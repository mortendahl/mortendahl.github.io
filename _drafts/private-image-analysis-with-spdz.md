---
layout:     post
title:      "Private Image Analysis with MPC"
subtitle:   "Training CNNs on Sensitive Data using SPDZ"
date:       2017-09-01 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-04.jpg"
---

<em><strong>This post is still very much a work in progress; in particular, its concrete practically is yet to be seen from implementation experiments.</strong></em>

<em><strong>TL;DR:</strong> in this blog post we take a typical CNN deep learning model and go through a series of steps that enable both training and prediction to instead be done on encrypted data.</em>

Using deep learning to analysis images through [convolutional neural networks](http://cs231n.github.io/convolutional-networks/) (CNNs) has gained enormous popularity over the last few years due to their success in out-performing many other approaches on this and related tasks. 

One recent application took the form of [skin cancer detection](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html), where anyone can quickly take a photo of a skin lesions using a mobile phone app and have it analysed with "performance on par with [..] experts" (see the [associated video](https://www.youtube.com/watch?v=toK1OSLep3s) for a demo). Having access to a large set of clinical photos played a key part in training this model -- a data set that could be considered sensitive.

Which brings us to privacy and eventually [secure multi-party computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) (MPC): how many applications are limited today due to the lack of access to data? In the above case, could the model be improved by letting anyone with a mobile phone app contribute to the training data set? And if so, how many would volunteer given the risk of exposing personal health related information?

With MPC we can potentially lower the risk of exposure and hence increase the incentive to participate. More concretely, by instead performing the training on encrypted data we can prevent anyone from ever seeing not only individual data, but also the learned model parameters. Further techniques such as [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) could additionally be used to hide any leakage from predictions as well, but we won't go into that here.

In this blog post we'll look at a simpler use case for image analysis but go over all required techniques. 


# Setting

We will assume that the training data set is held by a set of *input providers* and that the training is performed by two distinct *parties* that are trusted not to collaborate beyond what our protocol specifies. In practice, these parties could for instance each operate their own virtual server instance in a shared cloud environment. 

Note that the input providers are only needed in the very beginning to transmit their training data; after that all computations involve only the two parties. This means that input providers using mobile phones are indeed plausible.

For technical reasons we also assume a distinct *crypto provider* that generates certain raw material used during the computation for increased efficiency; there are ways to eliminate this additional entity but we won't go into that here. 

Finally, in more technical terms, we aim to both do training and prediction on encrypted data, with a guarantee of passive security in the two-party server-aided model and using an ideal functionality for triple generation. TODO small privacy leakage from softmax


# Image Analysis with CNNs

Our use case is the canonical [MNIST handwritten digit recognition](https://www.tensorflow.org/get_started/mnist/beginners), namely learning to identify the Arabic numeral in a given image, and we will use the following CNN model from a [Keras example](https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py) as our base. [TODO](https://github.com/ageron/handson-ml).

```python
feature_layers = [
    Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(.25),
    Flatten()
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(.50),
    Dense(NUM_CLASSES),
    Activation('softmax')
]

model = Sequential(feature_layers + classification_layers)

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test))
```

Using [Keras](https://keras.io/) has the benefit that we can perform quick experiments on unencrypted data to get an idea of the performance of the model itself, and it  provides a simple interface to later mimic in the encrypted setting.


# Secure Computation with SPDZ

With CNNs in place we next turn to MPC. Unlike the protocol used in [a previous blog post](/2017/04/17/private-deep-learning-with-mpc/), here we will use the state-of-the-art SPDZ protocol as it allows us to have only two parties and have received significant scientific attention over the last few years. As a result, several optimisations are known that can used to speed up our computation as shown later.

The protocol was first described in [SPZD'12](https://eprint.iacr.org/2011/535) and [DKLPSS'13](https://eprint.iacr.org/2012/642), but have also been the subject of at least [one series of blog posts](https://bristolcrypto.blogspot.fr/2016/10/what-is-spdz-part-1-mpc-circuit.html). Several implementations exist, including [one](https://github.com/bristolcrypto/SPDZ-2) from the cryptography group at the University of Bristol providing both high performance and full active security.

As usual, all computations take place in a field, here identified by a prime `Q`. As we will see, this means we also need a way to encode the fixed-point numbers used by the CNNs as integers modulo a prime, and we have to take care that these never "wrap around" as we then may not be able to recover the correct result.

Moreover, while the computational resources used by a procedure is often only measured in time complexity, i.e. the time it takes the CPU to perform the computation, with interactive computations such as the SPDZ protocol it also becomes relevant to consider communication and round complexity. The former measures the number of bits sent across the network, which is a relatively slow process, and the latter the number of synchronisation points needed between the two parties, which may block one of them with nothing to do until the other catches up. Both hence also have a big impact on overall executing time.

Concretely, we have an interest in keeping `Q` is small as possible, not only because we can then do arithmetic operations using only a single word sized operations (as opposed to arbitrary precision arithmetic which is significantly slower), but also because we have to transmit less bits when sending field elements across the network.

Note that while the protocol in general supports computations between any number of parties we here present it for the two-party setting only. Moreover, as mentioned earlier, we aim only for passive security and assume a crypto provider that will honestly generate the needed triples.


## Sharing and reconstruction

Sharing a private value between the two parties is done using the simple [additive scheme](/2017/06/04/secret-sharing-part1/#additive-sharing). This may be performed by either an input provider or by one of the parties, and keeps the value [perfectly private](https://en.wikipedia.org/wiki/Information-theoretic_security) as long as the two parties are not colluding.

```python
def share(secret):
    x0 = random.randrange(Q)
    x1 = (secret - x0) % Q
    return [x0, x1]
```

And when specified by the protocol, the private value can be reconstruct by a party sending his share to the other.

```python
def reconstruct(shares):
    return sum(shares) % Q
```

Of course, if both parties are to learn the private value then they can send their share simultaneously and hence still only use one round of communication.


## Linear operations

Having obtained sharings of private values we may next perform certain operations on these. The first set of these is what we call linear operations since they allow us to form linear combinations of private values.

The first are addition on subtraction, which are simple local computations on the shares already held by each party.

```python
def add(x, y):
    z0 = (x[0] + y[0]) % Q
    z1 = (y[1] + y[1]) % Q
    return [z0, z1]

def sub(x, y):
    z0 = (x[0] - y[0]) % Q
    z1 = (y[1] - y[1]) % Q
    return [z0, z1]
```

And if one of the values is public then we may simplify as follows.

```python
def add_public(x, k):
    y0 = (x[0] + k) % Q
    y1 =  x[1]
    return [y0, y1]

def sub_public(x, k):
    y0 = (x[0] - k) % Q
    y1 =  x[1]
    return [y0, y1]
```

Next we may also perform multiplication with a public value by again only performing a local operation on the share already held by each party.

```python
def mul_public(x, k):
    y0 = (x[0] * k) % Q
    y1 = (x[1] * k) % Q 
    return [y0, y1]
```

Note that the security of these operations is straight-forward since no communication is taking place between the two parties and hence nothing new could have been revealed.


## Multiplication

Multiplication of two private values is where we really start to deviate from the protocol used [previously](/2017/04/17/private-deep-learning-with-mpc/). The techniques used there inherently needed at least three parties so won't be much help in our two party setting. 

Perhaps more interesting though, is that the new techniques used here allow us to shift parts of the computation to an *offline phase* where raw material that doesn't depend on any of the private values can be generated at convenience. As we shall see later, this can be used to significantly speed up the *online phase* where training and prediction is taking place.

The raw material needed here is popularly called a *multiplication triple* (and sometimes *Beaver triple* due to their introduction in [Beaver'91](https://scholar.google.com/scholar?cluster=14306306930077045887)) and consists of independent sharings of three values `a`, `b`, and `c` such that `a` and `b` are uniformly random values and `c == a * b % Q`. Here we assume that these triples are generated by the crypto provider, and the resulting shares distributed to the two parties ahead of running the online phase. In other words, when we want to perform a multiplication we assume that `Pi` already knows `a[i]`, `b[i]`, and `c[i]`. 

```python
def generate_multiplication_triple():
    a = random.randrange(Q)
    b = random.randrange(Q)
    c = a * b % Q
    return share(a), share(b), share(c)
```

Note that a large portion of the effort in the original papers and the full implementation is spent on removing the crypto provider and instead letting the parties generate these triples on their own; we won't go into that here but see the resources pointed to earlier for details.

To use multiplication triples to compute the product of two private values `x` and `y` we proceed as follows. The idea is simply to use `a` and `b` to respectively mask `x` and `y` and then reconstruct the masked values as respectively `epsilon` and `delta`. As public values, `epsilon` and `delta` may then be combined locally by each party to form a sharing of `z == x * y`.

```python
def mul(x, y, triple):
    a, b, c = triple
    # local masking
    d = sub(x, a)
    e = sub(y, b)
    # communication: the players simultaneously send their shares to the other
    delta = reconstruct(d)
    epsilon = reconstruct(e)
    # local combination
    r = delta * epsilon % Q
    s = mul_public(a, epsilon)
    t = mul_public(b, delta)
    return add(s, add(t, add_public(c, r)))
```

If we write out the equations we see that `delta * epsilon == xy - xb - ay + ab`, `a * epsilon == ay - ab`, and `b * delta == bx - ab`, so that the sum of these with `c` cancels out everything except `xy`. In terms of complexity we see that communication of two field elements in one round is required. 

Finally, since `x` and `y` are [perfectly hidden](https://en.wikipedia.org/wiki/Information-theoretic_security) by `a` and `b`, neither party learns anything new as long as each triple is only used once. Moreover, the newly formed sharing of `z` is "fresh" in the sense that it contains no information about the sharings of `x` and `y` that were used in its construction, since `c` was independent of the sharings of `a` and `b`.


## Fixed-point encoding

The last step is to provide a mapping between the rational numbers used by the CNNs and the field elements used by the SPDZ protocol. As typically done, we here take a fixed-point approach where rational numbers are scaled by a fixed amount and then rounded off to an integer less than the field size `Q`.

```python
def encode(rational, precision=6):
    upscaled = int(rational * 10**precision)
    field_element = upscaled % Q
    return field_element

def decode(field_element, precision=6):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = upscaled / 10**precision
    return rational
```

In doing this we have to be careful not to "wrap around" by letting any encoding exceed `Q`; if this happens our decoding procedure will give wrong results.

To get around this we'll simply make sure to pick `Q` large enough relative to chosen precision and maximum magnitude. One place where we have to be careful is when doing multiplications as these double the precision. As done earlier we must hence actually leave enough room of double precision, and additionally include a truncation step after each multiplication where we bring the precision back down. Unlike earlier though, in the two party setting the truncation step can be performed as a local operation as pointed out in [SecureML](TODO).

```python
def truncate(x, amount=6):
    y0 = x[0] // 10**amount
    y1 = Q - ((Q - x[1]) // 10**amount)
    return [y0, y1]
```

With this in place we are now [in principle](TODO) set to perform any desired computation on encrypted data.


# Adapting the Model

While it is in principle possible to compute any function securely, and hence also the base model above, in practice it is still relevant to consider variants that are more MPC friendly. In other words, it is common to open up our black boxes and perform adaptations of both technologies to better fit each other.

The root of this comes from some operations being relatively cheap in the secure setting while other are surprisingly expensive. We saw above that addition and multiplication falls into the former category, while comparison and division with private denominator falls into the latter (division with *public denominator* is just a multiplication).

For this reason we first consider making a few changes to the model.


## Optimizer

The first issue involves the optimizer: while [*Adam*](http://ruder.io/optimizing-gradient-descent/index.html#adam) is a preferred choice in many implementations for its efficiency, it also involves operations that are currently not well suited for secure computation. Namely, it takes a square root of a private value and uses one as the denominator in a division. While it is theoretical possible to [compute these securely](TODO), in practice it will likely be a significant bottleneck for performance and hence relevant to avoid.

A simple remedy is to switch to the [*momentum SGD*](http://ruder.io/optimizing-gradient-descent/index.html#momentum) optimizer, which may imply longer training time but only uses simple operations.

```python
model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(clipnorm=10000, clipvalue=10000),
    metrics=['accuracy'])
```

An additional caveat is that many optimizers use clipping to prevent gradients from growing too small or too large. This requires a comparison on private values, which again is an expensive operation in the encrypted setting; to get realistic results from our Keras simulation we hence increase the bounds in an attempt to prevent clipping from happening.


## Layers

Seaking of comparisons, the *ReLU* and max-pooling layers poses similar problems. In [CryptoNets](https://arxiv.org/abs/1412.6181) this is done TODO while in [SecureML](https://eprint.iacr.org/2017/396) this id TODO. Here we take the approach of using respectively the sigmoid activation function and average-pooling layers instead. Note that average-pooling also uses a division, yet this time the denominator is a public value and hence easy to perform.

```python
feature_layers = [
    Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    Activation('sigmoid'),
    Conv2D(32, (3, 3), padding='same'),
    Activation('sigmoid'),
    AveragePooling2D(pool_size=(2,2)),
    Dropout(.25),
    Flatten()
]

classification_layers = [
    Dense(128),
    Activation('sigmoid'),
    Dropout(.50),
    Dense(5),
    Activation('softmax')
]

model = Sequential(feature_layers + classification_layers)
```

Note however that we now have to bump the number of epochs significantly, slowing down training time by an equal factor. Other choices of learning rate or momentum may improve this.

```python
model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test))
```

The remaining layers are easily dealt with. Dropout and flatten do not care about whether we're in an encrypted or unencrypted setting, and dense and convolution are matrix dot products which only require basic operations.


## Loss function

Next, the *softmax* layer also causes complications in the encrypted setting since we need to compute both an exponentiation using a private exponent as well as a division with a private denominator. While both are theoretical possible (using e.g. [TODO](TODO) for the exponentiation) we here choose a more abstract approach and simply assume we have an *ideal functionality*, or *oracle*, that will do this small computation for us; concretely, we allow the inputs to be leaked to one of the two parties how can then compute the result from the unencrypted values. This results in a privacy leakage, yet it only concerns the final class predictions and may hence pose limited risk. One heuristic improvement is for the other party to permute the values before revealing anything, thereby hiding which prediction corresponds to which class.

Finally, note that by preprocessing the labels we may deal with the cross-entropy loss in the encrypted setting despite the use of a logarithm. Namely, by 

For each training sample we end up with one value per class which we want to treat as a probability estimation of the sample belonging to the different classes. Concretely, if we can recognize five different symbols then the network will generate five values for each sample. TODO during training we leak these values, meaning one of the two parties will learn the probability estimations for each sample; however we can permute these so that he doesn't learn which estimation corresponds to which class. moreover, when later performing predictions using the trained network we can avoid leaking any data by skipping the softmax step and letting the input provider himself compute it: in most cases does he receive a value for each class so it's simply a question of how he interprets these.

and the cross-entropy loss

```python
model.compile(
    loss='mean_squared_error', 
    optimizer=SGD(clipnorm=10000, clipvalue=10000, lr=0.1, momentum=0.9), 
    metrics=['accuracy'])
```

At this point it seems we can actually train the model as-is (see notebook), but like often done in CNNs we can get significant speed-ups by using transfer learning.


## Transfer Learning

It is [apparently](https://yashk2810.github.io/Transfer-Learning/) somewhat common knowledge that <em>"very few people train their own convolutional net from scratch because they don’t have sufficient data"</em> and that <em>"it is always recommended to use transfer learning in practice"</em>.

- http://cs231n.github.io/transfer-learning/
-  "It is always recommended to use transfer learning in practice."
- https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8
- http://ruder.io/transfer-learning/
- https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py
- https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
- https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html

pre-train on public dataset

```python
(x_train, y_train), (x_test, y_test) = public_dataset

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test))
```

fix lower layers

```python
for layer in feature_layers:
    layer.trainable = False
```

train on private dataset

```python
(x_train, y_train), (x_test, y_test) = private_dataset

model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(clipnorm=10000, clipvalue=10000, lr=0.1, momentum=0.0),
    metrics=['accuracy'])

model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test))
```

## Preprocessing

using first layers as feature extraction; means the trained model must be distributed to clients and cannot be kept proprietary. can use much more powerful feature layers then though since we never need to run them on encrypted data.

Dimension reduction using PCA; not discussed further here


# Adapting the Protocol

Secure computation is not yet a complete blackbox (if we want performance): while possible to compute any function with general protocols, understanding what computation we want to perform can help speed up things. It is not a custom protocol but simply adapting the general approach.

Recall from earlier that its worth optimising both round complexity and communication complexity.


## Average pooling

Starting with the easiest type of layer, we see that average pooling only requires a summation followed by a division with a public denominator, and hence can be implemented by a multiplication with a public value: since the denominator is public we can easily find its inverse and then simply multiply and truncate. As such, pooling is an entirely local operation.


## Dropout


## Dense layers

The matrix dot product needed for dense layers can of course be implemented in the typical fashion using multiplication and addition. If we want to compute `dot(X, Y)` for matrices `X` and `Y` with shapes respectively `(M, K)` and `(K, N)` then this requires `M * N * K` multiplications, meaning we have to communicate an equal number of masked values. While these can all be sent in parallel so we only need one round, if we allow ourselves to use another kind of preprocessed triple then we can reduce the communication cost by one order of magnitude. 

For instance, the second dense layer in our model computes a dot product between a `(32, 128)` and a `(128, 5)` matrix. Using the typical approach requires sending `32 * 5 * 128 == 22400` masked values, but by using the preprocessed triples detailed below we instead only have to send of `32 * 128 + 5 * 128 == 4736` values, roughly a factor 4.7 improvement. For the first dense layer the factor is even greater.

To make this work we need triples `(R, S, T)` of random matrices `R` and `S` with the appropriate shapes and such that `T == dot(R, S)`. 

```python
def generate_matmul_triple(m, k, n):
    r = np.random.randint(Q, size=(m, k))
    s = np.random.randint(Q, size=(k, n))
    t = np.dot(r, s) % Q
    return share(r), share(s), share(t) # TODO vectorized version
```

Given such a triple we can instead communicate the values of `rho = X - R` and `sigma = Y - S` and perform local computation `dot(rho, sigma) + dot(R, sigma) + dot(rho, S) + T` to obtain `dot(X, Y)`.

```python
def matmul(x, y, triple):
    r, s, t = triple
    rho = reconstruct((x - r) % Q) # TODO vectorized version
    sigma = reconstruct((y - s) % Q)
    return np.dot(rho, sigma) + np.dot(r, sigma) + np.dot(rho, s) + t
```

Security of using these triples follows the same argument as for multiplication triples: the communicated masked values perfectly hides `X` and `Y` while `T` being an independent fresh sharing makes sure that TODO. This kind of triple was also used in [SecureML](TODO).

**TODO** mention that the original papers do not give solutions to have these new type of triples can be generated without a crypto provider, and hence it's open how this is to be concretely implemented and to what extend it's practical. we leave that as future research.


## Convolutions

- naive vs matrix mul vs special conv triples


## Sigmoid activations

Following [a previous blog post](/2017/04/17/private-deep-learning-with-mpc/#approximating-sigmoid) we may use polynomial approximations to implement the sigmoid activations. Specifically, we can use degree nine polynomial to get a fair level of accuracy. Computing the terms in this polynomial may of course be done by sequential multiplication with private value `x`. This however comes with an increased number of rounds.

As an alternative we may again use a new kind of preprocessed triple so that exponentiation to any degree may be done in a single round. TODO

```python
def generate_exponentiation_triple(exponent):
    a = random.randrange(Q)
    return [ share(pow(a, e, Q)) for e in range(1, exponent+1) ]
```

```python
def exp(x, exponent, triple):
    # local masking
    v = sub(x, triple[0])
    # communication: the players simultanously send their share to the other
    epsilon = reconstruct(v)
    # local combination
    cs = [ binom(exponent, k, exact=True) for k in range(exponent+1) ]
    es = [ pow(epsilon, e, Q) for e in reversed(range(exponent+1)) ]
    bs = [ONE] + triple
    return reduce(add, ( scalar_mul(b, c * e) for c,e,b in zip(cs, es, bs) ))
```

There is one caveat however, and that is that we now run a larger risk of wrapping around due to the lack of intermediate precision truncation. One way around this is to temporarily switch to a larger field and perform the exponentiation and truncation there. The conversion to and from this larger field each take one round of communication, meaning exponentiation to any degree may be done in three rounds given the appropriate triples. 

```python
def generate_statistical_mask():
    return random.randrange(Q) % BASE**(2*PRECISION+1 + KAPPA)

def generate_zero_triple(field):
    return share(0, field)

def upshare(x, large_zero_triple):
    # map to positive range
    x = scalar_add(x, BASE**(2*PRECISION+1), Q)
    # player 0
    r = generate_statistical_mask()
    e = (x[0] + r) % Q
    y0 = (large_zero_triple[0] - r) % P
    # player 1
    xr = (e + x[1]) % Q
    y1 = (large_zero_triple[1] + xr) % P
    # combine
    y = [y0, y1]
    y = scalar_sub(y, BASE**(2*PRECISION+1), P)
    return y

def downshare(x, small_zero_triple):
    # map to positive range
    x = scalar_add(x, BASE**(2*PRECISION+1), P)
    # player 0
    r = generate_statistical_mask()
    e = (x[0] + r) % P
    y0 = (small_zero_triple[0] - r) % Q
    # player 1
    xr = (e + x[1]) % P
    y1 = (small_zero_triple[1] + xr) % Q
    # combine
    y = [y0, y1]
    return scalar_sub(y, BASE**(2*PRECISION+1), Q)
```

Note that we could of course decide to simply do all computations in the larger field, thereby avoiding the conversion steps; this however may slow down the local computation as we may need arbitrary precision arithmetic as opposed to 64 or 128 bit native arithmetic.

let `epsilon = x - a`. then we know from [the binomal theorem](https://en.wikipedia.org/wiki/Binomial_theorem) that e.g. `x^2 == (epsilon + a)^2 == epsilon^2 + 2 * epsilon * a + a^2`, and in general that `x^n` can be expressed as a weighted sum of powers of `epsilon` and `a`, using the binomal coefficients as weights. in other words, if we know the right powers of `epsilon` and `a`, then computing `x^n` is a linear operation that can be performed locally. in still other words, `x^n` is a linear combination of the powers of `a`....

we have to tweak the field size vs number of rounds: the more exp we want to do in one round, the larger the field must be to avoid the fixed-point encoding wrapping around; larger fields means slower computations but fewer rounds. we need log(exponent) rounds





## Softmax and cross-entropy

Let's say that P1 is the one reconstructing the class likelihoods and performing the softmax computation to turn these into probabilities. This means P0 is responsible for permuting the vector `v` of encrypted likelihoods before, as well as applying the inverse permutation on the vector of encrypted probabilities `w` after. 

`E(v) -> pi(E(v)) -> pi(v) -> pi(w) -> E(pi(w)) -> E(w)`

Since P1 is holding a part of the encrypted values P0 cannot do the permutation as a local operation. Instead she can use a permutation matrix `P` by applying a random permutation `pi: [C] -> [C]` to the rows of an identity matrix, and then jointly compute vector `pi(v) = dot(P, v)`

Player P0 


- permutation matrix and its inverse




# Old

Pooling in MPC:
- max pooling inefficient
- average pool, scaled mean pool in CryptoNets
- doing entirely out of fashion: https://arxiv.org/abs/1412.6806 and http://cs231n.github.io/convolutional-networks/ -- use larger stride in CONV layer once in a while

general idea of CNNs for image recon:
- [CryptoNets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/)
- [SecureML](https://eprint.iacr.org/2017/396)

Gradient Compression
- https://arxiv.org/pdf/1610.02132.pdf

- https://stackoverflow.com/questions/36515202/why-is-the-cross-entropy-method-preferred-over-mean-squared-error-in-what-cases
- https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/

in numpy:
- https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
- https://github.com/andersbll/nnet
