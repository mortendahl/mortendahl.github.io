---
layout:     post
title:      "Private Image Analysis with MPC"
subtitle:   "Training CNNs on Sensitive Data using SPDZ"
date:       2017-09-19 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-04.jpg"
---

<em><strong>This post is still a work in progress; in particular, its concrete practically is yet to be seen from implementation experiments.</strong></em>

<em><strong>TL;DR:</strong> in this blog post we take a typical CNN deep learning model and go through a series of steps that enable both training and prediction to instead be done on encrypted data.</em> 

Using deep learning to analyse images through [convolutional neural networks](http://cs231n.github.io/) (CNNs) has gained enormous popularity over the last few years due to their success in out-performing many other approaches on this and related tasks. 

One recent application took the form of [skin cancer detection](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html), where anyone can quickly take a photo of a skin lesion using a mobile phone app and have it analysed with "performance on par with [..] experts" (see the [associated video](https://www.youtube.com/watch?v=toK1OSLep3s) for a demo). Having access to a large set of clinical photos played a key part in training this model -- a data set that could be considered sensitive.

Which brings us to privacy and eventually [secure multi-party computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) (MPC): how many applications are limited today due to the lack of access to data? In the above case, could the model be improved by letting anyone with a mobile phone app contribute to the training data set? And if so, how many would volunteer given the risk of exposing personal health related information?

With MPC we can potentially lower the risk of exposure and hence increase the incentive to participate. More concretely, by instead performing the training on encrypted data we can prevent anyone from ever seeing not only individual data, but also the learned model parameters. Further techniques such as [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) could additionally be used to hide any leakage from predictions as well, but we won't go into that here.

In this blog post we'll look at a simpler use case for image analysis but go over all required techniques. 

<em>A big thank you goes out to [Andrew Trask](https://twitter.com/iamtrask), [Nigel Smart](https://twitter.com/smartcryptology), [Adrià Gascón](https://twitter.com/adria), and the [OpenMined community](https://twitter.com/openminedorg) for inspiration and interesting discussions on this topic!</em>


# Setting

We will assume that the training data set is jointly held by a set of *input providers* and that the training is performed by two distinct *servers* (or *parties*) that are trusted not to collaborate beyond what our protocol specifies. In practice, these servers could for instance be virtual instances in a shared cloud environment operated by two different organisations. 

The input providers are only needed in the very beginning to transmit their training data; after that all computations involve only the two servers, meaning it is indeed plausible for the input providers to use e.g. mobile phones. Once trained, the model will remain jointly held in encrypted form by the two servers where anyone can use it to make further encrypted predictions.

For technical reasons we also assume a distinct *crypto provider* that generates certain raw material used during the computation for increased efficiency; there are ways to eliminate this additional entity but we won't go into that here. 

Finally, in terms of security we aim for a typical notion used in practice, namely *honest-but-curious (or passive) security*, where the servers are assumed to follow the protocol but may otherwise try to learn as much possible from the data they see. While a slightly weaker notion than *fully malicious (or active) security* with respect to the servers, this still gives strong protection against anyone who may compromise one of the servers *after* the computations, despite what they do. Note that for the purpose of this blog post we will actually allow a small privacy leakage during training as detailed later.


# Image Analysis with CNNs

Our use case is the canonical [MNIST handwritten digit recognition](https://www.tensorflow.org/get_started/mnist/beginners), namely learning to identify the Arabic numeral in a given image, and we will use the following CNN model from a [Keras example](https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py) as our base. 

We won't go into the details of this model here since the principles are already well-covered [elsewhere](https://github.com/ageron/handson-ml), but the basic idea is to first run an image through a set of *feature layers* that in some sense map the raw pixels of the input image into abstract properties. These properties are then subsequently combined by a set of *classification layers* to yield a probability distribution over the possible digits. The final outcome is then typically simply the digit with highest assigned probability.

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

With CNNs in place we next turn to MPC. Unlike the protocol used in [a previous blog post](/2017/04/17/private-deep-learning-with-mpc/), here we will use the state-of-the-art SPDZ protocol as it allows us to have only two servers and have received significant scientific attention over the last few years. As a result, several optimisations are known that can used to speed up our computation as shown later.

The protocol was first described in [SPZD'12](https://eprint.iacr.org/2011/535) and [DKLPSS'13](https://eprint.iacr.org/2012/642), but have also been the subject of at least [one series of blog posts](https://bristolcrypto.blogspot.fr/2016/10/what-is-spdz-part-1-mpc-circuit.html). Several implementations exist, including [one](https://www.cs.bris.ac.uk/Research/CryptographySecurity/SPDZ/) from the [cryptography group](http://www.cs.bris.ac.uk/Research/CryptographySecurity/) at the University of Bristol providing both high performance and full active security.

As usual, all computations take place in a field, here identified by a prime `Q`. As we will see, this means we also need a way to encode the fixed-point numbers used by the CNNs as integers modulo a prime, and we have to take care that these never "wrap around" as we then may not be able to recover the correct result.

Moreover, while the computational resources used by a procedure is often only measured in time complexity, i.e. the time it takes the CPU to perform the computation, with interactive computations such as the SPDZ protocol it also becomes relevant to consider communication and round complexity. The former measures the number of bits sent across the network, which is a relatively slow process, and the latter the number of synchronisation points needed between the two parties, which may block one of them with nothing to do until the other catches up. Both hence also have a big impact on overall executing time.

Concretely, we have an interest in keeping `Q` is small as possible, not only because we can then do arithmetic operations using only a single word sized operations (as opposed to arbitrary precision arithmetic which is significantly slower), but also because we have to transmit less bits when sending field elements across the network.

Note that while the protocol in general supports computations between any number of parties we here present it for the two-party setting only. Moreover, as mentioned earlier, we aim only for passive security and assume a crypto provider that will honestly generate the needed triples.


## Sharing and reconstruction

Sharing a private value between the two servers is done using the simple [additive scheme](/2017/06/04/secret-sharing-part1/#additive-sharing). This may be performed by anyone, including an input provider, and keeps the value [perfectly private](https://en.wikipedia.org/wiki/Information-theoretic_security) as long as the servers are not colluding.

```python
def share(secret):
    x0 = random.randrange(Q)
    x1 = (secret - x0) % Q
    return [x0, x1]
```

And when specified by the protocol, the private value can be reconstruct by a server sending his share to the other.

```python
def reconstruct(shares):
    return sum(shares) % Q
```

Of course, if both parties are to learn the private value then they can send their share simultaneously and hence still only use one round of communication.

Note that the use of an additive scheme means the servers are required to be highly robust, unlike e.g. [Shamir's scheme](http://127.0.0.1:4000/2017/06/04/secret-sharing-part1/) which may handle some servers dropping out. If this is a reasonable assumption though, then additive sharing provides significant advantages.


## Linear operations

Having obtained sharings of private values we may next perform certain operations on these. The first set of these is what we call linear operations since they allow us to form linear combinations of private values.

The first are addition and subtraction, which are simple local computations on the shares already held by each server.

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

Next we may also perform multiplication with a public value by again only performing a local operation on the share already held by each server.

```python
def mul_public(x, k):
    y0 = (x[0] * k) % Q
    y1 = (x[1] * k) % Q 
    return [y0, y1]
```

Note that the security of these operations is straight-forward since no communication is taking place between the two parties and hence nothing new could have been revealed.


## Multiplication

Multiplication of two private values is where we really start to deviate from the protocol used [previously](/2017/04/17/private-deep-learning-with-mpc/). The techniques used there inherently need at least three parties so won't be much help in our two party setting. 

Perhaps more interesting though, is that the new techniques used here allow us to shift parts of the computation to an *offline phase* where *raw material* that doesn't depend on any of the private values can be generated at convenience. As we shall see later, this can be used to significantly speed up the *online phase* where training and prediction is taking place.

This raw material is popularly called a *multiplication triple* (and sometimes *Beaver triple* due to their introduction in [Beaver'91](https://scholar.google.com/scholar?cluster=14306306930077045887)) and consists of independent sharings of three values `a`, `b`, and `c` such that `a` and `b` are uniformly random values and `c == a * b % Q`. Here we assume that these triples are generated by the crypto provider, and the resulting shares distributed to the two parties ahead of running the online phase. In other words, when performing a multiplication we assume that `Pi` already knows `a[i]`, `b[i]`, and `c[i]`. 

```python
def generate_multiplication_triple():
    a = random.randrange(Q)
    b = random.randrange(Q)
    c = a * b % Q
    return share(a), share(b), share(c)
```

Note that a large portion of the effort in the original papers and the full implementation is spent on removing the crypto provider and instead letting the parties generate these triples on their own; we won't go into that here but see the resources pointed to earlier for details.

To use multiplication triples to compute the product of two private values `x` and `y` we proceed as follows. The idea is simply to use `a` and `b` to respectively mask `x` and `y` and then reconstruct the masked values as respectively `epsilon` and `delta`. As public values, `epsilon` and `delta` may then be combined locally by each server to form a sharing of `z == x * y`.

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

Finally, since `x` and `y` are [perfectly hidden](https://en.wikipedia.org/wiki/Information-theoretic_security) by `a` and `b`, neither server learns anything new as long as each triple is only used once. Moreover, the newly formed sharing of `z` is "fresh" in the sense that it contains no information about the sharings of `x` and `y` that were used in its construction, since the sharing of `c` was independent of the sharings of `a` and `b`.


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

To get around this we'll simply make sure to pick `Q` large enough relative to the chosen precision and maximum magnitude. One place where we have to be careful is when doing multiplications as these double the precision. As done earlier we must hence leave enough room for double precision, and additionally include a truncation step after each multiplication where we bring the precision back down. Unlike earlier though, in the two server setting the truncation step can be performed as a local operation as pointed out in [SecureML](https://eprint.iacr.org/2017/396).

```python
def truncate(x, amount=6):
    y0 = x[0] // 10**amount
    y1 = Q - ((Q - x[1]) // 10**amount)
    return [y0, y1]
```

With this in place we are now (in theory) set to perform any desired computation on encrypted data.


# Adapting the Model

While it is in principle possible to compute any function securely with what we already have, and hence also the base model from above, in practice it is relevant to first consider variants of the model that are more MPC friendly, and vice versa. In slightly more picturesque words, it is common to open up our two black boxes and adapt the two technologies to better fit each other.

The root of this comes from some operations being surprisingly expensive in the encrypted setting. We saw above that addition and multiplication are relatively cheap, yet comparison and division with private denominator are not. For this reason we make a few changes to the model to avoid these.

The various changes presented here as well as their simulation performances are available as full Keras models in the [associated Python notebook](https://github.com/mortendahl/privateml/blob/master/image-analysis/Keras.ipynb).


## Optimizer

The first issue involves the optimizer: while [*Adam*](http://ruder.io/optimizing-gradient-descent/index.html#adam) is a preferred choice in many implementations for its efficiency, it also involves taking a square root of a private value and using one as the denominator in a division. While it is theoretically possible to [compute these securely](https://eprint.iacr.org/2012/164), in practice it could be a significant bottleneck for performance and hence relevant to avoid.

A simple remedy is to switch to the [*momentum SGD*](http://ruder.io/optimizing-gradient-descent/index.html#momentum) optimizer, which may imply longer training time but only uses simple operations.

```python
model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(clipnorm=10000, clipvalue=10000),
    metrics=['accuracy'])
```

An additional caveat is that many optimizers use [clipping](http://nmarkou.blogspot.fr/2017/07/deep-learning-why-you-should-use.html) to prevent gradients from growing too small or too large. This requires a [comparison on private values](https://www.iacr.org/archive/pkc2007/44500343/44500343.pdf), which again is a somewhat expensive operation in the encrypted setting, and as a result we aim to avoid using this technique altogether. To get realistic results from our Keras simulation we increase the bounds as seen above.


## Layers

Speaking of comparisons, the *ReLU* and max-pooling layers poses similar problems. In [CryptoNets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) the former is replaced by a squaring function and the latter by average pooling, while [SecureML](https://eprint.iacr.org/2017/396) implements a ReLU-like activation function by adding complexity that we wish to avoid to keep things simple. As such, we here instead use higher-degree sigmoid activation functions and average-pooling layers. Note that average-pooling also uses a division, yet this time the denominator is a public value, and hence division is simply a public inversion followed by a multiplication.

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

[Simulations](https://github.com/mortendahl/privateml/blob/master/image-analysis/Keras.ipynb) indicate that with this change we now have to bump the number of epochs, slowing down training time by an equal factor. Other choices of learning rate or momentum may improve this.

```python
model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test))
```

The remaining layers are easily dealt with. Dropout and flatten do not care about whether we're in an encrypted or unencrypted setting, and dense and convolution are matrix dot products which only require basic operations.


## Softmax and loss function

The final *softmax* layer also causes complications for training in the encrypted setting as we need to compute both an [exponentiation using a private exponent](https://cs.umd.edu/~fenghao/paper/modexp.pdf) as well as normalisation in the form of a division with a private denominator.

While both remain possible we here choose a much simpler approach and allow the predicted class likelihoods for each training sample to be revealed to one of the servers, who can then compute the result from the revealed values. This of course results in a privacy leakage that may or may not pose an acceptable risk. 

One heuristic improvement is for the servers to first permute the vector of class likelihoods for each training sample before revealing anything, thereby hiding which likelihood corresponds to which class. However, this may be of little effect if e.g. "healthy" often means a narrow distribution over classes while "sick" means a spread distribution.

Another is to introduce a dedicated third server who only does this small computation, doesn't see anything else from the training data, and hence cannot relate the labels with the sample data. Something is still leaked though, and this is hard to reason about.

Finally, we could also replace this [one-vs-all](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) approach with an [one-vs-one](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-one) approach using e.g. sigmoids. As argued earlier this allows us to fully compute the predictions without decrypting. We still need to compute the loss however, which be done by also considering a different loss function.

Note that none of the issues mentioned here occur when later performing predictions using the trained network, as there is no loss to be computed and the servers can there simply skip the softmax layer and let the recipient of the prediction compute it himself on the revealed values: for him it's simply a question of how the values are interpreted.


## Transfer Learning

At this point [it seems](https://github.com/mortendahl/privateml/blob/master/image-analysis/Keras.ipynb) that we can actually train the model as-is and get decent results. But as often done in CNNs we can get significant speed-ups by employing [transfer](http://cs231n.github.io/transfer-learning/) [learning](http://ruder.io/transfer-learning/); in fact, it is somewhat [well-known](https://yashk2810.github.io/Transfer-Learning/) that "very few people train their own convolutional net from scratch because they don’t have sufficient data" and that "it is always recommended to use transfer learning in practice".

A particular application to our setting here is that training may be split into a pre-training phase using non-sensitive public data and a fine-tuning phase using sensitive private data. For instance, in the case of a skin cancer detector, the researchers may choose to pre-train on a public set of photos and then afterwards ask volunteers to improve the model by providing additional photos. 

Moreover, besides a difference in cardinality, there is also room for differences in the two data sets in terms of subjects, as CNNs have a tendency to first decompose these into meaningful subcomponents, the recognition of which is what is being transferred. In other words, the technique is strong enough for pre-training to happen on a different type of images than fine-tuning.

Returning to our concrete use-case of character recognition, we will let the "public" images be those of digits `0-4` and the "private" images be those of digits `5-9`. As an alternative, it doesn't seem unreasonable to instead have used for instance characters `a-z` as the former and digits `0-9` as the latter.


### Pre-train on public dataset

In addition to avoiding the overhead of training on encrypted data for the public dataset, we also benefit from being able to train with more advanced optimizers. Here for instance, we switch back to the `Adam` optimizer for the public images and can take advantage of its improved training time. In particular, we see that we can again lower the number of epochs needed.

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

Once happy with this the servers simply shares the model parameters and move on to training on the private dataset.


### Fine-tune on private dataset

While we now begin encrypted training on model parameters that are already "half-way there" and hence can be expected to require fewer epochs, another benefit of transfer learning, as mentioned above, is that recognition of subcomponents tend to happen in the lower layers of the network and may in some cases be used as-is. As a result, we now freeze the parameters of the feature layers and focus training efforts exclusively on the classification layers.

```python
for layer in feature_layers:
    layer.trainable = False
```

Note however that we still need to run all private training samples forward through these layers; the only difference is that we skip them in the backward step and that there are few parameters to train.

Training is then performed as before, although now using a lower learning rate.

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

In the end we go from 25 epochs to 5 epochs in the simulations.


## Preprocessing

There are few preprocessing optimisations one could also apply but that we *won't* consider further here. 

The first is to move the computation of the frozen layers to the input provider so that it's the output of the flatten layer that is shared with the servers instead of the pixels of the images. In this case the layers are said to perform *feature extraction* and we could potentially also use more powerful layers. However, if we want to keep the model proprietary then this adds significant complexity as the parameters now have to be distributed to the clients in some form.

Another typical approach to speed up training is to first apply dimensionality reduction techniques such as a [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). This approach is taken in the encrypted setting in [BSS+'17](https://eprint.iacr.org/2017/857).


# Adapting the Protocol

Having looked at the model we next turn to the protocol: as well shall see, understanding the operations we want to perform can help speed things up. 

In particular, a lot of the computation can be moved to the crypto provider, who's generated raw material is independent of the private inputs and to some extend even the model. As such, its computation may be done in advance whenever it's convenient and at large scale.

Recall from earlier that it's relevant to optimising both round and communication complexity, and the extensions suggested here are often aimed at improving these at the expense of additional local computation. As such, practical experiments are needed to validate their benefits under concrete conditions.


## Average pooling

Starting with the easiest type of layer, we see that average pooling only requires a summation followed by a division with a public denominator, and hence can be implemented by a multiplication with a public value: since the denominator is public we can easily find its inverse and then simply multiply and truncate. As such, pooling is an entirely local operation.


## Dropout

Nothing special related to secure computation here, only thing is to make sure that the two servers agree on which values to drop in each training iteration. This can be done by simply agreeing on a seed value.


## Dense layers

The matrix dot product needed for dense layers can of course be implemented in the typical fashion using multiplication and addition. If we want to compute `dot(X, Y)` for matrices `X` and `Y` with shapes respectively `(m, k)` and `(k, n)` then this requires `m * n * k` multiplications, meaning we have to communicate the same number of masked values. While these can all be sent in parallel so we only need one round, if we allow ourselves to use another kind of preprocessed triple then we can reduce the communication cost by an order of magnitude.

For instance, the second dense layer in our model computes a dot product between a `(32, 128)` and a `(128, 5)` matrix. Using the typical approach requires sending `32 * 5 * 128 == 22400` masked values per batch, but by using the preprocessed triples described below we instead only have to send `32 * 128 + 5 * 128 == 4736` values, almost a factor 5 improvement. And for the first dense layer it is even greater, namely slightly more than a factor 25. 

The trick is to ensure that each private value is only sent masked once. To make this work we need triples `(R, S, T)` of random matrices `R` and `S` with the appropriate shapes and such that `T == dot(R, S)`. 

```python
def generate_matmul_triple(m, k, n):
    R = wrap(np.random.randint(Q, size=(m, k)))
    S = wrap(np.random.randint(Q, size=(k, n)))
    T = np.dot(R, S)
    return share(R), share(S), share(T)
```

Given such a triple we can instead communicate the values of `Rho = X - R` and `Sigma = Y - S` and perform local computation `dot(Rho, Sigma) + dot(R, Sigma) + dot(Rho, S) + T` to obtain `dot(X, Y)`.

```python
def matmul(X, Y, triple):
    R, S, T = triple
    Rho = reconstruct(X - R)
    Sigma = reconstruct(Y - S)
    return np.dot(Rho, Sigma) + np.dot(R, Sigma) + np.dot(Rho, S) + T
```

Security of using these triples follows the same argument as for multiplication triples: the communicated masked values perfectly hides `X` and `Y` while `T` being an independent fresh sharing makes sure that the result cannot leak anything about its constitutes. 

Note that this kind of triple is used in [SecureML](https://eprint.iacr.org/2017/396), which also give techniques allowing the servers to generate them without the help of the crypto provider.


## Convolutions

Like dense layers, convolutions can be treated either as a series of scalar multiplications or as [a matrix multiplication](http://cs231n.github.io/convolutional-networks/#conv), although the latter only after first expanding the tensor of training samples into a matrix with significant duplication. Unsurprisingly this leads to communication costs that in both cases can be improved by introducing another kind of triple.

As an example, the first convolution maps a tensor with shape `(m, 28, 28, 1)` to one with shape `(m, 28, 28, 32)` using `32` filters of shape `(3, 3, 1)` (excluding the bias vector). For batch size `m == 32` this means `7,225,344` communicated elements if we're using only scalar multiplications, and `226,080` if using a matrix multiplication. However, since there are only `(32*28*28) + (32*3*3) == 25,376` private values involved in total (again not counting bias since they only require addition), we see that there is roughly a factor `9` overhead. With a new kind of triple we can remove this overhead and save on communication cost: for 64 bit elements this means `200KB` per batch instead of respectively `1.7MB` and `55MB`.

The triples `(A, B, C)` we need here are similar to those used in matrix multiplication, with `A` and `B` having shapes matching the two inputs, i.e. `(m, 28, 28, 1)` and `(32, 3, 3, 1)`, and `C` matching output shape `(m, 28, 28, 32)`.


## Sigmoid activations

As we did [earlier](/2017/04/17/private-deep-learning-with-mpc/#approximating-sigmoid), we may use a degree-9 polynomial to approximate the sigmoid activation function with a sufficient level of accuracy. Evaluating this polynomial for a private value `x` requires computing a series of powers of `x`, which of course may be done by sequential multiplication. But this means several rounds and corresponding amount of communication.

As an alternative we can again use a new kind of preprocessed triple that allows exponentiation to all required powers to be done in a single round. The length of these "triples" is not fixed but equals the highest exponent, such that a triple for squaring, for instance, consists of independent sharings of `a` and `a^2`, while one for cubing consists of independent sharings of `a`, `a^2`, and `a^3`.

```python
def generate_powering_triple(exponent):
    a = random.randrange(Q)
    return [ share(pow(a, e, Q)) for e in range(1, exponent+1) ]
```

To use these we notice that if `epsilon = x - a` then `x^n == (epsilon + a)^n`, which by [the binomal theorem](https://en.wikipedia.org/wiki/Binomial_theorem) may be expressed as a weighted sum of `epsilon^n * a^0`, ..., `epsilon^0 * a^n` using the [binomial coefficients](https://en.wikipedia.org/wiki/Binomial_coefficient) as weights. For instance, we have `x^3 == (c0 * epsilon^3) + (c1 * epsilon^2 * a) + (c2 * epsilon * a^2) + (c3 * a^3)` with `ck = C(3, k)`.

Moreover, a triple for e.g. cubing `x` can also simultaneously be used for squaring `x` simply by skipping some powers and computing different binomial coefficients. Hence, all intermediate powers may be computed using a single triple and communication of one field element. The security of this again follows from the fact that all powers in the triple are independently shared.

```python
def pows(x, triple):
    # local masking
    a = triple[0]
    v = sub(x, a)
    # communication: the players simultanously send their share to the other
    epsilon = reconstruct(v)
    # local combination to compute all powers
    x_powers = []
    for exponent in range(1, len(triple)+1):
        # prepare all term values
        a_powers = [ONE] + triple[:exponent]
        e_powers = [ pow(epsilon, e, Q) for e in range(exponent+1) ]
        coeffs   = [ binom(exponent, k) for k in range(exponent+1) ]
        # compute and sum terms
        terms = ( mul_public(a,e*c) for a,e,c in zip(a_powers,reversed(e_powers),coeffs) )
        x_powers.append(reduce(lambda x,y: add(x, y), terms))
    return x_powers
```

Once we have these powers of `x`, evaluating a polynomial with public coefficients is then just a weighted sum.

```python
def pol_public(x, coeffs, triple):
    powers = [ONE] + pows(x, triple)
    terms = ( mul_public(xe, ce) for xe,ce in zip(powers, coeffs) )
    return reduce(lambda y,z: add(y, z), terms)
```

There is one caveat however, and that is that we now need room for the higher precision of the powers: `x^n` has `n` times the precision of `x` and we want to make sure that this value does not wrap around modulo `Q`.

One way around this is to temporarily switch to a larger field and compute the powers and truncation there. The conversion to and from this larger field `P` each take one round of communication, so polynomial evaluation ends up taking a total of three rounds. 

Security wise we also have to pay a small price, although from a practical perspective there is little difference. In particular, for this operation we rely on *statistical security* instead of perfect security: since `r` is not an uniform random element here, there's a tiny risk that something will be leaked about `x`.

```python
def generate_statistical_mask():
    return random.randrange(2*BOUND * 10**KAPPA)

def generate_zero_triple(field):
    return share(0, field)

def convert(x, from_field, to_field, zero_triple):
    # local mapping to positive representation
    x = add_public(x, BOUND, from_field)
    # local masking and conversion by player 0
    r = generate_statistical_mask()
    y0 = (zero_triple[0] - r) % to_field
    # exchange of masked share: one round of communication
    e = (x[0] + r) % from_field
    # local conversion by player 1
    xr = (e + x[1]) % from_field
    y1 = (zero_triple[1] + xr) % to_field
    # local mapping back from positive representation
    y = [y0, y1]
    y = sub_public(y, BOUND, to_field)
    return y

def upshare(x, large_zero_triple):
    return convert(x, Q, P, large_zero_triple)

def downshare(x, small_zero_triple):
    return convert(x, P, Q, small_zero_triple)
```

Note that we could of course decide to simply do all computations in the larger field `P`, thereby avoiding the conversion steps. This will likely slow down the local computations by a non-trivial factor however, as we may need arbitrary precision arithmetic for `P` as opposed to e.g. 64 bit native arithmetic for `Q`.

Practical experiments will show whether it best to stay in `Q` and use a few more rounds, or switch temporarily to `P` and pay for conversion and arbitrary precision arithmetic. Specifically, for low degree polynomials the former is likely better.



## Softmax and cross-entropy

TODO: permutation based on permuted identity matrix?

<!--
Let's say that P1 is the one reconstructing the class likelihoods and performing the softmax computation to turn these into probabilities. This means P0 is responsible for permuting the vector `v` of encrypted likelihoods before, as well as applying the inverse permutation on the vector of encrypted probabilities `w` after. 

`E(v) -> pi(E(v)) -> pi(v) -> pi(w) -> E(pi(w)) -> E(w)`

Since P1 is holding a part of the encrypted values P0 cannot do the permutation as a local operation. Instead she can use a permutation matrix `P` by applying a random permutation `pi: [C] -> [C]` to the rows of an identity matrix, and then jointly compute vector `pi(v) = dot(P, v)`

Player P0 

If `sick` e.g. means that several probabilities will be significant while `not sick` means that only a single probability will be non-zero then this heuristic doesn't hide anything.

- permutation matrix and its inverse
-->

# Thoughts

As always, when previous thoughts and questions have been answered there is already a new batch waiting.

## Activation functions

A natural question is which of the other typical activation functions are efficient in the encrypted setting. As mentioned above, [SecureML](https://eprint.iacr.org/2017/396) makes use of ReLU by switching to garbled circuits, and [CryptoDL](https://arxiv.org/abs/1711.05189) gives low-degree polynomial approximations to both Sigmoid, ReLU, and Tanh (using [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) for better accuracy).

It may also be relevant to consider non-typical but simpler activations functions, such as squaring as in e.g. [CryptoNets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/), if for nothing else than simplifying both computation and communication.


## Garbled circuits

While so far only mentioned in the context of evaluating activation functions, [garbled](https://oblivc.org/) [circuits](https://github.com/encryptogroup/ABY) could in fact also be used for larger parts, including as the main means of secure computation as done in for instance [DeepSecure](https://arxiv.org/abs/1705.08963). 

Compared to e.g. SPDZ this technique has the benefit of using only a constant number of communication rounds. The downside is that operations are now often happening on bits instead of on larger field elements, meaning more computation is involved.


## Generalised triples

When seeking to reduce communication, one may also wonder how much can be pushed to the preprocessing phase. Concretely, it might for instance be possible to have triples for advanced functions such as evaluating both a dense layer and its activation function with a single round of communication. Main question here again seems to be efficiency, this time in terms of triple storage and amount of computation needed for the recombination step.


## Precision

A lot of the research around [federated learning](https://research.googleblog.com/2017/04/federated-learning-collaborative.html) involve [gradient compression](https://arxiv.org/abs/1610.05492) in order to save on communication cost. Closer to our setting we have [BMMP'17](https://eprint.iacr.org/2017/1114) which uses quantization to apply homomorphic encryption to deep learning, and even [unencrypted](https://arxiv.org/abs/1610.02132) [production-ready](https://www.tensorflow.org/performance/quantization) systems often consider this technique as a way of improving performance.


## Floating point arithmetic

Above we used a fixed-point encoding of real numbers into field elements, yet unencrypted deep learning is typically using a floating point encoding. As shown in [ABZS'12](https://eprint.iacr.org/2012/405) and [the reference implementation of SPDZ](https://github.com/bristolcrypto/SPDZ-2/issues/7), it is also possible to use the latter in the encrypted setting, apparently with performance advantages for certain operations.


## GPUs

Since deep learning is today mostly done on GPUs for performance reasons, it is natural to consider whether similar speedups can be achieved by applying them in MPC computations. Some [work](https://www.cs.virginia.edu/~shelat/papers/hms13-gpuyao.pdf) exist on this topic for garbled circuits, yet it seems less popular in the secret sharing setting of e.g. SPDZ. One problem here might be to ensure enough room in the supported integer types or in the integral part of supported floats. 

A potential remedy to this is to decompose numbers using the [CRT](https://en.wikipedia.org/wiki/Chinese_remainder_theorem) into several components that are computed on in parallel. For this to work we would need to do our computations over a ring instead of a field, since our modulus must now be a composite number as opposed to a prime.


<!--

# Old

https://eprint.iacr.org/2017/262.pdf

Pooling in MPC:
- doing entirely out of fashion: https://arxiv.org/abs/1412.6806 and http://cs231n.github.io/convolutional-networks/ -- use larger stride in CONV layer once in a while

in numpy:
- https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
- https://github.com/andersbll/nnet

Gradient Compression
- https://arxiv.org/pdf/1610.02132.pdf

https://eprint.iacr.org/2016/1117.pdf

- https://stackoverflow.com/questions/36515202/why-is-the-cross-entropy-method-preferred-over-mean-squared-error-in-what-cases
- https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/

-->
