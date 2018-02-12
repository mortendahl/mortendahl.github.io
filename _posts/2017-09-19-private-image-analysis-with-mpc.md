---
layout:     post
title:      "Private Image Analysis with MPC"
subtitle:   "Training CNNs on Sensitive Data"
date:       2017-09-19 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-04.jpg"
---

<em><strong>TL;DR:</strong> we take a typical CNN deep learning model and go through a series of steps that enable both training and prediction to instead be done on encrypted data.</em> 

Using deep learning to analyse images through [convolutional neural networks](http://cs231n.github.io/) (CNNs) has gained enormous popularity over the last few years due to their success in out-performing many other approaches on this and related tasks. 

One recent application took the form of [skin cancer detection](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html), where anyone can quickly take a photo of a skin lesion using a mobile phone app and have it analysed with "performance on par with [..] experts" (see the [associated video](https://www.youtube.com/watch?v=toK1OSLep3s) for a demo). Having access to a large set of clinical photos played a key part in training this model -- a data set that could be considered sensitive.

Which brings us to privacy and eventually [secure multi-party computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) (MPC): how many applications are limited today due to the lack of access to data? In the above case, could the model be improved by letting anyone with a mobile phone app contribute to the training data set? And if so, how many would volunteer given the risk of exposing personal health related information?

With MPC we can potentially lower the risk of exposure and hence increase the incentive to participate. More concretely, by instead performing the training on encrypted data we can prevent anyone from ever seeing not only individual data, but also the learned model parameters. Further techniques such as [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy) could additionally be used to hide any leakage from predictions as well, but we won't go into that here.

In this blog post we'll look at a simpler use case for image analysis but go over all required techniques. A few notebooks are presented along the way, with the main one given as part of the [proof of concept implementation](#proof-of-concept-implementation).

<em>A big thank you goes out to [Andrew Trask](https://twitter.com/iamtrask), [Nigel Smart](https://twitter.com/smartcryptology), [Adrià Gascón](https://twitter.com/adria), and the [OpenMined community](https://twitter.com/openminedorg) for inspiration and interesting discussions on this topic!</em>


# Setting

We will assume that the training data set is jointly held by a set of *input providers* and that the training is performed by two distinct *servers* (or *parties*) that are trusted not to collaborate beyond what our protocol specifies. In practice, these servers could for instance be virtual instances in a shared cloud environment operated by two different organisations. 

The input providers are only needed in the very beginning to transmit their (encrypted) training data; after that all computations involve only the two servers, meaning it is indeed plausible for the input providers to use e.g. mobile phones. Once trained, the model will remain jointly held in encrypted form by the two servers where anyone can use it to make further encrypted predictions.

For technical reasons we also assume a distinct *crypto producer* that generates certain raw material used during the computation for increased efficiency; there are ways to eliminate this additional entity but we won't go into that here. 

Finally, in terms of security we aim for a typical notion used in practice, namely *honest-but-curious (or passive) security*, where the servers are assumed to follow the protocol but may otherwise try to learn as much possible from the data they see. While a slightly weaker notion than *fully malicious (or active) security* with respect to the servers, this still gives strong protection against anyone who may compromise one of the servers *after* the computations, despite what they do. Note that for the purpose of this blog post we will actually allow a small privacy leakage during training as detailed later.


# Image Analysis with CNNs

Our use case is the canonical [MNIST handwritten digit recognition](https://www.tensorflow.org/get_started/mnist/beginners), namely learning to identify the Arabic numeral in a given image, and we will use the following CNN model from a [Keras example](https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py) as our base. 

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

We won't go into the details of this model here since the principles are already [well-covered](http://cs231n.stanford.edu/) [elsewhere](https://github.com/ageron/handson-ml), but the basic idea is to first run an image through a set of *feature layers* that transforms the raw pixels of the input image into abstract properties that are more relevant for our classification task. These properties are then subsequently combined by a set of *classification layers* to yield a probability distribution over the possible digits. The final outcome is then typically simply the digit with highest assigned probability.

As we shall see, using [Keras](https://keras.io/) has the benefit that we can perform quick experiments on unencrypted data to get an idea of the performance of the model itself, as well as providing a simple interface to later mimic in the encrypted setting.


# Secure Computation with SPDZ

With CNNs in place we next turn to MPC. For this we will use the state-of-the-art SPDZ protocol as it allows us to only have two servers and to improve *online* performance by moving certain computations to an *offline* phase as described in detail in earlier [blog](/2017/09/03/the-spdz-protocol-part1) [posts](/2017/09/10/the-spdz-protocol-part2).

As typical in secure computation protocols, all computations take place in a field, here identified by a prime `Q`. This means we need to [encode](/2017/09/03/the-spdz-protocol-part1#fixed-point-encoding) the floating-point numbers used by the CNNs as integers modulo a prime, which puts certain constraints on `Q` and in turn has an affect on performance.

Moreover, [recall](/2017/09/10/the-spdz-protocol-part2) that in interactive computations such as the SPDZ protocol it becomes relevant to also consider communication and round complexity, in addition to the typical time complexity. Here, the former measures the number of bits sent across the network, which is a relatively slow process, and the latter the number of synchronisation points needed between the two servers, which may block one of them with nothing to do until the other catches up. Both hence also have a big impact on overall executing time.

Most importantly however, is that the only "native" operations we have in these protocols is addition and multiplication. Division, comparison, etc. can be done, but are more expensive in terms of our three performance measures. Later we shall see how to mitigate some of the issues raised due to this, but here we first recall the basic SPDZ protocol.


## Tensor operations

When we introduced the SPDZ protocol [earlier](/2017/09/03/the-spdz-protocol-part1) we did so in the form of classes `PublicValue` and `PrivateValue` representing respectively a (scalar) value known in clear by both servers and an encrypted value known only in secret shared form. In this blog post, we now instead present it more naturally via classes `PublicTensor` and `PrivateTensor` that reflect the heavy use of [tensors](https://www.tensorflow.org/programmers_guide/tensors) in our deep learning setting. 

```python
class PrivateTensor:
    
    def __init__(self, values, shares0=None, shares1=None):
        if not values is None:
            shares0, shares1 = share(values)
        self.shares0 = shares0
        self.shares1 = shares1
    
    def reconstruct(self):
        return PublicTensor(reconstruct(self.shares0, self.shares1))
        
    def add(x, y):
        if type(y) is PublicTensor:
            shares0 = (x.values + y.shares0) % Q
            shares1 =             y.shares1
            return PrivateTensor(None, shares0, shares1)
        if type(y) is PrivateTensor:
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateTensor(None, shares0, shares1)
            
    def mul(x, y):
        if type(y) is PublicTensor:
            shares0 = (x.shares0 * y.values) % Q
            shares1 = (x.shares1 * y.values) % Q
            return PrivateTensor(None, shares0, shares1)
        if type(y) is PrivateTensor:
            a, b, a_mul_b = generate_mul_triple(x.shape, y.shape)
            alpha = (x - a).reconstruct()
            beta  = (y - b).reconstruct()
            return alpha.mul(beta) + \
                   alpha.mul(b) + \
                   a.mul(beta) + \
                   a_mul_b
```

As seen, the adaptation is pretty straightforward using NumPy and the general form of for instance `PrivateTensor` is almost exactly the same, only occationally passing a shape around as well. There are a few technical details however, all of which are available in full in [the associated notebook](https://github.com/mortendahl/privateml/blob/master/spdz/Tensor%20SPDZ.ipynb). 

```python
def share(secrets):
    shares0 = sample_random_tensor(secrets.shape)
    shares1 = (secrets - shares0) % Q
    return shares0, shares1

def reconstruct(shares0, shares1):
    secrets = (shares0 + shares1) % Q
    return secrets

def generate_mul_triple(x_shape, y_shape):
    a = sample_random_tensor(x_shape)
    b = sample_random_tensor(y_shape)
    c = np.multiply(a, b) % Q
    return PrivateTensor(a), PrivateTensor(b), PrivateTensor(c)
```

As such, perhaps the biggest difference is in the above base utility methods where this shape is used.


# Adapting the Model

While it is in principle possible to compute any function securely with what we already have, and hence also the base model from above, in practice it is relevant to first consider variants of the model that are more MPC friendly, and vice versa. In slightly more picturesque words, it is common to open up our two black boxes and adapt the two technologies to better fit each other.

The root of this comes from some operations being surprisingly expensive in the encrypted setting. We saw above that addition and multiplication are relatively cheap, yet comparison and division with private denominator are not. For this reason we make a few changes to the model to avoid these.

The various changes presented in this section as well as their simulation performances are available in full in the [associated Python notebook](https://github.com/mortendahl/privateml/blob/master/image-analysis/Keras.ipynb).


## Optimizer

The first issue involves the optimizer: while [*Adam*](http://ruder.io/optimizing-gradient-descent/index.html#adam) is a preferred choice in many implementations for its efficiency, it also involves taking a square root of a private value and using one as the denominator in a division. While it is theoretically possible to [compute these securely](https://eprint.iacr.org/2012/164), in practice it could be a significant bottleneck for performance and hence relevant to avoid.

A simple remedy is to switch to the [*momentum SGD*](http://ruder.io/optimizing-gradient-descent/index.html#momentum) optimizer, which may imply longer training time but only uses simple operations.

```python
model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(clipnorm=10000, clipvalue=10000),
    metrics=['accuracy'])
```

An additional caveat is that many optimizers use [clipping](http://nmarkou.blogspot.fr/2017/07/deep-learning-why-you-should-use.html) to prevent gradients from growing too small or too large. This requires a [comparison on private values](https://www1.cs.fau.de/filepool/publications/octavian_securescm/smcint-scn10.pdf), which again is a somewhat expensive operation in the encrypted setting, and as a result we aim to avoid using this technique altogether. To get realistic results from our Keras simulation we increase the bounds as seen above.


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
    Dense(NUM_CLASSES),
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

Another is to introduce a dedicated third server who only does this small computation, doesn't see anything else from the training data, and hence cannot relate the labels with the sample data. Something is still leaked though, and this quantity is hard to reason about.

Finally, we could also replace this [one-vs-all](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) approach with an [one-vs-one](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-one) approach using e.g. sigmoids. As argued earlier this allows us to fully compute the predictions without decrypting. We still need to compute the loss however, which could be done by also considering a different loss function.

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

There are few preprocessing optimisations one could also apply but that we won't consider further here. 

The first is to move the computation of the frozen layers to the input provider so that it's the output of the flatten layer that is shared with the servers instead of the pixels of the images. In this case the layers are said to perform *feature extraction* and we could potentially also use more powerful layers. However, if we want to keep the model proprietary then this adds significant complexity as the parameters now have to be distributed to the clients in some form.

Another typical approach to speed up training is to first apply dimensionality reduction techniques such as a [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). This approach is taken in the encrypted setting in [BSS+'17](https://eprint.iacr.org/2017/857).


# Adapting the Protocol

Having looked at the model we next turn to the protocol: as well shall see, understanding the [operations](https://github.com/wiseodd/hipsternet) we want to perform can help speed things up. 

In particular, a lot of the computation can be moved to the crypto provider, who's generated raw material is independent of the private inputs and to some extend even the model. As such, its computation may be done in advance whenever it's convenient and at large scale.

Recall from earlier that it's relevant to optimise both round and communication complexity, and the extensions suggested here are often aimed at improving these at the expense of additional local computation. As such, practical experiments are needed to validate their benefits under concrete conditions.
 

## Dropout

Starting with the easiest type of layer, we notice that nothing special related to secure computation happens here, and the only thing is to make sure that the two servers agree on which values to drop in each training iteration. This can be done by simply agreeing on a seed value.
 

## Average pooling

The forward pass of average pooling only requires a summation followed by a division with a public denominator. Hence, it can be implemented by a multiplication with a public value: since the denominator is public we can easily find its inverse and then simply multiply and truncate. Likewise, the backward pass is simply a scaling, and hence both directions are entirely local operations.


## Dense layers

The dot product needed for both the forward and backward pass of dense layers can of course be implemented in the typical fashion using multiplication and addition. If we want to compute `dot(x, y)` for matrices `x` and `y` with shapes respectively `(m, k)` and `(k, n)` then this requires `m * n * k` multiplications, meaning we have to communicate the same number of masked values. While these can all be sent in parallel so we only need one round, if we allow ourselves to use another kind of preprocessed triple then we can reduce the communication cost by an order of magnitude.

For instance, the second dense layer in our model computes a dot product between a `(32, 128)` and a `(128, 5)` matrix. Using the typical approach requires sending `32 * 5 * 128 == 22400` masked values per batch, but by using the preprocessed triples described below we instead only have to send `32 * 128 + 5 * 128 == 4736` values, almost a factor 5 improvement. For the first dense layer it is even greater, namely slightly more than a factor 25. 

As also noted [previously](/2017/09/10/the-spdz-protocol-part2/), the trick is to ensure that each private value in the matrices is only sent masked once. To make this work we need triples `(a, b, c)` of random matrices `a` and `b` with the appropriate shapes and such that `c == dot(a, b)`. 

```python
def generate_dot_triple(x_shape, y_shape):
    a = sample_random_tensor(x_shape)
    b = sample_random_tensor(y_shape)
    c = np.dot(a, b) % Q
    return PrivateTensor(a), PrivateTensor(b), PrivateTensor(c)
```

Given such a triple we can instead communicate the values of `alpha = x - a` and `beta = y - b` followed by a local computation to obtain `dot(x, y)`.


```python
class PrivateTensor:
    
    ...
        
    def dot(x, y):
      
        if type(y) is PublicTensor:
            shares0 = x.shares0.dot(y.values) % Q
            shares1 = x.shares1.dot(y.values) % Q
            return PrivateTensor(None, shares0, shares1)
            
        if type(y) is PrivateTensor:
            a, b, a_dot_b = generate_dot_triple(x.shape, y.shape)
            alpha = (x - a).reconstruct()
            beta  = (y - b).reconstruct()
            return alpha.dot(beta) + \
                   alpha.dot(b) + \
                   a.dot(beta) + \
                   a_dot_b
```

Security of using these triples follows the same argument as for multiplication triples: the communicated masked values perfectly hides `x` and `y` while `c` being an independent fresh sharing makes sure that the result cannot leak anything about its constitutes. 

Note that this kind of triple is used in [SecureML](https://eprint.iacr.org/2017/396), which also give techniques allowing the servers to generate them without the help of the crypto provider.


## Convolutions

Like dense layers, convolutions can be treated either as a series of scalar multiplications or as [a matrix multiplication](http://cs231n.github.io/convolutional-networks/#conv), although the latter only after first expanding the tensor of training samples into a matrix with significant duplication. Unsurprisingly this leads to communication costs that in both cases can be improved by introducing another kind of triple.

As an example, the first convolution maps a tensor with shape `(m, 28, 28, 1)` to one with shape `(m, 28, 28, 32)` using `32` filters of shape `(3, 3, 1)` (excluding the bias vector). For batch size `m == 32` this means `7,225,344` communicated elements if we're using only scalar multiplications, and `226,080` if using a matrix multiplication. However, since there are only `(32*28*28) + (32*3*3) == 25,376` private values involved in total (again not counting bias since they only require addition), we see that there is roughly a factor `9` overhead. In other words, each private value is being masked and sent several times. With a new kind of triple we can remove this overhead and save on communication cost: for 64 bit elements this means `200KB` per batch instead of respectively `1.7MB` and `55MB`.

The triples `(a, b, c)` we need here are similar to those used in dot products, with `a` and `b` having shapes matching the two inputs, i.e. `(m, 28, 28, 1)` and `(32, 3, 3, 1)`, and `c` matching output shape `(m, 28, 28, 32)`.


## Sigmoid activations

As done [earlier](/2017/04/17/private-deep-learning-with-mpc/#approximating-sigmoid), we may use a degree-9 polynomial to approximate the sigmoid activation function with a sufficient level of accuracy. Evaluating this polynomial for a private value `x` requires computing a series of powers of `x`, which of course may be done by sequential multiplication -- but this means several rounds and corresponding amount of communication.

As an alternative we can again use a new kind of preprocessed triple that allows us to compute all required powers in a single round. As shown [previously](/2017/09/10/the-spdz-protocol-part2/), the length of these "triples" is not fixed but equals the highest exponent, such that a triple for e.g. squaring consists of independent sharings of `a` and `a**2`, while one for cubing consists of independent sharings of `a`, `a**2`, and `a**3`.

Once we have these powers of `x`, evaluating a polynomial with public coefficients is then just a local weighted sum. The security of this again follows from the fact that all powers in the triple are independently shared.

```python
def pol_public(x, coeffs, triple):
    powers = pows(x, triple)
    return sum( xe * ce for xe, ce in zip(powers, coeffs) )
```

We have the same caveat related to fixed-point precision as [earlier](/2017/09/10/the-spdz-protocol-part2/) though, namely that we need more room for the higher precision of the powers: `x**n` has `n` times the precision of `x` and we want to make sure that it does not wrap around modulo `Q` since then we cannot decode correctly anymore. As done there, we can solve this by introducing a sufficiently larger field `P` to which we temporarily [switch](/2017/09/10/the-spdz-protocol-part2/) while computing the powers, at the expense of two extra rounds of communication.


Practical experiments can show whether it best to stay in `Q` and use a few more multiplication rounds, or perform the switch and pay for conversion and arithmetic on larger numbers. Specifically, for low degree polynomials the former is likely better.


# Proof of Concept Implementation

A [proof-of-concept implementation](https://github.com/mortendahl/privateml/tree/master/image-analysis/) without networking is available for experimentation and reproducibility. Still a work in progress, the code currently supports training a new classifier from encrypted features, but not feature extraction on encrypted images. In other words, it assumes that the input providers themselves run their images through the feature extraction layers and send the results in encrypted form to the servers; as such, the weights for that part of the model are currently not kept private. A future version will address this and allow training and predictions directly from images by enabling the feature layers to also run on encrypted data.

```python
from pond.nn import Sequential, Dense, Sigmoid, Dropout, Reveal, Softmax, CrossEntropy
from pond.tensor import PrivateEncodedTensor

classifier = Sequential([
    Dense(128, 6272),
    Sigmoid(),
    Dropout(.5),
    Dense(5, 128),
    Reveal(),
    Softmax()
])

classifier.initialize()

classifier.fit(
    PrivateEncodedTensor(x_train_features), 
    PrivateEncodedTensor(y_train), 
    loss=CrossEntropy(), 
    epochs=3
)
```

The code is split into several Python notebooks, and comes with a set of precomputed weights that allows for skipping some of the steps:

- The first one deals with [pre-training on the public data](https://github.com/mortendahl/privateml/tree/master/image-analysis/Pre-training.ipynb) using Keras, and produces the model used for feature extraction. This step can be skipped by using the repository's precomputed weights instead.

- The second one applies the above model to do [feature extraction on the private data](https://github.com/mortendahl/privateml/tree/master/image-analysis/Feature%20extraction.ipynb), thereby producing the features used for training the new encrypted classifier. In future versions this will be done by first encrypting the data. This step cannot be skipped as the extracted data is too large.

- The third takes the extracted features and [trains a new encrypted classifier](https://github.com/mortendahl/privateml/tree/master/image-analysis/Fine-tuning.ipynb). This is by far the most expensive step and may be skipped by using the repository's precomputed weights instead.

- Finally, the fourth notebook uses the new classifier to perform [encrypted predictions](https://github.com/mortendahl/privateml/tree/master/image-analysis/Prediction.ipynb) from new images. Again feature extraction is currently done unencrypted.

Running the code is a matter of cloning the repository

```bash
$ git clone https://github.com/mortendahl/privateml.git && \
  cd privateml/image-analysis/
```

installing the dependencies
  
```bash
$ pip3 install jupyter numpy tensorflow keras h5py
```

launching a notebook

```bash
$ jupyter notebook
```

and navigating to either of the four notebooks mentioned above.


<!--
## Running on GCE

Since especially the encrypted training is a rather lengthy process, it might be worth running at least this part on e.g. a remote cloud instance. To use the [Google Compute Engine](https://cloud.google.com/compute/) one can do the following, after setting up [`gcloud`](https://cloud.google.com/sdk/) (which is also available in Homebrew as `brew cask info google-cloud-sdk`).

We first set up a fresh compute instance to function as out notebook server and connect to it.

```bash
laptop$ gcloud compute instances create server \ 
          --custom-cpu=1 \
          --custom-memory=6GB
          
laptop$ gcloud compute ssh server -- -L 8888:localhost:8888
```

Once connected we install dependencies, pull down the notebooks, and launch Jupyter. Note that we do the latter in a screen to let the notebook computations run even if we disconnect our SSH session.

```bash
server$ sudo apt-get update && \
        sudo apt-get install -y python3 python3-pip git && \
        sudo pip3 install jupyter numpy tensorflow keras
        
server$ git clone https://github.com/mortendahl/privateml.git && \
        cd privateml/image-analysis/
        
server$ screen jupyter notebook
```

```bash

```

```bash
server$ screen jupyter notebook
```

`ctrl+a d`

```bash
laptop$ gcloud compute ssh server -- -L 8888:localhost:8888
server$ screen -r
```

```bash
## Stop GCP instance
gcloud compute instances stop server
```
-->


# Thoughts

As always, when previous thoughts and questions have been answered there is already a new batch waiting.


## Generalised triples

When seeking to reduce communication, one may also wonder how much can be pushed to the preprocessing phase in the form of additional types of triples.

As mentioned several times (and also suggested in e.g. [BCG+'17](https://eprint.iacr.org/2017/1234)), we typically seek to ensure that each private value is only sent masked once. So if we are e.g. computing both `dot(x, y)` and `dot(x, z)` then it might make sense to have a triple `(r, s, t, u, v)` where `r` is used to mask `x`, `s` to mask `y`, `u` to mask `z`, and `t` and `u` are used to compute the result. This pattern happens during training for instance, where values computed during the forward pass are sometimes cached and reused during the backward pass. 

Perhaps more importantly though is when we are only making predictions with a model, i.e. computing with fixed private weights. In this case we only want to [mask the weights once and then reuse](/2017/09/10/the-spdz-protocol-part2) these for each prediction. Doing so means we only have to mask and communicate proportionally to the input tensor flowing through the model, as opposed to propotionally to both the input tensor and the weights, as also done in e.g. [JVC'18](https://arxiv.org/abs/1801.05507). More generally, we ideally want to communicate proportionally only to the values that change, which can be achieved (in an amortised sense) using tailored triples.

Finally, it is in principle also possible to have [triples for more advanced functions](/2017/09/10/the-spdz-protocol-part2) such as evaluating both a dense layer and its activation function with a single round of communication, but the big obstacle here seems to be scalability in terms of triple storage and amount of computation needed for the recombination step, especially when working with tensors.


## Activation functions

A natural question is which of the other typical activation functions are efficient in the encrypted setting. As mentioned above, [SecureML](https://eprint.iacr.org/2017/396) makes use of ReLU by temporarily switching to garbled circuits, and [CryptoDL](https://arxiv.org/abs/1711.05189) gives low-degree polynomial approximations to both Sigmoid, ReLU, and Tanh (using [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) for [better accuracy](http://www.chebfun.org/docs/guide/guide04.html#47-the-runge-phenomenon)).

It may also be relevant to consider non-typical but simpler activations functions, such as squaring as in e.g. [CryptoNets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/), if for nothing else than simplifying both computation and communication.


## Garbled circuits

While mentioned above only as a way of securely evaluating more advanced activation functions, [garbled](https://oblivc.org/) [circuits](https://github.com/encryptogroup/ABY) could in fact also be used for larger parts, including as the main means of secure computation as done in for instance [DeepSecure](https://arxiv.org/abs/1705.08963). 

Compared to e.g. SPDZ this technique has the benefit of using only a constant number of communication rounds. The downside is that operations are now often happening on bits instead of on larger field elements, meaning more computation is involved.


## Precision

A lot of the research around [federated learning](https://research.googleblog.com/2017/04/federated-learning-collaborative.html) involve [gradient compression](https://arxiv.org/abs/1610.05492) in order to save on communication cost. Closer to our setting we have [BMMP'17](https://eprint.iacr.org/2017/1114) which uses quantization to apply homomorphic encryption to deep learning, and even [unencrypted](https://arxiv.org/abs/1610.02132) [production-ready](https://www.tensorflow.org/performance/quantization) systems often consider this technique as a way of improving performance also in terms of [learning](https://ai.intel.com/lowering-numerical-precision-increase-deep-learning-performance/).


## Floating point arithmetic

Above we used a fixed-point encoding of real numbers into field elements, yet unencrypted deep learning is typically using a floating point encoding. As shown in [ABZS'12](https://eprint.iacr.org/2012/405) and [the reference implementation of SPDZ](https://github.com/bristolcrypto/SPDZ-2/issues/7), it is also possible to use the latter in the encrypted setting, apparently with performance advantages for certain operations.


## GPUs

Since deep learning is typically done on GPUs today for performance reasons, it is natural to consider whether similar speedups can be achieved by applying them in MPC computations. Some [work](https://www.cs.virginia.edu/~shelat/papers/hms13-gpuyao.pdf) exist on this topic for garbled circuits, yet it seems less popular in the secret sharing setting of e.g. SPDZ. 

Biggest problem here might be maturity and availability of arbitrary precision arithmetic on GPUs (but see e.g. [this](http://www.comp.hkbu.edu.hk/~chxw/fgc_2010.pdf) and [that](https://github.com/skystar0227/CUMP)) as needed for computations on field elements larger than e.g. 64 bits. Two things might be worth keeping in mind here though: firstly, while the values we compute on are larger than those natively supported, they are still bounded by the modulus; and secondly, we can do our secure computations over a ring instead of a field.

<!--
One potential remedy is to decompose numbers using the [CRT](https://en.wikipedia.org/wiki/Chinese_remainder_theorem) into several components that are computed on in parallel. For this to work we would need to do our computations over a ring instead of a field, since our modulus must now be a composite number as opposed to a prime.
-->

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
