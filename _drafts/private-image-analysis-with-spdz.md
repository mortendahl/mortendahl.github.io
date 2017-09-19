---
layout:     post
title:      "Private Image Analysis with SPDZ"
subtitle:   "Mixing Public and Private Data with Transfer Learning"
date:       2017-09-01 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---

`simple-spdz` in rust
- only the online
- raspberry pi
- mnist data set split between three parties
- dimension reduction using PCA

# Overview

- CNN in Keras

Pooling in MPC:
- max pooling inefficient
- average pool, scaled mean pool in CryptoNets
- doing entirely out of fashion: https://arxiv.org/abs/1412.6806 and http://cs231n.github.io/convolutional-networks/ -- use larger stride in CONV layer once in a while

Secure computation is not yet a complete blackbox (if we want performance): while possible to compute any function with general protocols, understanding what computation we want to perform can help speed up things. It is not a custom protocol but simply adapting the general approach.

# Character Recognition with CNNs

## Overview

### Keras

### TensorFlow

### Numpy

## Base line using custom (open) data type for fixed-point precision


# Secure Computation with SPDZ

Bristol blog posts:
- https://bristolcrypto.blogspot.fr/2016/10/what-is-spdz-part-1-mpc-circuit.html
- http://bristolcrypto.blogspot.fr/2016/10/what-is-spdz-part-2-circuit-evaluation.html
- https://bristolcrypto.blogspot.fr/2016/11/what-is-spdz-part-3-spdz-specifics.html
- https://www.cs.bris.ac.uk/Research/CryptographySecurity/SPDZ/
- papers

we can skip the resharing phase

## Encoding

As before we need to encode fixed point numbers as algebraic elements; and as in that case, our encoding is correct as long as no wrap-around with the modulo happened. Concretely this means that we have to be careful with precision since every multiplication doubles it, and hence will eventually cause a wrap-around if not kept in control.

## Beaver triples

(or *circuit randomization*)

let `epsilon = x - a`. then we know from [the binomal theorem](https://en.wikipedia.org/wiki/Binomial_theorem) that e.g. `x^2 == (epsilon + a)^2 == epsilon^2 + 2 * epsilon * a + a^2`, and in general that `x^n` can be expressed as a weighted sum of powers of `epsilon` and `a`, using the binomal coefficients as weights. in other words, if we know the right powers of `epsilon` and `a`, then computing `x^n` is a linear operation that can be performed locally. in still other words, `x^n` is a linear combination of the powers of `a`....

we have to tweak the field size vs number of rounds: the more exp we want to do in one round, the larger the field must be to avoid the fixed-point encoding wrapping around; larger fields means slower computations but fewer rounds. we need log(exponent) rounds

# Private Image Recognition

general idea of CNNs for image recon:
- http://cs231n.github.io/convolutional-networks/

- [CryptoNets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/)
- [SecureML](https://eprint.iacr.org/2017/396)

# Performance Improvements

## Dimension reduction

- PCA

## Transfer Learning

## Gradient Compression

- https://arxiv.org/pdf/1610.02132.pdf

# Dump

## DL

- http://cs231n.github.io/

## CNN



in numpy:
- https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
- https://github.com/andersbll/nnet

## Transfer learning

- http://cs231n.github.io/transfer-learning/
- https://yashk2810.github.io/Transfer-Learning/ "It is always recommended to use transfer learning in practice."
- https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8
- http://ruder.io/transfer-learning/
- https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py
- https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
- https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html