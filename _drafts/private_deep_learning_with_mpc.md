---
layout:     post
title:      "Private Deep Learning with MPC"
subtitle:   "A Simple Tutorial from Scratch"
date:       2017-04-02 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---

Inspired by a recent blog post about mixing deep learning and homomorphic encryption (see [*Building Safe A.I.*](http://iamtrask.github.io/2017/03/17/safe-ai/)) I thought it'd be interesting do to the same, but replacing homomorphic encryption with *secure multi-party computation*.


## HE vs MPC

Homomorphic encryption and secure multi-party computation are two closely related fields in modern cryptography, with one often using techniques from the other in order to solve roughly the same problem: privately computing on data, i.e. without revealing the inputs on which a function is being applied. As such, one is often replaceable by the other.

However, where they differ today can roughly be characterized by MPC exchanging computation for interaction. Or in other words, by having several parties instead of only one be part of the computation process, MPC currently offers significantly better performance compared to HE, to the point where one can argue that it's a significantly more mature technology.

But let's move on to something concrete in the hope of illustrating this.


## Multi-Party Computation

Assume we have three parties with inputs `x0`, `x1`, and `x2` respectively, and that they would like to compute some function `f` on these inputs: `y = f(x1, x2, x3)`. Here, the inputs may for instance be training sets and `f` a machine learning process that maps these to a trained model.

Moreover, assume that party `Pi` would like to keep `xi` private from everyone else, and is only interested in sharing the final result `y` (and implicitly what can be learned about `xi` from `y`).

This is the problem solved by MPC, and notice straight away the focus on inputs coming from several parties. In particular, MPC is naturally about mixing data from several parties, without any requirements that this data for instance being encrypted under the same key.


## A simple protocol

To start the tutorial, let's build a basic protocol that allows the three parties to start computing on their inputs.

The first step is to define a [field]() in which the inputs must be from, and in which the computation is to take place. Here, we'll pick a prime field `Zp = { 0, 1, ..., p-1 }` for some prime number `p`.

Next we'll need our first building block: [*secret sharing*](). The idea is simple. We want to take the private inputs, and split each in three shares in such a way that if anyone sees less than the three shares, then nothing at all is revealed about the input; yet, by seeing all three shares, the input can easily be reconstructed.

We'll use the following simple one. To secret share `xi`:

1. pick two random numbers `xi1` and `xi2` from `Zp`
2. compute `xi3 = xi - xi1 - xi2 mod p`
3. let `xi1`, `xi2`, and `xi3` be the three shares

This has the desired security properties (seeing less than three shares reveals nothing about `xi`) and the desired reconstruction properties (seeing `xi1`, `xi2`, and `xi3` one can reconstruct `xi` as `xi1 + xi2 + xi3 mod p`).
