---
layout:     post
title:      "Paillier Encryption"
subtitle:   "Overview, Implementation, and Applications"
date:       2017-02-08 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---



# The Paillier Encryption scheme

- Idea of mapping into algebraic structure to hide message
- should be easy in one direction, hard in the opposite without secret key
- in a HE scheme the mapping should preserve certain operations
- PHE vs SHE vs FHE; benchmarks?
- link to paper and [Introduction to Modern Cryptography](http://www.cs.umd.edu/~jkatz/imc.html).

## A deterministic simplification

- No randomness component
- mapping `g^m` for `Zn` and that it preserves some operations
- security rests on ...

## Paillier's probabilistic mapping

- The need for probabilistic encryption
- full scheme with mapping `g^m * r^n` for `Zn x Zn*`


# Implementation

- encoding
- keygen
- those used in rust-paillier
  - binary exp
  - binary? gcd
  - mult encryption
  - CRT decryption
  - etc
- precompute randomness
- side-channel challenges
- refs:
  - n1analytics (java-python)
  - rust-paillier
  - benchmark blog post


# Extensions

## Threshold decryption

- assume split key has already been generated; link to Gert's paper

## Proofs

- correct decryption
- correct HE operations
- knowledge of plaintext


# Tips and Tricks

Encodings
- signed
- rational
- packed

Multiplication


# Applications

## Voting

## Federated Learning

## General MPC
add for free, mult requires a bit more; see Carmit's book
