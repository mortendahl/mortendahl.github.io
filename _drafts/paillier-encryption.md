---
layout:     post
title:      "A Primer on Paillier Encryption"
subtitle:   "Overview, Applications, and Implementation"
date:       2017-02-08 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---



# The Paillier Encryption Scheme

- Idea of mapping into algebraic structure to hide message
- should be easy in one direction, hard in the opposite without secret key
- in a HE scheme the mapping should preserve certain operations
- PHE vs SHE vs FHE; benchmarks?
- link to paper and [Introduction to Modern Cryptography](http://www.cs.umd.edu/~jkatz/imc.html).

## Abstract properties

- `enc(m, r) == e(m, r)`
- `dec(e(m,r)) == m, r`
- `add`: `e(m, r) + e(n, s) == e(m+n, r*s)`
- `smul`: `n * e(m, r) == e(m*n, r^n)`

## A deterministic simplification

- No randomness component
- mapping `g^m` for `Zn` and that it preserves some operations
- security rests on ...

## Paillier's probabilistic mapping

- The need for probabilistic encryption
- full scheme with mapping `g^m * r^n` for `Zn x Zn*`
- security from hidden structure
- there are things we want from the mapping, besides being efficient to compute: 
    - efficient to invert knowning decryption key
    - provide security


# Applications


## Tips and Tricks

Encodings
- 8bit, 16bit, 32bit int
- signed
- rational
- packed

### Packed

Or *vector* encoding uses the fact that the modulus `N` allows us to operate on very large plaintexts, in fact often much larger than the values we typically need for an application: `N` is something like 2048 bits where most programs uses 32 or 64 bits numbers.

`[1,2,3,4]`

```
000001000002000003000004
== 1 * 10^18    + 2 * 10^12    + 3 * 10^6     + 4 * 10^0
== 1 * (10^6)^3 + 2 * (10^6)^2 + 3 * (10^6)^1 + 4 * (10^6)^0
== 1 * B^3      + 2 * B^2      + 3 * B^1      + 4 * B^0
```

for `B = 10^6`.

```
5 * encode([1,2,3,4])
== 5 * 000001000002000003000004
== 5 * (   1 * B^3 +     2 * B^2     + 3 * B^1     + 4 * B^0)
==     (5*1) * B^3 + (5*2) * B^2 + (5*3) * B^1 + (5*4) * B^0
== encode(scalar_mul([1,2,3,4], 5))
```



### Multiplication

Say we know ciphertexts `E(x)` and `E(y)` that we wish to multiply to obtain `E(x * y)`. While there is no known local way of during this for the Paillier scheme, by instead executing a simple protocol together with someone knowing the decryption key. If we don't mind that the decryption oracle learns `x` and `y` then of course we may simply ask him to decrypt, compute `z = x * y`, and send us back a fresh encryption of `z`.

However, we may want to mirror the security guarantees from addition, meaning the decryption oracle should not be able to learn `x`, `y`, nor `z` unless we explicitly want him to.

The protocol assumes that Alice starts out with inputs `E(x)` and `E(y)`, and in the end leaves Alice with `E(x * y)` without revealing anything to the oracle. It is as follows:
1. Alice picks two values `r` and `s` uniformly at random from `Zn`. She then computes `E(x) * r` and `E(y) * s`, and sends these to the oracle
2. The oracle decrypts `x * r` and `y * s`, multiplies, and sends back `E( (x * r) * (y * s) )`
3. Alice computes `t = (r * s)^-1` and in turn `E( (x * r) * (y * s) ) * t`

Two questions may come into mind. The first one is why this doesn't reveal anything to the oracle, the second whether Alice can always be sure to compute `t` (keeping in mind that not all values in `Zn` has a multiplicative inverse).



Knowing `E(x * r)` and 
`E(y * s)`



## Voting

## Federated Learning

## General MPC
add for free, mult requires a bit more; see Carmit's book



# Extensions

## Threshold decryption

- assume split key has already been generated; link to Gert's paper

## Proofs

- correct decryption
- correct HE operations
- knowledge of plaintext



# Implementation

- base on good library for arbitrary precision integer ops (framp, GMP)
- encoding
- keygen
- those used in rust-paillier
  - binary exp
  - binary? gcd
  - mult encryption
  - CRT decryption
  - CRT encryption
  - etc
- precompute randomness
- side-channel challenges
- refs:
  - n1analytics (java-python)
  - rust-paillier
  - benchmark blog post
- GPUs?

## Key generation

## Encryption

## Decryption

## Homomorphic operations