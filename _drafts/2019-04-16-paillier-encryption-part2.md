---
layout:     post
title:      "Paillier Encryption, Part 2"
subtitle:   "Applications in Machine Learning"
date:       2019-04-16 12:00:00
author:     "Morten Dahl"
header-img: "assets/paillier/autostereogram-mars-rover.jpeg"
summary:    "In the second part of the series on Paillier encryption we focus on it's use in privacy-preserving machine learning, including private predictions and secure aggregation for federated learning, and we go through how a bit of interaction allows us to support more operations and build general two-party secure computation."
---

<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>

With a basic understanding of the Paillier homomorphic encryption scheme we next turn to how it may be used in privacy-preserving machine learning. This includes serving private predictions and performing secure aggregation in federated learning. We also go through how to use Paillier for general two-party secure computation, using a bit of interaction to extend the homomorphic operations supported.

Voting comes up in a lot of papers but here we will look at other uses.

```python
class Ciphertext
    
    def __init__(self, ek, raw):
        self.ek = ek
        self.raw = raw

    def __add__(self, other):

        if isinstance(other, int):
            c1 = self.raw
            x2 = other
            return Ciphertext(self.ek, add_plain(self.ek, c1, x2))

        if isinstance(other, Ciphertext):
            c1 = self.raw
            c2 = self.raw
            return Ciphertext(self.ek, add_cipher(self.ek, c1, c2))

    def __mul__(self, other):

        if isinstance(other, int):
            c1 = self.raw
            x2 = other
            return Ciphertext(self.ek, mul_plain(self.ek, c1, x2))
```


# Privacy-Preserving Predictions

Prediction using a linear model

Linear model

see more examples in python-paillier and Andrew's blog post


# Secure Aggregation

Secure Aggregation for Federated Learning

## Packing

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

# General Two-Party Secure Computation

How do we get multiplication of ciphertexts as well?

dealing with fixedpoints (stat masking)

setting: server holds encryptions, client holds decryption key

add: `add_cipher`

mul: server sends 

```python
with server:
    c_masked = add_cipher(c, a)
    d_masked = add_cipher(d, b)

with client:
    e_masked = enc(dec(c_masked) * dec(d_masked))

with server:
    e = e_masked 
```


**insecure; follow CDN'01 and DNT'12 instead (convert to additive sharing first)**

Say we know ciphertexts `E(x)` and `E(y)` that we wish to multiply to obtain `E(x * y)`. While there is no known local way of during this for the Paillier scheme, by instead executing a simple protocol together with someone knowing the decryption key. If we don't mind that the decryption oracle learns `x` and `y` then of course we may simply ask him to decrypt, compute `z = x * y`, and send us back a fresh encryption of `z`.

However, we may want to mirror the security guarantees from addition, meaning the decryption oracle should not be able to learn `x`, `y`, nor `z` unless we explicitly want him to.

The protocol assumes that Alice starts out with inputs `E(x)` and `E(y)`, and in the end leaves Alice with `E(x * y)` without revealing anything to the oracle. It is as follows:
1. Alice picks two values `r` and `s` uniformly at random from `Zn`. She then computes `E(x) * r` and `E(y) * s`, and sends these to the oracle
2. The oracle decrypts `x * r` and `y * s`, multiplies, and sends back `E( (x * r) * (y * s) )`
3. Alice computes `t = (r * s)^-1` and in turn `E( (x * r) * (y * s) ) * t`

Two questions may come into mind. The first one is why this doesn't reveal anything to the oracle, the second whether Alice can always be sure to compute `t` (keeping in mind that not all values in `Zn` has a multiplicative inverse).

Knowing `E(x * r)` and 
`E(y * s)`

# Extensions

## Threshold decryption

- assume split key has already been generated; link to Gert's paper

## Proofs

- correct decryption
- correct HE operations
- knowledge of plaintext

[`zk-paillier`](https://github.com/KZen-networks/zk-paillier)
