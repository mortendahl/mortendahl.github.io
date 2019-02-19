---
layout:     post
title:      "An Illustrated Primer on Paillier"
subtitle:   "Overview of a Homomorphic Encryption Scheme"
date:       2019-01-15 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
summary:    "The Paillier homomorphic encryption scheme is not only interesting for allowing computation on encrypted values, it also provides an excellent illustration of modern security assumptions and a beautiful application of abstract algebra."
---

<em><strong>TL;DR:</strong> the Paillier encryption scheme is not only interesting for allowing computation on encrypted values, it also provides an excellent illustration of modern security assumptions and a beautiful application of abstract algebra.</em>

20 year anniversary (15 april)

[`python-paillier`](https://github.com/n1analytics/python-paillier)

# Basics

The Paillier encryption scheme is defined by a mapping `enc` that turns a message and randomness pair `(m, r)` into a ciphertext `c = enc(m, r)`. Here, `m` is the message we wish to encrypt while `r` is an independent random value that makes Paillier a [probabilistic encryption scheme](https://en.wikipedia.org/wiki/Probabilistic_encryption): even if we encrypt the same message `m` several times we end up different ciphertexts due to different randomness being used, in turn making it impossible to guess `m` by brute force even if it only takes on a few different values (as one would have to guess `r` as well, which can be picked large enough to make it impossible to guess, or at least only with negligable probability).


<img src="/assets/paillier/probabilistic.png" style="width: 50%;"/>

  This is done relative to a specific but here implicit public *encryption key* which 

To fully define this mapping we note that `m` can be any value from `Zn = {0, 1, ..., n-1}` while `r` is limited to those numbers in `Zn` that have a multiplication inverse, i.e. `Zn*`; together this implies that `c` is a value in `Zn^2*`, ie. amoung the values in `{0, 1, ..., n^2 - 1}` that have multiplication inverses. Finally, `n = p * q` is a typical RSA modulus consisting of two primes `p` and `q`, and `g` is a fixed generator, typically picked as `g = 1 + n`.

```python
P = sample_prime(2048)
Q = sample_prime(2048)

N = P * Q
NN = N * N
G = 1 + N
```

```python
def enc(m, r):
    Gm = pow(G, m, NN)
    rN = pow(r, N, NN)
    c = (Gm * rN) % NN
    return c
```


, as well as its inverse `dec` that recovers both `m` and `r` from a ciphertext. Here, `enc` is implicitly using a public encryption key `ek` that everyone can know while `dec` is using a private decryption key `dk` that only those allowed to decrypt should know.

The `m` is typically the message we want to encrypt while `r` is a randomness picked uniformly at random for each new encryption; as such, we get a 

## Homomorphic properties

Besides being probabilistic, a highly attractive property of the Paillier is that it allows for computing on encrypted values through homomorphic properties. In particuar, we can combine encryptions of `m1` and `m2` to get an encryption of `m1 + m2`, and an encryption of `m` with a constant `k` to get an encryption of `m * k`, in both cases without decrypting anything (and hence without learning anything about `m1` and `m2`.

### Addition

To see how this works let's start with addition: given `c1 = enc(m1, r1)` and `c2 = enc(m2, r2)` we compute `c1 * c2 == (g^m1 * r1^n) * (g^m2 * r2^n) == g^(m1 + m2) * (r1 * r2)^n == enc(m1 + m2, r1 * r2)`, leaving out the `mod n^2` to simplify notation.

```python
def add_encrypted(c, d):
    e = (c * d) % NN
    return e
```

```python
def add_plain(c, k):
    d = enc(k, 1)
    e = (c * d) % NN
    return e
```

### Multiplication

Likewise, given `c = enc(m, r)` and a `k` we compute `c^k = (g^m * r^n) ^ k == g^(m * k) * (r^k)^n == enc(m * k, r ^ k)`, again leaving out `mod n^2`.

```python
def mul_plain(c, k):
    d = pow(c, k, NN)
    return d
```

Note that the results do not exactly match the original form of encryptions: in the first case the resulting randomness is `r1 * r2` and in the latter it is `r^k`, whereas for fresh encryptions these values are uniformaly random. In some cases this may leak something about otherwise private values (see e.g. the voting application below TODO) and as a result we sometimes need a *re-randomize* operation that erases everything about how a ciphertext was created by making it look exactly as a fresh one. We do this by simply multiplying by a fresh encryption of zero: `enc(m, r) * enc(0, s) == enc(m, r*s) == enc(m, t)` for a uniformly random `t` if `s` is independent and uniformly random.

As we will see in more detail below this opens up for some powerful applications, including electronic voting, private machine learning, and general purpose secure computation. But first it's interesting to go into more details about how decryption works and why the scheme is secure.


### Re-randomization

```python
def rerandomize(c):
    s = random.randrange(N)
    d = enc(0, s)
    e = (c * d) % NN
    return e
```

could be done lazily

# Algebraic Interpretation

`(Zn2*, *) ~ (Zn, +) x (Zn*, *)`

HE
- `c1 * c2 == (m1, r1) * (m2, r2) == (m1+m2, r1*r2)`

- `c ^ k == (m, r) ^ k == (m * k, r^k)`

- `inv(c) == inv((m, r)) == (-m, r^-1)`

- `c * s^n == (m, r) * (0, s) == (m, r*s) == (m, t)`

dec
- `c^phi == (m*phi, r^phi) == (m*phi, 1) == g^(m*phi)`

<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>

<img src="/assets/paillier/nn.png" style="width: 50%;"/>
<img src="/assets/paillier/nninverse.png" style="width: 50%;"/>
<img src="/assets/paillier/nnstar.png" style="width: 50%;"/>



# Decryption

## Opening

# Security

In order to better understand the scheme, including its security, it's instructive to start with a fool-proof scheme and see how it compares.

Concretely, say you want to encrypt a value `x` from `Zn^2*`. One way of doing this is to pick a uniformly random value `s` also from `Zn2*` and multiply these together: `c = x * s mod n^2`. Since this is in fact the one-time pad (over a multiplicative group) **TODO TODO TODO really?** we get perfect secrecy, i.e. a theoretical guarantee that nothing whatsoever is leaked about `x` by `c` as long as the mask `s` remains unknown.

The problem with this scheme is when we want to decrypt `c` without knowing `s` (which would otherwise have to be communicated somehow). One might try to get rid of `s` by raising `c` to the order of `s`, which by TODO gives us `c ^ ord(s) == (x * s) ^ ord(s) == x^ord(s) * 1 == x^ord(s)`. This indeed removed `s`, but it is not clear how to extract `x` from `x^ord(s)`. In fact, in the case where `ord(s)` is `phi(n^2)` this is impossible.

So what if we 

let's see an example. say our modulus is `n^2` as in the Paillier scheme. these numbers we can convenient put into an `n` by `n` grid as follows.

(TODO)

if we then filter out those values that don't have multiplicative inverses we get the following grid

(TODO)

which shows that the remaining values are exactly those that do not share any factors with `n`. this means that an easy way to characterize these numbers is by 
`[x for x in range(NN) if gcd(x, N) == 1]`. moreover, multiplying any two values with multiplicative inverses implies that their product also has an multiplicative inverse; in other words, as long as we multiply numbers from `Zn*` we are guaranteeded to stay within `Zn*`.

To understand the scheme, not least in terms of security and how decryption works, it is useful to try to reconstruct the process of building it. moreover, some of the principles we'll see here are also used in more modern schemes, in particular around how security is argued. of course this likely involved much more back and forward originally, or perhaps even an entirely different line of thought -- short of asking Pascal Paillier himself we might never know. Nonetheless, let's go back to the late 1990s where RSA could very well have been a heavy inspiration.

the reasons for switching from computing mod `n` as in RSA to computing mod `n^2` instead of something else will become clearer later, but for now let's just say that in order to get a probabilistic scheme we have to let room for some randomness `r` (in ElGamal, another probabilistic scheme from roughly the same period does this by letting ciphertexts be pairs of values instead; however in Paillier we simply double the modulus).





# DUMP




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

Python object `Ciphertext` with two private field; use math to prevent access instead. impl eq, dec according to logic.

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

## Linear prediction


## Voting

## Federated Learning

## General MPC
add for free, mult requires a bit more; see Carmit's book






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


# Extensions

## Threshold decryption

- assume split key has already been generated; link to Gert's paper

## Proofs

- correct decryption
- correct HE operations
- knowledge of plaintext
