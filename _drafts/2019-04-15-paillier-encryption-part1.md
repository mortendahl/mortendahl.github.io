---
layout:     post
title:      "Paillier Encryption, Part 1"
subtitle:   "Basics of a Homomorphic Encryption Scheme"
date:       2019-04-15 12:00:00
author:     "Morten Dahl"
header-img: "assets/paillier/autostereogram-space-shuttle.jpeg"
summary:    "The Paillier homomorphic encryption scheme is not only interesting for allowing computation on encrypted values, it also provides an excellent illustration of modern security assumptions and a beautiful application of abstract algebra. In this first part of a series we cover the basics and the homomorphic operations it supports."
---

<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>

<em><strong>TL;DR:</strong> the Paillier encryption scheme is not only interesting for allowing computation on encrypted values, it also provides an excellent illustration of modern security assumptions and a beautiful application of abstract algebra.</em>

In this series of blog post we walk through and explain [Paillier encryption](https://en.wikipedia.org/wiki/Paillier_cryptosystem), a so called *partially homomorphic scheme* [first described](https://link.springer.com/chapter/10.1007%2F3-540-48910-X_16) by [Pascal Paillier](https://twitter.com/pascal_paillier) exactly 20 years ago. [More advanced schemes](https://en.wikipedia.org/wiki/Homomorphic_encryption) have since been developed, allowing more operations to be performed on encrypted data, yet Paillier encryption remains relevant not only for understanding modern cryptography but also from an applications point of view, as illustrated recently by for instance Google's [Private Join and Compute](https://eprint.iacr.org/2019/723) or the Snips' [Secure Distributed Aggregator](https://eprint.iacr.org/2017/643).

We will go through the basics, a few applications in privacy-preserving machine learning, why the scheme is believed to be secure, and some of the tricks that can be used to implement it more efficiently. As always, the full source code is available for experimentation, but inspired by the excellent [A Homomorphic Encryption Illustrated Primer](https://blog.n1analytics.com/homomorphic-encryption-illustrated-primer/) by [Stephen Hardy](https://twitter.com/proximation) and [Differential Privacy: An illustrated Primer](https://github.com/frankmcsherry/blog/blob/master/posts/2016-02-06.md) by [Frank McSherry](https://twitter.com/frankmcsherry) we also try to give a more visual presentation of material that is [typically](https://www.cs.umd.edu/~jkatz/imc.html) offered mostly in the form of equations.

<em>Parts of this blog post are based on [work](https://medium.com/snips-ai/benchmarking-paillier-encryption-15631a0b5ad8) done at [Snips](https://snips.ai).</em>

# Basics

Paillier is a [public-key encryption scheme](https://en.wikipedia.org/wiki/Public-key_cryptography), where a *public encryption key* allows anyone to turn a plaintext value into a ciphertext, but only those with the *private decryption key* can decrypt and go the other direction to recover the plaintext value inside a ciphertext. [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)) is perhaps the most well-known scheme offering this property, yet Paillier offers something that RSA does not: it allows anyone to compute on data while it remains encrypted.

Encryption is defined by a function `enc` that maps plaintext `x` and randomness `r` into ciphertext `c = enc(x, r, ek)`, relative to a given encryption key `ek` that we occasionally omit to simplify the notation. The effect of having the randomness `r` is that we end up with different ciphertexts `c` even if we encrypt the same `x` several times: if `r1`, `r2`, and `r3` are different then so are `c1 = enc(x, r1)`, `c2 = enc(x, r2)`, and `c3 = enc(x, r3)`.

<img src="/assets/paillier/probabilistic.png" style="width: 50%;"/>

This means that even if an adversary who has obtained a ciphertext `c` knows that there are only a few possibilities for `x`, say only `0` or `1`, then they cannot simply compare `c` to known all possible encryptions of `0` or all possible encryptions of `1` if there are sufficiently many choices for the randomness. these then this becomes impractical, and we have effectively removed the adversary's ability to guess `x` by removing its ability of check whether or not a guess was correct. Of course, there may be other ways for them to check a guess or learn something about `x` from `c` in general, and we shall return to security of the scheme in much more detail later.

<img src="/assets/paillier/enc.png" style="width: 50%;"/>

This means that even if an adversary who has obtained a ciphertext `c` knows that there are only a few possibilities for `x`, say only `0` or `1`, they cannot simply compare `c` to `c0 = enc(0, r0)` and `c1 = enc(1, r1)` without first guessing `r0` and `r1`. If there are sufficiently many choices for these then this becomes impractical, and we have effectively removed the adversary's ability to guess `x` by removing its ability of check whether or not a guess was correct. Of course, there may be other ways for them to check a guess or learn something about `x` from `c` in general, and we shall return to security of the scheme in much more detail later.

Let us take a closer look at that, and plot where encryptions of zero lies.

Below we will see concrete examples

When `r` is chosen independently at random, Paillier encryption becomes what is known as a [probabilistic encryption scheme](https://en.wikipedia.org/wiki/Probabilistic_encryption), an often desirable property of encryption schemes per the discussion above.

As detailed below, `x`, `r`, and `c` are all large integers.

  having enough  One motivation of including a randomness in the ciphertext is that it makes it impractical to guess `x` simply by trying to encrypt different values and check whether the ciphertexts match, as one would also have to guess `r` which is typically picked from a very large set: it is allowed to be any element from `Zn = {0, 1, ..., n-1}`, where `n` is a number with between 2000 and 4000 bits (we a lying a tiny bit here, but will return to that soon).


 Formally this makes Paillier a [probabilistic encryption scheme](https://en.wikipedia.org/wiki/Probabilistic_encryption), which is often a desirable property of encryption schemes.


```python
def keygen(n_bitlength=2048):
    p = sample_prime(n_bitlength // 2)
    q = sample_prime(n_bitlength // 2)
    n = p * q

    return EncryptionKey(n), DecryptionKey(p, q)
```


## Encryption

```python
def enc(ek: EncryptionKey, x, r):
    gx = pow(ek.g, x, ek.nn)
    rn = pow(r, ek.n, ek.nn)
    c = (gx * rn) % ek.nn
    return c
```


```python
class EncryptionKey:
    def __init__(self, n):
        self.n = n
        self.nn = n * n
        self.g = 1 + n
```

## Decryption


```python
class DecryptionKey:
    def __init__(self, p, q):
        n = p * q

        self.n = n
        self.nn = n * n
        self.g = 1 + n

        order_of_n = (p - 1) * (q - 1)
        self.d1 = order_of_n
        self.d2 = inv(order_of_n, n)
        self.e = inv(n, order_of_n)
```

```python
def dec(dk: DecryptionKey, c):
    gxd = pow(c, dk.d1, dk.nn)
    xd = dlog(gxd, dk.n)
    x = (xd * dk.d2) % dk.n
    return x
```

```python
def dlog(gy, n):
    y = (gy - 1) // n
    return y
```

## Extraction

```python
def extract(dk: DecryptionKey, c):
    x = dec(dk, c)
    gx = pow(dk.g, x, dk.nn)
    gx_inv = inv(gx, dk.nn)
    rn = (c * gx_inv) % dk.nn
    r = pow(rn, dk.e, dk.n)
    return r
```

## Dump


```python
p = 11
q = 13
n = p * q
ek = EncryptionKey(n)

assert enc(ek, 5, 2) ==
# from slides
```

```python
def sample_randomness(ek):
    while True:
        r = random.randrange(ek.n)
        if gcd(r, ek.n) == 1:
            return r
```


This means that `enc` is actually parameterized by `n` as seen below. In fact, it is also parameterized by a `g` and `nn` value that are however easily derived from `n`.


Concretely, `n = p * q` is a [RSA modulus](https://en.wikipedia.org/wiki/RSA_(cryptosystem)) consisting of two primes `p` and `q` that for security reasons are [recommended](https://www.keylength.com/en/compare/) to each be at least 1000 bits long.







Jointly we call `(n, g, nn)` the *public encryption key* and `(p, q, n)` the *private decryption key*.




 , and `g` is a fixed generator, typically picked as `g = 1 + n`.

 while `r` is limited to those numbers in `Zn` that have a [multiplication inverse](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse), which we denote as `Zn*`.



  This is done relative to a specific but here implicit public *encryption key* which 

To fully define this mapping we note that `x` can be any value from `Zn = {0, 1, ..., n-1}` while `r` is limited to those numbers in `Zn` that have a multiplication inverse, i.e. `Zn*`; together this implies that `c` is a value in `Zn^2*`, ie. amoung the values in `{0, 1, ..., n^2 - 1}` that have multiplication inverses. Finally, `n = p * q` is a typical RSA modulus consisting of two primes `p` and `q`, and `g` is a fixed generator, typically picked as `g = 1 + n`.

```python
ek, dk = keygen()
```

```python
def dec(dk, c):
    gx = pow(c, dk.d, dk.nn)
    x = (gx - 1) // dk.n
    return x
```




, as well as its inverse `dec` that recovers both `x` and `r` from a ciphertext. Here, `enc` is implicitly using a public encryption key `ek` that everyone can know while `dec` is using a private decryption key `dk` that only those allowed to decrypt should know.

The `x` is typically the message we want to encrypt while `r` is a randomness picked uniformly at random for each new encryption; as such, we get a 

# Homomorphic Operations

Perhaps the most attractive property of the Paillier encryption scheme is that it allows us to compute on data while it remains encrypted: given encryptions `c1` and `c2` of respectively `x1` and `x2`, it is possible to compute an encryption `c` of `x1 + x2` *without knowing the decryption key* or in other ways learn anything about the encrypted values.

This opens up for very powerful applications, including electronic voting, secure auctions, private-preserving machine learning, and even general purpose secure computation. We will go through some of these later in the series.

## Addition

Let us first see how one can do the above and compute the addition of two encrypted values, say `c1 = enc(x1, r1)` and `c2 = enc(x2, r2)`.

To do this we multiply the two ciphertexts, letting `c = c1 * c2` (modulus `nn`). To see that this indeed gives what we want, we can expand our formula for encryption to get the following:

```
c1 * c2
== enc(x1, r1) * enc(x2, r2)
== (g^x1 * r1^n) * (g^x2 * r2^n)
== g^(x1 + x2) * (r1 * r2)^n
== enc(x1 + x2, r1 * r2)
```

, leaving out the `mod n^2` to simplify notation.

In other words, if we multiply ciphertext values `c1` and `c2` then we get exactly the same result as if we had encrypted `x1 + x2` using randomness `r1 * r2`: `c = c1 * c2 == enc(x1 + x2, r1 * r2)`!

```python
def add_cipher(ek, c1, c2):
    c = (c1 * c2) % ek.nn
    return c
```

Note that `add_cipher` can also be used to compute the addition of a ciphertext and a plaintext value, by first encrypting the latter. In particular case we might as well use `1` as randomness when encrypting the plaintext value as shown in `add_plain`, and leading to a `c == enc(x1 + x2, r1)`.

```python
def add_plain(ek, c1, x2):
    c2 = enc(ek, x2, 1)
    c = add_cipher(ek, c1, c2)
    return c
```

We now know how to add encrypted values together without decrypting anything! Note however, that the resulting ciphertexts have a slightly different form than freshly generated ones, with a randomness that is no longer a uniformly random value but rather a composite such as `r1 * r2`. This does not affect correctness nor the ability to decrypt, but in some applications it may leak extra information to an adversary and hence have consequences for security. We return to this issue below after having introduced more operations.

## Subtraction

Subtraction follows easily from addition through the use of two negation functions, `neg_cipher` and `neg_plain`, given in full in the associated notebook.

```python
def neg_cipher(ek, c):
    return inv(c, ek.nn)

def neg_plain(ek, x):
    return ek.n - x
```

The former computes the [multiplicative inverse](https://en.wikipedia.org/wiki/Multiplicative_inverse) and the latter the [additive inverse](https://en.wikipedia.org/wiki/Additive_inverse), which simply means that `c * neg_cipher(c) == 1` modulus `nn` and `x + neg_plain(x) == 0` modulus `n`. This basically allows us to turn `x1 - x2` into `x1 + (-x2)` and use the addition operations from earlier.

```python
def sub_cipher(ek, c1, c2):
    minus_c2 = neg_cipher(ek, c2)
    c = add_cipher(ek, c1, minus_c2)
    return c

def sub_plain(ek, c1, x2):
    minus_x2 = neg_plain(ek, x2)
    c = add_plain(ek, c1, minus_x2)
    return c
```

Note that the resulting ciphertexts again have a slightly different form than freshly encrypted ones, namely `r1 * r2^-1`.

## Multiplication

The final operation supported by Paillier encryption is multiplication between an encrypted value and a plaintext value; the fact that it is not known how to multiply two encrypted values makes it a *partially homomorphic* scheme, and is the main thing that sets it apart from more recent *fully homomorphic* schemes where it is indeed possible.


Just like multiplication by a value `k` can be seen as adding `k`

Likewise, given `c = enc(x, r)` and a `k` we compute `c^k = (g^x * r^n) ^ k == g^(x * k) * (r^k)^n == enc(x * k, r ^ k)`, again leaving out `mod n^2`.

```python
def mul_plain(ek, c1, x2):
    c = pow(c1, x2, ek.nn)
    return c
```

## Linear functions

Combining the operations above we can derive a function `linear` for evaluating e.g. the [dot product](https://en.wikipedia.org/wiki/Dot_product#Algebraic_definition) between a vector of ciphertexts and a vector of plaintext.

```python
def linear(ek, cs, xs):
    terms = [
        mul_plain(ek, c, x)
        for c, x in zip(cs, xs)
    ]
    adder = lambda c1, c2: add_cipher(ek, c1, c2)
    return reduce(adder, terms)
```

As an example, this allows us to express the following

```python
cs = [enc(1), enc(2), enc(3)]
xs = [10, -20, 30]
c = linear(cs, xs)

assert dec(c) == (1 * 10) - (2 * 20) + (3 * 30)
```

and captures everything we can do with encrypted values in the Paillier scheme, with e.g. `add_cipher(c1, c2)` being essentially the same as `linear([c1, c2], [1, 1])`, `sub_cipher(c1, c2)` the same as `linear([c1, c2], [1, -1])`, and `mul_plain(c, x)` the same as `linear([c], [x])`.

## Re-randomization

As noted throughout, the ciphertexts resulting from homomorphic operations have randomness components with a structure that differs from the one found in freshly generated ciphertexts. In some cases, taking this into account may simply make analyzing the security of the system harder; in others, it may even leak something to an adversary about the encrypted values.

A freshly generated ciphertext will have a randomness component that was independently sampled .....

TODO: good examples of the above?

Fortunately, we can easily define a *re-randomize* operation that makes any ciphertext look exactly like a freshly generated one, effectively erasing everything about how it was created. To do this we have to make sure the randomness component looks uniformly random given anything that the adversary may know. To do this we simply add a fresh encryption of zero `enc(0, s)`, which for `enc(x, r)` will give us an encryption `enc(x, r*s)`; however, if `s` is independent and uniformly random then so is `r*s`. We are essentially 

```python
def rerandomize(ek, c):
    c_zero = enc(0, random.randint(ek.n))
    d = add_cipher(c, c_zero)
    return d
```

could be done lazily

# Next Steps

In the next post we will look at concrete applications of Paillier encryption, in particular when it comes to privacy-preserving machine learning.
