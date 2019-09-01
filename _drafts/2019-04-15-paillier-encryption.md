---
layout:     post
title:      "An Illustrated Primer on Paillier"
subtitle:   "Overview of a Homomorphic Encryption Scheme"
date:       2019-04-15 12:00:00
author:     "Morten Dahl"
header-img: "assets/paillier/autostereogram-space-shuttle.jpeg"
summary:    "The Paillier homomorphic encryption scheme is not only interesting for allowing computation on encrypted values, it also provides an excellent illustration of modern security assumptions and a beautiful application of abstract algebra."
---

<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>

<em><strong>TL;DR:</strong> the Paillier encryption scheme is not only interesting for allowing computation on encrypted values, it also provides an excellent illustration of modern security assumptions and a beautiful application of abstract algebra.</em>

In this blog post, we walk through and explain [Paillier encryption](https://en.wikipedia.org/wiki/Paillier_cryptosystem), a so called *partially homomorphic scheme* that was [first described](https://link.springer.com/chapter/10.1007%2F3-540-48910-X_16) by [Pascal Paillier](https://twitter.com/pascal_paillier) exactly 20 years ago. [More advanced schemes](https://en.wikipedia.org/wiki/Homomorphic_encryption) have since been developed, supporting additional operations to be performed on encrypted data, yet Paillier encryption remains relevant both from an application point of view and for understanding modern cryptography.

We will go through the basics, a few applications, why the scheme is believed to be secure, and some of the tricks that can be used to implement it more efficiently. As always, the full source code is available for experimentation, but inspired by the excellent [A Homomorphic Encryption Illustrated Primer](https://blog.n1analytics.com/homomorphic-encryption-illustrated-primer/) by [Stephen Hardy](https://twitter.com/proximation) and [Differential Privacy: An illustrated Primer](https://github.com/frankmcsherry/blog/blob/master/posts/2016-02-06.md) by [Frank McSherry](https://twitter.com/frankmcsherry) we also try to give a more visual presentation of material that is [typically](https://www.cs.umd.edu/~jkatz/imc.html) offered mostly in the form of equations.

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
def keygen(n_bitlength=2048):
    p = sample_prime(n_bitlength // 2)
    q = sample_prime(n_bitlength // 2)
    n = p * q

    return EncryptionKey(n), DecryptionKey(p, q)
```


```python
class DecryptionKey:
    def __init__(self, p, q):
        n = p * q

        self.n = p * q
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
    gx = pow(dk.g, x, ek.nn)
    gx_inv = inv(gx, ek.nn)
    rn = (c * gx_inv) % ek.nn
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


# Homomorphic Properties

Perhaps the most attractive property of the Paillier encryption scheme, is that anyone can compute on data while it remains encrypted. In other words, given only ciphertexts `c1` and `c2`, anyone can e.g. compute a new ciphertext `c` which encrypts the sum of the plainte     , i.e. without  if they cannot decrypt  anytinhgthe ability to decrypt anything. 



Concretely, anyone can combine encryptions `c1` and `c2` to get an encryption of their sum, and anyone can combine an encryption `c` with a plaintext `x` to get an encryption of their product -- without being able to learn anything about the encrypted values in the process.

As we will see in more detail below this opens up for some powerful applications, including basic electronic voting, private machine learning, and even general purpose secure computation. 

 two primitive operations 

## Addition

To see how this works let's start with addition: 

```python
def add_cipher(ek, c1, c2):
    c = (c1 * c2) % ek.nn
    return c
```

```python
def add_plain(ek, c1, x2):
    c2 = pow(ek.g, x2, ek.nn)
    c = (c1 * c2) % ek.nn
    return c
```

given `c1 = enc(x1, r1)` and `c2 = enc(x2, r2)` we compute `c1 * c2 == (g^x1 * r1^n) * (g^x2 * r2^n) == g^(x1 + x2) * (r1 * r2)^n == enc(x1 + x2, r1 * r2)`, leaving out the `mod n^2` to simplify notation.

## Subtraction

```python
def neg(ek, c):
    return inv(c, ek.nn)
```

```python
def sub_cipher(ek, c1, c2):
    c = add_cipher(ek, c1, neg(ek, c2))
    return c
```

```python
def sub_plain(ek, c1, x2):
    c = add_plain(ek, c1, ek.n - x2)
    return c
```

## Multiplication

Likewise, given `c = enc(x, r)` and a `k` we compute `c^k = (g^x * r^n) ^ k == g^(x * k) * (r^k)^n == enc(x * k, r ^ k)`, again leaving out `mod n^2`.

```python
def mul_plain(ek, c1, x2):
    c = pow(c1, x2, ek.nn)
    return c
```

## Linear functions

Combining the two primitive operations we can evaluate [linear functions](https://en.wikipedia.org/wiki/Linear_function) on two lists of ciphertexts and plaintexts.



```python
def linear(ek, cs, xs):
    terms = [
        mul_plain(ek, c, x)
        for c, x in zip(cs, xs)
    ]
    adder = lambda c1, c2: add_cipher(ek, c1, c2)
    return reduce(adder, terms)
```

```python
cs = [enc(1), enc(2), enc(3)]
xs = [10, 20, 30]
c = linear(cs, xs)

assert dec(c) == (1 * 10) + (2 * 20) + (3 * 30)
```

## Re-randomization


Note that the results do not exactly match the original form of encryptions: in the first case the resulting randomness is `r1 * r2` and in the latter it is `r^k`, whereas for fresh encryptions these values are uniformaly random. In some cases this may leak something about otherwise private values (see e.g. the voting application below TODO) and as a result we sometimes need a *re-randomize* operation that erases everything about how a ciphertext was created by making it look exactly as a fresh one. We do this by simply multiplying by a fresh encryption of zero: `enc(x, r) * enc(0, s) == enc(x, r*s) == enc(x, t)` for a uniformly random `t` if `s` is independent and uniformly random.

```python
def rerandomize(ek, c, s):
    sn = pow(s, ek.n, ek.nn)
    c_fresh = (c * sn) % ek.nn
    return c_fresh
```

could be done lazily

# Applications

<!-- 
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
``` -->

## Federated Learning

Secure Aggregation for Federated Learning

<em>(coming...)</em>

<!-- 

### Packing

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

-->

## Privacy-preserving predictions

Prediction using a linear model

<em>(coming...)</em>

<!-- 
Linear model

see more examples in python-paillier and Andrew's blog post -->



## General secure computation

How do we get multiplication of ciphertexts as well?

<em>(coming...)</em>

<!--
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
-->
<!-- 
## Extensions

### Threshold decryption

- assume split key has already been generated; link to Gert's paper

### Proofs

- correct decryption
- correct HE operations
- knowledge of plaintext

[`zk-paillier`](https://github.com/KZen-networks/zk-paillier) -->


# Algebraic Interpretation

<em>(coming...)</em>

<!-- 
`(1 + n)**x == 1 + nx mod nn` and `(1 + n)**x == 1 + nx mod pp`

`(Zn2*, *) ~ (Zn, +) x (Zn*, *)`

HE
- `c1 * c2 == (x1, r1) * (x2, r2) == (x1+x2, r1*r2)`

- `c ^ k == (x, r) ^ k == (x * k, r^k)`

- `inv(c) == inv((x, r)) == (-x, r^-1)`

- `c * s^n == (x, r) * (0, s) == (x, r*s) == (x, t)`

dec
- `c^phi == (x*phi, r^phi) == (x*phi, 1) == g^(x*phi)`
-->

# Security

<em>(coming...)</em>

<!-- <p style="align: center;">
<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-space-shuttle.jpeg" style="width: 85%;"/></a>
<br/><em>From World of Hidden 3D</em>
</p> -->

<!--

<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-mars-rover.jpeg" style="width: 85%; border: 2px solid black;"/></a>

If you are told [the secret](http://www.hidden-3d.com/how_to_view_stereogram.php) then you can see the pattern

like [autostereograms](https://en.wikipedia.org/wiki/Autostereogram) there's a hidden pattern; unlike them, discovering the underlying pattern via starring will simply take too long

In order to better understand the scheme, including its security, it's instructive to start with a fool-proof scheme and see how it compares.

<img src="/assets/paillier/nn.png" style="width: 45%"/>
<img src="/assets/paillier/nninverse.png" style="width: 45%"/>
<img src="/assets/paillier/nnstar.png" style="width: 45%"/>

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
-->

# Efficient Implementation

A [profiling tool](http://carol-nichols.com/2015/12/09/rust-profiling-on-osx-cpu-time/) will quickly show that the modular exponentiations in encryption and decryption are what takes up most of our compute, to the point where we can almost ignore everything else. This makes it a good place to focus for efficiency, and to do so we here switch from Python to Rust as it gives a more realistic image of performance and has better [tool](https://doc.rust-lang.org/cargo/commands/cargo-bench.html) [support](https://crates.io/crates/rayon).

Here we focus on the optimizations closest to the cryptography, and skip those that may be possible at the bignum (some from DJN03?) and applications level.

The code for this section is available [here](https://github.com/mortendahl/privateml/tree/master/paillier/benchmarks). We focus on CPU efficiency using [GMP](https://crates.io/crates/rust-gmp) but note that other work such as [cuda-fixnum](https://github.com/n1analytics/cuda-fixnum) target GPUs.

## Decryption

As a first step, we can get a sense of how the exponentiation in decryption scales with the bit-length of its inputs using the following micro-bench.

```rust
mod decrypt_pow {
    fn bench_core(b: &mut Bencher, x: &BigInt) {
        let xx = x * x;
        b.iter(|| {
            let _ = pow(&xx, x, &xx);
        });
    }
```

From the graph we see that doubling the bit-length increases the running time by more than a factor of four: as a [lower bound](https://en.wikipedia.org/wiki/Big_O_notation) we have `T(|x|) ∈ Ω(|x|^2)` so `T(2|x|) ≥ 2^2 * T(|x|) = 4 * T(|x|)`. Conversely, by cutting the size of our inputs in half we improve the running time by *at least* a factor of four.

<p style="align: center;"><img src="/assets/paillier/dec-pow.png" style="width: 85%;"/></p>

In fact, looking closer at the concrete numbers we see that going from a bit-length of 2048 down to 1024 boosts performance by a factor of `6.5`.

```text
test micro::decrypt_pow::bench_256  ... bench:      34,845 ns/iter (+/- 3,467)
test micro::decrypt_pow::bench_512  ... bench:     235,786 ns/iter (+/- 78,397)
test micro::decrypt_pow::bench_768  ... bench:     663,346 ns/iter (+/- 90,810)
test micro::decrypt_pow::bench_1024 ... bench:   1,601,292 ns/iter (+/- 373,027)
test micro::decrypt_pow::bench_1280 ... bench:   2,745,027 ns/iter (+/- 224,231)
test micro::decrypt_pow::bench_1536 ... bench:   4,722,906 ns/iter (+/- 527,313)
test micro::decrypt_pow::bench_1792 ... bench:   7,375,770 ns/iter (+/- 733,716)
test micro::decrypt_pow::bench_2048 ... bench:  10,461,714 ns/iter (+/- 701,832)
```

As shown in the original paper, the [Chinese Remainder Theorem](https://en.wikipedia.org/wiki/Chinese_remainder_theorem) (or CRT) can be used for this: instead of computing modulus `n^2` we use the fact that `n == p * q` and compute modulus `p^2` and `q^2` (since primes `p` and `q` are clearly co-prime). Although this means that we now have to do two computations, this should still improve performance by a factor of roughly `6.5/2 == 3.25`.

However, a nice property of the CRT is that the computations can be done almost entirely in parallel, meaning that if we have two cores available for our computations then we can get the full `6.5` factor improvement. This is almost the case as seen below, with `plain::bench_decrypt` being slower than `crt::bench_decrypt` and `crt::parallel::bench_decrypt` by factors `3.2` and `6.3`, respectively.

```text
test plain::bench_decrypt           ... bench:  11,218,307 ns/iter (+/- 1,743,463)
test crt::bench_decrypt             ... bench:   3,474,298 ns/iter (+/- 1,097,774)
test crt::parallel::bench_decrypt   ... bench:   1,778,522 ns/iter (+/- 477,601)
```

The following Rust [code](https://github.com/mortendahl/privateml/tree/master/paillier/benchmarks) shows the optimized decryption method, where [`rayon::join`](https://docs.rs/rayon/) is used to run the two computations in parallel.

```rust
fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
    let (mp, mq) = join(
        || {
            let cp = c % &dk.pp;
            decrypt_component(&cp, &dk.p, &dk.pp, &dk.p_order, &dk.hp)
        },
        || {
            let cq = c % &dk.qq;
            decrypt_component(&cq, &dk.q, &dk.qq, &dk.q_order, &dk.hq)
        },
    );
    crt(&mp, &mq, &dk.p, &dk.q, &dk.p_inv)
}
```

One caveat worth mentioning is that the use of the CRT opens up for some side-channel attacks as illustrated in [XXX](TODO).

## Extraction

```python
def extract(dk: DecryptionKey, c):
    rn = c % dk.n
    r = pow(rn, dk.e, dk.n)
    return r
```

```rust
fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
    let (rp, rq) = join(
        || {
            let rnp = c % &dk.p;
            pow(&rnp, &dk.ep, &dk.p)
        },
        || {
            let rnq = c % &dk.q;
            pow(&rnq, &dk.eq, &dk.q)
        },
    );
    crt(&rp, &rq, &dk.p, &dk.q, &dk.p_inv)
}
```

## Encryption

```python
def enc(ek: EncryptionKey, x, r):
    gx = 1 + x * ek.n
    rn = pow(r, ek.n, ek.nn)
    c = (gx * rn) % ek.nn
    return c
```

Given the above it would be natural to ask whether the CRT can also be used to speed up encryptions. However, the problem here is that doing so requires knowledge of `p` and `q`, which for security has to be kept private. The result is that this optimization is *not* applicable in general, but only by those who know the decryption key; in some of the applications seen earlier this is still useful.

```
test plain::bench_encrypt         ... bench:  10,862,975 ns/iter (+/- 1,150,426)
test crt::bench_encrypt           ... bench:   6,474,433 ns/iter (+/- 443,019)
test crt::parallel::bench_encrypt ... bench:   3,315,413 ns/iter (+/- 667,753)
```

From the concrete numbers we see that, when applicable, it can give a performance boost of roughly a factor of `3`. This smaller factor compared to decryption has ....

```rust
fn encrypt(x: &BigInt, r: &BigInt, ek: &EncryptionKey) -> BigInt {
    let (cp, cq) = join(
        || {
            let xp = x % &ek.pp;
            let rp = r % &ek.pp;
            encrypt_component(&xp, &rp, ek, &ek.pp)
        },
        || {
            let xq = x % &ek.qq;
            let rq = r % &ek.qq;
            encrypt_component(&xq, &rq, ek, &ek.qq)
        },
    );
    crt(&cp, &cq, &ek.pp, &ek.qq, &ek.pp_inv)
}
```

However, we can do a few other things.

The second is to note that the remaining exponentiation, `pow(r, ek.n, ek.nn)`, may be precomputed in an *offline* phase. This reduces the *online* encryption to two multiplications. Moreover, https://eprint.iacr.org/2015/864

```
DNJ03 proposed a generalization of the scheme in which the
expansion factor is reduced and implementations of both the generalized and
original scheme are optimized without losing the homomorphic property [3].
Their system achieves the speed of 0.262 milliseconds/bit for the original Paillier scheme, equivalent to 3,816.79 bits/sec. This performance was reached by a
clever choice of basis and using standard pre-computation techniques for fixed
basis exponentiation. However, the encryption performance can be increased
even further by using the techniques described in this paper. We reach a speed
of 9,197,824 to 48,810,496 bits/sec depending on the security parameter.
```

### Pre-computing randomness

```rust
fn precompute_randomness(ek: &EncryptionKey, r: &BigInt) -> BigInt {
    modpow(r, &ek.n, &ek.nn)
}

fn encrypt_with_precomputed(dk: &DecryptionKey, m: &BigInt, rn: &BigInt) -> BigInt {
    let gm = (1 + m * &dk.n) % &dk.nn;
    (gm * rn) % &dk.nn
}
```

### Smaller randomness

# Further Reading


- [`python-paillier`](https://github.com/n1analytics/python-paillier) [`rust-paillier`](https://github.com/mortendahl/rust-paillier) julia
- snips benchmark post
- [Public-Key Cryptosystems Based on Composite Degree Residuosity Classes](https://link.springer.com/chapter/10.1007%2F3-540-48910-X_16)
- DJ'01
- DJN'03
- link to paper and [Introduction to Modern Cryptography](http://www.cs.umd.edu/~jkatz/imc.html).

