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

20 year anniversary (15 april)

[`python-paillier`](https://github.com/n1analytics/python-paillier)
[`rust-paillier`](https://github.com/mortendahl/rust-paillier)
[Public-Key Cryptosystems Based on Composite Degree Residuosity Classes](https://link.springer.com/chapter/10.1007%2F3-540-48910-X_16)

- The need for probabilistic encryption
- full scheme with mapping `g^x * r^n` for `Zn x Zn*`
- security from hidden structure
- there are things we want from the mapping, besides being efficient to compute: 
    - efficient to invert knowning decryption key
    - provide security


- Idea of mapping into algebraic structure to hide message
- should be easy in one direction, hard in the opposite without secret key
- in a HE scheme the mapping should preserve certain operations
- PHE vs SHE vs FHE; benchmarks?
- link to paper and [Introduction to Modern Cryptography](http://www.cs.umd.edu/~jkatz/imc.html).

like [autostereograms](https://en.wikipedia.org/wiki/Autostereogram) there's a hidden pattern; unlike them, discovering the underlying pattern via starring will simply take too long

<em>Parts of this blog post are inspired from work done at [Snips](https://snips.ai), some of it originally published in a two [blog](https://medium.com/snips-ai/prime-number-generation-2a02f28508ff) [posts](https://medium.com/snips-ai/benchmarking-paillier-encryption-15631a0b5ad8).</em>


# Basics

The Paillier encryption scheme is defined by a function `enc` that maps a message `x` and randomness `r` to a ciphertext `c = enc(x, r)`. The effect of having both `x` and `r` is that we end up different ciphertexts even if we encrypt the same `x` several times: if `r1`, `r2`, and `r3` are different then so are `c1 = enc(x, r1)`, `c2 = enc(x, r2)`, and `c3 = enc(x, r3)`.

<img src="/assets/paillier/probabilistic.png" style="width: 50%;"/>

This means that even if an adversary who has obtained a ciphertext `c` knows that there are only a few options for `x`, say only either `0` or `1`, for them to figure out which one it is by simply comparing `c` to `c0 = enc(0, r0)` and `c1 = enc(1, r1)`, they would also have to figure out what `r0` and `r1` should be. So, if there are sufficiently many options for these then this guessing game becomes impractical. In other words, we have removed the adversary's ability to check whether or not a guess was correctly; of course, there may be other ways for them to check a guess or learn something about `x` from `c` in general, and we shall return to security of the scheme in much more detail later.

When `r` is chosen independently at random, Paillier encryption becomes what is known as a [probabilistic encryption scheme](https://en.wikipedia.org/wiki/Probabilistic_encryption), an often desirable property of encryption schemes per the discussion above.

  having enough  One motivation of including a randomness in the ciphertext is that it makes it impractical to guess `x` simply by trying to encrypt different values and check whether the ciphertexts match, as one would also have to guess `r` which is typically picked from a very large set: it is allowed to be any element from `Zn = {0, 1, ..., n-1}`, where `n` is a number with between 2000 and 4000 bits (we a lying a tiny bit here, but will return to that soon).


 Formally this makes Paillier a [probabilistic encryption scheme](https://en.wikipedia.org/wiki/Probabilistic_encryption), which is often a desirable property of encryption schemes.

```python
def enc(ek, x, r):
    gx = pow(ek.g, x, ek.nn)
    rn = pow(r, ek.n, ek.nn)
    c = (gx * rn) % ek.nn
    return c
```

```python
p = 5
q = 7
n = p * q
ek = EncryptionKey(n)

assert enc(ek, 5, 2) ==
# from slides
```

```python
def sample_randomenss(ek):
    return random.randrange(dk.n)
```


This means that `enc` is actually parameterized by `n` as seen below. In fact, it is also parameterized by a `g` and `nn` value that are however easily derived from `n`.


Concretely, `n = p * q` is a [RSA modulus](https://en.wikipedia.org/wiki/RSA_(cryptosystem)) consisting of two primes `p` and `q` that for security reasons are [recommended](https://www.keylength.com/en/compare/) to each be at least 1000 bits long.


```python
class EncryptionKey:
    def __init__(self, n):
        self.n = n
        self.nn = n * n
        self.g = n + 1

class DecryptionKey:
    def __init__(self, p, q):
        self.n = p * q
        self.nn = n * n
        self.g = n + 1

        self.d = (p - 1) * (q - 1)
        self.h = pow(g, -self.d, self.nn inv(g)
```

```python
def keygen(n_bitlength=2048):
    p = sample_prime(n_bitlength // 2)
    q = sample_prime(n_bitlength // 2)
    n = p * q

    return EncryptionKey(n), DecryptionKey(p, q)
```



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


## Homomorphic properties

Besides being probabilistic, a highly attractive property of the Paillier is that it allows for computing on data that remains encrypted throughout the entire process. through homomorphic properties. In particuar, we can combine encryptions of `x1` and `x2` to get an encryption of `x1 + x2`, and an encryption of `x` with a constant `k` to get an encryption of `x * k`, in both cases without decrypting anything (and hence without learning anything about `x1` and `x2`.

### Addition

To see how this works let's start with addition: given `c1 = enc(x1, r1)` and `c2 = enc(x2, r2)` we compute `c1 * c2 == (g^x1 * r1^n) * (g^x2 * r2^n) == g^(x1 + x2) * (r1 * r2)^n == enc(x1 + x2, r1 * r2)`, leaving out the `mod n^2` to simplify notation.

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

### Multiplication

Likewise, given `c = enc(x, r)` and a `k` we compute `c^k = (g^x * r^n) ^ k == g^(x * k) * (r^k)^n == enc(x * k, r ^ k)`, again leaving out `mod n^2`.

```python
def mul_plain(ek, c1, x2):
    c = pow(c1, x2, ek.nn)
    return c
```

Note that the results do not exactly match the original form of encryptions: in the first case the resulting randomness is `r1 * r2` and in the latter it is `r^k`, whereas for fresh encryptions these values are uniformaly random. In some cases this may leak something about otherwise private values (see e.g. the voting application below TODO) and as a result we sometimes need a *re-randomize* operation that erases everything about how a ciphertext was created by making it look exactly as a fresh one. We do this by simply multiplying by a fresh encryption of zero: `enc(x, r) * enc(0, s) == enc(x, r*s) == enc(x, t)` for a uniformly random `t` if `s` is independent and uniformly random.

As we will see in more detail below this opens up for some powerful applications, including electronic voting, private machine learning, and general purpose secure computation. But first it's interesting to go into more details about how decryption works and why the scheme is secure.


### Re-randomization

```python
def rerandomize(ek, c):
    s = random.randrange(ek.n)
    sn = pow(s, ek.n, ek.nn)
    c_fresh = (c * sn) % nn
    return c_fresh
```

could be done lazily

## Decryption

```python
c^len(X) == (g^x * r^n)^len(X) == g^(x * len(X)) * r^(n * len(X)) == g^(x * len(X)) * r^(len(C)) == g^(x * len(X))
pow(c, len(X), nn) == pow(pow(g, x, nn) * pow(r, n, nn), len(X), nn == g^(x * len(X)) * r^(n * len(X)) == g^(x * len(X)) * r^(len(C)) == g^(x * len(X))
```

## Opening


# Security

<p style="align: center;">
<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-space-shuttle.jpeg" style="width: 85%;"/></a>
<br/><em>From World of Hidden 3D</em>
</p>

<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-mars-rover.jpeg" style="width: 85%;"/></a>

If you are told [the secret](TODO-how-to-see-autosteoreograms) then you can see the pattern

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


# Algebraic Insights

`(1 + n)**x == 1 + nx mod nn` and `(1 + n)**x == 1 + nx mod pp`

`(Zn2*, *) ~ (Zn, +) x (Zn*, *)`

HE
- `c1 * c2 == (x1, r1) * (x2, r2) == (x1+x2, r1*r2)`

- `c ^ k == (x, r) ^ k == (x * k, r^k)`

- `inv(c) == inv((x, r)) == (-x, r^-1)`

- `c * s^n == (x, r) * (0, s) == (x, r*s) == (x, t)`

dec
- `c^phi == (x*phi, r^phi) == (x*phi, 1) == g^(x*phi)`






# Efficiency

The Chinese Remainder Theorem

- base on good library for arbitrary precision integer ops (framp, GMP)
- side-channel challenges
- refs:
  - n1analytics (java-python)
  - rust-paillier
  - benchmark blog post

https://github.com/n1analytics/cuda-fixnum for Paillier on GPU

```
test micro::bench_pow128          ... bench:      40,630 ns/iter (+/- 32,583)
test micro::bench_pow256          ... bench:     282,069 ns/iter (+/- 325,362)
test micro::bench_pow384          ... bench:     751,478 ns/iter (+/- 691,504)
test micro::bench_pow512          ... bench:   1,673,984 ns/iter (+/- 3,096,919)
test micro::bench_pow640          ... bench:   3,097,269 ns/iter (+/- 1,463,063)
test micro::bench_pow768          ... bench:   5,101,171 ns/iter (+/- 722,831)
test micro::bench_pow896          ... bench:   7,768,877 ns/iter (+/- 523,873)
test micro::bench_pow1024         ... bench:  11,239,404 ns/iter (+/- 6,250,901)
```

<p style="align: center;"><img src="/assets/paillier/pow-complexity.png" style="width: 85%;"/></p>

with `Oh(n) = n^2.3` we have `Oh(2n) = 2^2.3 * n^2.3 = 2^2.3 * Oh(n)`, or roughly `Oh(2n) = 5 * Oh(n)`, meaning that doubling the length of numbers makes the computation roughly five times slower, or conversely, working on numbers half the length cuts running time down by a factor of roughly five. and on top of that, with CRT we can do things in parallel, cutting the running time by an additional factor of two if enough cores are available.

above we used algebra and isomorphisms to understand the encryption scheme and argue why decryption works. here we'll use it to speed up computations. the crt defines an isomorphism.

before we had `Zn2* ~ Zn x Zn*` and for the same reasons we also have `Zp2* ~ Zp x Zp*` and `Zq2* ~ Zq x Zq*`. but from the general crt we also have `Zn2* ~ Zp2* x Zq2*`. and we can combine these so `Zn2* ~ Zp2* x Zq2* ~ (Zp x Zp*) x (Zq x Zq*)`.

[benchmarks](https://github.com/mortendahl/privateml/tree/master/paillier/benchmarks)

`cargo bench`




```rust
let (x1, x2) = (x % m1, x % m2);
```

```rust
fn crt(x1: &BigInt, x2: &BigInt, m1: &BigInt, m2: &BigInt, m1_inv: &BigInt) -> BigInt {
    let mut diff = x2 - x1;
    if diff.sign() == Sign::Negative {
        diff = (diff % m2) + m2;
    }
    let u = (diff * m1_inv) % m2;
    x1 + (u * m1)
}
```


### Decryption


<strong>raising to `n_order_n_order_inv` adds a lot, don't do this; use h instead</strong>

```
test plain::bench_decrypt         ... bench:  21,451,726 ns/iter (+/- 1,606,939)
test crt::bench_decrypt           ... bench:   3,269,054 ns/iter (+/- 511,768)
test crt::parallel::bench_decrypt ... bench:   1,723,580 ns/iter (+/- 694,941)
```

```rust
fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
    let gx = pow(c, &dk.n_order_n_order_inv, &dk.nn);
    l(&gx, &dk.n)
}
```

`decrypt(c, n, nn, n_order)`

```rust
fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
    let (cp, cq) = (c % &dk.pp, c % &dk.qq);
    let mp = decrypt_component(&cp, &dk.p, &dk.pp, &dk.p_order, &dk.hp);
    let mq = decrypt_component(&cq, &dk.q, &dk.qq, &dk.q_order, &dk.hq);
    crt(&mp, &mq, &dk.p, &dk.q, &dk.p_inv)
}

fn decrypt_component(
    c: &BigInt,
    m: &BigInt,
    mm: &BigInt,
    m_order: &BigInt,
    hm: &BigInt,
) -> BigInt {
    let dm = pow(c, m_order, mm);
    let lm = l(&dm, m);
    (lm * hm) % m
}
```

```rust
fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
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

### Opening

```rust
fn crt_open(
    c: &BigInt,
    p: &BigInt, pp: &BigInt, p_order: &BigInt,
    q: &BigInt, qq: &BigInt, q_order: &BigInt,
) -> BigInt
{
    let (cp, cq) = crt_decompose(c, pp, qq);
    let (mp, mq) = join(
        || { 
            decrypt(&cp, p, pp, p_order) 
            TODO
        },
        || { 
            decrypt(&cq, q, qq, q_order) 
            TODO
        },
    );
    crt(mp, mq, p, q)
}


fn extract_nroot(
    z: &BigInt,
    p: &BigInt, dp: &BigInt,
    q: &BigInt, dq: &BigInt,
) -> BigInt
{
    let (zp, zq) = crt_decompose(z, p, q);
    let (rp, rq) = join(
        || { modpow(&zp, dp, p) },
        || { modpow(&zq, dq, q) },
    );
    crt(rp, rq, p, q)
}
```


### Encryption


```
test plain::bench_encrypt         ... bench:  10,862,975 ns/iter (+/- 1,150,426)
test crt::bench_encrypt           ... bench:   6,474,433 ns/iter (+/- 443,019)
test crt::parallel::bench_encrypt ... bench:   3,315,413 ns/iter (+/- 667,753)
```

```rust
mod plain {
    fn encrypt(x: &BigInt, r: &BigInt, ek: &EncryptionKey) -> BigInt {
        let rm = pow(r, &ek.n, &ek.nn);
        let gx = (1 + x * &ek.n) % &ek.nn;
        (gx * rm) % &ek.nn
    }
}
```

```rust
mod crt {
    fn encrypt(x: &BigInt, r: &BigInt, ek: &EncryptionKey) -> BigInt {
        let (xp, xq) = (x % &ek.p, x % &ek.q);
        let (rp, rq) = (r % &ek.p, r % &ek.q);
        let cp = encrypt_component(&xp, &rp, &ek, &ek.pp);
        let cq = encrypt_component(&xq, &rq, &ek, &ek.qq);
        crt(&cp, &cq, &ek.pp, &ek.qq, &ek.pp_inv)
    }

    fn encrypt_component(x: &BigInt, r: &BigInt, ek: &EncryptionKey, mm: &BigInt) -> BigInt {
        let rm = pow(r, &ek.n, mm);
        let gx = (1 + x * &ek.n) % mm;
        (gx * rm) % mm
    }
}
```

```rust
mod crt {
    mod parallel {
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
    }
}
```


precomputed randomness

```rust
fn precompute_randomness(ek: &EncryptionKey, r: &BigInt) -> BigInt {
    modpow(r, &ek.n, &ek.nn)
}

fn encrypt_with_precomputed(dk: &DecryptionKey, m: &BigInt, rn: &BigInt) -> BigInt {
    let gm = (1 + m * &dk.n) % &dk.nn;
    (gm * rn) % &dk.nn
}
```



# Applications


Voting comes up in a lot of papers but here we will look at other uses.


## Encodings

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


### Fixedpoint

## Privacy-preserving predictions

Linear model

see more examples in python-paillier and Andrew's blog post


## Federated Learning


## General secure computation

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