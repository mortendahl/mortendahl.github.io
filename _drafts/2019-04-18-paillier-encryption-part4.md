---
layout:     post
title:      "Paillier Encryption, Part 2"
subtitle:   "Efficient Implementation"
date:       2019-06-16 12:00:00
author:     "Morten Dahl"
header-img: "assets/paillier/autostereogram-mars-rover.jpeg"
summary:    "In the first path of the series we gave a complete but simple Python implementation of Paillier encryption, without typical optimizations in order to stay focused. In the fourth part of series we look at these optimizations and justify them through concrete benchmarks using Rust."
---

<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>

<em><strong>TL;DR:</strong> We look at some of the typical optimizations applied when implementing Paillier encryption, and justify they use via concrete benchmarks on a Rust implementation.</em>

The Python implementation presented in [part 1](2019/04/15/paillier-encryption-part1/) can be optimized in at least two ways: rewritten in a more performance oriented language such as [Rust](https://www.rust-lang.org/)

A [profiling tool](http://carol-nichols.com/2015/12/09/rust-profiling-on-osx-cpu-time/) will quickly show that the modular exponentiations in encryption and decryption are what takes up most of our compute, to the point where we can almost ignore everything else. This makes it a good place to focus for efficiency, and to do so we here switch from Python to Rust as it gives a more realistic image of performance and has better [tool](https://doc.rust-lang.org/cargo/commands/cargo-bench.html) [support](https://crates.io/crates/rayon).

Here we focus on the optimizations closest to the cryptography, and skip those that may be possible at the bignum (some from DJN03?) and applications level.

The code for this section is available [here](https://github.com/mortendahl/privateml/tree/master/paillier/benchmarks). We focus on CPU efficiency using [GMP](https://crates.io/crates/rust-gmp) but note that other work such as [cuda-fixnum](https://github.com/n1analytics/cuda-fixnum) target GPUs.

# Plain

Note that we have already applied some optimizations in the Python implementation: we precompute and cache values derived from the minimal keys.

```rust
mod plain {

    fn encrypt(ek: &EncryptionKey, x: &BigInt, r: &BigInt) -> BigInt {
        let gx = pow(&ek.g, x, &ek.nn);
        let rm = pow(r, &ek.n, &ek.nn);
        let c = (gx * rm) % &ek.nn;
        c
    }

    fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let gxd = pow(c, &dk.d1, &dk.nn);
        let xd = l(&gxd, &dk.n);
        let x = (xd * &dk.d2) % &dk.n;
        x
    }

    fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
        let x = decrypt(dk, c);
        let gx = pow(&dk.g, &x, &dk.nn);
        let gx_inv = inv(&gx, &dk.nn);
        let rn = (c * gx_inv) % &dk.nn;
        let r = pow(&rn, &dk.e, &dk.n);
        r
    }

}
```

To establish a baseline ...

```text
test plain::bench_encrypt           ... bench:  22,255,119 ns/iter (+/- 6,827,724)
test plain::bench_decrypt           ... bench:  12,233,922 ns/iter (+/- 3,836,237)
test plain::bench_extract           ... bench:  26,036,841 ns/iter (+/- 8,889,650)
```

TODO: talk and show profiling, arguing that exp is much more costly than multiplication and hence a good target for optimization.

# Specialization

Paillier'99 shows that we are free to choose any suitable `g`: from a security perspective they are all equal in that an adversary who can break one can break all without doing much more work. As a result, we can fix `g = 1 + n` as done in e.g. DJ'01.

To see the benefits of this from a performance perspective, notice that `g^x == (1 + n)^x == 1 + x*n` when computing modulo `n^2` because of the [Binomial theorem](https://en.wikipedia.org/wiki/Binomial_theorem), and `g^x == (1 + n)^x == 1` when computing modulo `n`. The former means that we can replace one of the exponentiations in encryption with a multiplication, and the latter that in extraction we can obtain `r^n` from a ciphertext simply by computing a modulus reduction!

TODO: it is not worth computing in nn if we know our final result will fit in n.

```rust
fn encrypt(ek: &EncryptionKey, x: &BigInt, r: &BigInt) -> BigInt {
    let gx = 1 + x * &ek.n;
    let rn = pow(r, &ek.n, &ek.nn);
    let c = (gx * rn) % &ek.nn;
    c
}

fn extract(dk: &DecryptionKey, c: &BigInt) -> BigInt {
    let rn = c % &dk.n;
    let r = pow(&rn, &dk.e, &dk.n);
    r
}
```

Having seen above that exponentiation is far more costly than multiplication, it is not too surprising that this cuts encryption time in half. But notice that extraction improved by almost a factor of 8!

```text
test specialized::bench_encrypt     ... bench:  11,464,487 ns/iter (+/- 1,854,213)
test specialized::bench_extract     ... bench:   3,333,744 ns/iter (+/- 966,572)
```

Decryption has untouched by this optimization, but next  we will look at other means of improving it.

# Chinese Remainder Theorem



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


```text
test precomputed::bench_encrypt     ... bench:       7,600 ns/iter (+/- 652)
```

```text
test crt::bench_decrypt             ... bench:   3,287,835 ns/iter (+/- 231,732)
test crt::bench_encrypt             ... bench:   6,524,388 ns/iter (+/- 386,847)
test crt::bench_extract             ... bench:     975,329 ns/iter (+/- 119,651)

test crt::parallel::bench_decrypt   ... bench:   1,847,150 ns/iter (+/- 500,361)
test crt::parallel::bench_encrypt   ... bench:   3,556,157 ns/iter (+/- 725,629)
test crt::parallel::bench_extract   ... bench:     584,474 ns/iter (+/- 115,368)
```

The following Rust [code](https://github.com/mortendahl/privateml/tree/master/paillier/benchmarks) shows the optimized decryption method, where [`rayon::join`](https://docs.rs/rayon/) is used to run the two computations in parallel.

```rust
fn decrypt(c: &BigInt, dk: &DecryptionKey) -> BigInt {
    let (mp, mq) = join(
        || decrypt_component(c, &dk.p, &dk.pp, &dk.d1p, &dk.d2p),
        || decrypt_component(c, &dk.q, &dk.qq, &dk.d1q, &dk.d2q),
    );
    crt(&mp, &mq, &dk.p, &dk.q, &dk.p_inv)
}

fn decrypt_component(c: &BigInt, m: &BigInt, mm: &BigInt, d1: &BigInt, d2: &BigInt) -> BigInt {
    let cm = c % mm;
    let dm = pow(&cm, d1, mm);
    let lm = l(&dm, m);
    let xm = (lm * d2) % m;
    xm
}

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

# Randomness


However, while we generally postpone efficient implementation to a follow-up post, one optimization is so straight-forward that we might as well deal with it here. In particular, since `n == p * q` for primes `p` and `q`, we have that `gcd(r, n)` can only be one of three values: `1`, `p`, or `q`. And of course, if we know either `p` or `q` then we know both, meaning that we have recovered the decryption key.

In other words, if one of underlying security assumptions of Paillier is valid, namely that it is impractical to factor `n` into `p` and `q` given no other information, then it must also be impractical to do so by specific method of sampling a random number and computing the GCD.

So in summary, due to the size of `n`, we are unlikely to ever sample a number `r` such that `gcd(r, n) != 1`. In fact, we are so unlikely that in practice we can safely assume that it will never happen!

With this knowledge in hand we can simply remove the check in our approach above and obtain the following.

```python
def generate_randomness(ek):
    return secrets.randbelow(ek.n)
```


As mentioned, the above optimization of encryption can only be applied by someone who knows the decryption key, which means it's only applicable in special cases. However, we can do a few other things.

The second is to note that the remaining exponentiation, `pow(r, ek.n, ek.nn)`, may be precomputed in an *offline* phase. This reduces the *online* encryption to two multiplications. Moreover, https://eprint.iacr.org/2015/864

```text
DNJ03 proposed a generalization of the scheme in which the
expansion factor is reduced and implementations of both the generalized and
original scheme are optimized without losing the homomorphic property [3].
Their system achieves the speed of 0.262 milliseconds/bit for the original Paillier scheme, equivalent to 3,816.79 bits/sec. This performance was reached by a
clever choice of basis and using standard pre-computation techniques for fixed
basis exponentiation. However, the encryption performance can be increased
even further by using the techniques described in this paper. We reach a speed
of 9,197,824 to 48,810,496 bits/sec depending on the security parameter.
```

## Re-randomization

```python
    sn = pow(s, ek.n, ek.nn)
    c_fresh = (c * sn) % ek.nn
```

## Offline computation

```rust
fn precompute_randomness(ek: &EncryptionKey, r: &BigInt) -> BigInt {
    modpow(r, &ek.n, &ek.nn)
}

fn encrypt_with_precomputed(dk: &DecryptionKey, m: &BigInt, rn: &BigInt) -> BigInt {
    let gm = (1 + m * &dk.n) % &dk.nn;
    (gm * rn) % &dk.nn
}
```


## Smaller randomness

Systems typically already use PRNs, what more can we do?

# Further Reading

We have not looked at optimizations that could be done below the GMP abstraction, including e.g. modular exponentiation with fixed exponent; see eg section 14.6 in HAC (used by all three operations).


- [`python-paillier`](https://github.com/n1analytics/python-paillier) [`rust-paillier`](https://github.com/mortendahl/rust-paillier) julia
- snips benchmark post
- [Public-Key Cryptosystems Based on Composite Degree Residuosity Classes](https://link.springer.com/chapter/10.1007%2F3-540-48910-X_16)
- DJ'01
- DJN'03
- link to paper and [Introduction to Modern Cryptography](http://www.cs.umd.edu/~jkatz/imc.html).

