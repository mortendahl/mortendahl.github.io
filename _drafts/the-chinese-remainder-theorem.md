---
layout:     post
title:      "The Chinese Remainder Theorem"
subtitle:   "Decomposing Numbers for Efficiency"
date:       2018-08-19 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---

<em><strong>TL;DR:</strong> TODO.</em> 

Many operations in cryptography is done in finite algebraic structures such as [groups](https://en.wikipedia.org/wiki/Group_(mathematics)), [rings](https://en.wikipedia.org/wiki/Ring_(mathematics)), or [fields](https://en.wikipedia.org/wiki/Field_(mathematics)). Or more concretely, they involve [modular arithmetic](https://en.wikipedia.org/wiki/Modular_arithmetic) using the numbers `0, 1, 2, ..., M-1` for some given modulus `M`. As we will see in this blog post, how these numbers are represented can make a big difference, from speeding up computations to enabling the use of libraries not intended for computation with large numbers.

Taschwer01
DDV19

# The Chinese Remainder Theorem

Also called the Chinese Remainder Map due to the isomorphic map it defines.

Numbers can be represented in many ways, most typically relative to some base such as `10` in human  , for instance `10` or `2**32`. The CRT provides an alternative

Knuth 2, section 4.3.2

Knuth 2, page 289: positional notation vs modular form

residue number systems

ring arithmetic

https://www.uow.edu.au/~thomaspl/pdf/talk04a.pdf
https://www.ams.org/journals/mcom/1967-21-098/S0025-5718-1967-0224252-5/S0025-5718-1967-0224252-5.pdf

For all examples in this section we'll use ring `Z_105`, i.e. numbers `0, ..., 104`, where the modulus breaks down as `105 == 3 * 5 * 7` and gives us our moduli `ms = (3, 5, 7)`. 

Converting a number to its CRT representation is very straight-forward: we simply take its residue with respect to each modulus as shown below; we'll call this operation decomposition. See also CF'05

```python
def decompose(x):
    return [ x % mi for mi in ms ]
```

For example, `decompose(10) == [1, 0, 3]` since `10 % 3 == 1`, `10 % 5 == 0`, and `10 % 7] == 3`. Likewise, `decompose(5) == [2, 0, 5]` since `5 % 3 == 2`, `5 % 5 == 0`, and `5 % 7 == 5`.

Going the other direction, from a number's CRT representation to it's natural representation, is slightly more involved. However, for each set of moduli there is a set of values `ls` that will ... ; 

```python
# precomputation
Mis = [ M // mi for mi in ms ]
ls = [ Mi * inv(Mi, mi) % M for Mi, mi in zip(Mis, ms) ]

def recombine(xs):
    return sum( xi * li for xi, li in zip(xs, ls) ) % M
```

One caveat is that each `li` value generally belongs to `M` instead of the corresponding `mi`, with the consequence that the entire recombination computation takes place in `M` as opposed to in `mi`. Hence, even if the value is small we are still using large numbers to recombine it. We shall return to this issue later, including some mitigations.

Cost of going back and forth between representations: want to avoid this when possible so it's interesting to look at which operations are easy to perform in the CRT representation and which are harder and potentially requires temporarily switching back to the natural representation.

## Basic operations

In the CRT representation the basic operations are simply done independently on each residue. For instance, since `decompose(10) = [1, 0, 3]` and `decompose(5) = [2, 0, 5]` we can find `decompose(10 + 5)` by simple adding up component-wise: `decompose(15) == [1+2, 0+0, 3+5] == [0, 0, 1]`. And the same applies to subtraction and multiplication.

Instead of adding in `M` we add independently in each `mi`.

```python
def crt_add(x, y):
    return [ (xi + yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_sub(x, y):
    return [ (xi - yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_mul(x, y):
    return [ (xi * yi) % mi for xi, yi, mi in zip(x, y, m) ]
```


A few additional operations also carry over this way, including [modular exponentiation](https://en.wikipedia.org/wiki/Modular_exponentiation) and [modular inversion](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse).

```python
def crt_pow(x, e):
    return [ pow(xi, e, mi) for xi, mi in zip(x, m) ]

def crt_inv(x):
    return [ inv(xi, mi) for xi, mi in zip(x, m) ]
```

## Sampling

uniform and bit bounded

## Lazy reduction

pick moduli that with extra room for supporting lazy reductions

```python
def crt_dot(x, y):
    return [ np.dot(xi, yi) % mi for xi, yi, mi in zip(x, y, m) ]
```

## Garner's algorithm and comparisons

```python
def recombine_garner(x):
    pass
```

See Knuth p 290 based on Garner

See Modern Computer Algebra sec 5.6

what does HAC say? (section 14.5, Garner's algorithm)

## The Explicit CRT and mods

what does the [explicit](http://cr.yp.to/papers/mmecrt.pdf) [CRT](http://cr.yp.to/antiforgery/meecrt-20060914-ams.pdf) say?

```python
def recombine_explicit(x):
    pass
```

see explicit CRT

### Unnormalized

```python
def crt_mod(x, k):
    pass
```

### Normalized

## Bit extraction

chunk by chunk

# Benchmarks

Here we switch to rust


## Efficient moduli

which moduli should be used?

see Knuth 2, page 12 + 288.

CF'05 section 10.4.3: Mersenne primes, but we actually don't need something that strong for the CRT (only coprime) so have a larger selection of numbers to pick from (instead of just primes)

What does Knuth say? (4.3.2)

CP'05 sec 9.2.3 (and 9.5.9)

## Benchmarks

Python with NumPy benchmarks in notebook. But those are not taking advantage of running in parallel. So to bench that we try in Rust.

```rust
impl Add<CrtScalar> for CrtScalar {
    type Output = CrtScalar;

    fn add(self, other: CrtScalar) -> Self::Output {
        let mut z = [0; NB_MODULI];
        z.iter_mut().enumerate()
            .for_each(|(i, zi)| {
                let xi = self[i];
                let yi = other[i];
                let mi = MODULI[i];
                *zi = (xi + yi) % mi
            });
        CrtScalar(z)
    }
}
```

# Applications

https://en.wikipedia.org/wiki/Chinese_remainder_theorem

(fixed) BigInt support
- can simulate integer arithemtic as long as we don't wrap around

## Paillier decryption

A typical performance use case is that of ciphertext decryption in the RSA or Paillier crypto system. Here all operations happen modulu an `M` which is  the private decryption key 

Here for instance is the decryption procedure used in the [`rust-paillier`](https://crates.io/crates/paillier) library, which first decomposes the an integer from `N^2` into two integers in respectively `P^2` and `Q^2`. Since `N = P * Q` this already means a performance speedup since we are now working with integers half the size as before, but it also enables using multiple cores as done here via [`rayon`](https://crates.io/crates/rayon)giving a further speedup when a second core is available.

```rust
fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
    let dn = pow(&c, &dk.phi, &dk.nn);
    let ln = l(&dn, &dk.n);
    let mn = (&ln * &dk.hn) % &dk.n;
    mn
}
```

```rust
fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
    let dn = pow(&c, &dk.pminusone * &dk.qminusone, &dk.pp * &dk.qq);
    let ln = l(&dn, &dk.p * &dk.q);
    let mn = (&ln * &dk.hn) % (&dk.p * &dk.q);
    mn
}
```

```rust
fn decrypt(dk: &DecryptionKey, c: &BigInt) -> BigInt {
    // decompose the ciphertext
    let (cp, cq) = decompose(&c, &dk.pp, &dk.qq);
    
    // decrypt in parallel with respect to p and q
    let (mp, mq) = rayon::join(
        || {
            // process using p
            let dp = pow(&cp, &dk.pminusone, &dk.pp);
            let lp = l(&dp, &dk.p);
            let mp = (&lp * &dk.hp) % &dk.p;
            mp
        },
        || {
            // process using q
            let dq = pow(&cq, &dk.qminusone, &dk.qq);
            let lq = l(&dq, &dk.q);
            let mq = (&lq * &dk.hq) % &dk.q;
            mq
        }
    );
    
    // recombine into decryption with respect to n
    let m = recombine(mp, mq, &dk.p, &dk.q, &dk.pinv);
    m
}
```

It may also be natural to ask if we can also speed up encryptions this way, however the entire security of the Paillier scheme relies `P` and `Q` remaining private and hence the factorisation of `N` cannot be revealed, and hence it wouldn't be safe to reveal the moduli that breaks `M` down.

## Secure computation in TensorFlow

Several specialised libraries for matrix operations exist, the most popular of which having received a significant amount of research and engineering to make them as efficient as possible. However, using these for large integer operations may not have been a priority as most are limited to `int32` or `int64` values if any integer support at all.

One such library is TensorFlow, which current seem to only support `int64` and in some cases (e.g. matrix multiplication) only `int32` values, another is `ArrayFire` with the same restrictions. One possible explaination for this is a lack of GPU kernels for `int64` matrix multiplication.

# Dump


aka Chinese Remainder Map

https://en.wikipedia.org/wiki/Chinese_remainder_theorem

what does BZ say?

what does GG say?

what does GCL say? ([section 5.6](https://www.csee.umbc.edu/~lomonaco/s08/441/handouts/GarnerAlg.pdf))

what is mixed-radix representation?