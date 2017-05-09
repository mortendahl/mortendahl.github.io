---
layout:     post
title:      "Secret Sharing"
subtitle:   "and a way to securely combine large vectors"
date:       2016-08-12 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---


There already exist many implementations of what’s called [Shamir’s secret sharing](https://en.wikipedia.org/wiki/Shamir%27s_Secret_Sharing), but it turned out that for sharing a high volume of secrets, this is not always the best choice. As a result, we decided to implement a packed variant, with a focus on keeping it lightweight and efficient. To also achieve a high degree of portability we wrote it in Rust, and since we want to experiment with it in several applications we kept it as a self-contained library.

In a later post we’ll go into more details about how we can use this for secure computation, including summing high-dimensional vectors as part of our efforts to provide basic building blocks for analytics and machine learning.

<em>
A significant part of this blog post is derived from work done at [Snips](https://snips.ai/) and [originally appearing on their blog](https://medium.com/snips-ai/high-volume-secret-sharing-2e7dc5b41e9a). This includes an efficient [open source Rust implementation](https://github.com/snipsco/rust-threshold-secret-sharing) of the schemes.
</em>


## Basic Secret Sharing
Secret sharing is an old well-known cryptographic primitive, with strong links to [multi-party computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and with real-world applications in e.g. [Bitcoin signatures](https://bitcoinmagazine.com/articles/threshold-signatures-new-standard-wallet-security-1425937098) and [password management](https://www.vaultproject.io/docs/internals/security.html). As explained in detail [elsewhere](https://en.wikipedia.org/wiki/Secret_sharing), the essence of this primitive is that a *dealer* wants to split a *secret* into several *shares* to give to *shareholders*, such that if sufficiently many of them combine their shares then the secret can be reconstructed, yet nothing is revealed about it if only a few come together (specifically, their marginal distribution is independent of the secret).

Let’s first assume that we have fixed a [finite field](https://en.wikipedia.org/wiki/Finite_field) to which all secrets and shares belong and in which all computation will take place; this could for instance be [the integers modulo a prime number](https://en.wikipedia.org/wiki/Modular_arithmetic). Then, to split a secret `x` into three shares `x1`, `x2`, `x3`, we may simply pick `x1` and `x2` at random and let `x3 = x - x1 - x2`. This way, unless all three shares are known, nothing whatsoever is revealed about `x`. Yet, if all three shares are known then `x` can be reconstructed by simply computing `x1 + x2 + x3`. More generally, the scheme works for any `N` shares by picking `N - 1` random values and offers privacy as long as at most `T = N - 1` shareholders combine their shares. We call this *additive sharing*.

```python
def additive_share(secret):
    shares  = [ random.randrange(Q) for _ in range(N-1) ]
    shares += [ (secret - sum(shares)) % Q ]
    return shares

def additive_reconstruct(shares):
    return sum(shares) % Q
```

```rust
fn share(secret: i64, share_count: i64, modulus: i64) -> Vec<i64> {
    let mut rng = OsRng::new().expect("Unable to get randomness source");

    let mut shares: Vec<i64> = (0..share_count-1)
        .map(|_| rng.gen_range(0_i64, modulus))
        .collect();

    let mut last_share = shares.iter().fold(secret, |sum, &x| { (sum - x) % modulus });
    if last_share < 0 {
        last_share += modulus
    }
        
    shares.push(last_share);
    shares
}
```

```rust
fn reconstruct(shares: &[i64], modulus: i64) -> i64 {
    shares.iter().fold(0_i64, |sum, &x| { (sum + x) % modulus })
}
```

Notice that this simple scheme also has a homomorphic property that allows for certain degrees of [secure computation](https://en.wikipedia.org/wiki/Homomorphic_secret_sharing): it is additive, so if `x1`, `x2`, `x3` is a sharing of `x`, and `y1`, `y2`, `y3` is a sharing of `y`, then `x1+y1`, `x2+y2`, `x3+y3` is a sharing of `x + y`, which can be computed by the three shareholders by simply adding the shares they already have (i.e. respectively `x1` and `y1`, `x2` and `y2`, and `x3` and `y3`). More generally, we can compute functions of the secrets without seeing anything but the shares, and hence without learning anything about the secrets themselves.

Furthermore, the scheme also relies on a minimum of cryptographic assumptions: given only a source of (strong) random numbers there is no need for any further beliefs such as the hardness of [factoring integers](https://en.wikipedia.org/wiki/RSA_problem), [computing discrete logarithms](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange), or [finding short vectors](https://en.wikipedia.org/wiki/Ring_Learning_with_Errors). As a consequence, it is very efficient both in terms of time and space.

While the above scheme is particularly simple, below are two examples of slightly more advanced schemes. One way to characterise these is through the following four parameters:

- `N`: the number of shares that each secret is split into

- `R`: the minimum number of shares needed in order to reconstruct a secret

- `T`: the privacy threshold, i.e. the maximum number of shares that may be seen without learning nothing about the secret

- `K`: the number of secrets shared together

where, logically, we must have `R <= N` since otherwise reconstruction is never possible, and we must have `T < R` since otherwise privacy makes little sense. For the simple scheme above we furthermore have `R = N`, `K = 1`, and `T = R - K`, but below we’ll see how to get rid of the first two of these constraints so that in the end we are free to choose the parameters any way we like as long as `T + K = R <= N`.

## Shamir’s Scheme
By the constraint that `R = N`, the simple scheme above lacks some robustness, meaning that if one of the shareholders for some reason becomes unavailable then reconstruction is no longer possible. By moving to a different scheme we can remove this constraint and let `R` (and hence also `T`) be free to choose for any particular application.

In Shamir’s scheme, instead of picking random elements that sums up to the secret `x` as we did above, to share `x` we first sample a random polynomial `f` with the condition that `f(0) = x`. And by varying the degree of `f` we can choose how many shares are needed before reconstruction is possible, thereby removing the `R = N` constraint.

```python
def sample_shamir_polynomial(zero_value):
    return [zero_value] + [random.randrange(Q) for _ in range(T)]
```

Then, for the shares, we simply use the values of this polynomial at `N` non-zero points, say `f(1)`, `f(2)`, ..., `f(N)`. 

```python
def shamir_share(secret):
    polynomial = sample_shamir_polynomial(secret)
    shares = [ evaluate_at_point(polynomial, point) for point in range(1, N+1) ]
    return shares

def shamir_reconstruct(shares):
    points = range(1, N+1)
    values = shares
    points_values = [ (p, v) for p, v in zip(points, values) if v is not None ]
    return interpolate_at_zero(points_values)
```

Moreover, since it holds for polynomials in general that `f(i) + g(i) = (f + g)(i)`, we also here have an additive homomorphic property that allows us to obtain shares of `x + y` simply by adding together shares of `x` and `y`. 



## Packed Variant and Distributing Work
Shamir’s scheme gets rid of the `R = N` constraint, but still requires `K = 1`: if we want to share several secrets then we have to share each one independently.

Luckily, Shamir’s scheme can be generalised to what we’ll called the packed scheme, which removes this last constraint. Now, instead of picking a random polynomial such that `f(0) = x`, to share a vector of `K` secrets `x = [x1, x2, ..., xK]`, we pick a random polynomial such that `f(-1) = x1`, `f(-2) = x2`, ..., `f(-K) = xK`, and like before use `f(1)`, `f(2)`, ..., `f(N)` as the shares.

With this scheme we are free to choose `K` as high as we want, as long as our choice of parameters satisfy `T + K = R <= N`. And we still have the additive homomorphic property so that secrets can be added securely.


## Efficient Implementation
As mentioned in the introduction, there are already several implementations of Shamir’s scheme (e.g. in Java, JavaScript, Python, C, and Rust) while implementations of the packed scheme are sparse; in fact, a quick Google search didn’t immediately yield any results at the time of writing.

Part of the reason for this could be that it’s not always clear that the packed scheme is better (obviously only if there are many secrets to share, but also only if there are many shareholders), and part of it could be that it is slightly more involved to implement efficiently. Indeed, our current implementation of the packed scheme relies on the Fast Fourier Transform over finite fields, whereas the typical implementation of Shamir’s scheme only needs a simple evaluation of polynomials.

While there are still plenty of improvements to be made, we already have decent performance. For instance, using the packed scheme to share 10,000 secrets to roughly 200 shareholders takes around 100ms on a laptop and less than 2s on a Raspberry Pi (for comparison, this is around 25 times quicker than doing the same with a typical implementation of Shamir’s scheme).

```rust
pub fn fft2(a_coef: &[i64], omega: i64, prime: i64) -> Vec<i64> {
    if a_coef.len() == 1 {
        a_coef.to_vec()
    } else {
        // split A(x) into B(x) and C(x): A(x) = B(x^2) + x C(x^2)
        let b_coef: Vec<i64> = a_coef.iter().enumerate()
            .filter_map(|(x,&i)| if x % 2 == 0 { Some(i) } else { None } ).collect();
        let c_coef: Vec<i64> = a_coef.iter().enumerate()
            .filter_map(|(x,&i)| if x % 2 == 1 { Some(i) } else { None } ).collect();

        // recurse
        let b_point = fft2(&b_coef, mod_pow(omega, 2, prime), prime);
        let c_point = fft2(&c_coef, mod_pow(omega, 2, prime), prime);

        // combine
        let len = a_coef.len();
        let half_len = len >> 1;
        let mut a_point = vec![0; len];  // TODO trick: unsafe { Vec.set_len() }
        for i in 0..half_len {
            a_point[i]            = (b_point[i] + mod_pow(omega, i as u32, prime) * c_point[i]) % prime;
            a_point[i + half_len] = (b_point[i] - mod_pow(omega, i as u32, prime) * c_point[i]) % prime;
        }

        // return
        a_point
    }
}

pub fn fft2_inverse(a_point: &[i64], omega: i64, prime: i64) -> Vec<i64> {
    let omega_inv = mod_inverse(omega, prime);
    let len = a_point.len();
    let len_inv = mod_inverse(len as i64, prime);
    let scaled_a_coef = fft2(a_point, omega_inv, prime);
    let a_coef = scaled_a_coef.iter().map(|x| { x * len_inv % prime }).collect();
    a_coef
}
```

```rust
pub fn fft3(a_coef: &[i64], omega: i64, prime: i64) -> Vec<i64> {
    if a_coef.len() == 1 {
        a_coef.to_vec()
    } else {
        // split A(x) into B(x), C(x), and D(x): A(x) = B(x^3) + x C(x^3) + x^2 D(x^3)
        let b_coef: Vec<i64> = a_coef.iter().enumerate()
            .filter_map(|(x,&i)| if x % 3 == 0 { Some(i) } else { None } ).collect();
        let c_coef: Vec<i64> = a_coef.iter().enumerate()
            .filter_map(|(x,&i)| if x % 3 == 1 { Some(i) } else { None } ).collect();
        let d_coef: Vec<i64> = a_coef.iter().enumerate()
            .filter_map(|(x,&i)| if x % 3 == 2 { Some(i) } else { None } ).collect();

        // recurse
        let omega_cubed = mod_pow(omega, 3, prime);
        let b_point = fft3(&b_coef, omega_cubed, prime);
        let c_point = fft3(&c_coef, omega_cubed, prime);
        let d_point = fft3(&d_coef, omega_cubed, prime);

        // combine
        let len = a_coef.len();
        let third_len = len / 3;
        let mut a_point = vec![0; len];  // TODO trick: unsafe { Vec.set_len() }
        for i in 0..third_len {

            let j = i;
            let x = mod_pow(omega, j as u32, prime);
            let x_squared = (x * x) % prime;
            a_point[j] = (b_point[i] + x * c_point[i] + x_squared * d_point[i]) % prime;

            let j = i + third_len;
            let x = mod_pow(omega, j as u32, prime);
            let x_squared = (x * x) % prime;
            a_point[j] = (b_point[i] + x * c_point[i] + x_squared * d_point[i]) % prime;

            let j = i + third_len + third_len;
            let x = mod_pow(omega, j as u32, prime);
            let x_squared = (x * x) % prime;
            a_point[j] = (b_point[i] + x * c_point[i] + x_squared * d_point[i]) % prime;
        }

        // return
        a_point
    }
}

pub fn fft3_inverse(a_point: &[i64], omega: i64, prime: i64) -> Vec<i64> {
    let omega_inv = mod_inverse(omega, prime);
    let len = a_point.len();
    let len_inv = mod_inverse(len as i64, prime);
    let scaled_a_coef = fft3(a_point, omega_inv, prime);
    let a_coef = scaled_a_coef.iter().map(|x| { x * len_inv % prime }).collect();
    a_coef
}
```


## Conclusion
Although an old and simple primitive, secret sharing has several properties that makes it interesting as a way of delegating trust and computation to e.g. a community of users, even if the devices of these users are somewhat inefficient and unreliable.

Implementing in Rust also turned out to have many benefits, not least due to the strong guarantees its type system provides, its highly efficient binaries, and its ease of cross-compilation.

The source code for the library is now available on GitHub, including examples and performance benchmarks.
