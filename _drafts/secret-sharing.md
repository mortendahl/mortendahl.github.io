---
layout:     post
title:      "Secret Sharing"
subtitle:   "and a way to efficiently distribute trust and work"
date:       2016-08-12 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-02.jpg"
---

[Secret sharing](https://en.wikipedia.org/wiki/Secret_sharing) is an old well-known cryptographic primitive, with existing real-world applications in e.g. [Bitcoin signatures](https://bitcoinmagazine.com/articles/threshold-signatures-new-standard-wallet-security-1425937098) and [password management](https://www.vaultproject.io/docs/internals/security.html). But perhaps more interestingly, secret sharing also has strong links to [secure computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and can for instance be used to perform private machine learning. 

The essence of the primitive is that a *dealer* wants to split a *secret* into several *shares* given to *shareholders*, in such a way that each individual shareholder learns nothing about the secret, yet if sufficiently many combine their shares then the secret can be reliably reconstructed. Intuitively, the question of *trust* changes from being about the integrity of a single individual to the non-collaboration of several parties: trust becomes distributed in other words.

In this post we will look at a few concrete secret sharing schemes, as well as how these can be implemented efficiently. We won't talk too much about applications, but use private aggregation of large vectors as a running example (and see e.g. [this paper](TODO) for use cases of this).


<em>
Parts of this blog post are derived from work done at [Snips](https://snips.ai/) and [originally appearing in other blog post](https://medium.com/snips-ai/high-volume-secret-sharing-2e7dc5b41e9a). That work also included an efficient [open source Rust implementation](https://crates.io/crates/threshold-secret-sharing) of the schemes and techniques presented here.
</em>


## Additive Sharing

Let’s first assume that we have fixed a [finite field](https://en.wikipedia.org/wiki/Finite_field) to which all secrets and shares belong and in which all computation take place; this could for instance be [the integers modulo a prime number](https://en.wikipedia.org/wiki/Modular_arithmetic), i.e. `{ 0, 1, ..., Q-1 }` for a prime `Q`. 

An easy way to split a secret `x` from this field into say three shares `x1`, `x2`, `x3`, is then to simply pick `x1` and `x2` at random and let `x3 = x - x1 - x2`. As we will argue below, this hides the secret as long as no one knows more than two shares, yet if all three shares are known then `x` can be reconstructed by simply computing `x1 + x2 + x3`. More generally, this scheme is known as *additive sharing* and works for any `N` number of shares by picking `T = N - 1` random values.

```python
def additive_share(secret):
    shares  = [ random.randrange(Q) for _ in range(N-1) ]
    shares += [ (secret - sum(shares)) % Q ]
    return shares

def additive_reconstruct(shares):
    return sum(shares) % Q
```

An important property is that it hides the secret as long as at most `T = N - 1` shareholders collaborate. This can be stated formally by saying that the marginal distribution of the view of up to `T` shareholders is independent of the secret. More intuitively though, we can say that given only `T` shares, then *any* guess we would make at what the secret is can be explained by the remaining unseen share; as a result, the uncertainty we have of the unseen share carries over to an uncertainty of the secret.

```python
seen_shares = shares[:N-1]

def explain_guess(guess):
    # look for unseen share that matches guess
    for unseen_share in range(Q):
        # form sharing by combining seen and unseen shares
        simulated_shares = seen_shares + [unseen_share]
        if additive_reconstruct(simulated_shares) == guess:
            # found an explanation
            return unseen_share

for guess in range(Q):
    explanation = explain_guess(guess)
    if explanation is not None: 
        print("guess %d can be explained by %d" % (guess, explanation))
```

```
guess 0 can be explained by 28
guess 1 can be explained by 29
guess 2 can be explained by 30
guess 3 can be explained by 31
guess 4 can be explained by 32
guess 5 can be explained by 33
...
```

Moreover, we also see that the scheme relies on a minimum of cryptographic assumptions: given only a source of (strong) random numbers there is no need for any further beliefs such as the hardness of [factoring integers](https://en.wikipedia.org/wiki/RSA_problem), [computing discrete logarithms](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange), or [finding short vectors](https://en.wikipedia.org/wiki/Ring_Learning_with_Errors). As a consequence, secret sharing can be made very efficient both in terms of time and space.


### Homomorphic addition

While the above scheme is perhaps as simple as it gets, notice that it already has a homomorphic property that allows for certain degrees of secure computation: we can add secret together, so if e.g. `x1`, `x2`, `x3` is a sharing of `x` and `y1`, `y2`, `y3` is a sharing of `y`, then `x1+y1`, `x2+y2`, `x3+y3` is a sharing of `x + y`, which can be computed individually by the three shareholders by simply adding the shares they already have (respectively `x1` and `y1`, `x2` and `y2`, and `x3` and `y3`). Then, once added, we can reconstruct only the result of the addition.

```python
def additive_add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]
```

More generally, we can compute linear functions of secret inputs without seeing anything but the shares, and hence without learning anything besides the final output of the function.


## Comparing schemes

While the above scheme is particularly simple, below are two examples of slightly more advanced schemes. One way to compare these is through the following four parameters:

- `N`: the number of shares that each secret is split into

- `R`: the minimum number of shares needed to reconstruct the secret

- `T`: the privacy threshold, i.e. the maximum number of shares that may be seen without learning nothing about the secret

- `K`: the number of secrets shared together

where, logically, we must have `R <= N` since otherwise reconstruction is never possible, and we must have `T < R` since otherwise privacy makes little sense. 

For the additive scheme above we have `R = N`, `K = 1`, and `T = R - K`, but below we will see how to get rid of the first two of these constraints so that in the end we are free to choose the parameters any way we like as long as `T + K = R <= N`.

## Shamir’s Scheme
By the constraint that `R = N`, the additive scheme lacks some robustness, meaning that if one of the shareholders for some reason becomes unavailable or losses his share then reconstruction is no longer possible. By moving to a different scheme we can remove this constraint and let `R` (and hence also `T`) be free to choose for any particular application.

In [Shamir’s scheme](https://en.wikipedia.org/wiki/Shamir%27s_Secret_Sharing), instead of picking random field elements that sums up to the secret `x` as we did above, to share `x` we sample a random polynomial `f` with the condition that `f(0) = x` and evaluate this polynomial at `N` non-zero points to obtain the shares as e.g. `f(1)`, `f(2)`, ..., `f(N)`.

```python
def shamir_share(secret):
    points = range(1, N+1)
    polynomial = sample_shamir_polynomial(secret)
    shares = [ evaluate_at_point(polynomial, p) for p in points ]
    return shares
```

And by varying the degree of `f` we can choose how many shares are needed before reconstruction is possible through [interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial), thereby removing the `R = N` constraint.  

```python
def shamir_reconstruct(shares):
    points = range(1, N+1)
    points_values = [ (p, v) for p, v in zip(points, shares) if v is not None ]
    secret = interpolate_at_zero(points_values)
    return secret
```

Moreover, since it holds for polynomials in general that `f(i) + g(i) = (f + g)(i)`, we also here have an additive homomorphic property that allows us to compute linear functions of secrets by simply adding the individual shares.

```python
def shamir_add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]
```

Finally, since it also holds that `f(i) * g(i) = (f * g)(i)`, we in fact now also have a multiplicative homomorphic property that allows us to compute products without learning the secrets. 

```python
def shamir_mul(x, y):
    return [ (xi * yi) % Q for xi, yi in zip(x, y) ]
```

But while this means that we can in principle do *any* computation without seeing the inputs, it also comes with a new caveat: since every multiplication doubles the degree of the polynomial, we need `2T+1` shares to reconstruct a product instead of `T+1`. As a result, TODO


### The missing pieces


```python
def sample_shamir_polynomial(zero_value):
    coefs = [zero_value] + [random.randrange(Q) for _ in range(T)]
    return coefs
```

```python
def evaluate_at_point(coefs, point):
    result = 0
    for coef in reversed(coefs):
        result = (coef + point * result) % Q
    return result
```

```python
def interpolate_at_zero(points_values):
    points, values = zip(*points_values)
    result = 0
    for i in range(len(values)):
        # compute Lagrange coefficient
        pi = points[i]
        num = 1
        denum = 1
        for j in range(len(values)):
            if j != i:
                pj = points[j]
                num = (num * pj) % Q
                denum = (denum * (pj - xi)) % Q
        # update sum
        vi = values[i]
        result = (result + vi * num * inverse(denum)) % Q
    return result
```

Specifically, to sample a polynomial `f` with degree `T` such that `f(0) = x` we write `f = a0 + a1 * X + ... + aT * X^T`




## Packed Variant
Shamir’s scheme gets rid of the `R = N` constraint, but still requires `K = 1`: if we want to share several secrets then we have to share each one independently.

Luckily, Shamir’s scheme can be generalised to what we’ll called the packed scheme, which removes this last constraint. Now, instead of picking a random polynomial such that `f(0) = x`, to share a vector of `K` secrets `x = [x1, x2, ..., xK]`, we pick a random polynomial such that `f(-1) = x1`, `f(-2) = x2`, ..., `f(-K) = xK`, and like before use `f(1)`, `f(2)`, ..., `f(N)` as the shares.

With this scheme we are free to choose `K` as high as we want, as long as our choice of parameters satisfy `T + K = R <= N`. And we still have the additive homomorphic property so that secrets can be added securely.

### Distributing work


## Efficient Implementation
As mentioned in the introduction, there are already several implementations of Shamir’s scheme (e.g. in Java, JavaScript, Python, C, and Rust) while implementations of the packed scheme are sparse; in fact, a quick Google search didn’t immediately yield any results at the time of writing.

Part of the reason for this could be that it’s not always clear that the packed scheme is better (obviously only if there are many secrets to share, but also only if there are many shareholders), and part of it could be that it is slightly more involved to implement efficiently. Indeed, our current implementation of the packed scheme relies on the Fast Fourier Transform over finite fields (also known as a Number Theoretic Transform), whereas the typical implementation of Shamir’s scheme only needs a simple evaluation of polynomials.

While there are still plenty of improvements to be made, we already have decent performance. For instance, using the packed scheme to share 10,000 secrets to roughly 200 shareholders takes around 100ms on a laptop and less than 2s on a Raspberry Pi (for comparison, this is around 25 times quicker than doing the same with a typical implementation of Shamir’s scheme).

```python
# len(aX) must be a power of 2
def fft2_forward(aX, omega=OMEGA2):
    if len(aX) == 1:
        return aX

    # split A(x) into B(x) and C(x) -- A(x) = B(x^2) + x C(x^2) -- and recurse
    bX = aX[0::2]
    cX = aX[1::2]
    B = fft2_forward(bX, (omega**2) % Q)
    C = fft2_forward(cX, (omega**2) % Q)
        
    # combine subresults
    A = [0] * len(aX)
    Nhalf = len(aX) // 2
    for i in range(Nhalf):
        delta = omega**i * C[i]
        A[i]         = (B[i] + delta) % Q
        A[i + Nhalf] = (B[i] - delta) % Q
    return A

def fft2_backward(A):
    N_inv = inverse(len(A))
    return [ (a * N_inv) % Q for a in fft2_forward(A, inverse(OMEGA2)) ]
```

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
