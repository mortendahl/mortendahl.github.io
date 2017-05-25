---
layout:     post
title:      "Secret Sharing"
subtitle:   "and a way to efficiently distribute trust and work"
date:       2017-05-04 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-02.jpg"
---

[Secret sharing](https://en.wikipedia.org/wiki/Secret_sharing) is an old well-known cryptographic primitive, with existing real-world applications in e.g. [Bitcoin signatures](https://bitcoinmagazine.com/articles/threshold-signatures-new-standard-wallet-security-1425937098) and [password management](https://www.vaultproject.io/docs/internals/security.html). But perhaps more interestingly, secret sharing also has strong links to [secure computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and can for instance be used to perform private machine learning. 

The essence of the primitive is that a *dealer* wants to split a *secret* into several *shares* given to *shareholders*, in such a way that each individual shareholder learns nothing about the secret, yet if sufficiently many combine their shares then the secret can be reliably reconstructed. Intuitively, the question of *trust* changes from being about the integrity of a single individual to the non-collaboration of several parties: trust becomes distributed in other words.

In this post we will look at a few concrete secret sharing schemes, as well as a few hints on how these can be implemented efficiently. We won't talk too much about applications, but use private aggregation of large vectors as a running example (but see e.g. [our paper](TODO) for more use cases).

There is also a Python notebook containing [all the code](https://github.com/mortendahl/privateml/blob/master/secret-sharing/Schemes.ipynb) used here.

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

An important property is that it hides the secret as long as at most `T = N - 1` shareholders collaborate. This can be stated formally by saying that the marginal distribution of the view of up to `T` shareholders is independent of the secret. More intuitively though, we can say that given only `T` shares, then *any* guess we would make at what the secret could be can be explained by the remaining unseen share; as a result, the uncertainty we have of the unseen share carries over to an uncertainty of the secret.

```python
seen_shares = shares[:N-1]

def explain_guess(guess):
    # look for unseen share that matches guess
    for unseen_share in range(Q):
        # form would-be sharing by combining seen and unseen shares
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

Moreover, we also see that the scheme relies on a minimum of cryptographic assumptions: given only a source of (strong) random numbers there is no need for any further beliefs such as the hardness of [factoring integers](https://en.wikipedia.org/wiki/RSA_problem), [computing discrete logarithms](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange), or [finding short vectors](https://en.wikipedia.org/wiki/Ring_Learning_with_Errors). As a consequence, there are no requirements on the size of the finite field and hence secret sharing can be made very efficient both in terms of time and space.


### Homomorphic addition

While the above scheme is perhaps as simple as it gets, notice that it already has a homomorphic property that allows for certain degrees of secure computation: we can add secrets together, so if e.g. `x1`, `x2`, `x3` is a sharing of `x` and `y1`, `y2`, `y3` is a sharing of `y`, then `x1+y1`, `x2+y2`, `x3+y3` is a sharing of `x + y`, which can be computed individually by the three shareholders by simply adding the shares they already have (respectively `x1` and `y1`, `x2` and `y2`, and `x3` and `y3`). Then, once added, we can reconstruct only the result of the addition.

```python
def additive_add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]
```

More generally, we can ask the shareholders to compute linear functions of secret inputs without them seeing anything but the shares, and without learning anything besides the final output of the function.


### Comparing schemes

While the above scheme is particularly simple, below are two examples of slightly more advanced schemes. One way to compare these is through the following four parameters:

- `N`: the number of shares that each secret is split into

- `R`: the minimum number of shares needed to reconstruct the secret

- `T`: the privacy threshold, i.e. the maximum number of shares that may be seen without learning nothing about the secret

- `K`: the number of secrets shared together

where, logically, we must have `R <= N` since otherwise reconstruction is never possible, and we must have `T < R` since otherwise privacy makes little sense.

For the additive scheme we have `R = N`, `K = 1`, and `T = R - K`, but below we will see how to get rid of the first two of these constraints so that in the end we are free to choose the parameters any way we like as long as `T + K = R <= N`.


## Shamir’s Scheme

By the constraint that `R = N`, the additive scheme above lacks some robustness, meaning that if one of the shareholders for some reason becomes unavailable or losses his share then reconstruction is no longer possible. By moving to a different scheme we can remove this constraint and let `R` (and hence also `T`) be free to choose for any particular application.

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

More specifically, if the degree of `f` is `T` then reconstruction is possible given `R = T+1` shares. Yet given at most `T` shares again ensures that the secret is hidden, since any guess at the secret always allows us to find a polynomial explaining everything we have seen.


### Homomorphic addition and multiplication

Since it holds for polynomials in general that `f(i) + g(i) = (f + g)(i)`, we also here have an additive homomorphic property that allows us to compute linear functions of secrets by simply adding the individual shares.

```python
def shamir_add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]
```

And since it also holds that `f(i) * g(i) = (f * g)(i)`, we in fact now also have a multiplicative property that allows us to compute products in the same fashion. 

```python
def shamir_mul(x, y):
    return [ (xi * yi) % Q for xi, yi in zip(x, y) ]
```

But while this means that we can in principle do *any* computation without seeing the inputs (since addition and multiplication can be used to express any [boolean circuit](https://en.wikipedia.org/wiki/Boolean_circuit)), it also comes with a new caveat: every multiplication doubles the degree of the polynomial so we need `2T+1` shares to reconstruct a product instead of `T+1`. As a result, when used in secure computation, additional steps are taken to reduce the degree after each multiplication.


### The missing pieces

Above we ignored the questions of how to efficiently sample, evaluate, and interpolate polynomials so let's return to that.

The first one is easy. We want a random `T` degree polynomial with the constraint that `f(0) = x`, and we may obtain this in *coefficient representation* by simply letting the zero-degree coefficient be `x` and picking the remaining `T` coefficients at random: `f(X) = (x) + (r1 * X^1) + (r2 * X^2) + ... + (rT * X^T)` where `x` is the secret, `X` the indeterminate, and `r1`, ..., `rT` the random coefficients.

```python
def sample_shamir_polynomial(zero_value):
    coefs = [zero_value] + [random.randrange(Q) for _ in range(T)]
    return coefs
```

In to this representation the second task can also be performed somewhat efficiently using e.g. [Horner's rule](https://en.wikipedia.org/wiki/Horner%27s_method).

```python
def evaluate_at_point(coefs, point):
    result = 0
    for coef in reversed(coefs):
        result = (coef + point * result) % Q
    return result
```

Where it gets slightly more tricky is the interpolation. Here the polynomial is no longer represented as a list of `T+1` coefficients but instead in a *point-value representation* consisting of `T+1` pairs `(pi, vi)` that is not obviously suitable for computing `f(0)` as needed in reconstruction. 

Luckily however, using [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) we can express the value of the polynomial at a fixed point by a weighted sum of a fixed set of constants and the values at other points.

```python
def interpolate_polynomial_at_zero(points_values):
    points, values = zip(*points_values)
    constants = lagrange_constants_at_zero(points)
    return sum( ci * vi for ci, vi in zip(constants, values) ) % Q
```

And since these *Lagrange constants* depend only on the points and not on the values, their computation may essentially be amortized away in case several interpolations have to be performed, as in our running example of sharing a large vector of secrets.

```python
def lagrange_constants_at_zero(points):
    constants = [0] * len(points)
    for i in range(len(points)):
        xi = points[i]
        num = 1
        denum = 1
        for j in range(len(points)):
            if j != i:
                xj = points[j]
                num = (num * xj) % Q
                denum = (denum * (xj - xi)) % Q
        constants[i] = (num * inverse(denum)) % Q
    return constants
```

While Shamir's scheme gets rid of the `R = N` constraint and give us a flexibility in choosing `T` or `R`, it still has the limitation that `K = 1`. The next scheme we will get rid of that as well.


## Packed Variant

In Shamir's scheme each shareholder receives one share per secret, so a large vector of secrets means a large number of shares per shareholder. By removing the `K = 1` constraint we can do better and reduce the load on each individual shareholder.

To do so we can use a generalised variant of Shamir's scheme known as packed or ramp sharing, where instead of sampling a random polynomial under the constraint that `f(0) = x`, we now sample a random polynomial such that `f(-1) = x1`, `f(-2) = x2`, ..., `f(-K) = xK` in order to share a vector of `K` secrets `x = [x1, x2, ..., xK]`. Like before we then use `f(1)`, `f(2)`, ..., `f(N)` as the shares, yet now each shareholder receives a factor of `K` less shares. 

```python
def packed_share(secrets):
    pass TODO
```

As always there's a caveat, and this time it's that the degree of the polynomial increases from `T` to `T + K - 1` in order to have the same privacy threshold, which in turn means that more shares are needed to reconstruct.

For example, say we use Shamir's scheme to share a secret between `N = 10` shareholders and want a privacy guarantee against up to `T = 5` of them collaborating. Plugging this into our equation we get `5 + 1 = 6 <= 10`, meaning we can tolerate that up to `N - R = 4`, or 40%, of them go missing. However, if we want to instead use the packed scheme to share `K = 3` secrets together then we get `5 + 3 = 8 <= 10` and the tolerance drops to 20%.

One remedy is to simply multiply all parameters by `K`; in the example we get `15 + 3 = 18 <= 30` and we are back to the original privacy threshold of half the shareholders and tolerance of 40%. The cost is that we now also need `K` times as many shareholders, but we have effectively reduced the individual load by distributing it across a larger group. (Note that a similar reduction may be achieved by partitioning the secrets and shareholders into `K` groups; this however has a negative effect on tolerance since it is now per group.)

```python
def packed_reconstruct(shares):
    pass TODO
```

With this scheme we are free to choose `K` as high as we want, as long as our choice of parameters satisfy `T + K <= R <= N`. And we still have the additive homomorphic property so that secrets can be added securely.

### Sampling

### SIMD


## Fast Fourier Transform

Indeed, our current implementation of the packed scheme relies on the Fast Fourier Transform over finite fields (also known as a Number Theoretic Transform), whereas the typical implementation of Shamir’s scheme only needs a simple evaluation of polynomials.

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
