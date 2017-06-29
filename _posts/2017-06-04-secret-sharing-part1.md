---
layout:     post
title:      "Secret Sharing, Part 1"
subtitle:   "Distributing Trust and Work"
date:       2017-06-04 12:00:00
header-img: "img/post-bg-02.jpg"
author:     "Morten Dahl"
---

<em><strong>TL;DR:</strong> first part in a series where we look at secret sharing schemes, including the lesser known packed variant of Shamir's scheme, and give full and efficient implementations; here we start with the textbook approaches, with follow-up posts focusing on improvements from more advanced techniques for [sharing](/2017/06/24/secret-sharing-part2) and reconstruction.</em>

[Secret sharing](https://en.wikipedia.org/wiki/Secret_sharing) is an old well-known cryptographic primitive, with existing real-world applications in e.g. [Bitcoin signatures](https://bitcoinmagazine.com/articles/threshold-signatures-new-standard-wallet-security-1425937098) and [password management](https://www.vaultproject.io/docs/internals/security.html). But perhaps more interestingly, secret sharing also has strong links to [secure computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and may for instance be used for [private machine learning](/2017/04/17/private-deep-learning-with-mpc/).

The essence of the primitive is that a *dealer* wants to split a *secret* into several *shares* given to *shareholders*, in such a way that each individual shareholder learns nothing about the secret, yet if sufficiently many re-combine their shares then the secret can be reconstructed. Intuitively, the question of *trust* changes from being about the integrity of a single individual to the non-collaboration of several parties: it becomes distributed.

Secret sharing schemes are also interesting from a performance point of view, as they typically rely on a bare minimum of cryptographic assumptions. In particular, by not having to make any assumptions about the hardness of certain problems such as [factoring integers](https://en.wikipedia.org/wiki/RSA_problem), [computing discrete logarithms](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange), or [finding short vectors](https://en.wikipedia.org/wiki/Ring_Learning_with_Errors), secret sharing schemes can provide a computational advantage compared to other cryptographic tools such as [homomorphic encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption).

In this post we'll look at a few concrete secret sharing schemes, as well as hints on how to implement them efficiently (with a later post going into more detail). We won't focus too much on applications but simply use private aggregation of large vectors as a running example -- see e.g. our [paper](TODO) for more use cases.

There is a Python notebook containing [the code samples](https://github.com/mortendahl/privateml/blob/master/secret-sharing/Schemes.ipynb), yet for better performance our [open source Rust library](https://crates.io/crates/threshold-secret-sharing) is recommended.

<em>
Parts of this blog post are derived from work done at [Snips](https://snips.ai/) and [originally appearing in another blog post](https://medium.com/snips-ai/high-volume-secret-sharing-2e7dc5b41e9a). That work also included parts of the Rust implementation.
</em>
 


# Additive Sharing

Let’s first assume that we have fixed a [finite field](https://en.wikipedia.org/wiki/Finite_field) to which all secrets and shares belong, and in which all computation take place; this could for instance be [the integers modulo a prime number](https://en.wikipedia.org/wiki/Modular_arithmetic), i.e. `{ 0, 1, ..., Q-1 }` for a prime `Q`.

An easy way to split a secret `x` from this field into say three shares `x1`, `x2`, `x3`, is to simply pick `x1` and `x2` at random and let `x3 = x - x1 - x2`. As argued below, this hides the secret as long as no one knows more than two shares, yet if all three shares are known then `x` can be reconstructed by simply computing `x1 + x2 + x3`. More generally, this scheme is known as *additive sharing* and works for any `N` number of shares by picking `T = N - 1` random values.

```python
def additive_share(secret):
    shares  = [ random.randrange(Q) for _ in range(N-1) ]
    shares += [ (secret - sum(shares)) % Q ]
    return shares

def additive_reconstruct(shares):
    return sum(shares) % Q
```

That the secret remains hidden as long as at most `T = N - 1` shareholders collaborate follows from the marginal distribution of the view of up to `T` shareholders being independent of the secret. More intuitively, given at most `T` shares, *any* guess one may make at what the secret could be, can be explained by the remaining unseen share, and is hence an equally valid guess.

```python
def explain(seen_shares, guess):
    # compute the unseen share that justifies the seen shares and the guess
    simulated_unseen_share = (guess - sum(seen_shares)) % Q
    # and the would-be sharing by combining seen and unseen shares
    simulated_shares = seen_shares + [simulated_unseen_share]
    if additive_reconstruct(simulated_shares) == guess:
        # found an explanation
        return simulated_unseen_share

seen_shares = shares[:N-1]

for guess in range(Q):
    explanation = explain(seen_shares, guess)
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

And since all we need for this argument to go through is the ability to sample random field elements, with no additional constraints on the size of the field due to e.g. hardness assumptions, this scheme is highly efficient both in terms of time and space.


## Homomorphic addition

While it is also about as simple as it gets, notice that the scheme already has a homomorphic property that allows for certain degrees of secure computation: we can add secrets together, so if e.g. `x1`, `x2`, `x3` is a sharing of `x` and `y1`, `y2`, `y3` is a sharing of `y`, then `x1+y1`, `x2+y2`, `x3+y3` is a sharing of `x + y`, which can be computed individually by the three shareholders simply by adding the shares they already have (respectively `x1` and `y1`, `x2` and `y2`, and `x3` and `y3`). Then, once added, these new shares can be used reconstruct the result of the addition but not the addends.

```python
def additive_add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]
```

More generally, we can ask the shareholders to compute linear functions of secret inputs without them seeing anything but the shares, and without learning anything besides the final output of the function.


# Comparing schemes

While the above scheme is particularly simple, below are two examples of slightly more advanced schemes. One way to compare these is through the following four parameters:

- `N`: the number of shares that each secret is split into

- `R`: the minimum number of shares needed to reconstruct the secret

- `T`: the maximum number of shares that may be seen without learning nothing about the secret, also known as the *privacy threshold*

- `K`: the number of secrets shared together

where, logically, we must have `R <= N` since otherwise reconstruction is never possible, and we must have `T < R` since otherwise privacy makes little sense.

For the additive scheme we have `R = N`, `K = 1`, and `T = R - K`, but below we will get rid of the first two of these constraints so that in the end we are free to choose the parameters any way we like as long as `T + K = R <= N`.


# Shamir’s Scheme

The additive scheme lacks some robustness by the constraint that `R = N`, meaning that if one of the shareholders for some reason becomes unavailable or losses his share then reconstruction is no longer possible. By moving to a different scheme we can remove this constraint and let `R` (and hence also `T`) be free to choose for any particular application.

In [Shamir’s scheme](https://en.wikipedia.org/wiki/Shamir%27s_Secret_Sharing), instead of picking random field elements that sum up to the secret `x` as we did above, to share `x` we sample a random polynomial `f` with the condition that `f(0) = x` and evaluate this polynomial at `N` non-zero points to obtain the shares as `f(1)`, `f(2)`, ..., `f(N)`.

```python
def shamir_share(secret):
    polynomial = sample_shamir_polynomial(secret)
    shares = [ evaluate_at_point(polynomial, p) for p in SHARE_POINTS ]
    return shares
```

And by varying the degree of `f` we can choose how many shares are needed before reconstruction is possible, thereby removing the `R = N` constraint.   More specifically, if the degree of `f` is `T` then we know from [interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) that it is uniquely identified by either its `T+1` coefficients or by its value at `T+1` points, so that `R = T+1` shares allow us to reliably reconstruct. 

```python
def shamir_reconstruct(shares):
    polynomial = [ (p, v) for p, v in zip(SHARE_POINTS, shares) if v is not None ]
    secret = interpolate_at_point(polynomial, 0)
    return secret
```

And at the same time, given at most `T` shares, the secret is again guaranteed to be hidden since we also here can find an explanation for any guess: a guess is the value of `f` at point zero, so together with the `T` known shares, interpolation allows us to find a polynomial with the right degree that matches all values.

Before discussing how these operations can be done efficiently, let's first see the properties this scheme has in terms of secure computation.


## Homomorphic addition and multiplication

Since it holds for polynomials in general that `f(i) + g(i) = (f + g)(i)`, we also here have an additive homomorphic property that allows us to compute linear functions of secrets by simply adding the individual shares.

```python
def shamir_add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]
```

And because it also holds that `f(i) * g(i) = (f * g)(i)`, we in fact now have an additional multiplicative property that allows us to compute products in the same fashion. 

```python
def shamir_mul(x, y):
    return [ (xi * yi) % Q for xi, yi in zip(x, y) ]
```

But while this is in principle enough to perform *any* computation without seeing the inputs (since addition and multiplication can be used to express any [boolean circuit](https://en.wikipedia.org/wiki/Boolean_circuit)), it also comes with a caveat: unlike addition, every multiplication doubles the degree of the polynomial, so we need `2T+1` shares to reconstruct a product instead of `T+1`. 

As a result, when used in secure computation, additional steps must be taken to reduce the degree after even a small number of multiplications, which typically involve some level of interaction between the shareholders. In this light, when compared to homomorphic encryption, secret sharing in some respect replaces heavy computation with interaction.


## The missing pieces

Above we ignored the questions of how to efficiently sample, evaluate, and interpolate polynomials. The first one is easy. We want a random `T` degree polynomial with the constraint that `f(0) = x`, and we may obtain that by simply letting the zero-degree coefficient be `x` and picking the remaining `T` coefficients at random: `f(X) = (x) + (r1 * X^1) + (r2 * X^2) + ... + (rT * X^T)` where `x` is the secret, `X` the indeterminate, and `r1`, ..., `rT` the random coefficients. 

```python
def sample_shamir_polynomial(zero_value):
    coefs = [zero_value] + [ random.randrange(Q) for _ in range(T) ]
    return coefs
```

This gives us the polynomial in coefficient representation, which means we can perform the second task of evaluating the polynomial at `N` points somewhat efficiently using e.g. [Horner's rule](https://en.wikipedia.org/wiki/Horner%27s_method).

```python
def evaluate_at_point(coefs, point):
    result = 0
    for coef in reversed(coefs):
        result = (coef + point * result) % Q
    return result
```

The interpolation step needed in reconstruction is slightly trickier. Here the polynomial is instead given in a point-value representation consisting of `T+1` pairs `(pi, vi)` that is less obviously suitable for computing `f(0)`. 

However, using [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) we can express the value of a polynomial at any point by a weighted sum of a set of constants and its value at `T+1` other points.

```python
def interpolate_at_point(points_values, point):
    points, values = zip(*points_values)
    constants = lagrange_constants_for_point(points, point)
    return sum( ci * vi for ci, vi in zip(constants, values) ) % Q
```

Moreover, since these *Lagrange constants* depend only on the points and not on the values, their computation can be amortized away in case we have to perform several interpolations, as in our running example with large vectors of secrets.

```python
def lagrange_constants_for_point(points, point):
    constants = [0] * len(points)
    for i in range(len(points)):
        xi = points[i]
        num = 1
        denum = 1
        for j in range(len(points)):
            if j != i:
                xj = points[j]
                num = (num * (xj - point)) % Q
                denum = (denum * (xj - xi)) % Q
        constants[i] = (num * inverse(denum)) % Q
    return constants
```

Looking back at the sharing and reconstruction operations, we then see that the former takes `Oh(N * T)` steps (for each secret) and the latter `Oh(T)` steps (for each secret) if precomputation is allowed.


# Packed Variant

While Shamir's scheme gets rid of the `R = N` constraint and gives us flexibility in choosing `T` or `R`, it still has the limitation that `K = 1`. This means that each shareholder receives one share per secret, so a large number of secrets means a large number of shares for each shareholder. By using a generalised variant of Shamir's scheme known as packed or ramp sharing, we can remove this limitation and reduce the load on each individual shareholder. 

To share a vector of `K` secrets `x = [x1, x2, ..., xK]`, the shares are still computed as `f(1)`, `f(2)`, ..., `f(N)` but the random polynomial is now sampled such that it satisfies `f(-1) = x1`, `f(-2) = x2`, ..., `f(-K) = xK`. 

Since it's less obvious how to sample such a polynomial in coefficient representation as we did before, to achieve the desires privacy threshold we instead add `T` additional constraints `f(-K-1) = r1`, ..., `f(-K-T) = rT` and simply use a point-value representation of the degree `T+K-1` polynomial.

```python
def sample_packed_polynomial(secrets):
    points = SECRET_POINTS + RANDOMNESS_POINTS
    values = secrets + [ random.randrange(Q) for _ in range(T) ]
    return list(zip(points, values))
```

This however means that we now have to perform interpolation instead of evaluation during sharing, which has an impact on efficiency, even when using precomputation as it now means storing `N` different sets of constants. 

```python
def packed_share(secrets):
    polynomial = sample_packed_polynomial(secrets)
    shares = [ interpolate_at_point(polynomial, p) for p in SHARE_POINTS ]
    return shares
```

As we will see in the next blog post it is in fact also possible to sample a packed polynomial in the coefficient representation and regain efficient sharing, but it requires slightly more advanced techniques.

```python
def packed_reconstruct(shares):
    points = SHARE_POINTS
    values = shares
    polynomial = [ (p,v) for p,v in zip(points, values) if v is not None ]
    return [ interpolate_at_point(polynomial, p) for p in SECRET_POINTS ]
```

Leaving computational efficiency aside, with this scheme we have reduced the number of shares each shareholder gets by a factor of `K`, which is useful in our running example of aggregating large vectors.

However there's another a caveat: since the degree of the polynomial increased, from `T` to `T + K - 1`, we also have to adjust either the privacy threshold or the number of shares needed to reconstruct.

For example, say we use Shamir's scheme to share a secret between `N = 10` shareholders and want a privacy guarantee against up to half of them collaborating, i.e. `T = 5`. Plugging this into our equation we get `5 + 1 = 6 <= 10` for Shamir's scheme, meaning we can tolerate that up to `N - R = 4`, or 40%, of them go missing. However, if we use the packed scheme to share `K = 3` secrets together then we get `5 + 3 = 8 <= 10` and the tolerance drops to 20%.

One remedy is to simply multiply all parameters by `K`; in the example we get `15 + 3 = 18 <= 30` and we are back to the original privacy threshold of half the shareholders and tolerance of 40%. The cost is that we now also need `K` times as many shareholders, so we have effectively kept the same number of shares but distributed them across a larger population.

(Note that a similar distribution may be achieved by partitioning the secrets and shareholders into `K` groups; this however has a negative effect on overall tolerance as we need `R` shares from each group.)


## Homomorphic addition and multiplication

The scheme has the same homomorphic properties as Shamir's, yet now operate in a [SIMD](https://en.wikipedia.org/wiki/SIMD) fashion where each addition or multiplication is simultaneously performed on every secret shared together. This in itself can have benefits if it fits naturally with the application.


# Next Steps

Although an old and simple primitive, secret sharing has several properties that makes it interesting as a way of delegating trust and computation to e.g. a community of users, even if the devices of these users are somewhat inefficient and unreliable.

In this post we have seen a few classical schemes as well as a typical textbook algorithms to implement them. [The next blog post](/2017/06/24/secret-sharing-part2) will improve on these algorithms and obtain significantly better performance.

