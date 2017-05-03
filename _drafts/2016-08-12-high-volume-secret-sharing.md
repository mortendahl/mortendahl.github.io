---
layout:     post
title:      "High-Volume Secret Sharing"
subtitle:   "and a way to distribute work"
date:       2016-08-12 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---


There already exist many implementations of what’s called [Shamir’s secret sharing](https://en.wikipedia.org/wiki/Shamir%27s_Secret_Sharing), but it turned out that for sharing a high volume of secrets, this is not always the best choice. As a result, we decided to implement a packed variant, with a focus on keeping it lightweight and efficient. To also achieve a high degree of portability we wrote it in Rust, and since we want to experiment with it in several applications we kept it as a self-contained library.

In a later post we’ll go into more details about how we can use this for secure computation, including summing high-dimensional vectors as part of our efforts to provide basic building blocks for analytics and machine learning.

<em>
Most of this blog post is derived from work done at [Snips](https://snips.ai/) and [originally appearing on their blog](https://medium.com/snips-ai/high-volume-secret-sharing-2e7dc5b41e9a). This includes an efficient [open source Rust implementation](https://github.com/snipsco/rust-threshold-secret-sharing) of the schemes.
</em>


## Secret Sharing
Secret sharing is a well-known cryptographic primitive, with strong links to [multi-party computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and with real-world applications in e.g. [Bitcoin signatures](https://bitcoinmagazine.com/articles/threshold-signatures-new-standard-wallet-security-1425937098) and [password management](https://www.vaultproject.io/docs/internals/security.html). As explained in detail [elsewhere](https://en.wikipedia.org/wiki/Secret_sharing), the essence of this primitive is that a *dealer* wants to split a *secret* into several *shares* to give to *shareholders*, such that if many of them combine their shares then the secret can be reconstructed, yet nothing is revealed about the secret if only a few come together (specifically, their marginal distribution is independent of the secret).

Let’s first assume that we have fixed a [finite field](https://en.wikipedia.org/wiki/Finite_field) to which all secrets and shares belong and in which all computation will take place; this could for instance be [the integers modulo a prime number](https://en.wikipedia.org/wiki/Modular_arithmetic). Then, to split a secret `x` into three shares `x1`, `x2`, `x3`, we may simply pick `x1` and `x2` at random and let `x3 = x - x1 - x2`. This way, unless all three shares are known, nothing whatsoever is revealed about `x`. Yet, if all three shares are known, `x` can be reconstructed by simply computing `x1 + x2 + x3`. More generally, the scheme works for any `N` shares by picking `N - 1` random values and offers privacy as long as at most `T = N - 1` shareholders combine their shares.

Notice that this simple scheme also has a homomorphic property that allows for certain degrees of [secure computation](https://en.wikipedia.org/wiki/Homomorphic_secret_sharing): it is additive, so if `x1`, `x2`, `x3` is a sharing of `x`, and `y1`, `y2`, `y3` is a sharing of `y`, then `x1+y1`, `x2+y2`, `x3+y3` is a sharing of `x + y`, which can be computed by the three shareholders simply by adding the shares they already have (i.e. respectively `x1` and `y1`, `x2` and `y2`, and `x3` and `y3`). More generally, we can compute functions of the secrets without seeing anything but the shares, and hence without learning anything about the secrets themselves.

Furthermore, the scheme also relies on a minimum of cryptographic assumptions: given only a source of (strong) random numbers there is no need for any further beliefs such as the hardness of [factoring integers](https://en.wikipedia.org/wiki/RSA_problem), [computing discrete logarithms](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange), or [finding short vectors](https://en.wikipedia.org/wiki/Ring_Learning_with_Errors). As a consequence, it is very efficient both in terms of time and space.

While the above scheme is particularly simple, below are two examples of slightly more advanced schemes. One way to characterise these is through the following four parameters:

- `N`: the number of shares that each secret is split into

- `R`: the minimum number of shares needed in order to reconstruct a secret

- `T`: the privacy threshold, i.e. the number of shares one may see while still learning nothing about the secret

- `L`: the number of secrets shared together

where, logically, we must have `R <= N` since otherwise reconstruction is never possible, and we must have `T < R` since otherwise privacy makes little sense. For the simple scheme above we furthermore have `R = N`, `L = 1`, and `T = R - L`, but below we’ll see how to get rid of the first two of these constraints so that in the end we are free to choose the parameters any way we like as long as `T + L = R <= N`.

## Shamir’s Scheme
By the constraint that `R = N`, the simple scheme above lacks some robustness, meaning that if one of the shareholders for some reason becomes unavailable then reconstruction is no longer possible. By moving to a different scheme we can remove this constraint and let `R` (and hence also `T`) be free to choose for any particular application.

In Shamir’s scheme, instead of picking random elements that sums up to the secret `x` as we did above, to share `x` we first pick a random polynomial `f` with the condition that `f(0) = x`, and then use `f(1)`, `f(2)`, ..., `f(N)` as the shares. And by varying the degree of `f` we can choose how many shares are needed before reconstruction is possible, thereby removing the `R = N` constraint.

Moreover, since it holds for polynomials in general that `f(i) + g(i) = (f + g)(i)`, we also here have an additive homomorphic property that allows us to obtain shares of `x + y` simply by adding together shares of `x` and `y`.


## homomorphic properties
TODO

## Packed Variant
Shamir’s scheme gets rid of the `R = N` constraint, but still requires `L = 1`: if we want to share several secrets then we have to share each one independently.

Luckily, Shamir’s scheme can be generalised to what we’ll called the packed scheme, which removes this last constraint. Now, instead of picking a random polynomial such that `f(0) = x`, to share a vector of `L` secrets `x = [x1, x2, ..., xL]`, we pick a random polynomial such that `f(-1) = x1`, `f(-2) = x2`, ..., `f(-L) = xL`, and like before use `f(1)`, `f(2)`, ..., `f(N)` as the shares.

With this scheme we are free to choose `L` as high as we want, as long as our choice of parameters satisfy `T + L = R <= N`. And we still have the additive homomorphic property so that secrets can be added securely.


## Efficient Implementation
As mentioned in the introduction, there are already several implementations of Shamir’s scheme (e.g. in Java, JavaScript, Python, C, and Rust) while implementations of the packed scheme are sparse; in fact, a quick Google search didn’t immediately yield any results at the time of writing.

Part of the reason for this could be that it’s not always clear that the packed scheme is better (obviously only if there are many secrets to share, but also only if there are many shareholders), and part of it could be that it is slightly more involved to implement efficiently. Indeed, our current implementation of the packed scheme relies on the Fast Fourier Transform over finite fields, whereas the typical implementation of Shamir’s scheme only needs a simple evaluation of polynomials.

While there are still plenty of improvements to be made, we already have decent performance. For instance, using the packed scheme to share 10,000 secrets to roughly 200 shareholders takes around 100ms on a laptop and less than 2s on a Raspberry Pi (for comparison, this is around 25 times quicker than doing the same with a typical implementation of Shamir’s scheme).


## Distributing Work

## Conclusion
Although an old and simple primitive, secret sharing has several properties that makes it interesting as a way of delegating trust and computation to e.g. a community of users, even if the devices of these users are somewhat inefficient and unreliable.

Implementing in Rust also turned out to have many benefits, not least due to the strong guarantees its type system provides, its highly efficient binaries, and its ease of cross-compilation.

The source code for the library is now available on GitHub, including examples and performance benchmarks.
