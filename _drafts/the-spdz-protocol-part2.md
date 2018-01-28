---
layout:     post
title:      "More Fun with Triples"
subtitle:   "Adapting the SPDZ protocol"
date:       2017-09-10 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-04.jpg"
---

<em><strong>TL;DR:</strong> we take a typical CNN deep learning model and go through a series of steps that enable both training and prediction to instead be done on encrypted data.</em> 



## Underlying principles

## Squaring

## Powering

As an alternative we can again use a new kind of preprocessed triple that allows exponentiation to all required powers to be done in a single round. The length of these "triples" is not fixed but equals the highest exponent, such that a triple for squaring, for instance, consists of independent sharings of `a` and `a^2`, while one for cubing consists of independent sharings of `a`, `a^2`, and `a^3`.

```python
def generate_pows_triple(exponent, shape):
    a = np.random.randint(Q, size=shape)
    return [ share(np.power(a, e) % Q) for e in range(1, exponent+1) ]
```

To use these we notice that if `epsilon = x - a` then `x^n == (epsilon + a)^n`, which by [the binomal theorem](https://en.wikipedia.org/wiki/Binomial_theorem) may be expressed as a weighted sum of `epsilon^n * a^0`, ..., `epsilon^0 * a^n` using the [binomial coefficients](https://en.wikipedia.org/wiki/Binomial_coefficient) as weights. For instance, we have `x^3 == (c0 * epsilon^3) + (c1 * epsilon^2 * a) + (c2 * epsilon * a^2) + (c3 * a^3)` with `ck = C(3, k)`.

Moreover, a triple for e.g. cubing `x` can also simultaneously be used for squaring `x` simply by skipping some powers and computing different binomial coefficients. Hence, all intermediate powers may be computed using a single triple and communication of one field element. The security of this again follows from the fact that all powers in the triple are independently shared.

```python
def pows(x, triple):
    # local masking
    a = triple[0]
    v = sub(x, a)
    # communication: the players simultanously send their share to the other
    epsilon = reconstruct(v)
    # local combination to compute all powers
    x_powers = []
    for exponent in range(1, len(triple)+1):
        # prepare all term values
        a_powers = [ONE] + triple[:exponent]
        e_powers = [ pow(epsilon, e, Q) for e in range(exponent+1) ]
        coeffs   = [ binom(exponent, k) for k in range(exponent+1) ]
        # compute and sum terms
        terms = ( mul_public(a,e*c) for a,e,c in zip(a_powers,reversed(e_powers),coeffs) )
        x_powers.append(reduce(lambda x,y: add(x, y), terms))
    return x_powers
```

Once we have these powers of `x`, evaluating a polynomial with public coefficients is then just a weighted sum.

```python
def pol_public(x, coeffs, triple):
    powers = [ONE] + pows(x, triple)
    terms = ( mul_public(xe, ce) for xe,ce in zip(powers, coeffs) )
    return reduce(lambda y,z: add(y, z), terms)
```

## Share conversion

There is one caveat however, and that is that we now need room for the higher precision of the powers: `x^n` has `n` times the precision of `x` and we want to make sure that this value does not wrap around modulo `Q`.

One way around this is to temporarily switch to a larger field and compute the powers and truncation there. The conversion to and from this larger field `P` each take one round of communication, so polynomial evaluation ends up taking a total of three rounds. 

Security wise we also have to pay a small price, although from a practical perspective there is little difference. In particular, for this operation we rely on *statistical security* instead of perfect security: since `r` is not an uniform random element here, there's a tiny risk that something will be leaked about `x`.

```python
def generate_statistical_mask():
    return random.randrange(2*BOUND * 10**KAPPA)

def generate_zero_triple(field):
    return share(0, field)

def convert(x, from_field, to_field, zero_triple):
    # local mapping to positive representation
    x = add_public(x, BOUND, from_field)
    # local masking and conversion by player 0
    r = generate_statistical_mask()
    y0 = (zero_triple[0] - r) % to_field
    # exchange of masked share: one round of communication
    e = (x[0] + r) % from_field
    # local conversion by player 1
    xr = (e + x[1]) % from_field
    y1 = (zero_triple[1] + xr) % to_field
    # local mapping back from positive representation
    y = [y0, y1]
    y = sub_public(y, BOUND, to_field)
    return y

def upshare(x, large_zero_triple):
    return convert(x, Q, P, large_zero_triple)

def downshare(x, small_zero_triple):
    return convert(x, P, Q, small_zero_triple)
```

Note that we could of course decide to simply do all computations in the larger field `P`, thereby avoiding the conversion steps. This will likely slow down the local computations by a non-trivial factor however, as we may need arbitrary precision arithmetic for `P` as opposed to e.g. 64 bit native arithmetic for `Q`.

Practical experiments will show whether it best to stay in `Q` and use a few more rounds, or switch temporarily to `P` and pay for conversion and arbitrary precision arithmetic. Specifically, for low degree polynomials the former is likely better.

## Bit-decomposition


## Generalised triples

When seeking to reduce communication, one may also wonder how much can be pushed to the preprocessing phase in the form of additional types of triples.

As mentioned earlier, we might seek to ensure that each private value is only sent masked once. So if we are e.g. computing both `dot(X, Y)` and `dot(X, Z)` then it might make sense to have a triple `(R, S, T, U, V)` that allows us to compute both results yet only send `X` masked once, as done in e.g. [BCG+'17](https://eprint.iacr.org/2017/1234). 

One relevant case is training, where some values are used to compute both the output of the layer during the forward phase, but also typically cached and used again to update the weights during the backward phase (for instance in dense layers). 

Another, perhaps more important case, is if we are only interested in during prediction:  TODO TODO TODO

Additionally, it might also be possible to have triples for more advanced functions such as evaluating both a dense layer and its activation function with a single round of communication. Main question here again seems to be efficiency, this time in terms of triple storage and amount of computation needed for the recombination step.


