---
layout:     post
title:      "Secret Sharing, Part 3"
subtitle:   "Robust Reconstruction via Reed-Solomon Codes"
date:       2017-08-13 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-jupiter.gif"
---

<em><strong>TL;DR:</strong> due to redundancy in the way shares are generated, we can compensate not only for some of them being lost but also for some being manipulated; here we look at how to do this using decoding methods for Reed-Solomon codes.</em>

Returning to our motivation in [part one](/2017/06/04/secret-sharing-part1/) for using secret sharing, namely to distribute trust, we recall that the generated shares are given to shareholders that we may not trust individually. As such, if we later ask for the shares back in order to reconstruct the secret then it is natural to consider how reasonable it is to assume that we will receive the original shares back. 

Specifically, what if some shares are *lost*, or what if some shares are *manipulated* to differ from the initially ones? Both may happen due to simple systems failure, but may also be the result of malicious behaviour on the part of shareholders. Should we in these two cases still expect to be able to recover the secret?

In this blog post we will see how to handle both situations. We will use simpler algorithms, but note towards the end how techniques like those used in [part two](/2017/06/24/secret-sharing-part2/) can be used to make the process more efficient.

As usual, all code is available in the [associated Python notebook](https://github.com/mortendahl/privateml/blob/master/secret-sharing/Reed-Solomon.ipynb).


# Robust Reconstruction

In the [first part](/2017/06/04/secret-sharing-part1/#the-missing-pieces) we saw how [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) can be used to answer the first question, in that it allows us to reconstruct the secret as long as only a bounded number of shares are lost. As mentioned in the [second part](/2017/06/24/secret-sharing-part2/#polynomials), this is due to the redundancy that comes with point-value presentations of polynomials, namely that the original polynomial is uniquely defined by *any* large enough subset of the shares. Concretely, if `D` is the degree of the original polynomial then we can reconstruct given `R = D + 1` shares in case of Shamir's scheme and `R = D + K` shares in the packed variant; if `N` is the total number of shares we can hence afford to loose `N - R` shares.

But this is assuming that the received shares are unaltered, and the second question concerning recovery in the face of manipulated shares is intuitively harder as we now cannot easily identify when and where something went wrong. <i>(Note that it is also harder in a more formal sense, namely that a solution for manipulated shares can be used as a solution for lost shares, since dummy values, e.g. a constant, may be substituted for the lost shares and then instead treated as having been manipulated. This however, is not optimal.)</i>

To solve this issue we will use techniques from error-correction codes, specifically the well-known [Reed-Solomon codes](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction). The reason we can do this is that share generation is very similar to ([non-systemic](https://en.wikipedia.org/wiki/Systematic_code)) message encoding in these codes, and hence their decoding algorithms can be used to reconstruct even in the face of manipulated shares.

The robust reconstruct method for Shamir's scheme we end up with is as follows, with a straight forward generalisation to the packed scheme. The input is a complete list of length `N` of received shares, where missing shares are represented by `None` and manipulated shares by their new value. And if reconstruction goes well then the output is not only the secret, but also the indices of the shares that were manipulated.

```python
def shamir_robust_reconstruct(shares):
  
    # filter missing shares
    points_values = [ (p,v) for p,v in zip(POINTS, shares) if v is not None ]
    
    # decode remaining faulty
    points, values = zip(*points_values)
    polynomial, error_locator = gao_decoding(points, values, R, MAX_MANIPULATED)
    
    # check if recovery was possible
    if polynomial is None:
        # there were more errors than assumed by `MAX_ERRORS`
        raise Exception("Too many errors, cannot reconstruct")
    else:
        # recover secret
        secret = poly_eval(polynomial, 0)
    
        # find roots of error locator polynomial
        error_indices = [ i 
            for i,v in enumerate( poly_eval(error_locator, p) for p in POINTS ) 
            if v == 0 
        ]

        return secret, error_indices
```

Having the error indices may be useful for instance as a deterrent: since we can identify malicious shareholders we may also be able to e.g. publicly shame them, and hence incentivise correct behaviour in the first place. Formally this is known as [covert security](https://en.wikipedia.org/wiki/Secure_multi-party_computation#Security_definitions), where shareholders are willing to cheat only if they are not caught.

Finally note that reconstruction may however fail, yet it can be shown that this only happens when there indeed isn't enough information left to correctly identify the result; in other words, our method will never give a false negative. Parameters `MAX_MISSING` and `MAX_MANIPULATED` are used to characterise when failure can happen, giving respectively an upper bound on the number of lost and manipulated shares supported. What must hold in general is that the number of "redundancy shares" `N - R` must satisfy `N - R >= MAX_MISSING + 2 * MAX_MANIPULATED`, from which we see that we are paying a double price for manipulated shares compared to missing shares.


## Outline of decoding algorithm

The specific decoding procedure we use here works by first finding an erroneous polynomial in coefficient representation that matches all received shares, including the manipulated ones. Hence we must first find a way to interpolate not only values but also coefficients from a polynomial given in point-value representation; in other words, we must find a way to convert from point-value representation to coefficient representation. We saw in [part two](/2017/06/24/secret-sharing-part2/) how the backward FFT can do this in specific cases, but to handle missing shares we here instead adapt [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) as used in [part one](/2017/06/04/secret-sharing-part1/).

Given the erroneous polynomial we then extract a corrected polynomial from it to get our desired result. Surprisingly, this may simply be done by running the [extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Polynomial_extended_Euclidean_algorithm) on polynomials as shown below.

Finally, since both of these two steps are using polynomials as objects of computation, similarly to how one typically uses integers as objects of computation, we must first also give algorithms for polynomial arithmetic such as adding and multiplying.


# Computing on Polynomials

We assume we already have various functions `base_add`, `base_sub`, `base_mul`, etc. for computing in the base field; concretely this simply amounts to [integer arithmetic modulo a fixed prime](https://en.wikipedia.org/wiki/Modular_arithmetic) in our case.

We then represent polynomials over this base field by their list of coefficients: `A(x) = (a0) + (a1 * x) + ... + (aD * x^D)` is represented by `A = [a0, a1, ..., aD]`. Furthermore, we keep as an invariant that `aD != 0` and enforce this below through a `canonical` procedure that removes all trailing zeros.

```python
def canonical(A):
    for i in reversed(range(len(A))):
        if A[i] != 0:
            return A[:i+1]
    return []
```

However, as an intermediate step we will sometimes first need to expand one of two polynomials to ensure they have the same length. This is done by simply appending zero coefficients to the shorter list.

```python
def expand_to_match(A, B):
    diff = len(A) - len(B)
    if diff > 0: 
        return A, B + [0] * diff
    elif diff < 0:
        diff = abs(diff)
        return A + [0] * diff, B
    else: 
        return A, B
```

With this we can perform arithmetic on polynomials by simply using the [standard definitions](https://en.wikipedia.org/wiki/Polynomial_arithmetic). Specifically, to add two polynomials `A` and `B` given by coefficient lists `[a0, ..., aM]` and `[b0, ..., bN]` we perform component-wise addition of the coefficients `ai + bi`. For example, adding `A(x) = 2x + 3x^2` to `B(x) = 1 + 4x^3` we get `A(x) + B(x) = (0+1) + (2+0)x + (3+0)x^2 + (0+4)x^3`; the first two are represented by `[0,2,3]` and `[1,0,0,4]` respectively, and their sum by `[1,2,3,4]`. Subtraction is similarly done component-wise.

```python
def poly_add(A, B):
    F, G = expand_to_match(A, B)
    return canonical([ base_add(f, g) for f, g in zip(F, G) ])
    
def poly_sub(A, B):
    F, G = expand_to_match(A, B)
    return canonical([ base_sub(f, g) for f, g in zip(F, G) ])
```

We also do scalar multiplication component-wise, i.e. by scaling every coefficient of a polynomial by an element from the base field. For instance, with `A(x) = 1 + 2x + 3x^2` we have `2 * A(x) = 2 + 4x + 6x^2`, which as expected is the same as `A(x) + A(x)`.

```python
def poly_scalarmul(A, b):
    return canonical([ base_mul(a, b) for a in A ])

def poly_scalardiv(A, b):
    return canonical([ base_div(a, b) for a in A ])
```

Multiplication of two polynomials is only slightly more complex, with coefficient `cK` of the product being defined by `cK = sum( aI * bJ for i,aI in enumerate(A) for j,bJ in enumerate(B) if i + j == K )`, and by changing the computation slightly we avoid iterating over `K`.

```python
def poly_mul(A, B):
    C = [0] * (len(A) + len(B) - 1)
    for i in range(len(A)):
        for j in range(len(B)):
            C[i+j] = base_add(C[i+j], base_mul(A[i], B[j]))
    return canonical(C)
```

We also need to be able to divide a polynomial `A` by another polynomial `B`, effectively finding a *quotient polynomial* `Q` and a *remainder polynomial* `R` such that `A == Q * B + R` with `degree(R) < degree(B)`. The procedure works like long-division for integers and is explained in details [elsewhere](https://www.khanacademy.org/math/algebra2/arithmetic-with-polynomials#long-division-of-polynomials).


```python
def poly_divmod(A, B):
    t = base_inverse(lc(B))
    Q = [0] * len(A)
    R = copy(A)
    for i in reversed(range(0, len(A) - len(B) + 1)):
        Q[i] = base_mul(t, R[i + len(B) - 1])
        for j in range(len(B)):
            R[i+j] = base_sub(R[i+j], base_mul(Q[i], B[j]))
    return canonical(Q), canonical(R)
```

Note that we have used basic algorithms for these operations here but that more efficient versions exist. Some pointers to these are given at the end.


# Interpolating Polynomials

We next turn to the task of converting a polynomial given in (implicit) point-value representation to its (explicit) coefficient representation. Several procedures exist for this, including efficient algorithms for specific cases such as the backward FFT seen earlier, and general ones based e.g. on [Newton's method](https://en.wikipedia.org/wiki/Newton_polynomial) that seem popular in numerical analysis due to its better efficiency and ability to handle new data points. However, for this post we'll use Lagrange interpolation and see that although it's perhaps typically see as a procedure for interpolating the values of polynomials, it also works just as well for interpolating their coefficients.

Recall that we are given points `x0, x1, ..., xD` and values `y0, y1, ..., yD` implicitly defining a polynomial `F`. [Earlier](/2017/06/04/secret-sharing-part1/) we then used [Lagrange's method](https://en.wikipedia.org/wiki/Lagrange_polynomial) to find value `F(x)` at a potentially different point `x`. This works due to the constructive nature of Lagrange's proof, where a polynomial `H` is defined as `H(X) = y0 * L0(X) + ... + yD * LD(X)` for indeterminate `X` and *Lagrange basis polynomials* `Li`, and then shown identical to `F`. To find `F(x)` we then simply evaluated `H(x)`, although we precomputed `Li(x)` as the *Lagrange constants* `ci` so that this step simply reduced to a weighted sum `y1 * c1 + ... yD * cD`.

```python
def lagrange_constants_for_point(points, point):
    constants = []
    for i, xi in enumerate(points):
        numerator = 1
        denominator = 1
        for j, xj in enumerate(points):
            if i == j: continue
            numerator   = base_mul(numerator, base_sub(point, xj))
            denominator = base_mul(denominator, base_sub(xi, xj))
        constant = base_div(numerator, denominator)
        constants.append(constant)
    return constants
```

Now, when we want the coefficients of `F` instead of just its value `F(x)` at `x`, we see that while `H` is identical to `F` it only gives us a semi-explicit representation, made worse by the fact that the `Li` polynomials are also only given in a semi-explicit representation: `Li(X) = (X - x0) * ... * (X - xD) / (xi - x0) * ... * (xi - xD)`. However, since we developed algorithms for using polynomials as objects in computations, we can simply evaluate these expression with indeterminate `X` to find the reduced explicit form! See for instance the examples [here](https://en.wikipedia.org/wiki/Lagrange_polynomial#Examples).

```python
def lagrange_polynomials(points):
    polys = []
    for i, xi in enumerate(points):
        numerator = [1]
        denominator = 1
        for j, xj in enumerate(points):
            if i == j: continue
            numerator   = poly_mul(numerator, [base_sub(0, xj), 1])
            denominator = base_mul(denominator, base_sub(xi, xj))
        poly = poly_scalardiv(numerator, denominator)
        polys.append(poly)
    return polys
```

Doing this also for `H` gives us the interpolated polynomial in explicit coefficient representation.

```python
def lagrange_interpolation(points, values):
    ls = lagrange_polynomials(points)
    poly = []
    for i, yi in enumerate(values):
        term = poly_scalarmul(ls[i], yi)
        poly = poly_add(poly, term)
    return poly
```

While this may not be the most efficient way (see notes later), it is hard to beat its simplicity.


# Correcting Errors

In the non-systemic variants of [Reed-Solomon codes](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction), a message `m` represented by a vector `[m0, ..., mD]` is encoded by interpreting it as a polynomial `F(X) = (m0) + (m1 * X) + ... + (mD * X^D)` and then evaluating `F` at a fixed set of points to get the code word. Unlike share generation, no randomness is used in this process since the purpose is only to provide redundancy and not privacy (in fact, in the systemic variants, the message is directly readable from the code word), yet this doesn't change the fact that we can use decoding procedures to correct errors in shares.

Several such [decoding procedures](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction#Error_correction_algorithms) exist, some of which are explained [here](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders) and [there](https://jeremykun.com/2015/09/07/welch-berlekamp/), yet the one we'll use here is conceptually simple and has a certain beauty to it. Also keep in mind that some of the typical optimizations used in implementations of the alternative approaches get their speed-up by relying on properties of the more common setting over binary extension fields, while we here are interested in the setting over prime fields as we would like to simulate (bounded) integer arithmetic in our application of secret sharing to secure computation -- which is straight forward in prime fields but less clear in binary extension fields.

The approach we will use was first described in [SKHN'75](https://doi.org/10.1016/S0019-9958(75)90090-X), yet we'll follow the algorithm given in [Gao'02](http://www.math.clemson.edu/~sgao/papers/RS.pdf) (see also Section 17.5 in [Shoup'08](http://shoup.net/ntb/ntb-v2.pdf)). It works by first interpolating a potentially faulty polynomial `H` from all the available shares and then running the extended Euclidean algorithm to either extract the original polynomial `G` or (rightly) declare it impossible. That the algorithm can be used for this is surprising and is strongly related to [rational reconstruction](https://en.wikipedia.org/wiki/Rational_reconstruction_(mathematics)).


## Extended Euclidean algorithm on polynomials

Assume that we have two polynomials `H` and `F` and we would like to find linear combinations of these in the form of triples `(R, T, S)` of polynomials such that `R == H * T + F * S`. This may of course be done in many different ways, but one particular interesting approach is to consider the list of triples `(R0, T0, S0), ..., (RM, TM, SM)` generated by the [extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Polynomial_extended_Euclidean_algorithm) (EEA).

```python
def poly_eea(F, H):
    R0, R1 = F, H
    S0, S1 = [1], []
    T0, T1 = [], [1]
    
    triples = []
    
    while R1 != []:
        Q, R2 = poly_divmod(R0, R1)
        
        triples.append( (R0, S0, T0) )
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))
            
    return triples
```

The reason for this is that this list turns out to represent *all* triples up to a certain size that satisfy the equation, in the sense that every "small" triple `(R, T, S)` for which `R == T * H + S * F` is actually just a scaled version of a triple `(Ri, Ti, Si)` occurring in the list generated by the EEA: for some constant `a` we have `R == a * Ri`, `T == a * Ti`, and `S == a * Si`. Moreover, given a concrete interpretation of "small" in the form of a degree bound on `R` and `T`, we may find the unique `(Ri, Ti, Si)` that this holds for.

Why this is useful in decoding becomes apparent next.


## Euclidean decoding

Say that `T` is the unknown error locator polynomial, i.e. `T(xi) == 0` exactly when share `yi` has been manipulated. Say also that `R = T * G` where `G` is the original polynomial that was used to generate the shares. Clearly, if we actually knew `T` and `R` then we could get what we're after by a simple division `R / T` -- but since we don't we have to do something else.

Because we're only after the ratio `R / T`, we see that knowing `Ri` and `Ti` such that `R == a * Ri` and `T == a * Ti` actually gives us the same result: `R / T == (a * Ri) / (a * Ti) == Ri / Ti`, and these we could potentially get from the EEA! The only obstacles are that we need to define polynomials `H` and `F`, and we need to be sure that there is a "small" triple with the `R` and `T` as defined here that satisfies the linear equation, which in turn means making sure there exists a suitable `S`. Once done, the output of `poly_eea(H, F)` will give us the needed `Ri` and `Ti`.

Perhaps unsurprisingly, `H` is the polynomial interpolated using all available values, which may potentially be faulty in case some of them have been manipulated. `F = F1 * ... * FN` is the product of polynomials `Fi(X) = X - xi` where `X` it the indeterminate and `x1, ..., xN` are the points.

Having defined `H` and `F` like this, we can then show that our `R` and `T` as defined above are "small" when the number of errors that have occurred are below the bounds discussed earlier. Likewise it can be shown that there is an `S` such that `R == T * H + S * F`; this involves showing that `R - T * H == S * F`, which follows from `R == H * T mod F` and in turn `R == H * T mod Fi` for all `Fi`. See standard textbooks for further details.

With this in place we have our decoding algorithm!

```python
def gao_decoding(points, values, max_degree, max_error_count):

    # interpolate faulty polynomial
    H = lagrange_interpolation(points, values)
    
    # compute f
    F = [1]
    for xi in points:
        Fi = [base_sub(0, xi), 1]
        F = poly_mul(F, Fi)
    
    # run EEA-like algorithm on (F,H) to find EEA triple
    R0, R1 = F, H
    S0, S1 = [1], []
    T0, T1 = [], [1]
    while True:
        Q, R2 = poly_divmod(R0, R1)
        
        if deg(R0) < max_degree + max_error_count:
            G, leftover = poly_divmod(R0, T0)
            if leftover == []:
                decoded_polynomial = G
                error_locator = T0
                return decoded_polynomial, error_locator
            else:
                return None
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))
```

Note however that it actually does more than promised above: it breaks down gracefully, by returning `None` instead of a wrong result, in case our assumption on the maximum number of errors turns out to be false. The intuition behind this is that if the assumption is true then `T` by definition is "small" and hence the properties of the EEA triple kick in to imply that the division is the same as `R / T`, which by definition of `R` has a zero remainder. And vice versa, if the remainder was zero then the returned polynomial is in fact less than the assumed number of errors away from `H` and hence `T` by definition is "small". In other words, `None` is returned if and only if our assumption was false, which is pretty neat. See [Gao'02](http://www.math.clemson.edu/~sgao/papers/RS.pdf) for further details.

Finally, note that it also gives us the error locations in the form of the roots of `T`. As mentioned earlier this is very useful from an application point of view, but could also have been obtained by simply comparing the received shares against a re-sharing based on the decoded polynomial.


# Efficiency Improvements

The algorithms presented above have time complexity `Oh(N^2)` but are not the most efficient. Based on the [second part](/2017/06/24/secret-sharing-part2/) we may straight away see how interpolation can be sped up by using the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) instead of Lagrange's method. One downside is that we then need to assume that `x1, ..., xN` are Fourier points, i.e. with a special structure, and we need to fill in dummy values for the missing shares and hence pay the double price. [Newton's method](https://en.wikipedia.org/wiki/Newton_polynomial) alternatively avoids this constraint while potentially giving better concrete performance than Lagrange's.

However, there are also other fast interpolation algorithms without these constraints, as detailed in for instance Modern Computer Algebra or [this thesis](http://cr.yp.to/f2mult/mateer-thesis.pdf), which also reduces the asymptotic complexity to `Oh(N * log N)`. This former reference also contains fast `Oh(N * log N)` methods for arithmetic and the EEA.


# Next Steps

The first three posts have been a lot of theory and it's now time to turn to applications.

