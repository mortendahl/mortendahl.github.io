---
layout:     post
title:      "Secret Sharing, Part 3"
subtitle:   "Robust Reconstruction via Reed-Solomon Codes"
date:       2017-07-01 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---

<em><strong>TL;DR:</strong> TODO</em>

As usual, all code is available in the [associated Python notebook](https://github.com/mortendahl/privateml/blob/master/secret-sharing/Reed-Solomon.ipynb).

# Robust Reconstruction

Returning to our [motivations](/2017/06/04/secret-sharing-part1/) for using secret sharing, namely to distribute trust, we recall that the shares we generate are given to shareholders that we may not trust individually. As such, if we later ask for the shares back in order to reconstruct the secret then it is natural to consider how reasonable it is to assume that we will receive the original shares back. 

Specifically, what if some shares are lost, or what if some shares are manipulated to differ from what we initially generated? Both may happen due to simple systems failure, but may also be the result of malicious behaviour on the part of shareholders. Should we in these two cases still expect to be able to recover the secret?

In the [first part](/2017/06/04/secret-sharing-part1/#the-missing-pieces) we saw how [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) can be used to answer the first question, in that it allows us to reconstruct the secret as long as only a bounded number of shares are lost. As mentioned in the [second part](/2017/06/24/secret-sharing-part2/#polynomials), this is due to the redundancy that comes with point-value presentations of polynomials, namely that the original polynomial is uniquely defined by *any* large enough subset of the shares. Concretely, if `D` is the degree of the original polynomial then we can reconstruct given `R = D + 1` shares in case of Shamir's scheme and `R = D + K` shares in the packed variant; if `N` is the total number of shares we can hence afford to loose `N - R` shares.

But this is assuming that the received shares are unaltered, and the second question concerning recovery in the face of manipulated shares is intuitively harder as we now cannot easily identify when and where something went wrong. <i>(Note that it is also harder in a more formal sense, namely that a solution for manipulated shares can also be used as a solution for lost shares, since dummy values, e.g. a constant, may be substituted for the lost shares and then instead treated as having been manipulated. As we will see below, this may however not be optimal, and doing so with the approach taken here will in fact impose a "double price" on lost shares.)</i>

To solve this issue we will use techniques from error-correction codes, specifically the well-known [Reed-Solomon codes](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction). As we shall see, we can do this since our share generation procedure in fact corresponds to ([non-systemic](https://en.wikipedia.org/wiki/Systematic_code)) message encoding in these codes, and hence decoding algorithms can be used to reconstruct even in the face of manipulated shares.

The robust reconstruct method for Shamir's scheme we end up with is as follows (with a straight forward generalisation to the packed scheme). The input is a complete list of length `N` of received shares, where missing shares are represented by `None` and manipulated shares by their new value. And if reconstruction goes well then the output is not only the secret, but also the indices of the shares that were manipulated.

```python
def shamir_robust_reconstruct(shares):
  
    # filter missing shares
    points_values = [ (p,v) for p,v in zip(POINTS, shares) if v is not None ]
    
    # decode polynomial from remaining shares
    points, values = zip(*points_values)
    polynomial = gao_decoding(points, values, MAX_DEGREE, MAX_MANIPULATED)
    
    # check whether recovery was successful under constraints
    if polynomial is None:
        # there were more errors than assumed by `MAX_ERRORS`
        raise Exception("Too many errors, cannot reconstruct")
    else:
        # recover secret
        secret = poly_eval(polynomial, 0)
        
        # find indices of faulty shares
        corrected_shares = [ poly_eval(polynomial, p) for p in POINTS ]
        error_indices = [ 
            i for i,(s,cs) in enumerate(zip(shares, corrected_shares)) 
            if s is not None and s != cs
        ]

        return secret, error_indices
```

Having the error indices may be useful for instance as a deterrent: since we can identify malicious shareholders we can also e.g. publicly shame them, and hence incentivise correct behaviour in the first place. Formally this is known as [covert security](https://en.wikipedia.org/wiki/Secure_multi-party_computation#Security_definitions), where the shareholders "are willing to cheat only if they are not caught".

Note that reconstruction may however fail, yet it can be shown that this only happens when there indeed isn't enough data left to correctly identify the result; in other words, our method will never give a false negative. Parameters `MAX_MISSING` and `MAX_MANIPULATED` are used to characterise when failure can happen, giving respectively an upper bound on the number of lost and manipulated shares supported. What must hold in general is that the number of "redundancy shares" `N - R` must satisfy `N - R >= MAX_MISSING + 2 * MAX_MANIPULATED`, from which we see that we are in some sense paying twice the price for manipulated shares as we are for missing shares.


## Outline of decoding algorithm

The specific decoding procedure we use here works by first finding the coefficients of an erroneous polynomial matching all received shares, including the manipulated ones. Hence we must first find a way to interpolate not only values but also coefficients from a polynomial given in point-value representation; in other words, we must find a way to convert from point-value representation to coefficient representation. We saw in [part two](/2017/06/24/secret-sharing-part2/) how the backward FFT can do this in specific cases, but to handle missing shares we here instead adapt [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) as used in [part one](/2017/06/04/secret-sharing-part1/).

Given the erroneous polynomial we then extract a corrected polynomial from it to get our desired result. Surprisingly, this may simply be done by running the [extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Polynomial_extended_Euclidean_algorithm) on polynomials as shown below.

Finally, since both of these two steps are using polynomials as objects of computation, similarly to how one typically uses integers as objects of computation, we must first also give algorithms for polynomial arithmetic such as adding and multiplying.


# Computing on Polynomials

We assume we already have various functions `base_add`, `base_sub`, `base_mul`, etc. for computing in the base field; concretely this simply amounts to [integer arithmetic modulo a fixed prime](https://en.wikipedia.org/wiki/Modular_arithmetic) in our case.

We then represent polynomials over this base field by their list of coefficients: `A(x) = (a0) + (a1 * x) + ... + (aD * x^D)` is represented by `A = [a0, a1, ..., aD]`. Furthermore, we keep as an invariant that `aD != 0` and enforce this below through a `canonical` procedure that removes all tailing zeroes in a list.

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

We have used simple algorithms for these operations in this blog post but referred to Modern Computer Algebra or .. for more efficient versions. TODO 
The standard algorithms will suffice for this blog post but note that optimizations are possible.


# Interpolation of Polynomials

We next turn to the task of converting a polynomial given in point-value representation to its coefficient representation. Several procedures exist for this, including efficient algorithms for specific cases such as the backward FFT seen earlier, and general ones based e.g. on [Newton's method](https://en.wikipedia.org/wiki/Newton_polynomial) that apparently is popular in numerical analysis due to its efficiency in handling new data points. However, as mentioned above, for this post we'll use Lagrange interpolation and see that although it's perhaps typically see as a procedure for interpolating values, it can also very easily be made to work for interpolating coefficients.


Recall that given points `x0, x1, ..., xD` and values `y0, y1, ..., yD` we can use Lagrange's method to interpolate the value at another point of the implicit polynomial defined by these two sets. The way we do this is by defining `L(x) = y0 * L0(x) + ... y    reconstruct a polynomial `F` of degree at most `D`

and values `y1, ..., yD`



In the [first post](/2017/06/04/secret-sharing-part1/) in this series we saw how [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial) can be used to find the value `f(x)` of a polynomial `f` at point `x` when `f` is given in a point-value representation `[(x1, v1), ..., (xL, vL)]` with `vi == f(x1)`. Specifically, given evaluation points `x1, ..., xL` and target point `x` we computed Lagrange constants `c1, ..., cL` such that `f(x) == c1 * v1 + ... + cL * vL`.

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


```python
def lagrange_polynomials(xs):
    polys = []
    for i, xi in enumerate(xs):
        numerator = [1]
        denominator = 1
        for j, xj in enumerate(xs):
            if i == j: continue
            numerator   = poly_mul(numerator, [base_sub(0, xj), 1])
            denominator = base_mul(denominator, base_sub(xi, xj))
        poly = poly_scalardiv(numerator, denominator)
        polys.append(poly)
    return polys
```

```python
def lagrange_interpolation(xs, ys):
    ls = lagrange_polynomials(xs)
    poly = []
    for i in range(len(ys)):
        term = poly_scalarmul(ls[i], ys[i])
        poly = poly_add(poly, term)
    return poly
```


# Correcting Polynomials

## Reed-Solomon codes

Primer on RS codes

While there are several such decoding procedures, in this blog post we will use [an algorithm](http://www.math.clemson.edu/~sgao/papers/RS.pdf) based on the [extended GCD algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Polynomial_extended_Euclidean_algorithm) as detailed below. [Other tutorials](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders) present alternative approaches that may prove more efficient from either an in-depth analysis or implementation benchmarks; nonetheless, the algorithm chosen here is conceptually simple and has a certain beauty to it. Moreover, keep in mind that some of the typical optimizations used in other approaches are for the more common setting over binary extension fields, while we here are interested in the setting over prime fields as we would like to simulate (bounded) integer arithmetic in our application of secret sharing to secure computation -- which is straight forward in prime fields but less clear in binary extension fields.

Algorithm first described in [SKHN'75](https://doi.org/10.1016/S0019-9958(75)90090-X) yet here we will follow the approach described in [Gao'02](http://www.math.clemson.edu/~sgao/papers/RS.pdf)

  [Gao's algorithm](http://www.math.clemson.edu/~sgao/papers/RS.pdf) as detailed below

Gao's decoding procedure as introduced in [Gao'02](http://www.math.clemson.edu/~sgao/papers/RS.pdf) (see also Section 17.5 in [Shoup'08](http://shoup.net/ntb/ntb-v2.pdf)) works by first interpolating a potentially faulty polynomial `H` from all the available shares, and then running a partial [extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Polynomial_extended_Euclidean_algorithm) to either extract the original polynomial `G` or (rightly) declare it impossible. 

## Extended GCD on polynomials

We can, as for the integers, use the [extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Polynomial_extended_Euclidean_algorithm) to find the greatest common divisor `D` of two given polynomials `F` and `H`, along with polynomials `S` and `T` expressing it as a linear combination of these: `D == F * S + H * T`.

```python
def poly_egcd(A, B):
    R0, R1 = A, B
    S0, S1 = [1], []
    T0, T1 = [], [1]
    
    while R1 != []:
        Q, R2 = poly_divmod(R0, R1)
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))
            
    c = lc(R0)
    D = poly_scalardiv(R0, c)
    S = poly_scalardiv(S0, c)
    T = poly_scalardiv(T0, c)
    return D, S, T
```

We will see below that this forms the core of decoding. TODO (rational reconstruction?)



http://www.math.clemson.edu/~sgao/papers/RS.pdf (see also Section 17.5 in [Shoup'08](http://shoup.net/ntb/))


## Gao's decoding

```python
def gao_decoding(points, values, max_degree, max_error_count):
    
    # interpolate faulty polynomial
    H = lagrange_interpolation(points, values)
    
    # compute f
    F = [1]
    for xi in points:
        Fi = [base_sub(0, xi), 1]
        F = poly_mul(F, Fi)
    
    # run partial-EGCD algorithm on (F,H) to find triple
    R0, R1 = F, H
    S0, S1 = [1], []
    T0, T1 = [], [1]
    while True:
        Q, R2 = poly_divmod(R0, R1)
        
        if deg(R0) < max_degree + max_error_count:
            G, leftover = poly_divmod(R0, T0)
            if leftover == []:
                return G
            else:
                return None
        
        R0, S0, T0, R1, S1, T1 = \
            R1, S1, T1, \
            R2, poly_sub(S0, poly_mul(S1, Q)), poly_sub(T0, poly_mul(T1, Q))
```



# Efficient Special Case

- using the FFT
- Fourier points
- no lost shares, or willing to pay double price for them
- backward FFT for interpolation
- optimise EEA: f is easy to compute
- see also Modern Computer Algebra for efficient algorithms with fewer restrictions (ie not relying on the FFT)




# Dump

<em><strong>TL;DR</strong> TODO</em>

TODO: use image form Voyager or from Mars (who used Reed-Solomon codes, [Wiki](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction))

In the [first part](/2017/06/04/secret-sharing-part1/) we introduced typical secret sharing schemes and gave general algorithms for the two procedures involved: *share generation* and *secret reconstruction*. In the [second part](/2017/06/24/secret-sharing-part2/) we looked at how the Fast Fourier Transform can in some cases be used to speed up the share generation procedure by accepting a few additional constraints, in particular the use 

[blog post](https://jeremykun.com/2015/09/07/welch-berlekamp/) "And the best part, for each two additional points you include above the minimum, you get resilience to one additional error no matter where it happens in the message." Note that this is a very slow way.

[name for algo in Shoup](http://drona.csa.iisc.ac.in/~priti/euclid-algorithm.pdf)

Reed-Solomon

[tutorial for binary fields](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders)

There are various approaches to decoding Reed-Solomon comes, but went with the EEA approach after looking around due to its simplicity and apparent equal-or-better performance.

TODO compare efficiency of original encoding vs BCH view: "Since Reed–Solomon codes are a special case of BCH codes, the practical decoders designed for BCH codes are applicable to Reed–Solomon codes" ([Wiki](https://en.wikipedia.org/wiki/Reed–Solomon_error_correction))

TODO decoding using DFT (Wiki) -- why would you do that? more efficient? see also https://www.cse.buffalo.edu//faculty/atri/courses/coding-theory/lectures/lect27.pdf which mentions an FFT algo (the DFT algo from Wiki?) https://tmo.jpl.nasa.gov/progress_report2/42-35/35K.PDF

References:
- [Shoup's book](http://www.shoup.net/ntb/)
- [Gao's paper](http://www.math.clemson.edu/~sgao/papers/RS.pdf)
- [Mateer's thesis](http://cr.yp.to/f2mult/mateer-thesis.pdf)


Optimisations we can get from using Fourier points when no shares are lost (or we are willing to "pay double" for lost shares)


# Next Steps

