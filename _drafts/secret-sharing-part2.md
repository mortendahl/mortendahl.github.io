---
layout:     post
title:      "Secret Sharing, Part 2"
subtitle:   "Efficient Sharing and Reconstruction"
date:       2017-06-05 12:00:00
header-img: "img/post-bg-03.jpg"
author:           "Morten Dahl"
twitter_username: "mortendahlcs"
github_username:  "mortendahl"
---

In the [first part](/2017/06/04/secret-sharing-part1/) on secret sharing we looked at Shamir's scheme and its packed variant where several secrets are shared together. In both schemes do we use the values of a polynomial 

There is a Python notebook containing [the code samples](https://github.com/mortendahl/privateml/blob/master/secret-sharing/Fast%20Fourier%20Transform.ipynb), yet for better performance our [open source Rust library](https://crates.io/crates/threshold-secret-sharing) is recommended.

<em>
Parts of this blog post are derived from work done at [Snips](https://snips.ai/) and [originally appearing in another blog post](https://medium.com/snips-ai/high-volume-secret-sharing-2e7dc5b41e9a). That work also included parts of the Rust implementation.
</em>


# Fast Fourier Transform

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
        x = (omega**i * C[i]) % Q
        A[i]         = (B[i] + x) % Q
        A[i + Nhalf] = (B[i] - x) % Q
    return A

def fft2_backward(A):
    N_inv = inverse(len(A))
    return [ (a * N_inv) % Q for a in fft2_forward(A, inverse(OMEGA2)) ]
```

```python
# len(aX) must be a power of 3
def fft3_forward(aX, omega=OMEGA3):
    if len(aX) == 1:
        return aX

    # split A(x) into B(x), C(x), and D(x): A(x) = B(x^3) + x C(x^3) + x^2 D(x^3)
    bX = aX[0::3]
    cX = aX[1::3]
    dX = aX[2::3]
    
    # apply recursively
    omega_cubed = (omega**3) % Q
    B = fft3_forward(bX, omega_cubed)
    C = fft3_forward(cX, omega_cubed)
    D = fft3_forward(dX, omega_cubed)
        
    # combine subresults
    A = [0] * len(aX)
    Nthird = len(aX) // 3
    for i in range(Nthird):
        
        j = i
        x = (omega**j) % Q
        xx = (x * x) % Q
        A[j] = (B[i] + x * C[i] + xx * D[i]) % Q
        
        j = i + Nthird
        x = (omega**j) % Q
        xx = (x * x) % Q
        A[j] = (B[i] + x * C[i] + xx * D[i]) % Q
        
        j = i + Nthird + Nthird
        x = (omega**j) % Q
        xx = (x * x) % Q
        A[j] = (B[i] + x * C[i] + xx * D[i]) % Q

    return A

def fft3_backward(A):
    N_inv = inverse(len(A))
    return [ (a * N_inv) % Q for a in fft3_forward(A, inverse(OMEGA3)) ]
```

# Application to Secret Sharing

## Shamir's scheme

```python
def shamir_share(secret):
    small_coeffs = [secret] + [random.randrange(Q) for _ in range(T)]
    large_coeffs = small_coeffs + [0] * (ORDER3 - len(small_coeffs))
    large_values = fft3_forward(large_coeffs)
    shares = large_values
    return shares
```


## Packed scheme

```python
def packed_share(secrets):
    small_values = [0] + secrets + [random.randrange(Q) for _ in range(T)]
    small_coeffs = fft2_backward(small_values)
    large_coeffs = small_coeffs + [0] * (ORDER3 - ORDER2)
    large_values = fft3_forward(large_coeffs)
    shares = large_values[1:]
    return shares
```

```python
def packed_reconstruct(shares):
    large_values = [0] + shares
    large_coeffs = fft3_backward(large_values)
    small_coeffs = large_coeffs[:ORDER2]
    small_values = fft2_forward(small_coeffs)
    secrets = small_values[1:K+1]
    return secrets
```

# Parameter Generation

```python
def generate_parameters(min_bitsize, k, t, n):
    order2 = k + t + 1
    order3 = n + 1
    
    order_divisor = order2 * order3
    p, g = find_prime_field(min_bitsize, order_divisor)
    
    order = p - 1
    omega2 = pow(g, order // order2, p)
    omega3 = pow(g, order // order3, p)
    
    return p, omega2, omega3
```

```python
def find_prime_field(min_bitsize, order_divisor):
    p, order_prime_factors = find_prime(min_bitsize, order_divisor)
    g = find_generator(p, order_prime_factors)
    return p, g
```

```python
def find_generator(prime, order_prime_factors):
    order = prime - 1
    for candidate in range(2, Q):
        for factor in order_prime_factors:
            exponent = order // factor
            if pow(candidate, exponent, Q) == 1:
                break
        else:
            return candidate
```

## Finding primes

```python
def find_prime(min_bitsize, order_divisor):
    while True:
        k1 = sample_prime(min_bitsize)
        for k2 in range(128):
            p = k1 * k2 * order_divisor + 1
            if is_prime(p):
                order_prime_factors  = [k1]
                order_prime_factors += prime_factor(k2)
                order_prime_factors += prime_factor(order_divisor)
                return p, order_prime_factors
```

```python
def sample_prime(bitsize):
    lower = 1 << (bitsize-1)
    upper = 1 << (bitsize)
    while True:
        candidate = random.randrange(lower, upper)
        if is_prime(candidate):
            return candidate
```

```python
def prime_factor(x):
    factors = []
    for prime in SMALL_PRIMES:
        if prime > x: break
        if x % prime == 0:
            factors.append(prime)
            x = remove_factor(x, prime)
    assert(x == 1)
    return factors
```


TODO

# Conclusion

