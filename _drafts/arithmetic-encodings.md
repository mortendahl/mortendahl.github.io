---
layout:     post
title:      "Arithmetic Encodings"
subtitle:   "Secure Operations with Rational Numbers"
date:       2016-09-12 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-04.jpg"
---

<em><strong>TL;DR:</strong> we take a typical CNN deep learning model and go through a series of steps that enable both training and prediction to instead be done on encrypted data.</em> 

https://www1.cs.fau.de/filepool/publications/octavian_securescm/smcint-scn10.pdf

https://www.iacr.org/archive/pkc2007/44500343/44500343.pdf

# Integers

## Signed

```python
def encode_integer(integer):
    element = integer % Q
    return element

def decode_integer(element):
    integer = element if element <= Q//2 else element - Q
    return integer
```

# Rationals

## Fixedpoint

The last step is to provide a mapping between the rational numbers used by the CNNs and the field elements used by the SPDZ protocol. As typically done, we here take a fixed-point approach where rational numbers are scaled by a fixed amount and then rounded off to an integer less than the field size `Q`.

```python
def encode(rational, precision=6):
    upscaled = int(rational * 10**precision)
    field_element = upscaled % Q
    return field_element

def decode(field_element, precision=6):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = upscaled / 10**precision
    return rational
```

In doing this we have to be careful not to "wrap around" by letting any encoding exceed `Q`; if this happens our decoding procedure will give wrong results.

To get around this we'll simply make sure to pick `Q` large enough relative to the chosen precision and maximum magnitude. One place where we have to be careful is when doing multiplications as these double the precision. As done earlier we must hence leave enough room for double precision, and additionally include a truncation step after each multiplication where we bring the precision back down. Unlike earlier though, in the two server setting the truncation step can be performed as a local operation as pointed out in [SecureML](https://eprint.iacr.org/2017/396).

```python
def truncate(x, amount=6):
    y0 = x[0] // 10**amount
    y1 = Q - ((Q - x[1]) // 10**amount)
    return [y0, y1]
```

With this in place we are now (in theory) set to perform any desired computation on encrypted data.


## Floating Point
