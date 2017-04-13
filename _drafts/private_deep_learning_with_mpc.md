---
layout:     post
title:      "Private Deep Learning with MPC"
subtitle:   "A Simple Tutorial from Scratch"
date:       2017-04-02 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---

Inspired by a recent blog post about mixing deep learning and homomorphic encryption (see [*Building Safe A.I.*](http://iamtrask.github.io/2017/03/17/safe-ai/)) I thought it'd be interesting do to the same, but replacing homomorphic encryption with secure multi-party computation.

Below we'll build a simple secure computation protocol from scratch, and experiment with it for learning boolean functions using basic neural networks. As a result, we need to be able to securely compute on rational numbers with a certain precision, in particular to be able to add and multiply these.

One challenge is how to compute the Sigmoid function `1/(1+np.exp(-x))`, which in its traditional form results in surprisingly heavy operations in the secure setting. As a result we'll follow the approach of *Building Safe A.I.* and approximate it using polynomials, yet look at a few optimizations.


# Secure Multi-Party Computation

Homomorphic encryption (HE) and secure multi-party computation (MPC) are closely related fields in modern cryptography, with one often using techniques from the other in order to solve roughly the same problem: computing a function of some input data privately, i.e. without revealing any of the inputs. As such, one is often replaceable by the other.

Where they differ however, can roughly be characterized by HE using heavy computation and little interaction, whereas MPC instead uses light computation and significant interaction. As such, by letting a few parties be part of the computation process instead of only one, MPC currently offers significantly better practical performance, to the point where one can argue that it's a significantly more mature technology. For instance, [several](https://sepior.com/) [companies](https://www.dyadicsec.com/) [already](https://sharemind.cyber.ee/) [exist](https://z.cash/technology/paramgen.html) offering services based on MPC.

TODO

Assume we have three parties `P0`, `P1`, and `P2` with inputs `x0`, `x1`, and `x2` respectively, and that they would like to compute some function `f` on these inputs: `y = f(x1, x2, x3)`. Here, the inputs may for instance be training sets and `f` a machine learning process that maps these to a trained model.

Moreover, assume that party `Pi` would like to keep `xi` private from everyone else, and is only interested in sharing the final result `y` (and implicitly what can be learned about `xi` from `y`).

This is the problem solved by MPC, and notice straight away the focus on inputs coming from several parties. In particular, MPC is naturally about mixing data from several parties, such as training data from several users.

To start the tutorial, let's build a basic protocol that allows the three parties to start computing on their inputs.


But let's move on to something concrete.


## Fixed-point arithmetic

The computation is going to take place over a [field](https://en.wikipedia.org/wiki/Finite_field) and hence we first need to decide on how to represent rational numbers `r` as field elements, i.e. as integers `x` from `0, 1, ..., q-1` for some prime `q`. Taking a typical approach, we're going to scale a rational number by a fixed constant, say `10**6`, and let the integer part of the result be our fixed-point presentation; in this case with a fractional precision of `6` digits. For instance, with `Q = 10000019` we get `encode(0.5) == 500000` and `encode(-0.5) == 10000019 - 500000 == 9500019`.

```python
def encode(rational):
    upscaled = int(rational * 10**6)
    field_element = upscaled % Q
    return field_element

def decode(field_element):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = upscaled / 10**6
    return rational
```

Note that addition in this representation is straight-forward, `(r * 10**6) + (s * 10**6) == (r + s) * 10**6`, while multiplication adds an extra scaling factor that we will have to get rid of to keep the precision and avoid exploding the numbers: `(r * 10**6) * (s * 10**6) == (r * s) * 10**6 * 10**6`.


## Sharing and adding data

Having encoded the inputs, we next need a secure way of sharing these with the other parties so we can use them in the computation, yet still keep them private. 

The ingredient we need for this is [*secret sharing*](), which splits each value into three shares in such a way that if anyone sees less than the three shares, then nothing at all is revealed about the value; yet, by seeing all three shares, the input can easily be reconstructed. 

To keep it simple we'll use *replicated secret sharing* here, where each party receives more than one share. Concretely, private value `x` is split into three shares `x0`, `x1`, and `x2`, with party `P0` receiving (`x0`, `x1`), `P1` receiving (`x1`, `x2`), and `P2` receiving (`x2`, `x0`). For this tutorial we'll keep this implicit though, and simply store a sharing of `x` as a vector of the three shares `[x0, x1, x2]`.

```python
def share(x):
    x0 = random.randrange(Q)
    x1 = random.randrange(Q)
    x2 = (x - x0 - x1) % Q
    return [x0, x1, x2]

def reconstruct(sharing):
    return sum(sharing) % Q
```

With this we already have a way to do secure addition and subtraction: each party simply adds or subtracts its two shares. This works works since e.g. `(x0 + x1 + x2) + (y0 + y1 + y2) == (x0 + y0) + (x1 + y1) + (x2 + y2)`, which gives the three new shares of `x + y`.

```python
def add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]

def sub(x, y):
    return [ (xi - yi) % Q for xi, yi in zip(x, y) ]
```

 And note that no communication is needed since these are local computations.


## Multiplication

Since each party has two shares, multiplication can be done in similar way to addition and subtraction above, i.e. by each party computing a new share based on the two it already has. Specifically, for `z0`, `z1`, and `z2` as defined in the code below we have `x * y == z0 + z1 + z2`. 

However, our invariant of each party having two shares is not satisfied, and it wouldn't be secure for e.g. `P1` simply to send `z1` to `P0`. One easy fix is to simply share each `zi` as if it was a private input, and then have each party add the shares it receive together; this gives a correct and secure sharing `w` of `x * y`.

```python
def mul(x, y):
    # local computation
    z0 = (x[0]*y[0] + x[0]*y[1] + x[1]*y[0]) % Q
    z1 = (x[1]*y[1] + x[1]*y[2] + x[2]*y[1]) % Q
    z2 = (x[2]*y[2] + x[2]*y[0] + x[0]*y[2]) % Q
    # reshare and distribute
    Z = [ share(z0), share(z1), share(z2) ]
    w = [ sum(row) % Q for row in zip(*Z) ]
    # bring precision back down from double to single
    v = truncate(w)
    return v
```

One problem remains however, and as mentioned earlier this is the double precision of `w`: it is an encoding with scaling factor `10**6 * 10**6` instead of `10**6`. Dividing `w` by `10**6`, through multiplication by its inverse `10**(-6)`, fixes this, as long as there is no remainder. Specifically, we may write `w == v * 10**6 + u` where `u < 10**6`, so that after the division we have `v + u * 10**(-6)` in general, and the result we are after when `u == 0`. So, if we knew `u` in advance then by doing the division on `w' == (w - u)` instead we'd get `v' == v` and `u' == 0` as desired.

The question of course is how to securely get `u` so we may compute `w'`. The details are in [CS'10](https://www1.cs.fau.de/filepool/publications/octavian_securescm/secfp-fc10.pdf) but the basic idea is to first add a large mask to `w`, reveal this masked value to one of the parties who may then compute a masked `u`. Finally, this masked value is then reshared and unmasked, and used to compute `w'`.

```python
def truncate(a):
    # map to the positive range
    b = add(a, share(10**(6+6-1)))
    # apply mask known only by P0, and reconstruct masked b to P1 or P2
    mask = random.randrange(Q) % 10**(6+6+KAPPA)
    mask_low = mask % 10**6
    b_masked = reconstruct(add(b, share(mask)))
    # extract lower digits
    b_masked_low = b_masked % 10**6
    b_low = sub(share(b_masked_low), share(mask_low))
    # remove lower digits
    c = sub(a, b_low)
    # division
    d = imul(c, INVERSE)
    return d
```

Note that `imul` in the above is a local operation that multiplies each share with a public integer, is this case the inverse of `10**6`.

## Secure data type

As a final step we wrap the above procedures in a custom abstract data type, allowing us to use NumPy later when we express [the neural network](http://iamtrask.github.io/2015/07/12/basic-python-network/).

```python
class SecureRational(object):
    
    def __init__(self, secret=None):
        self.shares = share(encode(secret)) if secret is not None else []
    
    def reveal(self):
        return decode(reconstruct(self.shares))
    
    def __repr__(self):
        return "SecureRational(%f)" % self.reveal()
    
    def __add__(x, y):
        z = SecureRational()
        z.shares = add(x.shares, y.shares)
        return z
    
    def __sub__(x, y):
        z = SecureRational()
        z.shares = sub(x.shares, y.shares)
        return z
    
    def __mul__(x, y):
        z = SecureRational()
        z.shares = mul(x.shares, y.shares)
        return z
    
    def __pow__(x, e):
        z = SecureRational(1)
        for _ in range(e):
            z = z * x
        return z
```

With this type we can operate securely on values as we would any other type:

```python
x = SecureRational(.5)
y = SecureRational(-.25)
z = x * y
assert(z.reveal() == (.5) * (-.25))
```

Moreover, for debugging purposes we could switch to an insecure type without changing the rest of the (neural network) code, or we could isolated the use of counters to for instance see how many multiplications are performed, in turn allowing us to simulate how much communication is needed.


# Deep Learning

The term "deep learning" is an exaggeration of what we'll be doing here, since we'll simply play with the two and three layer neural networks from *Building Safe A.I.* to learn basic boolean functions.

One critical part of this is computing the [Sigmoid function](http://mathworld.wolfram.com/SigmoidFunction.html), which we'll first do via the standard Taylor expansion and later via custom polynomial interpolation. 


## A simple network

```python
# reseed to get reproducible results
random.seed(1)
np.random.seed(1)
    
# define Sigmoid approximation function
ONE = SecureRational(1)
W0 = SecureRational(1/2)
W1 = SecureRational(1/4)
W3 = SecureRational(1/48)
W5 = SecureRational(1/480)

def scalar_sigmoid(x):
    return W0 + (x * W1) - (x**3 * W3) + (x**5 * W5)

def scalar_sigmoid_deriv(x):
    return (ONE - x) * x

sigmoid = np.vectorize(scalar_sigmoid)
sigmoid_deriv = np.vectorize(scalar_sigmoid_deriv)

# helper function to map array of numbers to secure data type
secure = np.vectorize(lambda x: SecureRational(x))

# training inputs
X = secure(np.array([
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ]))

y = secure(np.array([[
            0,
            0,
            1,
            1
        ]]).T)

# initial weights 
synapse0 = secure(2 * np.random.random((3,1)) - 1)

# training
for i in range(1000):
    
    # forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, synapse0))
    
    # back propagation
    layer1_error = y - layer1
    layer1_delta = layer1_error * sigmoid_deriv(layer1)
    
    # update
    synapse0 += np.dot(layer0.T, layer1_delta)

# result
print("The learned weights are: \n %s" % synapse0)
print("Prediction on the training data is: \n %s" % layer1)
```

```python
The learned weights are: 
 [[SecureRational(4.936443)]
 [SecureRational(-0.010268)]
 [SecureRational(-2.460124)]]
Prediction on the training data is: 
 [[SecureRational(0.007459)]
 [SecureRational(0.004840)]
 [SecureRational(0.996689)]
 [SecureRational(0.994049)]]
```


## Approximating sigmoid

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_approx(x):
    return (.5) + (x * 1/4) - (x**3 * 1/48) + (x**5 * 1/480)
```
 
```python
x = -0.5
assert(0.00001 > abs(sigmoid(x) - sigmoid_approx(x)))

x = -1.0
assert(0.001 > abs(sigmoid(x) - sigmoid_approx(x)))

x = -2.5
assert(0.1 > abs(sigmoid(x) - sigmoid_approx(x)))
```

As the values become larger, the approximation becomes more inaccurate. Adding more terms from the Taylor expansion mitigates this, but doesn't scale since it quickly becomes inaccurate again. As such, when the magnitude of the synapsis weights become too large (the network becomes too confident) the approximation breaks down. This puts a limitation on the number of iterations we can perform with this approach.


## ...

Thank you to ...