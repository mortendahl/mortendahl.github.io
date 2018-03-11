---
layout:     post
title:      "Secure Computations as Dataflow Programs"
subtitle:   "Implementing the SPDZ Protocol using TensorFlow"
date:       2018-03-01 12:00:00
author:     "Morten Dahl"
header-img: "assets/tensorspdz/tensorflow.png"
---

<em><strong>TL;DR:</strong> TODO.</em> 

Why?
- in OpenMined we're making privacy tools available to practisiners

Implementing MPC protocols is non-trivial. Besides getting the underlying cryptography right there are also challenges such as:
- coordinating the simultanuous execution of player programmes 
- making sure data is sent efficiently between them across the network
- cross platform
- streaming triples
- making sure any value is only being masked once
- efficient local computations on available hardware (CPUs or GPUs)
- efficient storage on available hardware
- debugging and verification
- profiling and performance evaluation (incl bottlenecks)
- graph optimisations

Getting all this right can be overwhelming if starting from stratch, which is one reason that earlier blog posts focused on the principles behind MPC and simply did everything locally instead.

Luckily though, modern distributed computation frameworks have existed for a while and are receiving a lot of attention due to their use in modern machine learning: Tensorflow.

The main obstacle here though is that MPC protocols such as SPDZ, as we saw, perform addition, multiplication, and truncation on integer values that do not immediately fit into native words sizes such as 32 bit integers that are currently the largest type supported in Tensorflow and on GPUs if we want to compute dot products. It is possible to overcome this technicality but we will go into that in another blog post.

Tensorflow, being a framework for optimised distributed computations, is in retrospect an obvious candidate for quickly experimenting with MPC protocols.

https://www.tensorflow.org/programmers_guide/graphs
https://en.wikipedia.org/wiki/Dataflow_programming

https://github.com/ppwwyyxx/tensorpack
https://github.com/tensorflow/serving/issues/193
https://github.com/sandtable/ssl_grpc_example


# The SPDZ Protocol

If we look back at the [SPDZ](/2017/09/03/the-spdz-protocol-part1/) [protocol](/2017/09/10/the-spdz-protocol-part2/) and some of the [applications](/2017/09/19/private-image-analysis-with-mpc/) we've covered so far, we see that the typical secure operations we need are addition, subtraction, multiplication, dot products, and truncation.

Needed operations

Recall from earlier posts that the basic operations we needed in the SPDZ protocol for secure computation on fixedpoint values were uniform sampling, modulus reductions, addition, subtraction, multiplication, and truncation of ~120 bit integers. For efficiency we also introduced an explicit dot product.

We will go into detail in another blog post about how these are implemented over 32 bit words.

`crt_add`
`crt_sub`
`crt_mul`
`crt_dot`
`crt_mod`
`sample`

`encode`
`decode`


# Dataflow Programs


# TensorFlow

![](/assets/tensorspdz/structure.png)


# The Basics

Since one secure operation is often expressed in terms of several TensorFlow operations, as a convinient way of defining the computation graph we will use a few abstract operations such as `add` and `dot` that allow us manage this complexity and focus on the secure computation at hand. Since each private value during the computation is represented by two shares, each variable given as input to these abstract operations are also represented by two nodes in the graph. We call each such pair of nodes a private variable, and introduce a simple class to contain this abstraction; this will also come in handy later when we look at a few optimisations.

```python
class PrivateVariable:
    
    def __init__(self, share0, share1):
        self.share0 = share0
        self.share1 = share1
    
    @property
    def shape(self):
        # thanks to TensorFlow, tensor shapes are known at graph creation time,
        # meaning we don't have to do anything ourselves to keep track of this
        return self.share0[0].shape
```

With this in place we can begin to define the graph construction methods for secure operations. The first one is `add`, where the resulting graph simply instructs the servers to locally combine the two shares they each have using a subgraph created by `crt_add`.

```python
def add(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)
    
    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1
    
    with tf.name_scope("add"):
    
        with tf.device(SERVER_0):
            z0 = crt_add(x0, y0)

        with tf.device(SERVER_1):
            z1 = crt_add(x1, y1)

    return PrivateVariable(z0, z1)
```

Pay attention to how easy it is to express which server is doing what with `tf.device()`! Not only is it extremely easy to read, TensorFlow will also automatically insert appropiate networking operations to make sure that the value of e.g. `x0` is moved `SERVER_0` during execution if it was initially defined elsewhere (such as on the input provider)! All of this is abstracted away and the framework could potentially figure out the best strategy for optimising exactly when to perform sends and receives, including batching to better utilise the network and keeping the compute units busy.

**TODO** `tf.device` ties value to device

The `tf.name_scope()` on the other hand is simply a logical abstraction we can use to make the graphs easier to navigate in TensorBoard by essentially naming subgraphs as single components.

![](/assets/tensorspdz/add.png)

With addition in place we turn to dot products. This is significantly more complexity, not least since we now need to involve the crypto producer but also since the two servers have to communication with each other as part of the computation.

```python
def dot(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)

    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1

    with tf.name_scope("dot"):

        # triple generation

        with tf.device(CRYPTO_PRODUCER):
            a = sample(x.shape)
            b = sample(y.shape)
            ab = crt_dot(a, b)
            a0, a1 = share(a)
            b0, b1 = share(b)
            ab0, ab1 = share(ab)
        
        # masking
        
        with tf.device(SERVER_0):
            alpha0 = crt_sub(x0, a0)
            beta0  = crt_sub(y0, b0)

        with tf.device(SERVER_1):
            alpha1 = crt_sub(x1, a1)
            beta1  = crt_sub(y1, b1)

        # recombination after exchanging alphas and betas

        with tf.device(SERVER_0):
            alpha = reconstruct(alpha0, alpha1)
            beta  = reconstruct(beta0, beta1)
            z0 = crt_add(ab0,
                 crt_add(crt_dot(a0, beta),
                 crt_add(crt_dot(alpha, b0),
                         crt_dot(alpha, beta))))

        with tf.device(SERVER_1):
            alpha = reconstruct(alpha0, alpha1)
            beta  = reconstruct(beta0, beta1)
            z1 = crt_add(ab1,
                 crt_add(crt_dot(a1, beta),
                         crt_dot(alpha, b1)))
        
    z = PrivateVariable(z0, z1)
    return truncate(z)
```

However, we see that with `tf.device()` this is still relatively straight-forward, at least if the [secure protocol](/2017/09/03/the-spdz-protocol-part1/#multiplication) is already understood. We first construct a graph that makes the crypto producer generate a new dot triple. The output nodes of this graph is `a0, a1, b0, b1, ab0, ab1`

With `crt_sub` we then build graphs for the two servers that masks `x` and `y`. Note that TensorFlow will here take care of inserting networking code that sends the value of e.g. `a0` to `SERVER_0` during execution.

In the third step we then reconstruct `alpha` and `beta` on each server, and compute the recombination step to get the dot product. Note that we have to define `alpha` and `beta` twice, one for each server; although they contain the same value, if we had instead define them only on one server but used them on both, then we would implicitly have inserted additional networking operations and hence slowed down the computation.

Returning to TensorBoard we can verify that the nodes are indeed tied to the correct players, with yellow being the crypto producer, and green and turquoise being the two servers. Note the convinience of having `tf.name_scope()` here.

![](/assets/tensorspdz/dot.png)

To fully claim that this has made the distributed aspects of secure computations much easier to express, we also have to see what is actually needed for `td.device()` to work as intended. In the code below we first define an arbitrary job name followed by identifiers for our five players. More interestingly, we then simply specify the network hosts of the five players and wrap this together in a `ClusterSpec`. That's it!

```python
JOB_NAME = 'spdz'

SERVER_0        = '/job:{}/task:0'.format(JOB_NAME)
SERVER_1        = '/job:{}/task:1'.format(JOB_NAME)
CRYPTO_PRODUCER = '/job:{}/task:2'.format(JOB_NAME)
INPUT_PROVIDER  = '/job:{}/task:3'.format(JOB_NAME)
OUTPUT_RECEIVER = '/job:{}/task:4'.format(JOB_NAME)

HOSTS = [
    'localhost:4440',
    'localhost:4441',
    'localhost:4442',
    'localhost:4443',
    'localhost:4444'
]

CLUSTER = tf.train.ClusterSpec({
    JOB_NAME: HOSTS
})
```

Finally, the code that each player executes is about as simple as it gets.

```python
ROLE = 0

server = tf.train.Server(CLUSTER, job_name=JOB_NAME, task_index=ROLE)
server.start()
server.join()
```

Here the value of `ROLE` is the only thing that differs between the programs the five players run and typically given as a command-line argument.


# Profiling

**TODO** compute time in TensorBoard

![](/assets/tensorspdz/dot-computetime.png)

**TODO** tracing


# Optimisations

**TODO** `tf.staging`
**TODO** XLA


## Reusing Masked Values

In [previous](/2017/09/10/the-spdz-protocol-part2/) [posts](/2017/09/19/private-image-analysis-with-mpc/#generalised-triples) it was already mentioned that we'd ideally want to only mask every private value once to primarily save on networking. For instance, if we are computing both `dot(w, x)` and `dot(w, y)` then we want to use the same masked version of `w` in both. If we are for instance doing predictions with weights `w` then  ... **TODO** Turns out it's very straight forward to implement this in our setting.

For starters, the dataflow nature of TensorFlow will take care of only recomputing the parts of the graph that actually change when the input values are changed. For instance, an application that computes `dot(w, x)` will reuse the value of `w` between executions with different values of `x` changes (unless of course the value of `w` somehow depends on the value of `x` elsewhere).

We can do better though, since this will not help us in an application that computes both `dot(w, x)` and `dot(w, y)` in each execution.


We first extend `PrivateVariable` to hold a pointer to the previously masked version of the variable. When present, this field will point to several nodes in the graph as seen below.

```python
class PrivateVariable:
  
    x.masked = None
    
    ...
```

We next make masking an explicit operation separate from the operations that need it, allowing us to use the same masked version across all of them. Note that if a private variable has already been masked then we will now reuse that.

```python
def mask(x):
    assert isinstance(x, PrivateVariable)
    
    if x.masked is None:
      
        x0, x1 = x.share0, x.share1
        shape = x.shape
      
        with tf.name_scope("mask"):

            with tf.device(CRYPTO_PRODUCER):
                a = sample(shape)
                a0, a1 = share(a)

            with tf.device(SERVER_0):
                alpha0 = crt_sub(x0, a0)

            with tf.device(SERVER_1):
                alpha1 = crt_sub(x1, a1)

            # exchange of alphas

            with tf.device(SERVER_0):
                alpha = reconstruct(alpha0, alpha1)
                alpha_on_0 = alpha

            with tf.device(SERVER_1):
                alpha = reconstruct(alpha0, alpha1)
                alpha_on_1 = alpha

        x.masked = (a, a0, a1, alpha_on_0, alpha_on_1)
        
    return x.masked
```

Finally, `dot` may now be rewritten as below, where we additionally also cache the output node so that a secure computation such as `dot(x, y) + dot(x, y)` will reuses that node in the graph and hence only compute the three remaining steps of `dot(x, y)` once.

```python
def dot(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)

    cache_key = ('dot', x, y)
    z = cached_results.get(cache_key, None)

    if z is None:

        # make sure we have masked version of both inputs
        a, a0, a1, alpha_on_0, alpha_on_1 = mask(x)
        b, b0, b1,  beta_on_0,  beta_on_1 = mask(y)

        with tf.name_scope("dot"):

            with tf.device(CRYPTO_PRODUCER):
                ab = crt_dot(a, b)
                ab0, ab1 = share(ab)

            with tf.device(SERVER_0):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = crt_add(ab0,
                     crt_add(crt_dot(a0, beta),
                     crt_add(crt_dot(alpha, b0),
                             crt_dot(alpha, beta))))

            with tf.device(SERVER_1):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = crt_add(ab1,
                     crt_add(crt_dot(a1, beta),
                             crt_dot(alpha, b1)))

        z = PrivateVariable(z0, z1)
        z = truncate(z)
        cached_results[cache_key] = z

    return z
```

![](/assets/tensorspdz/masking-reuse.png)

# Experiments

## Setup

GCP