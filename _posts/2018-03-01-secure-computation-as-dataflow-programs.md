---
layout:     post
title:      "Secure Computations as Dataflow Programs"
subtitle:   "Implementing the SPDZ Protocol using TensorFlow"
date:       2018-03-01 12:00:00
author:     "Morten Dahl"
header-img: "assets/tensorspdz/tensorflow.png"
---

<em><strong>This post is a work in progress.</strong></em> 

<em><strong>TL;DR:</strong> using TensorFlow as a distributed computation framework for dataflow programs we give a full implementation of the SPDZ protocol with networking.</em> 

Unlike earlier [blog](/2017/09/03/the-spdz-protocol-part1/) [posts](/2017/09/10/the-spdz-protocol-part2/) where we focused on the higher-level concepts behind the SPDZ protocol and its [potential applications](/2017/09/19/private-image-analysis-with-mpc/), here we build a fully working (passively secure) implementation with players running on different machines and communicating via typical network stacks. As part of this, we also seek to investigate the benefits of using a [modern distributed computation](https://en.wikipedia.org/wiki/Dataflow_programming) platform when experimenting with secure computations, as opposed to building everything from scratch.

Moreover, this can also be seen as a step in the direction of getting private machine learning into the hands of practitioners, where integration with existing and popular tools such as [TensorFlow](https://www.tensorflow.org/) plays an important part. Concretely, while we here only do a relatively shallow integration that doesn't make use of powerful machine learning tools built into TensorFlow such as automatic gradient computations, we do show how some of the basic technical obstacles can be overcome, potentially paving the way for deeper integrations.

<em>A big thank you goes out to [Andrew Trask](https://twitter.com/iamtrask) and several members of the [OpenMined community](https://twitter.com/openminedorg) for inspiration and interesting discussions on this topic!</em>

<em><strong>Disclaimer</strong>: this implementation is meant for experimentation only and may not live up to required security. In particular, TensorFlow does not currently seem to have been designed with this application in mind, and although it does not appear to be the case right now, may for instance in future versions perform optimisations behind that scene that breaks the intended security properties.</em>


# Motivation

Due to their distributed nature, implementing secure computation protocols such as SPDZ can be a non-trivial task, and optimising them of course only adds to the complexity. 

Basic objectives might for instance include how to coordinate the simultanuous execution of several players and making sure data is sent efficiently across the network asynchronously and in batches, useful objectives such as supporting different hardware platforms including for instance both CPUs and GPUs, and conveniences such as tools for inspective, debugging, and profiling.

In particular, buffering and streaming of data is relevant in secure computations.

Implementing MPC protocols is non-trivial. Besides getting the underlying cryptography right there are also challenges such as:
- buffering/streaming triples
- making sure any value is only being masked once
- efficient storage on available hardware
- debugging and verification
- profiling and performance evaluation (incl bottlenecks)
- graph optimisations

Getting all this right can be overwhelming, which is one reason that earlier blog posts focused on the principles behind MPC and simply did everything locally instead.

Luckily though, modern distributed computation frameworks have existed for a while and are receiving a lot of attention due to their use in modern machine learning: Tensorflow.

The main obstacle here though is that MPC protocols such as SPDZ, as we saw, perform addition, multiplication, and truncation on integer values that do not immediately fit into native words sizes such as 32 bit integers that are currently the largest type supported in Tensorflow and on GPUs if we want to compute dot products. It is possible to overcome this technicality but we will go into that in another blog post.

Tensorflow, being a framework for optimised distributed computations and focused on machine learning, is in retrospect an obvious candidate for quickly experimenting with MPC protocols for private machine learning.


## The SPDZ Protocol

Looking back at the [SPDZ](/2017/09/03/the-spdz-protocol-part1/) [protocol](/2017/09/10/the-spdz-protocol-part2/) and some of the [applications](/2017/09/19/private-image-analysis-with-mpc/) covered so far, we see that the typical secure operations needed were tensor addition, subtraction, multiplication, dot products, and truncation. We also saw how these can easily be implemented using modular arithmetic, yet required working with numbers larger than what normally fits into a single machine word. Concretely, we often ended up working on ~120 bit integers using Python's built-in arbitrary precision objects.

Unfortunately such objects are not currently supported in most frameworks, including TensorFlow, which today for instance only supports dot products between tensors of 32 bit integers (a constraint that may have something to do with current GPUs, although see for instance **TODO**). Nonetheless, since we only need the operations mentioned above there are efficient ways around this as discussed [elsewhere](TODO).

In summary, below we will simply assume base operations `base_add`, `base_sub`, `base_mul`, `base_dot`, `base_mod`, and `sample` that allow us to simulate arithmetic on tensors of ~120 bit values using (a list of) tensors of 32 bit integers.


## TensorFlow

![](/assets/tensorspdz/structure.png)


# Basic Implementation

Since a secure operation is often expressed in terms of several TensorFlow operations, as a convinient way of defining the computation graph we will use a few abstract operations such as `add` and `dot` that allow us manage this complexity and focus on the secure computation at hand. Since each private value during the computation is represented by two shares, each variable given as input to these abstract operations are also represented by two nodes in the graph. We call each such pair of nodes a private tensor, and introduce a simple class to contain this abstraction; this will also come in handy later when we look at a few optimisations.

```python
class PrivateTensor:
    
    def __init__(self, share0, share1):
        self.share0 = share0
        self.share1 = share1
    
    @property
    def shape(self):
        # thanks to TensorFlow, tensor shapes are known at graph creation time,
        # meaning we don't have to do anything ourselves to keep track of this
        return self.share0[0].shape
```

With this in place we can begin to define the graph construction methods for secure operations. The first one is `add`, where the resulting graph simply instructs the servers to locally combine the two shares they each have using a subgraph created by `base_add`.

```python
def add(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)
    
    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1
    
    with tf.name_scope("add"):
    
        with tf.device(SERVER_0):
            z0 = base_add(x0, y0)

        with tf.device(SERVER_1):
            z1 = base_add(x1, y1)

    z = PrivateTensor(z0, z1)
    return z
```

Notice how easy it is to use [`tf.device()`](https://www.tensorflow.org/api_docs/python/tf/device) to express which server is doing what! This command ties the computation and its resulting value to the specified host, and instructs TensorFlow to automatically insert appropiate networking operations to make sure that the input values are available when needed! For instance, in the above, if `x0` was previous on, say, the input provider then TensorFlow will insert send and receive instructions that copies it to `SERVER_0` as part of computing `add`. All of this is abstracted away and the framework could potentially figure out the best strategy for optimising exactly when to perform sends and receives, including batching to better utilise the network and keeping the compute units busy.

The [`tf.name_scope()`](https://www.tensorflow.org/api_docs/python/tf/name_scope) command on the other hand is simply a logical abstraction that doesn't influence computations but can be used to make the graphs easier to navigate in TensorBoard by grouping subgraphs as single components.

![](/assets/tensorspdz/add.png)

With addition in place we turn to dot products. This is significantly more complexity, not least since we now need to involve the crypto producer, but also since the two servers have to communicate with each other as part of the computation.

```python
def dot(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)

    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1

    with tf.name_scope("dot"):

        # triple generation

        with tf.device(CRYPTO_PRODUCER):
            a = sample(x.shape)
            b = sample(y.shape)
            ab = base_dot(a, b)
            a0, a1 = share(a)
            b0, b1 = share(b)
            ab0, ab1 = share(ab)
        
        # masking after distributing the triple
        
        with tf.device(SERVER_0):
            alpha0 = base_sub(x0, a0)
            beta0  = base_sub(y0, b0)

        with tf.device(SERVER_1):
            alpha1 = base_sub(x1, a1)
            beta1  = base_sub(y1, b1)

        # recombination after exchanging alphas and betas

        with tf.device(SERVER_0):
            alpha = reconstruct(alpha0, alpha1)
            beta  = reconstruct(beta0, beta1)
            z0 = base_add(ab0,
                 base_add(base_dot(a0, beta),
                 base_add(base_dot(alpha, b0),
                          base_dot(alpha, beta))))

        with tf.device(SERVER_1):
            alpha = reconstruct(alpha0, alpha1)
            beta  = reconstruct(beta0, beta1)
            z1 = base_add(ab1,
                 base_add(base_dot(a1, beta),
                          base_dot(alpha, b1)))
        
    z = PrivateTensor(z0, z1)
    z = truncate(z)
    return z
```

However, with `tf.device()` we see that this is still relatively straight-forward, at least if the [protocol for secure dot products](/2017/09/03/the-spdz-protocol-part1/#multiplication) is already understood. We first construct a graph that makes the crypto producer generate a new dot triple. The output nodes of this graph is `a0, a1, b0, b1, ab0, ab1`

With `base_sub` we then build graphs for the two servers that mask `x` and `y` using `a` and `b` respectively. TensorFlow will again take care of inserting networking code that sends the value of e.g. `a0` to `SERVER_0` during execution.

In the third step we then reconstruct `alpha` and `beta` on each server, and compute the recombination step to get the dot product. Note that we have to define `alpha` and `beta` twice, one for each server, since although they contain the same value, if we had instead define them only on one server but used them on both, then we would implicitly have inserted additional networking operations and hence slowed down the computation.

Returning to TensorBoard we can verify that the nodes are indeed tied to the correct players, with yellow being the crypto producer, and green and turquoise being the two servers. Note the convenience of having `tf.name_scope()` here.

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
server = tf.train.Server(CLUSTER, job_name=JOB_NAME, task_index=ROLE)
server.start()
server.join()
```

Here the value of `ROLE` is the only thing that differs between the programs the five players run and typically given as a command-line argument.


# Optimisations

With the basics in place we can look at a few optimisations.


## Tracking intermediate results

Our first improvement allows us to reuse computations. For instance, if our goal is to compute `dot(x, y) + dot(x, y)` then we want to avoid computing `dot(x, y)` twice. Concretely, we want to keep track of intermediate nodes in the graph and reuse them when possible.

To do this we will maintain a glocal dictionary of `PrivateTensor` references as we build the graph, and use this for looking up already existing results before adding new nodes. For instance, `dot` now becomes as follows.

```python
def dot(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)

    node_key = ('dot', x, y)
    z = nodes.get(node_key, None)

    if z is None:

        # ... as before ...

        z = PrivateTensor(z0, z1)
        z = truncate(z)
        nodes[node_key] = z

    return z
``` 

While already significant, this change also opens up for our next improvement.


## Reusing masked tensors

In [previous](/2017/09/10/the-spdz-protocol-part2/) [posts](/2017/09/19/private-image-analysis-with-mpc/#generalised-triples) we already mentioned that we'd ideally want to only mask every private tensor once to primarily save on networking. For instance, if we are computing both `dot(w, x)` and `dot(w, y)` then we want to use the same masked version of `w` in both: if we are e.g. doing multiple predictions with the same weights `w` then in the long term the initial cost of masking `w` can be amortised away.

To implement this we first note that the dataflow nature of TensorFlow will take care of only recomputing the parts of the graph that actually change when the input values are changed. For instance, an application that computes `dot(w, x)` will reuse the value of `w` between executions with different values of `x` changes (unless of course the value of `w` somehow depends on the value of `x` elsewhere). However, `w` will currently be masked each time we compute `dot`.

To implement this we make masking an explicit operation, which allows us to use the same masked version across all uses.

```python
def mask(x):
    assert isinstance(x, PrivateTensor)
    
    node_key = ('mask', x)
    masked = nodes.get(node_key, None)
    
    if masked is None:
      
        x0, x1 = x.share0, x.share1
        shape = x.shape
      
        with tf.name_scope("mask"):

            with tf.device(CRYPTO_PRODUCER):
                a = sample(shape)
                a0, a1 = share(a)

            with tf.device(SERVER_0):
                alpha0 = base_sub(x0, a0)

            with tf.device(SERVER_1):
                alpha1 = base_sub(x1, a1)

            # exchange of alphas

            with tf.device(SERVER_0):
                alpha = reconstruct(alpha0, alpha1)
                alpha_on_0 = alpha

            with tf.device(SERVER_1):
                alpha = reconstruct(alpha0, alpha1)
                alpha_on_1 = alpha

        masked = (a, a0, a1, alpha_on_0, alpha_on_1)
        nodes[node_key] = masked
        
    return masked
```

With this `dot` may be rewritten as below, which is now only responsible for the recombination step.

```python
def dot(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)

    node_key = ('dot', x, y)
    z = nodes.get(node_key, None)

    if z is None:

        # make sure we have masked version of both inputs
        a, a0, a1, alpha_on_0, alpha_on_1 = mask(x)
        b, b0, b1,  beta_on_0,  beta_on_1 = mask(y)

        with tf.name_scope("dot"):

            with tf.device(CRYPTO_PRODUCER):
                ab = base_dot(a, b)
                ab0, ab1 = share(ab)

            with tf.device(SERVER_0):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = base_add(ab0,
                     base_add(base_dot(a0, beta),
                     base_add(base_dot(alpha, b0),
                              base_dot(alpha, beta))))

            with tf.device(SERVER_1):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = base_add(ab1,
                     base_add(base_dot(a1, beta),
                              base_dot(alpha, b1)))

        z = PrivateTensor(z0, z1)
        z = truncate(z)
        nodes[node_key] = z

    return z
```

As a verification, TensorBoard shows us that the graph structure is as expected (with some blocks still showing up white).

![](/assets/tensorspdz/masking-reuse.png)

Note here that the leftmost masking block, correspoding to `mask(w)`, is feeding into both dot blocks as intended.


## Precomputing triples

So far the crypto producer is creating triples online in synchronisation with the computation of the two servers. However, a main point of triples is to enable parts of the computation to be done in an offline phase.

<!--In principle this is entirely straight forward using e.g. `tf.FIFOQueue`, yet there is one obstacle we have to deal with.-->

<em>(Coming soon...)</em>


# Profiling

As a final reason to be excited about building dataflow programs in TensorFlow, we also look at the built-in profiling services. In particular, in TensorBoard we can easily see how expensive each operation was both in terms of compute and memory.

![](/assets/tensorspdz/computetime.png)

The heatmap in the above shows that masking and dot products are the most expensive operations we ran, and by clicking on e.g. the rightmost dot product we see that it took roughly 50ms to execute (using the [localhost configuration](TODO)). Moreover, below we navigate further into the dot block and see that sharing was probably the most expensive sub-operations, in this particular run taking 36ms.

![](/assets/tensorspdz/computetime-detailed.png)

But we can go even further and get detailed tracing information. Below we see all operations performed by our five players, with the two servers first, followed by the crypto producer, the input provider, and the output receiver. Clicking on one of the operations additionally allows us to pinpoint exactly where in the execution we were as shown in the bottom panel.

![](/assets/tensorspdz/tracing.png)
 
This also shows that TensorFlow automatically figured out to use six concurrent threads to run the operations of the first three players, and that the two servers began processing before the crypto producer had completed; in other words, it looks like it figured out how to perform streaming of the computation.


# Running the Code

<em>(Coming soon; see the [GitHub repo](https://github.com/mortendahl/privateml/tree/master/tensorflow/spdz/simple-dense) for now)</em>

<!--
# Dump

- https://learningtensorflow.com/lesson11/
- [XLA](https://www.tensorflow.org/performance/xla/)


https://www.tensorflow.org/programmers_guide/graphs
https://en.wikipedia.org/wiki/Dataflow_programming

https://github.com/ppwwyyxx/tensorpack
https://github.com/tensorflow/serving/issues/193
https://github.com/sandtable/ssl_grpc_example

[TensorBoard](https://www.tensorflow.org/programmers_guide/graph_viz)

-->