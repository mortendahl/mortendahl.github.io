---
layout:     post
title:      "Secure Computations as Dataflow Programs"
subtitle:   "Implementing the SPDZ Protocol using TensorFlow"
date:       2018-03-01 12:00:00
author:     "Morten Dahl"
header-img: "assets/tensorspdz/tensorflow.png"
---

<em><strong>TL;DR:</strong> using TensorFlow as a distributed computation framework for dataflow programs we give a full implementation of the SPDZ protocol with networking, in turn enabling optimised machine learning on encrypted data.</em> 

Unlike [earlier](/2017/09/03/the-spdz-protocol-part1/) where we focused on the concepts behind secure computation as well as [potential applications](/2017/09/19/private-image-analysis-with-mpc/), here we build a fully working (passively secure) implementation with players running on different machines and communicating via typical network stacks. And as part of this we investigate the benefits of using a [modern distributed computation](https://en.wikipedia.org/wiki/Dataflow_programming) platform when experimenting with secure computations, as opposed to building everything from scratch.

Additionally, this can also be seen as a step in the direction of getting private machine learning into the hands of practitioners, where integration with existing and popular tools such as [TensorFlow](https://www.tensorflow.org/) plays an important part. Concretely, while we here only do a relatively shallow integration that doesn't make use of some of the powerful tools that comes with TensorFlow (e.g. [automatic differentiation](https://www.tensorflow.org/api_docs/python/tf/gradients)), we do show how basic technical obstacles can be overcome, potentially paving the way for deeper integrations.

Jumping ahead, it is clear in retrospect that TensorFlow is an obvious candidate framework for quickly experimenting with secure computation protocols, at the very least in the context of private machine learning.

[All code](https://github.com/mortendahl/privateml/tree/master/tensorflow/spdz/) is available to play with, either locally or on the [Google Cloud](https://cloud.google.com/compute/). To keep it simple our running example throughout is private prediction using [logistic](https://beckernick.github.io/logistic-regression-from-scratch/) [regression](https://github.com/ageron/handson-ml/blob/master/04_training_linear_models.ipynb), meaning that given a private (i.e. encrypted) input `x` we wish to securely compute `sigmoid(dot(w, x) + b)` for private but pre-trained weights `w` and bias `b` (private training of `w` and `b` is considered in a follow-up post). [Experiments](#experiments) show that for a model with 100 features this can be done in TensorFlow with a latency as low as 60ms and at a rate of up to 20,000 prediction per second.

<em>A big thank you goes out to [Andrew Trask](https://twitter.com/iamtrask), [Kory Mathewson](https://twitter.com/korymath), [Jan Leike](https://twitter.com/janleike), and the [OpenMined community](https://twitter.com/openminedorg) for inspiration and interesting discussions on this topic!</em>

<em><strong>Disclaimer</strong>: this implementation is meant for experimentation only and may not live up to required security. In particular, TensorFlow does not currently seem to have been designed with this application in mind, and although it does not appear to be the case right now, may for instance in future versions perform optimisations behind that scene that break the intended security properties. [More notes below](#thoughts).</em>


# Motivation

As hinted above, implementing secure computation protocols such as [SPDZ](/2017/09/03/the-spdz-protocol-part1/) is a non-trivial task due to their distributed nature, which is only made worse when we start to introduce various optimisations ([but](https://github.com/rdragos/awesome-mpc) [it](https://github.com/bristolcrypto/SPDZ-2) [can](https://github.com/aicis/fresco) [be](http://oblivc.org/) [done](https://github.com/encryptogroup/ABY)). For instance, one has to consider how to best orchestrate the simultanuous execution of multiple programs, how to minimise the overhead of sending data across the network, and how to efficient interleave it with computation so that one server only rarely waits on the other. On top of that, we might also want to support different hardware platforms, including for instance both CPUs and GPUs, and for any serious work it is highly valuable to have tools for visual inspection, debugging, and profiling in order to identify issues and bottlenecks.

It should furthermore also be easy to experiment with various optimisations, such as transforming the computation for improved performance, [reusing intermediate results and masked values](/2017/09/19/private-image-analysis-with-mpc/#generalised-triples), and supplying fresh "raw material" in the form of [triples](/2017/09/03/the-spdz-protocol-part1/#multiplication) during the execution instead of only generating a large batch ahead of time in an offline phase. Getting all this right can be overwhelming, which is one reason earlier blog posts here focused on the principles behind secure computation protocols and simply did everything locally. 

Luckily though, modern distributed computation frameworks such as [TensorFlow](https://www.tensorflow.org/) are receiving a lot of research and engineering attention these days due to their use in advanced machine learning on large data sets. And since our focus is on private machine learning there is a natural large fundamental overlap. In particular, the secure operations we are interested in are tensor addition, subtraction, multiplication, dot products, truncation, and sampling, which all have insecure but highly optimised counterparts in TensorFlow.


## Prerequisites

We make the assumption that the main principles behind both TensorFlow and the SPDZ protocol are already understood -- if not then there are [plenty](https://www.tensorflow.org/tutorials/) [of](https://learningtensorflow.com/) [good](https://github.com/ageron/handson-ml) [resources](https://developers.google.com/machine-learning/crash-course/) for the former (including [whitepapers](https://www.tensorflow.org/about/bib)) and e.g. [previous](/2017/09/03/the-spdz-protocol-part1/) [blog](/2017/09/10/the-spdz-protocol-part2/) [posts](/2017-09-19-private-image-analysis-with-mpc.md) for the latter. As for the different parties involved, we also here assume a [setting](/2017/09/19/private-image-analysis-with-mpc/#setting) with two server, a crypto producer, an input provider, and an output receiver.

One important note though is that TensorFlow works by first constructing a static [computation graph](https://www.tensorflow.org/programmers_guide/graphs) that is subsequently executed in a [session](https://www.tensorflow.org/api_guides/python/client). For instance, inspecting the graph we get from `sigmoid(dot(w, x) + b)` in [TensorBoard](https://www.tensorflow.org/programmers_guide/graph_viz) shows the following.

![](/assets/tensorspdz/structure.png)

This means that our efforts in this post are concerned with building such a graph, as opposed to actual execution as earlier: we are to some extend making a small compiler that translates secure computations expressed in a simple language into TensorFlow programs. As a result we benefit not only from working at a higher level of abstraction but also from the large amount of efforts that have already gone into optimising graph execution in TensorFlow.

See the [experiments](#experiments) for a full code example.


# Basics

Our needs fit nicely with the operations already provided by TensorFlow as seen next, with one main exception: to match typical precision of floating point numbers when instead working with [fixedpoint numbers](/2017/09/03/the-spdz-protocol-part1/#fixedpoint-numbers) in the secure setting, we end up encoding into and operating on integers that are larger than what fits in the typical word sizes of 32 or 64 bits, yet today these are the only sizes for which TensorFlow provides operations (a constraint that may have something to do with current support on GPUs).

Luckily though, for the operations we need there are efficient ways around this that allow us to simulate arithmetic on tensors of ~120 bit integers using a list of tensors with identical shape but of e.g. 32 bit integers. And this decomposition moreover has the nice property that we can often operate on each tensor in the list independently, so in addition to enabling the use of TensorFlow, this also allows most operations to be performed in parallel and can actually [increase efficiency](https://en.wikipedia.org/wiki/Residue_number_system) compared to operating on single larger numbers, despite the fact that it may initially sound more expensive. 

We discuss the [details](/2018/01/29/the-chinese-remainder-theorem/) of this elsewhere and for the rest of this post simply assume operations `crt_add`, `crt_sub`, `crt_mul`, `crt_dot`, `crt_mod`, and `sample` that perform the expected operations on lists of tensors. Note that `crt_mod`, `crt_mul`, and `crt_sub` together allow us to define a right shift operation for [fixedpoint truncation](/2017/09/03/the-spdz-protocol-part1/#fixedpoint-numbers).


## Private tensors

Each private tensor we operate on is represented by a share on each of the two servers. And for the reasons mentioned above, each share is a list of tensors, which is represented by a list of nodes in the graph. To hide this complexity we introduce a simple class as follows.

```python
class PrivateTensor:
    
    def __init__(self, share0, share1):
        self.share0 = share0
        self.share1 = share1
    
    @property
    def shape(self):
        return self.share0[0].shape
        
    @property
    def unwrapped(self):
        return self.share0, self.share1
```

And thanks to TensorFlow we can know the shape of tensors at graph creation time, meaning we don't have to keep track of this ourselves.


## Simple operations

Since a secure operation will often be expressed in terms of several TensorFlow operations, we use abstract operations such as `add`, `mul`, and `dot` as a convenient way of constructing the computation graph. The first one is `add`, where the resulting graph simply instructs the two servers to locally combine the shares they each have using a subgraph constructed by `crt_add`.

```python
def add(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)
    
    x0, x1 = x.unwrapped
    y0, y1 = y.unwrapped
    
    with tf.name_scope('add'):
    
        with tf.device(SERVER_0):
            z0 = crt_add(x0, y0)

        with tf.device(SERVER_1):
            z1 = crt_add(x1, y1)

    z = PrivateTensor(z0, z1)
    return z
```

Notice how easy it is to use [`tf.device()`](https://www.tensorflow.org/api_docs/python/tf/device) to express which server is doing what! This command ties the computation and its resulting value to the specified host, and instructs TensorFlow to automatically insert appropiate networking operations to make sure that the input values are available when needed!

As an example, in the above, if `x0` was previous on, say, the input provider then TensorFlow will insert send and receive instructions that copies it to `SERVER_0` as part of computing `add`. All of this is abstracted away and the framework will attempt to [figure out](https://www.tensorflow.org/about/bib) the best strategy for optimising exactly when to perform sends and receives, including batching to better utilise the network and keeping the compute units busy.

The [`tf.name_scope()`](https://www.tensorflow.org/api_docs/python/tf/name_scope) command on the other hand is simply a logical abstraction that doesn't influence computations but can be used to make the graphs much easier to visualise in [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) by grouping subgraphs as single components as also seen earlier.

![](/assets/tensorspdz/add.png)

Note that by selecting *Device* coloring in TensorBoard as done above we can also use it to verify where the operations were actually computed, in this case that addition was indeed done locally by the two servers (green and turquoise).


## Dot products

We next turn to dot products. This is more complex, not least since we now need to involve the crypto producer, but also since the two servers have to communicate with each other as part of the computation.

```python
def dot(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)

    x0, x1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('dot'):

        # triple generation

        with tf.device(CRYPTO_PRODUCER):
            a = sample(x.shape)
            b = sample(y.shape)
            ab = crt_dot(a, b)
            a0, a1 = share(a)
            b0, b1 = share(b)
            ab0, ab1 = share(ab)
        
        # masking after distributing the triple
        
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
        
    z = PrivateTensor(z0, z1)
    z = truncate(z)
    return z
```

However, with `tf.device()` we see that this is still relatively straight-forward, at least if the [protocol for secure dot products](/2017/09/19/private-image-analysis-with-mpc/#dense-layers) is already understood. We first construct a graph that makes the crypto producer generate a new dot triple. The output nodes of this graph is `a0, a1, b0, b1, ab0, ab1`

With `crt_sub` we then build graphs for the two servers that mask `x` and `y` using `a` and `b` respectively. TensorFlow will again take care of inserting networking code that sends the value of e.g. `a0` to `SERVER_0` during execution.

In the third step we reconstruct `alpha` and `beta` on each server, and compute the recombination step to get the dot product. Note that we have to define `alpha` and `beta` twice, one for each server, since although they contain the same value, if we had instead define them only on one server but used them on both, then we would implicitly have inserted additional networking operations and hence slowed down the computation.

![](/assets/tensorspdz/dot.png)

Returning to TensorBoard we can verify that the nodes are indeed tied to the correct players, with yellow being the crypto producer, and green and turquoise being the two servers. Note the convenience of having `tf.name_scope()` here.


## Configuration

To fully claim that this has made the distributed aspects of secure computations much easier to express, we also have to see what is actually needed for `td.device()` to work as intended. In the code below we first define an arbitrary job name followed by identifiers for our five players. More interestingly, we then simply specify their network hosts and wrap this in a [`ClusterSpec`](https://www.tensorflow.org/deploy/distributed). That's it!

```python
JOB_NAME = 'spdz'

SERVER_0        = '/job:{}/task:0'.format(JOB_NAME)
SERVER_1        = '/job:{}/task:1'.format(JOB_NAME)
CRYPTO_PRODUCER = '/job:{}/task:2'.format(JOB_NAME)
INPUT_PROVIDER  = '/job:{}/task:3'.format(JOB_NAME)
OUTPUT_RECEIVER = '/job:{}/task:4'.format(JOB_NAME)

HOSTS = [
    '10.132.0.4:4440',
    '10.132.0.5:4441',
    '10.132.0.6:4442',
    '10.132.0.7:4443',
    '10.132.0.8:4444',
]

CLUSTER = tf.train.ClusterSpec({
    JOB_NAME: HOSTS
})
```

<em>Note that in the screenshots we are actually running the input provider and output receiver on the same host, and hence both show up as `3/device:CPU:0`.</em>

Finally, the code that each player executes is about as simple as it gets.

```python
server = tf.train.Server(CLUSTER, job_name=JOB_NAME, task_index=ROLE)
server.start()
server.join()
```

Here the value of `ROLE` is the only thing that differs between the programs the five players run and typically given as a command-line argument.


# Improvements

With the basics in place we can look at a few optimisations.


## Tracking nodes

Our first improvement allows us to reuse computations. For instance, if we need the result of `dot(x, y)` twice then we want to avoid computing it a second time and instead reuse the first. Concretely, we want to keep track of nodes in the graph and link back to them whenever possible.

To do this we simply maintain a global dictionary of `PrivateTensor` references as we build the graph, and use this for looking up already existing results before adding new nodes. For instance, `dot` now becomes as follows.

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

While already significant for some applications, this change also opens up for our next improvement.


## Reusing masked tensors

We have [already](/2017/09/10/the-spdz-protocol-part2/) [mentioned](/2017/09/19/private-image-analysis-with-mpc/#generalised-triples) that we'd ideally want to mask every private tensor at most once to primarily save on networking. For instance, if we are computing both `dot(w, x)` and `dot(w, y)` then we want to use the same masked version of `w` in both. Specifically, if we are doing many operations with the same masked tensor then the cost of masking it can be amortised away.

But with the current setup we mask every time we compute e.g. `dot` or `mul` since masking is baked into these. So to avoid this we simply make masking an explicit operation, additionally allowing us to also use the same masked version across different operations.

```python
def mask(x):
    assert isinstance(x, PrivateTensor)
    
    node_key = ('mask', x)
    masked = nodes.get(node_key, None)

    if masked is None:
      
        x0, x1 = x.unwrapped
        shape = x.shape
      
        with tf.name_scope('mask'):

            with tf.device(CRYPTO_PRODUCER):
                a = sample(shape)
                a0, a1 = share(a)

            with tf.device(SERVER_0):
                alpha0 = crt_sub(x0, a0)

            with tf.device(SERVER_1):
                alpha1 = crt_sub(x1, a1)

            # exchange of alphas

            with tf.device(SERVER_0):
                alpha_on_0 = reconstruct(alpha0, alpha1)

            with tf.device(SERVER_1):
                alpha_on_1 = reconstruct(alpha0, alpha1)

        masked = MaskedPrivateTensor(a, a0, a1, alpha_on_0, alpha_on_1)
        nodes[node_key] = masked
        
    return masked
```

Note that we introduce a `MaskedPrivateTensor` class as part of this, which is again simply a convenient way of abstracting over the five lists of tensors we get from `mask(x)`.

```python
class MaskedPrivateTensor(object):

    def __init__(self, a, a0, a1, alpha_on_0, alpha_on_1):
        self.a  = a
        self.a0 = a0
        self.a1 = a1
        self.alpha_on_0 = alpha_on_0
        self.alpha_on_1 = alpha_on_1

    @property
    def shape(self):
        return self.a[0].shape

    @property
    def unwrapped(self):
        return self.a, self.a0, self.a1, self.alpha_on_0, self.alpha_on_1
```


With this we may rewrite `dot` as below, which is now only responsible for the recombination step.

```python
def dot(x, y):
    assert isinstance(x, PrivateTensor) or isinstance(x, MaskedPrivateTensor)
    assert isinstance(y, PrivateTensor) or isinstance(y, MaskedPrivateTensor)

    node_key = ('dot', x, y)
    z = nodes.get(node_key, None)

    if z is None:
      
        if isinstance(x, PrivateTensor): x = mask(x)
        if isinstance(y, PrivateTensor): y = mask(y)

        a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
        b, b0, b1,  beta_on_0,  beta_on_1 = y.unwrapped

        with tf.name_scope('dot'):

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

        z = PrivateTensor(z0, z1)
        z = truncate(z)
        nodes[node_key] = z

    return z
```

As a verification we can see that TensorBoard shows us the expected graph structure, in this case inside the graph for [`sigmoid`](/2017/04/17/private-deep-learning-with-mpc/#approximating-sigmoid).

![](/assets/tensorspdz/masking-reuse.png)

Here the value of `square(x)` is first computed, then masked, and finally reused in four multiplications.

There is an inefficiency though: while the [dataflow nature](https://arxiv.org/abs/1603.04467) of TensorFlow will in general take care of only recomputing the parts of the graph that have changed between two executions, this does not apply to operations involving sampling via e.g. [`tf.random_uniform`](https://www.tensorflow.org/api_docs/python/tf/random_uniform), which is used in our sharing and masking. Consequently, masks are not being reused across executions.


## Caching values

To get around the above issue we can introduce caching of values that survive across different executions of the graph, and an easy way of doing this is to store tensors in [variables](https://www.tensorflow.org/api_docs/python/tf/Variable). Normal executions will read from these, while an explicit `cache_populators` set of operations allow us to populated them.

For example, wrapping our two tensors `w` and `b` with such `cache` operation gets us the following graph.

![](/assets/tensorspdz/cached.png)

When executing the cache population operations TensorFlow automatically figures out which subparts of the graph it needs to execute to generate the values to be cached, and which can be ignored.

![](/assets/tensorspdz/cached-populate.png)

And likewise when predicting, in this case skipping sharing and masking.

![](/assets/tensorspdz/cached-predict.png)



## Buffering triples

Recall that a main purpose of [triples](/2017/09/03/the-spdz-protocol-part1/#multiplication) is to move the computation of the crypto producer to an *offline phase* and distribute its results to the two servers ahead of time in order to speed up their computation later during the *online phase*. 

So far we haven't done anything to specify that this should happen though, and from reading the above code it's not unreasonable to assume that the crypto producer will instead compute in synchronisation with the two servers, injecting idle waiting periods throughout their computation. However, from experiments it seems that TensorFlow is already smart enough to optimise the graph to do the right thing and batch triple distribution, presumably to save on networking. We still have an initial waiting period though, that we could get rid of by introducing a separate compute-and-distribute execution that fills up buffers.

![](/assets/tensorspdz/tracing.png)

We'll skip this issue for now and instead return to it when looking at private training since it is not unreasonable to expect significant performance improvements there from distributing the training data ahead of time.


# Profiling

As a final reason to be excited about building dataflow programs in TensorFlow we also look at the built-in [runtime statistics](https://www.tensorflow.org/programmers_guide/graph_viz#runtime_statistics). We have already seen the built-in detailed tracing support above, but in TensorBoard we can also easily see how expensive each operation was both in terms of compute and memory. The numbers reported here are from the [experiments](#experiments) below.

![](/assets/tensorspdz/computetime.png)

The heatmap above e.g. shows that `sigmoid` was the most expensive operation in the run and that the dot product took roughly 30ms to execute. Moreover, in the below figure we have navigated further into the dot block and see that sharing in this particular run taking about 3ms.

![](/assets/tensorspdz/computetime-detailed.png)

This way we can potentially identify bottlenecks and compare performance of different approaches. And if needed we can of course switch to tracing for even more details.


# Experiments

The [GitHub repository](https://github.com/mortendahl/privateml/tree/master/tensorflow/spdz/) contains the code needed for experimentation, including examples and instructions for setting up either a [local configuration](https://github.com/mortendahl/privateml/tree/master/tensorflow/spdz/configs/localhost) or a [GCP configuration](https://github.com/mortendahl/privateml/tree/master/tensorflow/spdz/configs/gcp) of hosts. For the running example of private prediciton using a logistic regression model we use the GCP configuration, i.e. the parties are running on different virtual hosts located in the same Google Cloud zone, here on some of the weaker instances, namely dual core and 10GB memory.

A slightly simplified version of our program is as follows, where we first train a model in public, build a graph for the private prediction computation, and then run it in a fresh session. The model was somewhat arbitrarily picked to have 100 features.

```python
from config import session
from tensorspdz import (
    define_input, define_variable,
    add, dot, sigmoid, cache, mask,
    encode_input, decode_output
)

# publicly train `weights` and `bias`
weights, bias = train_publicly()

# define shape of unknown input
shape_x = X.shape

# construct graph for private prediction
input_x, x = define_input(shape_x, name='x')

init_w, w = define_variable(weights, name='w')
init_b, b = define_variable(bias, name='b')

if use_caching:
    w = cache(mask(w))
    b = cache(b)

y = sigmoid(add(dot(x, w), b))

# start session between all players
with session() as sess:

    # share and distribute `weights` and `bias` to the two servers
    sess.run([init_w, init_b])
    
    if use_caching:
        # compute and store cached values
        sess.run(cache_populators)

    # prepare to use `X` as private input for prediction
    feed_dict = encode_input([
        (input_x, X)
    ])

    # run secure computation and reveal output
    y_pred = sess.run(reveal(y), feed_dict=feed_dict)
    
    print decode_output(y_pred)
```

Running this a few times with different sizes of `X` gives the timings below, where the entire computation is considered including triple generation and distribution; slightly surprisingly there were no real difference between caching masked values or not.

<center>
<img width="80%" height="80%" src="/assets/tensorspdz/timings-10000.png" />
</center>

Processing batches of size 1, 10, and 100 took roughly the same time, ~60ms on average, which might suggest a lower latency bound due to networking. At 1000 the time jumps to ~110ms, at 10,000 to ~600ms, and finally at 100,000 to ~5s. As such, if latency is important than we can perform ~1600 predictions per second, while if more flexible then at least ~20,000 per second.

<center>
<img width="80%" height="80%" src="/assets/tensorspdz/timings-100000.png" />
</center>

This however is measuring only timings reported by profiling, with actual execution time taking a bit longer; hopefully some of the production-oriented tools such as [`tf.serving`](https://www.tensorflow.org/serving/) that come with TensorFlow can improve on this.


# Thoughts

After private prediction it'll of course also be interesting to look at private training. Caching of masked training data might be more relevant here since it remains fixed throughout the process.

The serving of models can also be improved, using for instance the production-ready [`tf.serving`](https://www.tensorflow.org/serving/) one might be able to avoid much of the current initial overhead for orchestration, as well as having endpoints that can be [safely exposed](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) to the public.

Finally, there are security improvements to be made on e.g. communication between the five parties. In particular, in the current version of TensorFlow all communication is happening over unencrypted and unauthenticated [gRPC](https://grpc.io/) connections, which means that someone listening in on the network traffic in principle could learn all private values. Since support for [TLS](https://grpc.io/docs/guides/auth.html) is already there in gRPC it might be straight-forward to make use of it in TensorFlow without a significant impact on performance. Likewise, TensorFlow does not currently use a strong pseudo-random generator for [`tf.random_uniform`](https://www.tensorflow.org/api_docs/python/tf/random_uniform) and hence sharing and masking are not as secure as they could be; adding an operation for cryptographically strong randomness might be straight-forward and should give roughly the same performance.


<!--
# Dump

- https://learningtensorflow.com/lesson11/
- [XLA](https://www.tensorflow.org/performance/xla/)

TODO overhead compared to plain (maybe wait until training?)

https://www.tensorflow.org/programmers_guide/graphs
https://en.wikipedia.org/wiki/Dataflow_programming

https://github.com/ppwwyyxx/tensorpack
https://github.com/tensorflow/serving/issues/193
https://github.com/sandtable/ssl_grpc_example

[TensorBoard](https://www.tensorflow.org/programmers_guide/graph_viz)

-->