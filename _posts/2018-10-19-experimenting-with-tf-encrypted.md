---
layout:     post
title:      "Experimenting with TF Encrypted"
subtitle:   "A Library for Privacy-Preserving Machine Learning in TensorFlow"
date:       2018-10-19 12:00:00
author:     "Morten Dahl"
header-img: "assets/tfe/iss-orbit.jpg"
summary:    "We apply TF Encrypted to a typical deep learning example, providing a good starting point for anyone wishing to get into this rapidly growing field. As shown, using state-of-the-art secure computation techniques to serve predictions on encrypted data requires nothing more than a basic familiarity with deep learning and TensorFlow."
---

Privacy-preserving machine learning offers many benefits and interesting applications: being able to train and predict on data while it remains in encrypted form unlocks the utility of data that were previously inaccessible due to privacy concerns. But to make this happen several technical fields must come together, including cryptography, machine learning, distributed systems, and high-performance computing.

The [TF Encrypted](https://tf-encrypted.io) open source project aims at bringing researchers and practitioners together in a familiar framework in order to accelerate exploration and adaptation. By building directly on [TensorFlow](https://www.tensorflow.org/) it provides a high performance framework with an easy-to-use interface that abstracts away most of the underlying complexity, allowing users with only a basic familiarity with machine learning and TensorFlow to apply state-of-the-art cryptographic techniques without first becoming cross-disciplinary experts.

In this blog post we apply the library to a traditional machine learning example, providing a good starting point for anyone wishing to get into this rapidly growing field.

<em>This is a [cross-posting](https://medium.com/dropoutlabs/experimenting-with-tf-encrypted-fe37977ff03c) of work done at [Dropout Labs](https://dropoutlabs.com) with [Jason Mancuso](https://twitter.com/jvmancuso).</em>

# TensorFlow and TF Encrypted

We start by looking at how our task can be solved in standard TensorFlow and then go through the changes needed to make the predictions private via TF Encrypted. Since the interface of the latter is meant to simulate the simple and concise expression of common machine learning operations that TensorFlow is well-known for, this requires only a small change that highlights what one must inherently think about when moving to the private setting.

Following standard practice, the following script shows our two-layer feedforward network with ReLU activations (more details in the [preprint](https://arxiv.org/abs/1810.08130)).

Concretely, we consider the classic [MNIST digit classification task](http://yann.lecun.com/exdb/mnist/). To keep things simple we use a small neural network and train it in the traditional way in TensorFlow using an unencrypted training set. However, for making predictions with the trained model we turn to TF Encrypted, and show how two servers can perform predictions for a client without learning anything about its input. While this is a basic yet somewhat standard benchmark in the literature, the techniques used carry over to many different use cases, including medical image analysis.

```python
import tensorflow as tf

# generic functions for loading model weights and input data

def provide_weights(): """Load model weights as TensorFlow objects."""
def provide_input(): """Load input data as TensorFlow objects."""
def receive_output(logits): return tf.print(tf.argmax(logits))

# get model weights/input data (both unencrypted)

w0, b0, w1, b1, w2, b2 = provide_weights()
x = provide_input()

# compute prediction

layer0 = tf.nn.relu((tf.matmul(x, w0) + b0))
layer1 = tf.nn.relu((tf.matmul(layer0, w1) + b1))
logits = tf.matmul(layer2, w2) + b2

# get result of prediction and print

prediction_op = receive_output(logits)

# run graph execution in a tf.Session

with tf.Session() as sess:
    sess.run(prediction_op)
```

Note that the [concrete implementation](https://github.com/tf-encrypted/tf-encrypted/tree/master/examples/mnist) of `provide_weights` and `provide_input` have been left out for the sake of readability. These two methods simply load their respective values from NumPy arrays stored on disk, and return them as tensor objects.

We next turn to making the predictions private, where for the notion of privacy and encryption to even make sense we first need to recast our setting to consider more than the single party implicit in the script above. As seen below, expressing our intentions about who should get to see which values is the biggest difference between the two scripts.

We can naturally identify two of the parties: the prediction client who knows its own input and a model owner who knows the weights. Moreover, for the secure computation protocol chosen here we also need two servers that will be doing the actual computation on encrypted values; this is often desirable in applications where the clients may be mobile devices that have significant restraints on computational power and networking bandwidth.

![](/assets/tfe/prediction-flow.png)

In summary, our data flow and privacy assumptions are as illustrated in the diagram above. Here a model owner first gives encryptions of the model weights to the two servers in the middle (known as a *private input*), the prediction client then gives encryptions of its input to the two servers (another private input), who can execute the model and send back encryptions of the prediction result to the client, who can finally decrypt; at no point can the two servers decrypt any values. Below we see our script expressing these privacy assumptions.

```python
import tensorflow as tf
import tf_encrypted as tfe

# generic functions for loading model weights and input data on each party

def provide_weights(): """Loads the model weights on the model-owner party."""
def provide_input(): """Loads the input data on the prediction-client party."""
def receive_output(): return tf.print(tf.argmax(logits))

# get model weights/input data as private tensors from each party

w0, b0, w1, b1, w2, b2 = tfe.define_private_input("model-owner", provide_weights)
x = tfe.define_private_input("prediction-client", provide_input)

# compute secure prediction

layer0 = tfe.relu((tfe.matmul(x, w0) + b0))
layer1 = tfe.relu((tfe.matmul(layer0, w1) + b1))
logits = tfe.matmul(layer1, w2) + b2

# send prediction output back to client

prediction_op = tfe.define_output("prediction-client", receive_output, logits)

# run secure graph execution in a tfe.Session

with tfe.Session() as sess:
    sess.run(prediction_op)
```

Note that most of the code remains essentially identical to the traditional TensorFlow code, using `tfe` instead of `tf`:

- The `provide_weights` method for loading model weights is now wrapped in a call to `tfe.define_private_input` in order to specify they should be owned and restricted to the model owner; by wrapping the method call, TF Encrypted will encrypt them before sharing with other parties in the computation.

- As with the weights, the prediction input is now also only accessible to the prediction client, who is also the only receiver of the output. Here the `tf.print` statement has been moved into `receive_output` as this is now the only point where the result is known in plaintext.

- We also tie the name of parties to their network hosts. Although omitted here, this information also needs to be available on these hosts, as typically shared via a simple configuration file.

# What’s the Point?

- **user-friendly**: very little boilerplate, very similar to traditional TensorFlow.

- **abstract and modular**: it integrates secure computation tightly with machine learning code, hiding advanced cryptographic operations underneath normal tensor operations.

- **extensible**: new protocols and techniques can be added under the hood, and the high-level API won’t change. Similarly, new machine learning layers can be added and defined on top of each underlying protocol as needed, just like in normal TensorFlow.

- **fast**: all of this is computed efficiently since it gets compiled down to ordinary TensorFlow graphs, and can hence take advantage of the optimized primitives for distributed computation that the TensorFlow backend provides.

These properties also make it easy to **benchmark** a diverse set of combinations of machine learning models and secure computation protocols. This allows for more fair comparisons, more confident experimental results, and a more rigorous empirical science, all while lowering the barrier to entry to private machine learning.

Finally, by operating directly in TensorFlow we also benefit from its ecosystem and can take advantage of existing tools such as [TensorBoard](https://www.tensorflow.org/guide/graph_viz). For instance, one can profile which operations are most expensive and where additional optimizations should be applied, and one can inspect where values reside and ensure correctness and security during implementation of the cryptographic protocols as shown below.

![](/assets/tensorspdz/masking-reuse.png)

Here, we visualize the various operations that make up a secure operation on two private values. Each of the nodes in the underlying computation graph are shaded according to which machine aided that node’s execution, and it comes with handy information about data flow and execution time. This gives the user a completely transparent yet effective way of auditing secure computations, while simultaneously allowing for program debugging.

# Conclusion

[TF Encrypted](https://github.com/tf-encrypted) is about providing researchers and practitioners with the open-source tools they need to quickly experiment with secure protocols and primitives for private machine learning.

The hope is that this will aid and inspire the next generation of researchers to implement their own novel protocols and techniques for secure computation in a fraction of the time, so that machine learning engineers can start to apply these techniques for their own use cases in a framework they’re already intimately familiar with.

To find out more have a look at the recent [preprint](https://arxiv.org/abs/1810.08130) or dive into the [examples on GitHub](https://github.com/tf-encrypted/tf-encrypted/tree/master/examples)!