---
layout:     post
title:      "Secure Aggregation, Part 2"
subtitle:   "Private Data Aggregation on a Budget"
date:       2019-01-09 12:00:00
author:     "Morten Dahl"
header-img: "assets/sda/constellation.png"
summary:    "In this blog post we walk through a protocol suitable for secure aggregation with sporadic devices, explaining the hows and whys of several cryptographic techniques in the process."
---

<style>
img {
    margin-left: auto;
    margin-right: auto;
}

.intro-header .post-heading h1 {
    font-size: 54px;
}
</style>

<em><strong>TL;DR:</strong> In this blog post we go through ways in which modern cryptography can be used to perform secure aggregation in for instance federated learning. As we will see, the right approach depends on the concrete scenario, including the stakes of the participating parties.

Secure Aggregation for Sporadic Devices

<!--
Federated learning has gained a lot of popularity in recent years as a way of keeping training data ...

What happens when you want to do secure multiparty computation but you're the only party that's really interested in investing resources in it?

As we will see, the main question we answer is, what happens when you want to do secure multiparty computation but you're the only party that's really interested in investing resources in it?

Secure multiparty computation (MPC) is an amazing technology when you need to ensure privacy in computations: you essentially have a bunch of parties that each have a data set they wish to keep private, yet using these techniques they can still evaluate a function on the joint data sets and only reveal the output.

In this blog post we'll step through a simple solution for federated learning we came up with a few years ago.

NOT JUST FEDERATED BUT ALSO STATISTICS: we want to support many users

this came out at about the same time as Google's secure aggregation paper; where they were training neural networks by averaging gradients, we wanted to generate global prior distributions for recommendations (see paper for more use cases)

in this post i want to not only show how it works but also how we built it
-->

<em>Parts of this blog post are based on [work](https://eprint.iacr.org/2017/643) [done](https://medium.com/snips-ai/private-analytics-with-sda-d98a0251ab32) at [Snips](https://snips.ai) together with [Valerio Pastro](https://www.google.com/search?q=Valerio+Pastro) and [Mathieu Poumeyrol](https://twitter.com/kalizoy). In particular, the presentation here roughly follows [the talk](https://github.com/mortendahl/talks/blob/master/HEAT17-slides.pdf) given at the HEAT'17 workshop.</em>

# Motivation


Scaling to Many Users

<img src="/assets/sda/scaling.png" style="width: 100%;"/>

<em>(coming ...)</em>

<!--

https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

https://eprint.iacr.org/2017/281

https://arxiv.org/abs/1902.01046

we had statistics in mind before thinking about DL

-->

## Secure aggregation

From the above use-cases we are going to focus on the question of how to aggregate large vectors in such a way that the inputs are kept private and only the output is revealed to the intended recipient. However, to keep things simple, in this blog post we will only talk about aggregating scalars values; adapting this to vectors as done in [the full paper](https://eprint.iacr.org/2017/643) is straight-forward.

Specifically, we assume a set of data owners that each hold a private integer `x`, as well as an output receiver without any inputs. Our goal is then for the latter to learn the sum of these values and nothing else.

```python
data_owners = [
    DataOwner('User1', x=10),
    DataOwner('User2', x=20),
    DataOwner('User3', x=30),
    DataOwner('User4', x=40),
    DataOwner('User5', x=50),
    DataOwner('User6', x=60),
]

output_receiver = OutputReceiver()
```

Note however, that this might allow *some* leakage about the inputs, namely that which can be derived from the output. For instance, in the above case, simply by learning that the sum is `210`, one also learns that for a randomly picked owner there is at least a 50% probability that it had an input below `210/6 == 35`.

# Fully Decentralised

this is typically a good solution when the aggregate happens between a few data owners connected on reilable high-speed networks

gives high security since each data owner only has to trust themselves

Someone knowledgeable in secure multi-party computation may immediately jump on this as a simple application of an old technique: [secret sharing](/2017/06/04/secret-sharing-part1/). Concretely, with e.g. six users one could very easily imagine the flow illustrated in the following figure and detailed next.

<img src="/assets/sda/slides.001.jpeg">

In the first step each user `ui` secret shares their vector `xi` into a list `si = share(xi)` consisting of six shares `si1, ..., si5`.

```python
# xi is known only by ui
assert x == [x1, x2, x3, x4, x5, x6]

for xi in x:
    # executed locally by user ui
    si = share(xi)
```

These shares are then distributed to the other users, sending `si1` to `u1`, `si2` to `u2`, and so on. As a result each user ends up holding a list `ti` of five shares `s1i`, `s2i`, ..., `s5i`, one of which generated by themself (`sii` to be precise) and four of which received from the others.

```python
# only ui knows all values in si
assert s == [s1, s2, s3, s4, s5]

# users distribute shares
t = zip(*s)
```

One way to visualize this is by writing all 25 shares as a 5x5 matrix, with `s1`, `s2`, ..., `s5` making up the rows and `t1`, `t2`, ..., `t5` making up the columns.

(visualize matrix)

Next, each user `ui` applies a homomorphic property of the secret sharing scheme to aggregate the five shares they hold in `ti`, which in our case simply amount to summing them.

```python
# only ui knows all values in ti
assert t == [t1, t2, t3, t4, t5]

for ti in t:
    # executed locally by user ui
    ri = sum(ti) % Q
```

Having done so we end up with a total of five shares `r1`, `r2`, ..., `r5`, which are finally sent to the output receiver and used for reconstructing `sum(x)`.

```python
# only the output receiver knows all values in r
assert r = [r1, r2, r3, r4, r5]

# executed locally by the output receiver
y = reconstruct(r)

assert y == sum(x)
```


```python
class FullyDecentralisedUser:

    def __init__(self, x):
        self.x = x

    def generate_shares(self):
        self.outgoing_shares = additive_share(self.x)

    def distribute_shares(self, receivers)



u1 = FullyDecentralisedUser(x1)
u2 = FullyDecentralisedUser(x2)
u3 = FullyDecentralisedUser(x3)
u4 = FullyDecentralisedUser(x4)
u5 = FullyDecentralisedUser(x5)

users = [u1, u2, u3, u4, u5]
```

This works and provides very strong privacy guarantees: using an [additive secret sharing sharing](/2017/06/04/secret-sharing-part1/#additive-sharing) means that privacy of `xi` is guaranteed even if all other users are colluding.  ... are guaranteed even if give us perfect privacy

it unfortunately also has a serious problem if we were to use it across e.g. the internet, where users are somewhat sporadic. In particular, the distribution of shares represents a significant synchronization point between all users, where even a single one of them can bring the protocol to a halt by failing to send their shares.

## Correlated randomness

zero-sharing

To some extend this is the basis of Google's approach, where thet pick a subsample of data owners and add recovery mechanism

https://eprint.iacr.org/2017/281
Google’s Approach: https://acmccs.github.io/papers/p1175-bonawitzA.pdf

Correlated randomness, Google’s approach, and sensor networks

Fully decentralized can be broken into offline and online phase, where shares of zero are first generated by the users. This is essentially the basis of the Google protocol. One problem is if some users drop out between the two phases, and the Google protocol extends the basic idea to mitigate that issue. From an operational perspective they also keep the iterations very short, and only aggregate when phones are most stable (plugged and on wifi).

# Server-aided

A typical solution to the above is what is known as the server-aided model. Here, instead of running the protocol directly between the users, we run it between a smaller set of more reliable servers. 

Note that this changes the trust model: before each user only had to trust themselves but now they have to trust the servers.

 is now run between a small set of servers  select no longer run between the users but  computation  set up may in particular feel like the right fit here as it is probably a good idea to limit the amount of synchronization between users as much as possible. Concretely, one could easily imagine the following: 

1. each user secret shares their vector `x` into shares `s1, s2, s3 = share(x)`; with `N` users we hence end up with a total of `3*N` shares
2. each user then distribute their shares to three servers,   them to a small set of servers who uses homomorphic properties of the secret sharing scheme to perform an aggregation on the individual shares, before finally revealing only the aggregated shares to the intended output receiver, thereby allowing him to reconstruct only the aggregate.

<img src="/assets/sda/slides.001.jpeg">
(update photo to use generic servers instead of clerks)

```python
class Server:

    pass

class User(InputProvider):

class InputProvider:


```

2. Each user then uses a homomorphic property of the secret sharing scheme to aggregate his five shares. Concretely, when our aggregation is a summation he simply adds the shares obtaining `y = s1 + ... + 
  three servers,   them to a small set of servers who uses homomorphic properties of the secret sharing scheme to perform an aggregation on the individual shares, before finally revealing only the aggregated shares to the intended output receiver, thereby allowing him to reconstruct only the aggregate.

```python

shares

```


we were quick to say "secret sharing" but as Nigel later phrased it <em>the problem is to find someone to play with</em>

## Sporadic servers

Our Approach, SDA
clerks

### Robustness

<img src="/assets/sda/slides.001.jpeg">

Switch to threshold secret sharing scheme to mitigate clerk availability: lower recovery threshold

Means privacy threshold drops as well, so enough clerks coming together could learn values

```python
def shamir_share(threshold)

```

```python
class SimpleClerk(Server):
    pass
```

### Privacy

blinding

no amount of clerks on their own can learn anything; now need privacy threshold together with output receiver

<img src="/assets/sda/slides.002.jpeg">

<img src="/assets/sda/slides.002.jpeg">
(with PRG)

### Availability

<img src="/assets/sda/slides.003.jpeg">

```python
class BulletinBoard:

    def put(self, sender, receiver, message):

    def get(self, sender, receiver):
```

## Resource-contrained servers

By nature of hardware or by incentive

Scaling to any number of users

<img src="/assets/sda/slides.004.jpeg">


packed PHE for reducing data transfer

packed SS for scaling with ???


## Cheating servers

Detecting cheating clerks

<img src="/assets/sda/slides.005.jpeg">

# Further Considerations

## Differential privacy

## Input pollution
