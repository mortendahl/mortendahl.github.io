---
layout:     post
title:      "Privacy-Preserving Machine Learning"
subtitle:   "And How It Mitigates Bottlenecks"
date:       2018-08-29 12:00:00
author:     "Morten Dahl"
header-img: "assets/bottlenecks/impact.jpg"
summary:    "The need for privacy-enhancing techniques in machine learning can be understood in the light of bottlenecks that may otherwise exist. We here review some of those bottlenecks together with concrete technologies available to mitigate and create new opportunities.
"
---

<em><strong>TL;DR:</strong> TODO.</em> 

Entering the field of privacy-preserving machine learning can be confusing, not least because of the many different techniques that may be new to most. In this post I give a quick overview of some of these, and illustrate which ones compliment each other and which ones solve the same problem but have different characteristics.

The techniques I want to talk about are *secure computation*, *differential privacy*, and *federated learning*. There are other [privacy-enhancing technologies](https://en.wikipedia.org/wiki/Privacy-enhancing_technologies), such as encrypted search, private information retrieval, and private set intersection, that may also in one form or another be used as part of privacy-preserving machine learning; however, these are more specialised tools where the utility depends more strongly on the concrete use cases.

<em>Much of this post is based on [recent talks](/talks/) and some of the work have been done as part of my work as a research scientist at [Dropout Labs](https://dropoutlabs.com/).</em>

- SGX
- HE
- SS
- GC
- PATE
- FL
- DP

# The Machine Learning Process

It is useful to use the typical machine learning process to frame

<img src="/assets/bottlenecks/process-simple.png" style="width: 50%;"/>

<img src="/assets/bottlenecks/process-context.png" style="width: 50%;"/>

# Bottlenecks

## Access to Training Data

liability

price

(consumers)

## Holding Training Data

Data as a toxic assest

## Incentivizing Use


<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>
