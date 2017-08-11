---
layout:     post
title:      "Recent Talks on Privacy"
subtitle:   "Slides from PMPML'16, TPMPC'17, and PSA'17"
date:       2016-06-28 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-06.jpg"
---

During winter and spring I was fortunate enough to have a few occasions to talk about some of the work done at [Snips](https://snips.ai) on applying [privacy-enhancing technologies](https://en.wikipedia.org/wiki/Privacy-enhancing_technologies) to concrete problems encountered while building privacy-aware machine learning systems for mobile devices. 

These were mostly centered around our [*Secure Distributed Aggregator*](https://github.com/mortendahl/sda) (SDA) for learning from user data distributed on mobile devices in a privacy-preserving manner, i.e. without learning any individual data only the final aggregation, but there was also room for some discussion around privacy at a higher level, including how it has played into the business of the company.


# Private Data Aggregation on a Budget

SDA protocol from above further explain in detail, as well as more details reports on applications and experiments


concrete protocol for application of secure computation to federated learning 

- present concrete application of MPC to primarily academic
- propose practical considerations taken in implementation, such as packed secret sharing and packed Paillier
- highlight difficulty of independent parties to employ MPC due to the need of other parties to split trust with; proposed `community MPC` as a potential model for further protocols

This talk, given at workshop on [*Theory and Practice of Multi-Party Computation 2017 (TPMPC'17)*](http://www.multipartycomputation.com/tpmpc-2017)

[*Private Multi-Party Machine Learning 2016 (PMPML'16)*](https://pmpml.github.io/PMPML16/), a [*NIPS'16*](https://nips.cc/Conferences/2016) associated workshop.

[paper](https://eprint.iacr.org/2017/643)

[poster](https://github.com/mortendahl/privateml/raw/master/talks/PMPML16-poster.pdf)

[slides](https://github.com/mortendahl/privateml/raw/master/talks/TPMPC17-slides.pdf)

# What Privacy Has Meant For Snips

The aim of [this talk](https://github.com/mortendahl/privateml/raw/master/talks/PSA17-slides.pdf) at the workshop on [*Privacy in Statistical Analysis 2017 (PSA'17)*](http://wwwf.imperial.ac.uk/~nadams/events/ic-rss2017/ic-rss2017.html) was to give an industrial overview of how privacy has played a role at Snips from its beginning. To this end the talk was divided into a few areas where privacy has been involved.


Starting with the most recent since this is the more general one.

### Accessing Data
By keeping data sets locally on mobile devices, Snips obtained a high percentage of users who were willing to give access to sensitive data such as emails, chats, location tracking, and screen content. Since the success of the user experience has heavily dependent on having this access, privacy played the role of gaining the user's trust and minimizing his perceived risk in using the app. Running locally is easy to explain.

### Protecting the Company
By not having any user data there is less to worry about if company systems are compromised. Some services hosted by the company may however build up a set of metadata that in itself could hurt the trust the company is trying to build with the user. Powerful cryptographic techniques exist to minimize this (such as the Tor network and private information retrieval), however experiments indicated that the overhead is still too large. Luckily, under the assumption that the company is generally behaving honestly, we may instead decide to protect only against e.g. accidental logging of sensitive data, and simpler techniques are enough to limit the number of places where we have to pay special attention.

risk management
Cannot overspend on security
*Data is a Toxic Asset*

### Learning from Data
Very relevant for the company to get insight into the distributed and locally stored data sets (for feedback, recommendations, etc.) yet often there is no need to see individual data. How can we learn from these data sets without seen individual data, especially under the constraints of mobile users.

Brief comparison of
- sensor networks: high performance; but requires a lot of coordination between users
- differential privacy: high performance and strong privacy guarantees; but a lot of data is needed
- homomorphic encryption: flexible and explainable; but still not very efficient and has the issue of who keeps the decryption keys
- multi-party computation: flexible and decent performance; but requires other players

Focusing on MPC the idea is to use a community to find other players, in particular a subset of the users, and then minimize the amount of resources they have to spend on the computation by out sourcing as much as possible to the company.

Concrete protocol of aggregation high-dimensional vectors (Secure Distributed Aggregator) proposed and reported on implementation results. Efficient for mobile and web users.



