---
layout:     post
title:      "Recent Talks on Privacy"
subtitle:   "Slides from PMPML'16, TPMPC'17, and PSA'17"
date:       2017-08-12 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-06.jpg"
summary:    "Overview of recent work done at Snips on applying privacy-enhancing technologies as a start-up building privacy-aware machine learning systems for mobile devices. Mainly centered around secure aggregation for federated learning from user data but also some discussion around privacy from a broader perspective."
---

During winter and spring I was fortunate enough to have a few occasions to talk about some of the work done at [Snips](https://snips.ai) on applying [privacy-enhancing technologies](https://en.wikipedia.org/wiki/Privacy-enhancing_technologies) in a start-up building privacy-aware machine learning systems for mobile devices. 

These were mainly centered around the [*Secure Distributed Aggregator*](https://github.com/snipsco/sda) (SDA) for learning from user data distributed on mobile devices in a privacy-preserving manner, i.e. without learning any individual data only the final aggregation, but there was also room for discussion around privacy from a broader perspective, including how it has played into decisions made by the company.


# What Privacy Has Meant For Snips

Given at the workshop on [*Privacy in Statistical Analysis (PSA'17)*](http://wwwf.imperial.ac.uk/~nadams/events/ic-rss2017/ic-rss2017.html), this invited [talk](https://github.com/mortendahl/privateml/raw/master/talks/PSA17-slides.pdf) aimed at giving an industrial perspective on privacy, including how it has played a role at Snips from its beginning. To this end the talk was divided into four areas where privacy had been involved, three of which briefly discussed below.

### Accessing Data
Access to personal data was essential for the success of its first mobile app, so to ensure that this was given the company decided to earn users' trust by focusing on privacy. To this end, it was decided to keep all data locally on users' devices and do the processing there instead of on company servers. 

These on-device privacy solutions have the extra benefit of being easy to explain, and may have accounted for the high percentage of users willing to give the mobile app access to sensitive information such as emails, chats, location tracking, and even screen content.

### Protecting the Company
By the principle of [*Data is a Toxic Asset*](https://www.schneier.com/blog/archives/2016/03/data_is_a_toxic.html), not storing any user data means less to worry about if company servers are ever compromised. However, some services hosted by third parties, including the company, may build up a set of metadata that in itself could reveal something about the users and e.g. damage reputation. One such example is *point-of-interest* services where a user reveals his location in order to obtain e.g. a list of nearby restaurants.

Powerful cryptographic techniques, such as the [Tor network](https://www.torproject.org/) and [private information retrieval](https://en.wikipedia.org/wiki/Private_information_retrieval), may make it possible for companies to make private versions of these services, yet also impose a significant overhead. Instead, by assuming that the company is generally honest, a more efficient compromise can be reached by shifting the focus from deliberate malicious behaviour to easier problems such as accidental storing or logging. 

One concrete approach taken for this was to strip sensitive information at the server entry point so that it was never exposed to subcomponents.

### Learning from Data
While it is great for user privacy to only have locally stored data sets, it is also relevant for both users and the company to get insights from these, for instance as a way of making cross-user recommendations or getting model feedback.

The key to this contradiction is that often there is no need to share individual data as long as a global view can be computed. A brief comparison between techniques was made, including:

- **sensor networks**: high performance but requires a lot of coordination between users

- **differential privacy**: high performance and strong privacy guarantees, but a lot of data is needed for the signal to overcome the noise

- **homomorphic encryption**: flexible and explainable, but still not very efficient and has the issue of who's holding the decryption keys

- **multi-party computation**: flexible and decent performance, but requires several players to distribute trust to

and concluding with the specialised multi-party computation protocol underlying SDA and further detailed below.


# Private Data Aggregation on a Budget

Given at the workshop on [*Theory and Practice of Multi-Party Computation (TPMPC'17)*](http://www.multipartycomputation.com/tpmpc-2017), this [talk](https://github.com/mortendahl/privateml/raw/master/talks/TPMPC17-slides.pdf) was technical in nature in that it presented the [SDA protocol](https://eprint.iacr.org/2017/643), but also aimed at illustrating the problem that a company may experience when wanting to solve a privacy problem by employing a secure multi-party computation (MPC) protocol: namely, that it may find itself to be the only party that is naturally motivated to invest resources into it. 

Moreover, to remain open to as many potential other parties as possible, it is interesting to minimise the requirements on these in terms of computation, communication, and coordination. By doing so parties running e.g. mobile devices or web browsers may be considered. These concerns however, are not always considered in typical MPC protocols.

### Community-based MPC
To this end SDA presents a simple but concrete proposal in a *community-based model* where members from a community are used as parties. 

These parties only have to make a minimum of investment as most of the computation is out-sourced to the company and very little coordination is required between the selected members. Furthermore, a mechanism for distributing work is also presented that allows for lowering the individual load by involving more members. 

The result is a practical protocol for *aggregating high-dimensional vectors* that is suitable for a single company with a community of sporadic members.

### Applications
Concrete and realistic applications was also considered, including analytics, surveys, and place discovery based on users' location history.

As illustrated, the load on community members in these applications were low enough to be reasonably run on mobile phones and even web browsers.

This work was also presented at [*Private Multi-Party Machine Learning (PMPML'16)*](https://pmpml.github.io/PMPML16/) in the form of a [poster](https://github.com/mortendahl/privateml/raw/master/talks/PMPML16-poster.pdf).
