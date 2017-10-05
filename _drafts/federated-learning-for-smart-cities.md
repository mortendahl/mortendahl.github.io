---
layout:     post
title:      "Federated Learning for Smart Cities"
subtitle:   "Crowdsourcing Location-based Data with Privacy"
date:       2016-04-02 12:00:00
author:     "Morten Dahl"
header-img: "img/post-bg-01.jpg"
---


the general idea is that 

we capture applications that can be expressed as local training followed by an aggregation in the form of a weighted linear combinations

# Federated Learning


# Capturing Location Data

- local operation done on mobile devices
- show Google API: tracking and geofences
- https://developer.android.com/training/location/receive-location-updates.html
- https://developer.apple.com/documentation/corelocation/getting_the_user_s_location


# Smart City Applications

Validate vs discover: train on public or private data.

give concrete performance estimates for each: "using building blocks below"



## Traffic count
- spots already known, perhaps from previous data; purpose is to validate and detect deviation; potentially short-lived experiments in real time
- physical counters are expensive, in limited number, and hard to move in real time
- virtual counters via geofences


## Flow into city
- geofencing
- potentially longer-running experiments
- 35 portes in Paris; could be public to keep dimension down but if longer running then maybe okay with 35x blow-up
- patterns not known; purpose is to discover patterns, potentially for further investigation
- path as north, east, south, west to keep dimension down
- weighted by counting number of contributions: public for technical reasons (Paillier)

  
## Hotspots
- where to put efforts in making biking safer
- find heavy bike-trafficked intersections in city
- use sketches to keep dimension down
- http://theory.stanford.edu/~tim/s15/l/l2.pdf


## Busy Hours
- Google Maps recommendations


# Private Aggregation



# Dump

SDA use cases
- assume we have community/network that will do MPC
- min-count sketch
- on test data
- crowd sourcing data
- mini golf
- Tirpitz
- train on sensitive data, go into production/get feedback on aggregated data

