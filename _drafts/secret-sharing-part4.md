---
layout:     post
title:      "Secret Sharing, Part 4"
subtitle:   "Application to Secure Computation"
date:       2017-06-20 12:00:00
header-img: "img/post-bg-03.jpg"
author:     "Morten Dahl"
twitter:    "mortendahlcs"
github:     "mortendahl"
linkedin:   "mortendahlcs"
---

motivation for using a prime field (as opposed to an extension field): it allows us to naturally use modular arithmetic to simulate bounded integer arithmetic, as useful for instance in secure computation. in the previous post we used prime fields, but all the algorithms there would carry directly over to an extension field (they will here as well, but again we focus on prime fields). binary extension fields would make the computations more efficient but are less suitable since it is less obvious which encoding/embedding to use in order to simulate integer arithmetic

