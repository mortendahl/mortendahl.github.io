---
layout:     post
title:      "Paillier Encryption, Part 3"
subtitle:   "Algebraic Interpretation and Security"
date:       2019-04-17 12:00:00
author:     "Morten Dahl"
header-img: "assets/paillier/autostereogram-hidden-treasures.jpg"
summary:    "Having seen the basics of Paillier encryption and how it may be used, in this third part of the series we look at why it is believe to be security. We also extend our understanding of the scheme through an algebraic interpretation, revealing an underlying mathematical structure that gives better insight into what is doing on."
---

<style>
img {
    margin-left: auto;
    margin-right: auto;
}
</style>

# Algebraic Interpretation

`(1 + n)**x == 1 + nx mod nn` and `(1 + n)**x == 1 + nx mod pp`

`(Zn2*, *) ~ (Zn, +) x (Zn*, *)`

HE
- `c1 * c2 == (x1, r1) * (x2, r2) == (x1+x2, r1*r2)`

- `c ^ k == (x, r) ^ k == (x * k, r^k)`

- `inv(c) == inv((x, r)) == (-x, r^-1)`

- `c * s^n == (x, r) * (0, s) == (x, r*s) == (x, t)`

dec
- `c^phi == (x*phi, r^phi) == (x*phi, 1) == g^(x*phi)`

# Security

<p style="align: center;">
<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-space-shuttle.jpeg" style="width: 85%;"/></a>
<br/><em>From World of Hidden 3D</em>
</p>

<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-mars-rover.jpeg" style="width: 85%; border: 2px solid black;"/></a>

<a href="http://www.hidden-3d.com/index.php?id=gallery&pk=116"><img src="/assets/paillier/autostereogram-hidden-treasures.jpg" style="width: 85%; border: 2px solid black;"/></a

If you are told [the secret](http://www.hidden-3d.com/how_to_view_stereogram.php) then you can see the pattern

like [autostereograms](https://en.wikipedia.org/wiki/Autostereogram) there's a hidden pattern; unlike them, discovering the underlying pattern via starring will simply take too long

In order to better understand the scheme, including its security, it's instructive to start with a fool-proof scheme and see how it compares.

<img src="/assets/paillier/nn.png" style="width: 45%"/>
<img src="/assets/paillier/nninverse.png" style="width: 45%"/>
<img src="/assets/paillier/nnstar.png" style="width: 45%"/>

Concretely, say you want to encrypt a value `x` from `Zn^2*`. One way of doing this is to pick a uniformly random value `s` also from `Zn2*` and multiply these together: `c = x * s mod n^2`. Since this is in fact the one-time pad (over a multiplicative group) **TODO TODO TODO really?** we get perfect secrecy, i.e. a theoretical guarantee that nothing whatsoever is leaked about `x` by `c` as long as the mask `s` remains unknown.

The problem with this scheme is when we want to decrypt `c` without knowing `s` (which would otherwise have to be communicated somehow). One might try to get rid of `s` by raising `c` to the order of `s`, which by TODO gives us `c ^ ord(s) == (x * s) ^ ord(s) == x^ord(s) * 1 == x^ord(s)`. This indeed removed `s`, but it is not clear how to extract `x` from `x^ord(s)`. In fact, in the case where `ord(s)` is `phi(n^2)` this is impossible.

So what if we 

let's see an example. say our modulus is `n^2` as in the Paillier scheme. these numbers we can convenient put into an `n` by `n` grid as follows.

(TODO)

if we then filter out those values that don't have multiplicative inverses we get the following grid

(TODO)

which shows that the remaining values are exactly those that do not share any factors with `n`. this means that an easy way to characterize these numbers is by 
`[x for x in range(NN) if gcd(x, N) == 1]`. moreover, multiplying any two values with multiplicative inverses implies that their product also has an multiplicative inverse; in other words, as long as we multiply numbers from `Zn*` we are guaranteeded to stay within `Zn*`.

To understand the scheme, not least in terms of security and how decryption works, it is useful to try to reconstruct the process of building it. moreover, some of the principles we'll see here are also used in more modern schemes, in particular around how security is argued. of course this likely involved much more back and forward originally, or perhaps even an entirely different line of thought -- short of asking Pascal Paillier himself we might never know. Nonetheless, let's go back to the late 1990s where RSA could very well have been a heavy inspiration.

the reasons for switching from computing mod `n` as in RSA to computing mod `n^2` instead of something else will become clearer later, but for now let's just say that in order to get a probabilistic scheme we have to let room for some randomness `r` (in ElGamal, another probabilistic scheme from roughly the same period does this by letting ciphertexts be pairs of values instead; however in Paillier we simply double the modulus).
