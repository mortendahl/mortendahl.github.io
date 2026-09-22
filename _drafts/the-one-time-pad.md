---
layout:     post
title:      "The One-Time Pad"
subtitle:   "Perfect Secrecy and Modern Uses"
date:       2018-08-04 12:00:00
author:     "Morten Dahl"
header-img: "assets/aarhus.jpg"
---

<em><strong>TL;DR:</strong> The one-time pad is perhaps considered a toy encryption scheme that has strong secrecy guarantees but no-one in their right mind would use in practice. However, modern cryptography makes several references to it in various forms and knowing why and where it works helps understand many widely-used schemes and protocols.</em> 

# Work pad

Audience is as you discovered: "hackers" (software engineers) instead of cryptographers.

We want to refer to this blog post later when we talk about why ElGamal is secure (one component looks like a one time pad, but has more structure that allows )

My rough thoughts on the flow are something like, with Python code sprinkled in to illustrate from a practical perspective:

- applications:
    - build on previous code and show how a PRG can be used as a way to efficiently agree on a huge "table" ahead of time by knowing just a seed; point out that while the pad gives perfect security, the PRG introduces computational security; let's discuss which PRG to use: maybe we use a library or maybe we use a toy one that's easy to build from scratch and understand
    - mention that key exchange can be used to agree on a secret for the PRG, but siply say that this will be the subject of a future post

TODO:
- more concrete examples of what the one time pad looks like in various rings
- is Shamir sharing an example of the OTP in a polynomial field?

# Introduction

The one-time pad has the distinction of being the only encryption scheme that is provably unbreakable — not just hard to break with today's computers, but impossible to break with any computer, now or in the future. Yet it rarely appears in practice, and in courses on cryptography it is often covered briefly before being set aside in favour of more "realistic" schemes.

That dismissal misses something important. The one-time pad is not just a historical curiosity or a pedagogical warm-up: it shows up, in one form or another, in stream ciphers, in secret sharing, and in the security arguments for schemes like ElGamal. Knowing why it works and where its limits lie pays dividends throughout.

In this post we cover the basics of the scheme and why it achieves perfect secrecy, what goes wrong when a pad is reused, and how a pseudorandom generator can turn it into a practical encrypted channel. Along the way we will see that the scheme's apparent impracticality stems not from any weakness in the cryptography, but from a single logistical constraint: the pad must be as long as the message, and must itself be distributed securely in advance.

# The Scheme

To encrypt a plaintext `x` we sample a uniformly random pad `r` and compute the ciphertext `c = (x + r) % Q` for some integer `Q`. To decrypt, the holder of `r` computes `x = (c - r) % Q` to recover the original plaintext. In other words, the scheme requires nothing more than addition and subtraction modulo `Q` — a [ring](https://en.wikipedia.org/wiki/Ring_(mathematics)).

```python
Q = 100

def encrypt(plaintext, pad):
    return (plaintext + pad) % Q

def decrypt(ciphertext, pad):
    return (ciphertext - pad) % Q
```

The only constraint on `Q` is that it must be large enough to represent all possible plaintexts relevant to the application: if messages are numbers from `0` to `99` then `Q = 100` works, while `Q = 2` means we can only encrypt a single bit. For illustration we will use `Q = 100` throughout.

As a concrete example, say we wish to encrypt `x = 42`. We sample a random pad, say `r = 73`, and compute `c = encrypt(42, 73) = (42 + 73) % 100 = 15`. Given `c = 15` and `r = 73`, decryption gives back `decrypt(15, 73) = (15 - 73) % 100 = 42`.

The perhaps more familiar XOR-based one-time pad is exactly this scheme applied bit by bit with `Q = 2`, or byte by byte with `Q = 256`.

# Security

The security of the one-time pad follows from a surprisingly simple observation: for any ciphertext `c` and any plaintext `x`, there is exactly one pad `r` that could have produced it, namely `r = (c - x) % Q`. This means that no matter what value of `x` an attacker guesses, there is always a valid explanation for the ciphertext — it just requires a different pad.

We can verify this in code: given a ciphertext `c = 2`, we enumerate all plaintexts and find the unique pad that explains each one.

```python
c = 2

for x in range(Q):
    r = (c - x) % Q
    assert encrypt(x, r) == c
    print(f"plaintext={x}, pad={r}")
```

```
plaintext=0, pad=2
plaintext=1, pad=1
plaintext=2, pad=0
plaintext=3, pad=9
plaintext=4, pad=8
plaintext=5, pad=7
plaintext=6, pad=6
plaintext=7, pad=5
plaintext=8, pad=4
plaintext=9, pad=3
```

Since the pad is chosen uniformly at random, each row is equally likely. This is where uniformity matters: if some pad values were more probable than others, then the plaintexts associated with those pads would also become more probable, giving the attacker a statistical edge. Uniformity is precisely what ensures every row has the same weight, so an attacker who sees `c = 2` has no reason to prefer any plaintext over any other — every one is equally consistent with the ciphertext. This is what is formally known as [perfect secrecy](https://en.wikipedia.org/wiki/Information-theoretic_security): `P(plaintext | ciphertext) == P(plaintext)`, meaning the probability of any particular plaintext does not change upon observing the ciphertext.

The intuition is this: the best any encryption scheme can do is to not reveal any *additional* information beyond what the attacker already knew. Perfect secrecy achieves exactly that — the ciphertext is statistically independent of the plaintext.

Finally, note that `Q` has no effect on security. The one-pad-per-plaintext argument holds for any ring size, so it is the application that dictates the choice of `Q`.

## What goes wrong if the pad is used more than once

If the same pad is used to encrypt two different messages, the scheme breaks. Suppose an attacker observes two ciphertexts encrypted under the same pad `r`. Subtracting one from the other, `r` cancels out entirely:

```python
r = 73

c1 = encrypt(42, r)  # = 15
c2 = encrypt(17, r)  # = 90

leak = (c1 - c2) % Q  # = 25 = (42 - 17) % Q
```

The attacker now knows `x1 - x2` — the difference between the two plaintexts — without knowing `r` or either message individually. This is no longer perfect secrecy: the ciphertext pair has revealed information. And if one of the plaintexts happens to be known, the other is revealed completely:

```python
x1 = 42  # known to the attacker

x2 = (x1 - leak) % Q  # = 17
```

In practice, with structured messages such as natural language text, knowing the difference between two plaintexts is often enough to recover both even without knowing either outright.

This is the core practical limitation of the one-time pad: every pad must be used exactly once, and the total length of pads consumed grows with the total length of all messages ever sent. We return to this in the next section.

# Sending Encrypted Messages

Say Alice wants to send encrypted messages to Bob. The simplest approach is for them to agree ahead of time on a table of random pads — one per message — and number them so each message uses a fresh one.

```python
import secrets

# generated once and shared securely between sender and receiver
pad_table = [secrets.randbelow(Q) for _ in range(1000)]

def send(plaintext, index):
    pad = pad_table[index]
    ciphertext = encrypt(plaintext, pad)
    return ciphertext, index

def receive(ciphertext, index):
    pad = pad_table[index]
    return decrypt(ciphertext, pad)
```

This works, and the security follows directly from what we saw above: as long as no pad is used twice, every message is perfectly secret. But there is an obvious problem: the pad table must be at least as long as all messages Alice and Bob will ever send, and it must itself be distributed securely before any communication can take place. In other words, to solve the problem of exchanging secrets, we first need to exchange secrets.

## Stretching the Pad with a PRG

A [pseudorandom generator](https://en.wikipedia.org/wiki/Pseudorandom_generator) (PRG) offers a way out. Instead of storing a large table of random pads, Alice and Bob agree only on a short `secret`. For each message, a fresh pad is derived by feeding the secret and a unique `nonce` — a number used exactly once — into the PRG:

```python
def send(plaintext, nonce):
    pad = prg(secret, nonce)
    ciphertext = encrypt(plaintext, pad)
    return ciphertext, nonce

def receive(ciphertext, nonce):
    pad = prg(secret, nonce)
    return decrypt(ciphertext, pad)
```

The `nonce` plays the same role as the index in the table above: it ensures that different messages get different pads. In practice it can simply be a counter, incremented with each message. And as before, using the same `nonce` twice under the same `secret` produces the same pad twice — so the pad reuse attack from the previous section applies unchanged.

The scheme now requires only that Alice and Bob share a short `secret` — say 32 bytes — rather than a table that grows without bound. But this convenience comes at a cost: we have given up perfect secrecy. The security of the scheme now rests on the PRG being hard to distinguish from a truly random function. An attacker with unbounded computation could in principle try all possible secrets and check which produces pads consistent with the observed ciphertexts; the scheme is only secure against attackers who cannot feasibly do this. This is known as [computational security](https://en.wikipedia.org/wiki/Computational_hardness_assumption), as opposed to the information-theoretic guarantee of the original one-time pad.

The implementation of `prg` is in the associated notebook. In practice it should be instantiated with a well-studied construction such as AES in counter mode or ChaCha20.

The remaining question — how Alice and Bob agree on `secret` in the first place, without meeting in person — is the subject of a future post on key exchange.

# Other Applications

The one-time pad shows up in a surprising number of places once you know to look for it.

The [additive secret sharing scheme](/2017/06/04/secret-sharing-part1/#additive-sharing) is in essence a multi-party one-time pad. To share a secret `x` among three parties, we pick two uniformly random values `x0` and `x1` and set `x2 = (x - x0 - x1) % Q`, giving each party one share. The security argument is identical to the one we made above: for any guess at `x`, there is exactly one value of `x2` consistent with `x0` and `x1`, so any two shares together reveal nothing about `x`. The one-time pad and additive secret sharing are really the same idea, one framed as encryption and the other as distribution of trust.

The one-time pad also appears inside the security arguments of more advanced encryption schemes. In [ElGamal encryption](https://en.wikipedia.org/wiki/ElGamal_encryption), for instance, one component of the ciphertext takes the form `c = m * k`, where `k` is a value that — under a standard hardness assumption — looks uniformly random to anyone without the decryption key. The argument that this hides `m` is exactly the one-time pad argument in multiplicative form: for any guess at `m`, there is a unique `k` that explains `c`, so `c` reveals nothing about `m`. We explore this in more detail in a future post on the principles of encryption.
