---
layout:     post
title:      "The Zama Protocol, Part 1"
subtitle:   "Symbolic Execution and Coprocessors"
date:       2026-09-22 12:00:00
author:     "Morten Dahl"
header-img: "assets/aarhus.jpg"
---

<em><strong>TL;DR:</strong> The one-time pad is perhaps considered a toy encryption scheme that has strong secrecy guarantees but no-one in their right mind would use in practice. However, modern cryptography makes several references to it in various forms and knowing why and where it works helps understand many widely-used schemes and protocols.</em> 

# Work pad

Encrypted types & symbolic execution — a mini "EVM" that stores handles instead of values, dispatches ops onto ciphertexts, resolves them asynchronously

Rough content (on no particular order):
- Toy fhEVM protocol implementation, in Python.
- "Smart contracts" are Python classes.
- There are application contracts, and trusted fhEVM contracts.
- The "blockchain" consists of a number of smart contract class instances.
- For now, we only consider "honest" application contracts; when we talk about the ACL later, we'll also consider "malicious" application contracts.
- Transactions (to application contracts) are function calls on instances.
- The trusted contract is used to obtain handles to encrypted values, and to compute on them; each of these add a node to an implicit computational graph.
- The coprocessor maps a ciphertext to each node in the computational graph.
- The computaional graph can be built in many ways, but to match the actual fhEVM protocol, we say that new nodes are emitted one by one using "events".
- We call what happens on the blockchain for "symbolic computation" where handles are "symbolic values"; we call what happens on the coprocessor for "concrete computation"
- This paradigm is to split the heavy concrete computation from the lighter symbolic computation; in our case it's to move the heavy FHE computation offchain, but it could be applied in many other cases as well (list examples)
- The blockchain settles which computation must be performed; after that anyone can perform it; the coprocessor is just one such party
- The terms "Ssymbolic" and "concrete" values and execution come from symbolic protocol analysis; the term "handle" is inspired by the BPW model for mapping symbolic analysis to computational analysis in the UC model.
- "Zama’s fhEVM is a real-world, production implementation of the exact abstraction paradigm envisioned by computational soundness theorists. By converting complex computational objects (TFHE ciphertexts) into clean symbolic entities (bytes32 handles), developers can write secure, deterministic code on-chain, while the true cryptographic complexity runs safely underneath."
- This was heavily inspired by my previous research in academia on computational soundness: "The exact abstraction paradigm he researched during his PhD—shielding upper-level protocol logic from lower-level cryptographic complexity using a formal mapping—is precisely how the fhEVM functions today. By packaging homomorphic ciphertexts into secure, high-level bytes32 handles, his team has successfully implemented the exact type of universally composable symbolic framework he mathematically proved over a decade ago"
- While the blockchain provides consensus on the computational graph, the coprocessor must provide consensus on the mapping from symbolic values )handles) to concrete values (ciphertexts); we can do multiple things here, including running multiple coprocessor nodes that must reach consensus, or do sampling to catch a cheating coprocessor with high probability (how high?)
- The benefit of this separate of light and heavy computation, is that the heavy computation can be added as a layer on top of any blockchain without impacting performance, and without having to change the blockchain software

Audience is as you discovered: "hackers" (software engineers) instead of cryptographers.

For a future post, it could also be interesting to talk about security, including how to model the protocol in the UC framework (where the core protocol is perhaps one ideal functionality, and contracts running on top are indepedent ideal functionalities)

Table of content:
- introduction
- 

# Introduction

