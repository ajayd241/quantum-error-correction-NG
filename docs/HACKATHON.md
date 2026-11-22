# Hackathon Goal: Surface-Code Decoders

> **One-line pitch:** We built three surface-code decoders from scratch (Union-Find, Belief Propagation, and BP-guided Union-Find), benchmarked them against the industry-standard MWPM decoder (PyMatching), and measured how much accuracy each one buys versus how fast and how parallel it can run.

---

## 1. The challenge

| | |
|---|---|
| **Event** | Northrop Grumman Decoder Challenge |
| **Time budget** | 6–8 hours total |
| **Deliverable 1** | Public git repo: decoder code + tests that compute decoder metrics |
| **Deliverable 2** | ~10-minute presentation |
| **Reference reading** | deMarti iOlius et al., *Decoding algorithms for surface codes*, arXiv:2307.14989 (Quantum, 2024) |

### What "decoding" means here (30-second version)
A quantum computer stores one *logical* qubit across many noisy *physical* qubits (the **surface code**). We can't look at the qubits directly, but we can repeatedly measure parity checks. Errors light up the checks at the **ends** of error chains (these lit checks are called **defects**). A **decoder** takes the pattern of defects (the **syndrome**) and guesses which error happened, so it can be undone. It is judged on two things:
- **Accuracy:** how often its guess leaves a *logical* error behind.
- **Speed and parallelism:** real hardware produces syndromes every ~1 µs. A decoder that can't keep up is useless, however accurate it is.

---

## 2. Judging criteria → how we score on each

| Criterion | Our plan |
|---|---|
| **Presentation quality** | One clear story: *accuracy vs speed vs parallelism*. Every slide has one plot or one picture. |
| **Novelty** | Belief-find itself is published (see §5), so our new part is a study of the **BP iteration budget**: how few BP iterations buy most of the accuracy gain, plus a "BP-converged fast path" that skips UF. Both matter for real-time hardware. |
| **Parallelizability** | UF grows clusters with only local steps. BP updates every node at the same time and is vectorized across edges *and* shots. We measure scaling, not just claim it. |
| **Code quality** | One `Decoder` interface, one harness, typed and documented modules, pytest suite with physics-based tests (single-fault correction, k=0 equivalence, agreement with PyMatching). |
| **Number of decoders** | 3 core decoders of our own (UF, BP, BP+UF), plus a stretch decoder (our own MWPM via networkx), plus the PyMatching reference. |

---

## 3. The decoder lineup

| # | Decoder | Ours? | Role in the story |
|---|---|---|---|
| — | **PyMatching (MWPM)** | No (reference) | The ruler. Near-optimal accuracy for surface codes, but hard to parallelize. |
| 1 | **Union-Find (UF)** | Yes | Near-linear time, local operations, slightly less accurate. Has been implemented on FPGAs in the literature. |
| 2 | **Belief Propagation (min-sum)** | Yes | Fully parallel message passing. Weak on its own for surface codes because of *degeneracy* (many equally good explanations). We show this rather than hide it. |
| 3 | **BP-guided UF ("belief-find")** | Yes (our implementation) | BP's soft beliefs re-weight the graph and UF finishes the job. This is our headline result. |
| S | **Exact MWPM (networkx blossom)** | Yes (stretch) | Small d only. Proves we understand matching, and its answers should match PyMatching. |

---

## 4. Success criteria

**Minimum viable (must have):**
- [ ] Harness producing logical error rate (LER) vs physical error rate (p) for d = 3, 5, 7 with PyMatching
- [ ] Our UF decoder passing all tests, with its LER curve on the same plot
- [ ] Public repo with README and a green `pytest`
- [ ] 10-minute talk

**Target:**
- [ ] BP + belief-find working, with the threshold plot for all decoders
- [ ] Iteration-budget study (LER and latency vs BP iterations k)
- [ ] Latency table (mean and p99 per shot) for every decoder

**Stretch:**
- [ ] Our own exact MWPM for d = 3 (and d = 5 if fast enough)
- [ ] Shot-level parallel sweep (multiprocessing or `sinter`) with a speed-up plot

---

## 5. Honesty notes (state these in the talk)

- **Belief-find is not our invention.** BP-reweighted matching and UF ("belief-matching" / "belief-find") were introduced by Higgott, Bohdanowicz, Kubica, Flammia & Campbell, *Improved decoding of circuit noise and fragile boundaries of tailored surface codes*, PRX 13, 031007 (2023), arXiv:2203.04948. We implement it from scratch and credit it. What we add is the **iteration-budget / fast-path latency study**.
- **Related:** Caune et al., *Belief propagation as a partial decoder*, arXiv:2306.17142 (2023), uses BP as a first-stage pre-decoder. Cite it if we present the fast path.
- **Python is not hardware.** Our latencies show *relative* cost and scaling, not real-time feasibility. Say so.
- **Sanity ranges, not targets:** the PRX paper reports a circuit-noise threshold of about 0.82% for MWPM and about 0.94% for belief-matching. Our noise model (stim's uniform depolarizing) is different, so expect our MWPM threshold to land somewhere around 0.5–1%, UF slightly below it, and belief-find above UF.

---

## 6. Scope

**In scope:** rotated surface code memory experiments (Z basis), circuit-level noise from stim, d ∈ {3, 5, 7}, rounds = d.

**Out of scope (say "future work"):** lattice surgery, sliding-window / streaming decoding, real FPGA implementation, neural decoders, biased noise.

---

## 7. Timeline (6:45 planned, with buffer)

| Phase | File | Time | Output |
|---|---|---|---|
| P0 Harness + baseline | `phases/P0_harness.md` | 0:45 | PyMatching LER vs p plot |
| P1 Union-Find | `phases/P1_union_find.md` | 2:00 | UF decoder + tests |
| P2 Belief Propagation | `phases/P2_belief_propagation.md` | 1:00 | BP decoder + convergence stats |
| P3 Belief-find + study | `phases/P3_belief_find.md` | 0:45 | Hybrid decoder + iteration-budget plot |
| P4 Our MWPM (stretch) | `phases/P4_mwpm_stretch.md` | 0:30 | networkx MWPM for small d |
| P5 Metrics + repo | `phases/P5_metrics_repo.md` | 1:00 | Final plots, README, public repo |
| P6 Talk | `phases/P6_talk.md` | 0:45 | Slides + rehearsal |

**Cut order if we run late:** P4 first, then the fast-path part of P3, then BP α-tuning. **Never cut:** tests, the main threshold plot, or the README.

---

## 8. References
1. deMarti iOlius et al., *Decoding algorithms for surface codes*, arXiv:2307.14989.
2. Delfosse & Nickerson, *Almost-linear time decoding algorithm for topological codes*, Quantum 5, 595 (2021). (Union-Find)
3. Higgott et al., PRX 13, 031007 (2023), arXiv:2203.04948. (Belief-matching / belief-find)
4. Caune et al., arXiv:2306.17142. (BP as a partial decoder)
5. Higgott & Gidney, *Sparse Blossom*, arXiv:2303.15933. (PyMatching v2)
6. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021).
