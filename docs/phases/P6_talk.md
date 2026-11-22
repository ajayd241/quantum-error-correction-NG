# P6: 10-Minute Talk

**Time box:** 0:45  **Depends on:** P5 figures

## Story arc (one sentence)
*"Decoders trade accuracy for speed and parallelism. We built three of them from scratch, measured that trade-off against the gold standard, and found how little BP it takes to make a fast, parallel decoder more accurate."*

## Slide outline (≈ 10 slides, 10:00)
| # | Slide | Time | Visual |
|---|---|---|---|
| 1 | Title + the one-sentence story | 0:20 | — |
| 2 | Why decoders matter: noisy qubits, ~1 µs per round, backlog problem | 0:50 | surface-code grid with a lit syndrome |
| 3 | Decoding = pairing defects (the 7-bit chain → 2D grid) | 1:00 | repetition chain + matching picture |
| 4 | Our setup: stim → DEM → one harness, one interface, five decoders | 0:40 | architecture diagram |
| 5 | Union-Find: grow, merge, peel | 1:20 | 3-frame cluster picture |
| 6 | BP: passing notes in parallel, and why degeneracy hurts | 1:00 | Tanner-graph cartoon + convergence plot |
| 7 | Belief-find (credit Higgott et al.) + our iteration-budget study | 1:30 | Plot A (LER vs k) |
| 8 | Results: thresholds and LER comparison | 1:30 | `ler_vs_p_d5_all.png` + threshold table |
| 9 | Speed and parallelism: latency, batch throughput, worker speed-up | 1:00 | latency + speed-up plots |
| 10 | Takeaways, limitations, future work, repo link/QR | 0:50 | — |

## Checklist
- [ ] Slides built from P5 figures (no screenshots of terminals)
- [ ] Every slide has one message, written as its title (e.g. "4 BP iterations recover most of the gain")
- [ ] Credit slide/footnote for belief-find and for the reference review
- [ ] Repo URL + QR code on the last slide
- [ ] One timed rehearsal, ≤ 10:00
- [ ] Backup: figures saved as PNGs in case the slides fail

## Likely judge questions (prepare one-line answers)
- *Why not just use PyMatching?* It's sequential and hard to put on hardware. UF and BP are local and parallel.
- *Is belief-find yours?* No, it's from Higgott et al. 2023. Our contribution is the iteration-budget and fast-path latency study, plus a clean from-scratch implementation.
- *Are your latencies real-time?* No. It's Python. They show relative cost and scaling. Real-time needs FPGA/ASIC.
- *Why is BP alone bad?* Degeneracy and short loops in the surface-code graph.
- *What would you do next?* Sliding-window decoding, a numba/C++ UF, an FPGA-style cycle count for UF, and other noise models.

## Kick-off prompt
> Read docs/HACKATHON.md, docs/PROGRESS.md and docs/phases/P6_talk.md. Draft the 10-slide deck from the P5 figures, with message-style titles and speaker notes of ≤ 60 words each, then help me rehearse with a timer.
