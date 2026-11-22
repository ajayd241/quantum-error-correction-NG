# PROGRESS.md (living tracker: agents update this)

## Status
| Phase | Status | Key result |
|---|---|---|
| P0 Harness + baseline | ✅ done | PyMatching threshold ≈ 0.695%; single-fault 100% |
| P1 Union-Find | ✅ done | Weighted threshold ≈ 0.658%; single-fault 100%; max LER ratio 1.22× PM |
| P2 Belief Propagation | ✅ done | d=5, p=0.005: LER 7.165%, convergence 64.66%; batch speed-up 1.31× |
| P3 Belief-find + budget study | ✅ done | k=2 matched k=32 accuracy at 50% of mean p99 latency; BF significantly beat UF |
| P4 Our MWPM (stretch) | ⬜ not started | |
| P5 Metrics + repo | ⬜ not started | |
| P6 Talk | ⬜ not started | |

Legend: ⬜ not started · 🟨 in progress · ✅ done · ✂️ cut

## Results snapshot (fill as phases finish)
| Decoder | Threshold est. | LER/shot @ d=5, p=0.005 | p50 / p99 latency @ d=5 (µs) | Notes |
|---|---|---|---|---|
| PyMatching (ref) | 0.695% (pairwise 0.688–0.699%) | 0.0137 | 9.50 / 35.10 | 20k shots; seed 15459343435847105904 |
| UF (unweighted) | 0.539% | 0.0197 | 467.02 / 952.47 | weighted UF wins at all 24 sweep points |
| UF (weighted) | 0.658% | 0.0153 | 543.40 / 1321.15 | 1.12× PM LER; growth rounds mean/p99 7.06/15 |
| BP (min-sum) | no crossing in tested grid | 0.07165 | 1525.65 / 8749.08 | α=1.0; convergence rate 64.66% |
| Belief-find | budget study complete | 0.0118 | 8381.40 / 11427.77 | 5k shots; fast-path fraction 34.96% |
| MWPM (networkx) | | | | d ≤ 5 only |

## Log
<!-- One line per meaningful step: YYYY-MM-DD HH:MM | phase | what changed | numbers -->
- 2026-11-21 | planning | Agent assets created (HACKATHON, IMPLEMENTATION_PLAN, phases P0–P6, CLAUDE.md). Learning: Lesson 1 (repetition code) done; Lessons 2–6 will be taught inline during P0–P2.
- 2026-11-21 | planning | Pre-flight check (stim 1.x, PyMatching 2.x), p=0.005, rounds=d. d=3: D=24, 286 mechanisms → 78 merged edges. d=5: D=120, 1953 mechanisms → 502 edges. 0 observable conflicts. PyMatching single-fault test = 100% at d=3 and d=5, so the test design is valid.
- 2026-11-21 | P0 | Created `p0-harness`, added the `qecdec` src-layout package, and installed development dependencies. Versions: stim 1.16.0, PyMatching 2.4.0; host Python 3.9.6.
- 2026-11-21 | P0 Checkpoint A | Built circuit/DEM infrastructure and PyMatching reference. At p=0.005, rounds=d: d=3 has D=24, 286 mechanisms, 78 merged edges; d=5 has D=120, 1953 mechanisms, 502 merged edges; 0 observable conflicts. Single-fault pass rate 100% at both distances; fast suite 8 passed.
- 2026-11-22 | P0 Checkpoint B | PyMatching baseline completed: d=3,5,7; 8 p-values from 0.002 to 0.015; 20k shots/point; base seed 1. Pairwise log-log crossings: 0.699%, 0.695%, 0.688% (median threshold ≈ 0.695%). At p=0.002, LER d=3/d=7 = 0.0028/0.0003; at p=0.015 = 0.1089/0.2704. Talk point: added distance suppresses logical chains below threshold but adds harmful fault opportunities above it.
- 2026-11-22 | P1 unweighted UF | Implemented half-edge cluster growth, disjoint-set merging, and BFS forest peeling. Single-fault pass rate: d=3 286/286, d=5 1953/1953; correction-syndrome invariant: 1,000/1,000 sampled d=3 shots; focused suite 9 passed.
- 2026-11-22 | P1 weighted UF | Added likelihood-weighted growth with median integer edge length 10 and event jumps. Both UF variants pass every single fault at d=3/5 and the 1,000-shot peeling invariant; weighted UF agrees with PyMatching on ≥99% of 10k low-noise d=5 shots. At p=0.003, weighted/PM failure ratios were 0.98× (d=3) and 1.06× (d=5), while weighted beat unweighted by 24–26%.
- 2026-11-22 | P1 sweep | Completed 72 points (PyMatching, weighted UF, unweighted UF; d=3,5,7; 8 p-values; 20k shots/point). Pairwise-crossing medians: PM 0.695%, weighted UF 0.658%, unweighted UF 0.539%; weighted beat unweighted at all 24 matched points and its worst failure-count ratio to PM was 1.22×. At d=5, p=0.005: weighted LER 0.0153, p50/p99 543.40/1321.15 µs; unweighted 0.0197, 467.02/952.47 µs. Weighted growth mean/p99 at p≈0.00474: d=3 3.97/10, d=5 6.97/15, d=7 8.25/16. Talk point: probability-aware growth recovers most of MWPM's accuracy while retaining local, parallel cluster operations, but this pure-Python reference is much slower than optimized PyMatching.
- 2026-11-22 | P2 BP core | Implemented batched normalized min-sum BP over whole DEM mechanisms, with deterministic chunking and active-shot compaction. Tree-code exactness and batch/loop bit-equivalence pass. At d=3, p=0.005, logical single-fault accuracy was 266/286 (93.0%) and hard-decision convergence was 178/286 (62.2%).
- 2026-11-22 | P2 alpha + throughput | On 2,000 paired d=5, p=0.005 shots (seed 20261122), α=1.0 beat 0.75 and 0.625: LER 0.0665 vs 0.0955/0.1705 and convergence 65.95% vs 52.95%/35.70%. On the first 1,000 shots, batch=1,000 delivered 342 shots/s vs 261 shots/s at batch=1 (1.31×); predictions were identical.
- 2026-11-22 | P2 sweep | Completed 24 BP points at α=1.0 (d=3,5,7; 8 p-values; 1k shots/point; base seed 1), plus a 20k-shot d=5, p=0.005 point. At p≈0.00474, LER/convergence were d=3 0.032/92.7%, d=5 0.070/66.3%, d=7 0.147/23.0%. At exact d=5, p=0.005: LER 0.07165, convergence 64.66%, p50/p99 1525.65/8749.08 µs. Full suite: 41 passed. Talk point: BP's local updates are parallel and produce useful soft beliefs, but loops and degeneracy make standalone hard decisions worse as distance grows; belief-find uses UF to finish the job.
- 2026-11-22 | P3 belief-find core | Implemented BP-posterior XOR reweighting with sparse edge×mechanism incidence, per-shot UF weights at a fixed scale, and the optional BP-converged fast path. k=0 matched weighted UF bit-for-bit on 2,000 shots; default α=0.625 corrected all 286 d=3 single faults. At d=5, p=0.005 on 5,000 paired shots (seed 20261123), BF had 52 failures (LER 1.04% ± 0.14%) versus UF's 71 (1.42% ± 0.17%); the one-sigma intervals do not overlap.
- 2026-11-22 | P3 budget study | Completed 96 rows: k={0,1,2,4,8,16,32}, fast path on/off, d={3,5,7}, p={0.0035,0.00625}, 1,000 shots/point, base seed 3, with UF/PyMatching baselines. Finding: k=2 matched k=32 accuracy within sampling noise (83 vs 87 aggregate failures) using 50% of k=32's mean p99 latency; k=32 fast-path use fell from 72.5–84.8% at d=3 to 1.0–5.3% at d=7. Exact d=5, p=0.005 (5k shots, sweep seed 1): LER 1.18%, p50/p99 8381.40/11427.77 µs, fast-path fraction 34.96%. Full suite: 45 passed. Talk point: extra loopy-BP iterations are not monotonically helpful, and a BP fast path cuts average work more readily than p99 because the tail is dominated by shots that do not converge.

## Decision log
| # | Decision | Why |
|---|---|---|
| 1 | PyMatching (MWPM) is the reference, not one of "our" decoders | Gold-standard accuracy. Using it as the ruler keeps the comparison honest |
| 2 | All decoders built from one decomposed DEM (graph + check matrices) | One harness for every decoder; exact BP-to-edge mapping in P3 |
| 3 | Core decoders: UF → BP → belief-find; networkx MWPM is the stretch goal | Covers the accuracy/speed/parallelism spectrum and maximizes decoder count within the time |
| 4 | Novelty = the iteration-budget / fast-path latency study, not belief-find itself | Belief-find is published (Higgott et al., PRX 2023). Claim only what's new |
| 5 | Noise: stim uniform depolarizing, all four knobs = p; rotated memory Z; rounds = d | Standard, simple, comparable to the literature |
| 6 | Lessons 2–6 taught just-in-time during coding | The human wants to start building. Concepts stick better next to real stim output |
| 7 | Support Python 3.9+ and postpone annotation evaluation | The available system interpreter is Python 3.9.6; this preserves the specified interfaces without adding an environment-management dependency |
| 8 | Resolve equal-probability observable-mask conflicts with a deterministic bytewise tie-break | The plan says to keep the more likely mask but does not define ties; a stable tie-break prevents DEM iteration order from changing the graph |
| 9 | Derive each sweep seed from base seed, distance, and the exact float bits of p | A `(d, p)` point must retain the same noise seed when the probability grid is reordered or subset; all decoders at that point still share the seed |
| 10 | Scale UF log-likelihood weights so the median integer edge length is 10, then jump directly to the next growth event | This preserves useful probability resolution without paying one Python loop per support unit |
| 11 | Use α=1.0 for the P2 BP sweep while retaining the specified API default of 0.75 | It had the lowest paired failure count and highest convergence of α ∈ {0.625, 0.75, 1.0} at d=5, p=0.005; keeping the default preserves the planned interface and makes tuning explicit |
| 12 | Use α=0.625 as the belief-find default | On the P3 hybrid, it preserved 100% d=3 single-fault accuracy and still significantly beat UF near threshold; the standalone-BP optimum did not transfer directly to BP-guided UF |
