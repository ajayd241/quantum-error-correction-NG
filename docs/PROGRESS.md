# PROGRESS.md (living tracker: agents update this)

## Status
| Phase | Status | Key result |
|---|---|---|
| P0 Harness + baseline | ✅ done | PyMatching threshold ≈ 0.695%; single-fault 100% |
| P1 Union-Find | ⬜ not started | |
| P2 Belief Propagation | ⬜ not started | |
| P3 Belief-find + budget study | ⬜ not started | |
| P4 Our MWPM (stretch) | ⬜ not started | |
| P5 Metrics + repo | ⬜ not started | |
| P6 Talk | ⬜ not started | |

Legend: ⬜ not started · 🟨 in progress · ✅ done · ✂️ cut

## Results snapshot (fill as phases finish)
| Decoder | Threshold est. | LER/shot @ d=5, p=0.005 | p50 / p99 latency @ d=5 (µs) | Notes |
|---|---|---|---|---|
| PyMatching (ref) | 0.695% (pairwise 0.688–0.699%) | 0.0137 | 9.50 / 35.10 | 20k shots; seed 15459343435847105904 |
| UF (unweighted) | | | | |
| UF (weighted) | | | | |
| BP (min-sum) | | | | convergence rate: |
| Belief-find | | | | fast-path fraction: |
| MWPM (networkx) | | | | d ≤ 5 only |

## Log
<!-- One line per meaningful step: YYYY-MM-DD HH:MM | phase | what changed | numbers -->
- 2026-11-21 | planning | Agent assets created (HACKATHON, IMPLEMENTATION_PLAN, phases P0–P6, CLAUDE.md). Learning: Lesson 1 (repetition code) done; Lessons 2–6 will be taught inline during P0–P2.
- 2026-11-21 | planning | Pre-flight check (stim 1.x, PyMatching 2.x), p=0.005, rounds=d. d=3: D=24, 286 mechanisms → 78 merged edges. d=5: D=120, 1953 mechanisms → 502 edges. 0 observable conflicts. PyMatching single-fault test = 100% at d=3 and d=5, so the test design is valid.
- 2026-11-21 | P0 | Created `p0-harness`, added the `qecdec` src-layout package, and installed development dependencies. Versions: stim 1.16.0, PyMatching 2.4.0; host Python 3.9.6.
- 2026-11-21 | P0 Checkpoint A | Built circuit/DEM infrastructure and PyMatching reference. At p=0.005, rounds=d: d=3 has D=24, 286 mechanisms, 78 merged edges; d=5 has D=120, 1953 mechanisms, 502 merged edges; 0 observable conflicts. Single-fault pass rate 100% at both distances; fast suite 8 passed.
- 2026-11-22 | P0 Checkpoint B | PyMatching baseline completed: d=3,5,7; 8 p-values from 0.002 to 0.015; 20k shots/point; base seed 1. Pairwise log-log crossings: 0.699%, 0.695%, 0.688% (median threshold ≈ 0.695%). At p=0.002, LER d=3/d=7 = 0.0028/0.0003; at p=0.015 = 0.1089/0.2704. Talk point: added distance suppresses logical chains below threshold but adds harmful fault opportunities above it.

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
