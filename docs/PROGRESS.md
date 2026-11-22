# PROGRESS.md (living tracker: agents update this)

## Status
| Phase | Status | Key result |
|---|---|---|
| P0 Harness + baseline | 🟨 in progress | Setup complete: stim 1.16.0, PyMatching 2.4.0 |
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
| PyMatching (ref) | | | | |
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
