# CLAUDE.md: Agent Notes for `qec-decoders`

**Every session:** read this file → `docs/PROGRESS.md` (where are we?) → the current phase file in `docs/phases/`.

## What this repo is
A hackathon entry for the Northrop Grumman Decoder Challenge: surface-code decoders written from scratch (Union-Find, BP, belief-find, plus a stretch MWPM), benchmarked against PyMatching. Goals and judging: `docs/HACKATHON.md`. Architecture and interfaces: `docs/IMPLEMENTATION_PLAN.md`.

## Who you're working with
The human is **new to quantum error correction** and is learning while building. Act as an expert researcher and a patient teacher.
- **Before each phase:** explain its core idea in plain language (≤ ~10 lines, one picture or analogy). Each phase file has a *Concept primer* to draw from.
- **After each non-trivial piece of code:** say in 2–3 sentences what it does and why.
- **Before cheap experiments:** ask the human to predict the outcome (each phase lists one *prediction exercise*).
- **Never hide bad results.** A decoder that underperforms is a talk point. Explain why.
- Define jargon the first time you use it (defect, detector, DEM, LLR, threshold…).

## Working rules
1. **One phase at a time.** Don't start phase N+1 until phase N's acceptance criteria pass, unless the human says so.
2. **Tick the checkboxes** in the phase file as you go. Append a dated line to `docs/PROGRESS.md` with what changed and the key numbers.
3. **One interface.** Every decoder subclasses `qecdec.base.Decoder`. The harness never special-cases a decoder.
4. **Don't touch the ruler.** `decoders/pymatching_ref.py` is the reference. Never tune it or change it to flatter our decoders.
5. **Tests before claims.** Run `pytest -q -m "not slow"` while iterating and `pytest -q` before saying a phase is done.
6. **Reproducible.** Seed every sampler and RNG. Every number in `results/` comes from a script plus a seed, never from a notebook cell.
7. **Code quality is judged.** Small typed functions, and docstrings that explain the *physics*, not just the code. No dead code, no commented-out blocks.
8. **Performance:** correct first, then numpy-vectorize, then profile, then (only if needed) numba with a pure-Python reference kept and tested.
9. **Dependencies:** stick to the list below. Note any addition in PROGRESS.md's decision log.
10. **Decisions:** record non-obvious choices and why in PROGRESS.md's decision log.
11. **Ask before anything outward-facing or irreversible:** creating or pushing to GitHub, deleting results, renaming the package.
12. **Credit honestly:** belief-find is from Higgott et al., PRX 13, 031007 (2023). Never describe it as our invention.

## Commands
```bash
pip install -e ".[dev]"
pytest -q -m "not slow"          # fast loop (< 30 s)
pytest -q                        # full suite
python scripts/run_sweep.py --decoders pymatching,uf --distances 3,5,7 --shots 20000 --seed 1 --out results/sweep.csv
python scripts/plot_results.py results/sweep.csv --out results/figs/
python scripts/run_budget_study.py --out results/budget.csv      # P3
```

## Dependencies
Core: `stim`, `pymatching`, `numpy`, `scipy`, `networkx`, `matplotlib`, `pandas`, `pytest`.
Optional: `sinter` (parallel sampling), `numba` (speed), `ldpc` (**cross-checks in tests only**, never as one of our decoders).

## Domain gotchas (read before touching DEM / decoder code)
- **`decompose_errors=True`.** Always build from the decomposed DEM. Graph decoders need edges with at most 2 detectors.
- **Separators.** One `error(p)` instruction can hold several components split by `^` (`target.is_separator()`). Each component becomes a graph edge with probability p. For BP, the whole instruction is **one** variable whose detector set is the XOR of its components.
- **Boundary.** A component with one detector is an edge to the virtual boundary node `D = num_detectors`. Clusters touching it are neutral. Root peeling trees at it. It never carries a defect flag.
- **Parallel edges.** Merge with `p = p1(1−p2) + p2(1−p1)`. If the observable masks differ, keep the more likely one's mask and count the conflict.
- **Weights.** `w = ln((1−p)/p)`. Clamp p to `[1e-15, 0.5−1e-12]`, because p ≥ 0.5 gives w ≤ 0 and breaks matching/UF.
- **LLR sign.** `λ = ln(P(no error)/P(error))`: positive means "probably fine". Hard decision is `Λ < 0`.
- **What decoders output.** `bool[K]` predicted observable flips, not the full error. A shot fails if the prediction ≠ the actual value for any observable.
- **Per-round LER.** `1 − (1 − P_shot)^(1/rounds)`. Report both per-shot and per-round.
- **Error bars.** `sqrt(P(1−P)/N)`. A point with fewer than ~100 failures is noisy: add shots rather than read the noise.
- **Single-fault test ★** For every mechanism j, the syndrome `H[:, j]` must decode to `L[:, j]`. This catches most bugs. MWPM-family decoders must pass 100%.
- **dtypes.** stim returns `bool` arrays (or bit-packed `uint8` with `bit_packed=True`). Be explicit and don't mix them.

## Definition of done (any phase)
- [ ] Phase checklist fully ticked
- [ ] `pytest -q` green
- [ ] Figures/CSVs regenerated by a script
- [ ] PROGRESS.md updated with the numbers
- [ ] The human got a 2–3 sentence plain-language summary, plus one "talk material" item saved

## When stuck (> 15 min on one bug)
1. Shrink: d=3, rounds=1, one shot, and print everything.
2. Draw the graph with the defects (networkx).
3. Compare against PyMatching on that exact shot.
4. Tell the human what you tried, and propose a cut or a simplification.
