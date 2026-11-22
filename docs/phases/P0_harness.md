# P0: Harness + PyMatching Baseline

**Time box:** 0:45  **Depends on:** nothing  **Unlocks:** every other phase

## Goal
Build the "measuring tape": generate surface-code noise with stim, turn the detector error model into our own graph and matrices, and measure PyMatching's logical error rate. Every later decoder plugs into this harness unchanged.

## Concept primer (tell the human this before coding)
- **Detectors.** A stabilizer measurement can be +1 or −1 even with no error, so the raw value isn't useful. What matters is whether it *changed* since the last round. A **detector** is "did this check's answer change?" A data-qubit error lights two detectors that are next to each other *in space*. A measurement error lights the same check in two consecutive rounds, so the two lit detectors are next to each other *in time*. Decoding becomes a 3D matching problem (2D space × time).
- **Detector error model (DEM).** stim looks at the noisy circuit and lists every possible fault as a line like: "with probability p, this fault flips detectors 4 and 11, and flips the logical observable." Each line is an edge in a graph, so this list is our decoding graph.
- **Observable.** The decoder doesn't need to recover the exact error. It only needs to answer one question: did the logical qubit get flipped? If the answer is right, we win.
- **Prediction exercise:** before plotting, ask *"At very low p, should d=7 beat d=3? At very high p?"*

## Inputs / outputs
- **In:** `(d, p, shots, seed)`
- **Out:** `results/baseline_pymatching.csv`, `results/figs/baseline_pymatching.png`

## Checklist
### Setup
- [x] `pyproject.toml` (package `qecdec`, src layout, `[dev]` extras: pytest, matplotlib, pandas)
- [x] `pip install -e ".[dev]"` works; `python -c "import stim, pymatching"` prints versions
- [x] `pytest.ini` / pyproject section registering the `slow` marker

### Circuits
- [x] `circuits.make_circuit(d, p, rounds=None, basis="z")` using `surface_code:rotated_memory_z` and all four noise knobs set to `p`
- [x] Docstring explains each of the four noise parameters in one line

### DEM → graph and matrices (`dem_graph.py`)
- [x] Parse `dem.flattened()` error instructions; split on separators
- [x] Build `DecodingGraph` with boundary node `D`, merged parallel edges, and an observable mask per edge
- [x] Build `CheckMatrices` (`H`, `L`, `priors`, `mech_to_edges`) from the **same** decomposed DEM
- [x] Count and expose `num_obs_conflicts` (parallel edges whose observable masks differ)
- [x] CSR adjacency arrays

### Decoder interface + reference
- [x] `base.Decoder` ABC exactly as in IMPLEMENTATION_PLAN §4
- [x] `decoders/pymatching_ref.py`: `pymatching.Matching.from_detector_error_model(dem)` and `decode_batch`
- [x] `decoders/__init__.py` REGISTRY

### Harness
- [x] `harness.sample()` with a seeded `compile_detector_sampler(seed=...)` and `separate_observables=True`
- [x] `harness.evaluate()` → `Result` (LER per shot, error bar, LER per round, latency mean/p50/p99)
- [x] `harness.sweep()` → CSV
- [ ] `scripts/run_sweep.py` CLI (`--decoders --distances --ps --shots --seed --out`)
- [ ] `scripts/plot_results.py`: log-log LER vs p with one colour per d, one line style per decoder, error bars

### Tests
- [x] `test_dem_graph.py`: D+1 nodes; all probs in (0, 0.5); every mechanism's H column matches its detector targets; `mech_to_edges` covers every edge
- [x] `test_harness.py`: zero syndrome → zero prediction (PyMatching); a fixed seed gives the same failures twice
- [x] Single-fault test for PyMatching (d = 3, 5): 100%
- [x] `@slow`: LER(d=5) < LER(d=3) at p = 0.002

### Results
- [ ] Baseline sweep: d ∈ {3,5,7}, 8 p-values, ≥ 20k shots
- [ ] Plot saved. The curves should cross somewhere around p ≈ 0.5–1% (the threshold)
- [ ] Numbers logged in `PROGRESS.md`

## Acceptance criteria
1. `pytest -q -m "not slow"` is green.
2. The single-fault test passes 100% for PyMatching at d=3 and d=5. This proves the DEM parsing *and* the test itself are right before we trust it on our own decoders.
3. The baseline plot shows curves for different d crossing at a threshold.

## Agent notes
- Print `dem` for d=3, rounds=1 once and show the human three lines of it. Seeing real DEM lines makes the concept click.
- Keep `dem_graph.py` free of decoder logic. It is shared infrastructure.
- If `num_obs_conflicts > 0`, don't panic: log it in PROGRESS.md and move on. PyMatching makes a similar choice when it merges edges.
- Don't optimize anything yet.

## Talk material to capture
- Screenshot of three DEM lines with annotations.
- Baseline LER vs p plot. This becomes the background for every later comparison.

## Kick-off prompt (paste into the coding agent)
> Read CLAUDE.md, docs/IMPLEMENTATION_PLAN.md and docs/phases/P0_harness.md. Explain the P0 concept primer to me in plain language, then implement P0 step by step, ticking the checklist as you go. Write the single-fault test early. Stop after the acceptance criteria pass and show me the baseline plot and the numbers.
