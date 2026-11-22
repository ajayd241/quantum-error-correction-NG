# P2: Belief Propagation (core decoder #2)

**Time box:** 1:00  **Depends on:** P0  **Unlocks:** P3

## Goal
Implement normalized min-sum BP on the check matrix `H`, vectorized across edges and shots. Report how often it converges and what accuracy it gets. Its posterior beliefs (LLRs) are the raw material for P3.

## Concept primer (tell the human this before coding)
Picture a room of people passing notes.
- Each **fault** (a column of `H`) starts with a hunch: "I'm unlikely, only probability p."
- Each **detector** (a row) knows one fact: "an odd/even number of my faults happened."
- Each round, every detector tells each of its faults: "based on what the *others* told me, here's how likely you are." Each fault combines its hunch with what its other detectors said and reports back.
- After a few rounds, every fault has a **belief**. If the beliefs explain the syndrome exactly, BP has converged.

All notes are passed **at the same time**, so it's ideal for GPUs and FPGAs.

**Why BP struggles on surface codes:** *degeneracy*. Many different errors produce the same syndrome and are equally good fixes. BP tries to pick a single answer, gets torn between two equally likely ones, and its beliefs oscillate or settle at 50/50. The surface code's graph also has lots of short loops, which break BP's assumptions. That's why we pair it with UF in P3.

**Prediction exercise:** *"Will BP alone beat UF?"* (Expected: no.)

## Math (LLR convention: λ = ln P(0)/P(1), positive = "probably no error")
- Prior: `λ_j = ln((1 − p_j)/p_j)`
- Variable → check: `μ_{j→i} = λ_j + Σ_{i' ∈ N(j)\i} ν_{i'→j}`
- Check → variable (normalized min-sum): `ν_{i→j} = (−1)^{s_i} · α · Π_{j' ∈ N(i)\j} sign(μ_{j'→i}) · min_{j' ∈ N(i)\j} |μ_{j'→i}|`
- Posterior: `Λ_j = λ_j + Σ_{i ∈ N(j)} ν_{i→j}`. Hard decision: `ê_j = [Λ_j < 0]`
- Stop when `H ê = s (mod 2)` or after `max_iter` iterations
- Prediction: `L ê (mod 2)`

## Checklist
### Implementation (`bp.py`)
- [x] Edge list of the Tanner graph from `H`: `chk[e]`, `var[e]`, sorted by check (for `reduceat`) plus a permutation sorted by variable
- [x] Messages as `float64[S, n_edges]`, so the batch dimension is the shots
- [x] Check update: per-check min1, min2, argmin and sign product via `np.minimum.reduceat` / masking. "Min of the others" = min2 at the argmin position, min1 everywhere else
- [x] Variable update via `np.add.reduceat` on the variable-sorted order
- [x] Per-shot convergence mask; freeze converged shots (or just keep iterating and record the first convergence iteration)
- [x] Params: `max_iter` (default 30), `alpha` (default 0.75; try 0.625 and 1.0)
- [x] Return the prediction, plus `self.last_converged: bool[S]`, `self.last_iters: int[S]` and `self.last_posterior_llr: float[S, M]` (needed by P3)
- [x] `stats()` → convergence rate, mean iterations

### Tests (`test_bp.py`)
- [x] Zero syndrome → converges at iteration 0 or 1, zero prediction
- [x] 5-bit repetition code (a tree graph, where BP is exact): every single error is decoded correctly
- [x] Hand-computed one-iteration message check on a 2×3 toy `H`
- [x] Batched result == looped-per-shot result (bit-for-bit)
- [x] Single-fault rate on d=3 reported (not asserted to be 100%)

### Results
- [x] Add `bp` to the sweep
- [x] Plot: convergence rate vs p for each d
- [x] Small α sweep at one (d, p); record the best α in the decision log
- [x] Throughput: shots/s with batch size 1 vs 1,000 (the vectorization story)

## Acceptance criteria
1. Exact on the tree-like toy code.
2. Batched == looped.
3. Convergence rate and LER reported for d ∈ {3,5,7}. A *poor* LER is acceptable and expected.

## Agent notes
- Clip LLRs to ±50 to avoid overflow.
- Variables with degree 1 are fine. Checks with degree 1 are fine (min over an empty set = +∞ → use a large constant).
- Don't use the decomposed graph's merged edges for BP. Use `CheckMatrices` (one variable per DEM error mechanism), because that keeps the correlations between X and Z (hyperedges) that make BP useful.
- If time is short, skip the α sweep and use 0.75.

## Talk material to capture
- The "passing notes" picture: one check, three faults, arrows.
- Convergence rate vs p, as the motivation for belief-find.
- Batch-size throughput bar chart.

## Kick-off prompt
> Read CLAUDE.md and docs/phases/P2_belief_propagation.md. Explain BP with the passing-notes picture and explain why degeneracy hurts it. Then implement batched min-sum BP with its tests (tree-code exactness and batched == looped first). Report the convergence rate and LER for d=3,5,7. Stop at the acceptance criteria.
