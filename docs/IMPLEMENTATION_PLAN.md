# Implementation Plan

This is the technical blueprint. `HACKATHON.md` covers *why*. This doc covers *what* and *how*. Each phase has its own checklist in `docs/phases/`.

---

## 1. Architecture

```
            ┌──────────────────────────────┐
  (d, p) ──▶│ circuits.make_circuit()      │  stim.Circuit (rotated_memory_z, rounds=d)
            └──────────────┬───────────────┘
               ┌───────────┴────────────┐
               ▼                        ▼
   compile_detector_sampler()   detector_error_model(decompose_errors=True)
               │                        │
     dets: bool[shots, D]               ▼
     obs:  bool[shots, K]     ┌──────────────────────┐
               │              │ dem_graph.build()    │ single source of truth
               │              │  ├─ DecodingGraph    │ → UF, MWPM, PyMatching
               │              │  └─ CheckMatrices    │ → BP
               │              └──────────┬───────────┘
               │                         ▼
               │              Decoder.from_dem(...)      (one interface)
               │                         │
               └──────────▶  decoder.decode_batch(dets) → pred: bool[shots, K]
                                         │
                                         ▼
                        harness.evaluate() → Result(LER, LER/round, err bar, latency)
                                         │
                                         ▼
                           results/*.csv → scripts/plot_results.py → figs
```

**Key design decision:** everything is derived from **one** decomposed detector error model (DEM). The matching graph uses its components as edges. BP's check matrix uses each whole error instruction as one variable (the XOR of its components). Because both come from the same object, mapping BP beliefs back onto graph edges (needed in P3) is exact rather than a guess.

---

## 2. Repo layout

```
qec-decoders/
├── CLAUDE.md                 # agent rules (read first)
├── AGENTS.md                 # pointer to CLAUDE.md for other agent tools
├── README.md                 # written in P5
├── pyproject.toml
├── src/qecdec/
│   ├── __init__.py
│   ├── circuits.py           # make_circuit(d, p, rounds=None, basis="z")
│   ├── dem_graph.py          # DecodingGraph, CheckMatrices, build_from_dem()
│   ├── base.py               # Decoder ABC
│   ├── harness.py            # sample(), evaluate(), sweep(), Result
│   └── decoders/
│       ├── __init__.py       # REGISTRY = {"pymatching": ..., "uf": ..., ...}
│       ├── pymatching_ref.py # reference: never tune this
│       ├── union_find.py
│       ├── bp.py
│       ├── belief_find.py
│       └── mwpm_nx.py        # stretch
├── scripts/
│   ├── run_sweep.py          # CLI → results/*.csv
│   └── plot_results.py       # CLI → results/figs/*.png
├── tests/
│   ├── conftest.py           # shared fixtures: small DEMs, rng
│   ├── test_dem_graph.py
│   ├── test_harness.py
│   ├── test_union_find.py
│   ├── test_bp.py
│   ├── test_belief_find.py
│   └── test_mwpm_nx.py
├── results/                  # csv + figs (committed: judges should see them)
└── docs/
    ├── HACKATHON.md
    ├── IMPLEMENTATION_PLAN.md
    ├── PROGRESS.md
    └── phases/P0..P6_*.md
```

---

## 3. Core data structures

```python
@dataclass(frozen=True)
class DecodingGraph:
    num_detectors: int          # D; the boundary node has index D
    edge_u: np.ndarray          # int32[E]
    edge_v: np.ndarray          # int32[E]  (== D for boundary edges)
    edge_prob: np.ndarray       # float64[E], merged probability
    edge_weight: np.ndarray     # float64[E], ln((1-p)/p)
    edge_obs: np.ndarray        # bool[E, K]: which observables this edge flips
    # adjacency in CSR form for fast neighbour loops
    adj_ptr: np.ndarray         # int32[D+2]
    adj_edges: np.ndarray       # int32[2E]  edge ids incident to each node

@dataclass(frozen=True)
class CheckMatrices:
    H: scipy.sparse.csr_matrix  # uint8[D, M]: detector × error mechanism
    L: scipy.sparse.csr_matrix  # uint8[K, M]: observable × error mechanism
    priors: np.ndarray          # float64[M]
    mech_to_edges: list[np.ndarray]  # mechanism j → graph edge ids of its components (for P3)
```

`build_from_dem(dem: stim.DetectorErrorModel) -> tuple[DecodingGraph, CheckMatrices]`

Rules (also listed in CLAUDE.md "gotchas"):
- Iterate `dem.flattened()`, keep only `instruction.type == "error"`.
- Split targets on `t.is_separator()`. Each component is one or two detectors plus some observables.
- One detector → edge to the boundary node `D`.
- Parallel edges (same unordered `(u, v)`): merge with `p = p1(1−p2) + p2(1−p1)`. If their observable masks conflict, keep the mask of the more likely one and log a counter.
- Clamp `p` to `[1e-15, 0.5 − 1e-12]` before taking logs.

---

## 4. Decoder interface

```python
class Decoder(ABC):
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_dem(cls, dem: stim.DetectorErrorModel, **kwargs) -> "Decoder": ...

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """bool[D] → bool[K] predicted observable flips."""

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """bool[S, D] → bool[S, K]. Default loops; override when vectorizable."""
        return np.stack([self.decode(s) for s in syndromes])

    def stats(self) -> dict:
        """Optional extra metrics (e.g. BP convergence rate). Default {}."""
        return {}
```

`from_dem` builds the graph and matrices **once**. Never rebuild them per shot.

---

## 5. Harness API

```python
make_circuit(d: int, p: float, rounds: int | None = None, basis: str = "z") -> stim.Circuit
#   stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=d, rounds=rounds or d,
#       after_clifford_depolarization=p, after_reset_flip_probability=p,
#       before_measure_flip_probability=p, before_round_data_depolarization=p)

sample(circuit, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]   # dets, obs

@dataclass
class Result:
    decoder: str; d: int; p: float; rounds: int; shots: int; failures: int
    ler_shot: float; ler_shot_err: float     # binomial sqrt(P(1-P)/N)
    ler_round: float                          # 1 - (1 - ler_shot)**(1/rounds)
    t_mean_us: float; t_p50_us: float; t_p99_us: float   # per-shot decode time
    extra: dict                               # decoder.stats()

evaluate(decoder_name, d, p, shots, seed, **decoder_kwargs) -> Result
sweep(decoders, distances, ps, shots, seed, out_csv) -> pd.DataFrame
```

Timing: time `decode()` per shot with `time.perf_counter_ns` on a subsample (e.g. 1,000 shots) for the p50/p99 figures. Time the full `decode_batch` for throughput. Report both, because the batch figure is where vectorization shows up.

---

## 6. Experiment grid

| Parameter | Values |
|---|---|
| d | 3, 5, 7 |
| rounds | d |
| p | `np.geomspace(0.002, 0.015, 8)` for threshold plots. Add 0.001 for the low-p regime. |
| shots | 20k default. Raise until each point has ≥ 100 failures, or cap by time. |
| seeds | fixed per (d, p), recorded in the CSV |

Python UF at d = 7 might take ~1 ms/shot. Budget roughly 20k shots × 8 p-values × 3 d × 4 decoders, and use multiprocessing across (d, p) points if needed.

---

## 7. Parallelism plan (talk material)

| Level | What | How we show it |
|---|---|---|
| **Shot-level** | Shots are independent, so decoding them is embarrassingly parallel | `multiprocessing.Pool` (or `sinter`) over chunks. Plot speed-up vs #workers. |
| **Algorithm-level: BP** | Every check and variable updates at the same time | Messages stored as arrays over edges × shots. One iteration is a few numpy ops with no Python loops. |
| **Algorithm-level: UF** | Growing and merging clusters only touches neighbours | Count growth rounds per shot (≈ the parallel depth on a mesh of processors) and plot vs d. |
| **MWPM contrast** | The blossom algorithm is sequential and uses irregular memory access | Cite the literature. No full FPGA implementation of matching is known to the authors of arXiv:2506.01779. |

---

## 8. Testing strategy

| Test type | Example | Applies to |
|---|---|---|
| **Structural** | graph has D+1 nodes; all probs in (0, 0.5); H·(single mechanism) = its detectors | dem_graph |
| **Trivial** | all-zero syndrome → all-zero prediction | all decoders |
| **Single fault** ★ | for every error mechanism j, syndrome = H[:, j] → prediction must equal L[:, j] | MWPM-family must be 100%. BP reports its rate. |
| **Hand-built** | 5-bit repetition code graph with known answers | UF, BP, MWPM |
| **Equivalence** | belief-find with k=0 BP iterations ≡ weighted UF (bit-for-bit) | belief-find |
| **Agreement** | ≥ 99% prediction agreement with PyMatching at p = 0.001, d = 5 | UF, MWPM |
| **Statistical** (`@pytest.mark.slow`) | LER(d=5) < LER(d=3) at p = 0.002; LER_UF ≤ 3× LER_PM below threshold | all |

★ The single-fault test catches most bugs in DEM parsing, boundary handling and peeling. Write it first.

---

## 9. Phase map

| Phase | Deliverable | Acceptance (short) |
|---|---|---|
| [P0](phases/P0_harness.md) | harness + PyMatching baseline plot | PyMatching passes the single-fault test; LER drops with d below threshold |
| [P1](phases/P1_union_find.md) | `UnionFindDecoder` (unweighted → weighted) | single-fault 100%; ≥ 99% agreement at low p; LER ≤ 3× PyMatching |
| [P2](phases/P2_belief_propagation.md) | `BPDecoder` (min-sum, batched) | exact on tree-like toy graph; convergence stats reported |
| [P3](phases/P3_belief_find.md) | `BeliefFindDecoder` + iteration-budget study | k=0 ≡ UF; LER_BF < LER_UF near threshold |
| [P4](phases/P4_mwpm_stretch.md) | `NetworkxMWPMDecoder` (stretch) | LER statistically equal to PyMatching at d=3 |
| [P5](phases/P5_metrics_repo.md) | final sweep, plots, README, public repo | `pytest` green on a fresh clone |
| [P6](phases/P6_talk.md) | 10-min slides | one full timed rehearsal ≤ 10:00 |

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| UF peeling / boundary bugs | High | Single-fault test first; hand-built repetition-code tests; draw the d=3 graph |
| Parallel-edge observable conflicts in the DEM | Medium | Count conflicts; if non-zero, log them and compare against PyMatching's graph |
| BP doesn't converge | Expected | That's a result. Report convergence rate and use it to motivate belief-find |
| Python too slow at d=7 | Medium | Fewer shots for d=7; event-jump growth in UF; multiprocessing; numba only if needed |
| Time overrun | Medium | Follow the cut order in HACKATHON.md §7 |
| Over-claiming novelty | Medium | Credit Higgott et al. for belief-find; claim only the iteration-budget/fast-path study |
