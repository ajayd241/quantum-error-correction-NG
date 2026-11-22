# Running and extending the decoders

This guide is the practical interface to the project. Commands assume you are
in the repository root and have completed the installation steps in the
[README](../README.md#quick-start).

## Which command should I use?

- To verify the project: `pytest -q`
- To run one or more decoders: `python scripts/run_sweep.py ...`
- To reproduce the submission: `python scripts/run_sweep.py --config configs/final.yaml`
- To regenerate figures: `python scripts/plot_results.py ...`
- To call a decoder from your own Python code: use `qecdec.harness.evaluate`
  or construct a class from `qecdec.decoders.REGISTRY`.

Do not run files such as `src/qecdec/decoders/union_find.py` directly. They are
importable implementations behind the common `Decoder` interface; the sweep
script constructs the noisy circuit, samples detector events, runs the selected
decoder, scores logical failures, and measures latency.

## Decoder names

Pass one or more of these comma-separated identifiers to `--decoders`:

| CLI name | Decoder | Main implementation |
|---|---|---|
| `pymatching` | Optimized PyMatching reference | [`pymatching_ref.py`](../src/qecdec/decoders/pymatching_ref.py) |
| `uf_unweighted` | Unweighted Union-Find | [`union_find.py`](../src/qecdec/decoders/union_find.py) |
| `uf` | Likelihood-weighted Union-Find | [`union_find.py`](../src/qecdec/decoders/union_find.py) |
| `bp` | Normalized min-sum belief propagation | [`bp.py`](../src/qecdec/decoders/bp.py) |
| `belief_find` | BP-guided weighted Union-Find | [`belief_find.py`](../src/qecdec/decoders/belief_find.py) |

`pymatching` is the external reference. The other four identifiers expose the
three decoder families implemented in this repository, including both UF
weighting modes.

## Run one decoder

The general form is:

```bash
python scripts/run_sweep.py \
  --decoders DECODER_NAME \
  --distances DISTANCE \
  --ps PHYSICAL_ERROR_PROBABILITY \
  --shots NUMBER_OF_SAMPLES \
  --seed RANDOM_SEED \
  --workers NUMBER_OF_PROCESSES \
  --out OUTPUT.csv
```

### PyMatching reference

```bash
python scripts/run_sweep.py \
  --decoders pymatching \
  --distances 5 \
  --ps 0.005 \
  --shots 1000 \
  --seed 1 \
  --out results/pymatching_demo.csv
```

### Unweighted Union-Find

```bash
python scripts/run_sweep.py \
  --decoders uf_unweighted \
  --distances 5 \
  --ps 0.005 \
  --shots 1000 \
  --seed 1 \
  --out results/uf_unweighted_demo.csv
```

### Weighted Union-Find

```bash
python scripts/run_sweep.py \
  --decoders uf \
  --distances 5 \
  --ps 0.005 \
  --shots 1000 \
  --seed 1 \
  --out results/uf_demo.csv
```

### Belief propagation

```bash
python scripts/run_sweep.py \
  --decoders bp \
  --distances 5 \
  --ps 0.005 \
  --shots 250 \
  --seed 1 \
  --alpha 1.0 \
  --out results/bp_demo.csv
```

`--alpha` controls min-sum message normalization. The final BP sweep used
`alpha=1.0`; the class default is `0.75`. BP is intentionally given fewer
shots in quick examples because this pure-Python implementation is slower.

### Belief-find

Use the final study's `alpha=0.625`, two BP iterations, and fast path by loading
the committed config and overriding only the desired grid:

```bash
python scripts/run_sweep.py \
  --config configs/final.yaml \
  --decoders belief_find \
  --distances 5 \
  --ps 0.005 \
  --shots 250 \
  --workers 1 \
  --skip-benchmark \
  --out results/belief_find_demo.csv
```

Without `--config`, belief-find uses its class defaults: 30 BP iterations,
`alpha=0.625`, and the BP-converged fast path enabled.

## Change distance, noise, samples, and parallelism

| Option | Meaning | Examples |
|---|---|---|
| `--distances` | Surface-code distance `d`; the CLI sets measurement rounds equal to `d` | `3` or `3,5,7` |
| `--ps` | Physical error probability applied to all four circuit-noise processes | `0.005` means 0.5%; grid: `0.002,0.005,0.01` |
| `--shots` | Number of independently sampled experiments at each grid point | `100` for a smoke test; thousands for estimates |
| `--seed` | Base random seed; each `(d,p)` gets a stable derived seed shared across decoders | `1` |
| `--workers` | Process count for independent `(decoder,d,p)` jobs | `1`, `2`, or `4` |
| `--alpha` | BP normalization override applied to selected BP-family decoders | `1.0` |
| `--out` | Destination CSV path | `results/my_sweep.csv` |

Distances 3, 5, and 7 are the tested submission grid. A run with two decoders,
three distances, and four probabilities contains `2 × 3 × 4 = 24` independent
jobs. More shots improve statistical precision but also increase runtime.

Example multi-decoder grid:

```bash
python scripts/run_sweep.py \
  --decoders pymatching,uf \
  --distances 3,5,7 \
  --ps 0.003,0.005,0.008,0.012 \
  --shots 1000 \
  --seed 7 \
  --workers 4 \
  --out results/pm_vs_uf.csv
```

## Customize algorithm-specific settings

For options not exposed as simple flags, copy `configs/final.yaml` and edit its
`decoder_options`. The file uses JSON syntax, which is valid YAML 1.2 and needs
no extra parser dependency.

```json
{
  "decoders": ["bp", "belief_find"],
  "distances": [3, 5],
  "probabilities": [0.003, 0.005, 0.008],
  "shots": 500,
  "seed": 11,
  "workers": 2,
  "out": "results/custom.csv",
  "decoder_options": {
    "bp": {
      "alpha": 1.0,
      "max_iter": 10,
      "batch_size": 256
    },
    "belief_find": {
      "alpha": 0.625,
      "bp_iters": 2,
      "fast_path": true
    }
  }
}
```

Save that object as `configs/custom.yaml`, then run:

```bash
python scripts/run_sweep.py --config configs/custom.yaml
```

Explicit CLI values override the corresponding top-level config values. For
example, `--shots 100 --distances 3` makes a quick version of a larger config.

## Regenerate plots

For a useful threshold figure, include distances 3, 5, and 7 and at least a few
physical-error probabilities spanning the expected crossing. Then run:

```bash
python scripts/plot_results.py results/pm_vs_uf.csv --out results/figs/custom
```

To reproduce every submission figure, including the iteration-budget and
parallel-scaling plots:

```bash
python scripts/plot_results.py results/final.csv \
  --parallel-csv results/parallel_speedup.csv \
  --budget-csv results/budget.csv \
  --out results/figs
```

All published CSVs and PNGs are committed under `results/`, so judges can read
the results without rerunning the full benchmark.

## Understand the CSV output

Each row is one `(decoder, d, p)` operating point.

| Column | Meaning |
|---|---|
| `decoder` | Registry name listed above |
| `d`, `rounds` | Code distance and syndrome-measurement rounds; the CLI uses `rounds=d` |
| `p` | Physical error probability |
| `shots`, `failures` | Samples and shots with at least one incorrect logical prediction |
| `ler_shot` | Logical error rate per complete shot: `failures / shots` |
| `ler_shot_err` | Binomial standard error of `ler_shot` |
| `ler_round` | Logical error rate converted to a per-round value |
| `t_mean_us` | Batched decode time per shot in microseconds |
| `t_p50_us`, `t_p99_us` | Median and 99th-percentile individual-decode latency |
| `extra` | JSON-encoded decoder-specific statistics |
| `seed` | Stable derived seed for that `(d,p)` point |
| `sweep_workers`, `sweep_wall_s` | Process count and total sweep wall time |

Points with fewer than roughly 100 failures have noisy logical-error estimates.
Increase `--shots` before making a close accuracy comparison.

## Use the harness from Python

For one scored operating point, `evaluate` is the smallest API:

```python
from dataclasses import asdict

from qecdec.harness import evaluate

result = evaluate(
    "uf",
    d=5,
    p=0.005,
    shots=1000,
    seed=1,
)
print(asdict(result))
```

Algorithm-specific constructor options are accepted as keywords:

```python
from qecdec.harness import evaluate

result = evaluate(
    "belief_find",
    d=5,
    p=0.005,
    shots=250,
    seed=1,
    alpha=0.625,
    bp_iters=2,
    fast_path=True,
)
print(result.ler_shot, result.t_p99_us, result.extra)
```

To obtain raw predictions for your own analysis:

```python
import numpy as np

from qecdec.circuits import make_circuit
from qecdec.decoders import REGISTRY
from qecdec.harness import sample

circuit = make_circuit(d=3, p=0.005, rounds=3)
dem = circuit.detector_error_model(decompose_errors=True)
decoder = REGISTRY["uf"].from_dem(dem)

detector_events, true_observables = sample(circuit, shots=100, seed=1)
predicted_observables = decoder.decode_batch(detector_events)
logical_error_rate = np.mean(
    np.any(predicted_observables != true_observables, axis=1)
)
print(f"LER = {logical_error_rate:.4f}")
```

`decode(syndrome)` accepts `bool[D]` and returns predicted logical flips
`bool[K]`. `decode_batch(syndromes)` accepts `bool[S,D]` and returns
`bool[S,K]`, where `S` is shots, `D` is detectors, and `K` is logical
observables.

## Add another decoder

1. Subclass `qecdec.base.Decoder`.
2. Implement `from_dem` and `decode`; override `decode_batch` if it can be
   vectorized.
3. Register the class by its unique `name` in
   [`qecdec.decoders.REGISTRY`](../src/qecdec/decoders/__init__.py).
4. Add zero-syndrome, single-fault, and fixed-seed tests.
5. Pass the new registry name to `--decoders`; the harness needs no special
   case.
