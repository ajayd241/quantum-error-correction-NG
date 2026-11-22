#!/usr/bin/env python3
"""Draw one Union-Find decode as defects, grown clusters, and correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stim

from qecdec.circuits import make_circuit
from qecdec.decoders.union_find import UnionFindDecoder
from qecdec.harness import sample


def _positions(
    dem: stim.DetectorErrorModel,
    boundary: int,
) -> dict[int, tuple[float, float]]:
    """Project Stim's space-time detector coordinates onto a readable plane."""
    coordinates = dem.get_detector_coordinates()
    positions: dict[int, tuple[float, float]] = {}
    for detector in range(boundary):
        coordinate = coordinates.get(detector, [float(detector), 0.0])
        x = float(coordinate[0])
        y = float(coordinate[1]) if len(coordinate) > 1 else 0.0
        time = float(coordinate[2]) if len(coordinate) > 2 else 0.0
        positions[detector] = (x + 0.28 * time, y - 0.28 * time)
    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    positions[boundary] = (float(np.mean(xs)), min(ys) - 2.0)
    return positions


def _draw_panel(
    axis,
    graph: nx.Graph,
    positions: dict[int, tuple[float, float]],
    defects: set[int],
    boundary: int,
    highlighted_edges: list[tuple[int, int]],
    highlight_color: str,
    title: str,
) -> None:
    """Draw the common graph background and one stage's selected edges."""
    detector_nodes = list(range(boundary))
    detector_colors = [
        "#d62728" if node in defects else "#ffffff"
        for node in detector_nodes
    ]
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edge_color="#c8c8c8",
        width=1.0,
    )
    if highlighted_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=highlighted_edges,
            edge_color=highlight_color,
            width=3.2,
        )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        nodelist=detector_nodes,
        node_color=detector_colors,
        edgecolors="#222222",
        node_size=240,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        nodelist=[boundary],
        node_color="#333333",
        node_shape="s",
        node_size=290,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        ax=axis,
        labels={node: f"D{node}" for node in detector_nodes},
        font_size=7,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        ax=axis,
        labels={boundary: "B"},
        font_color="white",
        font_size=8,
    )
    axis.set_title(title)
    axis.set_axis_off()


def plot_example(out: Path) -> Path:
    """Generate a deterministic three-stage d=3, one-round decode figure."""
    circuit = make_circuit(d=3, p=0.02, rounds=1)
    dem = circuit.detector_error_model(decompose_errors=True)
    decoder = UnionFindDecoder.from_dem(dem)
    syndromes, _ = sample(circuit, shots=2_000, seed=1_903)

    selected_shot = -1
    selected_syndrome = None
    selected_outcome = None
    for shot, syndrome in enumerate(syndromes):
        defect_count = int(np.count_nonzero(syndrome))
        if not 2 <= defect_count <= 4:
            continue
        outcome = decoder.decode_outcome(syndrome)
        if np.any(outcome.correction):
            selected_shot = shot
            selected_syndrome = syndrome
            selected_outcome = outcome
            break
    if selected_syndrome is None or selected_outcome is None:
        raise RuntimeError("fixed sample did not contain a suitable syndrome")

    graph_data = decoder.graph
    boundary = graph_data.num_detectors
    graph = nx.Graph()
    graph.add_nodes_from(range(boundary + 1))
    edge_pairs = [
        (int(first), int(second))
        for first, second in zip(graph_data.edge_u, graph_data.edge_v)
    ]
    graph.add_edges_from(edge_pairs)
    positions = _positions(dem, boundary)
    defects = set(int(node) for node in np.flatnonzero(selected_syndrome))
    grown_edges = [
        edge_pairs[int(edge)]
        for edge in np.flatnonzero(selected_outcome.fully_grown)
    ]
    correction_edges = [
        edge_pairs[int(edge)]
        for edge in np.flatnonzero(selected_outcome.correction)
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    _draw_panel(
        axes[0],
        graph,
        positions,
        defects,
        boundary,
        [],
        "#9467bd",
        "1. Detector defects",
    )
    _draw_panel(
        axes[1],
        graph,
        positions,
        defects,
        boundary,
        grown_edges,
        "#9467bd",
        f"2. Grown clusters ({selected_outcome.growth_rounds} rounds)",
    )
    _draw_panel(
        axes[2],
        graph,
        positions,
        defects,
        boundary,
        correction_edges,
        "#2ca02c",
        "3. Peeled correction",
    )
    figure.suptitle(
        f"Weighted Union-Find example: d=3, one round, sampled shot {selected_shot}",
        fontsize=13,
    )
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figs/union_find_example.png"),
    )
    args = parser.parse_args()
    output = plot_example(args.out)
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
