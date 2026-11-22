"""Tests for final-sweep configuration and threshold reporting."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.plot_results import estimate_threshold
from scripts.run_sweep import load_config


def test_log_log_threshold_interpolation() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["toy"] * 4,
            "d": [5, 5, 7, 7],
            "p": [0.004, 0.008, 0.004, 0.008],
            "ler_shot": [0.02, 0.04, 0.01, 0.08],
        }
    )

    threshold = estimate_threshold(frame, "toy")

    assert threshold == pytest.approx((0.004 * 0.008) ** 0.5)


def test_threshold_ignores_reverse_crossing_from_low_shot_noise() -> None:
    frame = pd.DataFrame(
        {
            "decoder": ["toy"] * 6,
            "d": [5, 5, 5, 7, 7, 7],
            "p": [0.002, 0.004, 0.008, 0.002, 0.004, 0.008],
            "ler_shot": [0.001, 0.02, 0.04, 0.002, 0.01, 0.08],
        }
    )

    threshold = estimate_threshold(frame, "toy")

    assert threshold == pytest.approx((0.004 * 0.008) ** 0.5)


def test_config_loader_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps({"decoders": ["pymatching"], "typo": 3}))

    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)
