from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.run_pipeline import top_bottom, weighted_available


ROOT = Path(__file__).resolve().parents[1]


def test_config_contains_required_sections():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert {"project", "analysis", "missing", "benchmark_2014", "style"} <= set(
        config
    )
    assert set(config["analysis"]["stress_columns"]) == {
        "thermal",
        "humidity",
        "wind",
        "precipitation",
        "air_quality",
    }


def test_season_months_cover_calendar_once():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    months = [
        month
        for season_months in config["analysis"]["seasons"].values()
        for month in season_months
    ]
    assert sorted(months) == list(range(1, 13))
    assert len(months) == len(set(months))


def test_weighted_available_renormalizes_observed_components():
    scores = pd.DataFrame(
        {
            "thermal": [100.0, 100.0, np.nan],
            "humidity": [50.0, np.nan, 50.0],
            "wind": [0.0, 0.0, 0.0],
        }
    )
    weights = {"thermal": 0.5, "humidity": 0.3, "wind": 0.2}
    result = weighted_available(scores, weights, min_n=2)

    assert np.isclose(result.iloc[0], 65.0)
    assert np.isclose(result.iloc[1], 100.0 * 0.5 / 0.7)
    assert np.isclose(result.iloc[2], 50.0 * 0.3 / 0.5)


def test_top_bottom_returns_expected_difference():
    index = pd.Series(np.arange(100), dtype=float)
    target = index / 10.0
    top, bottom, difference, n = top_bottom(target, index, q=0.20)

    assert n == 100
    assert top > bottom
    assert np.isclose(difference, top - bottom)


def test_benchmark_weights_are_valid():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for season, weights in config["benchmark_2014"].items():
        if season == "note":
            continue
        assert all(value >= 0 for value in weights.values())
        assert np.isclose(sum(weights.values()), 1.0, atol=1e-4)

