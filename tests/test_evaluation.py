"""Tests for the repeatable experiment pipeline and metric calculations."""

import os

import numpy as np
import pandas as pd

from src.evaluation import compare_strategies, run_experiment


def test_experiment_is_reproducible_with_same_seed():
    a = run_experiment(n_trials=5, seed=1234)
    b = run_experiment(n_trials=5, seed=1234)
    cols = [
        "mean_dispatch", "mean_end_to_end", "mean_hospital_delay",
        "mean_risk_score", "n_served", "n_unserved",
    ]
    pd.testing.assert_frame_equal(
        a["per_trial"][cols], b["per_trial"][cols], check_exact=True
    )


def test_experiment_differs_with_different_seed():
    a = run_experiment(n_trials=5, seed=1)
    b = run_experiment(n_trials=5, seed=2)
    assert not np.allclose(
        a["per_trial"]["mean_dispatch"].values,
        b["per_trial"]["mean_dispatch"].values,
    )


def test_trial_count_and_seeds_are_correct():
    m = run_experiment(n_trials=7, seed=100)
    assert m["n_trials"] == 7
    assert len(m["per_trial"]) == 7
    # Trial t must use seed + t so scenarios are reproducible and shared.
    assert list(m["per_trial"]["seed"]) == [100 + t for t in range(7)]


def test_all_reported_metrics_are_calculated_and_finite():
    m = run_experiment(n_trials=5, seed=42)
    for key in [
        "mean_dispatch", "median_dispatch", "mean_end_to_end",
        "mean_hospital_delay", "p90_dispatch", "mean_risk_score",
        "mean_coverage_remaining",
    ]:
        assert np.isfinite(m[key]), f"{key} is not finite"
    assert m["mean_dispatch"] > 0
    assert m["mean_end_to_end"] >= m["mean_dispatch"]
    assert m["p90_dispatch"] >= m["median_dispatch"]  # quantiles are monotonic
    assert 0 <= m["mean_coverage_remaining"] <= 1
    assert m["n_trials_unserved"] >= 0
    assert m["high_risk_assignments"] >= 0


def test_compare_strategies_uses_identical_scenarios():
    tbl, resqx, baseline = compare_strategies(n_trials=6, seed=777)
    assert len(tbl) == 6
    # Same trial seeds on both sides -> identical scenarios.
    assert list(resqx["per_trial"]["seed"]) == list(baseline["per_trial"]["seed"])
    assert list(tbl["seed"]) == list(resqx["per_trial"]["seed"])
    # Strategy labels are not swapped.
    assert (resqx["per_trial"]["strategy"] == "resqx").all()
    assert (baseline["per_trial"]["strategy"] == "baseline").all()


def test_metrics_match_manual_recomputation():
    m = run_experiment(n_trials=4, seed=42)
    df = m["per_trial"]
    assert abs(m["mean_dispatch"] - df["mean_dispatch"].mean()) < 1e-9
    assert abs(m["median_dispatch"] - df["mean_dispatch"].median()) < 1e-9
    assert m["n_trials_unserved"] == int(df["n_unserved"].sum())


def test_csv_export_roundtrip(tmp_path):
    m = run_experiment(n_trials=5, seed=42)
    path = os.path.join(str(tmp_path), "results.csv")
    m["per_trial"].to_csv(path, index=False)
    assert os.path.exists(path)
    back = pd.read_csv(path)
    assert len(back) == 5
    assert "mean_dispatch" in back.columns


def test_scenario_overrides_change_pressure_metrics():
    calm = run_experiment(n_trials=5, seed=42, scenario_overrides={"congestion": 5})
    jammed = run_experiment(n_trials=5, seed=42, scenario_overrides={"congestion": 95})
    assert jammed["mean_dispatch"] > calm["mean_dispatch"]
