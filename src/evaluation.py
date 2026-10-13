"""Experiment pipeline and repeated-trial evaluation.

``run_experiment`` runs ``n_trials`` independent repeated trials on the *same*
input scenario (same seed), comparing ``baseline`` against ``resqx`` on every
trial. This guarantees a fair, apples-to-apples comparison: the only thing that
changes between strategies is the dispatch rule, not the generated data.

Every reported number is computed from the executed simulation outputs. Nothing
is hard-coded and no performance claim is made -- results are returned for the
caller to render and to export to CSV.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .data_generator import generate_scenario
from .decision_engine import dispatch, summarize


# ---------------------------------------------------------- result schema
def _as_dict(r: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if v is None else float(v)) for k, v in r.items()}


def run_experiment(
    n_trials: int = 30,
    seed: int = 1234,
    n_ambulances: int = 6,
    n_hospitals: int = 4,
    n_incidents: int = 9,
    strategy: str = "resqx",
    scenario_overrides: dict[str, float] | None = None,
) -> dict:
    """Run a reproducible experiment and return calculated metrics.

    Parameters
    ----------
    n_trials : number of repeated trials to average over.
    seed : base seed; trial ``t`` uses ``seed + t``.
    strategy : strategy to evaluate (baseline or resqx).
    scenario_overrides : optional(dict) of scenario parameters to vary per
        trial (e.g. {"congestion": 70}) while keeping everything else fixed.

    Returns
    -------
    dict with per-trial lists plus a summary block computed from the executed
    trial results.
    """
    overrides = scenario_overrides or {}
    allowed = {"congestion", "demand", "load", "uncertainty", "risk_weight"}
    t_start = time.perf_counter()
    trials = []
    for t in range(n_trials):
        s = generate_scenario(
            n_ambulances=n_ambulances,
            n_hospitals=n_hospitals,
            n_incidents=n_incidents,
            seed=seed + t,
            **{k: v for k, v in overrides.items() if k in allowed},
        )
        summary = summarize(s, strategy=strategy)
        summary["trial"] = int(t)
        summary["seed"] = int(s["seed"])
        summary["strategy"] = strategy
        trials.append(summary)

    df = pd.DataFrame(trials)

    def _mean(col: str) -> float:
        return float(df[col].mean())

    def _median(col: str) -> float:
        return float(df[col].median())

    def _p90(col: str) -> float:
        return float(df[col].quantile(0.90))

    metrics = {
        "n_trials": int(len(df)),
        "n_trials_served": int(df["n_served"].sum()),
        "n_trials_unserved": int(df["n_unserved"].sum()),
        "mean_dispatch": _mean("mean_dispatch"),
        "median_dispatch": _median("mean_dispatch"),
        "mean_transport": _mean("mean_transport"),
        "mean_hospital_delay": _mean("mean_hospital_delay"),
        "mean_end_to_end": _mean("mean_end_to_end"),
        "p90_dispatch": _p90("mean_dispatch"),
        "mean_risk_score": _mean("mean_risk_score"),
        "mean_decision_score": _mean("mean_decision_score"),
        "mean_coverage_remaining": _mean("coverage_remaining"),
        "mean_ambulances_available": _mean("ambulances_available"),
        "high_risk_assignments": int(df["high_risk_assignments"].sum()),
        "computation_seconds": float(time.perf_counter() - t_start),
        "per_trial": df,
    }
    return metrics


def compare_strategies(
    n_trials: int = 30,
    seed: int = 1234,
    **kwargs,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run identical scenarios for both strategies and compare per trial.

    Both strategies are evaluated on the *same* trial seeds, so the only
    difference between rows is the dispatch rule. Returns
    ``(comparison_table, resqx_metrics, baseline_metrics)``. All values are
    calculated from executed trials -- nothing is hard-coded.
    """
    resqx = run_experiment(n_trials=n_trials, seed=seed, strategy="resqx", **kwargs)
    baseline = run_experiment(n_trials=n_trials, seed=seed, strategy="baseline", **kwargs)

    b_df = baseline["per_trial"].set_index("trial")
    r_df = resqx["per_trial"].set_index("trial")

    rows = []
    for t in range(n_trials):
        b = b_df.loc[t]
        r = r_df.loc[t]
        rows.append(
            {
                "trial": t,
                "seed": int(b["seed"]),
                "baseline_mean_dispatch": b["mean_dispatch"],
                "resqx_mean_dispatch": r["mean_dispatch"],
                "baseline_end_to_end": b["mean_end_to_end"],
                "resqx_end_to_end": r["mean_end_to_end"],
                "baseline_hospital_delay": b["mean_hospital_delay"],
                "resqx_hospital_delay": r["mean_hospital_delay"],
                "baseline_coverage": b["coverage_remaining"],
                "resqx_coverage": r["coverage_remaining"],
            }
        )
    return pd.DataFrame(rows), resqx, baseline

