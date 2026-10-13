"""Tests for the decision engine (baseline vs RESQ-X dispatch)."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_scenario
from src.decision_engine import dispatch, summarize
from src.simulator import city_distance


def test_unknown_strategy_raises():
    s = generate_scenario(seed=1)
    with pytest.raises(ValueError):
        dispatch(s, strategy="magic")


def test_dispatch_is_deterministic_for_fixed_scenario():
    s = generate_scenario(seed=42)
    a = dispatch(s, strategy="resqx")
    b = dispatch(s, strategy="resqx")
    pd.testing.assert_frame_equal(a, b)


def test_baseline_picks_nearest_available_ambulance():
    s = generate_scenario(seed=42, n_incidents=3)
    res = dispatch(s, strategy="baseline")
    served = res[res["Assignment status"] == "Served"].iloc[0]
    inc = s["incidents"].set_index("Incident").loc[served["Incident"]]
    # The chosen ambulance must be the closest available one at decision time.
    avail = s["ambulances"][s["ambulances"]["status"] == "Available"]
    dists = avail.apply(
        lambda a: city_distance(a["x"], a["y"], inc["x"], inc["y"]), axis=1
    )
    nearest_amb = avail.loc[dists.idxmin(), "Ambulance"]
    assert served["Ambulance"] == nearest_amb


def test_resqx_returns_feasible_assignments_with_explanations():
    s = generate_scenario(seed=42)
    res = dispatch(s, strategy="resqx")
    served = res[res["Assignment status"] == "Served"]
    # Every served row carries the fields the dashboard must display.
    for col in ["Ambulance", "Hospital", "Dispatch (min)", "Risk score", "Decision score"]:
        assert served[col].notna().all(), f"missing {col}"
    assert (served["Dispatch (min)"] >= 0).all()
    assert served["Risk score"].between(0, 100).all()
    # A hospital must be assigned for every served incident.
    assert s["hospitals"]["Hospital"].isin(served["Hospital"]).any()


def test_no_reuse_across_all_incidents():
    s = generate_scenario(seed=99, n_incidents=9)
    for strategy in ("baseline", "resqx"):
        res = dispatch(s, strategy=strategy)
        served = res[res["Assignment status"] == "Served"]
        assert served["Ambulance"].is_unique


def test_both_strategies_share_identical_input_scenario():
    s = generate_scenario(seed=42)
    b = dispatch(s, strategy="baseline")
    x = dispatch(s, strategy="resqx")
    # Same scenario metadata stamped on both result tables.
    assert (b["Seed"] == x["Seed"]).all()
    assert (b["Congestion"] == x["Congestion"]).all()
    assert sorted(b["Incident"]) == sorted(x["Incident"])


def test_scenario_pressure_changes_decisions():
    normal = generate_scenario(seed=42, congestion=10, load=10, demand=10)
    pressed = generate_scenario(seed=42, congestion=95, load=95, demand=90)
    n = dispatch(normal, strategy="resqx")
    p = dispatch(pressed, strategy="resqx")
    ns = summarize(normal, strategy="resqx")
    ps = summarize(pressed, strategy="resqx")
    # Higher congestion must never make dispatch times cheaper.
    assert ps["mean_dispatch"] > ns["mean_dispatch"]
    # And the risk score must rise with pressure.
    assert ps["mean_risk_score"] > ns["mean_risk_score"]
    assert len(n) == len(p)


def test_summarize_reports_coverage_and_unserved_counts():
    s = generate_scenario(seed=42, n_incidents=9)
    m = summarize(s, strategy="baseline", amb_df=s["ambulances"])
    assert m["n_unserved"] >= 0
    assert m["n_served"] + m["n_unserved"] == 9
    assert m["total_ambulances"] == len(s["ambulances"])
