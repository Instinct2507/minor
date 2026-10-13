"""Tests for the synthetic data generator and core simulator."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_scenario
from src.simulator import (
    build_candidates,
    estimate_times,
    risk_score,
    schedule_incidents,
    summary_of,
    travel_time,
)


# ----------------------------------------------------------- reproducibility
def test_same_seed_reproduces_identical_scenario():
    a = generate_scenario(seed=42)
    b = generate_scenario(seed=42)
    pd.testing.assert_frame_equal(a["ambulances"], b["ambulances"])
    pd.testing.assert_frame_equal(a["hospitals"], b["hospitals"])
    pd.testing.assert_frame_equal(a["incidents"], b["incidents"])


def test_different_seed_produces_different_scenario():
    a = generate_scenario(seed=1)
    b = generate_scenario(seed=2)
    assert not a["incidents"].equals(b["incidents"])


# ----------------------------------------------------------- data validity
def test_scenario_value_ranges_are_valid():
    s = generate_scenario(seed=7)
    hos = s["hospitals"]
    assert hos["occupancy"].between(30, 98).all()
    assert (hos["base_delay"] > 0).all()
    assert set(s["ambulances"]["status"]) <= {"Available", "Busy"}
    assert s["incidents"]["urgency"].between(0, 1).all()
    assert s["incidents"]["severity"].isin(["Critical", "High", "Medium", "Low"]).all()


def test_scenario_parameters_are_clipped_to_0_100():
    s = generate_scenario(seed=3, congestion=250, demand=-40, load=130)
    assert 0 <= s["congestion"] <= 100
    assert 0 <= s["demand"] <= 100
    assert 0 <= s["load"] <= 100


# ----------------------------------------------------------- time model
def test_travel_time_is_positive_and_grows_with_congestion():
    calm = travel_time(0, 0, 3, 4, congestion=0)
    jammed = travel_time(0, 0, 3, 4, congestion=100)
    assert calm > 0
    assert jammed > calm


def test_estimate_times_are_non_negative():
    s = generate_scenario(seed=11)
    amb = s["ambulances"].iloc[0]
    hosp = s["hospitals"].iloc[0]
    inc = s["incidents"].iloc[0]
    t = estimate_times(amb, hosp, inc, congestion=80, load=90)
    assert t["dispatch"] >= 0
    assert t["transport"] >= 0
    assert t["hospital_delay"] >= 0
    assert all(np.isfinite(list(t.values())))


def test_risk_score_is_bounded_between_0_and_100():
    s = generate_scenario(seed=5)
    for _, hosp in s["hospitals"].iterrows():
        r = risk_score(hosp, congestion=100, demand=100, load=100)
        assert 0 < r <= 100
        r_low = risk_score(hosp, congestion=0, demand=0, load=0)
        assert r_low <= r  # more pressure never lowers risk


# ----------------------------------------------------------- scheduling
def test_no_ambulance_is_reused_in_either_strategy():
    s = generate_scenario(seed=42, n_incidents=9)
    for strategy in ("baseline", "resqx"):
        res = schedule_incidents(s, strategy=strategy)
        served = res[res["Assignment status"] == "Served"]
        assert served["Ambulance"].is_unique, f"{strategy} reused an ambulance"


def test_both_strategies_cover_the_same_incidents():
    s = generate_scenario(seed=42, n_incidents=9)
    b = schedule_incidents(s, strategy="baseline")
    x = schedule_incidents(s, strategy="resqx")
    assert sorted(b["Incident"]) == sorted(x["Incident"])
    b_served = set(b.loc[b["Assignment status"] == "Served", "Incident"])
    x_served = set(x.loc[x["Assignment status"] == "Served", "Incident"])
    assert b_served == x_served  # same feasibility constraints


def test_unserved_incidents_are_reported_not_silently_dropped():
    s = generate_scenario(seed=42, n_incidents=9)
    res = schedule_incidents(s, strategy="baseline")
    # 9 incidents but only ~5 ambulances are available -> explicit unserved rows
    assert len(res) == 9
    assert (res["Assignment status"] == "Unserved").any()
    unserved = res[res["Assignment status"] == "Unserved"]
    assert unserved["Ambulance"].isna().all()  # no fake assignment invented


def test_no_available_ambulances_reports_everything_unserved():
    s = generate_scenario(seed=42, n_incidents=4)
    s["ambulances"] = s["ambulances"].assign(status="Busy")
    res = schedule_incidents(s, strategy="resqx")
    assert len(res) == 4
    assert (res["Assignment status"] == "Unserved").all()


def test_summary_metrics_are_finite_and_sane():
    s = generate_scenario(seed=42)
    res = schedule_incidents(s, strategy="resqx")
    m = summary_of(res, s["ambulances"])
    assert np.isfinite(m["mean_dispatch"])
    assert np.isfinite(m["mean_end_to_end"])
    assert m["n_served"] + m["n_unserved"] == m["n_incidents"]
    assert 0 <= m["coverage_remaining"] <= 1
    assert m["mean_risk_score"] >= 0


def test_candidates_only_use_available_ambulances():
    s = generate_scenario(seed=42)
    cands = build_candidates(
        s["ambulances"], s["hospitals"], s["incidents"],
        congestion=s["congestion"], load=s["load"], demand=s["demand"],
    )
    available = set(s["ambulances"].loc[s["ambulances"]["status"] == "Available", "Ambulance"])
    assert set(cands["Ambulance"]) <= available
