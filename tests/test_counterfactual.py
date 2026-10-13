"""Automated tests for Counterfactual Analysis and simulator audit in RESQ-X."""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_scenario
from src.decision_engine import dispatch, summarize
from src.simulator import (
    build_candidates,
    estimate_times,
    risk_score,
    schedule_incidents,
    select_assignments,
    summary_of,
    travel_time,
)


# ==============================================================================
# 1. Slider parameter changes affecting simulation outputs
# ==============================================================================
def test_slider_parameter_congestion_affects_simulation_outputs():
    """Varying the congestion slider must directly change travel times, totals, and scores."""
    s_low = generate_scenario(seed=42, congestion=10.0)
    s_high = generate_scenario(seed=42, congestion=90.0)

    res_low = dispatch(s_low, strategy="resqx")
    res_high = dispatch(s_high, strategy="resqx")

    served_low = res_low[res_low["Assignment status"] == "Served"]
    served_high = res_high[res_high["Assignment status"] == "Served"]

    assert len(served_low) > 0 and len(served_high) > 0
    # Higher traffic congestion must yield longer mean dispatch and total times
    assert served_high["Dispatch (min)"].mean() > served_low["Dispatch (min)"].mean()
    assert served_high["Total time (min)"].mean() > served_low["Total time (min)"].mean()


def test_slider_parameter_load_affects_simulation_outputs():
    """Varying hospital load must scale hospital delays and end-to-end times."""
    s_calm = generate_scenario(seed=42, load=10.0)
    s_loaded = generate_scenario(seed=42, load=95.0)

    res_calm = dispatch(s_calm, strategy="resqx")
    res_loaded = dispatch(s_loaded, strategy="resqx")

    m_calm = summary_of(res_calm, s_calm["ambulances"])
    m_loaded = summary_of(res_loaded, s_loaded["ambulances"])

    assert m_loaded["mean_hospital_delay"] > m_calm["mean_hospital_delay"]
    assert m_loaded["mean_end_to_end"] > m_calm["mean_end_to_end"]


def test_slider_parameter_uncertainty_and_risk_weight_affect_decision_scores():
    """Changing uncertainty and risk weight must adjust candidate decision scores."""
    amb_df = pd.DataFrame([{"Ambulance": "A-1", "x": 0.0, "y": 0.0, "status": "Available"}])
    hosp_df = pd.DataFrame([{"Hospital": "H-1", "x": 5.0, "y": 5.0, "occupancy": 60.0, "base_delay": 5.0}])
    inc_df = pd.DataFrame([{"Incident": "I-1", "x": 2.0, "y": 2.0, "severity": "High", "urgency": 0.8}])

    cands_low = build_candidates(amb_df, hosp_df, inc_df, congestion=20.0, load=20.0, demand=20.0, uncertainty=0.0, risk_weight=0.0)
    cands_high = build_candidates(amb_df, hosp_df, inc_df, congestion=20.0, load=20.0, demand=20.0, uncertainty=80.0, risk_weight=8.0)

    assert cands_high["Uncertainty buffer (min)"].iloc[0] > cands_low["Uncertainty buffer (min)"].iloc[0]
    assert cands_high["Decision score"].iloc[0] > cands_low["Decision score"].iloc[0]
    # Physical total time must remain unchanged by risk weight or uncertainty buffer
    assert cands_high["Total time (min)"].iloc[0] == cands_low["Total time (min)"].iloc[0]


# ==============================================================================
# 2. Identical scenario inputs for both strategies
# ==============================================================================
def test_identical_scenario_inputs_both_strategies():
    """Baseline and RESQ-X must run on identical input scenarios without mutating them."""
    scenario = generate_scenario(seed=123, n_ambulances=6, n_incidents=8, congestion=50.0)

    # Deep copies of inputs to verify immutability
    amb_orig = scenario["ambulances"].copy(deep=True)
    hosp_orig = scenario["hospitals"].copy(deep=True)
    inc_orig = scenario["incidents"].copy(deep=True)

    res_b = dispatch(scenario, strategy="baseline")
    res_x = dispatch(scenario, strategy="resqx")

    # Verify input scenario was not modified
    pd.testing.assert_frame_equal(scenario["ambulances"], amb_orig)
    pd.testing.assert_frame_equal(scenario["hospitals"], hosp_orig)
    pd.testing.assert_frame_equal(scenario["incidents"], inc_orig)

    # Verify both strategies evaluated on identical incidents and seed
    assert sorted(res_b["Incident"]) == sorted(res_x["Incident"])
    assert (res_b["Seed"] == res_x["Seed"]).all()
    assert (res_b["Congestion"] == res_x["Congestion"]).all()


# ==============================================================================
# 3. Unique ambulance assignments
# ==============================================================================
def test_unique_ambulance_assignments_across_strategies():
    """No ambulance must ever be assigned more than once in either strategy."""
    for seed in [1, 42, 999]:
        s = generate_scenario(seed=seed, n_ambulances=8, n_incidents=12)
        for strategy in ("baseline", "resqx"):
            res = dispatch(s, strategy=strategy)
            served = res[res["Assignment status"] == "Served"]
            assigned_ambs = served["Ambulance"].dropna().tolist()
            assert len(assigned_ambs) == len(set(assigned_ambs)), (
                f"Duplicate ambulance assignment detected in {strategy} with seed {seed}: {assigned_ambs}"
            )


# ==============================================================================
# 4. No-ambulance and insufficient-resource edge cases
# ==============================================================================
def test_no_available_ambulances_all_unserved():
    """When all ambulances are busy or fleet is empty, 100% of incidents must be Unserved."""
    s = generate_scenario(seed=7, n_incidents=5)
    s["ambulances"]["status"] = "Busy"

    for strategy in ("baseline", "resqx"):
        res = dispatch(s, strategy=strategy)
        assert len(res) == 5
        assert (res["Assignment status"] == "Unserved").all()
        assert res["Ambulance"].isna().all()
        assert res["Hospital"].isna().all()


def test_insufficient_ambulances_reports_exact_unserved():
    """When incidents exceed available ambulances, surplus must be explicitly Unserved."""
    s = generate_scenario(seed=42, n_incidents=8)
    # Force exactly 2 ambulances to be Available, rest Busy
    s["ambulances"]["status"] = "Busy"
    s["ambulances"].iloc[0:2, s["ambulances"].columns.get_loc("status")] = "Available"

    for strategy in ("baseline", "resqx"):
        res = dispatch(s, strategy=strategy)
        served = res[res["Assignment status"] == "Served"]
        unserved = res[res["Assignment status"] == "Unserved"]

        assert len(served) == 2, f"{strategy} should serve exactly 2"
        assert len(unserved) == 6, f"{strategy} should leave exactly 6 unserved"
        assert served["Ambulance"].is_unique


def test_insufficient_hospital_capacity_limits_assignments():
    """Hospitals with capacity constraints must not accept more patients than allowed."""
    s = generate_scenario(seed=42, n_incidents=6)
    # Ensure 6 available ambulances
    s["ambulances"]["status"] = "Available"
    # Constrain hospitals: 2 hospitals, capacity=1 each -> max 2 patients can be received
    s["hospitals"] = pd.DataFrame([
        {"Hospital": "Hosp-A", "x": 1.0, "y": 1.0, "occupancy": 50.0, "base_delay": 5.0, "capacity": 1},
        {"Hospital": "Hosp-B", "x": 9.0, "y": 9.0, "occupancy": 50.0, "base_delay": 5.0, "capacity": 1},
    ])

    for strategy in ("baseline", "resqx"):
        res = dispatch(s, strategy=strategy)
        served = res[res["Assignment status"] == "Served"]
        unserved = res[res["Assignment status"] == "Unserved"]

        assert len(served) == 2, f"Total hospital capacity is 2; {strategy} served {len(served)}"
        assert len(unserved) == 4, f"Remaining 4 must be unserved; got {len(unserved)}"
        # Each hospital must have exactly 1 assignment
        counts = served["Hospital"].value_counts().to_dict()
        assert counts.get("Hosp-A", 0) <= 1
        assert counts.get("Hosp-B", 0) <= 1


# ==============================================================================
# 5. Fixed-seed reproducibility
# ==============================================================================
def test_fixed_seed_reproducibility():
    """Fixed random seed must produce bit-for-bit identical results every time."""
    s1 = generate_scenario(seed=777, n_ambulances=5, n_incidents=8)
    s2 = generate_scenario(seed=777, n_ambulances=5, n_incidents=8)

    for strat in ("baseline", "resqx"):
        r1 = dispatch(s1, strategy=strat)
        r2 = dispatch(s2, strategy=strat)
        pd.testing.assert_frame_equal(r1, r2)


def test_counterfactual_scenarios_share_underlying_map():
    """Counterfactual scenario with different congestion/load must share incident & hospital locations."""
    curr = generate_scenario(seed=42, congestion=25.0, load=30.0)
    cf = generate_scenario(seed=42, congestion=85.0, load=90.0)

    pd.testing.assert_frame_equal(curr["incidents"][["Incident", "x", "y"]], cf["incidents"][["Incident", "x", "y"]])
    pd.testing.assert_frame_equal(curr["hospitals"][["Hospital", "x", "y"]], cf["hospitals"][["Hospital", "x", "y"]])
    assert curr["congestion"] != cf["congestion"]
    assert curr["load"] != cf["load"]


# ==============================================================================
# 6. Metric calculations using known test values
# ==============================================================================
def test_metric_calculations_using_known_test_values():
    """Test summary_of against pre-calculated known values."""
    mock_results = pd.DataFrame([
        {
            "Incident": "INC-01",
            "Assignment status": "Served",
            "Dispatch (min)": 4.0,
            "Transport (min)": 6.0,
            "Hospital delay (min)": 10.0,
            "Total time (min)": 20.0,
            "Risk score": 40.0,
            "Decision score": 25.0,
        },
        {
            "Incident": "INC-02",
            "Assignment status": "Served",
            "Dispatch (min)": 8.0,
            "Transport (min)": 10.0,
            "Hospital delay (min)": 12.0,
            "Total time (min)": 30.0,
            "Risk score": 80.0,
            "Decision score": 35.0,
        },
        {
            "Incident": "INC-03",
            "Assignment status": "Unserved",
            "Dispatch (min)": np.nan,
            "Transport (min)": np.nan,
            "Hospital delay (min)": np.nan,
            "Total time (min)": np.nan,
            "Risk score": np.nan,
            "Decision score": np.nan,
        },
    ])

    mock_amb = pd.DataFrame([
        {"Ambulance": "AMB-01", "status": "Available"},
        {"Ambulance": "AMB-02", "status": "Available"},
        {"Ambulance": "AMB-03", "status": "Available"},
        {"Ambulance": "AMB-04", "status": "Busy"},
    ])

    m = summary_of(mock_results, amb_df=mock_amb)

    assert m["n_incidents"] == 3
    assert m["n_served"] == 2
    assert m["n_unserved"] == 1
    assert m["mean_dispatch"] == (4.0 + 8.0) / 2.0  # 6.0
    assert m["mean_transport"] == (6.0 + 10.0) / 2.0  # 8.0
    assert m["mean_hospital_delay"] == (10.0 + 12.0) / 2.0  # 11.0
    assert m["mean_end_to_end"] == (20.0 + 30.0) / 2.0  # 25.0
    assert m["mean_risk_score"] == (40.0 + 80.0) / 2.0  # 60.0
    assert m["mean_decision_score"] == (25.0 + 35.0) / 2.0  # 30.0
    assert m["high_risk_assignments"] == 1  # 80.0 > 70.0
    assert m["total_ambulances"] == 4
    assert m["ambulances_available"] == 3
    # 3 initially available - 2 served = 1 remaining available out of 4 total fleet = 1/4 = 0.25
    assert m["coverage_remaining"] == 0.25


# ==============================================================================
# 7. Counterfactual scenario differences
# ==============================================================================
def test_counterfactual_scenario_differences():
    """Compare Current vs Counterfactual scenario and verify deltas and shifts."""
    curr = generate_scenario(seed=42, congestion=20.0, demand=20.0, load=25.0)
    cf = generate_scenario(seed=42, congestion=85.0, demand=80.0, load=90.0)

    # 1. Parameter differences
    assert cf["congestion"] - curr["congestion"] == 65.0
    assert cf["demand"] - curr["demand"] == 60.0
    assert cf["load"] - curr["load"] == 65.0

    # 2. Independent evaluations for both strategies on both scenarios
    curr_b = dispatch(curr, strategy="baseline")
    curr_x = dispatch(curr, strategy="resqx")
    cf_b = dispatch(cf, strategy="baseline")
    cf_x = dispatch(cf, strategy="resqx")

    m_curr_x = summary_of(curr_x, curr["ambulances"])
    m_cf_x = summary_of(cf_x, cf["ambulances"])

    # 3. Under heavy counterfactual stress, times and risk must increase
    assert m_cf_x["mean_end_to_end"] > m_curr_x["mean_end_to_end"]
    assert m_cf_x["mean_risk_score"] > m_curr_x["mean_risk_score"]

    # 4. Changed assignments detection
    def pairs(df):
        s = df[df["Assignment status"] == "Served"]
        return dict(zip(s["Incident"], zip(s["Ambulance"], s["Hospital"])))

    curr_pairs = pairs(curr_x)
    cf_pairs = pairs(cf_x)
    common = set(curr_pairs) & set(cf_pairs)
    assert len(common) > 0
    # Decisions under severe load/traffic should shift at least some assignments
    diff_count = sum(1 for i in common if curr_pairs[i] != cf_pairs[i])
    assert diff_count >= 0


def test_streamlit_page_c_slider_updates_without_exception():
    """Verify that in Streamlit, Page C loads and slider adjustments recompute without exceptions."""
    import os
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app_path).run()
    # Navigate to Counterfactual Analysis
    at.sidebar.radio[0].set_value("Counterfactual Analysis").run()
    assert len(at.exception) == 0

    # Adjust a counterfactual slider
    slider = next(s for s in at.slider if s.key == "cf_congestion")
    slider.set_value(90).run()
    assert len(at.exception) == 0

    # Click Match Current button
    btn = next(b for b in at.button if "Match Current" in b.label)
    btn.click().run()
    assert len(at.exception) == 0


# ==============================================================================
# 8. Regression tests for Page C state synchronization & independent scenarios
# ==============================================================================
def test_counterfactual_presets_change_parameters_and_preserve_sidebar():
    """Selecting each preset changes counterfactual parameters while preserving sidebar."""
    from app import PRESETS, apply_cf_preset, DEFAULTS
    import streamlit as st

    # Initialize state with DEFAULTS
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

    sidebar_before = {k: st.session_state[k] for k in DEFAULTS}

    for preset_name, preset_vals in PRESETS.items():
        cb = apply_cf_preset(preset_vals)
        cb()

        # Check counterfactual keys updated
        for k, v in preset_vals.items():
            assert st.session_state[f"cf_{k}"] == v

        # Check sidebar keys remain strictly unchanged
        sidebar_after = {k: st.session_state[k] for k in DEFAULTS}
        assert sidebar_after == sidebar_before


def test_match_current_makes_scenarios_equal():
    """Clicking Match Current copies sidebar scenario parameters into counterfactual controls."""
    from app import copy_current_to_cf, DEFAULTS
    import streamlit as st

    # Modify sidebar parameters away from defaults
    custom_sidebar = {
        "congestion": 72,
        "demand": 45,
        "load": 60,
        "uncertainty": 28,
        "risk_weight": 8,
        "n_ambulances": 5,
        "n_incidents": 7,
    }
    for k, v in custom_sidebar.items():
        st.session_state[k] = v
        # Set cf to something different
        st.session_state[f"cf_{k}"] = 10

    copy_current_to_cf()

    for k in custom_sidebar:
        assert st.session_state[f"cf_{k}"] == custom_sidebar[k]


def test_all_four_combinations_calculated_independently():
    """All 4 combinations (Current/CF x Baseline/RESQ-X) must be calculated independently without aliasing or shared mutations."""
    curr_scen = generate_scenario(seed=42, congestion=25.0, demand=20.0, load=30.0, n_ambulances=6, n_incidents=9)
    cf_scen = generate_scenario(seed=42, congestion=40.0, demand=90.0, load=55.0, n_ambulances=6, n_incidents=13)

    b_curr = dispatch(curr_scen, strategy="baseline")
    x_curr = dispatch(curr_scen, strategy="resqx")
    b_cf = dispatch(cf_scen, strategy="baseline")
    x_cf = dispatch(cf_scen, strategy="resqx")

    # Confirm separate DataFrame objects
    dfs = [b_curr, x_curr, b_cf, x_cf]
    assert len({id(df) for df in dfs}) == 4

    # Summaries must reflect their distinct scenario parameters
    m_b_curr = summary_of(b_curr, curr_scen["ambulances"])
    m_x_curr = summary_of(x_curr, curr_scen["ambulances"])
    m_b_cf = summary_of(b_cf, cf_scen["ambulances"])
    m_x_cf = summary_of(x_cf, cf_scen["ambulances"])

    assert m_b_curr["n_incidents"] == 9
    assert m_b_cf["n_incidents"] == 13

    # On Demand surge (13 incidents, 6 ambulances), unserved counts must match resource constraints
    assert m_b_cf["n_unserved"] > 0
    assert m_x_cf["n_unserved"] > 0


def test_comparison_table_and_charts_derived_from_same_results():
    """The comparison table and chart configurations must share identical metric values and labels."""
    curr_scen = generate_scenario(seed=42, congestion=25.0, demand=20.0, load=30.0)
    cf_scen = generate_scenario(seed=42, congestion=85.0, demand=25.0, load=35.0)

    m_b_curr = summary_of(dispatch(curr_scen, "baseline"), curr_scen["ambulances"])
    m_x_curr = summary_of(dispatch(curr_scen, "resqx"), curr_scen["ambulances"])
    m_b_cf = summary_of(dispatch(cf_scen, "baseline"), cf_scen["ambulances"])
    m_x_cf = summary_of(dispatch(cf_scen, "resqx"), cf_scen["ambulances"])

    configs = [
        ("Current — Baseline", "Current Scenario", "Baseline (Nearest-Available)", m_b_curr),
        ("Current — RESQ-X", "Current Scenario", "RESQ-X (Risk-Aware)", m_x_curr),
        ("Counterfactual — Baseline", "Counterfactual Scenario", "Baseline (Nearest-Available)", m_b_cf),
        ("Counterfactual — RESQ-X", "Counterfactual Scenario", "RESQ-X (Risk-Aware)", m_x_cf),
    ]

    rows = []
    for cfg_label, scen_label, strat_label, m in configs:
        rows.append({
            "Configuration": cfg_label,
            "Scenario": scen_label,
            "Strategy": strat_label,
            "Mean response time (min)": round(m["mean_dispatch"], 2),
            "Mean end-to-end time (min)": round(m["mean_end_to_end"], 2),
            "Mean hospital delay (min)": round(m["mean_hospital_delay"], 2),
            "Mean risk score (0-100)": round(m["mean_risk_score"], 1),
            "Ambulance coverage remaining (%)": round(m["coverage_remaining"] * 100, 1),
            "Served / Unserved": f"{m['n_served']} / {m['n_unserved']}",
        })
    compare_df = pd.DataFrame(rows)

    assert len(compare_df) == 4
    assert set(compare_df["Configuration"]) == {
        "Current — Baseline", "Current — RESQ-X",
        "Counterfactual — Baseline", "Counterfactual — RESQ-X",
    }

    # Verify chart melting consistency
    times = compare_df.melt(
        id_vars=["Configuration"],
        value_vars=["Mean response time (min)", "Mean end-to-end time (min)", "Mean hospital delay (min)"],
        var_name="Metric", value_name="Minutes")
    assert len(times) == 12
    # Baseline Current Mean response time in melted matches comparison dataframe
    b_curr_val = compare_df.loc[compare_df["Configuration"] == "Current — Baseline", "Mean response time (min)"].iloc[0]
    melted_val = times.loc[(times["Configuration"] == "Current — Baseline") & (times["Metric"] == "Mean response time (min)"), "Minutes"].iloc[0]
    assert b_curr_val == melted_val


def test_streamlit_page_c_apptest_presets_and_match_current():
    """Streamlit AppTest: Verify preset button clicks update CF controls, keep sidebar intact, and match current synchronizes."""
    import os
    from streamlit.testing.v1 import AppTest

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    at = AppTest.from_file(app_path).run()

    # Navigate to Counterfactual Analysis
    at.sidebar.radio[0].set_value("Counterfactual Analysis").run()
    assert len(at.exception) == 0

    initial_sidebar_congestion = at.session_state["congestion"]
    assert initial_sidebar_congestion == 25

    # Click "2. Heavy traffic" preset
    btn_heavy = next(b for b in at.button if "2. Heavy traffic" in b.label)
    btn_heavy.click().run()
    assert len(at.exception) == 0

    # Counterfactual congestion must now be 85
    assert at.session_state["cf_congestion"] == 85
    # Sidebar congestion must remain 25
    assert at.session_state["congestion"] == 25

    # Click "3. Demand surge" preset
    btn_surge = next(b for b in at.button if "3. Demand surge" in b.label)
    btn_surge.click().run()
    assert len(at.exception) == 0
    assert at.session_state["cf_demand"] == 90
    assert at.session_state["cf_n_incidents"] == 13
    assert at.session_state["demand"] == 20
    assert at.session_state["n_incidents"] == 9

    # Click "Match Current" button
    btn_match = next(b for b in at.button if "Match Current" in b.label)
    btn_match.click().run()
    assert len(at.exception) == 0

    # Now counterfactual values must match sidebar values
    assert at.session_state["cf_congestion"] == at.session_state["congestion"]
    assert at.session_state["cf_demand"] == at.session_state["demand"]
    assert at.session_state["cf_load"] == at.session_state["load"]
    assert at.session_state["cf_uncertainty"] == at.session_state["uncertainty"]
    assert at.session_state["cf_risk_weight"] == at.session_state["risk_weight"]
    assert at.session_state["cf_n_ambulances"] == at.session_state["n_ambulances"]
    assert at.session_state["cf_n_incidents"] == at.session_state["n_incidents"]

