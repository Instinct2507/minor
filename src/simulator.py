"""Core simulation engine for RESQ-X.

This module turns a synthetic scenario into a set of simulated dispatch
decisions. It implements:

* a consistent distance/time model used by both strategies,
* candidate-assignment generation and feasibility filtering,
* the baseline "nearest available ambulance" policy,
* the RESQ-X risk-aware policy (deterministic decision score),
* hospital congestion and emergency-coverage accounting.

All time numbers are kept in float minutes internally and rounded for the
UI. The module deliberately avoids negative times, invalid probabilities, and
impossible assignments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generator import CITY_BOUNDS

# --------------------------------------------------------------- constants
DISPATCH_LAG_BASE = 1.6          # minutes to leave station
TRAVEL_PER_UNIT = 2.6            # minutes per map unit
TRANSPORT_LAG_BASE = 1.9         # minutes scene -> hospital handover
BASE_DELAY_MEAN = 5.0            # hospital decision/handover minutes
BASE_DELAY_MAX = 14.0

# --------------------------------------------------------------- distance
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between (lat,lon) pairs, in miles."""
    R = 3958.8
    p1, p2 = np.radians([lat1, lat2])
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def city_distance(amb_x: float, amb_y: float, inc_x: float, inc_y: float) -> float:
    """Euclidean distance (map units) between an ambulance and an incident."""
    return float(np.hypot(amb_x - inc_x, amb_y - inc_y))



# ------------------------------------------------------------- estimate
def estimate_times(
    amb: pd.Series,
    hosp: pd.Series,
    inc: pd.Series,
    congestion: float,
    load: float,
) -> dict:
    """Return nominal dispatch / transport / hospital-delay minutes."""
    d_scene = city_distance(amb["x"], amb["y"], inc["x"], inc["y"])
    d_hosp = city_distance(hosp["x"], hosp["y"], inc["x"], inc["y"])

    dispatch = travel_time(amb["x"], amb["y"], inc["x"], inc["y"], congestion, d_scene)
    transport = travel_time(hosp["x"], hosp["y"], inc["x"], inc["y"], congestion, d_hosp)

    # Hospital delay: base delay, scaled up by occupancy and system load.
    occ = float(hosp.get("occupancy", 0.0))
    delay = hosp.get("base_delay", BASE_DELAY_MEAN) * (
        0.65 + occ / 100.0
    ) * (1.0 + load / 100.0)

    return {
        "dispatch": dispatch,
        "transport": transport,
        "hospital_delay": delay,
    }


def risk_score(
    hosp: pd.Series,
    congestion: float,
    demand: float,
    load: float,
    occupancy: float | None = None,
) -> float:
    """Explainable risk score in [0, 100] for a given hospital/congestion mix."""
    occ = occupancy if occupancy is not None else float(hosp.get("occupancy", 0.0))
    val = (
        8.0
        + occ * 0.42
        + congestion * 0.22
        + demand * 0.18
        + load * 0.25
    )
    return float(min(100.0, max(0.0, val)))


def occ_occupancy(hosp: pd.Series) -> float:
    """Read occupancy from a hospital row, tolerating the earlier naming."""
    for key in ("occupancy", "Occupancy (%)"):
        if key in hosp.index:
            return float(hosp[key])
    return 0.0


def travel_time(
    from_x: float,
    from_y: float,
    to_x: float,
    to_y: float,
    congestion: float,
    distance: float | None = None,
) -> float:
    """Simulate travel time between two points (minutes)."""
    d = distance if distance is not None else city_distance(from_x, from_y, to_x, to_y)
    factor = 1.0 + (congestion / 100.0) * 2.4
    return float(d * TRAVEL_PER_UNIT * factor + DISPATCH_LAG_BASE)


def build_candidates(
    amb_df: pd.DataFrame,
    hosp_df: pd.DataFrame,
    inc_df: pd.DataFrame,
    congestion: float,
    load: float,
    demand: float,
    uncertainty: float = 0.0,
    risk_weight: float = 1.0,
) -> pd.DataFrame:
    """Build one candidate row per (incident, ambulance, hospital) triple.

    ``uncertainty`` (0-100) adds a robustness buffer for travel-time and
    hospital-delay variability; ``risk_weight`` scales the congestion-risk
    term in the decision score. Both are transparent, documented heuristics.
    """
    rows = []
    for _, inc in inc_df.iterrows():
        for _, amb in amb_df[amb_df["status"] == "Available"].iterrows():
            for _, hosp in hosp_df.iterrows():
                t = estimate_times(amb, hosp, inc, congestion, load)
                total = t["dispatch"] + t["transport"] + t["hospital_delay"]
                r = risk_score(hosp, congestion, demand, load)
                # Uncertainty buffer: expected variability of dispatch + delay.
                buffer = (uncertainty / 100.0) * 0.35 * (t["dispatch"] + t["hospital_delay"])
                # Decision score: nominal time + uncertainty buffer + risk terms.
                score = (
                    total
                    + buffer
                    + risk_weight * (r / 10.0)
                    + max(0.0, occ_occupancy(hosp) - 80.0) * 0.18
                )
                rows.append(
                    {
                        "Incident": inc["Incident"],
                        "Severity": inc["severity"],
                        "Urgency": inc["urgency"],
                        "Ambulance": amb["Ambulance"],
                        "Hospital": hosp["Hospital"],
                        "Hospital occupancy (%)": occ_occupancy(hosp),
                        "Dispatch (min)": round(t["dispatch"], 2),
                        "Transport (min)": round(t["transport"], 2),
                        "Hospital delay (min)": round(t["hospital_delay"], 2),
                        "Total time (min)": round(total, 2),
                        "Uncertainty buffer (min)": round(buffer, 2),
                        "Risk score": round(r, 2),
                        "Decision score": round(score, 2),
                        "Hospital capacity": hosp.get("capacity", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def select_assignments(
    candidates: pd.DataFrame,
    incident_ids: list[str] | None = None,
    strategy: str = "baseline",
    hospital_capacities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Select assignments; incidents with no feasible option are Unserved.

    * ``baseline`` : nearest available ambulance per incident (no risk term).
    * ``resqx``    : cheapest RESQ-X decision score per incident.

    Both enforce the "no ambulance reuse" constraint explicitly: an ambulance
    dispatched to one incident is never dispatched to another.
    Hospital capacity constraints are also respected when specified.
    Incidents that cannot be served with the remaining ambulances or hospital
    capacities are returned as rows with ``Assignment status = "Unserved"``.
    """
    incident_ids = incident_ids or []
    if hospital_capacities is None and not candidates.empty and "Hospital capacity" in candidates.columns:
        valid_caps = candidates.dropna(subset=["Hospital capacity"])
        if not valid_caps.empty:
            hospital_capacities = dict(zip(valid_caps["Hospital"], valid_caps["Hospital capacity"].astype(int)))

    sel = _select_baseline if strategy == "baseline" else _select_resqx
    served = sel(candidates, incident_ids, hospital_capacities=hospital_capacities) if not candidates.empty else pd.DataFrame()

    base_cols = [
        "Incident",
        "Severity",
        "Urgency",
        "Ambulance",
        "Hospital",
        "Hospital occupancy (%)",
        "Dispatch (min)",
        "Transport (min)",
        "Hospital delay (min)",
        "Total time (min)",
        "Risk score",
        "Decision score",
    ]

    if served.empty:
        served = pd.DataFrame(columns=base_cols)
    else:
        served = served.copy()
    served["Assignment status"] = "Served"

    served_ids = set(served["Incident"]) if not served.empty else set()
    unserved_ids = [i for i in incident_ids if i not in served_ids]
    if unserved_ids:
        unserved = pd.DataFrame({"Incident": unserved_ids})
        for col in base_cols:
            if col != "Incident":
                unserved[col] = np.nan
        # Keep severity/urgency if the incident appeared in any candidate.
        if not candidates.empty:
            lookup = candidates.drop_duplicates("Incident").set_index("Incident")
            for col in ("Severity", "Urgency"):
                if col in lookup.columns:
                    unserved[col] = unserved["Incident"].map(lookup[col])
        unserved["Assignment status"] = "Unserved"
        served = pd.concat([served, unserved], ignore_index=True)

    return served


def _select_baseline(
    candidates: pd.DataFrame,
    incident_ids: list[str],
    hospital_capacities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Nearest-available: each incident gets the closest available ambulance, respecting capacity."""
    out = []
    remaining = candidates.copy()
    remaining_caps = dict(hospital_capacities) if hospital_capacities is not None else None
    for inc_id in incident_ids or []:
        sub = remaining[remaining["Incident"] == inc_id]
        if sub.empty:
            continue
        best = sub.sort_values(
            ["Dispatch (min)", "Transport (min)"], ascending=[True, True]
        ).iloc[0]
        out.append(best.to_dict())
        remaining = remaining[remaining["Ambulance"] != best["Ambulance"]]
        if remaining_caps is not None:
            h = best["Hospital"]
            if h in remaining_caps:
                remaining_caps[h] -= 1
                if remaining_caps[h] <= 0:
                    remaining = remaining[remaining["Hospital"] != h]
    return pd.DataFrame(out)


def _select_resqx(
    candidates: pd.DataFrame,
    incident_ids: list[str],
    hospital_capacities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """RESQ-X: greedy assignment by cheapest decision score, no reuse, respecting capacity."""
    out = []
    remaining = candidates.copy()
    remaining_caps = dict(hospital_capacities) if hospital_capacities is not None else None
    for inc_id in incident_ids or []:
        sub = remaining[remaining["Incident"] == inc_id]
        if sub.empty:
            continue
        best = sub.sort_values("Decision score", ascending=True).iloc[0]
        out.append(best.to_dict())
        remaining = remaining[remaining["Ambulance"] != best["Ambulance"]]
        if remaining_caps is not None:
            h = best["Hospital"]
            if h in remaining_caps:
                remaining_caps[h] -= 1
                if remaining_caps[h] <= 0:
                    remaining = remaining[remaining["Hospital"] != h]
    return pd.DataFrame(out)


def schedule_incidents(
    scenario: dict,
    strategy: str = "baseline",
    incident_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Run a full simulation: build candidates and select assignments."""
    amb_df = scenario["ambulances"]
    hosp_df = scenario["hospitals"]
    inc_df = scenario["incidents"]
    congestion = float(scenario.get("congestion", 25.0))
    demand = float(scenario.get("demand", 20.0))
    load = float(scenario.get("load", 30.0))
    uncertainty = float(scenario.get("uncertainty", 0.0))
    risk_weight = float(scenario.get("risk_weight", 1.0))

    incident_ids = incident_ids or inc_df["Incident"].tolist()
    candidates = build_candidates(
        amb_df, hosp_df, inc_df, congestion, load, demand,
        uncertainty=uncertainty, risk_weight=risk_weight,
    )
    hospital_capacities = None
    if "capacity" in hosp_df.columns:
        hospital_capacities = dict(zip(hosp_df["Hospital"], hosp_df["capacity"].astype(int)))

    result = select_assignments(
        candidates, incident_ids, strategy=strategy, hospital_capacities=hospital_capacities
    )

    result.insert(0, "Congestion", congestion)
    result.insert(1, "Demand", demand)
    result.insert(2, "Load", load)
    result.insert(3, "Strategy", strategy)
    result.insert(4, "Seed", scenario.get("seed"))
    return result


def summary_of(result: pd.DataFrame, amb_df: pd.DataFrame | None = None) -> dict:
    """Compute aggregate metrics from a result table."""
    amb_df = amb_df if amb_df is not None else pd.DataFrame(columns=["Ambulance", "status"])

    served = result[result["Assignment status"] != "Unserved"] if "Assignment status" in result else result
    n_total_amb = int(len(amb_df))
    n_avail_amb = int((amb_df["status"] == "Available").sum()) if ("status" in amb_df.columns and len(amb_df)) else n_total_amb
    n_served = int(len(served))
    rem_avail = max(0, n_avail_amb - n_served) if n_total_amb > 0 else 0
    coverage_rem = float(rem_avail) / max(1, n_total_amb) if n_total_amb > 0 else 0.0

    return {
        "n_incidents": int(len(result)),
        "n_served": n_served,
        "n_unserved": int((result["Assignment status"] == "Unserved").sum())
        if "Assignment status" in result
        else 0,
        "mean_dispatch": float(served["Dispatch (min)"].mean()) if len(served) else 0.0,
        "mean_transport": float(served["Transport (min)"].mean()) if len(served) else 0.0,
        "mean_hospital_delay": float(served["Hospital delay (min)"].mean()) if len(served) else 0.0,
        "mean_end_to_end": float(served["Total time (min)"].mean()) if len(served) else 0.0,
        "p90_dispatch": float(served["Dispatch (min)"].quantile(0.9)) if len(served) else 0.0,
        "mean_risk_score": float(served["Risk score"].mean()) if len(served) else 0.0,
        "mean_decision_score": float(served["Decision score"].mean()) if len(served) else 0.0,
        "ambulances_available": n_avail_amb,
        "total_ambulances": n_total_amb,
        "coverage_remaining": coverage_rem,
        "high_risk_assignments": int((served["Risk score"] > 70).sum()) if len(served) else 0,
    }
