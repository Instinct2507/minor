"""Synthetic data generation for the RESQ-X emergency-resource simulator.

This module builds a reproducible, fictional scenario: a grid of ambulance
stations, a set of hospitals, and a set of emergency incidents. All coordinates,
statuses, occupancies, and timings are synthetic fiction -- no real city data is
used and nothing here is live or clinically validated.

The generator is driven by a single integer ``seed`` so experiments can ask for
the *same* scenario twice and get bit-for-bit identical data. Each call to
``generate_scenario`` returns a fresh, random scenario unless a fixed seed is
passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Coordinate model
# --------------------------------------------------------------------------
# We work in a simple 2-D "city" model. Distances are in abstract map units
# (≈1 map unit ~ a few city blocks). The same distance model is used for
# ambulances, incidents, and hospitals so every time estimate is in compatible
# seconds/minutes.
#
# Time model:
#   * dispatch_lag    : base seconds an ambulance needs to leave its station
#   * travel_per_unit : seconds per map unit of distance
#   * transport_lag   : base seconds from scene to hospital handover
#   * base_delay      : hospital-side decision/handover seconds per row
# All time numbers are converted to minutes for display (divide by 60.0).

CITY_BOUNDS = {"min_x": 0.0, "max_x": 10.0, "min_y": 0.0, "max_y": 10.0}


def _seed_rng(seed: int | None) -> np.random.Generator:
    """Turn a seed into a reproducible NumPy random generator."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


def generate_ambulances(
    n: int,
    rng: np.random.Generator,
    station_centers: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build a synthetic ambulance fleet DataFrame.

    Columns: Ambulance, station_x, station_y, status, type, x, y.
    About a quarter of the fleet starts Busy; the rest Available.
    """
    stations = station_centers or [
        (1.4, 7.8),
        (3.8, 5.9),
        (6.9, 7.2),
        (8.1, 3.0),
        (4.6, 2.0),
        (7.1, 5.1),
    ]
    rows = []
    for k in range(n):
        sx, sy = stations[k % len(stations)]
        is_busy = k < max(0, int(n * 0.25))
        rows.append(
            {
                "Ambulance": f"AMB-{k + 1:02d}",
                "station_x": sx,
                "station_y": sy,
                "status": "Busy" if is_busy else "Available",
                "type": "ALS" if k % 3 == 0 else "BLS",
                "x": sx,
                "y": sy,
            }
        )
    return pd.DataFrame(rows)


def generate_hospitals(
    n: int,
    rng: np.random.Generator,
    bounds: dict | None = None,
) -> pd.DataFrame:
    """Build a hospital DataFrame with synthetic occupancy (30-98%)."""
    bounds = bounds or CITY_BOUNDS
    occ = np.clip(62.0 + rng.uniform(-20.0, 20.0, size=n), 30.0, 98.0)
    x = rng.uniform(bounds["min_x"] + 0.8, bounds["max_x"] - 0.8, size=n)
    y = rng.uniform(bounds["min_y"] + 0.8, bounds["max_y"] - 0.8, size=n)
    names = [
        "Central Medical",
        "Eastside Hospital",
        "South General",
        "West Care",
        "Northstar Regional",
        "Sunset Community",
    ]
    rows = []
    for k in range(n):
        rows.append(
            {
                "Hospital": names[k % len(names)],
                "x": float(x[k]),
                "y": float(y[k]),
                "occupancy": round(float(occ[k]), 1),
                "base_delay": float(5 + rng.uniform(0, 14)),
            }
        )
    return pd.DataFrame(rows)


def generate_incidents(
    n: int,
    rng: np.random.Generator,
    severity_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build an incidents DataFrame with synthetic severity labels."""
    severity_counts = severity_counts or {"Critical": 2, "High": 3, "Medium": 4, "Low": 3}
    x = rng.uniform(0.5, 9.5, size=n)
    y = rng.uniform(0.5, 9.5, size=n)

    pool: list[str] = []
    for sev, cnt in severity_counts.items():
        pool.extend([sev] * cnt)
    rng.shuffle(pool)
    if not pool:
        pool = ["Medium"]
    severities = [pool[i % len(pool)] for i in range(n)]
    urgency_map = {"Critical": 0.9, "High": 0.7, "Medium": 0.5, "Low": 0.3}

    return pd.DataFrame(
        {
            "Incident": [f"INC-{i + 100:03d}" for i in range(n)],
            "x": x,
            "y": y,
            "severity": severities,
            "urgency": [urgency_map[s] for s in severities],
        }
    )



# ---------------------------------------------------------- full scenario
def generate_scenario(
    n_ambulances: int = 6,
    n_hospitals: int = 4,
    n_incidents: int = 9,
    seed: int | None = None,
    congestion: float = 25.0,
    demand: float = 20.0,
    load: float = 30.0,
    uncertainty: float = 0.0,
    risk_weight: float = 1.0,
) -> dict:
    """Generate a complete, self-consistent synthetic scenario.

    Returns a dict with keys ``ambulances``, ``hospitals``, ``incidents`` plus
    the scenario parameters used. All generators receive the *same* rng, so the
    incident positions and resource states are tied to each other.
    """
    rng = _seed_rng(seed)
    ambulances = generate_ambulances(n_ambulances, rng)
    hospitals = generate_hospitals(n_hospitals, rng)
    incidents = generate_incidents(n_incidents, rng)

    scenario = {
        "ambulances": ambulances,
        "hospitals": hospitals,
        "incidents": incidents,
        "congestion": float(_clip(congestion, 0.0, 100.0)),
        "demand": float(_clip(demand, 0.0, 100.0)),
        "load": float(_clip(load, 0.0, 100.0)),
        "uncertainty": float(_clip(uncertainty, 0.0, 100.0)),
        "risk_weight": float(_clip(risk_weight, 0.0, 10.0)),
        "seed": seed,
    }
    return scenario


if __name__ == "__main__":
    import pprint

    s = generate_scenario(seed=42, n_incidents=9)
    pprint.pprint(s)
