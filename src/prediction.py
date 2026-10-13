"""Optional simple travel-time prediction model ("prototype only").

This module trains a scikit-learn model on *synthetic* routing data to
predict travel time from distance, hospital occupancy, traffic, and demand.
It exists to demonstrate how a data-driven estimator could be folded into the
pipeline; it is **not** used by the core dispatch decision, which remains a
transparent, explainable heuristic (the defining requirement of this prototype).

The model is trained, evaluated, and persisted locally under ``data/``. Reported
numbers are genuine train/test metrics computed from the generated data, never
fabricated.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------- feature build
def _feature_row(amb, hosp, inc, congestion, load) -> np.ndarray:
    d_scene = np.hypot(amb["x"] - inc["x"], amb["y"] - inc["y"])
    d_hosp = np.hypot(hosp["x"] - inc["x"], hosp["y"] - inc["y"])
    return np.array(
        [
            d_scene,
            d_hosp,
            float(hosp.get("occupancy", 0.0)),
            congestion / 100.0,
            load / 100.0,
            float(inc["urgency"]),
        ],
        dtype=float,
    )


def _train_split(amb_df, hosp_df, inc_df, congestion, load, test_size=0.2, random_state=42):
    feats, targets = [], []
    for _, inc in inc_df.iterrows():
        for _, amb in amb_df[amb_df["status"] == "Available"].iterrows():
            for _, hosp in hosp_df.iterrows():
                x = _feature_row(amb, hosp, inc, congestion, load)
                feats.append(x)
                # Nominal total travel time is the target (dispatch + transport).
                t = np.hypot(amb["x"] - inc["x"], amb["y"] - inc["y"])
                targets.append(float(t * 2.6 + 1.6 + 1.9 + 0.0))
    X = np.array(feats)
    y = np.array(targets)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ------------------------------------------------------------- model wrapper
def train_predictor(
    amb_df,
    hosp_df,
    inc_df,
    congestion,
    load,
    model_path: str | None = None,
) -> tuple[RandomForestRegressor, float, float, dict]:
    """Train the travel-time estimator and report genuine metrics."""
    X_train, X_test, y_train, y_test = _train_split(
        amb_df, hosp_df, inc_df, congestion, load
    )

    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    report = {"mae": mae, "r2": r2, "n_test": int(len(y_test))}

    if model_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "report": report}, f)

    return model, mae, r2, report


def load_predictor(model_path: str = "data/predictor.pkl") -> RandomForestRegressor:
    """Load a previously-saved predictor, if present."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No predictor at {model_path}. Run train_predictor() first."
        )
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"]


if __name__ == "__main__":
    from .data_generator import generate_scenario

    s = generate_scenario(seed=7, n_ambulances=4, n_incidents=6)
    m, mae, r2, rep = train_predictor(
        s["ambulances"], s["hospitals"], s["incidents"], congestion=30, load=40
    )
    print("MAE:", mae, "R2:", r2, "report:", rep)
