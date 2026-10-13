"""Decision-engine API.

Exposes a single, easy-to-read entry point for both dispatch strategies:

    result = dispatch(scenario, strategy="resqx" | "baseline", ids=...)

It also wraps the underlying *simulator* so the dashboard and the experiment
pipeline stay in sync. No hidden randomness: every call uses the scenario as
given and returns deterministic results.
"""

from __future__ import annotations

import pandas as pd

from .simulator import schedule_incidents, summary_of


def dispatch(
    scenario: dict,
    strategy: str = "baseline",
    incident_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Run the named strategy over a pre-generated scenario."""
    strategy = strategy.lower()
    if strategy not in {"baseline", "resqx"}:
        raise ValueError(f"Unknown strategy {strategy!r}. Use 'baseline' or 'resqx'.")
    return schedule_incidents(scenario, strategy=strategy, incident_ids=incident_ids)


def summarize(
    scenario: dict,
    strategy: str = "baseline",
    incident_ids: list[str] | None = None,
    amb_df: pd.DataFrame | None = None,
) -> dict:
    """Return aggregate metrics for one strategy on one scenario."""
    res = dispatch(scenario, strategy=strategy, incident_ids=incident_ids)
    if amb_df is None:
        amb_df = scenario.get("ambulances")
    return summary_of(res, amb_df=amb_df)
