# data/ — synthetic data notes

**All data used by RESQ-X is generated at runtime and is entirely fictional.**

Nothing in this folder (or anywhere in the project) contains real patient
information, real ambulance locations, or live dispatch feeds.

## What is generated

The module `src/data_generator.py` builds every scenario from a single integer
seed:

| Entity      | Columns                                        | Synthetic assumptions |
|-------------|------------------------------------------------|-----------------------|
| Ambulances  | `Ambulance`, `station_x/y`, `status`, `type`, `x`, `y` | Positions snap to six fictional stations; ~25% start `Busy`. |
| Hospitals   | `Hospital`, `x`, `y`, `occupancy`, `base_delay` | Occupancy sampled in 30–98 % around a 62 % mean; `base_delay` 5–19 min. |
| Incidents   | `Incident`, `x`, `y`, `severity`, `urgency`    | Uniform positions on a 10×10 abstract city grid; severity mix 2 Critical / 3 High / 4 Medium / 3 Low (scaled by count). |

Coordinates are **abstract map units on a 10×10 grid**, not latitude/longitude.
The dashboard map is illustrative only and does not represent any real city.

## Reproducibility

```python
from src.data_generator import generate_scenario
a = generate_scenario(seed=42)   # identical to
b = generate_scenario(seed=42)   # this
```

## Generated artifacts

Experiment exports (CSV) are written by the Evaluation page when you click
"Export results". They contain only simulated metrics computed by the code —
no measurements from the real world.

Optional: the `src/prediction.py` module can persist a trained travel-time
model to `data/predictor.pkl`. That file is a local artifact of your own run
and is safe to delete at any time.
