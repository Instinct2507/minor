# RESQ-X Prototype v1

A runnable Streamlit demo for **Risk-Aware Counterfactual AI for Emergency Resource Decision-Making Under Uncertainty**.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Features
- Synthetic ambulance, incident, and hospital data
- Operations map and recommended assignments
- Nearest-available baseline comparison
- Traffic, demand, hospital load, and risk-aversion controls
- Counterfactual stress scenario
- Data and assumptions view

## Limitations
All data are fictional. The current score is a transparent rule-based heuristic, not a trained predictive model. It is not clinically validated and must not be used for real emergency dispatch. Run repeated randomized simulations and document metrics before claiming improvements.
