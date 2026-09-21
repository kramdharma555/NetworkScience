# RIS-Assisted 6G Topology Analysis: Data and Code

Companion repository for the paper *Network-Science Analysis of Dynamic Multi-Layer RIS-Assisted 6G Networks:
Topology Impact and Predictive Control Framework* (submitted to IEEE Open Journal of the Communications Society).

## Contents
| Folder | Description |
|---|---|
| `data/` | Result tables of the paper (Tables 2-9) as CSV, plus a data dictionary |
| `scripts/` | `make_fig2.py` regenerates Fig. 2 from `data/table8_scaling_ris.csv` |
| `figures/` | Generated figures |
| `simulation/` | Simulator code (to be added) |

## Quick start
```bash
pip install -r requirements.txt
python scripts/make_fig2.py
```

## Setup used in the paper
28 GHz, 2 BS, 2 RIS (64 elements), 20 UE, 500x500 m, Random Waypoint, dt = 0.1 s, T = 50 steps, seed 42.

## Citation
See `CITATION.cff`. License: MIT (code); the CSV data may be reused with attribution.
