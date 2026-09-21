# RISNET: closed-loop RIS control study (Section 7.9)

Self-contained re-implementation of the RIS-assisted 6G topology model of the paper and the closed-loop
evaluation of predictive RIS activation. **This is not the code that produced Tables 3-9.** It follows the
same parameters (28 GHz, 2 BS, 20 UE, 500 x 500 m, d_max = 300 m, SINR threshold -5 dB, Random Waypoint) but its
absolute topology statistics differ (see Section 7.9 of the paper).

## Files
- `sim.py`      mobility (uniform and crowd Random Waypoint), link admission, graph metrics
- `data.py`     episodes, features, labels, windows
- `predictor.py` GAT-LSTM link predictor and the no-attention ablation (PyTorch)
- `control.py`  utility and the policies (static, random, reactive, constant-velocity, learned, oracle)
- `run_all.py`  end-to-end run (training + evaluation), writes `results/*.csv`
- `analyze.py`  tables and Fig. 4 -> `results/summary.json`, `results/fig4_closed_loop.png`
- `results/`    the outputs used in the paper (per-episode CSVs, link-prediction CSV, summary)

## Run
```
pip install numpy scipy pandas matplotlib torch
python run_all.py --out results      # about 14 min on one CPU core
python analyze.py results
python run_all.py --quick --out results_quick --taus 3   # smoke test, about 15 s
```
Seeds: train 1000+, validation 2000+, test 3000+ (per scenario/speed block of 100). Torch seed 0.
Numerical results can differ slightly across PyTorch versions and CPUs.
