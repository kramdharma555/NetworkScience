# Data dictionary
CSV files reproduce the result tables of the paper (Tables 2-9). They are the published
aggregate results, not raw per-step traces.

Notes
- `table8_scaling_ris.csv`: density and average degree are recomputed from node/edge counts
  (density = 2E / (V(V-1))); `delta_edges_per_added_ris` is the increment relative to the previous row.
- `table3_topology_scenarios.csv`: the No-RIS scenario keeps the RIS nodes (|V|=24) with reflection disabled,
  so it is not comparable with N_R=0 in table 8 (|V|=22).
- Add raw per-step time series (e.g. `raw/seed_42_timeseries.csv`) here once exported from the simulator.
