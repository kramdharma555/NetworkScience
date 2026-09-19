"""Regenerate Fig. 2 (edges and density vs. number of RIS) from data/table8_scaling_ris.csv."""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("data/table8_scaling_ris.csv")))
nr = [int(r["N_R"]) for r in rows]
edges = [int(r["edges"]) for r in rows]
rho = [float(r["density"]) for r in rows]

fig, ax = plt.subplots(figsize=(3.3, 2.2), dpi=600)
ax2 = ax.twinx()
l1, = ax.plot(nr, edges, "-s", ms=3.2, lw=1.0, color="#1f3a93")
l2, = ax2.plot(nr, rho, "--^", ms=3.4, lw=1.0, color="#c0392b")
ax.set_xlabel("Number of RIS, $N_R$", fontsize=8)
ax.set_ylabel("Number of edges", fontsize=8)
ax2.set_ylabel(r"Network density $\rho$", fontsize=8)
ax.set_ylim(0, 250); ax2.set_ylim(0, 0.5); ax.set_xticks(nr)
ax.legend([l1, l2], ["Edges", r"Density $\rho$"], loc="upper left", fontsize=7)
fig.tight_layout(pad=0.4)
fig.savefig("figures/fig2_scalability_ris.png", dpi=600)
