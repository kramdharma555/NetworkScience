"""Summarize results/*.csv into the tables and figure of Section 7.9."""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else 'results'
link = pd.read_csv(f'{R}/link_prediction.csv')
ep = pd.read_csv(f'{R}/closed_loop_episodes.csv')
POL = ['reactive', 'cv', 'learned LSTM (no graph attention)', 'learned GAT-LSTM', 'oracle']
NAME = {'reactive': 'Reactive', 'cv': 'Constant-velocity predictive', 'learned LSTM (no graph attention)': 'LSTM predictive (no graph attention)',
        'learned GAT-LSTM': 'GAT–LSTM predictive', 'oracle': 'Oracle (upper bound)', 'random': 'Random pair', 'none': 'No RIS'}
COMBOS = [('uniform', 'pedestrian'), ('uniform', 'vehicular'), ('crowd', 'pedestrian'), ('crowd', 'vehicular')]

def ci(x):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return m, h

def delta(pol, tau, scen, reg, ref='static'):
    a = ep[(ep.tau == tau) & (ep.scenario == scen) & (ep.regime == reg) & (ep.policy == pol)].sort_values('episode')
    b = ep[(ep.tau == tau) & (ep.scenario == scen) & (ep.regime == reg) & (ep.policy == ref)].sort_values('episode')
    return 100.0 * (a['J'].values - b['J'].values) / b['J'].values

out = {}
# ---- Table 10: link prediction
rows = []
for m in ['Persistence', 'Constant velocity', 'LSTM (no graph attention)', 'GAT-LSTM']:
    r = [m if m != 'GAT-LSTM' else 'GAT–LSTM']
    for tau in sorted(link.tau.unique()):
        for reg in ['pedestrian', 'vehicular']:
            x = link[(link.method == m) & (link.tau == tau) & (link.regime == reg)].iloc[0]
            r += ['%.1f' % (100 * x.accuracy), '%.2f' % x.flip_f1]
    rows.append(r)
out['table10'] = rows
out['flip_rates'] = {f'{t}-{g}': float(link[(link.tau == t) & (link.regime == g)].flip_rate.iloc[0]) for t in link.tau.unique() for g in ['pedestrian', 'vehicular']}
# ---- Table 11: closed loop
t11 = []
for tau in sorted(ep.tau.unique()):
    for pol in POL:
        r = [NAME[pol], int(tau)]
        for scen, reg in COMBOS:
            m, h = ci(delta(pol, tau, scen, reg)); r.append('%+.2f ± %.2f' % (m, h))
        t11.append(r)
out['table11'] = t11
# extra: differences vs reactive, reconfiguration rates, means
extra = {}
for tau in sorted(ep.tau.unique()):
    for scen, reg in COMBOS:
        key = f'{int(tau)}|{scen}|{reg}'
        d = {}
        for pol in ['cv', 'learned GAT-LSTM', 'learned LSTM (no graph attention)', 'oracle', 'reactive']:
            m, h = ci(delta(pol, tau, scen, reg, ref='reactive')); d['vs_reactive_' + pol] = (m, h)
        m, h = ci(delta('learned GAT-LSTM', tau, scen, reg, ref='cv')); d['gat_vs_cv'] = (m, h)
        sub = ep[(ep.tau == tau) & (ep.scenario == scen) & (ep.regime == reg)]
        d['reconf'] = {p: float(sub[sub.policy == p].reconf_per_100s.mean()) for p in ['reactive', 'cv', 'learned GAT-LSTM', 'oracle', 'random']}
        d['abs'] = {p: dict(J=float(sub[sub.policy == p].J.mean()), E=float(sub[sub.policy == p].E.mean()), lam2=float(sub[sub.policy == p].lam2.mean()),
                            lcc=float(sub[sub.policy == p].lcc.mean())) for p in ['none', 'static', 'random', 'reactive', 'cv', 'learned GAT-LSTM', 'oracle']}
        extra[key] = d
out['extra'] = extra
json.dump(out, open(f'{R}/summary.json', 'w'), indent=1, default=float)

# ---- figure: gain over static, crowd scenario
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
pols = ['reactive', 'cv', 'learned GAT-LSTM', 'oracle']
lab = ['Reactive', 'Const.-vel.', 'GAT–LSTM', 'Oracle']
hatch = ['', '//', 'xx', '..']; grey = ['0.85', '0.6', '0.35', '1.0']
taus = sorted(ep.tau.unique())
for ax, reg in zip(axes, ['pedestrian', 'vehicular']):
    w = 0.19
    for i, p in enumerate(pols):
        ms, hs = [], []
        for tau in taus:
            m, h = ci(delta(p, tau, 'crowd', reg)); ms.append(m); hs.append(h)
        ax.bar(np.arange(len(taus)) + (i - 1.5) * w, ms, w, yerr=hs, capsize=1.5, error_kw=dict(lw=0.7, capthick=0.7), color=grey[i], edgecolor='k', hatch=hatch[i], linewidth=0.6, label=lab[i])
    ax.set_xticks(range(len(taus))); ax.set_xticklabels(['τ = %d s' % t for t in taus], fontsize=8)
    ax.set_title('Crowd scenario, %s users' % reg, fontsize=8); ax.tick_params(labelsize=7); ax.axhline(0, color='k', lw=0.5)
axes[0].set_ylabel('Utility gain over static RIS pair (%)', fontsize=8)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7, loc='lower center', ncol=4, frameon=False)
fig.tight_layout(pad=0.4, rect=(0, 0.08, 1, 1)); fig.savefig(f'{R}/fig4_closed_loop.png', dpi=300)
print(json.dumps(out['table10'], indent=0)); print(json.dumps(out['table11'], indent=0))
