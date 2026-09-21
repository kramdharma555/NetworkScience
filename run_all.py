"""End-to-end closed-loop study (Section 7.9).  Usage:  python run_all.py [--quick]"""
import argparse, json, os, time, itertools
import numpy as np
import sim, data, control

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--out', default='results')
ap.add_argument('--taus', default='3,10')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
import predictor, torch
torch.set_num_threads(1)

N_TRAIN, N_VAL, N_TEST, EPOCHS, N_CTRL = (3, 1, 3, 2, 120) if args.quick else (24, 6, 30, 8, 300)
COMBOS = list(itertools.product(['uniform', 'crowd'], ['pedestrian', 'vehicular']))
SEED0 = {'train': 1000, 'val': 2000, 'test': 3000}

def make(split, n):
    out = []
    for ci, (scen, reg) in enumerate(COMBOS):
        for k in range(n):
            out.append(data.episode(SEED0[split] + 100 * ci + k, reg, N_CTRL, scen))
    return out

t0 = time.time()
log = lambda *a: print('[%5.0fs]' % (time.time() - t0), *a, flush=True)
train_eps, val_eps, test_eps = make('train', N_TRAIN), make('val', N_VAL), make('test', N_TEST)
log('episodes', len(train_eps), len(val_eps), len(test_eps))

# utility references from the static policy on training episodes (fixed before any evaluation)
base = [control.run_episode(e, 'static', control.Utility(1, 1)) for e in train_eps]
util = control.Utility(lam_ref=float(np.mean([b['lam2'] for b in base])), e_ref=float(np.mean([b['E'] for b in base])))
log('utility refs: lambda2 %.3f  E %.2f' % (util.lam_ref, util.e_ref))
json.dump(dict(lam_ref=util.lam_ref, e_ref=util.e_ref), open(f'{args.out}/utility_refs.json', 'w'))

def prf(pred, true):
    tp = float((pred & true).sum()); fp = float((pred & ~true).sum()); fn = float((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

link_rows, ctl_rows = [], []
for tau in [int(x) for x in args.taus.split(',')]:
    data.TAU = tau; control.TAU = tau
    log('=== horizon / activation delay tau = %d s ===' % tau)
    Xtr, Atr, Ytr, Ctr, _ = data.make_samples(train_eps)
    Xva, Ava, Yva, Cva, _ = data.make_samples(val_eps)
    models = {}
    for name, use_gat in [('GAT-LSTM', True), ('LSTM (no graph attention)', False)]:
        log('training', name, 'samples', len(Xtr))
        m, best = predictor.train(Xtr, Atr, Ctr, Ytr, Xva, Ava, Cva, Yva, use_gat=use_gat, epochs=EPOCHS, seed=0, log=log)
        models[name] = m
    # ---- link prediction on the test episodes -------------------------------------------------
    Xte, Ate, Yte, Y0te, meta = data.make_samples(test_eps)
    tags = np.array([(test_eps[k]['scenario'], test_eps[k]['regime']) for k in meta[:, 0]])
    probs = {n: predictor.predict(m, Xte, Ate, Y0te) for n, m in models.items()}
    cvp = np.zeros_like(Yte, dtype=bool)
    for i, (k, t) in enumerate(meta):
        q = data.cv_positions(test_eps[k]['pos'], t, tau)
        cvp[i] = data.link_labels(q).astype(bool)
    preds = {'Persistence': Y0te.astype(bool), 'Constant velocity': cvp}
    preds.update({n: p >= 0.5 for n, p in probs.items()})
    true = Yte.astype(bool); cur = Y0te.astype(bool); flip = true != cur
    for name, pr in preds.items():
        for reg in ['pedestrian', 'vehicular']:
            sel = tags[:, 1] == reg
            acc = float((pr[sel] == true[sel]).mean())
            _, _, f1 = prf(pr[sel], true[sel])
            fp_, fr_, ff = prf((pr != cur)[sel], flip[sel])
            link_rows.append(dict(tau=tau, method=name, regime=reg, accuracy=acc, link_f1=f1, flip_precision=fp_, flip_recall=fr_, flip_f1=ff,
                                  flip_rate=float(flip[sel].mean())))
    # ---- closed-loop control ------------------------------------------------------------------
    per_ep = {}
    mp = {n: {} for n in models}
    for i, (k, t) in enumerate(meta):
        for n in models:
            mp[n].setdefault(k, {})[int(t)] = probs[n][i]
    for k, ep in enumerate(test_eps):
        rng = np.random.default_rng(9000 + k)
        row = {}
        for pol in ['none', 'static', 'random', 'reactive', 'cv', 'oracle']:
            row[pol] = control.run_episode(ep, pol, util, rng)
        row['learned GAT-LSTM'] = control.run_episode(ep, 'learned', util, model_probs=mp['GAT-LSTM'][k])
        row['learned LSTM (no graph attention)'] = control.run_episode(ep, 'learned', util, model_probs=mp['LSTM (no graph attention)'][k])
        for pol, r in row.items():
            ctl_rows.append(dict(tau=tau, scenario=ep['scenario'], regime=ep['regime'], episode=k, policy=pol, **{a: float(b) for a, b in r.items()}))
    log('control done for tau', tau)

import pandas as pd
pd.DataFrame(link_rows).to_csv(f'{args.out}/link_prediction.csv', index=False)
pd.DataFrame(ctl_rows).to_csv(f'{args.out}/closed_loop_episodes.csv', index=False)
log('saved results to', args.out)
