"""Closed-loop RIS activation control: policies and evaluation (Section 7.9)."""
import itertools, numpy as np
import sim, data

PAIRS = list(itertools.combinations(range(sim.N_RIS), 2))
DEFAULT = (0, 1)                        # the two RIS sites used in the paper
TAU = data.TAU

class Utility:
    """J = 0.5 * (lambda_2 / lambda_ref + E / E_ref); the references are set on training episodes"""
    def __init__(self, lam_ref=1.0, e_ref=1.0):
        self.lam_ref, self.e_ref = lam_ref, e_ref
    def __call__(self, e, lam2):
        return 0.5 * (lam2 / self.lam_ref + e / self.e_ref)

def eval_graph(bs_ue, ris_ue, active, util):
    e, lam2, lcc = sim.metrics(sim.adjacency(bs_ue, ris_ue, active))
    return util(e, lam2), e, lam2, lcc

def best_pair(bs_ue, ris_ue, util):
    best, arg = -1e9, DEFAULT
    for p in PAIRS:
        j = eval_graph(bs_ue, ris_ue, p, util)[0]
        if j > best + 1e-12:
            best, arg = j, p
    return arg

def decisions(ep, policy, util, rng=None, model_probs=None):
    """decision made at control step t (takes effect at t+TAU); returns list of RIS pairs (or None)"""
    pos = ep['pos']; T = len(pos); out = [DEFAULT] * T
    for t in range(T):
        if policy == 'static' or policy == 'none':
            out[t] = DEFAULT
        elif policy == 'random':
            out[t] = PAIRS[rng.integers(len(PAIRS))]
        elif policy == 'reactive':
            out[t] = best_pair(sim.bs_ue_edges(pos[t]), sim.ris_ue_edges(pos[t]), util)
        elif policy == 'cv':
            q = data.cv_positions(pos, t)
            out[t] = best_pair(sim.bs_ue_edges(q), sim.ris_ue_edges(q), util)
        elif policy == 'oracle':
            q = pos[min(t + TAU, T - 1)]
            out[t] = best_pair(sim.bs_ue_edges(q), sim.ris_ue_edges(q), util)
        elif policy.startswith('learned'):
            P = model_probs.get(t)                    # (8,20) predicted link probability at t+TAU
            if P is None:
                out[t] = DEFAULT
            else:
                L = P >= 0.5
                out[t] = best_pair(L[:sim.N_BS], L[sim.N_BS:], util)
    return out

def run_episode(ep, policy, util, rng=None, model_probs=None):
    pos = ep['pos']; T = len(pos)
    dec = decisions(ep, policy, util, rng, model_probs) if policy != 'none' else None
    J, E, L, C, sw = [], [], [], [], 0
    prev = None
    for t in range(T):
        if policy == 'none':
            active = ()
        elif t < TAU:
            active = DEFAULT
        else:
            active = dec[t - TAU]
        if prev is not None and policy != 'none':
            sw += len(set(active) ^ set(prev)) // 2
        prev = active
        j, e, lam2, lcc = eval_graph(sim.bs_ue_edges(pos[t]), sim.ris_ue_edges(pos[t]), active, util)
        J.append(j); E.append(e); L.append(lam2); C.append(lcc)
    return dict(J=np.mean(J), E=np.mean(E), lam2=np.mean(L), lcc=np.mean(C), reconf_per_100s=sw / T * 100)
