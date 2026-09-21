"""Episode generation, features and labels for the link-prediction task."""
import numpy as np
import sim

TAU = 3          # prediction / activation horizon in control steps (3 s)
W = 5            # history window (snapshots)
N_INFRA = sim.N_BS + sim.N_RIS          # 8
N_NODES = N_INFRA + sim.N_UE            # 28
V_NORM = 15.0
REGIMES = {'pedestrian': (1.0, 5.0), 'vehicular': (5.0, 15.0)}

def link_labels(ue):
    """(8, 20) link existence: BS rows = admitted direct links, RIS rows = candidate hops."""
    return np.vstack([sim.bs_ue_edges(ue), sim.ris_ue_edges(ue)])

_INFRA_POS = np.vstack([sim.BS, sim.RIS_SITES])

def velocities(pos):
    v = np.zeros_like(pos)
    v[1:] = pos[1:] - pos[:-1]          # displacement per 1-s control step = m/s
    return v

def features(pos):
    """pos (T,20,2) -> X (T,28,7), A (T,28,28) potential-link graph."""
    T = len(pos)
    v = velocities(pos)
    X = np.zeros((T, N_NODES, 7), dtype=np.float32)
    X[:, :sim.N_BS, 0] = 1
    X[:, sim.N_BS:N_INFRA, 1] = 1
    X[:, N_INFRA:, 2] = 1
    X[:, :N_INFRA, 3:5] = _INFRA_POS[None] / sim.AREA
    X[:, N_INFRA:, 3:5] = pos / sim.AREA
    X[:, N_INFRA:, 5:7] = v / V_NORM
    A = np.zeros((T, N_NODES, N_NODES), dtype=np.uint8)
    A[:, :N_INFRA, :N_INFRA] = 1
    d = np.linalg.norm(_INFRA_POS[None, :, None, :] - pos[:, None, :, :], axis=3)   # (T,8,20)
    m = (d <= sim.D_MAX).astype(np.uint8)
    A[:, :N_INFRA, N_INFRA:] = m
    A[:, N_INFRA:, :N_INFRA] = m.transpose(0, 2, 1)
    A[:, np.arange(N_NODES), np.arange(N_NODES)] = 1
    return X, A

def episode(seed, regime, n_ctrl=300, scenario='uniform'):
    vmin, vmax = REGIMES[regime]
    pos = sim.trajectory(seed, vmin, vmax, n_ctrl, scenario)
    X, A = features(pos)
    Y = np.stack([link_labels(p) for p in pos]).astype(np.uint8)     # (T,8,20)
    return dict(pos=pos, X=X, A=A, Y=Y, seed=seed, regime=regime, scenario=scenario)

def cv_positions(pos, t, h=None):
    """constant-velocity extrapolation of the UE positions from time t to t+h"""
    h = TAU if h is None else h
    v = pos[t] - pos[t - 1] if t > 0 else np.zeros_like(pos[t])
    return np.clip(pos[t] + v * h, 0, sim.AREA)

def make_samples(eps):
    """stack windows: X (n,W,28,7), A (n,W,28,28), Y (n,8,20) = links at t+TAU, plus current Y0 and t index"""
    Xs, As, Ys, Y0, meta = [], [], [], [], []
    for k, e in enumerate(eps):
        T = len(e['pos'])
        for t in range(W - 1, T - TAU):
            Xs.append(e['X'][t - W + 1:t + 1]); As.append(e['A'][t - W + 1:t + 1])
            Ys.append(e['Y'][t + TAU]); Y0.append(e['Y'][t]); meta.append((k, t))
    return (np.stack(Xs), np.stack(As), np.stack(Ys), np.stack(Y0), np.array(meta))
