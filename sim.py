"""Minimal, self-contained re-implementation of the RIS-assisted 6G topology model of the paper
(Sections 3-4): free-space links at 28 GHz, Random Waypoint mobility, and the link-admission rule
(distance <= d_max, SINR >= gamma_min with inter-BS interference, RIS hops admitted by distance).
It is used for the closed-loop study of Section 7.9 and is NOT the code that produced Tables 3-9."""
import numpy as np

AREA = 500.0
BS = np.array([[150.0, 250.0], [350.0, 250.0]])
# candidate RIS sites: the two sites of the paper (first two rows) plus four corner-side sites
RIS_SITES = np.array([[250.0, 375.0], [250.0, 125.0], [80.0, 80.0], [420.0, 80.0], [80.0, 420.0], [420.0, 420.0]])
N_BS, N_RIS, N_UE = 2, len(RIS_SITES), 20
D_MAX = 300.0
FC, C0 = 28e9, 3e8
LAM = C0 / FC
P_BS_DBM, G_BS_DBI, G_UE_DBI = 46.0, 15.0, 0.0
NOISE_DBM = -94.0 + 9.0            # thermal noise power + noise figure
GAMMA_MIN_DB = -5.0
DT = 0.1                            # simulator step (s)
PAUSE = 2.0                         # RWP pause time (s)
CTRL_STEP = 10                      # one control interval = 10 simulator steps = 1 s

def pl_db(d):
    return 20 * np.log10(4 * np.pi * np.maximum(d, 1.0) / LAM)

def rx_dbm(d):
    return P_BS_DBM + G_BS_DBI + G_UE_DBI - pl_db(d)

HOTSPOTS = np.array([[100.0, 100.0], [400.0, 100.0], [250.0, 420.0]])
HOT_SIGMA, HOT_FRAC, HOT_PERIOD = 40.0, 0.8, 100.0     # crowd scenario: std (m), share of waypoints, dwell (s)

class RWP:
    """Random Waypoint mobility for N_UE users. scenario='uniform' draws waypoints uniformly in the area
    (the paper's model); scenario='crowd' draws a share HOT_FRAC of them around the currently popular
    hotspot, which changes cyclically every HOT_PERIOD seconds (non-stationary demand)."""
    def __init__(self, rng, vmin, vmax, n=N_UE, scenario='uniform'):
        self.rng, self.vmin, self.vmax, self.n, self.scenario = rng, vmin, vmax, n, scenario
        self.time = 0.0
        self.pos = rng.uniform(0, AREA, (n, 2))
        self.wp = self._waypoints(n)
        self.spd = rng.uniform(vmin, vmax, n)
        self.pause = np.zeros(n)

    def _waypoints(self, k):
        w = self.rng.uniform(0, AREA, (k, 2))
        if self.scenario == 'crowd':
            h = HOTSPOTS[int(self.time // HOT_PERIOD) % len(HOTSPOTS)]
            near = self.rng.random(k) < HOT_FRAC
            w[near] = np.clip(h + self.rng.normal(0, HOT_SIGMA, (int(near.sum()), 2)), 0, AREA)
        return w

    def step(self):
        self.time += DT
        moving = self.pause <= 0
        vec = self.wp - self.pos
        dist = np.linalg.norm(vec, axis=1)
        stepl = self.spd * DT
        arrive = moving & (dist <= stepl)
        go = moving & ~arrive
        self.pos[go] += vec[go] / dist[go, None] * stepl[go, None]
        self.pos[arrive] = self.wp[arrive]
        self.pause[arrive] = PAUSE
        # paused users: count down, then draw a new waypoint and speed
        paused = (~moving)
        self.pause[paused] -= DT
        resume = paused & (self.pause <= 0)
        k = int(resume.sum())
        if k:
            self.wp[resume] = self._waypoints(k)
            self.spd[resume] = self.rng.uniform(self.vmin, self.vmax, k)
            self.pause[resume] = 0.0

def trajectory(seed, vmin, vmax, n_ctrl, scenario='uniform'):
    """UE positions sampled at every control interval: array (n_ctrl, N_UE, 2)."""
    rng = np.random.default_rng(seed)
    m = RWP(rng, vmin, vmax, scenario=scenario)
    out = np.empty((n_ctrl, N_UE, 2))
    for t in range(n_ctrl):
        out[t] = m.pos
        for _ in range(CTRL_STEP):
            m.step()
    return out

# ---------------------------------------------------------------- link admission
def bs_ue_edges(ue):
    """(N_BS, N_UE) boolean: direct BS-UE links (distance and SINR with inter-BS interference)."""
    d = np.linalg.norm(BS[:, None, :] - ue[None, :, :], axis=2)             # (2, U)
    p = 10 ** (rx_dbm(d) / 10.0)                                            # mW
    noise = 10 ** (NOISE_DBM / 10.0)
    interf = p[::-1]                                                        # the other BS
    sinr_db = 10 * np.log10(p / (interf + noise))
    return (d <= D_MAX) & (sinr_db >= GAMMA_MIN_DB)

def ris_ue_edges(ue):
    """(N_RIS, N_UE) boolean candidate RIS-UE hops (exist only if that RIS is active)."""
    d = np.linalg.norm(RIS_SITES[:, None, :] - ue[None, :, :], axis=2)
    return d <= D_MAX

_BS_RIS = (np.linalg.norm(BS[:, None, :] - RIS_SITES[None, :, :], axis=2) <= D_MAX)      # (2, N_RIS)
_BS_BS = np.linalg.norm(BS[0] - BS[1]) <= D_MAX

def adjacency(bs_ue, ris_ue, active):
    """Adjacency of the graph on BS + active RIS + UE nodes. active: list of RIS indices."""
    a = list(active)
    n = N_BS + len(a) + N_UE
    A = np.zeros((n, n))
    if _BS_BS:
        A[0, 1] = A[1, 0] = 1
    for k, r in enumerate(a):
        for b in range(N_BS):
            if _BS_RIS[b, r]:
                A[b, N_BS + k] = A[N_BS + k, b] = 1
        A[N_BS + k, N_BS + len(a):] = ris_ue[r]
        A[N_BS + len(a):, N_BS + k] = ris_ue[r]
    u0 = N_BS + len(a)
    for b in range(N_BS):
        A[b, u0:] = bs_ue[b]
        A[u0:, b] = bs_ue[b]
    return A

def metrics(A):
    """(#edges, algebraic connectivity, LCC fraction)"""
    n = len(A)
    deg = A.sum(1)
    L = np.diag(deg) - A
    ev = np.linalg.eigvalsh(L)
    lam2 = float(ev[1])
    # number of components = multiplicity of eigenvalue 0
    ncomp = int((ev < 1e-9).sum())
    if ncomp == 1:
        lcc = 1.0
    else:
        seen = np.zeros(n, bool); best = 0
        for s in range(n):
            if not seen[s]:
                stack = [s]; seen[s] = True; size = 0
                while stack:
                    v = stack.pop(); size += 1
                    for w in np.nonzero(A[v])[0]:
                        if not seen[w]:
                            seen[w] = True; stack.append(w)
                best = max(best, size)
        lcc = best / n
    return A.sum() / 2, lam2, lcc

def graph_metrics(ue, active, bs_ue=None, ris_ue=None):
    bs_ue = bs_ue_edges(ue) if bs_ue is None else bs_ue
    ris_ue = ris_ue_edges(ue) if ris_ue is None else ris_ue
    return metrics(adjacency(bs_ue, ris_ue, active))
