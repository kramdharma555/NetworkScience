"""GAT-LSTM link predictor (Section 6.2 of the paper) and its ablation without graph attention."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import data

class GATLayer(nn.Module):
    def __init__(self, din, dout, heads=2):
        super().__init__()
        self.h, self.d = heads, dout
        self.W = nn.Linear(din, dout * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, dout) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, dout) * 0.1)
    def forward(self, h, A):                       # h (B,N,din)  A (B,N,N) 0/1
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.h, self.d)
        es = (Wh * self.a_src).sum(-1)             # (B,N,H)
        ed = (Wh * self.a_dst).sum(-1)
        e = Fn.leaky_relu(es.unsqueeze(2) + ed.unsqueeze(1), 0.2)       # (B,N,N,H)
        e = e.masked_fill(A.unsqueeze(-1) == 0, -1e9)
        alpha = torch.softmax(e, dim=2)
        out = torch.einsum('bijh,bjhd->bihd', alpha, Wh).reshape(B, N, self.h * self.d)
        return Fn.elu(out)

class GATLSTM(nn.Module):
    def __init__(self, fin=7, hid=32, use_gat=True):
        super().__init__()
        self.use_gat = use_gat
        self.inp = nn.Linear(fin, hid)
        if use_gat:
            self.g1 = GATLayer(hid, hid // 2, 2)
            self.g2 = GATLayer(hid, hid // 2, 2)
        else:
            self.m1 = nn.Linear(hid, hid); self.m2 = nn.Linear(hid, hid)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(3 * hid + 4, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, X, A, C):                    # X (B,W,N,F)  A (B,W,N,N)  C (B,8,U) current link state
        B, W, N, Fd = X.shape
        h = Fn.relu(self.inp(X.reshape(B * W, N, Fd)))
        Af = A.reshape(B * W, N, N)
        if self.use_gat:
            h = h + self.g1(h, Af)
            h = h + self.g2(h, Af)
        else:
            h = h + Fn.elu(self.m1(h)); h = h + Fn.elu(self.m2(h))
        h = h.view(B, W, N, -1).permute(0, 2, 1, 3).reshape(B * N, W, -1)
        o, _ = self.lstm(h)
        hT = o[:, -1].view(B, N, -1)
        infra, ue = hT[:, :data.N_INFRA], hT[:, data.N_INFRA:]
        U = ue.shape[1]
        # pairwise geometry at the latest snapshot (edge attributes): relative offset and distance
        pi = X[:, -1, :data.N_INFRA, 3:5] * 500.0
        pu = X[:, -1, data.N_INFRA:, 3:5] * 500.0
        rel = pu[:, None, :, :] - pi[:, :, None, :]                       # (B,8,U,2)
        dist = rel.norm(dim=-1, keepdim=True)
        geo = torch.cat([rel / 500.0, dist / 300.0, C.unsqueeze(-1)], dim=-1)
        pair = torch.cat([infra[:, :, None].expand(-1, -1, U, -1),
                          ue[:, None].expand(-1, infra.shape[1], -1, -1),
                          infra[:, :, None] * ue[:, None], geo], dim=-1)
        return self.head(pair).squeeze(-1)         # logits (B,8,U)

def _batches(n, bs, rng=None):
    idx = np.arange(n) if rng is None else rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]

def predict(model, X, A, C, bs=512):
    model.eval(); out = []
    with torch.no_grad():
        for b in _batches(len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[b]), torch.from_numpy(A[b]),
                                           torch.from_numpy(C[b]).float())).numpy())
    return np.concatenate(out)

def train(Xtr, Atr, Ctr, Ytr, Xva, Ava, Cva, Yva, use_gat=True, epochs=10, seed=0, lr=2e-3, bs=128, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = GATLSTM(use_gat=use_gat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); tot = 0.0
        for b in _batches(len(Xtr), bs, rng):
            x = torch.from_numpy(Xtr[b]); a = torch.from_numpy(Atr[b]); y = torch.from_numpy(Ytr[b]).float()
            c = torch.from_numpy(Ctr[b]).float()
            loss = Fn.binary_cross_entropy_with_logits(model(x, a, c), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        pv = predict(model, Xva, Ava, Cva)
        vl = float(-np.mean(Yva * np.log(pv + 1e-7) + (1 - Yva) * np.log(1 - pv + 1e-7)))
        log('  epoch %2d  train BCE %.4f  val BCE %.4f' % (ep + 1, tot / len(Xtr), vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best
