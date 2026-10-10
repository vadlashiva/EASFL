"""Shared parameters, data generator and model for the EASFL-WSN study."""
import numpy as np

# ---------------- network / radio (first-order radio model, Heinzelman et al.) ----
N_NODES = 100
FIELD = 100.0
BS = np.array([50.0, 175.0])
E_ELEC = 50e-9          # J/bit
EPS_FS = 10e-12         # J/bit/m^2
EPS_MP = 0.0013e-12     # J/bit/m^4
D0 = np.sqrt(EPS_FS / EPS_MP)   # ~87.7 m
E_DA = 5e-9             # J/bit/signal, aggregation at CH
E_INIT_LO, E_INIT_HI = 0.25, 0.75   # heterogeneous initial energy (J)
HEADER = 200            # bits per packet
CTRL = 200              # bits per control packet (CH advert / join)
P_CH = 0.1              # desired CH fraction
E_SAMPLE_EPOCH = 2e-6   # J per training-sample pass (MLP, ~4k ops at 0.5 nJ)

# ---------------- model -------------------------------------------------------------
N_FEAT = 24
HID = 16
N_PARAM = N_FEAT * HID + HID + HID + 1   # 417
BITS_F32 = 32 * N_PARAM
BITS_F16 = 16 * N_PARAM
TOPK = 0.10
K_TOP = int(np.ceil(TOPK * N_PARAM))     # 42
IDX_BITS = int(np.ceil(np.log2(N_PARAM)))  # 9
BITS_TOPK8 = K_TOP * (8 + IDX_BITS) + 32
BITS_Q8 = 8 * N_PARAM + 32

def tx_energy(bits, d):
    d = np.asarray(d, float)
    return np.where(d < D0, bits * E_ELEC + bits * EPS_FS * d ** 2,
                    bits * E_ELEC + bits * EPS_MP * d ** 4)

def rx_energy(bits):
    return bits * E_ELEC

# ---------------- synthetic WSN data ------------------------------------------------
WIN = 10
CH_SCALE = np.array([0.15, 0.5, 10.0, 0.005])   # noise std: temp, hum, light, volt

def make_node_profiles(rng, n=N_NODES, rate_scale=1.0, alpha=0.5, amp_scale=1.0):
    """rate_scale: multiplies anomaly rate; alpha: Dirichlet concentration of the
    per-node fault-type mix (small = strongly non-IID); amp_scale: fault magnitude."""
    prof = []
    for i in range(n):
        rate = float(np.clip(rate_scale * rng.lognormal(np.log(0.07), 0.8), 0.01 * rate_scale, min(0.30 * rate_scale, 0.6)))
        mix = rng.dirichlet(np.full(4, alpha))        # spike, stuck, drift, burst
        prof.append(dict(
            T=rng.uniform(18, 30), A=rng.uniform(1, 5), phi=rng.uniform(0, 2 * np.pi),
            H=rng.uniform(30, 60), L=rng.uniform(100, 600), outdoor=rng.random() < 0.4,
            noise=rng.uniform(0.7, 1.6), rate=rate, mix=mix, amp=amp_scale))
    return prof

def gen_windows(rng, p, n_win, t0=0):
    """Return feature matrix (n_win, 24) and labels for one node."""
    n = n_win * WIN
    t = t0 + np.arange(n)                       # minutes
    s = np.sin(2 * np.pi * t / 1440 + p['phi'])
    sc = CH_SCALE * p['noise'] * (np.array([1, 1, 3, 1]) if p['outdoor'] else 1)
    T = p['T'] + p['A'] * s + rng.normal(0, sc[0], n)
    H = p['H'] - 1.5 * (T - p['T']) + rng.normal(0, sc[1], n)
    L = p['L'] * np.maximum(0, s) + rng.normal(0, sc[2], n)
    V = 2.9 - 1e-5 * t + rng.normal(0, sc[3], n)
    X = np.stack([T, H, L, V], 1).reshape(n_win, WIN, 4)
    y = (rng.random(n_win) < p['rate']).astype(int)
    for w in np.where(y)[0]:
        kind = rng.choice(4, p=p['mix'])
        c = rng.integers(4)
        amp = rng.uniform(3, 8) * sc[c] * p.get('amp', 1.0)
        if kind == 0:     # spike
            j = rng.integers(WIN); X[w, j, c] += amp * rng.choice([-1, 1])
        elif kind == 1:   # stuck-at
            X[w, :, c] = X[w, rng.integers(WIN), c]
        elif kind == 2:   # drift
            X[w, :, c] += np.linspace(0, amp * rng.uniform(1, 2), WIN) * rng.choice([-1, 1])
        else:             # noise burst
            X[w, :, c] += rng.normal(0, amp / 2, WIN)
    return features(X), y

def features(X):
    d = np.diff(X, axis=1)
    tt = np.arange(WIN) - (WIN - 1) / 2
    slope = (X * tt[None, :, None]).sum(1) / (tt ** 2).sum()
    resid = X - X.mean(1, keepdims=True) - slope[:, None, :] * tt[None, :, None]
    f = [X.std(1), X.max(1) - X.min(1), np.abs(d).max(1), np.abs(slope),
         np.abs(resid).max(1), (np.abs(d) < 1e-9).mean(1)]
    F = np.concatenate(f, 1) / np.tile(CH_SCALE, 6)
    F[:, -4:] *= CH_SCALE        # stuck fraction is unitless
    return np.log1p(F)

def build_dataset(seed, n_train=200, n_test=100, **gen):
    rng = np.random.default_rng(seed)
    prof = make_node_profiles(rng, **gen)
    tr, te = [], []
    for p in prof:
        tr.append(gen_windows(rng, p, n_train, 0))
        te.append(gen_windows(rng, p, n_test, n_train * WIN))
    allX = np.concatenate([x for x, _ in tr])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6       # calibration statistics
    tr = [((x - mu) / sd, y) for x, y in tr]
    te = [((x - mu) / sd, y) for x, y in te]
    return prof, tr, te

# ---------------- tiny MLP (24-16-1) ------------------------------------------------
def init_params(rng):
    w1 = rng.normal(0, np.sqrt(2 / N_FEAT), (N_FEAT, HID))
    w2 = rng.normal(0, np.sqrt(1 / HID), HID)
    return np.concatenate([w1.ravel(), np.zeros(HID), w2, [0.0]])

def unpack(w):
    i = N_FEAT * HID
    return w[:i].reshape(N_FEAT, HID), w[i:i + HID], w[i + HID:i + 2 * HID], w[-1]

def predict_proba(w, X):
    W1, b1, w2, b2 = unpack(w)
    h = np.maximum(0, X @ W1 + b1)
    return 1 / (1 + np.exp(-np.clip(h @ w2 + b2, -30, 30)))

POS_W = 4.0
LR = 0.15

def sgd(w, X, y, rng, epochs=1, lr=0.05, bs=16, mu=0.0):
    w0 = w; w = w.copy()
    n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            xb, yb = X[b], y[b]
            W1, b1, w2, b2 = unpack(w)
            z = xb @ W1 + b1; h = np.maximum(0, z)
            p = 1 / (1 + np.exp(-np.clip(h @ w2 + b2, -30, 30)))
            g = (p - yb) * np.where(yb == 1, POS_W, 1.0) / len(b)
            gw2 = h.T @ g; gb2 = g.sum()
            gh = np.outer(g, w2) * (z > 0)
            gW1 = xb.T @ gh; gb1 = gh.sum(0)
            w -= lr * (np.concatenate([gW1.ravel(), gb1, gw2, [gb2]]) + mu * (w - w0))
    return w

def metrics(w_or_p, te, thr=0.5, proba=None):
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([x for x, _ in te]); y = np.concatenate([t for _, t in te])
    p = proba if proba is not None else predict_proba(w_or_p, X)
    yh = (p >= thr).astype(int)
    tp = ((yh == 1) & (y == 1)).sum(); fp = ((yh == 1) & (y == 0)).sum()
    fn = ((yh == 0) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec, f1=f1,
                fpr=fp / max(fp + tn, 1), auc=roc_auc_score(y, p))


def best_threshold(p, y):
    """Threshold maximising F1 on a node's own training windows (local, no communication)."""
    if y.sum() == 0:
        return 0.5
    cand = np.unique(np.concatenate([[0.5], np.quantile(p, np.linspace(0.5, 0.99, 25))]))
    best, bt = -1, 0.5
    for t in cand:
        yh = p >= t; tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best: best, bt = f, t
    return bt
