"""E12b: detector baselines (centralized, pooled training windows of all nodes = upper bound).

  MLP 24-16-1 on the 24 summary features (the paper's detector, 417 parameters)
  Isolation forest on the same features (unsupervised; contamination = training anomaly rate)
  1D-CNN autoencoder on raw 10x4 windows (trained on normal windows; score = reconstruction error)
  1D-CNN classifier on raw windows (supervised)
  GRU classifier on raw windows (supervised, hidden 16)
Raw windows are centred per window and channel and scaled by the channel noise scale.
Thresholds for the unsupervised scores are chosen to maximise F1 on the training windows.
Seeds 1-3. Implemented with autograd (NumPy). -> results_detectors.json
"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as onp
import autograd.numpy as np
from autograd import grad
from common import build_dataset, init_params, sgd, predict_proba, best_threshold, CH_SCALE


def f1(p, y):
    yh = p >= 0.5; tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def thr_f1(score_tr, ytr, score_te, yte):
    cand = onp.quantile(score_tr, onp.linspace(0.5, 0.995, 60))
    best = max(cand, key=lambda t: f1((score_tr >= t).astype(float), ytr))
    return f1((score_te >= best).astype(float), yte)


def prep(R):
    R = R - R.mean(1, keepdims=True)
    return R / CH_SCALE[None, None, :]


# ---------------------------------------------------------------- tiny autograd models
def conv1d(x, W, b):           # x (B,T,C), W (k,C,F) 'same' padding
    k = W.shape[0]; p = k // 2
    xp = np.concatenate([np.zeros((x.shape[0], p, x.shape[2])), x, np.zeros((x.shape[0], p, x.shape[2]))], 1)
    out = 0
    for j in range(k):
        out = out + np.einsum('btc,cf->btf', xp[:, j:j + x.shape[1], :], W[j])
    return out + b


def init(shapes, rng):
    return [rng.normal(0, onp.sqrt(2.0 / max(1, int(onp.prod(s[:-1])))), s) if len(s) > 1 else onp.zeros(s) for s in shapes]


def cnn_ae(P, x):
    W1, b1, W2, b2 = P
    return conv1d(np.maximum(0, conv1d(x, W1, b1)), W2, b2)


def cnn_clf(P, x):
    W1, b1, W2, b2, w3, b3 = P
    h = np.maximum(0, conv1d(np.maximum(0, conv1d(x, W1, b1)), W2, b2))
    z = np.concatenate([h.max(1), h.mean(1)], 1)
    return z @ w3 + b3


def gru_clf(P, x):
    Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh, w3, b3 = P
    h = np.zeros((x.shape[0], Uz.shape[0]))
    sig = lambda a: 1 / (1 + np.exp(-a))
    for t in range(x.shape[1]):
        xt = x[:, t, :]
        z = sig(xt @ Wz + h @ Uz + bz); r = sig(xt @ Wr + h @ Ur + br)
        hh = np.tanh(xt @ Wh + (r * h) @ Uh + bh)
        h = (1 - z) * h + z * hh
    return h @ w3 + b3


def bce(logit, y, pw=4.0):
    p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    return -np.mean(pw * y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


def adam_train(loss, P, data, epochs, rng, bs=256, lr=3e-3):
    g = grad(loss); m = [onp.zeros_like(p) for p in P]; v = [onp.zeros_like(p) for p in P]; t = 0
    n = len(data[0])
    for _ in range(epochs):
        idx = rng.permutation(n)
        for k in range(0, n, bs):
            j = idx[k:k + bs]; t += 1
            G = g(P, *[d[j] for d in data])
            for i in range(len(P)):
                m[i] = 0.9 * m[i] + 0.1 * G[i]; v[i] = 0.999 * v[i] + 0.001 * G[i] ** 2
                P[i] = P[i] - lr * (m[i] / (1 - 0.9 ** t)) / (onp.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return P


def nparams(P):
    return int(sum(p.size for p in P))


EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def run_seed(sd):
    raw = []
    _, tr, te = build_dataset(sd, raw=raw)
    Ftr = onp.concatenate([x for x, _ in tr]); ytr = onp.concatenate([y for _, y in tr]).astype(float)
    Fte = onp.concatenate([x for x, _ in te]); yte = onp.concatenate([y for _, y in te]).astype(float)
    Rtr = prep(onp.concatenate(raw[0])); Rte = prep(onp.concatenate(raw[1]))
    rng = onp.random.default_rng(sd); out = {}
    t0 = time.time()
    w = init_params(rng)
    for _ in range(30):
        w = sgd(w, Ftr, ytr.astype(int), rng, 1, 0.05, 32)
    out['MLP 24-16-1 (features)'] = dict(f1=f1(predict_proba(w, Fte), yte), params=417)
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=100, contamination=float(ytr.mean()), random_state=sd).fit(Ftr)
    out['Isolation forest (features)'] = dict(f1=thr_f1(-iso.score_samples(Ftr), ytr, -iso.score_samples(Fte), yte), params=None)
    # 1D-CNN autoencoder, trained on normal windows
    P = init([(3, 4, 16), (16,), (3, 16, 4), (4,)], rng)
    loss = lambda P, x: np.mean((cnn_ae(P, x) - x) ** 2)
    P = adam_train(loss, P, [Rtr[ytr == 0]], EP, rng)
    err = lambda X: ((cnn_ae(P, X) - X) ** 2).reshape(len(X), -1).mean(1)
    out['1D-CNN autoencoder (raw)'] = dict(f1=thr_f1(err(Rtr), ytr, err(Rte), yte), params=nparams(P))
    # 1D-CNN classifier
    P = init([(3, 4, 16), (16,), (3, 16, 16), (16,), (32,), (1,)], rng); P[4] = rng.normal(0, 0.2, 32); P[5] = onp.zeros(())
    loss = lambda P, x, y: bce(cnn_clf(P, x), y)
    P = adam_train(loss, P, [Rtr, ytr], EP, rng)
    p = 1 / (1 + onp.exp(-cnn_clf(P, Rte)))
    out['1D-CNN classifier (raw)'] = dict(f1=f1(p, yte), params=nparams(P))
    # GRU classifier
    H = 16
    P = init([(4, H), (H, H), (H,), (4, H), (H, H), (H,), (4, H), (H, H), (H,), (H,), (1,)], rng); P[9] = rng.normal(0, 0.3, H); P[10] = onp.zeros(())
    loss = lambda P, x, y: bce(gru_clf(P, x), y)
    P = adam_train(loss, P, [Rtr, ytr], EP, rng)
    p = 1 / (1 + onp.exp(-gru_clf(P, Rte)))
    out['GRU classifier (raw)'] = dict(f1=f1(p, yte), params=nparams(P))
    print(sd, {k: round(v['f1'], 3) for k, v in out.items()}, round(time.time() - t0), 's', flush=True)
    return out


if __name__ == '__main__':
    R = {sd: run_seed(sd) for sd in (1, 2, 3)}
    json.dump(R, open('results_detectors.json', 'w'))
    for k in R[1]:
        v = [R[s][k]['f1'] for s in R]
        print(f'{k:30s} F1={onp.mean(v):.3f}±{onp.std(v):.3f} params={R[1][k]["params"]}')
