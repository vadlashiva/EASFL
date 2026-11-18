"""E2 + E3 (revision): detection quality over the whole network lifetime, and local-only.

Learning continues every round until the network is dead (or MAXR). Every EVAL rounds we
record, with per-node fine-tuning + threshold (ft+thr):
  net_f1   network-level F1: windows of dead nodes are scored normal (no alarm raised)
  live_f1  detection F1 on the windows of nodes that are still alive
  coverage fraction of nodes alive
Schemes: FL-flat, FL-LEACH, FL-LEACH matched (EASFL compression/participation/CH policy),
EASFL (dense filter), EASFL-M, centralized (model trained on collected data; nodes die
by the centralized collection energy), local-only (each node trains on its own data every
round at the same per-round training energy as an FL participant, and reports each alarm
to the BS in one HEADER+CTRL packet; no other radio use).
Usage: python run_lifetime_f1.py  -> results_lifetime_f1.json
"""
import json, time, warnings, os, sys
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool

EVAL = 25
MAXR = 1500
SEEDS = [1, 2, 3, 4, 5]
OUT = 'results_lifetime_f1.json'


def f1(p, y):
    yh = p >= 0.5
    tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def score(s_or_w, tr, te, alive, rng, ft=True, Pfix=None):
    """Returns net_f1, live_f1 using ft+thr personalization for alive nodes."""
    from common import sgd, predict_proba, best_threshold, metrics
    from sim import FT_EP
    P, Pl, Yl = [], [], []
    for i in range(len(tr)):
        (Xt, yt), (Xe, ye) = tr[i], te[i]
        if not alive[i]:
            P.append(np.zeros(len(ye))); continue
        if Pfix is not None:
            p = Pfix[i]
        else:
            w = sgd(s_or_w, Xt, yt, rng, epochs=FT_EP, lr=0.05) if ft else s_or_w
            p = predict_proba(w, Xe)
            t = best_threshold(predict_proba(w, Xt), yt)
            p = np.clip(p - t + 0.5, 0, 1)
        P.append(p); Pl.append(p); Yl.append(ye)
    Y = np.concatenate([y for _, y in te])
    net = f1(np.concatenate(P), Y)
    live = f1(np.concatenate(Pl), np.concatenate(Yl)) if Pl else float('nan')
    return float(net), float(live)


def fl_run(args):
    warnings.filterwarnings('ignore')
    from sim import Sim
    from common import build_dataset
    scheme, seed, kw = args
    data = build_dataset(seed)
    s = Sim(scheme, seed, learn=True, data=data, **kw)
    _, tr, te = data
    rows = []
    while s.r < MAXR:
        if not s.step():
            break
        if s.r % EVAL == 0:
            al = s.E > 0
            net, live = score(s.w, tr, te, al, np.random.default_rng(7))
            rows.append([s.r, net, live, float(al.mean())])
            if not al.any():
                break
    return dict(scheme=scheme, seed=seed, kw=kw, rows=rows)


def personal_probs(w, tr, te):
    from common import predict_proba, best_threshold
    P = []
    for (Xt, yt), (Xe, _) in zip(tr, te):
        t = best_threshold(predict_proba(w, Xt), yt)
        P.append(np.clip(predict_proba(w, Xe) - t + 0.5, 0, 1))
    return P


def cl_run(seed):
    warnings.filterwarnings('ignore')
    from sim import Sim
    from common import build_dataset, init_params, sgd
    _, tr, te = build_dataset(seed)
    X = np.concatenate([x for x, _ in tr]); y = np.concatenate([t for _, t in tr])
    rng = np.random.default_rng(seed); w = init_params(rng)
    for _ in range(30):
        w = sgd(w, X, y, rng, 1, 0.05, 32)
    P = personal_probs(w, tr, te)
    s = Sim('CL', seed)
    rows = []
    while s.r < MAXR and s.step():
        if s.r % EVAL == 0:
            al = s.E > 0
            net, live = score(None, tr, te, al, None, Pfix=P)
            rows.append([s.r, net, live, float(al.mean())])
            if not al.any():
                break
    return dict(scheme='CL', seed=seed, kw={}, rows=rows)


def local_run(seed):
    warnings.filterwarnings('ignore')
    from common import (build_dataset, init_params, sgd, predict_proba, best_threshold,
                        E_SAMPLE_EPOCH, HEADER, CTRL, tx_energy, N_NODES, E_INIT_LO, E_INIT_HI, FIELD, BS)
    _, tr, te = build_dataset(seed)
    net = np.random.default_rng(seed)
    pos = net.uniform(0, FIELD, (N_NODES, 2)); E0 = net.uniform(E_INIT_LO, E_INIT_HI, N_NODES)
    dBS = np.linalg.norm(pos - BS, axis=1)
    rng = np.random.default_rng(seed)
    P, rate = [], []
    for (X, yy), (Xe, ye) in zip(tr, te):
        w = init_params(rng)
        for _ in range(30):
            w = sgd(w, X, yy, rng, 1, 0.05, 16)
        t = best_threshold(predict_proba(w, X), yy)
        p = np.clip(predict_proba(w, Xe) - t + 0.5, 0, 1)
        P.append(p); rate.append(float((p >= 0.5).mean()))
    # per-round energy: local training (same as an FL participant) + alarm reports
    # (60 samples/round = 6 windows of 10; each flagged window sends one packet to the BS)
    e_round = np.array([150 * E_SAMPLE_EPOCH + 6 * r * float(tx_energy(HEADER + CTRL, d))
                        for r, d in zip(rate, dBS)])
    death = E0 / e_round
    rows = []
    for r in range(EVAL, MAXR + 1, EVAL):
        al = death > r
        n, l = score(None, tr, te, al, None, Pfix=P)
        rows.append([r, n, l, float(al.mean())])
        if not al.any():
            break
    return dict(scheme='Local', seed=seed, kw={}, rows=rows, e_round_mJ=float(e_round.mean() * 1e3))


def dispatch(a):
    if a[0] == 'CL':
        return cl_run(a[1])
    if a[0] == 'Local':
        return local_run(a[1])
    return fl_run(a)


if __name__ == '__main__':
    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {json.dumps([r['scheme'], r['seed'], r['kw']], sort_keys=True) for r in out}
    J = []
    for sd in SEEDS:
        J += [('Local', sd, {}), ('CL', sd, {}), ('FL-flat', sd, {}), ('FL-LEACH', sd, {}),
              ('FL-LEACH', sd, dict(matched=True)), ('EASFL', sd, {}), ('EASFL', sd, dict(mcos=True))]
    J = [j for j in J if json.dumps([j[0], j[1], j[2]], sort_keys=True) not in done]
    print(len(J), 'jobs', flush=True)
    t = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        for k, r in enumerate(p.imap_unordered(dispatch, J)):
            out.append(r); json.dump(out, open(OUT, 'w'), default=float)
            print(k + 1, '/', len(J), r['scheme'], r['kw'], round(time.time() - t), 's', flush=True)
