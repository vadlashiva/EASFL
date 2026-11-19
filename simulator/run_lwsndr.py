"""E13 (R1-8, R2-4, AE-5): detection of physically recorded, expert-labelled anomalies.

Data: Labelled Wireless Sensor Network Data Repository (LWSNDR; Suthaharan et al., ISSNIP 2010):
TelosB motes, temperature and humidity every 5 s for 6 h; one single-hop deployment (9 May 2010) and
one multi-hop deployment (10 Jul 2010), four motes each. Anomalies are real events: steam from hot
water raised humidity and temperature near a mote, and every reading is labelled 0/1 by the authors.

Protocol (train on injected faults, test on real events):
* Each of the 8 motes is one FL client. Readings are cut into non-overlapping windows of W=10
  readings (50 s); a window is anomalous if any reading in it is labelled 1.
* Training data per client: the first 60% (in time) of the mote's windows, with every window that
  touches a real event removed, plus synthetic faults injected exactly as in the Intel Lab
  experiment (spike, stuck-at, drift, noise burst; amplitudes scaled by the mote's own noise),
  restricted to the two measured channels.  The real anomalies are never seen in training.
* Test data per client: all real-event windows plus the normal windows of the last 40% in time.
* Light and voltage are not recorded; their channels are held constant, so their features are
  constant and carry no information (the detector and its 417 parameters are unchanged).
Usage: python run_lwsndr.py  -> results_lwsndr.json
"""
import json, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
import common, sim as simmod
from common import features, make_node_profiles, WIN, CH_SCALE, init_params, sgd, predict_proba, best_threshold

N = 8
for mod in (common, simmod):
    mod.N_NODES = N
SEEDS = list(range(1, 11))
R_LEARN = 200
TRAIN_FRAC = 0.6


def load():
    motes = []
    for fn, dep in (('lwsndr/single_hop.csv', 'single'), ('lwsndr/multi_hop.csv', 'multi')):
        d = pd.read_csv(fn)
        for m, g in d.groupby('mote_id'):
            g = g.sort_values('reading')
            X = np.zeros((len(g), 4)); X[:, 0] = g['temperature'].values; X[:, 1] = g['humidity'].values
            X[:, 2] = 0.0; X[:, 3] = 2.9
            lab = g['label'].values
            nw = len(g) // WIN
            Wx = X[:nw * WIN].reshape(nw, WIN, 4); Wy = lab[:nw * WIN].reshape(nw, WIN).max(1)
            dif = np.diff(X[:, :2], axis=0)
            mad = np.median(np.abs(dif - np.median(dif, 0)), 0)
            sc = np.concatenate([np.maximum(1.4826 * mad / np.sqrt(2), 0.2 * CH_SCALE[:2]), CH_SCALE[2:]])
            ev = np.where(Wy == 1)[0]
            # event ids: contiguous runs of anomalous windows
            eid = np.full(nw, -1); k = -1
            for i in ev:
                if i == 0 or Wy[i - 1] == 0: k += 1
                eid[i] = k
            motes.append(dict(dep=dep, mote=int(m), W=Wx, y=Wy, sc=sc, eid=eid))
    return motes


def inject2(rng, W, sc, p):
    X = W.copy(); n = len(X)
    y = (rng.random(n) < p['rate']).astype(int)
    for w in np.where(y)[0]:
        kind = rng.choice(4, p=p['mix']); c = rng.integers(2)          # temperature or humidity
        amp = rng.uniform(3, 8) * sc[c] * p.get('amp', 1.0)
        if kind == 0:
            j = rng.integers(WIN); X[w, j, c] += amp * rng.choice([-1, 1])
        elif kind == 1:
            X[w, :, c] = X[w, rng.integers(WIN), c]
        elif kind == 2:
            X[w, :, c] += np.linspace(0, amp * rng.uniform(1, 2), WIN) * rng.choice([-1, 1])
        else:
            X[w, :, c] += rng.normal(0, amp / 2, WIN)
    return features(X), y


MOTES = load()


def dataset(seed):
    rng = np.random.default_rng(seed)
    prof = make_node_profiles(rng, n=N)
    tr, te, meta = [], [], []
    for m, p in zip(MOTES, prof):
        nw = len(m['y']); cut = int(TRAIN_FRAC * nw)
        touch = np.convolve(m['y'], np.ones(3), 'same') > 0          # event windows and their neighbours
        itr = [i for i in range(cut) if not touch[i]]
        ite = [i for i in range(nw) if m['y'][i] == 1 or (i >= cut and not touch[i])]
        tr.append(inject2(rng, m['W'][itr], m['sc'], p))
        te.append((features(m['W'][ite]), m['y'][ite]))
        meta.append(dict(eid=m['eid'][ite]))
    allX = np.concatenate([x for x, _ in tr]); mu, sd = allX.mean(0), allX.std(0) + 1e-6
    tr = [((x - mu) / sd, y) for x, y in tr]; te = [((x - mu) / sd, y) for x, y in te]
    return prof, tr, te, meta


def score(P, te, meta, hours_per_window=WIN * 5 / 3600):
    """window-level P/R/F1, event recall and false alarms per hour (pooled over motes)."""
    yh = np.concatenate([p >= 0.5 for p in P]); y = np.concatenate([t for _, t in te])
    tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    ev_tot = ev_hit = 0; delays = []
    for p, (_, t), md in zip(P, te, meta):
        for e in sorted(set(md['eid'][md['eid'] >= 0])):
            idx = np.where(md['eid'] == e)[0]; ev_tot += 1
            hit = np.where(p[idx] >= 0.5)[0]
            if len(hit):
                ev_hit += 1; delays.append(int(hit[0]))
    from sklearn.metrics import roc_auc_score
    pr = np.concatenate(P)
    return dict(prec=prec, rec=rec, f1=f1, auc=float(roc_auc_score(y, pr)), ev_rec=ev_hit / max(ev_tot, 1), events=ev_tot,
                delay_win=float(np.mean(delays)) if delays else float('nan'),
                fa_per_h=fp / max((y == 0).sum() * hours_per_window, 1e-9), n_pos=int(y.sum()), n_neg=int((y == 0).sum()))


def probs(w, tr, te, personal):
    P = []
    rng = np.random.default_rng(7)
    for (Xt, yt), (Xe, _) in zip(tr, te):
        ww = sgd(w, Xt, yt, rng, epochs=simmod.FT_EP, lr=0.05) if personal == 'ft+thr' else w
        p = predict_proba(ww, Xe)
        if personal in ('thr', 'ft+thr'):
            t = best_threshold(predict_proba(ww, Xt), yt); p = np.clip(p - t + 0.5, 0, 1)
        P.append(p)
    return P


CONF = {'EASFL-M': ('EASFL', dict(mcos=True)), 'EASFL (dense filter)': ('EASFL', dict()),
        'FedAvg (matched)': ('FL-LEACH', dict(matched=True)), 'Trimmed mean (matched)': ('FL-LEACH', dict(matched=True, agg='trimmed')),
        'FedAvg-LEACH (full)': ('FL-LEACH', dict())}


def fl_run(args):
    name, seed = args
    prof, tr, te, meta = dataset(seed)
    sch, kw = CONF[name]
    s = simmod.Sim(sch, seed, learn=True, data=(prof, tr, te), **kw)
    for _ in range(R_LEARN):
        s.step()
    return dict(method=name, seed=seed, alive=int((s.E > 0).sum()),
                res={p or 'global': score(probs(s.w, tr, te, p), te, meta) for p in (None, 'thr', 'ft+thr')})


def central_run(seed):
    prof, tr, te, meta = dataset(seed)
    X = np.concatenate([x for x, _ in tr]); y = np.concatenate([t for _, t in tr])
    rng = np.random.default_rng(seed); w = init_params(rng)
    for _ in range(30):
        w = sgd(w, X, y, rng, 1, 0.05, 32)
    return dict(method='Centralized', seed=seed, res={p or 'global': score(probs(w, tr, te, p), te, meta) for p in (None, 'thr')})


def local_run(seed):
    prof, tr, te, meta = dataset(seed)
    rng = np.random.default_rng(seed); P, Pt = [], []
    for (X, y), (Xe, _) in zip(tr, te):
        w = init_params(rng)
        for _ in range(30):
            w = sgd(w, X, y, rng, 1, 0.05, 16)
        p = predict_proba(w, Xe); P.append(p)
        t = best_threshold(predict_proba(w, X), y); Pt.append(np.clip(p - t + 0.5, 0, 1))
    return dict(method='Local only', seed=seed, res={'global': score(P, te, meta), 'thr': score(Pt, te, meta)})


def iforest_run(seed):
    """Unsupervised reference: Isolation Forest per mote, fitted on the mote's clean training windows."""
    from sklearn.ensemble import IsolationForest
    prof, tr, te, meta = dataset(seed)
    P = []
    for (X, y), (Xe, _) in zip(tr, te):
        f = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed).fit(X[y == 0])
        P.append((f.predict(Xe) == -1).astype(float))
    return dict(method='Isolation Forest (unsup.)', seed=seed, res={'global': score(P, te, meta)})


if __name__ == '__main__':
    for m in MOTES:
        print(m['dep'], m['mote'], 'windows', len(m['y']), 'anomalous', int(m['y'].sum()), 'events', int(m['eid'].max() + 1))
    jobs = [(n, s) for n in CONF for s in SEEDS]
    with Pool(2) as P:
        out = P.map(fl_run, jobs)
        out += P.map(central_run, SEEDS) + P.map(local_run, SEEDS) + P.map(iforest_run, SEEDS)
    json.dump(out, open('results_lwsndr.json', 'w'), indent=1)
    meth = list(dict.fromkeys(o['method'] for o in out))
    for mname in meth:
        R = [o for o in out if o['method'] == mname]
        for p in R[0]['res']:
            v = lambda k: np.nanmean([o['res'][p][k] for o in R])
            print(f"{mname:28s} {p:7s} F1={v('f1'):.3f} P={v('prec'):.3f} R={v('rec'):.3f} AUC={v('auc'):.3f} "
                  f"events={v('ev_rec'):.2f} FA/h={v('fa_per_h'):.1f}")
