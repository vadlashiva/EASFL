"""E2b (revision): train-then-detect deployment. Every scheme first trains (FL schemes:
200 federated rounds; local-only: 30 local epochs once), then all nodes only infer and
report each flagged window to the BS (HEADER+CTRL bits); no further training or model
exchange. Lifetime after training is set by the alarm-report energy. Network-level F1 vs
round, same accounting for every scheme. -> results_ttd.json"""
import json, warnings, numpy as np
warnings.filterwarnings('ignore')
from multiprocessing import Pool
EVAL, MAXR, SEEDS = 25, 6000, [1, 2, 3, 4, 5]

def f1(p, y):
    yh = p >= 0.5; tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

def curve(P, te, E_after, dBS, t0):
    from common import tx_energy, HEADER, CTRL
    Y = np.concatenate([y for _, y in te])
    rate = np.array([(p >= 0.5).mean() for p in P])
    e = np.array([6 * r * float(tx_energy(HEADER + CTRL, d)) for r, d in zip(rate, dBS)]) + 1e-12
    death = t0 + np.clip(E_after, 0, None) / e
    rows = []
    for t in range(EVAL, MAXR + 1, EVAL):
        al = death > t
        Pn = [p if a else np.zeros_like(p) for p, a in zip(P, al)]
        rows.append([t, float(f1(np.concatenate(Pn), Y)), float(al.mean())])
    return rows, float(np.median(death)), float(e.mean() * 1e3)

def job(a):
    warnings.filterwarnings('ignore')
    from sim import Sim, FT_EP
    from common import build_dataset, init_params, sgd, predict_proba, best_threshold, E_SAMPLE_EPOCH
    name, seed, kw = a
    prof, tr, te = data = build_dataset(seed)
    if name == 'Local':
        s = Sim('EASFL', seed)            # only for the deployment (positions, E0, dBS)
        rng = np.random.default_rng(seed); P = []
        for (X, y), (Xe, _) in zip(tr, te):
            w = init_params(rng)
            for _ in range(30): w = sgd(w, X, y, rng, 1, 0.05, 16)
            t = best_threshold(predict_proba(w, X), y); P.append(np.clip(predict_proba(w, Xe) - t + 0.5, 0, 1))
        E = s.E0 - 30 * np.array([len(y) for _, y in tr]) * E_SAMPLE_EPOCH
        rows, med, e = curve(P, te, E, s.dBS, 0)
    else:
        s = Sim(name, seed, learn=True, data=data, **kw)
        for _ in range(200): s.step()
        rng = np.random.default_rng(7); P = []
        for (Xt, yt), (Xe, _) in zip(tr, te):
            w = sgd(s.w, Xt, yt, rng, epochs=FT_EP, lr=0.05)
            t = best_threshold(predict_proba(w, Xt), yt); P.append(np.clip(predict_proba(w, Xe) - t + 0.5, 0, 1))
        E = s.E - FT_EP * np.array([len(y) for _, y in tr]) * E_SAMPLE_EPOCH
        rows, med, e = curve(P, te, E, s.dBS, 200)
        rows = [[t, 0.0, 0.0] if False else r for t, r in zip(range(len(rows)), rows)]
    return dict(scheme=name, seed=seed, kw=kw, rows=rows, median_death=med, alarm_mJ=e)

if __name__ == '__main__':
    J = [(n, s, k) for s in SEEDS for n, k in [('Local', {}), ('FL-LEACH', {}), ('EASFL', dict(mcos=True))]]
    with Pool(2) as p: out = p.map(job, J)
    json.dump(out, open('results_ttd.json', 'w'), default=float)
    for n in ('Local', 'FL-LEACH', 'EASFL'):
        R = [r for r in out if r['scheme'] == n]
        def at(rows, t, j): return next((x[j] for x in rows if x[0] >= t), 0.0)
        print(f'{n:9s} median death {np.mean([r["median_death"] for r in R]):7.0f}  alarm {np.mean([r["alarm_mJ"] for r in R]):.3f} mJ/rnd  netF1 '
              + ' '.join(f't={t}:{np.mean([at(r["rows"], t, 1) for r in R]):.3f}' for t in (200, 500, 1000, 2000, 3000, 4000, 5000)))
