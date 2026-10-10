"""Extended experiment suite (v2) for the EASFL-WSN study.

Adds to run_all.py:
  * per-node personalisation (local threshold calibration and local fine-tuning) for every scheme
  * stronger attacks (stealth sign flip, label flip, Gaussian) and robust-aggregation baselines
    (coordinate median, trimmed mean, Krum) on FL-LEACH
  * sensitivity of the conclusions to the synthetic generator (non-IID degree alpha,
    fault magnitude, anomaly rate)
Results are written to results_v2.json; summarize2.py prints the tables.
"""
import numpy as np, json, time, sys
from multiprocessing import Pool
from sim import Sim
from common import *
from run_quick import lifetime

SEEDS = [1, 2, 3, 4, 5]
R_LEARN = 200
PERS = [None, 'thr', 'ft+thr']


def key(p):
    return 'global' if p is None else p


def learn_run(args):
    scheme, seed, kw, gen = args
    data = build_dataset(seed, **gen)
    s = Sim(scheme, seed, learn=True, data=data, **kw)
    curve = []
    for r in range(R_LEARN):
        s.step()
        if (r + 1) % 10 == 0:
            m = s.evaluate(); row = [r + 1, m['f1'], m['auc']]
            if (r + 1) % 50 == 0:
                mp = s.evaluate('ft+thr'); row += [mp['f1'], mp['auc']]
            curve.append(row)
    final = {key(p): s.evaluate(p) for p in PERS}
    final['net'] = s.evaluate('ft+thr', alive_only=True)   # dead nodes raise no alarms
    used = float(s.E0.sum() - np.clip(s.E, 0, None).sum())
    return dict(scheme=scheme, seed=seed, kw=kw, gen=gen, final=final, curve=curve,
                mbit=s.bits_up / 1e6, energy=used, alive=int((s.E > 0).sum()))


def central_run(args):
    seed, gen = args
    _, tr, te = build_dataset(seed, **gen)
    X = np.concatenate([x for x, _ in tr]); y = np.concatenate([t for _, t in tr])
    rng = np.random.default_rng(seed); w = init_params(rng)
    for _ in range(30):
        w = sgd(w, X, y, rng, 1, 0.05, 32)
    out = {'global': metrics(w, te)}
    P = []
    for (Xt, yt), (Xe, _) in zip(tr, te):
        t = best_threshold(predict_proba(w, Xt), yt)
        P.append(np.clip(predict_proba(w, Xe) - t + 0.5, 0, 1))
    out['thr'] = metrics(None, te, proba=np.concatenate(P))
    return dict(seed=seed, gen=gen, final=out)


def local_run(args):
    seed, gen = args
    _, tr, te = build_dataset(seed, **gen)
    rng = np.random.default_rng(seed)
    P, Pt = [], []
    for (X, y), (Xe, _) in zip(tr, te):
        w = init_params(rng)
        for _ in range(30):
            w = sgd(w, X, y, rng, 1, 0.05, 16)
        p = predict_proba(w, Xe); P.append(p)
        t = best_threshold(predict_proba(w, X), y); Pt.append(np.clip(p - t + 0.5, 0, 1))
    return dict(seed=seed, gen=gen, final={'global': metrics(None, te, proba=np.concatenate(P)),
                                           'thr': metrics(None, te, proba=np.concatenate(Pt))})


def life_run(args):
    scheme, seed, kw = args
    r = lifetime(scheme, seed, **kw)
    r['alive'] = r['alive'].tolist()
    return dict(scheme=scheme, seed=seed, kw=kw, **r)


GEN_SWEEP = ([dict(alpha=a) for a in [0.1, 0.3, 1.0, 10.0]] +
             [dict(amp_scale=a) for a in [0.5, 0.75, 1.5]] +
             [dict(rate_scale=a) for a in [0.5, 2.0]])
ATTACKS = ['signflip', 'stealth', 'labelflip', 'gauss']
DEFENCES = [('FL-LEACH', {}), ('FL-LEACH', dict(agg='median')), ('FL-LEACH', dict(agg='trimmed')),
            ('FL-LEACH', dict(agg='krum')), ('EASFL', dict(robust=False)),
            ('EASFL', dict(cosf=False)), ('EASFL', {})]

if __name__ == '__main__':
    parts = sys.argv[1:] or ['main', 'life', 'gen', 'attack', 'loss', 'rate']
    t = time.time()
    try:
        out = json.load(open('results_v2.json'))
    except FileNotFoundError:
        out = {}
    def save():
        json.dump(out, open('results_v2.json', 'w'), default=float)
    with Pool(2) as pool:
        if 'main' in parts:
            out['central'] = pool.map(central_run, [(s, {}) for s in SEEDS])
            out['local'] = pool.map(local_run, [(s, {}) for s in SEEDS])
            jobs = [(sc, sd, kw, {}) for sc, kw in [('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}),
                                                    ('EASFL', dict(mu=0.1))] for sd in SEEDS]
            out['learn'] = pool.map(learn_run, jobs); save(); print('main', time.time() - t, flush=True)
        if 'life' in parts:
            variants = [('CL', {}), ('CL', dict(ea_ch=True)), ('FL-flat', {}), ('FL-LEACH', {}),
                        ('EASFL', {}), ('EASFL', dict(psel=False)), ('EASFL', dict(compress=False)),
                        ('EASFL', dict(ea_ch=False))]
            out['life'] = pool.map(life_run, [(sc, sd, kw) for sc, kw in variants for sd in SEEDS])
            save(); print('life', time.time() - t, flush=True)
        if 'gen' in parts:
            out['gen_central'] = pool.map(central_run, [(s, g) for g in GEN_SWEEP for s in SEEDS])
            jobs = [(sc, sd, {}, g) for g in GEN_SWEEP for sc in ['FL-LEACH', 'EASFL'] for sd in SEEDS]
            out['gen'] = pool.map(learn_run, jobs); save(); print('gen', time.time() - t, flush=True)
        if 'attack' in parts:
            jobs = [(sc, sd, dict(kw, mal_frac=0.0), {}) for sc, kw in DEFENCES for sd in SEEDS]
            for m in [0.1, 0.2, 0.3]:
                for a in ATTACKS:
                    jobs += [(sc, sd, dict(kw, mal_frac=m, attack=a), {}) for sc, kw in DEFENCES for sd in SEEDS]
            out['attack'] = pool.map(learn_run, jobs); save(); print('attack', time.time() - t, flush=True)
        if 'loss' in parts:
            jobs = [(sc, sd, dict(p_loss=p), {}) for p in [0.0, 0.1, 0.2, 0.3]
                    for sc in ['FL-LEACH', 'EASFL'] for sd in SEEDS]
            out['loss'] = pool.map(learn_run, jobs); save(); print('loss', time.time() - t, flush=True)
        if 'rate' in parts:
            jobs = [(sc, sd, dict(samples_per_round=S, **kw)) for S in [10, 30, 60, 120, 240]
                    for sc, kw in [('CL', {}), ('CL', dict(ea_ch=True))] for sd in SEEDS]
            res = pool.map(life_run, jobs)
            for r in res:
                r.pop('alive')
            out['rate'] = res; save(); print('rate', time.time() - t, flush=True)
    print('done', time.time() - t)
