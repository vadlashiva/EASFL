import numpy as np, json, time
from multiprocessing import Pool
from sim import Sim
from common import *
from run_quick import lifetime

SEEDS = [1, 2, 3, 4, 5]
R_LEARN = 200

def learn_run(args):
    scheme, seed, kw = args
    data = build_dataset(seed)
    s = Sim(scheme, seed, learn=True, data=data, **kw)
    curve = []
    for r in range(R_LEARN):
        s.step()
        if (r + 1) % 10 == 0:
            m = s.evaluate(); curve.append((r + 1, m['f1'], m['auc']))
    m = s.evaluate()
    used = float(s.E0.sum() - np.clip(s.E, 0, None).sum())
    return dict(scheme=scheme, seed=seed, kw=kw, final=m, curve=curve,
                mbit=s.bits_up / 1e6, energy=used, alive=int((s.E > 0).sum()))

def central_run(seed):
    _, tr, te = build_dataset(seed)
    X = np.concatenate([x for x, _ in tr]); y = np.concatenate([t for _, t in tr])
    rng = np.random.default_rng(seed); w = init_params(rng)
    for _ in range(30):
        w = sgd(w, X, y, rng, 1, 0.05, 32)
    return metrics(w, te)

def local_run(seed):
    _, tr, te = build_dataset(seed)
    rng = np.random.default_rng(seed)
    P = []
    for (X, y) in tr:
        w = init_params(rng)
        for _ in range(30):
            w = sgd(w, X, y, rng, 1, 0.05, 16)
        P.append(w)
    proba = np.concatenate([predict_proba(w, x) for w, (x, _) in zip(P, te)])
    return metrics(None, te, proba=proba)

def life_run(args):
    scheme, seed, kw = args
    r = lifetime(scheme, seed, **kw)
    r['alive'] = r['alive'].tolist()
    return dict(scheme=scheme, seed=seed, kw=kw, **r)

if __name__ == '__main__':
    t = time.time(); out = {}
    with Pool(2) as pool:
        out['central'] = pool.map(central_run, SEEDS)
        out['local'] = pool.map(local_run, SEEDS)
        print('central/local', time.time() - t, flush=True)
        jobs = [(sc, sd, {}) for sc in ['FL-flat', 'FL-LEACH', 'EASFL'] for sd in SEEDS]
        out['learn'] = pool.map(learn_run, jobs); print('learn', time.time() - t, flush=True)
        jobs = []
        for m in [0.0, 0.1, 0.2, 0.3]:
            for sd in SEEDS:
                jobs += [('FL-LEACH', sd, dict(mal_frac=m)), ('EASFL', sd, dict(mal_frac=m)),
                         ('EASFL', sd, dict(mal_frac=m, robust=False))]
        out['mal'] = pool.map(learn_run, jobs); print('mal', time.time() - t, flush=True)
        jobs = []
        for p in [0.0, 0.1, 0.2, 0.3]:
            for sd in SEEDS:
                jobs += [('FL-LEACH', sd, dict(p_loss=p)), ('EASFL', sd, dict(p_loss=p))]
        out['loss'] = pool.map(learn_run, jobs); print('loss', time.time() - t, flush=True)
        variants = [('CL', {}), ('CL', dict(ea_ch=True)), ('FL-flat', {}), ('FL-LEACH', {}),
                    ('EASFL', {}), ('EASFL', dict(psel=False)), ('EASFL', dict(compress=False)),
                    ('EASFL', dict(ea_ch=False))]
        jobs = [(sc, sd, kw) for sc, kw in variants for sd in SEEDS]
        out['life'] = pool.map(life_run, jobs); print('life', time.time() - t, flush=True)
        jobs = [(sc, sd, dict(samples_per_round=S, **kw)) for S in [10, 30, 60, 120, 240]
                for sc, kw in [('CL', {}), ('CL', dict(ea_ch=True))] for sd in SEEDS]
        out['rate'] = pool.map(life_run, jobs); print('rate', time.time() - t, flush=True)
    for k in ['rate', 'life']:
        for r in out[k]:
            if k == 'rate':
                r.pop('alive')
    json.dump(out, open('results.json', 'w'), default=float)
    print('done', time.time() - t)
