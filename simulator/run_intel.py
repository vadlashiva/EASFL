"""Intel Lab experiment suite for the EASFL-WSN study (see intel.py for the protocol).

Usage: python run_intel.py [main] [life] [attack]
Expects intel_clean.npz (from prep_intel.py) and mote_locs.txt in the working directory,
or paths in the environment variables INTEL_NPZ and INTEL_LOCS.
Results are written to results_intel.json; summarize_intel.py prints the tables.
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
import intel
from intel import IntelSim, build_intel_dataset, use_intel_layout
from common import init_params, sgd, metrics, predict_proba, best_threshold
import common

NPZ = os.environ.get('INTEL_NPZ', 'intel_clean.npz')
LOCS = os.environ.get('INTEL_LOCS', 'mote_locs.txt')
N = use_intel_layout(NPZ, LOCS)
SEEDS = [1, 2, 3, 4, 5]
R_LEARN = 200
PERS = [None, 'thr', 'ft+thr']


def data(seed):
    return build_intel_dataset(seed, NPZ, LOCS)


def learn_run(args):
    scheme, seed, kw = args
    s = IntelSim(scheme, seed, learn=True, data=data(seed), **kw)
    for _ in range(R_LEARN):
        s.step()
    final = {('global' if p is None else p): s.evaluate(p) for p in PERS}
    final['net'] = s.evaluate('ft+thr', alive_only=True)
    used = float(s.E0.sum() - np.clip(s.E, 0, None).sum())
    return dict(scheme=scheme, seed=seed, kw=kw, final=final, mbit=s.bits_up / 1e6,
                energy=used, alive=int((s.E > 0).sum()))


def central_run(seed):
    _, tr, te = data(seed)
    X = np.concatenate([x for x, _ in tr]); y = np.concatenate([t for _, t in tr])
    rng = np.random.default_rng(seed); w = init_params(rng)
    for _ in range(30):
        w = sgd(w, X, y, rng, 1, 0.05, 32)
    P = []
    for (Xt, yt), (Xe, _) in zip(tr, te):
        t = best_threshold(predict_proba(w, Xt), yt)
        P.append(np.clip(predict_proba(w, Xe) - t + 0.5, 0, 1))
    # network level: run centralized collection for R_LEARN rounds; dead nodes raise no alarms
    s = IntelSim('CL', seed)
    for _ in range(R_LEARN):
        s.step()
    Pn = [p if s.E[i] > 0 else np.zeros_like(p) for i, p in enumerate(P)]
    return dict(seed=seed, alive=int((s.E > 0).sum()),
                final={'global': metrics(w, te), 'thr': metrics(None, te, proba=np.concatenate(P)),
                       'net': metrics(None, te, proba=np.concatenate(Pn))})


def local_run(seed):
    _, tr, te = data(seed)
    rng = np.random.default_rng(seed)
    P, Pt = [], []
    for (X, y), (Xe, _) in zip(tr, te):
        w = init_params(rng)
        for _ in range(30):
            w = sgd(w, X, y, rng, 1, 0.05, 16)
        p = predict_proba(w, Xe); P.append(p)
        t = best_threshold(predict_proba(w, X), y); Pt.append(np.clip(p - t + 0.5, 0, 1))
    return dict(seed=seed, final={'global': metrics(None, te, proba=np.concatenate(P)),
                                  'thr': metrics(None, te, proba=np.concatenate(Pt))})


def life_run(args):
    scheme, seed, kw = args
    s = IntelSim(scheme, seed, **kw)
    alive, e200, maxr = [], None, 4000
    while s.r < maxr and s.step():
        alive.append(int((s.E > 0).sum()))
        if s.r == 200:
            e200 = float(s.E0.sum() - np.clip(s.E, 0, None).sum())
    a = np.array(alive)
    first = lambda m: int(np.argmax(m)) + 1 if m.any() else maxr
    return dict(scheme=scheme, seed=seed, kw=kw, fnd=first(a < N), hnd=first(a <= N // 2),
                lnd=first(a <= N // 10), e200=e200)


ATTACKS = ['signflip', 'stealth', 'labelflip', 'gauss']
DEFENCES = [('FL-LEACH', {}), ('FL-LEACH', dict(agg='median')), ('FL-LEACH', dict(agg='trimmed')),
            ('FL-LEACH', dict(agg='krum')), ('EASFL', {})]

if __name__ == '__main__':
    parts = sys.argv[1:] or ['main', 'life', 'attack']
    t = time.time()
    try:
        out = json.load(open('results_intel.json'))
    except FileNotFoundError:
        out = {}
    out['meta'] = dict(n_nodes=N, bs=intel.BS_INTEL.tolist(), seeds=SEEDS, rounds=R_LEARN)
    save = lambda: json.dump(out, open('results_intel.json', 'w'), default=float)
    with Pool(2) as pool:
        if 'main' in parts:
            out['central'] = pool.map(central_run, SEEDS)
            out['local'] = pool.map(local_run, SEEDS)
            jobs = [(sc, sd, kw) for sc, kw in [('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}),
                                                ('EASFL', dict(compress=False)), ('EASFL', dict(robust=False)),
                                                ('EASFL', dict(psel=False))] for sd in SEEDS]
            out['learn'] = pool.map(learn_run, jobs); save(); print('main', time.time() - t, flush=True)
        if 'life' in parts:
            variants = [('CL', {}), ('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}),
                        ('EASFL', dict(psel=False)), ('EASFL', dict(compress=False)),
                        ('EASFL', dict(ea_ch=False))]
            out['life'] = pool.map(life_run, [(sc, sd, kw) for sc, kw in variants for sd in SEEDS])
            save(); print('life', time.time() - t, flush=True)
        if 'attack' in parts:
            jobs = [(sc, sd, dict(kw, mal_frac=0.2, attack=a)) for a in ATTACKS
                    for sc, kw in DEFENCES for sd in SEEDS]
            out['attack'] = pool.map(learn_run, jobs); save(); print('attack', time.time() - t, flush=True)
    print('done', time.time() - t)
