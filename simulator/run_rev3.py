"""Revision experiments, part 3 (synthetic unless the stage name says intel).

  e9      energy and traffic by protocol component over 200 rounds (no learning; seeds 1-5)
  e6      inner-product manipulation and stealth attacks, EASFL-M with and without the cosine test
  e7      unseen fault family: one family left out of the training windows; recall on that family
          (also E14: thresholds chosen on a held-out validation split of each node's training data)
  e12     larger federated models (hidden 80 -> P=2,081; hidden 400 -> P=10,401); run in a fresh
          process with EASFL_HID set
  t2      Table II with 20 seeds (seeds 6-20 added; EASFL-M from results_matched.json)
  t3      Table III lifetime with 20 seeds
  t4      Table IV poisoning with 20 seeds (full-precision baselines and EASFL-M ablations)
  t5      Table V Intel Lab with 20 seeds (run with: python run_rev3.py t5 intel)
Output: results_rev3_<stage>.json (resumable)
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool

STAGE = sys.argv[1]
OUT = f'results_rev3_{STAGE}' + (f'_hid{os.environ["EASFL_HID"]}' if 'EASFL_HID' in os.environ else '') + '.json'
M = dict(mcos=True)
ATT = ['signflip', 'stealth', 'labelflip', 'gauss']


def f1_of(p, y):
    yh = p >= 0.5; tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


# ------------------------------------------------------------------------------------- jobs
def jobs():
    J = []
    if STAGE == 'e9':
        for sd in range(1, 6):
            for sc, kw in [('CL', {}), ('FL-flat', {}), ('FL-LEACH', {}), ('FL-LEACH', dict(matched=True)),
                           ('EASFL', {}), ('EASFL', dict(audit=True))]:
                J.append(('e9', sc, sd, kw))
    elif STAGE == 'e6':
        for sd in range(1, 11):
            for m in (0.2, 0.3):
                for a in ('ipm', 'stealth'):
                    for kw in (M, dict(M, cosf=False), dict(matched=True, agg='trimmed')):
                        J.append(('learn', 'FL-LEACH' if 'matched' in kw else 'EASFL', sd, dict(kw, mal_frac=m, attack=a)))
    elif STAGE == 'e7':
        for sd in range(1, 6):
            for ex in (None, 0, 1, 2, 3):
                J.append(('e7', 'EASFL', sd, dict(M), ex))
                J.append(('e7', 'CL', sd, {}, ex))
    elif STAGE == 'e12':
        for sd in range(1, 6):
            for kw in (M, dict(matched=True)):
                for a in (None, 'signflip'):
                    k = dict(kw, mal_frac=0.2, attack=a) if a else dict(kw)
                    J.append(('learn', 'FL-LEACH' if 'matched' in kw else 'EASFL', sd, k))
    elif STAGE == 't2':
        for sd in range(6, 21):
            J += [('central', 'CL', sd, {}), ('local', 'Local', sd, {}),
                  ('learn', 'FL-flat', sd, {}), ('learn', 'FL-LEACH', sd, {})]
    elif STAGE == 't3':
        for sd in range(1, 21):
            for sc, kw in [('CL', {}), ('CL', dict(ea_ch=True)), ('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}),
                           ('EASFL', dict(psel=False)), ('EASFL', dict(compress=False)), ('EASFL', dict(ea_ch=False))]:
                J.append(('life', sc, sd, kw))
    elif STAGE == 't4':
        for sd in range(1, 21):
            for a in ATT:
                if sd > 5:
                    for agg in ('mean', 'median', 'trimmed', 'krum'):
                        J.append(('learn', 'FL-LEACH', sd, dict(mal_frac=0.2, attack=a, **({} if agg == 'mean' else dict(agg=agg)))))
                for kw in (dict(M, robust=False), dict(M, cosf=False)):
                    J.append(('learn', 'EASFL', sd, dict(kw, mal_frac=0.2, attack=a)))
    elif STAGE == 't5':
        for sd in range(1, 21):
            if sd > 5:
                J += [('icentral', 'CL', sd, {}), ('ilocal', 'Local', sd, {}),
                      ('ilearn', 'FL-flat', sd, {}), ('ilearn', 'FL-LEACH', sd, {})]
            for kw in (dict(M, compress=False), dict(M, robust=False), dict(M, psel=False)):
                J.append(('ilearn', 'EASFL', sd, kw))
            for sc, kw in [('CL', {}), ('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}), ('EASFL', dict(compress=False)),
                           ('EASFL', dict(psel=False))]:
                J.append(('ilife', sc, sd, kw))
            J.append(('icl_traffic', 'CL', sd, {}))
    return J


# ------------------------------------------------------------------------------------- runners
def run(a):
    warnings.filterwarnings('ignore')
    kind, sc, sd, kw = a[:4]
    if kind == 'e9':
        from sim import Sim
        s = Sim(sc, sd, **kw)
        for _ in range(200):
            s.step()
        return dict(cat=s.cat_E, mbit=s.bits_up / 1e6)
    if kind == 'learn':
        import run_all2 as S
        r = S.learn_run((sc, sd, kw, {})); r.pop('curve', None); return r
    if kind == 'central':
        import run_all2 as S
        return S.central_run((sd, {}))
    if kind == 'local':
        import run_all2 as S
        return S.local_run((sd, {}))
    if kind == 'life':
        from run_quick import lifetime
        r = lifetime(sc, sd, **kw); r.pop('alive', None); return {k: float(v) for k, v in r.items()}
    if kind == 'e7':
        return e7(sc, sd, kw, a[4])
    if kind.startswith('i'):
        import run_intel as R
        if kind == 'icentral':
            return R.central_run(sd)
        if kind == 'ilocal':
            return R.local_run(sd)
        if kind == 'ilearn':
            return R.learn_run((sc, sd, kw))
        if kind == 'ilife':
            return R.life_run((sc, sd, kw))
        if kind == 'icl_traffic':
            from intel import IntelSim
            s = IntelSim('CL', sd)
            for _ in range(200):
                s.step()
            return dict(mbit=s.bits_up / 1e6)
    raise ValueError(kind)


def e7(sc, sd, kw, ex):
    from common import build_dataset, init_params, sgd, predict_proba, best_threshold
    from sim import Sim, FT_EP
    kinds = []
    prof, tr, te = build_dataset(sd, exclude=ex, kinds=kinds)
    if sc == 'CL':
        X = np.concatenate([x for x, _ in tr]); y = np.concatenate([t for _, t in tr])
        rng = np.random.default_rng(sd); w = init_params(rng)
        for _ in range(30):
            w = sgd(w, X, y, rng, 1, 0.05, 32)
    else:
        s = Sim(sc, sd, learn=True, data=(prof, tr, te), **kw)
        for _ in range(200):
            s.step()
        w = s.w
    Y = np.concatenate([y for _, y in te]); K = np.concatenate(kinds)
    out = {}
    rng = np.random.default_rng(7)
    P = {'global': [], 'ft+thr': [], 'ft+thr_val': []}
    for (Xt, yt), (Xe, _) in zip(tr, te):
        P['global'].append(predict_proba(w, Xe))
        wl = sgd(w, Xt, yt, rng, epochs=FT_EP, lr=0.05) if sc != 'CL' else w
        t = best_threshold(predict_proba(wl, Xt), yt)
        P['ft+thr'].append(np.clip(predict_proba(wl, Xe) - t + 0.5, 0, 1))
        # E14: fine-tune on 80% of the node's training windows, choose the threshold on the other 20%
        n = len(yt); idx = np.random.default_rng(sd * 1000 + n).permutation(n); a_, b_ = idx[:int(0.8 * n)], idx[int(0.8 * n):]
        wv = sgd(w, Xt[a_], yt[a_], rng, epochs=FT_EP, lr=0.05) if sc != 'CL' else w
        tv = best_threshold(predict_proba(wv, Xt[b_]), yt[b_])
        P['ft+thr_val'].append(np.clip(predict_proba(wv, Xe) - tv + 0.5, 0, 1))
    for k, v in P.items():
        p = np.concatenate(v)
        rec = {int(f): float(((p >= 0.5) & (K == f)).sum() / max((K == f).sum(), 1)) for f in range(4)}
        out[k] = dict(f1=f1_of(p, Y), recall_by_family=rec,
                      f1_by_family={int(f): f1_of(p[(K == f) | (Y == 0)], Y[(K == f) | (Y == 0)]) for f in range(4)})
    return dict(final=out)


def tag(a):
    return json.dumps([a[0], a[1], a[2], a[3]] + list(a[4:]), sort_keys=True, default=str)


if __name__ == '__main__':
    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r['tag'] for r in out}
    todo = [j for j in jobs() if tag(j) not in done]
    print(len(todo), 'jobs', OUT, flush=True)
    t = time.time()
    with Pool(2) as p:
        for k, (j, r) in enumerate(zip(todo, p.imap(run, todo, chunksize=1))):
            r = dict(r); r['tag'] = tag(j); r['job'] = list(j[:4]) + list(j[4:])
            out.append(r)
            if k % 10 == 0 or k == len(todo) - 1:
                json.dump(out, open(OUT, 'w'), default=float)
                print(k + 1, '/', len(todo), round(time.time() - t), 's', flush=True)
