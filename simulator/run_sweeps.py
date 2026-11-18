"""E11: sensitivity of energy and lifetime to network and radio parameters (no learning; the energy
of a learning round does not depend on the model's values). Each setting runs in a fresh process
with EASFL_* environment variables (common.py). Seeds 1-5.
Usage: python run_sweeps.py -> results_sweeps.json"""
import json, os, subprocess, sys, itertools
from multiprocessing import Pool

SETTINGS = [
    ('default', {}),
    ('N=50', dict(N_NODES=50)), ('N=200', dict(N_NODES=200)),
    ('field 200 m', dict(FIELD=200.0, BS_X=100.0, BS_Y=275.0)),
    ('BS inside (50,50)', dict(BS_X=50.0, BS_Y=50.0)), ('BS far (50,300)', dict(BS_Y=300.0)),
    ('p=0.05', dict(P_CH=0.05)), ('p=0.2', dict(P_CH=0.2)),
    ('E_elec 25 nJ', dict(E_ELEC=25e-9)), ('E_elec 100 nJ', dict(E_ELEC=100e-9)),
    ('e_s 0.5 uJ', dict(E_SAMPLE=0.5e-6)), ('e_s 8 uJ', dict(E_SAMPLE=8e-6)),
]
SCHEMES = [('CL', {}), ('FL-LEACH', {}), ('FL-LEACH', {'matched': True}), ('EASFL', {})]
CODE = r'''
import json, sys
from run_quick import lifetime
sc, sd, kw = json.loads(sys.argv[1])
r = lifetime(sc, sd, **kw); r.pop("alive", None)
print(json.dumps({k: (None if v is None else float(v)) for k, v in r.items()}))
'''


def job(a):
    name, env, sc, kw, sd = a
    e = dict(os.environ, **{'EASFL_' + k: repr(v) if isinstance(v, float) else str(v) for k, v in env.items()})
    out = subprocess.run([sys.executable, '-c', CODE, json.dumps([sc, sd, kw])], env=e, capture_output=True, text=True)
    try:
        return dict(setting=name, scheme=sc, kw=kw, seed=sd, **json.loads(out.stdout.strip().splitlines()[-1]))
    except Exception:
        return dict(setting=name, scheme=sc, kw=kw, seed=sd, error=out.stderr[-400:])


if __name__ == '__main__':
    J = [(n, e, sc, kw, sd) for (n, e), (sc, kw), sd in itertools.product(SETTINGS, SCHEMES, range(1, 6))]
    with Pool(2) as p:
        R = p.map(job, J)
    json.dump(R, open('results_sweeps.json', 'w'))
    import numpy as np, collections
    G = collections.defaultdict(list)
    for r in R:
        if 'error' in r:
            print('ERROR', r['setting'], r['scheme'], r['kw'], r['seed'], r['error'][-200:]); continue
        G[(r['setting'], r['scheme'] + (' matched' if r['kw'] else ''))].append(r)
    for n, _ in SETTINGS:
        print(n)
        for sc in ('CL', 'FL-LEACH', 'FL-LEACH matched', 'EASFL'):
            L = G[(n, sc)]
            print(f'   {sc:18s} FND={np.mean([x["fnd"] for x in L]):7.0f} HND={np.mean([x["hnd"] for x in L]):7.0f} '
                  f'LND={np.mean([x["lnd"] for x in L]):7.0f} E200={np.nanmean([np.nan if x["e200"] is None else x["e200"] for x in L]):7.2f}')
