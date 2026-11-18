"""E1 (revision): matched baselines on the synthetic workload.

Every baseline runs with EASFL's own top-k 8-bit compression with error feedback,
energy-aware participant selection (rho = 0.5), energy-aware CH selection, 8-bit CH->BS
and 16-bit broadcast (Sim(..., matched=True)). Only the aggregation rule differs:
mean, coordinate median, trimmed mean, Krum, support-aware median (no filter/quarantine),
EASFL with the dense filter, and EASFL-M (mask-aware filter + trimmed mean + quarantine).

Seeds 1-20 for clean and 20% attacks; seeds 1-5 for 10% and 30%.
Usage: python run_matched.py [intel] -> results_matched[_intel].json (resumable;
intel: Intel Lab deployment, clean and 20% attacks, seeds 1-20)
"""
import json, time, warnings, os
warnings.filterwarnings('ignore')
from multiprocessing import Pool

ATT = ['signflip', 'stealth', 'labelflip', 'gauss']
DEF = [('FL-LEACH', dict(matched=True)),
       ('FL-LEACH', dict(matched=True, agg='median')),
       ('FL-LEACH', dict(matched=True, agg='trimmed')),
       ('FL-LEACH', dict(matched=True, agg='krum')),
       ('FL-LEACH', dict(matched=True, agg='mmedian')),
       ('EASFL', {}),
       ('EASFL', dict(mcos=True))]
import sys
INTEL = 'intel' in sys.argv[1:]
OUT = 'results_matched_intel.json' if INTEL else 'results_matched.json'


def job(a):
    warnings.filterwarnings('ignore')
    if INTEL:
        import run_intel as S
        r = S.learn_run(a[:3])
    else:
        import run_all2 as S
        r = S.learn_run(a)
    r.pop('curve', None)
    return r


def jobs():
    J = []
    for seed in range(1, 21):
        for sc, kw in DEF:
            J.append((sc, seed, dict(kw), {}))
            for a in ATT:
                J.append((sc, seed, dict(kw, mal_frac=0.2, attack=a), {}))
    for seed in ([] if INTEL else range(1, 6)):
        for m in (0.1, 0.3):
            for a in ATT:
                for sc, kw in DEF:
                    J.append((sc, seed, dict(kw, mal_frac=m, attack=a), {}))
    return J


def tag(a):
    return json.dumps([a[0], a[1], a[2]], sort_keys=True)


if __name__ == '__main__':
    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {tag((r['scheme'], r['seed'], r['kw'])) for r in out}
    todo = [j for j in jobs() if tag(j) not in done]
    print(len(todo), 'jobs to run', flush=True)
    t = time.time()
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(job, todo, chunksize=2)):
            out.append(r)
            if k % 10 == 0 or k == len(todo) - 1:
                json.dump(out, open(OUT, 'w'), default=float)
                print(k + 1, '/', len(todo), round(time.time() - t), 's', flush=True)
