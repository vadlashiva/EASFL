"""Mask-aware robust aggregation (EASFL-M): mask-aware direction filter + mask-aware
trimmed mean at the CH (sim.py, mcos=True). Runs the Intel Lab and synthetic suites for
EASFL-M; baselines are reused from results_intel.json and results_v2.json.
Usage: python run_mask.py [intel] [synth]  -> results_mask.json"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
ATT = ['signflip', 'stealth', 'labelflip', 'gauss']
SEEDS = [1, 2, 3, 4, 5]

def intel_job(a):
    import run_intel as R
    return R.learn_run(a)

def synth_job(a):
    import run_all2 as S
    return S.learn_run(a)

if __name__ == '__main__':
    parts = sys.argv[1:] or ['intel', 'synth']
    try: out = json.load(open('results_mask.json'))
    except FileNotFoundError: out = {}
    save = lambda: json.dump(out, open('results_mask.json', 'w'), default=float)
    t = time.time()
    with Pool(2) as p:
        if 'intel' in parts:
            jobs = [('EASFL', s, dict(mcos=True)) for s in SEEDS]
            jobs += [('EASFL', s, dict(mcos=True, mal_frac=0.2, attack=a)) for a in ATT for s in SEEDS]
            out['intel'] = p.map(intel_job, jobs); save(); print('intel', time.time() - t, flush=True)
        if 'synth' in parts:
            jobs = [('EASFL', s, dict(mcos=True), {}) for s in SEEDS]
            jobs += [('EASFL', s, dict(mcos=True, mal_frac=m, attack=a), {}) for m in (0.1, 0.2, 0.3) for a in ATT for s in SEEDS]
            out['synth'] = p.map(synth_job, jobs); save(); print('synth', time.time() - t, flush=True)
