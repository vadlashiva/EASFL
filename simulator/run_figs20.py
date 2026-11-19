"""Re-run the data behind Figs. 2 (alive) , 3 (convergence) and 5 (ablation) with 20 seeds (revision: R1-9)."""
import json, time
from multiprocessing import Pool
import run_all2 as ra
SEEDS = list(range(1, 21))
if __name__ == '__main__':
    R = json.load(open('results_v2.json'))
    t = time.time()
    with Pool(2) as pool:
        variants = [('CL', {}), ('CL', dict(ea_ch=True)), ('FL-flat', {}), ('FL-LEACH', {}),
                    ('EASFL', {}), ('EASFL', dict(ea_ch=False)), ('EASFL', dict(psel=False)), ('EASFL', dict(compress=False))]
        R['life'] = pool.map(ra.life_run, [(sc, sd, kw) for sc, kw in variants for sd in SEEDS]); print('life', time.time() - t, flush=True)
        R['central'] = pool.map(ra.central_run, [(s, {}) for s in SEEDS])
        jobs = [(sc, sd, kw, {}) for sc, kw in (('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}), ('EASFL', dict(mcos=True))) for sd in SEEDS]
        R['learn'] = pool.map(ra.learn_run, jobs); print('learn', time.time() - t, flush=True)
    R['_note'] = 'life, central and learn re-run with 20 seeds; other parts as in results_v2.json'
    json.dump(R, open('results_v2_20.json', 'w'))
