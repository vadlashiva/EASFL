"""Fig. 4 data with 10 seeds and EASFL-M (revision R1-6, R1-9): adds seeds 6-10 for every method and seeds 1-10 for EASFL-M."""
import json, time
from multiprocessing import Pool
import run_all2 as ra
if __name__ == '__main__':
    R = json.load(open('results_v2_20.json'))
    D = list(ra.DEFENCES)
    jobs = []
    for sc, kw in D:
        seeds = range(6, 11)
        jobs += [(sc, sd, dict(kw, mal_frac=0.0), {}) for sd in seeds]
        for m in (0.1, 0.2, 0.3):
            for a in ra.ATTACKS:
                jobs += [(sc, sd, dict(kw, mal_frac=m, attack=a), {}) for sd in seeds]
    for sd in range(1, 11):
        jobs.append(('EASFL', sd, dict(mcos=True, mal_frac=0.0), {}))
        for m in (0.1, 0.2, 0.3):
            for a in ra.ATTACKS:
                jobs.append(('EASFL', sd, dict(mcos=True, mal_frac=m, attack=a), {}))
    t = time.time()
    with Pool(2) as pool:
        new = pool.map(ra.learn_run, jobs)
    R['attack'] = R['attack'] + new
    R['_note'] += '; attack: seeds 1-10, EASFL-M added'
    json.dump(R, open('results_v2_20.json', 'w'))
    print('done', len(jobs), time.time() - t)
