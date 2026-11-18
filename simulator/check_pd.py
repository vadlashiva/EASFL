"""Check that participant-only downlink (pd) leaves detection unchanged (synthetic and Intel)."""
import json, numpy as np
from multiprocessing import Pool
import common
from common import build_dataset
from sim import Sim
from intel import build_dataset_intel, load_net

def run(args):
    which, pd, seed = args
    common.set_radio('first-order')
    if which == 'syn':
        s = Sim('EASFL', seed, learn=True, data=build_dataset(seed), pd=pd)
    else:
        s = Sim('EASFL', seed, learn=True, data=build_dataset_intel(seed), net=load_net(), pd=pd)
    for _ in range(200):
        s.step()
    return dict(which=which, pd=pd, seed=seed, g=s.evaluate()['f1'], p=s.evaluate('ft+thr')['f1'],
                net=s.evaluate('ft+thr', alive_only=True)['f1'], alive=int((s.E > 0).sum()))

if __name__ == '__main__':
    with Pool(2) as pool:
        out = pool.map(run, [(w, pd, sd) for w in ['syn', 'intel'] for pd in [False, True] for sd in [1, 2, 3, 4, 5]])
    json.dump(out, open('results_pd_check.json', 'w'))
    for w in ['syn', 'intel']:
        for pd in [False, True]:
            xs = [x for x in out if x['which'] == w and x['pd'] == pd]
            print(w, 'PD' if pd else '  ', ' '.join(f"{k}={np.mean([x[k] for x in xs]):.3f}" for k in ['g', 'p', 'net', 'alive']))
