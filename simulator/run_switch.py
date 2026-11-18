import json, numpy as np, sys
from multiprocessing import Pool
from run_quick import lifetime
POL = {'score': {}, 'LEACH': dict(ea_ch=False), 'switch 10% dead': dict(ch_switch=0.10), 'switch 50% dead': dict(ch_switch=0.50), 'switch E<50%': dict(ch_switch_energy=0.5), 'switch E<25%': dict(ch_switch_energy=0.25), 'score+cooldown': dict(ch_cooldown=True)}

def job(a):
    sd, name = a
    r = lifetime('EASFL', sd, **POL[name])
    al = r['alive']; n = 100
    t80 = int(np.argmax(al <= 20)) + 1 if (al <= 20).any() else len(al)
    return dict(seed=sd, policy=name, fnd=r['fnd'], hnd=r['hnd'], lnd=r['lnd'], node_rounds=int(al.sum()), t80=t80)
if __name__ == '__main__':
    J = [(sd, p) for sd in range(1, 21) for p in POL]
    with Pool(2) as P: out = P.map(job, J)
    json.dump(out, open('results_switch.json', 'w'), indent=1)
    for p in POL:
        R = [o for o in out if o['policy'] == p]
        f = lambda k: (np.mean([o[k] for o in R]), 1.96*np.std([o[k] for o in R], ddof=1)/np.sqrt(len(R)))
        print(p, *[f"{k}={f(k)[0]:.0f}±{f(k)[1]:.0f}" for k in ('fnd', 'hnd', 'lnd', 'node_rounds')])
