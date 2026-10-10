import numpy as np, time, sys
from sim import Sim
from common import *

def lifetime(scheme, seed, maxr=4000, **kw):
    s = Sim(scheme, seed, **kw)
    alive = []
    e_tot0 = s.E0.sum()
    e200 = None
    while s.r < maxr:
        if not s.step():
            break
        alive.append(int((s.E > 0).sum()))
        if s.r == 200:
            e200 = e_tot0 - np.clip(s.E, 0, None).sum()
    a = np.array(alive)
    fnd = int(np.argmax(a < N_NODES)) + 1 if (a < N_NODES).any() else maxr
    hnd = int(np.argmax(a <= N_NODES // 2)) + 1 if (a <= N_NODES // 2).any() else maxr
    lnd = int(np.argmax(a <= N_NODES // 10)) + 1 if (a <= N_NODES // 10).any() else maxr
    return dict(fnd=fnd, hnd=hnd, lnd=lnd, e200=e200, alive=a)

if __name__ == '__main__':
    for sc in ['CL', 'FL-flat', 'FL-LEACH', 'EASFL']:
        t = time.time()
        r = lifetime(sc, 0)
        print(sc, r['fnd'], r['hnd'], r['lnd'], round(r['e200'], 3), round(time.time() - t, 1))
