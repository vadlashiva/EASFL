import warnings, numpy as np
warnings.filterwarnings('ignore')
from multiprocessing import Pool
def job(a):
    warnings.filterwarnings('ignore')
    from sim import Sim; from common import build_dataset
    sd, kw = a
    s = Sim('EASFL', sd, learn=True, data=build_dataset(sd), mcos=True, **kw); al = []
    while s.r < 1500 and s.step():
        al.append(int((s.E > 0).sum()))
        if al[-1] <= 10: break
    a_ = np.array(al); f = lambda m: int(np.argmax(m)) + 1 if m.any() else 1500
    return f(a_ < 100), f(a_ <= 50), f(a_ <= 10)
cfgs = {'EASFL-M': {}, '+audit': dict(audit=True), '+all+audit': dict(bs_filter=True, ch_cooldown=True, e_check=True, audit=True)}
J = [(sd, kw) for kw in cfgs.values() for sd in range(1, 6)]
with Pool(2) as p: out = p.map(job, J)
for i, k in enumerate(cfgs):
    o = np.array(out[i*5:(i+1)*5]); print(k, 'FND/HND/LND', o.mean(0).round(0))
