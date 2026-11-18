"""Print the Intel Lab tables from results_intel.json."""
import json, numpy as np
from collections import defaultdict
o = json.load(open('results_intel.json'))
m = lambda v: (np.mean(v), np.std(v))
life = defaultdict(list)
for r in o['life']:
    life[(r['scheme'], json.dumps(r['kw'], sort_keys=True))].append(r)
def lt(sc, kw={}):
    rs = life[(sc, json.dumps(kw, sort_keys=True))]
    return [np.mean([r[k] for r in rs]) for k in ('fnd', 'hnd', 'lnd', 'e200')]
print('scheme | F1 global | F1 pers | F1 net | FPR | AUC | Mbit | J(200) | alive | FND HND LND')
for name in ('central', 'local'):
    rs = o[name]
    g = m([r['final']['global']['f1'] for r in rs]); t = m([r['final']['thr']['f1'] for r in rs])
    net = m([r['final']['net']['f1'] for r in rs]) if 'net' in rs[0]['final'] else (np.nan, 0)
    print(name, '%.3f±%.3f' % g, '%.3f±%.3f' % t, '%.3f' % net[0], '%.3f' % np.mean([r['final']['global']['fpr'] for r in rs]),
          '%.3f' % np.mean([r['final']['global']['auc'] for r in rs]), np.mean([r.get('alive', 0) for r in rs]),
          lt('CL') if name == 'central' else '')
L = defaultdict(list)
for r in o['learn']:
    L[(r['scheme'], json.dumps(r['kw'], sort_keys=True))].append(r)
for (sc, kw), rs in L.items():
    f = lambda p, k='f1': m([r['final'][p][k] for r in rs])
    print(sc, kw, '%.3f±%.3f' % f('global'), '%.3f±%.3f' % f('ft+thr'), '%.3f±%.3f' % f('net'),
          '%.3f' % f('global', 'fpr')[0], '%.3f' % f('global', 'auc')[0], '%.1f' % np.mean([r['mbit'] for r in rs]),
          '%.2f' % np.mean([r['energy'] for r in rs]), np.mean([r['alive'] for r in rs]), lt(sc, json.loads(kw)) if (sc, kw) in life else '')
A = defaultdict(list)
for r in o['attack']:
    A[(r['kw']['attack'], r['scheme'], r['kw'].get('agg', 'mean'))].append(r['final']['global']['f1'])
print('attack (20%):')
for d in [('FL-LEACH', 'mean'), ('FL-LEACH', 'median'), ('FL-LEACH', 'trimmed'), ('FL-LEACH', 'krum'), ('EASFL', 'mean')]:
    print(d, ' '.join('%.2f±%.2f' % m(A[(a,) + d]) for a in ('signflip', 'stealth', 'labelflip', 'gauss')))
