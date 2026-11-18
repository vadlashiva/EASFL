"""Tables for E1 (matched baselines) and E10 (paired statistics) from results_matched.json.

Paired comparisons use the same seed (same deployment, data and attacker set) for both
methods: mean difference, 95% bootstrap CI of the paired difference (10,000 resamples),
Wilcoxon signed-rank p-value, Holm-corrected within each table."""
import json, sys, collections
import numpy as np
from scipy.stats import wilcoxon

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_matched.json'))
NAME = {('FL-LEACH', ''): 'Mean (FedAvg)', ('FL-LEACH', 'median'): 'Coord. median',
        ('FL-LEACH', 'trimmed'): 'Trimmed mean', ('FL-LEACH', 'krum'): 'Krum',
        ('FL-LEACH', 'mmedian'): 'Support-aware median', ('EASFL', 'dense'): 'EASFL (dense filter)',
        ('EASFL', 'mask'): 'EASFL-M'}
ORDER = list(NAME.values())


def name(r):
    kw = r['kw']
    if r['scheme'] == 'EASFL':
        return NAME[('EASFL', 'mask' if kw.get('mcos') else 'dense')]
    return NAME[('FL-LEACH', kw.get('agg', '') if kw.get('agg') != 'mean' else '')]


def cond(r):
    return (r['kw'].get('attack', 'clean') if r['kw'].get('mal_frac', 0) else 'clean', r['kw'].get('mal_frac', 0.0))


D = collections.defaultdict(dict)   # (cond, method) -> seed -> result
for r in R:
    D[(cond(r), name(r))][r['seed']] = r


def val(r, m):
    if m == 'G':
        return r['final']['global']['f1']
    if m == 'P':
        return r['final']['ft+thr']['f1']
    if m == 'mbit':
        return r['mbit']
    if m == 'E':
        return r['energy']


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, (n, len(d))).mean(1)
    return np.percentile(b, [2.5, 97.5])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


conds = sorted({c for c, _ in D}, key=lambda c: (c[1], c[0]))
print('== E1: matched baselines (all methods: top-10% 8-bit + error feedback, rho=0.5, EA-CH, '
      '8-bit CH->BS, 16-bit broadcast). Global F1 mean+-std (n seeds); personalized F1 in []')
for c in conds:
    print(f'-- {c[0]} {int(c[1]*100)}%')
    for m in ORDER:
        S = D.get((c, m), {})
        if not S:
            continue
        g = np.array([val(r, 'G') for r in S.values()]); p = np.array([val(r, 'P') for r in S.values()])
        mb = np.mean([val(r, 'mbit') for r in S.values()]); e = np.mean([val(r, 'E') for r in S.values()])
        print(f'   {m:24s} G={g.mean():.3f}±{g.std():.3f} [P={p.mean():.3f}±{p.std():.3f}] '
              f'n={len(g):2d} Mbit={mb:.1f} E200={e:.2f} J')

print('\n== E10: paired differences EASFL-M minus baseline (global F1), seeds with both runs')
for c in conds:
    rows = []
    for m in ORDER:
        if m == 'EASFL-M' or (c, m) not in D or (c, 'EASFL-M') not in D:
            continue
        A, B = D[(c, 'EASFL-M')], D[(c, m)]
        s = sorted(set(A) & set(B))
        if len(s) < 5:
            continue
        d = np.array([val(A[k], 'G') - val(B[k], 'G') for k in s])
        try:
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        rows.append((m, d.mean(), boot_ci(d), p, len(s)))
    if not rows:
        continue
    adj = holm([r[3] for r in rows])
    print(f'-- {c[0]} {int(c[1]*100)}%')
    for (m, mu, ci, p, n), pa in zip(rows, adj):
        print(f'   vs {m:24s} diff={mu:+.3f} 95%CI[{ci[0]:+.3f},{ci[1]:+.3f}] '
              f'Wilcoxon p={p:.4f} Holm p={pa:.4f} n={n}')
