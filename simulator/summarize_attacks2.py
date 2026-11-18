"""Tables for AE-1 and AE-2 from results_attacks2[_intel].json (global F1 mean±std over seeds,
paired Wilcoxon + Holm for the key comparisons)."""
import json, sys, collections
import numpy as np
from scipy.stats import wilcoxon

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_attacks2.json'))


def nm(r):
    k = r['kw']
    if r['scheme'] == 'FL-LEACH':
        return {None: 'Matched FedAvg', 'trimmed': 'Matched trimmed mean', 'mmedian': 'Matched support-aware median'}[k.get('agg')]
    if not k.get('mcos'):
        return 'EASFL (dense)'
    if k.get('sup_test'):
        return 'EASFL-M + support test'
    d = [n for n, f in (('BS filter', 'bs_filter'), ('cooldown', 'ch_cooldown'), ('energy check', 'e_check'), ('audit', 'audit')) if k.get(f)]
    return 'EASFL-M' + (' + ' + ' + '.join(d) if d else '')


def scen(r):
    k = r['kw']
    if k.get('ch_attack'):
        s = 'CH attack' + (' + false energy' if k.get('falsify') else '') + (' + member sign flip' if k.get('attack') == 'signflip' else '')
        return s
    if k.get('mal_frac', 0) == 0:
        return 'clean'
    return k['attack']


G = collections.defaultdict(dict)
for r in R:
    G[(scen(r), r['kw'].get('mal_frac', 0.0), nm(r))][r['seed']] = r
f1 = lambda r: r['final']['global']['f1']


def show(title, keys):
    print('\n== ' + title)
    for k in keys:
        S = G.get(k)
        if not S:
            continue
        v = np.array([f1(x) for x in S.values()])
        extra = ''
        if 'CH' in k[0]:
            extra = (f"  malicious CH share={np.mean([x['mal_ch_share'] for x in S.values()]):.2f}"
                     f" rounds with a malicious CH={np.mean([x['mal_ch_rounds'] for x in S.values()]):.2f}"
                     f" quarantined mal/honest={np.mean([x['quarantined_mal'] for x in S.values()]):.1f}/"
                     f"{np.mean([x['quarantined_honest'] for x in S.values()]):.1f}")
        print(f'{k[0]:36s} {int(k[1]*100):3d}%  {k[2]:44s} F1={v.mean():.3f}±{v.std():.3f} n={len(v)}'
              f' E200={np.mean([x["energy"] for x in S.values()]):.2f} J{extra}')


def paired(a, b):
    A, B = G.get(a, {}), G.get(b, {})
    s = sorted(set(A) & set(B))
    if len(s) < 5:
        return None
    d = np.array([f1(A[i]) - f1(B[i]) for i in s])
    try:
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
    except ValueError:
        p = 1.0
    return d.mean(), p, len(s)


keys = sorted(G, key=lambda k: (k[0], k[1], k[2]))
show('Clean cost of the new defences', [k for k in keys if k[0] == 'clean'])
show('AE-1 coordinate-targeting collusion', [k for k in keys if k[0].startswith('coordtarget')])
show('AE-2 compromised cluster heads', [k for k in keys if 'CH' in k[0]])

print('\n== Paired tests (difference in global F1, Wilcoxon; Holm within each block)')
blocks = []
for att in ('coordtarget', 'coordtarget_adapt'):
    for m in (0.1, 0.2, 0.3):
        a = (att, m, 'EASFL-M + support test')
        rows = [(b[2], paired(a, b)) for b in keys if b[0] == att and b[1] == m and b != a]
        blocks.append((f'{att} {int(m*100)}%: EASFL-M + support test vs', rows))
for sc in ('CH attack', 'CH attack + false energy', 'CH attack + false energy + member sign flip'):
    a = (sc, 0.2, 'EASFL-M + BS filter + cooldown + energy check + audit')
    rows = [(b[2], paired(a, b)) for b in keys if b[0] == sc and b[1] == 0.2 and b != a]
    blocks.append((f'{sc} 20%: all defences vs', rows))
for title, rows in blocks:
    rows = [(n, x) for n, x in rows if x]
    if not rows:
        continue
    p = np.array([x[1] for _, x in rows]); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1, run)
    print('-- ' + title)
    for (n, (d, pv, nn)), pa in zip(rows, adj):
        print(f'   {n:44s} diff={d:+.3f} p={pv:.4f} Holm={pa:.4f} n={nn}')
