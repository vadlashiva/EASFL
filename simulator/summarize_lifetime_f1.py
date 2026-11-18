"""E2/E3 summary: network-level F1, live-node F1 and coverage over the whole lifetime,
plus the cumulative detection over the lifetime (mean network F1 over rounds 0..T)."""
import json, sys, collections
import numpy as np

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_lifetime_f1.json'))
T_LIST = (200, 400, 600, 800, 1000, 1500)


def nm(r):
    kw = r['kw']
    if r['scheme'] == 'EASFL':
        return 'EASFL-M' if kw.get('mcos') else 'EASFL (dense)'
    if r['scheme'] == 'FL-LEACH' and kw.get('matched'):
        return 'FedAvg matched'
    return {'Local': 'Local only', 'CL': 'Centralized'}.get(r['scheme'], r['scheme'])


G = collections.defaultdict(list)
for r in R:
    G[nm(r)].append(r)


def at(rows, t, j):
    for row in rows:
        if row[0] >= t:
            return row[j]
    return 0.0 if j in (1, 3) else float('nan')   # past the end: network dead


def auc(rows, T):
    """Mean network F1 over evaluation points up to T (dead network counts as 0)."""
    return float(np.mean([at(rows, t, 1) for t in range(25, T + 1, 25)]))


ORDER = ['Local only', 'Centralized', 'FL-flat', 'FL-LEACH', 'FedAvg matched', 'EASFL (dense)', 'EASFL-M']
print('Network-level F1 (dead nodes silent) at round t; mean over seeds')
print('scheme            ' + ''.join(f't={t:<6d}' for t in T_LIST))
for k in ORDER:
    if k in G:
        print(f'{k:18s}' + ''.join(f'{np.mean([at(r["rows"], t, 1) for r in G[k]]):<8.3f}' for t in T_LIST))
print('\nLive-node detection F1 at round t (nan = no node alive)')
for k in ORDER:
    if k in G:
        print(f'{k:18s}' + ''.join(f'{np.nanmean([at(r["rows"], t, 2) for r in G[k]]):<8.3f}' for t in T_LIST))
print('\nCoverage (fraction of nodes alive) at round t')
for k in ORDER:
    if k in G:
        print(f'{k:18s}' + ''.join(f'{np.mean([at(r["rows"], t, 3) for r in G[k]]):<8.2f}' for t in T_LIST))
print('\nCumulative detection: mean network F1 over rounds 1..T (mean±std over seeds)')
for k in ORDER:
    if k in G:
        print(f'{k:18s}' + ''.join(
            f'{np.mean([auc(r["rows"], T) for r in G[k]]):.3f}±{np.std([auc(r["rows"], T) for r in G[k]]):.3f} '
            for T in (200, 500, 1000, 1500)))
for r in G.get('Local only', [])[:1]:
    print('\nLocal-only mean energy per node-round (training + alarm reports):', round(r['e_round_mJ'], 3), 'mJ')
