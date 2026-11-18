"""Summaries for run_rev3.py stages (e6, e12, t2, t3, t5, t4) and the earlier 20-seed results."""
import json, os, sys, collections
import numpy as np
from scipy.stats import wilcoxon

L = lambda f: json.load(open(f)) if os.path.exists(f) else []
ms = lambda v: f'{np.mean(v):.3f}±{np.std(v):.3f}'
f1 = lambda r, k='global': r['final'][k]['f1']
which = sys.argv[1:] or ['e6', 'e12', 't2', 't3', 't5', 't4']

if 'e6' in which:
    print('== E6 inner-product manipulation (IPM, -0.5x own update) and stealth (-1x), global F1, 10 seeds')
    G = collections.defaultdict(dict)
    for r in L('results_rev3_e6.json'):
        _, sc, sd, kw = r['job']
        name = 'Matched trimmed mean' if sc == 'FL-LEACH' else ('EASFL-M w/o cosine' if kw.get('cosf') is False else 'EASFL-M')
        G[(kw['attack'], kw['mal_frac'], name)][sd] = f1(r)
    for k in sorted(G):
        print(f'   {k[0]:8s} {int(k[1]*100)}%  {k[2]:22s} {ms(list(G[k].values()))}')
    for a in ('ipm', 'stealth'):
        for m in (0.2, 0.3):
            A, B = G[(a, m, 'EASFL-M')], G[(a, m, 'EASFL-M w/o cosine')]
            d = np.array([A[s] - B[s] for s in sorted(A)])
            print(f'   cosine effect {a} {int(m*100)}%: diff={d.mean():+.3f} Wilcoxon p={wilcoxon(d).pvalue:.4f}')

if 'e12' in which:
    print('\n== E12 larger federated models (global F1, 5 seeds; traffic Mbit; E200 J)')
    base = {r['seed']: r for r in L('results_matched.json')}
    for hid, P in (('80', 2081), ('400', 10401)):
        R = L(f'results_rev3_e12_hid{hid}.json')
        if not R:
            continue
        G = collections.defaultdict(list)
        for r in R:
            _, sc, sd, kw = r['job']
            G[('EASFL-M' if sc == 'EASFL' else 'Matched FedAvg', kw.get('attack') or 'clean')].append(r)
        for k in sorted(G):
            v = G[k]
            print(f'   P={P:6d} {k[0]:15s} {k[1]:8s} F1={ms([f1(x) for x in v])} pers={ms([f1(x, "ft+thr") for x in v])} '
                  f'Mbit={np.mean([x["mbit"] for x in v]):.1f} E200={np.mean([x["energy"] for x in v]):.2f} alive={np.mean([x["alive"] for x in v]):.0f}')

if 't2' in which:
    print('\n== Table II with 20 seeds')
    old = json.load(open('results_v2.json')); new = L('results_rev3_t2.json')
    rows = collections.defaultdict(list)
    for r in old['central']:
        if not r['gen']: rows['Centralized'].append(r['final'])
    for r in old['local']:
        if not r['gen']: rows['Local only'].append(r['final'])
    for r in old['learn']:
        if r['kw'] == {} and r['scheme'] in ('FL-flat', 'FL-LEACH'): rows[r['scheme']].append(dict(r['final'], mbit=r['mbit'], energy=r['energy'], alive=r['alive']))
    for r in new:
        kind, sc = r['job'][0], r['job'][1]
        if kind == 'central': rows['Centralized'].append(r['final'])
        elif kind == 'local': rows['Local only'].append(r['final'])
        else: rows[sc].append(dict(r['final'], mbit=r['mbit'], energy=r['energy'], alive=r['alive']))
    for r in L('results_matched.json'):
        if r['scheme'] == 'EASFL' and r['kw'] == {'mcos': True}:
            rows['EASFL-M'].append(dict(r['final'], mbit=r['mbit'], energy=r['energy'], alive=r['alive']))
    for k, v in rows.items():
        g = [x['global']['f1'] for x in v]; auc = np.mean([x['global']['auc'] for x in v]); fpr = np.mean([x['global']['fpr'] for x in v])
        pk = 'ft+thr' if 'ft+thr' in v[0] else 'thr'
        p = [x[pk]['f1'] for x in v]; pf = np.mean([x[pk]['fpr'] for x in v])
        net = [x['net']['f1'] for x in v if 'net' in x]
        extra = (f' net={ms(net)} Mbit={np.mean([x["mbit"] for x in v]):.1f} E200={np.mean([x["energy"] for x in v]):.1f} '
                 f'alive={np.mean([x["alive"] for x in v]):.1f}') if net else ''
        print(f'   {k:12s} n={len(v):2d} G={ms(g)} AUC={auc:.3f} FPR={fpr:.3f} P={ms(p)} FPR(P)={pf:.3f}{extra}')

if 't3' in which:
    print('\n== Table III with 20 seeds (FND/HND/LND, E200)')
    G = collections.defaultdict(list)
    for r in L('results_rev3_t3.json'):
        G[(r['job'][1], json.dumps(r['job'][3]))].append(r)
    for k, v in G.items():
        print(f'   {k[0]:9s} {k[1]:22s} FND={np.mean([x["fnd"] for x in v]):5.0f}±{np.std([x["fnd"] for x in v]):3.0f} '
              f'HND={np.mean([x["hnd"] for x in v]):5.0f}±{np.std([x["hnd"] for x in v]):3.0f} LND={np.mean([x["lnd"] for x in v]):5.0f}±{np.std([x["lnd"] for x in v]):3.0f} '
              f'E200={np.mean([x["e200"] for x in v]):6.2f}')

if 't5' in which:
    print('\n== Table V (Intel Lab) with 20 seeds')
    old = json.load(open('results_intel.json')); new = L('results_rev3_t5.json')
    rows = collections.defaultdict(list); life = collections.defaultdict(list); cltr = []
    for r in old.get('central', []): rows['Centralized'].append(r)
    for r in old.get('local', []): rows['Local only'].append(r)
    for r in old.get('learn', []):
        if r['kw'] == {} and r['scheme'] in ('FL-flat', 'FL-LEACH'): rows[r['scheme']].append(r)
    for r in new:
        kind, sc, sd, kw = r['job']
        if kind == 'icentral': rows['Centralized'].append(r)
        elif kind == 'ilocal': rows['Local only'].append(r)
        elif kind == 'ilearn':
            nm = sc if sc != 'EASFL' else 'EASFL-M ' + ','.join(f'{k}={v}' for k, v in kw.items() if k != 'mcos')
            rows[nm].append(r)
        elif kind == 'ilife': life[(sc, json.dumps(kw))].append(r)
        elif kind == 'icl_traffic': cltr.append(r['mbit'])
    for r in L('results_matched_intel.json'):
        if r['scheme'] == 'EASFL' and r['kw'] in ({'mcos': True}, {}):
            rows['EASFL-M' if r['kw'] else 'EASFL (dense)'].append(r)
    for k, v in rows.items():
        g = [x['final']['global']['f1'] for x in v]
        pk = 'ft+thr' if 'ft+thr' in v[0]['final'] else 'thr'
        p = [x['final'][pk]['f1'] for x in v]
        net = [x['final']['net']['f1'] for x in v if 'net' in x['final']]
        extra = f' Mbit={np.mean([x["mbit"] for x in v]):.1f} E={np.mean([x["energy"] for x in v]):.2f} alive={np.mean([x["alive"] for x in v]):.1f}' if 'mbit' in v[0] else ''
        if 'alive' in v[0] and 'mbit' not in v[0]: extra = f' alive={np.mean([x["alive"] for x in v]):.1f}'
        print(f'   {k:30s} n={len(v):2d} G={ms(g)} P={ms(p)}' + (f' net={ms(net)}' if net else '') + extra)
    print(f'   Centralized traffic over 200 rounds: {np.mean(cltr):.1f} Mbit (n={len(cltr)})')
    for k, v in life.items():
        print(f'   life {k[0]:9s} {k[1]:22s} FND={np.mean([x["fnd"] for x in v]):5.0f} HND={np.mean([x["hnd"] for x in v]):5.0f} '
              f'LND={np.mean([x["lnd"] for x in v]):5.0f} E200={np.mean([x["e200"] for x in v]):.2f} n={len(v)}')

if 't4' in which:
    print('\n== Table IV with 20 seeds (global / personalized F1, 20% malicious)')
    old = json.load(open('results_v2.json'))['attack']; new = L('results_rev3_t4.json')
    G = collections.defaultdict(list)
    def nm(sc, kw):
        if sc == 'FL-LEACH':
            return {None: 'FedAvg (mean)', 'median': 'Median', 'trimmed': 'Trimmed mean', 'krum': 'Krum'}[kw.get('agg')]
        return 'EASFL-M w/o robust' if kw.get('robust') is False else 'EASFL-M w/o cosine'
    for r in old:
        kw = r['kw']
        if r['scheme'] == 'FL-LEACH' and kw.get('mal_frac') == 0.2:
            G[(kw['attack'], nm('FL-LEACH', kw))].append(r['final'])
    for r in new:
        _, sc, sd, kw = r['job']
        G[(kw['attack'], nm(sc, kw))].append(r['final'])
    for r in L('results_matched.json'):
        if r['scheme'] == 'EASFL' and r['kw'].get('mcos') and r['kw'].get('mal_frac') == 0.2:
            G[(r['kw']['attack'], 'EASFL-M')].append(r['final'])
    names = ['FedAvg (mean)', 'Median', 'Trimmed mean', 'Krum', 'EASFL-M w/o robust', 'EASFL-M w/o cosine', 'EASFL-M']
    for n in names:
        cells = []
        for a in ('signflip', 'stealth', 'labelflip', 'gauss'):
            v = G.get((a, n), [])
            cells.append(f'{np.mean([x["global"]["f1"] for x in v]):.2f}/{np.mean([x["ft+thr"]["f1"] for x in v]):.2f}(n={len(v)})' if v else '--')
        print(f'   {n:20s} ' + '  '.join(cells))
