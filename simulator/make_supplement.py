"""Builds paper_iotj_revision/supplementary.tex (Tables S1-S4) from the result files."""
import json, collections, numpy as np
ms = lambda v: f'{np.mean(v):.3f}$\\pm${np.std(v):.3f}'
out = [r'''\documentclass[journal,onecolumn]{IEEEtran}
\usepackage{booktabs,amsmath}
\begin{document}
\title{Supplementary Material for ``EASFL: Energy-Aware Federated Anomaly Detection in Wireless Sensor Networks With Mask-Aware Robust Aggregation''}
\author{Shiva~Kumar~V and~Choupiri~Shivakeshi}
\maketitle\renewcommand{\thetable}{S\arabic{table}}
All results are produced by the scripts in \texttt{simulator/} of https://github.com/vadlashiva/EASFL (branch \texttt{iotj-revision}); the file that produces each table is named in its caption.
''']
# S1 sweeps
R = json.load(open('results_sweeps.json')); G = collections.defaultdict(list)
for r in R:
    G[(r['setting'], r['scheme'] + (' (matched)' if r['kw'] else ''))].append(r)
LBL = {'e_s 0.5 uJ': '$e_s$ 0.5\\,$\\mu$J', 'e_s 8 uJ': '$e_s$ 8\\,$\\mu$J', 'E_elec 25 nJ': '$E_{\\mathrm{elec}}$ 25\\,nJ/bit', 'E_elec 100 nJ': '$E_{\\mathrm{elec}}$ 100\\,nJ/bit', 'p=0.05': '$p=0.05$', 'p=0.2': '$p=0.2$', 'N=50': '$N=50$', 'N=200': '$N=200$'}
settings = []
for r in R:
    if r['setting'] not in settings: settings.append(r['setting'])
out.append(r'''\begin{table}[h]\caption{Sensitivity of Lifetime and Energy to Network and Radio Parameters (Rounds; $E_{200}$ in J; Mean of 5 Seeds; \texttt{run\_sweeps.py}). --: Network Dead Before Round 200}\centering\small
\begin{tabular}{@{}llrrrr@{}}\toprule Setting & Scheme & FND & HND & LND & $E_{200}$\\\midrule''')
for st in settings:
    for sc in ('CL', 'FL-LEACH', 'EASFL'):
        L = G[(st, sc)]
        e = [x['e200'] for x in L if x['e200'] is not None]
        nm = {'CL': 'Centralized', 'FL-LEACH': 'FedAvg-LEACH', 'EASFL': 'EASFL'}[sc]
        out.append(f"{LBL.get(st, st) if sc == 'CL' else ''} & {nm} & {np.mean([x['fnd'] for x in L]):.0f} & {np.mean([x['hnd'] for x in L]):.0f} & "
                   f"{np.mean([x['lnd'] for x in L]):.0f} & {np.mean(e):.2f}\\\\" if len(e) == len(L) else
                   f"{LBL.get(st, st) if sc == 'CL' else ''} & {nm} & {np.mean([x['fnd'] for x in L]):.0f} & {np.mean([x['hnd'] for x in L]):.0f} & {np.mean([x['lnd'] for x in L]):.0f} & --\\\\")
    out.append(r'\midrule' if st != settings[-1] else '')
out.append(r'\bottomrule\end{tabular}\end{table}')
out.append('Matched FedAvg (FedAvg with EASFL\'s compression, participation and CH policy) has the same lifetime as EASFL in every setting, because the aggregation rule has no radio cost.\n')
# S2 unseen faults
R = json.load(open('results_rev3_e7.json')); G = collections.defaultdict(list)
for r in R:
    G[(r['job'][1], r['job'][4])].append(r['final'])
fam = ['Spike', 'Stuck-at', 'Drift', 'Noise burst']
out.append(r'''\begin{table}[h]\caption{Recall per Fault Family When One Family Is Absent From the Training Windows (Synthetic, 5 Seeds; \texttt{run\_rev3.py e7}). Diagonal Entries Are the Held-Out Family}\centering\small
\begin{tabular}{@{}llccccc@{}}\toprule Scheme & Left out & Spike & Stuck-at & Drift & Burst & F1 (all)\\\midrule''')
for sc, key, nm in (('CL', 'ft+thr', 'Centralized'), ('EASFL', 'ft+thr', 'EASFL-M, personalized')):
    for ex in (None, 0, 1, 2, 3):
        L = G[(sc, ex)]
        rec = [np.mean([x[key]['recall_by_family'][str(f)] for x in L]) for f in range(4)]
        cells = [(f'\\textbf{{{v:.2f}}}' if ex == f else f'{v:.2f}') for f, v in enumerate(rec)]
        out.append(f"{nm if ex is None else ''} & {'none' if ex is None else fam[ex]} & " + ' & '.join(cells) + f" & {np.mean([x[key]['f1'] for x in L]):.3f}\\\\")
    out.append(r'\midrule' if sc == 'CL' else '')
out.append(r'\bottomrule\end{tabular}\end{table}')
# S3 detectors + model size
D = json.load(open('results_detectors.json'))
out.append(r'''\begin{table}[h]\caption{Detector Architectures, Centralized Training on Pooled Synthetic Windows (3 Seeds; \texttt{run\_detectors.py}), and Larger Federated Models (5 Seeds; \texttt{run\_rev3.py e12})}\centering\small
\begin{tabular}{@{}lrc@{}}\toprule Detector (input) & Parameters & Test F1\\\midrule''')
for k in D['1']:
    v = [D[s][k]['f1'] for s in D]; p = D['1'][k]['params']
    out.append(f"{k} & {p if p else '--'} & {ms(v)}\\\\")
out.append(r'\midrule Federated, 20\% sign flip & & Global F1 (clean / attack); alive at 200\\\midrule')
for hid, P in (('80', 2081), ('400', 10401)):
    R = json.load(open(f'results_rev3_e12_hid{hid}.json')); G = collections.defaultdict(list)
    for r in R:
        _, sc, sd, kw = r['job']; G[(sc, kw.get('attack') or 'clean')].append(r)
    for sc, nm in (('EASFL', 'EASFL-M'), ('FL-LEACH', 'Matched FedAvg')):
        c, a = G[(sc, 'clean')], G[(sc, 'signflip')]
        out.append(f"MLP 24-{hid}-1, {nm} & {P:,} & {np.mean([x['final']['global']['f1'] for x in c]):.3f} / {np.mean([x['final']['global']['f1'] for x in a]):.3f}; "
                   f"{np.mean([x['alive'] for x in c]):.0f} nodes, $E_{{200}}$={np.mean([x['energy'] for x in c]):.1f}\\,J\\\\")
out.append(r'\bottomrule\end{tabular}\end{table}')
# S4 IPM / stealth cosine ablation
R = json.load(open('results_rev3_e6.json')); G = collections.defaultdict(list)
for r in R:
    _, sc, sd, kw = r['job']
    nm = 'Matched trimmed mean' if sc == 'FL-LEACH' else ('EASFL-M without direction test' if kw.get('cosf') is False else 'EASFL-M')
    G[(kw['attack'], kw['mal_frac'], nm)].append(r['final']['global']['f1'])
out.append(r'''\begin{table}[h]\caption{Direction-Test Ablation Under Inner-Product Manipulation ($-0.5\Delta_i$) and Stealth Sign Flip ($-\Delta_i$) (Global F1, 10 Seeds; \texttt{run\_rev3.py e6})}\centering\small
\begin{tabular}{@{}lcccc@{}}\toprule Rule & IPM 20\% & IPM 30\% & Stealth 20\% & Stealth 30\%\\\midrule''')
for nm in ('Matched trimmed mean', 'EASFL-M without direction test', 'EASFL-M'):
    out.append(nm + ' & ' + ' & '.join(ms(G[(a, m, nm)]) for a in ('ipm', 'stealth') for m in (0.2, 0.3)) + '\\\\')
out.append(r'\bottomrule\end{tabular}\end{table}')
out.append(r'\end{document}')
open('/home/claude/easfl/paper_iotj_revision/supplementary.tex', 'w').write('\n'.join(out))
