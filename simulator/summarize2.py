"""Print the v2 result tables (mean ± std over seeds) from results_v2.json."""
import json, numpy as np, sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_v2.json'))


def fmt(a, d=3):
    a = [x for x in a if x is not None]
    if not a:
        return 'n/a (dead<200)'
    return f"{np.mean(a):.{d}f}±{np.std(a):.{d}f}"


def sel(rows, scheme=None, **kw):
    out = []
    for r in rows:
        if scheme and r.get('scheme') != scheme:
            continue
        if all(r.get('kw', {}).get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def det(rows, p, keys=('acc', 'prec', 'rec', 'f1', 'fpr', 'auc')):
    return ' '.join(f"{k}={fmt([r['final'][p][k] for r in rows])}" for k in keys)


if 'learn' in R:
    print('== Table A: detection after 200 rounds (global model | per-node threshold | fine-tune+threshold)')
    for name, rows in [('Centralized', R['central']), ('Local-only', R['local'])]:
        for p in ['global', 'thr']:
            print(f"{name:12s} {p:7s} {det(rows, p)}")
    for sc, kw, lab in [('FL-flat', {}, 'FedAvg-flat'), ('FL-LEACH', {}, 'FedAvg-LEACH'),
                        ('EASFL', {}, 'EASFL'), ('EASFL', {'mu': 0.1}, 'EASFL+prox')]:
        rows = [r for r in R['learn'] if r['scheme'] == sc and r['kw'] == kw]
        for p in ['global', 'thr', 'ft+thr', 'net']:
            if p in rows[0]['final']:
                print(f"{lab:12s} {p:7s} {det(rows, p)}")
        print(f"{'':12s} Mbit={fmt([r['mbit'] for r in rows], 1)} E200={fmt([r['energy'] for r in rows], 2)} "
              f"alive@200={fmt([r['alive'] for r in rows], 1)}")

if 'life' in R:
    print('\n== Table B: lifetime (rounds) and energy used in first 200 rounds (J)')
    keys = []
    for r in R['life']:
        k = (r['scheme'], json.dumps(r['kw'], sort_keys=True))
        if k not in keys:
            keys.append(k)
    for k in keys:
        rows = [r for r in R['life'] if (r['scheme'], json.dumps(r['kw'], sort_keys=True)) == k]
        print(f"{k[0]:9s} {k[1]:22s} FND={fmt([r['fnd'] for r in rows], 0)} HND={fmt([r['hnd'] for r in rows], 0)} "
              f"LND={fmt([r['lnd'] for r in rows], 0)} E200={fmt([r['e200'] for r in rows], 2)}")

if 'gen' in R:
    print('\n== Table C: sensitivity to the synthetic generator (F1 global / F1 ft+thr)')
    gens = []
    for r in R['gen']:
        if r['gen'] not in gens:
            gens.append(r['gen'])
    for g in gens:
        c = [r for r in R['gen_central'] if r['gen'] == g]
        line = f"{json.dumps(g):24s} CL={fmt([r['final']['global']['f1'] for r in c])} CL+thr={fmt([r['final']['thr']['f1'] for r in c])}"
        for sc in ['FL-LEACH', 'EASFL']:
            rows = [r for r in R['gen'] if r['gen'] == g and r['scheme'] == sc]
            line += f" | {sc} {fmt([r['final']['global']['f1'] for r in rows])} / {fmt([r['final']['ft+thr']['f1'] for r in rows])}"
        print(line)

if 'attack' in R:
    print('\n== Table D: F1 under poisoning (global / ft+thr)')
    labs = {}
    for r in R['attack']:
        kw = {k: v for k, v in r['kw'].items() if k not in ('mal_frac', 'attack')}
        labs.setdefault((r['scheme'], json.dumps(kw, sort_keys=True)), None)
    for att in ['none', 'signflip', 'stealth', 'labelflip', 'gauss']:
        for m in ([0.0] if att == 'none' else [0.1, 0.2, 0.3]):
            print(f"-- attack={att} mal={m}")
            for sc, kws in labs:
                kw = json.loads(kws)
                rows = [r for r in R['attack'] if r['scheme'] == sc and r['kw']['mal_frac'] == m
                        and (att == 'none' or r['kw'].get('attack') == att)
                        and {k: v for k, v in r['kw'].items() if k not in ('mal_frac', 'attack')} == kw]
                print(f"   {sc:9s} {kws:22s} {fmt([r['final']['global']['f1'] for r in rows])} / "
                      f"{fmt([r['final']['ft+thr']['f1'] for r in rows])}")

if 'loss' in R:
    print('\n== Table E: packet loss (F1 global / ft+thr)')
    for p in [0.0, 0.1, 0.2, 0.3]:
        for sc in ['FL-LEACH', 'EASFL']:
            rows = [r for r in R['loss'] if r['scheme'] == sc and r['kw']['p_loss'] == p]
            print(f"p={p} {sc:9s} {fmt([r['final']['global']['f1'] for r in rows])} / {fmt([r['final']['ft+thr']['f1'] for r in rows])}")

if 'rate' in R:
    print('\n== Table F: centralized baseline vs sampling rate (samples/node/round)')
    for S in [10, 30, 60, 120, 240]:
        for ea in [None, True]:
            rows = [r for r in R['rate'] if r['kw']['samples_per_round'] == S and r['kw'].get('ea_ch') == ea]
            print(f"S={S:3d} {'EA-CH' if ea else 'LEACH':6s} FND={fmt([r['fnd'] for r in rows], 0)} "
                  f"HND={fmt([r['hnd'] for r in rows], 0)} E200={fmt([r['e200'] for r in rows], 2)}")
