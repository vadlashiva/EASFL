"""Re-run lifetime and E200 with the radio constants measured on FIT IoT-LAB (radio_measured.json from
testbed/analyze_testbed.py) and compare the ranking with the first-order model. Seeds 1-20.
Usage: python calibrate_sim.py radio_measured.json -> results_calibrated.json"""
import json, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
PATH = sys.argv[1] if len(sys.argv) > 1 else 'radio_measured.json'
SCHEMES = [('CL', {}), ('FL-flat', {}), ('FL-LEACH', {}), ('FL-LEACH', {'matched': True}), ('EASFL', {}),
           ('EASFL', {'compress': False}), ('EASFL', {'psel': False})]

def job(a):
    import common
    if a[0] == 'measured':
        common.set_radio('measured', PATH)
    from run_quick import lifetime
    r = lifetime(a[1], a[3], **a[2]); r.pop('alive', None)
    return dict(radio=a[0], scheme=a[1], kw=a[2], seed=a[3], **{k: (None if v is None else float(v)) for k, v in r.items()})

if __name__ == '__main__':
    J = [(rad, sc, kw, sd) for rad in ('first-order', 'measured') for sc, kw in SCHEMES for sd in range(1, 21)]
    with Pool(2) as p:
        R = p.map(job, J)
    json.dump(R, open('results_calibrated.json', 'w'))
    for rad in ('first-order', 'measured'):
        print(f'== radio model: {rad}')
        rows = []
        for sc, kw in SCHEMES:
            L = [r for r in R if r['radio'] == rad and r['scheme'] == sc and r['kw'] == kw]
            f = np.mean([r['fnd'] for r in L]); h = np.mean([r['hnd'] for r in L]); l = np.mean([r['lnd'] for r in L])
            rows.append((sc + (' ' + json.dumps(kw) if kw else ''), f, h, l))
            print(f'   {rows[-1][0]:32s} FND={f:7.0f} HND={h:7.0f} LND={l:7.0f}')
        print('   ranking by FND:', ' > '.join(r[0] for r in sorted(rows, key=lambda x: -x[1])))
        print('   ranking by HND:', ' > '.join(r[0] for r in sorted(rows, key=lambda x: -x[2])))
