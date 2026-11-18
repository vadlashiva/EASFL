"""Tables for the realistic-radio validation (results_radio.json)."""
import json, sys
import numpy as np

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results_radio.json'))
runs = R['runs']
SCHEMES = ['CL', 'FL-flat', 'FL-LEACH', 'EASFL', 'EASFL-PD']
CATS = ['train', 'up', 'ch', 'ctrl', 'down']


def sel(rad, sc):
    return [x for x in runs if x['radio'] == rad and x['scheme'] == sc]


def ms(v):
    v = np.array(v, float)
    return f"{v.mean():.0f}±{v.std():.0f}"


def e_round(rad, sc):
    """Mean energy per node per round (J) over seeds, and its category split."""
    xs = sel(rad, sc)
    tot = np.mean([sum(x['energy'].values()) for x in xs])
    cat = {c: np.mean([x['energy'].get(c, 0.0) for x in xs]) for c in CATS}
    return tot, cat


print("== Table R1: lifetime (rounds, mean±std over 5 seeds) and energy per node per round (mJ)")
for rad in ['first-order', 'CC2420', 'CC2650']:
    for sc in SCHEMES:
        xs = sel(rad, sc); tot, cat = e_round(rad, sc)
        split = ' '.join(f"{c}={100 * cat[c] / tot:.0f}%" for c in CATS if cat[c] > 0)
        print(f"{rad:12s} {sc:9s} FND={ms([x['fnd'] for x in xs]):9s} HND={ms([x['hnd'] for x in xs]):9s} "
              f"LND={ms([x['lnd'] for x in xs]):9s} E={1e3 * tot:6.3f} mJ  [{split}]")
    e_l = e_round(rad, 'FL-LEACH')[0]; e_e = e_round(rad, 'EASFL')[0]
    f = lambda k: np.mean([x[k] for x in sel(rad, 'EASFL')]) / np.mean([x[k] for x in sel(rad, 'FL-LEACH')])
    print(f"{rad:12s} EASFL vs FedAvg-LEACH: energy/round {e_l / e_e:.1f}x lower, FND {f('fnd'):.1f}x, "
          f"HND {f('hnd'):.1f}x, LND {f('lnd'):.1f}x")

print("\n== Table R2: average-node lifetime gain with the sleep floor (radio standby current x round period)")
print("   gain = (floor + E_baseline) / (floor + E_EASFL), per node per round")
for rad in ['CC2420', 'CC2650']:
    v, i = R['radios'][rad]['v'], R['radios'][rad]['i_sleep']
    e = {sc: e_round(rad, sc)[0] for sc in SCHEMES}
    for T in [1, 10, 60]:
        fl = v * i * T * 60
        for ours in ['EASFL', 'EASFL-PD']:
            print(f"{rad} T={T:2d} min floor={1e3 * fl:7.3f} mJ  {ours} vs FedAvg-LEACH {(fl + e['FL-LEACH']) / (fl + e[ours]):.2f}x  "
                  f"vs FedAvg-flat {(fl + e['FL-flat']) / (fl + e[ours]):.2f}x  vs CL {(fl + e['CL']) / (fl + e[ours]):.2f}x")
