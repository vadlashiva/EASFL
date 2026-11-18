"""Realistic-radio validation: lifetime and per-round energy under CC2420 and CC2650
(IEEE 802.15.4, datasheet currents, frame overheads), plus the effect of the sleep floor
for different round periods. Writes results_radio.json; summarize_radio.py prints the tables."""
import json, sys
import numpy as np
import common
from sim import Sim
from run_quick import lifetime

SEEDS = [1, 2, 3, 4, 5]
SCHEMES = [('CL', {}), ('FL-flat', {}), ('FL-LEACH', {}), ('EASFL', {}), ('EASFL', {'pd': True})]
RADIOS = ['first-order', 'CC2420', 'CC2650']
WARM = 10            # rounds used for the per-round energy breakdown (all nodes alive)


def breakdown(scheme, seed, **kw):
    s = Sim(scheme, seed, **kw)
    for _ in range(WARM):
        s.step()
    per_node_round = {k: v / WARM / common.N_NODES for k, v in s.ecat.items()}
    return per_node_round


def main():
    out = []
    for rad in RADIOS:
        common.set_radio(rad)
        for sc, kw in SCHEMES:
            for sd in SEEDS:
                r = lifetime(sc, sd, maxr=20000, **kw)
                out.append(dict(radio=rad, scheme=sc + ('-PD' if kw.get('pd') else ''), seed=sd, fnd=r['fnd'],
                                hnd=r['hnd'], lnd=r['lnd'], energy=breakdown(sc, sd, **kw)))
                print(rad, sc, sd, r['fnd'], r['hnd'], r['lnd'], flush=True)
    json.dump(dict(runs=out, radios=common.RADIOS, rate=common.RATE, max_payload=common.MAX_PAYLOAD,
                   frame_ovh=common.FRAME_OVH, ack=common.ACK_BYTES, t_startup=common.T_STARTUP),
              open('results_radio.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
