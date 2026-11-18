#!/usr/bin/env python3
"""Analyse an EASFL testbed run (run on the IoT-LAB frontend or on any PC after copying the files).

    python3 analyze_testbed.py <exp_id> [--oml-dir ~/.iot-lab/<exp_id>/consumption]

Inputs: roles.json and phases.csv written by easfl_testbed.py, and the power traces m3_<n>.oml.
Outputs (printed and saved as testbed_results.json):
  * idle power of every node (radio listening, no traffic)
  * per scheme and role (CH, participating member, non-participating member): energy per round
    ABOVE the idle level (transmission, retries, MCU work), and the time per round the radio is busy
    sending or receiving for that node
  * energy per round under duty cycling: above-idle energy + idle power x busy time, i.e. the radio
    is on only while the node sends or receives
  * per-frame calibration constants for the simulator (radio model 'measured' in common.py):
    e_tx_inc (J per transmitted frame above listening), t_frame (s of radio time per frame,
    including CSMA/CA backoff, ACK wait and retransmissions as they occurred), p_listen (W)
"""
import argparse, collections, csv, glob, json, math, os, re, statistics as st, sys


def load_oml(path):
    t, p = [], []
    for line in open(path, errors='replace'):
        f = line.split('\t')
        if len(f) >= 8:
            try:
                t.append(int(f[3]) + int(f[4]) * 1e-6); p.append(float(f[5]))
            except ValueError:
                pass
    return t, p


def integrate(t, p, a, b, base=0.0):
    """Energy (J) of (p - base) over [a, b] by the trapezoid rule on the samples inside."""
    import bisect
    i, j = bisect.bisect_left(t, a), bisect.bisect_right(t, b)
    e = 0.0
    for k in range(max(i, 1), j):
        e += 0.5 * ((p[k] - base) + (p[k - 1] - base)) * (t[k] - t[k - 1])
    return e


def mean_power(t, p, a, b):
    import bisect
    i, j = bisect.bisect_left(t, a), bisect.bisect_right(t, b)
    return st.mean(p[i:j]) if j > i else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_id')
    ap.add_argument('--oml-dir', default=None)
    a = ap.parse_args()
    oml_dir = os.path.expanduser(a.oml_dir or f'~/.iot-lab/{a.exp_id}/consumption')
    roles = json.load(open('roles.json'))
    phases = [r for r in csv.DictReader(open('phases.csv')) if r['t_start'] != 't_start']
    for r in phases:
        r['t_start'], r['t_end'] = float(r['t_start']), float(r['t_end'])
    traces = {}
    for f in glob.glob(os.path.join(oml_dir, '*.oml')):
        name = os.path.basename(f).split('.')[0].replace('_', '-')
        traces[name] = load_oml(f)
    if not traces:
        sys.exit(f'no .oml files in {oml_dir}')
    nodes = [roles['bs'], roles['ch']] + roles['members']
    missing = [n for n in nodes if n not in traces]
    if missing:
        print('warning: no power trace for', missing)

    idle = [r for r in phases if r['phase'] == 'idle'][0]
    p_idle = {n: mean_power(*traces[n], idle['t_start'] + 2, idle['t_end'] - 2) for n in nodes if n in traces}
    print('idle (listening) power, W: median %.4f  min %.4f  max %.4f' %
          (st.median(p_idle.values()), min(p_idle.values()), max(p_idle.values())))

    out = dict(exp_id=a.exp_id, p_idle=p_idle, schemes={})
    cfgs = roles['schemes']
    for scheme in cfgs:
        P = [r for r in phases if r['scheme'] == scheme and r['phase'] not in ('stats',)]
        rounds = sorted({int(r['round']) for r in P})
        if not rounds:
            continue
        per_role = collections.defaultdict(lambda: dict(above=0.0, busy=0.0, frames_tx=0, n=0))
        for rd in rounds:
            R = [r for r in P if int(r['round']) == rd]
            parts = set()
            for r in R:
                if r['phase'] == 'uplink':
                    parts = set(r['detail'].split())
            for n in nodes[1:]:
                if n not in traces:
                    continue
                role = 'CH' if n == roles['ch'] else ('member (participant)' if n in parts else 'member (idle)')
                e = 0.0; busy = 0.0; ftx = 0
                for r in R:
                    dur = r['t_end'] - r['t_start']
                    e += integrate(*traces[n], r['t_start'], r['t_end'], base=p_idle[n])
                    ph = r['phase']
                    # the node's radio must be on during phases in which it sends or receives
                    sends = ((ph == 'control' and role == 'CH') or (ph == 'join' and role != 'CH') or
                             (ph == 'uplink' and n in parts) or (ph == 'ch_bs' and role == 'CH'))
                    recvs = ((ph == 'control' and role != 'CH') or (ph == 'join' and role == 'CH') or
                             (ph == 'uplink' and role == 'CH') or ph == 'bcast')
                    if sends or recvs:
                        busy += dur
                    if sends:
                        nb = {'control': roles['ctrl'], 'join': roles['ctrl'], 'uplink': cfgs[scheme]['up'],
                              'ch_bs': cfgs[scheme]['ch_bs'] * (len(roles['members']) if cfgs[scheme]['ch_bs_per_member'] else 1)}[ph]
                        ftx += math.ceil(nb / 100)
                d = per_role[role]; d['above'] += e; d['busy'] += busy; d['frames_tx'] += ftx; d['n'] += 1
        res = {}
        for role, d in per_role.items():
            n = max(d['n'], 1); pi = st.median(p_idle.values())
            res[role] = dict(above_idle_mJ=1e3 * d['above'] / n, busy_s=d['busy'] / n, frames_tx=d['frames_tx'] / n,
                             duty_cycled_mJ=1e3 * (d['above'] + pi * d['busy']) / n)
        # cluster total per round (CH + all members), duty cycled
        tot = 0.0
        for role, d in per_role.items():
            tot += 1e3 * (d['above'] + st.median(p_idle.values()) * d['busy'])
        res['cluster_total_duty_cycled_mJ_per_round'] = tot / len(rounds)
        out['schemes'][scheme] = res
        print(f'\n== {scheme} ({len(rounds)} rounds)')
        for role, v in res.items():
            if isinstance(v, dict):
                print(f'   {role:22s} above idle {v["above_idle_mJ"]:8.3f} mJ  busy {v["busy_s"]:6.3f} s  '
                      f'tx frames {v["frames_tx"]:5.1f}  duty-cycled {v["duty_cycled_mJ"]:8.3f} mJ/round')
        print(f'   cluster total (duty cycled): {res["cluster_total_duty_cycled_mJ_per_round"]:.2f} mJ/round')

    # per-frame calibration from all transmitting phases of members and CH
    e_inc, t_frame = [], []
    for r in phases:
        if r['phase'] in ('uplink', 'join'):
            senders = r['detail'].split() if r['phase'] == 'uplink' else roles['members']
            cfg = cfgs.get(r['scheme'])
            if not cfg:
                continue
            nb = cfg['up'] if r['phase'] == 'uplink' else roles['ctrl']
            nf = math.ceil(nb / 100) * len(senders)
            e = sum(integrate(*traces[s], r['t_start'], r['t_end'], base=p_idle[s]) for s in senders if s in traces)
            e_inc.append(e / nf)
            t_frame.append((r['t_end'] - r['t_start']) / math.ceil(nb / 100))
    cal = dict(p_listen=st.median(p_idle.values()), e_tx_inc=st.median(e_inc), t_frame=st.median(t_frame),
               note='per IEEE 802.15.4 frame of up to 100 payload bytes, measured on FIT IoT-LAB M3 (AT86RF231)')
    out['calibration'] = cal
    print('\ncalibration for common.py (radio "measured"):', json.dumps(cal))
    for sc in ('FedAvg-LEACH', 'Centralized'):
        if sc in out['schemes'] and 'EASFL' in out['schemes']:
            r = out['schemes'][sc]['cluster_total_duty_cycled_mJ_per_round'] / out['schemes']['EASFL']['cluster_total_duty_cycled_mJ_per_round']
            print(f'measured energy per round, {sc} / EASFL = {r:.2f}x')
    json.dump(out, open('testbed_results.json', 'w'), indent=1)
    json.dump(cal, open('radio_measured.json', 'w'), indent=1)
    print('saved testbed_results.json and radio_measured.json')


if __name__ == '__main__':
    main()
