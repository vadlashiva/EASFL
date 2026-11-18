#!/usr/bin/env python3
"""EASFL revision (IoT-76474-2026): replay federated rounds on FIT IoT-LAB M3 nodes.

Run on the IoT-LAB SSH frontend of the site where the experiment runs (e.g. grenoble), while the
experiment is running with easfl_traffic.elf flashed on every node and a power-monitoring profile:

    python3 easfl_testbed.py --rounds 100            # full run (~25 min)
    python3 easfl_testbed.py --rounds 3 --smoke      # 2-minute check that everything works

Roles (by node number): the lowest-numbered node is the base station (BS, mains powered, its energy
is not counted), the next is the cluster head (CH), the rest are members. One cluster is replayed;
payload sizes are the bit counts of the paper (Table I) rounded up to bytes, without the abstract
200-bit header, because the real IEEE 802.15.4 PHY/MAC headers and ACKs are now paid by the radio.

Per round and scheme:
  control  CH broadcasts an advertisement (25 B); every member sends a join (25 B) to the CH
  uplink   participants send their update to the CH (EASFL: 50% of members, 94 B each;
           FedAvg-LEACH: all members, 1,668 B; centralized: all members, 480 B of raw samples)
  ch_bs    CH sends the aggregate to the BS (EASFL 421 B; FedAvg 1,668 B; centralized members x 480 B)
  bcast    BS broadcasts the model (EASFL 834 B, FedAvg 1,668 B; centralized: none)
A quiet interval with all radios listening and no traffic gives the idle baseline.
All phase boundaries are logged with wall-clock time to phases.csv; analyze_testbed.py aligns them
with the power traces in ~/.iot-lab/<exp_id>/consumption/.
"""
import argparse, json, os, re, socket, subprocess, sys, threading, time

SCHEMES = {
    #               up_bytes participation ch_bs_bytes(per round or per member)  bcast_bytes
    'EASFL':        dict(up=94,   part=0.5, ch_bs=421,  ch_bs_per_member=False, bcast=834),
    'FedAvg-LEACH': dict(up=1668, part=1.0, ch_bs=1668, ch_bs_per_member=False, bcast=1668),
    'Centralized':  dict(up=480,  part=1.0, ch_bs=480,  ch_bs_per_member=True,  bcast=0),
}
CTRL = 25


class Node:
    def __init__(self, host):
        self.host = host
        self.name = host.split('.')[0]
        self.num = int(re.findall(r'\d+', self.name)[-1])
        self.sock = socket.create_connection((self.name, 20000), timeout=10)
        self.sock.settimeout(0.2)
        self.buf = b''
        self.lock = threading.Lock()
        self.log = open(f'serial_{self.name}.log', 'a')

    def cmd(self, line, wait=0.0, expect=None, timeout=30):
        with self.lock:
            self.sock.sendall((line + '\n').encode())
            t0 = time.time(); out = ''
            while True:
                try:
                    chunk = self.sock.recv(4096)
                    if chunk:
                        s = chunk.decode(errors='replace'); out += s; self.log.write(s)
                except socket.timeout:
                    pass
                if expect is None and time.time() - t0 >= wait:
                    break
                if expect is not None and re.search(expect, out):
                    break
                if time.time() - t0 > timeout:
                    break
            self.log.flush()
            return out


def nodes_of_experiment():
    out = subprocess.run(['iotlab-experiment', 'get', '-n'], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    hosts = [it['network_address'] for it in d['items'] if it['archi'].startswith('m3')]
    return sorted(hosts, key=lambda h: int(re.findall(r'\d+', h.split('.')[0])[-1]))


def l2addr(node):
    out = node.cmd('ifconfig', wait=1.0)
    m = re.search(r'HWaddr:\s*([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})\b', out)
    if not m:
        raise RuntimeError(f'no short address from {node.name}:\n{out}')
    return m.group(1)


def parallel(calls):
    th = [threading.Thread(target=f, args=a) for f, a in calls]
    [t.start() for t in th]; [t.join() for t in th]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=100)
    ap.add_argument('--period', type=float, default=4.0, help='seconds per round (must exceed the longest phase)')
    ap.add_argument('--quiet', type=float, default=120.0, help='seconds of idle baseline')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--exp', type=int, default=None, help='experiment id (default: the running one)')
    a = ap.parse_args()
    if a.smoke:
        a.quiet = 20.0

    # power monitoring starts some tens of seconds after the experiment is 'Running':
    # wait until every node's consumption file is being written
    exp = a.exp or json.loads(subprocess.run(['iotlab-experiment', 'get', '-e'], capture_output=True, text=True).stdout or '{}').get('Running', [None])[0]
    if exp:
        d = os.path.expanduser(f'~/.iot-lab/{exp}/consumption')
        print(f'waiting for power data in {d} ...', flush=True)
        for _ in range(120):
            fs = [os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else []
            if fs and all(os.path.getsize(f) > 20000 for f in fs):
                break
            time.sleep(5)
        print('power data flowing', flush=True)
    else:
        print('experiment id unknown; waiting 120 s for power monitoring to start', flush=True)
        time.sleep(120)
    hosts = nodes_of_experiment()
    print('nodes:', [h.split('.')[0] for h in hosts], flush=True)
    nodes = [Node(h) for h in hosts]
    bs, ch, members = nodes[0], nodes[1], nodes[2:]
    addr = {n.name: l2addr(n) for n in nodes}
    print('addresses:', addr, flush=True)
    json.dump(dict(bs=bs.name, ch=ch.name, members=[m.name for m in members], addr=addr,
                   schemes=SCHEMES, ctrl=CTRL, rounds=a.rounds, period=a.period),
              open('roles.json', 'w'), indent=1)
    ph = open('phases.csv', 'a'); ph.write('t_start,t_end,scheme,round,phase,detail\n')

    def phase(scheme, r, name, calls, detail=''):
        t0 = time.time(); parallel(calls); t1 = time.time()
        ph.write(f'{t0:.6f},{t1:.6f},{scheme},{r},{name},{detail}\n'); ph.flush()

    burst = lambda n, dst, nbytes: (n.cmd, (f'burst {dst} {nbytes}', 0, r'BURST .*t_us=\d+', 60))

    # idle baseline: all radios listening, no traffic
    t0 = time.time(); time.sleep(a.quiet); ph.write(f'{t0:.6f},{time.time():.6f},idle,0,idle,\n'); ph.flush()
    for n in nodes:
        n.cmd('stats', wait=0.5)

    for scheme, cfg in SCHEMES.items():
        print(f'== {scheme}', flush=True)
        k = max(1, int(round(cfg['part'] * len(members))))
        for r in range(a.rounds):
            tr = time.time()
            phase(scheme, r, 'control', [burst(ch, 'bcast', CTRL)])
            phase(scheme, r, 'join', [burst(m, addr[ch.name], CTRL) for m in members])
            parts = members[(r * k) % len(members):] + members[:(r * k) % len(members)]
            parts = parts[:k]                                  # rotate participants over rounds
            phase(scheme, r, 'uplink', [burst(m, addr[ch.name], cfg['up']) for m in parts],
                  detail=' '.join(m.name for m in parts))
            nb = cfg['ch_bs'] * (len(members) if cfg['ch_bs_per_member'] else 1)
            phase(scheme, r, 'ch_bs', [burst(ch, addr[bs.name], nb)])
            if cfg['bcast']:
                phase(scheme, r, 'bcast', [burst(bs, 'bcast', cfg['bcast'])])
            rest = a.period - (time.time() - tr)
            if rest > 0:
                time.sleep(rest)
            if r % 10 == 0:
                print(f'   round {r} took {time.time() - tr:.2f} s', flush=True)
        for n in nodes:
            out = n.cmd('stats', wait=0.5)
            m = re.search(r'STATS .*', out)
            ph.write(f'{time.time():.6f},{time.time():.6f},{scheme},-1,stats,{n.name} {m.group(0) if m else "?"}\n')
        ph.flush()
    t0 = time.time(); time.sleep(a.quiet / 2); ph.write(f'{t0:.6f},{time.time():.6f},idle,1,idle,\n'); ph.flush()
    print(f'done; now run: python3 analyze_testbed.py {exp or "<exp_id>"}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
