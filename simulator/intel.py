"""Intel Berkeley Research Lab workload for the EASFL-WSN study.

Real temperature, humidity, light and supply-voltage readings from the Intel Lab
deployment (54 Mica2Dot motes, 28 Feb - 5 Apr 2004; Bodik et al.,
https://db.csail.mit.edu/labdata/labdata.html) replace the synthetic generator, and the
real mote coordinates replace the random deployment.

Protocol
--------
* Period: 28 Feb - 9 Mar 2004 (50-52 motes report; later days lose motes to battery
  failure). Readings outside physical ranges (temp 5-45 C, humidity 0-100 %,
  light 0-3000 lux, voltage 2.0-3.1 V) are dropped.
* Nodes: the 50 motes with at least 300 gap-free windows (motes 5 and 28 never report,
  12 and 15 are too sparse).
* Windows: 10 consecutive received samples spanning at most 15 epochs (31 s each).
  300 windows per mote, evenly spaced over the period; the first 200 in time are used
  for training and the last 100 for testing (chronological split, no leakage).
* Ground truth: the raw traces carry no expert labels, so faults are injected into the
  real windows with the same four fault types, per-node rates and non-IID mixes as the
  synthetic workload (common.gen_windows). Injection amplitudes are scaled by each
  node's own measured noise level per channel (robust std of first differences), so
  faults are judged against real sensor noise, real diurnal cycles and real lighting
  changes.
* Network: the real coordinates (metres, lab of about 40.5 m x 31 m); the BS is placed
  75 m beyond the lab edge, the same offset as in the synthetic layout.
"""
import numpy as np
import common
import sim as simmod
from common import features, make_node_profiles, WIN, CH_SCALE

LAB_W, LAB_H = 40.5, 31.0
BS_INTEL = np.array([LAB_W / 2, LAB_H + 75.0])
N_TRAIN, N_TEST = 200, 100
MIN_WIN = N_TRAIN + N_TEST
MAX_SPAN = 14          # epochs between first and last sample of a window


def load_locs(path):
    a = np.loadtxt(path)
    return {int(m): (x, y) for m, x, y in a}


def load_windows(npz_path):
    """Return {mote: (raw windows (n, WIN, 4), noise scale (4,))} for usable motes."""
    d = np.load(npz_path)
    mote, epoch, X = d['mote'], d['epoch'], d['X'].astype(float)
    out = {}
    for m in np.unique(mote):
        sel = mote == m
        e, x = epoch[sel], X[sel]
        o = np.argsort(e); e, x = e[o], x[o]
        wins, i = [], 0
        while i + WIN <= len(e):
            if e[i + WIN - 1] - e[i] <= MAX_SPAN:
                wins.append(x[i:i + WIN]); i += WIN
            else:
                i += 1
        if len(wins) < MIN_WIN:
            continue
        W = np.array(wins)
        dif = np.diff(x, axis=0)
        mad = np.median(np.abs(dif - np.median(dif, 0)), 0)
        sc = np.maximum(1.4826 * mad / np.sqrt(2), 0.2 * CH_SCALE)   # floor: quantised channels
        out[int(m)] = (W, sc)
    return out


_CACHE = {}


def node_table(npz_path, locs_path):
    key = (npz_path, locs_path)
    if key not in _CACHE:
        wins = load_windows(npz_path)
        locs = load_locs(locs_path)
        motes = sorted(m for m in wins if m in locs)
        pos = np.array([locs[m] for m in motes], float)
        _CACHE[key] = (motes, pos, wins)
    return _CACHE[key]


def inject(rng, W, sc, p):
    """Inject faults into real windows W (n, WIN, 4) using profile p; return X, y."""
    X = W.copy()
    n = len(X)
    y = (rng.random(n) < p['rate']).astype(int)
    for w in np.where(y)[0]:
        kind = rng.choice(4, p=p['mix'])
        c = rng.integers(4)
        amp = rng.uniform(3, 8) * sc[c] * p.get('amp', 1.0)
        if kind == 0:
            j = rng.integers(WIN); X[w, j, c] += amp * rng.choice([-1, 1])
        elif kind == 1:
            X[w, :, c] = X[w, rng.integers(WIN), c]
        elif kind == 2:
            X[w, :, c] += np.linspace(0, amp * rng.uniform(1, 2), WIN) * rng.choice([-1, 1])
        else:
            X[w, :, c] += rng.normal(0, amp / 2, WIN)
    return features(X), y


def build_intel_dataset(seed, npz_path, locs_path, **gen):
    motes, _, wins = node_table(npz_path, locs_path)
    rng = np.random.default_rng(seed)
    prof = make_node_profiles(rng, n=len(motes), **gen)
    tr, te = [], []
    for m, p in zip(motes, prof):
        W, sc = wins[m]
        idx = np.linspace(0, len(W) - 1, MIN_WIN).round().astype(int)
        Ws = W[idx]
        tr.append(inject(rng, Ws[:N_TRAIN], sc, p))
        te.append(inject(rng, Ws[N_TRAIN:], sc, p))
    allX = np.concatenate([x for x, _ in tr])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    tr = [((x - mu) / sd, y) for x, y in tr]
    te = [((x - mu) / sd, y) for x, y in te]
    return prof, tr, te


def use_intel_layout(npz_path, locs_path):
    """Switch the simulator to the Intel Lab node count, positions and BS location."""
    motes, pos, _ = node_table(npz_path, locs_path)
    n = len(motes)
    for mod in (common, simmod):
        mod.N_NODES = n
        mod.BS = BS_INTEL
    simmod.INTEL_POS = pos
    return n


class IntelSim(simmod.Sim):
    """Sim with the real Intel Lab coordinates (energies stay heterogeneous per seed)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pos = simmod.INTEL_POS.copy()
        self.dBS = np.linalg.norm(self.pos - simmod.BS, axis=1)
        D = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        self.D = D
        self.deg = (D < 25).sum(1) - 1
