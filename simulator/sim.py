"""Round-based WSN simulator with energy accounting and optional federated learning.

Schemes: 'CL' (centralized, raw data via LEACH clusters), 'FL-flat' (FedAvg, direct to BS),
'FL-LEACH' (FedAvg over LEACH clusters), 'EASFL' (proposed). EASFL flags allow ablations.
"""
import numpy as np
from common import *

class Sim:
    def __init__(self, scheme, seed, learn=False, data=None, samples_per_round=60,
                 p_loss=0.05, mal_frac=0.0, rho=0.5, ea_ch=None, compress=True,
                 robust=True, psel=True, local_cap=150, attack='signflip', agg='mean', cosf=True, mu=0.0):
        self.s = scheme; self.rng = np.random.default_rng(1000 + seed)
        net = np.random.default_rng(seed)               # same deployment per seed
        self.pos = net.uniform(0, FIELD, (N_NODES, 2))
        self.E0 = net.uniform(E_INIT_LO, E_INIT_HI, N_NODES)
        self.E = self.E0.copy()
        self.dBS = np.linalg.norm(self.pos - BS, axis=1)
        D = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        self.D = D; self.deg = (D < 25).sum(1) - 1
        self.S = samples_per_round; self.p_loss = p_loss
        self.rho, self.ea_ch_flag, self.compress, self.robust, self.psel = rho, ea_ch, compress, robust, psel
        self.ea_ch = (scheme == 'EASFL') if self.ea_ch_flag is None else self.ea_ch_flag
        self.last_ch = np.full(N_NODES, -10**6)
        self.r = 0
        self.mal = np.zeros(N_NODES, bool)
        if mal_frac > 0:
            self.mal[net.choice(N_NODES, int(round(mal_frac * N_NODES)), replace=False)] = True
        self.flags = np.zeros(N_NODES); self.ban_until = np.zeros(N_NODES)
        self.learn = learn; self.cap = local_cap
        self.attack, self.agg, self.cosf, self.mu = attack, agg, cosf, mu
        self.bits_up = 0.0
        if learn:
            self.prof, self.tr, self.te = data
            self.w = init_params(np.random.default_rng(seed))
            self.resid = np.zeros((N_NODES, N_PARAM))

    # ------------------------------------------------------------------ clustering
    def alive(self):
        return self.E > 0

    def leach_ch(self, al):
        p = P_CH; period = int(round(1 / p))
        elig = al & (self.r - self.last_ch >= period)
        T = p / (1 - p * (self.r % period))
        ch = elig & (self.rng.random(N_NODES) < T)
        return np.where(ch)[0]

    def ea_select_ch(self, al):
        idx = np.where(al)[0]
        k = max(1, int(round(P_CH * len(idx))))
        e = self.E[idx] / self.E[idx].max()
        d = self.dBS[idx]; dn = 1 - (d - d.min()) / (d.max() - d.min() + 1e-9)
        g = self.deg[idx] / max(self.deg[idx].max(), 1)
        rep = (self.ban_until[idx] <= self.r)
        score = (0.5 * e + 0.3 * dn + 0.2 * g) * rep
        chosen = []
        for j in idx[np.argsort(-score)]:
            if all(self.D[j, c] >= 20 for c in chosen):
                chosen.append(j)
            if len(chosen) == k:
                break
        return np.array(chosen, int)

    def spend(self, i, e):
        self.E[i] -= e

    # ------------------------------------------------------------------ one round
    def step(self):
        al = self.alive()
        if not al.any():
            return False
        s = self.s
        updates = {}          # node -> (vector, n_samples) delivered to its aggregator
        clusters = {}
        if s == 'FL-flat':
            parts = np.where(al)[0]
            for i in parts:
                self.spend(i, min(self.cap, 200) * E_SAMPLE_EPOCH)
                b = BITS_F32 + HEADER
                self.spend(i, tx_energy(b, self.dBS[i])); self.bits_up += b
            delivered = parts[self.rng.random(len(parts)) >= self.p_loss]
            if self.learn:
                ups = [self.local_update(i) for i in delivered]
                if ups:
                    self.w = self.w + robust_agg(np.array(ups), self.agg)
            for i in np.where(al)[0]:
                self.spend(i, rx_energy(BITS_F32 + HEADER))
        else:
            chs = self.ea_select_ch(al) if self.ea_ch else self.leach_ch(al)
            if len(chs) == 0:
                chs = np.array([np.where(al)[0][np.argmin(self.dBS[al])]])
            self.last_ch[chs] = self.r
            for c in chs:  # CH advertisement broadcast (radius ~ 30 m)
                self.spend(c, tx_energy(CTRL, 30.0))
            mem = np.setdiff1d(np.where(al)[0], chs)
            if len(mem):
                dmc = self.D[np.ix_(mem, chs)]
                near = chs[np.argmin(dmc, 1)]; dnear = dmc.min(1)
            for c in chs:
                clusters[c] = [c]
            for k, i in enumerate(mem):
                self.spend(i, rx_energy(CTRL * len(chs)) + tx_energy(CTRL, dnear[k]))
                self.spend(near[k], rx_energy(CTRL))
                clusters[near[k]].append(i)
            agg_bs = []
            for c, members in clusters.items():
                members = np.array(members)
                dist = self.D[c, members]
                if s == 'CL':
                    b = self.S * 4 * 16 + HEADER
                    raw_bits = 0
                    for i, d in zip(members, dist):
                        if i != c:
                            self.spend(i, tx_energy(b, d)); self.spend(c, rx_energy(b))
                            self.bits_up += b          # member -> CH hop (counted as in FL schemes)
                        raw_bits += b
                    self.spend(c, tx_energy(raw_bits, self.dBS[c])); self.bits_up += raw_bits
                    continue
                # ---- FL schemes: participant selection
                if s == 'EASFL' and self.psel:
                    ok = members[(self.E[members] > 0.1 * E_INIT_LO) & (self.ban_until[members] <= self.r)]
                    n_take = int(np.ceil(self.rho * len(members)))
                    parts = ok[np.argsort(-self.E[ok])][:n_take]
                else:
                    parts = members
                b_up = (BITS_TOPK8 if (s == 'EASFL' and self.compress) else BITS_F32) + HEADER
                recv = []
                for i in parts:
                    self.spend(i, min(self.cap, 200) * E_SAMPLE_EPOCH)
                    if i != c:
                        self.spend(i, tx_energy(b_up, self.D[c, i])); self.bits_up += b_up
                        if self.rng.random() < self.p_loss:
                            continue
                        self.spend(c, rx_energy(b_up) + E_DA * b_up)
                    recv.append(i)
                b_ch = (BITS_Q8 if (s == 'EASFL' and self.compress) else BITS_F32) + HEADER
                if len(recv):
                    self.spend(c, tx_energy(b_ch, self.dBS[c])); self.bits_up += b_ch
                    lost = self.rng.random() < self.p_loss
                    if self.learn and not lost:
                        agg_bs.append((self.cluster_aggregate(c, recv), len(recv)))
            if s == 'CL':
                return self.after()
            if self.learn and agg_bs:
                if s == 'EASFL' and self.robust and len(agg_bs) >= 5:
                    A = np.array([a for a, _ in agg_bs])
                    self.w = self.w + trimmed_mean(A, 0.1)
                elif self.agg != 'mean' and s != 'EASFL':
                    self.w = self.w + robust_agg(np.array([a for a, _ in agg_bs]), self.agg)
                else:
                    A = np.array([a for a, _ in agg_bs]); n = np.array([m for _, m in agg_bs], float)
                    self.w = self.w + (A * n[:, None]).sum(0) / n.sum()
            b_dn = (BITS_F16 if (s == 'EASFL' and self.compress) else BITS_F32) + HEADER
            for i in np.where(al)[0]:
                self.spend(i, rx_energy(b_dn))
        return self.after()

    def after(self):
        self.r += 1
        return True

    # ------------------------------------------------------------------ learning
    def local_update(self, i):
        X, y = self.tr[i]
        if len(y) > self.cap:
            j = self.rng.choice(len(y), self.cap, replace=False); X, y = X[j], y[j]
        if self.mal[i] and self.attack == 'labelflip':
            y = 1 - y                          # data poisoning: flipped labels
        wl = sgd(self.w, X, y, self.rng, epochs=1, lr=LR, mu=self.mu)
        u = wl - self.w
        if self.mal[i]:
            if self.attack == 'signflip':
                u = -4.0 * u                   # model poisoning: scaled sign flip
            elif self.attack == 'stealth':
                u = -1.0 * u                   # norm-preserving sign flip (evades norm filter)
            elif self.attack == 'gauss':
                g = self.rng.normal(0, 1, u.shape)
                u = g / np.linalg.norm(g) * 2.0 * np.linalg.norm(u)
        if self.s == 'EASFL' and self.compress:
            v = u + self.resid[i]
            k = np.argpartition(-np.abs(v), K_TOP)[:K_TOP]
            q = np.zeros_like(v)
            sc = np.abs(v[k]).max() / 127 + 1e-12
            q[k] = np.round(v[k] / sc) * sc
            self.resid[i] = v - q
            u = q
        return u

    def cluster_aggregate(self, c, recv):
        ups = np.array([self.local_update(i) for i in recv])
        if self.s == 'EASFL' and self.robust:
            nrm = np.linalg.norm(ups, axis=1); med = np.median(nrm) + 1e-12
            bad = nrm > 3 * med
            if self.cosf and len(ups) >= 3:
                ref = np.median(ups, 0); rn = np.linalg.norm(ref) + 1e-12
                cos = ups @ ref / (nrm * rn + 1e-12)
                bad = bad | (cos < 0)
            for i, f in zip(recv, bad):
                self.flags[i] = 0.7 * self.flags[i] + (1 if f else 0)
                if self.flags[i] > 1.5:
                    self.ban_until[i] = self.r + 20
            ups = ups[~bad] if (~bad).any() else ups
            return trimmed_mean(ups, 0.2) if len(ups) >= 5 else np.median(ups, 0)
        return robust_agg(ups, self.agg)

    def evaluate(self, personal=None, alive_only=False):
        """personal=None: global model, threshold 0.5.
        'thr': each node calibrates its own decision threshold on its local training windows.
        'ft' : each node fine-tunes the global model locally (FT_EP epochs) before inference.
        Both run on-node after training and need no extra radio communication.
        alive_only=True: a node whose battery is exhausted cannot raise alarms, so its
        windows are scored 0 (network-level detection; recall counts every anomaly)."""
        if personal is None and not alive_only:
            return metrics(self.w, self.te)
        P, Y = [], []
        rng = np.random.default_rng(7)
        for i in range(N_NODES):
            (Xt, yt), (Xe, ye) = self.tr[i], self.te[i]
            w = self.w
            if personal in ('ft', 'ft+thr'):
                w = sgd(self.w, Xt, yt, rng, epochs=FT_EP, lr=0.05)
            p = predict_proba(w, Xe)
            if personal in ('thr', 'ft+thr'):
                t = best_threshold(predict_proba(w, Xt), yt)
                p = np.clip(p - t + 0.5, 0, 1)      # shift so that 0.5 is the local threshold
            if alive_only and self.E[i] <= 0:
                p = np.zeros_like(p)
            P.append(p); Y.append(ye)
        return metrics(None, self.te, proba=np.concatenate(P))


FT_EP = 3

def krum(A, f):
    n = len(A)
    if n <= 2 * f + 2:
        return np.median(A, 0)
    D = ((A[:, None] - A[None]) ** 2).sum(2)
    sc = np.sort(D, 1)[:, 1:n - f - 1].sum(1)
    return A[np.argmin(sc)]

def robust_agg(A, rule):
    if rule == 'mean' or len(A) < 3:
        return A.mean(0)
    if rule == 'median':
        return np.median(A, 0)
    if rule == 'trimmed':
        return trimmed_mean(A, 0.2)
    if rule == 'krum':
        return krum(A, max(1, int(0.2 * len(A))))
    raise ValueError(rule)

def trimmed_mean(A, beta):
    n = len(A); k = int(np.floor(beta * n))
    S = np.sort(A, 0)
    return S[k:n - k].mean(0) if n - 2 * k > 0 else S.mean(0)
