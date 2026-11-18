"""Round-based WSN simulator with energy accounting and optional federated learning.

Schemes: 'CL' (centralized, raw data via LEACH clusters), 'FL-flat' (FedAvg, direct to BS),
'FL-LEACH' (FedAvg over LEACH clusters), 'EASFL' (proposed). EASFL flags allow ablations.
"""
import numpy as np
from common import *

class Sim:
    def __init__(self, scheme, seed, learn=False, data=None, samples_per_round=60,
                 p_loss=0.05, mal_frac=0.0, rho=0.5, ea_ch=None, compress=True,
                 robust=True, psel=True, local_cap=150, attack='signflip', agg='mean', cosf=True, mu=0.0, mcos=False, matched=False,
                 min_support=0, ch_attack=False, falsify=False, bs_filter=False, ch_cooldown=False, e_check=False, sup_test=False, sup_tau=0.7, bs_mode='median', audit=False):
        self.s = scheme; self.rng = np.random.default_rng(1000 + seed)
        net = np.random.default_rng(seed)               # same deployment per seed
        self.pos = net.uniform(0, FIELD, (N_NODES, 2))
        self.E0 = net.uniform(E_INIT_LO, E_INIT_HI, N_NODES)
        self.E = self.E0.copy()
        self.n = len(self.E0)
        self.dBS = np.linalg.norm(self.pos - BS, axis=1)
        D = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=2)
        self.D = D; self.deg = (D < 25).sum(1) - 1
        self.S = samples_per_round; self.p_loss = p_loss
        self.rho, self.ea_ch_flag, self.compress, self.robust, self.psel = rho, ea_ch, compress, robust, psel
        self.matched = matched   # baseline runs with EASFL's compression, participation and CH policy
        self.cmp = compress and (scheme == 'EASFL' or matched)
        self.ea_ch = (scheme == 'EASFL' or matched) if self.ea_ch_flag is None else self.ea_ch_flag
        self.last_ch = np.full(N_NODES, -10**6)
        self.r = 0
        self.mal = np.zeros(N_NODES, bool)
        if mal_frac > 0:
            self.mal[net.choice(N_NODES, int(round(mal_frac * N_NODES)), replace=False)] = True
        self.flags = np.zeros(N_NODES); self.ban_until = np.zeros(N_NODES)
        self.learn = learn; self.cap = local_cap
        self.attack, self.agg, self.cosf, self.mu = attack, agg, cosf, mu
        self.mcos = mcos   # mask-aware direction filter for sparsified updates
        # revision (AE-1/AE-2): minimum support per coordinate; compromised CHs; defences
        self.min_support = min_support
        self.sup_test, self.sup_tau = sup_test, sup_tau
        self.audit = audit; self.audit_flags = [0, 0, 0, 0]
        self.bs_mode = bs_mode; self.prev_step = None; self.ch_strikes = np.zeros(N_NODES)
        self.ch_attack, self.falsify = ch_attack, falsify
        self.bs_filter, self.ch_cooldown, self.e_check = bs_filter, ch_cooldown, e_check
        self.spent = np.zeros(N_NODES)          # energy spent, known to the protocol from node roles
        self.mal_ch_rounds = 0; self.ch_rounds = 0; self.mal_ch_count = 0; self.ch_count = 0
        self.T = net.choice(N_PARAM, K_TOP, replace=False)          # colluders' shared coordinates
        self.Tsign = np.where(net.random(K_TOP) < 0.5, -1.0, 1.0)   # colluders' shared push direction
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
        rep_E = self.E[idx].copy()
        if self.falsify:
            fake = self.mal[idx]
            rep_E[fake] = E_INIT_HI                     # claims a full battery
            if self.e_check:
                # BS bound: no node can hold more than the largest initial energy minus what its
                # known roles have cost; an adaptive liar reports exactly the bound
                bound = E_INIT_HI - self.spent[idx]
                rep_E = np.minimum(rep_E, bound)
        e = rep_E / rep_E.max()
        d = self.dBS[idx]; dn = 1 - (d - d.min()) / (d.max() - d.min() + 1e-9)
        g = self.deg[idx] / max(self.deg[idx].max(), 1)
        rep = (self.ban_until[idx] <= self.r)
        score = (0.5 * e + 0.3 * dn + 0.2 * g) * rep
        if self.ch_cooldown:
            # a node may serve as CH at most once every 1/p rounds (relaxed if too few are eligible)
            cool = (self.r - self.last_ch[idx]) >= int(round(1 / P_CH))
            if cool.sum() >= k:
                score = np.where(cool, score, -1.0)
        chosen = []
        for j in idx[np.argsort(-score)]:
            if all(self.D[j, c] >= 20 for c in chosen):
                chosen.append(j)
            if len(chosen) == k:
                break
        return np.array(chosen, int)

    def spend(self, i, e):
        self.E[i] -= e
        self.spent[i] += e

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
            self.ch_rounds += 1; self.ch_count += len(chs)
            nm = int(self.mal[chs].sum()); self.mal_ch_count += nm; self.mal_ch_rounds += int(nm > 0)
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
                if (s == 'EASFL' or self.matched) and self.psel:
                    ok = members[(self.E[members] > 0.1 * E_INIT_LO) & (self.ban_until[members] <= self.r)]
                    n_take = int(np.ceil(self.rho * len(members)))
                    parts = ok[np.argsort(-self.E[ok])][:n_take]
                else:
                    parts = members
                b_up = (BITS_TOPK8 if self.cmp else BITS_F32) + HEADER
                recv = []
                for i in parts:
                    self.spend(i, min(self.cap, 200) * E_SAMPLE_EPOCH)
                    if i != c:
                        self.spend(i, tx_energy(b_up, self.D[c, i])); self.bits_up += b_up
                        if self.rng.random() < self.p_loss:
                            continue
                        self.spend(c, rx_energy(b_up) + E_DA * b_up)
                    recv.append(i)
                b_ch = (BITS_Q8 if self.cmp else BITS_F32) + HEADER
                if len(recv):
                    self.spend(c, tx_energy(b_ch, self.dBS[c])); self.bits_up += b_ch
                    lost = self.rng.random() < self.p_loss
                    if self.learn and not lost:
                        a = self.cluster_aggregate(c, recv)
                        if self.ch_attack and self.mal[c]:
                            a = -4.0 * a                # compromised CH: scaled sign flip of its aggregate
                        if self.audit and len(recv) >= 2:
                            # member audit: one random participant (not the CH) also sends its compressed
                            # update straight to the BS; the CH aggregate must not oppose it on its support
                            cand = [i for i in recv if i != c]
                            j = cand[self.rng.integers(len(cand))]
                            self.spend(j, tx_energy(b_up, self.dBS[j])); self.bits_up += b_up
                            uj = self._last_up.get(j)
                            if uj is not None:
                                S_ = uj != 0
                                cs = (a[S_] @ uj[S_]) / (np.linalg.norm(a[S_]) * np.linalg.norm(uj[S_]) + 1e-12)
                                flagged = cs < 0
                                self.audit_flags[2 * int(self.mal[c]) + int(flagged)] += 1
                                if flagged:
                                    self.ch_strikes[c] += 1
                                    if self.ch_strikes[c] >= 2:
                                        self.ban_until[c] = self.r + 100
                                    continue            # aggregate rejected
                        agg_bs.append((a, len(recv), c))
            if s == 'CL':
                return self.after()
            if self.learn and agg_bs:
                if self.bs_filter and len(agg_bs) >= 3:
                    agg_bs = self.bs_screen(agg_bs)
                w_before = self.w.copy()
                if s == 'EASFL' and self.robust and len(agg_bs) >= 5:
                    A = np.array([a for a, _, _ in agg_bs])
                    self.w = self.w + trimmed_mean(A, 0.1)
                elif self.agg != 'mean' and s != 'EASFL':
                    self.w = self.w + robust_agg(np.array([a for a, _, _ in agg_bs]), self.agg)
                else:
                    A = np.array([a for a, _, _ in agg_bs]); n = np.array([m for _, m, _ in agg_bs], float)
                    self.w = self.w + (A * n[:, None]).sum(0) / n.sum()
                self.prev_step = self.w - w_before
            b_dn = (BITS_F16 if self.cmp else BITS_F32) + HEADER
            for i in np.where(al)[0]:
                self.spend(i, rx_energy(b_dn))
        return self.after()

    def after(self):
        self.r += 1
        return True

    def bs_screen(self, agg_bs):
        """BS-level filter on cluster aggregates (dense): norm and direction test against the
        coordinate median of the aggregates; a flagged CH accrues suspicion and is quarantined
        (excluded from CH candidacy and participation) exactly like a flagged member."""
        A = np.array([a for a, _, _ in agg_bs]); nrm = np.linalg.norm(A, axis=1)
        if self.bs_mode == 'median':
            ref = np.median(A, 0); rn = np.linalg.norm(ref) + 1e-12
            bad = (nrm > 3 * np.median(nrm) + 1e-12) | ((A @ ref) / (nrm * rn + 1e-12) < 0)
            for (_, _, c), f in zip(agg_bs, bad):
                self.flags[c] = 0.7 * self.flags[c] + (1 if f else 0)
                if self.flags[c] > 1.5:
                    self.ban_until[c] = self.r + 20
        else:
            # leave-one-out: each aggregate is compared with the median of the OTHER aggregates plus
            # the previous global step (the BS's own history); norms are compared per participant
            # sample so that small clusters are not penalised
            n = np.array([m for _, m, _ in agg_bs], float)
            bad = np.zeros(len(A), bool)
            for k in range(len(A)):
                o = np.delete(A, k, 0); ref = np.median(o, 0)
                if self.prev_step is not None:
                    ref = ref / (np.linalg.norm(ref) + 1e-12) + self.prev_step / (np.linalg.norm(self.prev_step) + 1e-12)
                cos = A[k] @ ref / (nrm[k] * np.linalg.norm(ref) + 1e-12)
                bad[k] = (cos < 0) or (nrm[k] > 3 * np.median(np.delete(nrm, k)) + 1e-12)
            for (_, _, c), f in zip(agg_bs, bad):
                if f:
                    # strikes on the CH role do not decay between terms: two strikes -> 100-round ban
                    self.ch_strikes[c] += 1
                    if self.ch_strikes[c] >= 2:
                        self.ban_until[c] = self.r + 100
        keep = [x for x, f in zip(agg_bs, bad) if not f]
        return keep if len(keep) >= 2 else agg_bs

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
        if self.mal[i] and self.attack == 'coordtarget' and self.cmp:
            # colluding sparse attack: all attackers send only the shared coordinate set T with the
            # same signs, scaled to 2.5x the norm of their own honest top-k update (passes 3x norm test)
            v = u + self.resid[i]
            k = np.argpartition(-np.abs(v), K_TOP)[:K_TOP]
            amp = 2.5 * np.linalg.norm(v[k]) / np.sqrt(K_TOP)
            q = np.zeros_like(v); q[self.T] = amp * self.Tsign
            return q
        if self.mal[i] and self.attack == 'coordtarget_adapt' and self.cmp:
            # adaptive colluder: a fresh random half of T (same signs) plus its own honest top
            # coordinates for the other half, so supports of two colluders overlap like honest ones
            v = u + self.resid[i]
            h = K_TOP // 2
            Ti = self.rng.choice(K_TOP, h, replace=False)
            own = np.argsort(-np.abs(v))
            own = own[~np.isin(own, self.T)][:K_TOP - h]
            amp = 2.5 * np.linalg.norm(v[own]) / np.sqrt(K_TOP)
            q = np.zeros_like(v); q[self.T[Ti]] = amp * self.Tsign[Ti]
            sc = np.abs(v[own]).max() / 127 + 1e-12; q[own] = np.round(v[own] / sc) * sc
            return q
        if self.cmp:
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
        self._last_up = {i: u for i, u in zip(recv, ups)}
        if self.s == 'EASFL' and self.robust:
            nrm = np.linalg.norm(ups, axis=1); med = np.median(nrm) + 1e-12
            bad = nrm > 3 * med
            if self.cosf and len(ups) >= 3:
                if self.mcos:
                    # mask-aware: reference = median over the updates that kept each coordinate;
                    # cosine of each update with the reference on that update's own support only
                    M = ups != 0
                    with np.errstate(all='ignore'):
                        ref = np.nan_to_num(np.nanmedian(np.where(M, ups, np.nan), 0))
                    if self.min_support:
                        ref = np.where(M.sum(0) >= self.min_support, ref, 0.0)
                    R = np.where(M, ref[None, :], 0.0)
                    cos = (ups * R).sum(1) / (nrm * np.linalg.norm(R, axis=1) + 1e-12)
                else:
                    ref = np.median(ups, 0); rn = np.linalg.norm(ref) + 1e-12
                    cos = ups @ ref / (nrm * rn + 1e-12)
                bad = bad | (cos < 0)
                if self.sup_test and self.mcos and len(ups) >= 2:
                    # support-collusion test: honest top-k supports overlap little (Jaccard <= 0.56
                    # observed); updates sharing almost the same support with another are flagged
                    Mf = (ups != 0).astype(float); inter = Mf @ Mf.T
                    cnt = Mf.sum(1); uni = cnt[:, None] + cnt[None, :] - inter
                    Jc = inter / np.maximum(uni, 1); np.fill_diagonal(Jc, 0)
                    bad = bad | (Jc.max(1) > self.sup_tau)
            for i, f in zip(recv, bad):
                self.flags[i] = 0.7 * self.flags[i] + (1 if f else 0)
                if self.flags[i] > 1.5:
                    self.ban_until[i] = self.r + 20
            ups = ups[~bad] if (~bad).any() else ups
            if self.mcos:
                return masked_trimmed_mean(ups, 0.2, self.min_support)
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
    if rule == 'mmedian':
        return masked_median(A)
    if rule == 'krum':
        return krum(A, max(1, int(0.2 * len(A))))
    raise ValueError(rule)

def trimmed_mean(A, beta):
    n = len(A); k = int(np.floor(beta * n))
    S = np.sort(A, 0)
    return S[k:n - k].mean(0) if n - 2 * k > 0 else S.mean(0)


def masked_trimmed_mean(A, beta, min_support=0):
    """Coordinate-wise trimmed mean over the updates that kept each coordinate
    (non-zero after top-k), rescaled by the fraction that kept it. Without outliers it
    equals the plain mean of the sparse updates; zeros from sparsification are not
    treated as votes, so honest sparse updates are not shrunk."""
    n = len(A)
    M = A != 0
    c = M.sum(0)
    S = np.sort(np.where(M, A, np.inf), 0)
    t = np.floor(beta * c).astype(int)
    t = np.where(c >= 5, t, 0)
    i = np.arange(n)[:, None]
    V = (i >= t) & (i < c - t)
    tot = np.where(V, S, 0.0).sum(0)
    k = V.sum(0)
    out = np.where(k > 0, tot / np.maximum(k, 1), 0.0) * c / n
    if min_support:
        out = np.where(c >= min_support, out, 0.0)   # a coordinate sent by too few nodes is ignored
    return out


def masked_median(A):
    """Support-aware coordinate median for sparse updates: median over the updates that
    kept each coordinate, rescaled by the fraction that kept it (no filter, no quarantine)."""
    M = A != 0
    c = M.sum(0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nan_to_num(np.nanmedian(np.where(M, A, np.nan), 0))
    return med * c / len(A)
