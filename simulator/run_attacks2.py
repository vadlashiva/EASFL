"""Revision experiments AE-1 and AE-2 (synthetic workload, seeds 1-10; Intel Lab with 'intel').

AE-1  coordinate-targeting collusion: attackers send only a shared coordinate set T with shared
      signs ('coordtarget'), or an adaptive variant that mixes a random half of T with its own
      top coordinates to evade the support test ('coordtarget_adapt'). Defences compared at matched
      cost: matched FedAvg/trimmed mean/support-aware median, EASFL (dense), EASFL-M, and EASFL-M with
      the support-collusion test (sup_test).
AE-2  compromised cluster heads: a malicious node elected CH sends -4x its cluster aggregate
      (ch_attack), optionally after reporting a full battery to win CH election (falsify), optionally
      while also sign-flipping its own updates as a member. Defences: BS-level screening of cluster
      aggregates with quarantine (bs_filter), CH cooldown of 1/p rounds (ch_cooldown), and an energy
      bound check on reported energy (e_check).
Usage: python run_attacks2.py [intel] -> results_attacks2[_intel].json (resumable)
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool

INTEL = 'intel' in sys.argv[1:]
OUT = 'results_attacks2_intel.json' if INTEL else 'results_attacks2.json'
SEEDS = range(1, 11)
M = dict(mcos=True)
AE1_DEF = [('FL-LEACH', dict(matched=True)), ('FL-LEACH', dict(matched=True, agg='trimmed')),
           ('FL-LEACH', dict(matched=True, agg='mmedian')), ('EASFL', {}), ('EASFL', M),
           ('EASFL', dict(M, sup_test=True))]
D_ALL = dict(bs_filter=True, ch_cooldown=True, e_check=True)
AE2_DEF = [('none', {}), ('bs', dict(bs_filter=True)), ('cool', dict(ch_cooldown=True)),
           ('echeck', dict(e_check=True)), ('bs+cool', dict(bs_filter=True, ch_cooldown=True)),
           ('all', D_ALL), ('audit', dict(audit=True)), ('all+audit', dict(D_ALL, audit=True))]
AE2_SCEN = [('ch', dict(attack='none', ch_attack=True)),
            ('ch+falsify', dict(attack='none', ch_attack=True, falsify=True)),
            ('ch+falsify+member', dict(attack='signflip', ch_attack=True, falsify=True))]


def jobs():
    J = []
    fr = (0.2,) if INTEL else (0.1, 0.2, 0.3)
    for sd in SEEDS:
        # AE-1
        J.append(('EASFL', sd, dict(M, sup_test=True)))                       # clean cost of the test
        for att in ('coordtarget', 'coordtarget_adapt'):
            for m in fr:
                for sc, kw in AE1_DEF:
                    J.append((sc, sd, dict(kw, mal_frac=m, attack=att)))
        # AE-2
        J.append(('EASFL', sd, dict(M, **D_ALL)))                              # clean cost of defences
        J.append(('EASFL', sd, dict(M, **D_ALL, audit=True)))
        for scn, skw in AE2_SCEN:
            J.append(('FL-LEACH', sd, dict(matched=True, mal_frac=0.2, **skw)))
            for dn, dkw in AE2_DEF:
                if scn != 'ch+falsify' and dn not in ('none', 'all', 'all+audit'):
                    continue
                J.append(('EASFL', sd, dict(M, mal_frac=0.2, **skw, **dkw)))
            if not INTEL and scn == 'ch+falsify':
                for m in (0.1, 0.3):
                    for dn, dkw in (AE2_DEF[0], AE2_DEF[5], AE2_DEF[-1]):
                        J.append(('EASFL', sd, dict(M, mal_frac=m, **skw, **dkw)))
    return J


def job(a):
    warnings.filterwarnings('ignore')
    import numpy as np
    if INTEL:
        import run_intel as R
        from intel import IntelSim as S
        data = R.data(a[1])
    else:
        from sim import Sim as S
        from common import build_dataset
        data = build_dataset(a[1])
    s = S(a[0], a[1], learn=True, data=data, **a[2])
    for _ in range(200):
        s.step()
    fin = {'global': s.evaluate(), 'ft+thr': s.evaluate('ft+thr')}
    return dict(scheme=a[0], seed=a[1], kw=a[2], final=fin, mbit=s.bits_up / 1e6,
                energy=float(s.E0.sum() - np.clip(s.E, 0, None).sum()), alive=int((s.E > 0).sum()),
                mal_ch_share=s.mal_ch_count / max(s.ch_count, 1),
                mal_ch_rounds=s.mal_ch_rounds / max(s.ch_rounds, 1),
                quarantined_mal=int(((s.ban_until > s.r) & s.mal).sum()),
                quarantined_honest=int(((s.ban_until > s.r) & ~s.mal).sum()),
                audit_flags=list(s.audit_flags))


def tag(a):
    return json.dumps([a[0], a[1], a[2]], sort_keys=True)


if __name__ == '__main__':
    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {tag((r['scheme'], r['seed'], r['kw'])) for r in out}
    todo = [j for j in jobs() if tag(j) not in done]
    print(len(todo), 'jobs', flush=True)
    t = time.time()
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(job, todo, chunksize=2)):
            out.append(r)
            if k % 10 == 0 or k == len(todo) - 1:
                json.dump(out, open(OUT, 'w'), default=float)
                print(k + 1, '/', len(todo), round(time.time() - t), 's', flush=True)
