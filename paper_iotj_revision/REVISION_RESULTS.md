# EASFL revision (IoT-76474-2026) - results so far, 18 Nov 2026

Code: copy these files into the `code/` folder of EASFL_IoTJ_submission (sim.py replaces the old one;
it adds `matched=True` and the support-aware median `agg='mmedian'`; default behaviour is unchanged and
reproduces results_mask.json, results_v2.json and results_intel.json exactly).
Intel Lab runs need intel_clean.npz: `python prep_intel.py data.txt.gz` (data.txt.gz is in EASFL BUNDLE).

| Exp | Script | Output | Tables |
|---|---|---|---|
| E0 table check | summarize2.py, summarize_intel.py | - | all tables match logs; 3.564/3.561 not in PDF |
| E1 matched baselines, synthetic (980 runs) | run_matched.py | results_matched.json | tables_matched.txt |
| E1 matched baselines, Intel Lab (700 runs) | run_matched.py intel | results_matched_intel.json | tables_matched_intel.txt |
| E10 20 seeds, paired bootstrap CI, Wilcoxon + Holm | summarize_matched.py | - | bottom of tables_matched*.txt |
| E2/E3 F1 over the whole lifetime, local-only | run_lifetime_f1.py | results_lifetime_f1.json | tables_lifetime_f1.txt |
| E2b train-then-detect deployment | run_train_then_detect.py | results_ttd.json | printed by the script |

Text for the manuscript that needs no new runs: revision_snippets.tex.

Key findings
- Matched baselines (same traffic and energy for all): without attack EASFL-M equals matched FedAvg and a
  support-aware median (synthetic 0.680 vs 0.674/0.676; Intel 0.570 vs 0.569/0.570). Coordinate median and Krum
  lose 0.12-0.29 on sparse updates (zeros counted as votes).
- Mask-aware vs dense filter: Intel clean +0.038 (p < 0.001), Gaussian +0.037, sign flip +0.064 (all Holm p < 0.001);
  synthetic differences are small and not significant.
- Under 20% sign flip EASFL-M beats every matched baseline on both data sets (synthetic +0.06 to +0.43,
  Intel +0.15 to +0.35, Holm p <= 0.04). Label flip: beats mean/median/Krum; equal to trimmed mean on Intel.
  Stealth and Gaussian: equal to the best matched baseline. 30%: no method is reliable.
- Lifetime: live-node F1 of EASFL-M stays 0.83-0.84 to round 800; network F1 falls with coverage only.
- Local-only with per-round retraining outlasts all FL schemes (mean network F1 over 1000 rounds 0.63 vs 0.55);
  with FL as a 200-round training phase, EASFL-M leads at every horizon (0.76 vs 0.64 at round 1000).
- 20-seed values are lower than the paper's 5-seed values in some attack cells (e.g. synthetic sign flip
  0.64 +- 0.12, not 0.72): the revised tables must use the 20-seed numbers.

## AE-1 / AE-2 (insider attacks), added 18 Nov 2026

Script: `simulator/run_attacks2.py [intel]` -> `results_attacks2[_intel].json`; tables: `summarize_attacks2.py`
(saved as `results_attacks2[_intel]_tables.txt`); lifetime cost of the defences: `life_def.py`.

- Coordinate-targeting collusion (shared support, 2.5x honest norm): EASFL-M 0.488 at 20% (clean 0.667).
  The support-collusion test (Jaccard > 0.7) restores 0.590 (20%) and 0.575 (30%); Intel 0.536 vs <= 0.458
  for every other rule (Holm p <= 0.012). No clean cost. An adaptive colluder (half shared support) evades
  the test; then the dense filter (0.649) and trimmed mean (0.624) beat EASFL-M (0.596) on synthetic data.
- Compromised CHs (-4x aggregate) with false full-battery reports: EASFL-M 0.09 (synthetic) / 0.15 (Intel);
  malicious nodes take 80% of CH slots. Energy bound or BS screening alone: no help. Member audit alone:
  0.683 / 0.570 (clean 0.667 / 0.574), +4.7% / +1.5% energy, FND 546 -> 525. All checks + audit hold to 30%
  (0.671). Cooldown shortens FND 546 -> 324.
