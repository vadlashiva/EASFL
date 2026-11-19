# EASFL: Energy-Aware, Poisoning-Resilient Federated Learning for WSN Anomaly Detection

Simulator, experiment scripts, seeds and raw results for the paper:

> Shiva Kumar V and Choupiri Shivakeshi, "EASFL: Energy-Aware, Poisoning-Resilient Federated Learning for Anomaly Detection and Lifetime Extension in Wireless Sensor Networks," submitted to *IEEE Transactions on Green Communications and Networking*.

Department of Computer Science and Engineering, Kishkinda University, Ballari, Karnataka, India.
Corresponding author: shivakumar@kishkindauniversity.edu.in

## Repository layout

```
simulator/
  common.py        network/energy constants, synthetic sensing workload and fault model, 24-16-1 MLP detector
  sim.py           round-based WSN + FL simulator (EASFL, FedAvg-flat, FedAvg-LEACH, centralized, local-only;
                   attacks, robust-aggregation baselines, personalization)
  run_quick.py     lifetime runs (FND / HND / LND) for one scheme and seed
  run_all.py       original experiment suite  -> results.json
  run_all2.py      extended experiment suite  -> results_v2.json (used in the paper)
  summarize.py     tables from results.json
  summarize2.py    tables from results_v2.json (tables_v2.txt is its saved output)
  make_figs.py     paper figures -> figs/*.pdf, figs/*.png
  results*.json    raw results of the runs reported in the paper
paper/
  main.tex, main.pdf          manuscript for IEEE Trans. Green Commun. Netw. (single-column, double-spaced)
  main_twocolumn.tex          two-column IEEE layout of the same manuscript
  cover_letter.tex/.pdf       cover letter; refs.bib, figs/ shared
```

## Requirements

Python 3.10+ with NumPy and Matplotlib (tested with Python 3.13, NumPy 2.5, Matplotlib 3.11):

```
pip install -r requirements.txt
```

## Reproducing the results

All commands run from `simulator/`.

```
cd simulator
python summarize2.py          # print the paper's tables from the shipped results_v2.json
python make_figs.py           # regenerate the figures from results_v2_20.json (20 seeds; 10 for poisoning)
python run_all2.py            # re-run every experiment (multi-core; several hours)
python run_figs20.py          # 20-seed learning/lifetime runs for the figures (revision)
python run_poison10.py        # 10-seed poisoning runs incl. EASFL-M (revision)
python run_all2.py main life  # or run selected stages only
```

Stages of `run_all2.py`: `main` (scheme comparison and personalization), `life` (lifetime and ablations),
`gen` (sensitivity to non-IID degree, fault magnitude and anomaly rate), `attack` (four poisoning attacks vs.
median, trimmed mean and Krum), `loss` (packet loss) and `rate` (sampling rate).
Results from completed stages are merged into `results_v2.json`.

## Seeds and setup

Seeds 1-5 (`SEEDS` in `run_all2.py`); 100 nodes, 200 learning rounds; every scheme uses the same detector
and the same synthetic dataset for a given seed. See Section III of the paper for the full model.

## Citation

Please cite the paper above if you use this code. A BibTeX entry will be added on publication.

## IEEE IoT-J revision (IoT-76474-2026), branch `iotj-revision`

`paper_iotj_revision/` holds the revised manuscript: `main_revised.tex` marks every change in blue
(`\rev{...}`), `main_clean.tex` builds the same text without colour, and `main_submitted.tex` is the
version submitted on 14 Oct 2026. `apply_revision.py` regenerates `main_revised.tex` from the submitted
source. Build: `pdflatex main_revised && bibtex main_revised && pdflatex main_revised && pdflatex main_revised`.

New experiments (run from `simulator/`; Intel Lab runs need `python prep_intel.py data.txt.gz` first):

| Experiment | Command | Output |
|---|---|---|
| Matched baselines, synthetic (20 seeds) | `python run_matched.py` | results_matched.json, tables_matched.txt |
| Matched baselines, Intel Lab (20 seeds) | `python run_matched.py intel` | results_matched_intel.json, tables_matched_intel.txt |
| Paired statistics (bootstrap CI, Wilcoxon + Holm) | `python summarize_matched.py [file]` | printed |
| F1 over the whole lifetime, local-only | `python run_lifetime_f1.py` | results_lifetime_f1.json, tables_lifetime_f1.txt |
| Train-then-detect deployment | `python run_train_then_detect.py` | results_ttd.json |

`sim.py` adds `matched=True` (a baseline with EASFL's compression, participation and CH policy) and the
support-aware median (`agg='mmedian'`); with these options off it reproduces every earlier result exactly.
See `paper_iotj_revision/REVISION_RESULTS.md` for the findings.
