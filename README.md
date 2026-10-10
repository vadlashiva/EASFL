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
python make_figs.py           # regenerate the figures from results_v2.json
python run_all2.py            # re-run every experiment (multi-core; several hours)
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
