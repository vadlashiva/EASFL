"""Prepare the Intel Lab readings for intel.py.

Usage: python prep_intel.py path/to/data.txt.gz [out.npz]
Download data.txt.gz and mote_locs.txt from https://db.csail.mit.edu/labdata/labdata.html
"""
import sys
import numpy as np
import pandas as pd

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else 'intel_clean.npz'
df = pd.read_csv(src, sep=r'\s+', header=None, on_bad_lines='skip',
                 names=['date', 'time', 'epoch', 'mote', 'temp', 'hum', 'light', 'volt']).dropna()
df = df[(df.mote >= 1) & (df.mote <= 54) & df.date.between('2004-02-28', '2004-03-09')]
ok = (df.temp.between(5, 45) & df.hum.between(0, 100) & df.light.between(0, 3000)
      & df.volt.between(2.0, 3.1))
df = df[ok].sort_values(['mote', 'epoch']).drop_duplicates(['mote', 'epoch'])
np.savez_compressed(dst, mote=df.mote.astype(int).values, epoch=df.epoch.astype(int).values,
                    X=df[['temp', 'hum', 'light', 'volt']].values.astype(np.float32))
print(f'{len(df)} readings from {df.mote.nunique()} motes -> {dst}')
