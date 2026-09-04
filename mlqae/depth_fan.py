"""Data for the depth-limited 'fan' figure: eps95 vs N_tot for fixed depth
caps M, each swept over uniform-scaling factors s. Each fixed-M family
follows eps ~ sqrt(Ctilde/(M N)) (slope -1/2 on log-log axes), departing the
Heisenberg envelope eps ~ C95/N at its endpoint; growing M with budget as
M ~ N^beta selects any intermediate power N^-(1+beta)/2.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from mlqae import geom_ladder, canonical_shots, evaluate_schedule

NT = 20000
rows = []
print(f'{"M":>5s} {"s":>4s} {"Ntot":>8s} {"eps95":>9s} {"Ctilde95":>9s}')
for M in (16, 64, 256, 1024):
    ladder = geom_ladder(M, 1.45)
    if ladder[-1] != M:
        ladder.append(M)
    base = canonical_shots(len(ladder))
    for s in (1, 4, 16, 64, 256):
        shots = [b * s for b in base]
        r = evaluate_schedule(ladder, shots, NT, seed=120000 + 3 * M + s, chunk=2500)
        rows.append((M, s, r['nq'], r['p95'], r['p99']))
        print(f'{M:5d} {s:4d} {r["nq"]:8d} {r["p95"]:9.2e} '
              f'{r["p95"]**2 * M * r["nq"]:9.3f}', flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth_fan.pkl'), 'wb') as h:
    pickle.dump(rows, h)
