"""Open item A: find a shot-allocation rule whose C95 stays flat with scale.

Sweeps the base of the linear-decay allocation (deepest level gets `base`
shots, one more per shallower level) across scales, for r = 1.40 and 1.45.
Hypothesis: optimal base grows ~logarithmically with nmax (union bound over
the growing number of branch decisions).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from mlqae import geom_ladder, canonical_shots, evaluate_schedule

NT = 20000
SCALES = [60, 125, 260, 540, 1150, 2400]

rows = []
for r in (1.40, 1.45):
    print(f'--- ratio {r} ---')
    for nmax in SCALES:
        ladder = geom_ladder(nmax, r)
        best = None
        for base in (1, 2, 3, 4, 5):
            shots = canonical_shots(len(ladder), base=base)
            res = evaluate_schedule(ladder, shots, NT, seed=71000 + nmax + base,
                                    chunk=1500)
            rows.append((r, nmax, base, res['nq'], res['C95'], res['C99']))
            tag = f"r={r} nmax={nmax:4d} base={base}: q={res['nq']:6d} " \
                  f"C95={res['C95']:5.2f} C99={res['C99']:6.2f}"
            print(tag, flush=True)
            if best is None or res['C95'] < best[0]:
                best = (res['C95'], base)
        print(f'  -> optimal base at nmax={nmax}: {best[1]} (C95={best[0]:.2f})', flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scale_rule.pkl'), 'wb') as h:
    pickle.dump(rows, h)
