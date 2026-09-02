"""Fine sweep of the ladder ratio r, canonical untuned shots, two scales.
Also records the basin-flip rate (trials landing outside the deepest basin),
the quantity the aliasing theory (alias_analysis.py) predicts.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pickle
from mlqae import geom_ladder, canonical_shots, evaluate_schedule

NT = 20000
rows = []
for nmax in (128, 256):
    for r in [round(1.15 + 0.05 * i, 2) for i in range(18)]:  # 1.15 .. 2.00
        ladder = geom_ladder(nmax, r)
        shots = canonical_shots(len(ladder))
        res = evaluate_schedule(ladder, shots, NT, seed=31000 + nmax + int(100 * r),
                                return_extra=True)
        # basin-flip rate: |theta_hat - theta| beyond one deepest-level basin
        th = np.arcsin(res['a_true'])
        th_hat_err = np.abs(np.arcsin(np.clip(res['errors'] + 0, 0, 1)))  # not used
        basin = np.pi / (2 * (2 * res['nmax'] + 1))
        # errors are in amplitude; convert threshold per-trial: |da| ~ cos(theta)*dtheta
        flip = np.mean(res['errors'] > np.cos(th) * basin)
        rows.append((nmax, r, res['nq'], res['C95'], res['C99'], res['C95par'], flip,
                     len(ladder)))
        print(f"nmax={nmax:3d} r={r:4.2f} levels={len(ladder):2d} q={res['nq']:5d} "
              f"C95={res['C95']:5.2f} C99={res['C99']:6.2f} "
              f"C95par={res['C95par']:5.3f} flip={100*flip:5.2f}%", flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ratio_sweep.pkl'), 'wb') as h:
    pickle.dump(rows, h)
