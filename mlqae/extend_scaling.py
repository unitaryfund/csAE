"""Extend the Fig. 3 head-to-head down to eps ~ 1e-6.

Adds four deeper flagship points (nmax up to ~2.9e5) and matched chebAE runs
at the achieved eps95 targets. Results are merged with head_to_head.pkl into
head_to_head_ext.pkl for the figure.

These points use the same mlqae.evaluate_schedule as every other number in the
paper; the small `chunk` keeps the (chunk x grid) intermediate in memory at
nmax ~ 2e5, where the likelihood grid holds ~3.4e6 points.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import time
from mlqae import (geom_ladder, canonical_shots, flagship_schedule,
                   evaluate_schedule, boot_ci_constant)
from head_to_head import run_chebae, cheb_constants

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = [3500, 16000, 69000, 290000]
NTRIALS = {3500: 20000, 16000: 20000, 69000: 20000, 290000: 10000}


if __name__ == '__main__':
    with open(os.path.join(HERE, 'head_to_head.pkl'), 'rb') as h:
        base = pickle.load(h)
    ext = {'ours': list(base['ours']), 'cheb95': list(base['cheb95'])}

    print('=== deeper flagship points ===')
    for nmax in SCALES:
        ladder, shots = flagship_schedule(nmax)
        t0 = time.time()
        r = evaluate_schedule(ladder, shots, NTRIALS[nmax], seed=440000 + nmax,
                              return_errors=True, chunk=40)
        ci68 = boot_ci_constant(r['errors'], r['nq'], 68)
        ci95 = boot_ci_constant(r['errors'], r['nq'], 95)
        ci99 = boot_ci_constant(r['errors'], r['nq'], 99)
        ext['ours'].append((nmax, ladder, shots, r, ci95, ci99, ci68))
        print(f"nmax={r['nmax']:6d} q={r['nq']:8d} eps95={r['p95']:.2e}  "
              f"C95={r['C95']:.2f} [{ci95[0]:.2f},{ci95[1]:.2f}]  "
              f"C99={r['C99']:.2f} [{ci99[0]:.2f},{ci99[1]:.2f}]  "
              f"C95par={r['C95par']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print('=== chebAE at the new targets (delta=0.05, 2000 trials) ===')
    for (nmax, ladder, shots, r, _, _, _) in ext['ours'][-len(SCALES):]:
        eps = r['p95']
        err, q, md = run_chebae(eps, 0.05, 2000, seed0=333000 + (nmax % 99991))
        (cav, cmx, cpar), (ciav, cimx, cipar) = cheb_constants(err, q, md, 95)
        ext['cheb95'].append((nmax, eps, err, q, md))
        print(f"target eps={eps:.2e}: achieved eps95={np.percentile(err,95):.2e}  "
              f"C_ave={cav:.2f} [{ciav[0]:.2f},{ciav[1]:.2f}]  "
              f"C_max={cmx:.2f} [{cimx[0]:.2f},{cimx[1]:.2f}]  C_par={cpar:.2f}",
              flush=True)

    with open(os.path.join(HERE, 'head_to_head_ext.pkl'), 'wb') as h:
        pickle.dump(ext, h)
    print('saved head_to_head_ext.pkl')
