"""Paired comparison of the published csAE pipeline vs global-ML readout on
the SAME schedule and the SAME simulated measurement records.

Replays the RNG sequence of run_ae_sims.py (seed 8) so the csAE column
reproduces the r=3 row of sims/csae_C4.000_mc0500.pkl trial-for-trial, then
scores three estimators per trial:
  1. published csAE (sign heuristic + ESPRIT + correction)
  2. csAE + ML polish (1-D likelihood refinement around the csAE estimate)
  3. global ML (exact likelihood over all basins; no signs, no ESPRIT)

Run from the repository root: python mlqae/compare_to_csae.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import multiprocessing
from scipy.optimize import minimize_scalar
from csae.signals import TwoqULASignal
from csae.frequencyestimator import ESPIRIT
from csae.estimator import get_heavy_signs, csae_with_local_minimization
from csae.util import simulate_signal

R = 3


def nll(theta, depths, ks, Ns):
    p = np.clip(np.cos((2 * depths + 1) * theta) ** 2, 1e-12, 1 - 1e-12)
    return -np.sum(ks * np.log(p) + (Ns - ks) * np.log1p(-p))


def refine(t, s, depths, ks, Ns):
    lo, hi = max(t - s / 2, 1e-9), min(t + s / 2, np.pi / 2 - 1e-9)
    r = minimize_scalar(nll, bounds=(lo, hi), args=(depths, ks, Ns),
                        method='bounded', options={'xatol': 1e-10})
    return r.x, r.fun


def ml_polish(theta0, depths, ks, Ns):
    s = np.pi / (2 * (2 * depths[-1] + 1))
    best_t, best_v = theta0, nll(theta0, depths, ks, Ns)
    for j in range(-4, 5):
        t = theta0 + j * s
        if 1e-9 < t < np.pi / 2 - 1e-9:
            x, v = refine(t, s, depths, ks, Ns)
            if v < best_v:
                best_t, best_v = x, v
    return best_t


def global_ml(depths, ks, Ns):
    s = np.pi / (2 * (2 * depths[-1] + 1))
    grid = np.arange(1e-6, np.pi / 2, s / 4)
    p = np.clip(np.cos(np.outer(grid, 2 * depths + 1)) ** 2, 1e-12, 1 - 1e-12)
    vals = -(np.log(p) @ ks + np.log1p(-p) @ (Ns - ks))
    order = np.argsort(vals)
    picked, best_t, best_v = [], None, np.inf
    for idx in order:
        t = grid[idx]
        if any(abs(t - q) < s for q in picked):
            continue
        picked.append(t)
        x, v = refine(t, s, depths, ks, Ns)
        if v < best_v:
            best_t, best_v = x, v
        if len(picked) >= 5:
            break
    return best_t


def trial(theta, ula, heavy_signs, seed):
    np.random.seed(seed)
    esp = ESPIRIT()
    _, meas = simulate_signal(ula.depths, ula.n_samples, theta)
    ula.set_measurements(meas)
    res = csae_with_local_minimization(ula, esp, heavy_signs, sample=True,
                                       correction=True, optimize=True, adjacency=5)
    t0 = res['theta_est']
    depths = np.array(ula.depths)
    Ns = np.array(ula.n_samples, dtype=float)
    ks = np.rint(Ns * np.array(meas))
    a = np.sin(theta)
    return (abs(a - np.sin(t0)),
            abs(a - np.sin(ml_polish(t0, depths, ks, Ns))),
            abs(a - np.sin(global_ml(depths, ks, Ns))))


if __name__ == '__main__':
    np.random.seed(8)
    avals = [np.random.uniform(0.1, 0.9) for _ in range(500)]
    thetas = np.arcsin(np.array(avals))
    hs = None
    for r in range(R + 1):
        narray = [2] * (2 * r + 2)
        u = TwoqULASignal(M=narray, C=4)
        h = get_heavy_signs(u.depths, u.n_samples, len(narray) ** 2)
        if r == R:
            hs = h
    ula = TwoqULASignal(M=[2] * (2 * R + 2), C=4)
    with multiprocessing.Pool(7) as pool:
        jobs = [pool.apply_async(trial, args=(t, ula, hs, 8 + i + 1))
                for i, t in enumerate(thetas)]
        out = np.array([j.get() for j in jobs])
    nq = int(np.sum(np.array(ula.depths) * np.array(ula.n_samples)) + ula.n_samples[0])
    for i, name in enumerate(('csAE (published)', 'csAE + ML polish', 'global ML')):
        e = out[:, i]
        q = np.percentile(e, [50, 95, 99])
        print(f'{name:18s} p50={q[0]:.3e} p95={q[1]:.3e} p99={q[2]:.3e} '
              f'max={e.max():.3e}  C95={q[1]*nq:.2f}  C99={q[2]*nq:.2f}')
