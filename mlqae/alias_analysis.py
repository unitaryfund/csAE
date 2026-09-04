"""Aliasing theory for depth ladders: a computable Chernoff bound that
explains the measured optimal ratio.

For true angle theta and a rival angle theta + d, the probability that the
exact likelihood prefers the rival is bounded by the Chernoff/Bhattacharyya
bound

    P[flip to theta+d] <= prod_j BC_j(theta, d)^{N_j},
    BC_j = sqrt(p_j p'_j) + sqrt((1-p_j)(1-p'_j)),

with p_j = cos^2((2n_j+1)theta), p'_j = cos^2((2n_j+1)(theta+d)). Define the
exponent E(theta, d) = -sum_j N_j ln BC_j. Large E everywhere outside the
central basin = no dangerous aliases. A union bound over rival basins,
averaged over theta, predicts the basin-flip rate measured in ratio_sweep.py.

Mechanism: for a power-of-two ladder the fringe patterns of different levels
re-align at specific offsets d (all levels' (2n+1)d near multiples of pi
simultaneously), carving deep notches in E; a ratio ~1.45 ladder has
incommensurate fringe periods, so E has a much higher floor at equal cost.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from mlqae import geom_ladder, canonical_shots

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alias_analysis.pkl')


def exponent_profile(depths, shots, thetas, ds):
    """E(theta, d) matrix, shape (len(thetas), len(ds))."""
    depths = np.asarray(depths, float)
    shots = np.asarray(shots, float)
    d1 = 2 * depths + 1
    p = np.cos(np.outer(thetas, d1)) ** 2                      # (T, D)
    E = np.zeros((len(thetas), len(ds)))
    for k, d in enumerate(ds):
        q = np.cos(np.outer(thetas + d, d1)) ** 2
        bc = np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))
        bc = np.clip(bc, 1e-300, 1.0)
        E[:, k] = -np.sum(shots * np.log(bc), axis=1)
    return E


def predicted_flip_rate(depths, shots, ntheta=400):
    """Union bound over rival basins, averaged over theta in (arcsin .1, arcsin .9)."""
    depths = np.asarray(depths, float)
    nmax = depths.max()
    basin = np.pi / (2 * (2 * nmax + 1))
    thetas = np.linspace(np.arcsin(0.1), np.arcsin(0.9), ntheta)
    # rival offsets: one point per basin, both sides, out to +-0.35 rad
    ds = np.concatenate([np.arange(basin, 0.35, basin),
                         -np.arange(basin, 0.35, basin)])
    E = exponent_profile(depths, shots, thetas, ds)
    pflip = np.minimum(1.0, np.sum(np.exp(-E), axis=1))
    return float(np.mean(pflip)), E, ds, thetas


def split_flip_rates(depths, shots, far_basins=10, ntheta=400):
    """Union-bound flip predictions split into NEAR rivals (within far_basins
    deepest-level basins; benign, small error) and FAR rivals (catastrophic)."""
    depths = np.asarray(depths, float)
    nmax = depths.max()
    basin = np.pi / (2 * (2 * nmax + 1))
    thetas = np.linspace(np.arcsin(0.1), np.arcsin(0.9), ntheta)
    ds = np.concatenate([np.arange(basin, 0.35, basin),
                         -np.arange(basin, 0.35, basin)])
    E = exponent_profile(depths, shots, thetas, ds)
    far = np.abs(ds) >= far_basins * basin
    p_near = np.minimum(1.0, np.sum(np.exp(-E[:, ~far]), axis=1))
    p_far = np.minimum(1.0, np.sum(np.exp(-E[:, far]), axis=1))
    return (float(np.mean(p_near)), float(np.mean(p_far)),
            float(E[:, far].min()), E, ds)


def measured_flip_rates(ladder, shots, far_basins=10, ntrials=20000, seed=606):
    from mlqae import evaluate_schedule
    res = evaluate_schedule(ladder, shots, ntrials, seed=seed, return_extra=True)
    th = np.arcsin(res['a_true'])
    basin = np.pi / (2 * (2 * res['nmax'] + 1))
    any_flip = np.mean(res['errors'] > np.cos(th) * basin)
    far_flip = np.mean(res['errors'] > np.cos(th) * far_basins * basin)
    return float(any_flip), float(far_flip)


if __name__ == '__main__':
    NMAX = 128
    FAR = 10
    configs = {}
    for name, r in (('geom1.25', 1.25), ('geom1.45', 1.45), ('geom1.8', 1.8), ('pow2', 2.0)):
        ladder = geom_ladder(NMAX, r)
        shots = canonical_shots(len(ladder))
        configs[name] = (ladder, shots)

    print(f'{"ladder":10s} {"q":>6s} {"minE(far)":>9s} {"pred near":>10s} {"pred far":>10s} '
          f'{"meas any":>9s} {"meas far":>9s}')
    summary = {}
    for name, (ladder, shots) in configs.items():
        nq = int(np.sum(np.asarray(ladder) * np.asarray(shots)) + shots[0])
        pn, pf, minEf, E, ds = split_flip_rates(ladder, shots, FAR)
        ma, mf = measured_flip_rates(ladder, shots, FAR)
        summary[name] = dict(ladder=ladder, shots=shots, nq=nq, pred_near=pn,
                             pred_far=pf, minE_far=minEf, meas_any=ma, meas_far=mf,
                             E_min_over_theta=E.min(axis=0), ds=ds)
        print(f'{name:10s} {nq:6d} {minEf:9.2f} {pn:10.2e} {pf:10.2e} '
              f'{ma:9.2%} {mf:9.2%}')

    with open(OUT, 'wb') as h:
        pickle.dump(summary, h)
