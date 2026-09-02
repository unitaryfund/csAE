"""Appendix A: mode width, grid resolution, and the bracketing certification.

Produces every number quoted in Appendix A of the paper:

  Part 1 (Table V, cols 2-3)  predicted basin/mode-width ratio pi sqrt(Lambda)
                              against the median measured from the curvature of
                              ell at its peak. Eq. (A3) is an identity, so this
                              is the only empirical check of the claim that a
                              likelihood mode is really sigma_theta wide.
  Part 2 (Table V, cols 4-5)  eps95 under the two grid rules, relative to a
                              reference search at Delta = w/512, on identical
                              measurement records.
  Part 3 (Sec. A.3 prose)     basin-level disagreement rate against a 32x finer
                              coarse search, and the log-likelihood gap on the
                              trials that disagree.

Run from the repository root, with the venv active:

    source .venv/bin/activate && python research/grid_resolution.py

Runtime ~50 s on a ~10-core laptop.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pickle
from mlqae import geom_ladder, canonical_shots, grid_spacing, evaluate_schedule

HERE = os.path.dirname(os.path.abspath(__file__))

NMAX, RATIO = 125, 1.45          # the ladder Table V is built on
SCALES = (1, 16, 64, 256, 1024)
NT_WIDTH = 4000                  # trials for the curvature measurement
NT_GRID = 20000                  # paired trials for the resolution comparison
NT_BRACKET = 40000               # trials for the bracketing certification


def ladder_and_shots(nmax, r, s):
    lad = geom_ladder(nmax, r)
    return lad, [s * b for b in canonical_shots(len(lad))]


def basin_width(nmax):
    return np.pi / (2 * (2 * nmax + 1))


def log_lik(theta, K, shots, d):
    """Exact log-likelihood, Eq. (3), evaluated per trial at one theta each."""
    p = np.clip(np.cos(theta[:, None] * d[None, :]) ** 2, 1e-12, 1 - 1e-12)
    return np.sum(K * np.log(p) + (shots[None, :] - K) * np.log1p(-p), axis=1)


# ------------------------------------------------- Part 1: mode width (cols 2-3)
def measure_mode_width(nmax, r, s, ntrials, seed):
    """Median (and inter-decile) basin/mode-width from the curvature of ell.

    The mode width is (-ell'')^{-1/2} at the peak; we evaluate the curvature at
    the true theta, which is the mode centre in expectation. Compare against
    pi sqrt(Lambda) = basin / sigma_theta, Eq. (A3).
    """
    lad, shots = ladder_and_shots(nmax, r, s)
    shots = np.asarray(shots, int)
    d = 2 * np.asarray(lad, float) + 1
    w = basin_width(nmax)
    sigma = 1.0 / np.sqrt(4 * np.sum(shots * d ** 2))
    lam = np.sum(shots * (d / d[-1]) ** 2)

    rng = np.random.default_rng(seed)
    a = rng.uniform(0.1, 0.9, ntrials)
    th = np.arcsin(a)
    K = rng.binomial(shots[None, :], np.cos(np.outer(th, d)) ** 2)

    h = sigma / 4
    curv = -(log_lik(th + h, K, shots, d) - 2 * log_lik(th, K, shots, d)
             + log_lik(th - h, K, shots, d)) / h ** 2
    ratio = w / (1.0 / np.sqrt(curv[curv > 0]))
    return dict(nmax=nmax, s=s, predicted=float(np.pi * np.sqrt(lam)),
                measured=float(np.median(ratio)),
                d10=float(np.percentile(ratio, 10)),
                d90=float(np.percentile(ratio, 90)),
                usable=float(np.mean(curv > 0)))


# -------------------------------------------- Part 2: grid resolution (cols 4-5)
def compare_grids(nmax, r, s, ntrials, seed):
    """eps95 under Delta = w/8 and under Eq. (A5), vs a Delta = w/512 reference.

    All three use identical measurement records (same seed), so the comparison
    is paired and the differences are pure estimator effects.
    """
    lad, shots = ladder_and_shots(nmax, r, s)
    w = basin_width(nmax)
    rule = grid_spacing(lad, shots)
    d = 2 * np.asarray(lad, float) + 1
    sigma = 1.0 / np.sqrt(4 * np.sum(np.asarray(shots, float) * d ** 2))
    out = dict(nmax=nmax, s=s, w_over_sigma=float(w / sigma))
    eps = {}
    for name, step in (('w/8', w / 8), ('rule', rule), ('ref', w / 512)):
        res = evaluate_schedule(lad, shots, ntrials, seed=seed,
                                grid_step=step, chunk=250)
        eps[name] = res['p95']
    out.update(eps95_w8=eps['w/8'], eps95_rule=eps['rule'], eps95_ref=eps['ref'],
               rel_w8=eps['w/8'] / eps['ref'] - 1,
               rel_rule=eps['rule'] / eps['ref'] - 1,
               rule_binds='4 sigma' if rule < w / 8 else 'w/8')
    return out


# ------------------------------------- Part 3: bracketing certification (prose)
def certify_bracketing(nmax, r, s, ntrials, seed):
    """Compare the shipped coarse grid with a 32x finer one on identical records.

    Reports how often the two land in different deepest-level basins, and -- on
    exactly those trials -- the exact log-likelihood gap between the two
    answers. Small gaps mean the coarse grid arbitrated a statistical near-tie
    rather than missing a clear maximum.
    """
    lad, shots = ladder_and_shots(nmax, r, s)
    shots_a = np.asarray(shots, int)
    d = 2 * np.asarray(lad, float) + 1
    w = basin_width(nmax)
    rule = grid_spacing(lad, shots)

    kw = dict(seed=seed, return_extra=True, chunk=250)
    coarse = evaluate_schedule(lad, shots, ntrials, grid_step=rule, **kw)
    fine = evaluate_schedule(lad, shots, ntrials, grid_step=rule / 32, **kw)

    disagree = np.abs(coarse['theta_hat'] - fine['theta_hat']) > w / 2
    K = coarse['counts'][disagree]
    gap = (log_lik(fine['theta_hat'][disagree], K, shots_a, d)
           - log_lik(coarse['theta_hat'][disagree], K, shots_a, d))
    return dict(nmax=nmax, r=r, s=s, ntrials=ntrials,
                n_disagree=int(disagree.sum()),
                rate=float(disagree.mean()),
                max_gap=float(gap.max()) if disagree.any() else 0.0,
                median_gap=float(np.median(gap)) if disagree.any() else 0.0,
                eps95_coarse=coarse['p95'], eps95_fine=fine['p95'])


if __name__ == '__main__':
    results = {}

    print('Part 1 -- mode width (Table V, cols 2-3)')
    print(f'{"s":>6} {"pi sqrt(Lambda)":>16} {"measured":>10} {"[d10, d90]":>18}')
    results['mode_width'] = []
    for s in SCALES:
        row = measure_mode_width(NMAX, RATIO, s, NT_WIDTH, seed=99)
        results['mode_width'].append(row)
        print(f'{s:6d} {row["predicted"]:16.2f} {row["measured"]:10.2f}'
              f'   [{row["d10"]:6.2f}, {row["d90"]:6.2f}]', flush=True)

    print('\nPart 2 -- grid resolution (Table V, cols 4-5)')
    print(f'{"s":>6} {"binds":>8} {"eps95 (ref)":>12} {"Delta=w/8":>11} {"Eq. (A5)":>11}')
    results['grid'] = []
    for s in SCALES:
        row = compare_grids(NMAX, RATIO, s, NT_GRID, seed=20240)
        results['grid'].append(row)
        print(f'{s:6d} {row["rule_binds"]:>8} {row["eps95_ref"]:12.4e} '
              f'{100*row["rel_w8"]:+10.2f}% {100*row["rel_rule"]:+10.2f}%', flush=True)

    print('\nPart 3 -- bracketing vs a 32x finer coarse search (Sec. A.3)')
    print(f'{"ladder":>16} {"trials":>8} {"disagree":>9} {"rate":>9} '
          f'{"max gap":>9} {"eps95 coarse/fine":>22}')
    results['bracketing'] = []
    for nmax, r in ((125, 1.45), (803, 1.45), (125, 2.0)):
        row = certify_bracketing(nmax, r, 1, NT_BRACKET, seed=7)
        results['bracketing'].append(row)
        print(f'{f"nmax={nmax}, r={r}":>16} {row["ntrials"]:8d} '
              f'{row["n_disagree"]:9d} {row["rate"]:9.1e} {row["max_gap"]:9.3f} '
              f'   {row["eps95_coarse"]:.5e} / {row["eps95_fine"]:.5e}', flush=True)

    with open(os.path.join(HERE, 'grid_resolution.pkl'), 'wb') as h:
        pickle.dump(results, h)
    print('\nwrote grid_resolution.pkl')
