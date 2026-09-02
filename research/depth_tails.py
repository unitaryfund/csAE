"""Tail behavior of anchored vs uniformly scaled depth-limited schedules.

Backs the quantitative claims of Sec. V and Sec. VI.B about the depth-limited
alternatives (companion to depth_tradeoff.py, which measures the percentile
constants themselves):

  block A (stage-1 stall):   single cap anchor, M=64, anchors up to 2^20.
                             eps95*M plateaus at ~0.16 while Ctilde99 grows
                             without bound with the anchor (mirror locking).
  block B (pair, no stall):  {0.8M, M} anchor, M=64, anchors 2^12..2^20.
                             eps95/CRLB ~ 1.0 throughout, Ctilde95 -> ~0.19:
                             the 95th percentile never stalls.
  block C (pair, thin tail): the same design at 1e6 trials (M=64, a4096) and
                             1e5 trials (M=256, a4096/a65536): wrong-basin
                             rate 1e-5..1e-4 with errors ~200x eps95;
                             Ctilde99.9 from ~0.9 up to ~90 (M=256, a4096).
                             The danger-zone measure shrinks like the anchor
                             fringe width, i.e. polynomially in the budget.
  block D (uniform, clean):  uniform scaling s in {4, 16} at M in {64, 256},
                             2e5 trials: no far flips resolved, Ctilde99.9 =
                             1.9-4.2 and falling with s (exponential
                             suppression of every failure mode).

A "far flip" is a trial whose theta error exceeds two deepest-level basin
widths 2 * pi/(2(2M+1)). Runtime ~4 minutes; ~2 GB peak (block C, M=256).

Run from the repository root: python research/depth_tails.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pickle
from mlqae import geom_ladder, canonical_shots, evaluate_schedule, KAPPA

HDR = (f'{"M":>5s} {"cfg":>12s} {"ntrials":>8s} {"Ntot":>9s} {"eps95*M":>8s} '
       f'{"eps/CRLB":>9s} {"Ct95":>7s} {"Ct99":>8s} {"Ct999":>8s} '
       f'{"farflips":>12s} {"emax":>9s}')


def capped_ladder(M):
    ladder = geom_ladder(M, 1.45)
    if ladder[-1] != M:
        ladder.append(M)
    return ladder


def pair_schedule(M, anchor):
    M2 = int(round(0.8 * M))
    ladder = sorted(set(geom_ladder(M, 1.45) + [M2, M]))
    shots = list(canonical_shots(len(ladder)))
    shots[ladder.index(M)] += anchor // 2
    shots[ladder.index(M2)] += anchor // 2
    return ladder, shots


def run(M, cfg, ladder, shots, ntrials, seed, chunk=2500):
    r = evaluate_schedule(ladder, shots, ntrials, seed=seed, chunk=chunk,
                          return_extra=True)
    basin = np.pi / (2 * (2 * M + 1))
    th_err = np.abs(r['theta_hat'] - np.arcsin(r['a_true']))
    nfar = int((th_err > 2 * basin).sum())
    p999 = float(np.percentile(r['errors'], 99.9))
    crlb = KAPPA[95] * r['fisher_sigma']
    row = dict(M=M, cfg=cfg, ntrials=ntrials, nq=r['nq'], eps95=r['p95'],
               ratio=r['p95'] / crlb, Ct95=r['p95'] ** 2 * M * r['nq'],
               Ct99=r['p99'] ** 2 * M * r['nq'], Ct999=p999 ** 2 * M * r['nq'],
               nfar=nfar, far_rate=nfar / ntrials, emax=r['emax'])
    print(f'{M:5d} {cfg:>12s} {ntrials:8d} {r["nq"]:9d} {r["p95"]*M:8.4f} '
          f'{row["ratio"]:9.3f} {row["Ct95"]:7.3f} {row["Ct99"]:8.3f} '
          f'{row["Ct999"]:8.3f} {nfar:5d}={nfar/ntrials:6.1e} '
          f'{r["emax"]:9.2e}', flush=True)
    return row


if __name__ == '__main__':
    rows = {'A': [], 'B': [], 'C': [], 'D': []}

    print('--- block A: single cap anchor (stall + diverging tails) ---')
    print(HDR)
    M = 64
    ladder = capped_ladder(M)
    base = canonical_shots(len(ladder))
    for anchor in (4096, 65536, 1048576):
        shots = list(base)
        shots[-1] += anchor
        rows['A'].append(run(M, f'a{anchor}', ladder, shots, 20000,
                             seed=550000 + 7 * M + anchor))

    print('--- block B: pair anchor {0.8M, M} (no stall at the 95th pct) ---')
    print(HDR)
    for anchor in (4096, 65536, 1048576):
        ladder, shots = pair_schedule(64, anchor)
        rows['B'].append(run(64, f'a{anchor}', ladder, shots, 20000,
                             seed=31337 + anchor))

    print('--- block C: pair anchor, deep-trial tail quantification ---')
    print(HDR)
    ladder, shots = pair_schedule(64, 4096)
    rows['C'].append(run(64, 'a4096', ladder, shots, 1000000, seed=424242,
                         chunk=4000))
    for anchor in (4096, 65536):
        ladder, shots = pair_schedule(256, anchor)
        rows['C'].append(run(256, f'a{anchor}', ladder, shots, 100000,
                             seed=606060 + anchor, chunk=2000))

    print('--- block D: uniform scaling (clean tails at every budget) ---')
    print(HDR)
    for M in (64, 256):
        ladder = capped_ladder(M)
        base = canonical_shots(len(ladder))
        for s in (4, 16):
            shots = [b * s for b in base]
            rows['D'].append(run(M, f's{s}', ladder, shots, 200000,
                                 seed=990000 + M + s, chunk=4000))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'depth_tails.pkl'), 'wb') as h:
        pickle.dump(rows, h)
