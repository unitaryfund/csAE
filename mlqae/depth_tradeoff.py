"""Depth vs total-query tradeoff with dense ladders + global ML.

In the depth-limited regime (max depth M << 1/eps) the optimal tradeoff is
M * N_tot ~ eps^-2 up to polylog factors [Giurgica-Tiron et al., Quantum 6,
745 (2022); Huang & Koczor, arXiv:2603.05475; Erle & Koczor,
arXiv:2608.24434]. This script measures the explicit constants achievable
with the r=1.45 ladder + global ML, via four schedule families that document
a cascade of degeneracy mechanisms:

  stage 1 (naive):    ladder + single anchor at the cap M. FAILS: p_M is
                      symmetric about each fringe extremum (mirror
                      degeneracy), so eps95*M plateaus at ~0.14-0.28
                      regardless of anchor size, and the tails GROW with
                      the anchor (confident mirror locking); see
                      depth_tails.py.
  stage 2 (pair):     anchor split over {0.8M, M}. Breaks the mirror; no
                      stall at any tested anchor (eps95/CRLB ~ 1.0 up to
                      anchor 2^20), Ctilde95 -> ~0.19. The pair's joint
                      (beat) degeneracies survive only as a thin
                      catastrophic tail (rate 1e-5..1e-4, errors ~200x
                      eps95, Ctilde99.9 up to ~90 at moderate budgets):
                      quantified in depth_tails.py.
  stage 3 (band):     anchor spread over {0.68, 0.78, 0.9, 1.0}*M.
                      Ctilde95 = eps95^2*M*N_tot ~ 0.20-0.23 at
                      eps95/CRLB ~ 1.0; same thin-tail character.
  stage 4 (uniform):  simply scale ALL canonical shots by s. Every
                      rejection exponent grows linearly in s, so all
                      failure modes die off exponentially: no stall and
                      clean tails at every budget, eps95/CRLB 1.02-1.30
                      falling with s, Ctilde95 = 0.39-0.80 (0.39-0.58 for
                      s >= 4). The recommended rule; its ~2x premium in
                      Ctilde95 over stage 2 buys exponential (rather than
                      polynomial) tail suppression.

Run: python mlqae/depth_tradeoff.py [stage]   (default: all stages)
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from mlqae import (geom_ladder, canonical_shots, evaluate_schedule,
                   KAPPA)

NT = 20000
HDR = (f'{"M":>5s} {"cfg":>12s} {"Ntot":>8s} {"eps95":>9s} {"eps95*M":>8s} '
       f'{"Ctilde68":>9s} {"Ctilde95":>9s} {"Ctilde99":>9s} {"eps/CRLB":>9s}')


def report(M, cfg, r):
    Ct68 = r['p68'] ** 2 * M * r['nq']
    Ct95 = r['p95'] ** 2 * M * r['nq']
    Ct99 = r['p99'] ** 2 * M * r['nq']
    crlb95 = KAPPA[95] * r['fisher_sigma']
    print(f'{M:5d} {cfg:>12s} {r["nq"]:8d} {r["p95"]:9.2e} {r["p95"]*M:8.3f} '
          f'{Ct68:9.3f} {Ct95:9.3f} {Ct99:9.3f} {r["p95"]/crlb95:9.2f}', flush=True)
    return (M, cfg, r['nq'], r['p68'], r['p95'], r['p99'], Ct68, Ct95, Ct99,
            r['p95'] / crlb95)


def capped_ladder(M):
    ladder = geom_ladder(M, 1.45)
    if ladder[-1] != M:
        ladder.append(M)
    return ladder


def stage1():
    print('--- stage 1: single anchor at the cap (mirror-degeneracy plateau) ---')
    print(HDR)
    rows = []
    for M in (16, 64, 256, 1024):
        ladder = capped_ladder(M)
        base = canonical_shots(len(ladder))
        for anchor in (0, 8, 64, 512, 4096):
            shots = list(base)
            shots[-1] += anchor
            r = evaluate_schedule(ladder, shots, NT,
                                  seed=550000 + 7 * M + anchor, chunk=2500)
            rows.append(report(M, f'a{anchor}', r))
    return rows


def stage2():
    print('--- stage 2: split anchor {0.8M, M} + spine scaling ---')
    print(HDR)
    rows = []
    for M in (64, 256):
        for anchor in (512, 4096):
            for k in (1, 2, 4):
                M2 = int(round(0.8 * M))
                ladder = sorted(set(geom_ladder(M, 1.45) + [M2, M]))
                shots = [s * k for s in canonical_shots(len(ladder))]
                shots[ladder.index(M)] += anchor // 2
                shots[ladder.index(M2)] += anchor // 2
                r = evaluate_schedule(ladder, shots, NT,
                                      seed=770000 + M + anchor + k, chunk=2500)
                rows.append(report(M, f'a{anchor}x{k}', r))
    return rows


def stage3():
    print('--- stage 3: anchor band over the top four rungs ---')
    print(HDR)
    rows = []
    for M in (64, 256):
        anchors_at = sorted(set(int(round(f * M)) for f in (0.68, 0.78, 0.9, 1.0)))
        for anchor in (512, 4096, 32768):
            for k in (2, 4):
                ladder = sorted(set(geom_ladder(M, 1.45) + anchors_at))
                shots = [s * k for s in canonical_shots(len(ladder))]
                for d in anchors_at:
                    shots[ladder.index(d)] += anchor // len(anchors_at)
                r = evaluate_schedule(ladder, shots, NT,
                                      seed=880000 + M + anchor + k, chunk=2500)
                rows.append(report(M, f'a{anchor}x{k}', r))
    return rows


def stage4():
    print('--- stage 4: uniform scaling of the canonical schedule (recommended) ---')
    print(HDR)
    rows = []
    for M in (64, 256):
        ladder = capped_ladder(M)
        base = canonical_shots(len(ladder))
        for s in (1, 4, 16, 64):
            shots = [b * s for b in base]
            r = evaluate_schedule(ladder, shots, NT,
                                  seed=990000 + M + s, chunk=2500)
            rows.append(report(M, f's{s}', r))
    return rows


if __name__ == '__main__':
    stages = {'1': stage1, '2': stage2, '3': stage3, '4': stage4}
    todo = sys.argv[1:] if len(sys.argv) > 1 else ['1', '2', '3', '4']
    allrows = {}
    for t in todo:
        allrows[t] = stages[t]()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'depth_tradeoff.pkl'), 'wb') as h:
        pickle.dump(allrows, h)
