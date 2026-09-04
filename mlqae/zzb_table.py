"""Ziv-Zakai lower-bound columns of Table V (Appendix B protocol).

For the flagship schedule at nmax = 125 and 382, evaluates the bound

    Pr[|a_hat - a| >= g/2] >= A(g) := \\int [p(a)+p(a+g)] P_min(a, a+g) da

with the unnormalised binary-test error mass computed EXACTLY (no
Bhattacharyya inequality) via

    sum_x min(p0 P, p1 Q) = p0 E_{x~P}[min(1, (p1/p0) e^{Delta})],

a plain Monte Carlo average over exactly sampled binomial datasets, Delta the
log-likelihood ratio. Valley-filling (running maximum from large g) and the
percentile crossing eps_delta >= g*/2 at A(g*) = 1 - delta follow Appendix B.

Settings (520 separations, 600 amplitudes, 2e3 datasets per amplitude) are
the ones quoted in the appendix; coarsening each grid 2x and the Monte Carlo
5x moves the entries by up to 3%, so read the printed digits to ~1%. The
CRLB column of Table V is printed alongside; the achieved column is Table I.
Runtime ~15 min for both scales.

Run from the repository root: python mlqae/zzb_table.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from mlqae import flagship_schedule, crlb_constant

A_LO, A_HI = 0.1, 0.9


def A_of_g_exact(depths, shots, gs, na=600, mc=2000, seed=7):
    """A(g) with P_min evaluated exactly by Monte Carlo (Appendix B.3)."""
    d = 2 * np.asarray(depths, float) + 1
    N = np.asarray(shots, int)
    rng = np.random.default_rng(seed)
    P = 1.0 / (A_HI - A_LO)
    out = np.empty(len(gs))
    for i, g in enumerate(gs):
        hi = A_HI - g
        if hi <= A_LO:
            out[i] = 0.0
            continue
        a = np.linspace(A_LO, hi, na)
        p0 = np.clip(np.cos(np.outer(np.arcsin(a), d)) ** 2, 1e-12, 1 - 1e-12)
        p1 = np.clip(np.cos(np.outer(np.arcsin(a + g), d)) ** 2, 1e-12, 1 - 1e-12)
        # mc datasets per amplitude, drawn under hypothesis a
        k = rng.binomial(N[None, None, :], p0[:, None, :], size=(na, mc, len(N)))
        dl = np.sum(k * (np.log(p1) - np.log(p0))[:, None, :]
                    + (N - k) * (np.log1p(-p1) - np.log1p(-p0))[:, None, :], axis=2)
        overlap = np.mean(np.minimum(1.0, np.exp(np.minimum(dl, 0.0)))
                          * (dl <= 0) + (dl > 0) * 1.0, axis=1)
        out[i] = np.trapz(P * overlap, a)
    return out


if __name__ == '__main__':
    for nmax in (125, 382):
        lad, sh = flagship_schedule(nmax)
        nq = int(np.sum(np.asarray(lad) * np.asarray(sh)) + sh[0])
        gs = np.geomspace(1e-5, 0.2, 520)
        t0 = time.time()
        A = np.maximum.accumulate(
            A_of_g_exact(lad, sh, gs)[::-1])[::-1]      # valley-filling
        print(f'\nflagship nmax={nmax}  Ntot={nq}  ({time.time()-t0:.0f}s)')
        for pct in (68, 95, 99):
            ok = np.where(A >= 1 - pct / 100)[0]
            z = (gs[ok[-1]] / 2 * nq) if len(ok) else 0.0
            print(f'  delta={pct}:  CRLB {crlb_constant(lad, sh, pct):.3f}   '
                  f'ZZB {z:.3f}', flush=True)
