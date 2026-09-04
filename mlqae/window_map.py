"""Window-MAP: the Bayes rule for a tolerance loss, measured on the flagship.

For the loss 1{|a_hat - a| > h} the Bayes rule maximises the posterior mass
inside a window of half-width h, i.e. reports the CENTRE of the highest-mass
window -- not the peak. The MLE is the h -> 0 member of that family, which is
why it is not the right rule for a percentile metric with a finite tolerance.

Reported against the plain MLE on identical measurement records.
Backs the window-rule numbers of Sec. VIII (h ~ 2.8 sigma: C99 down 14-16%,
C95 up ~2%, C68 up ~9% at nmax = 125 and 382).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mlqae import flagship_schedule, grid_spacing, boot_ci_constant

NT = 200_000


def run(nmax, halfwidths, ntrials=NT, seed=880000):
    lad, sh = flagship_schedule(nmax)
    d = 2 * np.asarray(lad, float) + 1
    N = np.asarray(sh, int)
    step = grid_spacing(lad, sh)
    grid = np.arange(1e-6, np.pi / 2, step)
    P = np.clip(np.cos(np.outer(grid, d)) ** 2, 1e-12, 1 - 1e-12)
    logP, log1mP = np.log(P), np.log1p(-P)
    sig = 1 / np.sqrt(4 * np.sum(N * d ** 2))
    nq = int(np.sum(np.asarray(lad) * N) + N[0])

    rng = np.random.default_rng(seed + nmax)
    a = rng.uniform(0.1, 0.9, ntrials)
    K = rng.binomial(N[None, :], np.cos(np.outer(np.arcsin(a), d)) ** 2)

    out = {h: [] for h in halfwidths}
    for lo in range(0, ntrials, 2000):
        hi = min(lo + 2000, ntrials)
        Kc = K[lo:hi]
        ll = Kc @ logP.T + (N[None, :] - Kc) @ log1mP.T
        ll -= ll.max(axis=1, keepdims=True)
        L = np.exp(ll)
        c = np.pad(np.cumsum(L, axis=1), ((0, 0), (1, 0)))
        ar = np.arange(L.shape[1])
        for h in halfwidths:
            if h == 0:
                idx = np.argmax(ll, axis=1)
            else:
                up = np.clip(ar + h + 1, 0, L.shape[1])
                dn = np.clip(ar - h, 0, L.shape[1])
                idx = np.argmax(c[:, up] - c[:, dn], axis=1)
            out[h].append(grid[idx])
    return {h: np.abs(np.sin(np.concatenate(v)) - a) for h, v in out.items()}, nq, step / sig


if __name__ == '__main__':
    for nmax in (125, 382):
        errs, nq, pt = run(nmax, (0, 1, 2, 3, 4, 5))
        base = np.percentile(errs[0], 95) * nq
        print(f'\nflagship nmax={nmax}, {NT} trials  (1 grid pt = {pt:.2f} sigma)')
        print(f'{"h/sigma":>9}{"C68":>9}{"C95":>9}{"C99":>9}'
              f'{"dC68":>8}{"dC95":>8}{"dC99":>8}')
        b = {p: np.percentile(errs[0], p) * nq for p in (68, 95, 99)}
        for h, e in errs.items():
            v = {p: np.percentile(e, p) * nq for p in (68, 95, 99)}
            ci = boot_ci_constant(e, nq, 99)
            print(f'{h*pt:>9.2f}{v[68]:>9.3f}{v[95]:>9.3f}'
                  f'{v[99]:>9.3f}'
                  f'{100*(v[68]/b[68]-1):>7.1f}%{100*(v[95]/b[95]-1):>7.1f}%'
                  f'{100*(v[99]/b[99]-1):>7.1f}%', flush=True)
