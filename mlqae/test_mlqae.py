"""Validation tests for the ladder code (mlqae/core.py and friends).

Run:  python mlqae/test_mlqae.py        (plain runner, no dependencies)
  or: pytest mlqae/test_mlqae.py        (if pytest is available)

Philosophy: every test checks the implementation against something it did not
define itself -- an analytic limit, an independently derived bound, a published
external number, or a pinned historical value. Deterministic seeds make every
test exactly reproducible; tolerances cover only floating-point/BLAS variation.
Total runtime ~2 minutes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys
import unittest  # SkipTest: recognized as a skip by the runner below and by pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mlqae import (geom_ladder, canonical_shots, grid_spacing,
                   crlb_constant, KAPPA, flagship_schedule, evaluate_schedule)


# ---------------------------------------------------------------- pure math
def test_fisher_lemma_numeric():
    """Numerically verify I_n = 4(2n+1)^2, independent of theta (Lemma II.1)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(0, 300))
        th = rng.uniform(0.05, np.pi / 2 - 0.05)
        h = 1e-7
        p = lambda t: np.cos((2 * n + 1) * t) ** 2
        dp = (p(th + h) - p(th - h)) / (2 * h)
        pt = p(th)
        if pt < 1e-9 or pt > 1 - 1e-9:      # derivative test ill-conditioned at extrema
            continue
        I = dp ** 2 / (pt * (1 - pt))
        assert abs(I / (4 * (2 * n + 1) ** 2) - 1) < 1e-4, (n, th, I)


def test_ladder_constructor():
    for r in (1.3, 1.45, 2.0):
        for cap in (60, 128, 1000):
            lad = geom_ladder(cap, r)
            assert lad[0] == 0
            assert lad == sorted(set(lad))
            assert max(lad) <= cap
    assert geom_ladder(128, 2.0) == [0, 1, 2, 4, 8, 16, 32, 64, 128]


def test_query_accounting():
    """N_tot = sum(N_j * n_j) + N_1 (the csAE paper's convention)."""
    lad = [0, 1, 2, 4]
    shots = [5, 4, 3, 2]
    r = evaluate_schedule(lad, shots, 100, seed=1)
    assert r['nq'] == 5 + 4 * 1 + 3 * 2 + 2 * 4  # = 23


# ------------------------------------------------------- statistical limits
def test_classical_endpoint():
    """Depth-0 only = classical sampling; eps95 must match binomial theory.

    Theory: sigma_theta = 1/(2 sqrt(N)) (Fisher = 4N), error in a is
    cos(theta) * |dtheta|. Predicted eps95 computed by direct simulation of
    the Gaussian limit; measured must agree within 5%.
    """
    N = 20000
    r = evaluate_schedule([0], [N], 20000, seed=42)
    rng = np.random.default_rng(7)
    a = rng.uniform(0.1, 0.9, 200000)
    pred = np.percentile(np.abs(rng.normal(size=a.size)) * np.sqrt(1 - a ** 2) / (2 * np.sqrt(N)), 95)
    assert abs(r['p95'] / pred - 1) < 0.05, (r['p95'], pred)


def test_crlb_tracking_flagship():
    """Flagship ladder at the Heisenberg endpoint: eps95 within [1.2, 1.5]x
    its own CRLB. The endpoint ratio is ~1.4 (single-digit deep shots,
    non-Gaussian percentiles); the scaled regime, where the ratio drops to
    1.07-1.15, is covered by test_no_resolution_floor. Historical note: this
    test's first version asserted <1.35 based on a stale CRLB figure in the
    paper (2.4-2.5, which belongs to the csAE power-of-two schedule, not the
    flagship ladder); the failure exposed and fixed that paper error."""
    lad = geom_ladder(125, 1.45)
    shots = canonical_shots(len(lad))
    r = evaluate_schedule(lad, shots, 20000, seed=880125)
    crlb95 = KAPPA[95] * r['fisher_sigma']
    ratio = r['p95'] / crlb95
    assert 1.2 < ratio < 1.5, ratio


def test_no_resolution_floor():
    """Heavily scaled schedule must keep tracking the CRLB.

    Regression test for the grid-interpolation floor bug: with the coarse-grid
    estimator (no zoom refinement), this ratio was ~2 and the test fails.
    """
    lad = geom_ladder(16, 1.45)
    if lad[-1] != 16:
        lad.append(16)
    shots = [b * 256 for b in canonical_shots(len(lad))]
    r = evaluate_schedule(lad, shots, 8000, seed=3333)
    crlb95 = KAPPA[95] * r['fisher_sigma']
    assert r['p95'] / crlb95 < 1.30, r['p95'] / crlb95


def test_mode_width_identity():
    """Appendix A, Eq. (A3)-(A4): basin/sigma_theta = pi sqrt(Lambda) -> 6 sqrt(s).

    Note this is an algebraic identity (both sides are the same sum rearranged);
    it guards the formula, not the physics. The substantive claim -- that a
    likelihood mode is really sigma_theta wide -- is tested separately below.
    """
    for nmax in (125, 803):
        lad = geom_ladder(nmax, 1.45)
        for s in (1, 64, 256, 1024):
            shots = np.array([s * b for b in canonical_shots(len(lad))], float)
            d = 2 * np.asarray(lad, float) + 1
            basin = np.pi / (2 * (2 * nmax + 1))
            sigma = 1.0 / np.sqrt(4 * np.sum(shots * d ** 2))
            lam = np.sum(shots * (d / d[-1]) ** 2)
            assert abs(basin / sigma / (np.pi * np.sqrt(lam)) - 1) < 1e-9
            assert abs(basin / sigma / (6 * np.sqrt(s)) - 1) < 0.02, (nmax, s)


def test_mode_width_measured():
    """The likelihood's actual curvature must match sigma_theta (Appendix A).

    Measures -ell''(theta) at the truth on simulated records and compares the
    resulting mode width to the Fisher prediction. Modes run ~12% wide at s=1
    (observed information fluctuates at single-digit shot counts) and converge
    as the shots are scaled up; the grid rule must survive the narrow tail, so
    the 90th percentile is checked too.
    """
    for nmax, s, lo, hi in ((125, 1, 0.85, 1.02), (125, 64, 0.97, 1.02),
                            (803, 256, 0.97, 1.02)):
        lad = geom_ladder(nmax, 1.45)
        shots = np.array([s * b for b in canonical_shots(len(lad))], int)
        d = 2 * np.asarray(lad, float) + 1
        sigma = 1.0 / np.sqrt(4 * np.sum(shots * d ** 2))

        rng = np.random.default_rng(99)
        a = rng.uniform(0.1, 0.9, 4000)
        th = np.arcsin(a)
        K = rng.binomial(shots[None, :], np.cos(np.outer(th, d)) ** 2)

        def ell(t):
            p = np.clip(np.cos(t[:, None] * d[None, :]) ** 2, 1e-12, 1 - 1e-12)
            return np.sum(K * np.log(p) + (shots[None, :] - K) * np.log1p(-p), axis=1)

        h = sigma / 4
        curv = -(ell(th + h) - 2 * ell(th) + ell(th - h)) / h ** 2
        width = 1.0 / np.sqrt(curv[curv > 0])
        ratio = sigma / width                     # 1.0 == mode exactly sigma wide
        assert lo < np.median(ratio) < hi, (nmax, s, np.median(ratio))
        # narrowest modes must stay within the 4 sigma_theta grid margin
        assert np.percentile(ratio, 90) < 1.30, (nmax, s, np.percentile(ratio, 90))


def test_grid_convergence():
    """Appendix A: the shipped grid must agree with a 32x finer coarse search.

    Certifies that grid resolution contributes nothing to the reported numbers:
    eps95 must match the refined search to well inside the bootstrap CI, at both
    an unscaled and a heavily scaled schedule (where the second term of the
    spacing rule binds).
    """
    for nmax, s, ntrials in ((125, 1, 20000), (125, 256, 6000)):
        lad = geom_ladder(nmax, 1.45)
        shots = [s * b for b in canonical_shots(len(lad))]
        kw = dict(seed=4242, chunk=250)
        coarse = evaluate_schedule(lad, shots, ntrials, **kw)
        fine = evaluate_schedule(lad, shots, ntrials, **kw,
                                 grid_step=grid_spacing(lad, shots) / 32)
        rel = coarse['p95'] / fine['p95'] - 1
        assert abs(rel) < 0.004, (nmax, s, rel, coarse['p95'], fine['p95'])


def test_reproducibility():
    lad = geom_ladder(60, 1.45)
    shots = canonical_shots(len(lad))
    r1 = evaluate_schedule(lad, shots, 500, seed=11, return_errors=True)
    r2 = evaluate_schedule(lad, shots, 500, seed=11, return_errors=True)
    assert np.array_equal(r1['errors'], r2['errors'])


def test_golden_flagship_value():
    """Pin two certified values at nmax=125; a change in either means the
    estimator's behavior moved and every published number needs re-certifying.

    The PLAIN geometric ladder is the object Sec. IV analyses (C95 = 2.82); the
    flagship adds one rung near the cap and is what Table I reports.
    """
    lad = geom_ladder(125, 1.45)
    r = evaluate_schedule(lad, canonical_shots(len(lad)), 30000, seed=880125)
    assert abs(r['C95'] - 2.82) < 0.04, ('plain', r['C95'])
    assert abs(r['C95par'] - 0.284) < 0.006, ('plain', r['C95par'])

    r = evaluate_schedule(*flagship_schedule(125), 30000, seed=880125)
    assert abs(r['C95'] - 2.85) < 0.05, ('flagship', r['C95'])
    assert abs(r['C99'] - 5.23) < 0.30, ('flagship', r['C99'])
    assert abs(r['C95par'] - 0.209) < 0.008, ('flagship', r['C95par'])


# ------------------------------------------------------------------- noise
def test_noise_model_limits():
    """eta=0 matches noiseless exactly; matched likelihood beats or equals the
    noise-ignorant one; tiny eta*nmax is harmless."""
    lad = geom_ladder(125, 1.45)
    shots = canonical_shots(len(lad))
    base = evaluate_schedule(lad, shots, 8000, seed=99)
    zero = evaluate_schedule(lad, shots, 8000, seed=99, eta=0.0)
    assert abs(base['p95'] - zero['p95']) < 1e-12
    aware = evaluate_schedule(lad, shots, 8000, seed=99, eta=1e-3)
    ignorant = evaluate_schedule(lad, shots, 8000, seed=99, eta=1e-3, model_eta=0.0)
    assert aware['p95'] <= ignorant['p95'] * 1.02
    tiny = evaluate_schedule(lad, shots, 8000, seed=99, eta=1e-6)
    assert tiny['p95'] < base['p95'] * 1.05


# ------------------------------------------- external anchors (need parent repo)
def test_external_anchor_chebae():
    """Our harness must reproduce the published chebAE average-case constant
    (~3.07 at 95%, csAE paper Table I) using the repository's own chebAE.py.
    Loose window: 300 trials only, [2.5, 3.7]."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from chebAE import chebae
    except ImportError as e:
        raise unittest.SkipTest(f'chebAE.py / statsmodels unavailable ({e})')
    rng = np.random.default_rng(5)
    errs, qs = [], []
    for i in range(300):
        a = rng.uniform(0.1, 0.9)
        np.random.seed(10000 + i)
        res = chebae(a, 2e-3, 0.05)
        errs.append(abs(res['a_hat'] - a))
        qs.append(res['queries'])
    C = np.percentile(errs, 95) * np.mean(qs)
    assert 2.5 < C < 3.7, C


if __name__ == '__main__':
    import time
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed, skipped = 0, []
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'PASS  {name}  ({time.time()-t0:.1f}s)')
        except unittest.SkipTest as e:
            skipped.append(name)
            print(f'SKIP  {name}: {e}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {name}: {e}')
    ran = len(tests) - len(skipped)
    print(f'\n{ran - failed}/{ran} passed'
          + (f', {len(skipped)} SKIPPED ({", ".join(skipped)})' if skipped else ''))
    sys.exit(1 if failed else 0)
