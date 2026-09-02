"""Maximum-likelihood amplitude estimation on non-uniform Grover-depth ladders.

Measurement model (identical to csAE / the paper's simulations): at depth n the
circuit G^n U|0> is measured in the Z basis, giving Bernoulli samples with
p(n, theta) = cos^2((2n+1) theta), a = sin(theta), theta in (0, pi/2).

Estimator: exact global maximum likelihood over theta on a coarse grid of
spacing min(basin/8, 4 sigma_theta), followed by a zoom grid across the winning
coarse step and quadratic interpolation (see Appendix A of the paper). This is
statistically optimal for a given schedule (it is the exact likelihood, globally
optimized) and costs one matrix multiplication per batch of trials.

Query accounting matches the paper: total = sum_n N_n * n + N_0 (the extra N_0
counts the single U preceding the Grover iterations at depth 0).
"""
import numpy as np

__all__ = ['geom_ladder', 'fib_ladder', 'canonical_shots', 'grid_spacing',
           'crlb_constant', 'KAPPA', 'flagship_schedule',
           'evaluate_schedule', 'boot_ci_constant']


def geom_ladder(nmax, r):
    """Depths {0} + geometric sequence round(r^j) up to nmax (unique, sorted)."""
    ds, x = [0], 1.0
    while round(x) <= nmax:
        if round(x) not in ds:
            ds.append(int(round(x)))
        x *= r
    return ds


def fib_ladder(nmax):
    ds, a, b = [0], 1, 2
    while a <= nmax:
        ds.append(a)
        a, b = b, a + b
    return ds


def flagship_schedule(nmax, r=1.45, cap_divisor=1.3, base=1, slope=1.0):
    """The paper's flagship: geometric ladder plus ONE extra rung near the cap.

    The plain geometric ladder is the object the theory of Sec. IV analyses; the
    extra rung at round(nmax/cap_divisor) is what the numerics of Sec. VI use.
    It sits in the gap below the deepest rung and breaks that rung's mirror
    degeneracy (Sec. V), which is what the 99% tail is made of: C99 falls
    16-20% and the parallel constant ~27% at no cost in C95, for ~7% of C68.

    Set cap_divisor=None to get the plain ladder. Placement is a shallow dial,
    not a sharp optimum: anything in [1.1, 2.5] works, with smaller divisors
    favouring C95 and the parallel constant, larger ones favouring C99.
    """
    lad = geom_ladder(nmax, r)
    if cap_divisor is not None:
        extra = int(round(max(lad) / cap_divisor))
        if extra not in lad and extra > 0:
            lad = sorted(lad + [extra])
    return lad, canonical_shots(len(lad), base=base, slope=slope)


def canonical_shots(nlev, base=1, slope=1.0):
    """Linear decay: deepest level gets `base` shots, each shallower level
    slope more. The untuned default (base=1, slope=1) gives [nlev, ..., 2, 1]."""
    return [max(1, int(round(base + slope * (nlev - 1 - i)))) for i in range(nlev)]


# Percentile of |a_hat - a| in units of sigma_theta, for an efficient estimator
# and a ~ U(0.1, 0.9). The amplitude error is cos(theta)*|N(0, sigma_theta)|, a
# scale MIXTURE (theta is random), so the percentile must be taken after mixing:
# kappa_delta solves E_a[2 Phi(kappa/cos theta) - 1] = delta. Using the natural-
# looking z_delta * <cos theta> instead averages the scale first and understates
# the floor by 3.7% at delta=95. Computed by quadrature (see the kappa_delta definition in the paper, Sec. II).
KAPPA = {68: 0.802408, 95: 1.668608, 99: 2.258032}


def crlb_constant(depths, shots, pct=95):
    """CRLB floor on the constant C_pct = eps_pct * N_tot for this schedule."""
    depths = np.asarray(depths, float)
    shots = np.asarray(shots, float)
    sigma = 1.0 / np.sqrt(4 * np.sum(shots * (2 * depths + 1) ** 2))
    nq = float(np.sum(depths * shots) + shots[0])
    return KAPPA[pct] * sigma * nq


def grid_spacing(depths, shots):
    """Coarse-grid spacing of Appendix A, Eq. (A5): min(basin/8, 4 sigma_theta).

    The first term samples every likelihood basin ~8 times so the coarse stage
    cannot skip a rival mode; the second keeps the coarse step below twice the
    mode width, which the zoom window +-step/2 has to bracket (it binds only for
    heavily scaled schedules, s >~ 30). sigma_theta is the noiseless CRLB; under
    noise the true sigma is larger, so this stays conservative.
    """
    depths = np.asarray(depths, float)
    shots = np.asarray(shots, float)
    d = 2 * depths + 1
    basin = np.pi / (2 * (2 * depths.max() + 1))
    sigma_theta = 1.0 / np.sqrt(4.0 * np.sum(shots * d ** 2))
    return float(min(basin / 8, 4 * sigma_theta))


def evaluate_schedule(depths, shots, ntrials, seed, a_range=(0.1, 0.9),
                      return_errors=False, return_extra=False,
                      eta=0.0, model_eta=None, chunk=4000, grid_step=None):
    """Run `ntrials` simulated estimations of the schedule; report constants.

    C68 / C95 / C99 are epsilon_{68,95,99} * total queries (Table I's metric);
    C95par is epsilon_95 * max depth (parallel query complexity constant).

    Noise: with per-oracle depolarizing rate `eta`, the sampled probability at
    depth n is V_n p + (1 - V_n)/2 with fringe visibility V_n = (1-eta)^(2n+1)
    (2n+1 oracle calls at depth n). `model_eta` is the noise rate assumed by
    the likelihood: None -> same as eta (noise-aware / matched); 0.0 -> the
    estimator ignores the noise (mismatched). Trials are processed in chunks
    of `chunk` to bound memory at large depths. `grid_step` overrides the coarse
    grid spacing; it exists for the resolution study of Appendix A (see
    grid_resolution.py) and should be None -- i.e. grid_spacing() -- everywhere
    else.
    """
    depths = np.asarray(depths, float)
    shots = np.asarray(shots, int)
    d = 2 * depths + 1
    nmax = int(depths.max())
    step = grid_spacing(depths, shots) if grid_step is None else float(grid_step)
    grid = np.arange(1e-6, np.pi / 2, step)
    if model_eta is None:
        model_eta = eta
    Vm = (1.0 - model_eta) ** d
    P = np.clip(Vm * np.cos(np.outer(grid, d)) ** 2 + (1 - Vm) / 2, 1e-12, 1 - 1e-12)
    logP, log1mP = np.log(P), np.log1p(-P)

    rng = np.random.default_rng(seed)
    a_true = rng.uniform(a_range[0], a_range[1], ntrials)
    th = np.arcsin(a_true)
    Vt = (1.0 - eta) ** d
    p_true = Vt * np.cos(np.outer(th, d)) ** 2 + (1 - Vt) / 2
    K = rng.binomial(shots[None, :], p_true)

    Vm_r = Vm[None, None, :]

    def _local_refine(th0, Kc, ns):
        """Zoom on the exact likelihood around th0: 25 points across one
        coarse step, then a parabola. Needed when the statistical error is
        finer than the coarse grid (heavily scaled schedules)."""
        offs = np.linspace(-step / 2, step / 2, 25)
        th_loc = th0[:, None] + offs[None, :]
        p = Vm_r * np.cos(th_loc[:, :, None] * d[None, None, :]) ** 2 \
            + (1 - Vm_r) / 2
        p = np.clip(p, 1e-12, 1 - 1e-12)
        L = -np.sum(Kc[:, None, :] * np.log(p)
                    + (ns[None, None, :] - Kc[:, None, :]) * np.log1p(-p), axis=2)
        j = np.clip(np.argmin(L, axis=1), 1, len(offs) - 2)
        ar = np.arange(len(th0))
        ym1, y0, yp1 = L[ar, j - 1], L[ar, j], L[ar, j + 1]
        den = ym1 - 2 * y0 + yp1
        off = np.where(np.abs(den) > 1e-12,
                       0.5 * (ym1 - yp1) / np.maximum(den, 1e-12), 0.0)
        dstep = offs[1] - offs[0]
        return th_loc[ar, j] + np.clip(off, -0.5, 0.5) * dstep

    th_hat = np.empty(ntrials)
    ns = shots.astype(float)
    for lo in range(0, ntrials, chunk):
        hi = min(lo + chunk, ntrials)
        Kc = K[lo:hi]
        NLL = -(Kc @ logP.T + (shots[None, :] - Kc) @ log1mP.T)
        idx = np.clip(np.argmin(NLL, axis=1), 1, len(grid) - 2)
        th_hat[lo:hi] = _local_refine(grid[idx], Kc.astype(float), ns)
    err = np.abs(np.sin(th_hat) - a_true)

    nq = int(np.sum(depths * shots) + shots[0])
    out = dict(nq=nq, nmax=nmax,
               p50=float(np.percentile(err, 50)), p68=float(np.percentile(err, 68)),
               p95=float(np.percentile(err, 95)),
               p99=float(np.percentile(err, 99)), emax=float(err.max()),
               C68=float(np.percentile(err, 68) * nq),
               C95=float(np.percentile(err, 95) * nq),
               C99=float(np.percentile(err, 99) * nq),
               C95par=float(np.percentile(err, 95) * nmax),
               fisher_sigma=float(1.0 / np.sqrt(4 * np.sum(shots * (2 * depths + 1) ** 2))))
    if return_errors or return_extra:
        out['errors'] = err
    if return_extra:
        out['a_true'] = a_true
        out['theta_hat'] = th_hat
        out['counts'] = K
    return out


def boot_ci_constant(errors, scale, pct, B=2000, seed=1):
    """Bootstrap 95% CI for percentile(errors, pct) * scale."""
    rng = np.random.default_rng(seed)
    n = len(errors)
    cs = np.array([np.percentile(errors[rng.integers(0, n, n)], pct) * scale
                   for _ in range(B)])
    return float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))
