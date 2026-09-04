"""Figures for the dense-ladder ML-QAE paper. Reads the result pickles.
Palette: Okabe-Ito subset, CVD-validated; identity always doubled by marker
shape / linestyle and direct labels.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figures')
os.makedirs(FIGS, exist_ok=True)

BLUE, VERM, GREEN, PINK = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.6, 'figure.dpi': 200,
})


# ---------------- Fig 1: ratio sweep ----------------
with open(os.path.join(HERE, 'ratio_sweep.pkl'), 'rb') as h:
    rows = pickle.load(h)
rows = np.array([(nm, r, q, c95, c99, cpar, flip) for nm, r, q, c95, c99, cpar, flip, _ in rows])

fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True,
                         gridspec_kw={'height_ratios': [1.3, 1]})
ax = axes[0]
for nm, color, ls in ((128, BLUE, '-'), (256, VERM, '--')):
    m = rows[:, 0] == nm
    ax.plot(rows[m, 1], rows[m, 3], ls, color=color, marker='o', ms=3.5,
            label=f'$C_{{95}}$, $n_{{\\max}}\\!=\\!{nm}$')
    ax.plot(rows[m, 1], rows[m, 4], ls, color=color, marker='s', ms=3.5, alpha=0.45,
            label=f'$C_{{99}}$, $n_{{\\max}}\\!=\\!{nm}$')
ax.axvline(2.0, color='#999999', lw=0.8, ls=':')
ax.text(1.985, 30, 'powers of two', rotation=90, fontsize=7, color='#666666',
        ha='right', va='bottom')
ax.set_yscale('log')
ax.set_ylabel(r'constant $\varepsilon_\delta \cdot N_{\mathrm{tot}}$')
ax.legend(frameon=False, ncol=1, loc='upper left')

ax = axes[1]
for nm, color, ls in ((128, BLUE, '-'), (256, VERM, '--')):
    m = rows[:, 0] == nm
    ax.semilogy(rows[m, 1], np.maximum(rows[m, 6], 2e-5), ls, color=color,
                marker='o', ms=3.5, label=f'$n_{{\\max}}={nm}$')
ax.axvline(2.0, color='#999999', lw=0.8, ls=':')
ax.set_ylabel('basin-flip rate')
ax.set_xlabel(r'ladder ratio $r$')
ax.legend(frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'ratio_sweep.pdf'))
plt.close(fig)

# ---------------- Fig 2: aliasing exponent ----------------
with open(os.path.join(HERE, 'alias_analysis.pkl'), 'rb') as h:
    summary = pickle.load(h)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for name, label, color in (('pow2', r'$r=2$ (powers of two)', VERM),
                           ('geom1.8', r'$r=1.8$', PINK),
                           ('geom1.45', r'$r=1.45$', BLUE),
                           ('geom1.25', r'$r=1.25$', GREEN)):
    s = summary[name]
    ds = np.asarray(s['ds'])
    E = np.asarray(s['E_min_over_theta'])
    order = np.argsort(ds)
    basin = np.pi / (2 * (2 * max(s['ladder']) + 1))
    pos = ds[order] > 0
    ax.plot(ds[order][pos] / basin, E[order][pos], color=color, lw=1.2, label=label)
ax.set_xscale('log')
ax.set_xlabel(r'rival offset $d$  [deepest-level basins]')
ax.set_ylabel(r'rejection exponent $\min_\theta E(\theta, d)$')
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'alias_exponent.pdf'))
plt.close(fig)

# ---------------- Fig 3: scaling head-to-head ----------------
_hh_ext = os.path.join(HERE, 'head_to_head_ext.pkl')
with open(_hh_ext if os.path.exists(_hh_ext) else
          os.path.join(HERE, 'head_to_head.pkl'), 'rb') as h:
    hh = pickle.load(h)

fig, ax = plt.subplots(figsize=(3.4, 2.8))
q_ours = np.array([r[3]['nq'] for r in hh['ours']])
e_ours = np.array([r[3]['p95'] for r in hh['ours']])
q_cheb_ave, q_cheb_max, e_cheb = [], [], []
for nmax, eps, err, q, md in hh['cheb95']:
    e_cheb.append(np.percentile(err, 95))
    q_cheb_ave.append(q.mean())
    q_cheb_max.append(q.max())
e_cheb, q_cheb_ave, q_cheb_max = map(np.array, (e_cheb, q_cheb_ave, q_cheb_max))

xs = np.array([np.min(q_ours) * 0.6, np.max(q_cheb_max) * 1.6])
Cfit = float(np.mean(e_ours * q_ours))
ax.loglog(xs, Cfit / xs, color=BLUE, lw=1.0, alpha=0.6)
ax.loglog(q_ours, e_ours, 'o', color=BLUE, ms=5, label=f'this work  ${Cfit:.1f}/N$')
Ca = float(np.mean(e_cheb * q_cheb_ave))
ax.loglog(xs, Ca / xs, color=VERM, lw=1.0, alpha=0.6, ls='--')
ax.loglog(q_cheb_ave, e_cheb, 's', color=VERM, ms=5, label=f'chebAE ave.  ${Ca:.1f}/N$')
Cm = float(np.mean(e_cheb * q_cheb_max))
ax.loglog(xs, Cm / xs, color=GREEN, lw=1.0, alpha=0.6, ls=':')
ax.loglog(q_cheb_max, e_cheb, '^', color=GREEN, ms=5, label=f'chebAE max.  ${Cm:.1f}/N$')
ax.loglog(xs, 4.3 / xs, color='#888888', lw=1.0, ls='-.',
          label='csAE  $4.3/N$')
ax.set_xlabel(r'total oracle queries $N$')
ax.set_ylabel(r'estimation error $\varepsilon_{95}$')
ax.legend(frameon=False, fontsize=7.5, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'scaling.pdf'))
plt.close(fig)

# ---------------- Fig 4: depth-limited fan ----------------
fan_file = os.path.join(HERE, 'depth_fan.pkl')
if os.path.exists(fan_file):
    with open(fan_file, 'rb') as h:
        fan = pickle.load(h)
    fan = np.array(fan)  # (M, s, nq, p95, p99)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    xs = np.array([1e2, 5e6])
    ax.loglog(xs, 2.9 / xs, color='#888888', lw=1.0, ls='-.',
              label=r'Heisenberg limit  $2.9/N$')
    for M, color in ((16, BLUE), (64, VERM), (256, GREEN), (1024, PINK)):
        m = fan[:, 0] == M
        q, e = fan[m, 2], fan[m, 3]
        Ct = np.mean((e ** 2 * M * q)[1:])
        qq = np.array([q.min() * 0.7, q.max() * 1.5])
        ax.loglog(qq, np.sqrt(Ct / (M * qq)), color=color, lw=1.0, alpha=0.5)
        ax.loglog(q, e, 'o', color=color, ms=4, label=f'$M={M}$')
    ax.text(1.1e5, np.sqrt(0.4 / (16 * 1.1e5)) * 1.55,
            r'fixed $M$: $\propto N^{-1/2}$', fontsize=7.5, color='#666666')
    # a beta = 1/2 depth-growth policy: M ~ sqrt(N), realized by the existing
    # points (16,4) -> (64,16) -> (256,64) -> (1024,256); slope -(1+beta)/2 = -3/4
    path = [(16, 4), (64, 16), (256, 64), (1024, 256)]
    pq, pe = [], []
    for M, s in path:
        row = fan[(fan[:, 0] == M) & (fan[:, 1] == s)][0]
        pq.append(row[2]); pe.append(row[3])
    pq, pe = np.array(pq), np.array(pe)
    Cp = float(np.mean(pe * pq ** 0.75))
    qq = np.array([pq.min() * 0.5, pq.max() * 2.5])
    ax.loglog(qq, Cp / qq ** 0.75, color='#333333', lw=1.2, ls='--',
              label=r'$M \propto N^{1/2}$:  $\propto N^{-3/4}$')
    ax.loglog(pq, pe, 'D', mfc='none', mec='#333333', ms=7, mew=1.2)
    ax.set_xlabel(r'total oracle queries $N$')
    ax.set_ylabel(r'estimation error $\varepsilon_{95}$')
    ax.set_xlim(1e2, 5e6)
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'depth_fan.pdf'))
    plt.close(fig)

print('figures written to', FIGS)
