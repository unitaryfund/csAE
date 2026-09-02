"""Certified final numbers for the paper: scale table, tail-tuned config,
and depolarizing-noise study. Fresh holdout seeds throughout, bootstrap CIs.
Writes paper_numbers.pkl.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pickle
from mlqae import (geom_ladder, canonical_shots, flagship_schedule,
                   evaluate_schedule, boot_ci_constant)

HERE = os.path.dirname(os.path.abspath(__file__))

# Trial counts. At 1e6 the CI on C95 is +-0.6%, already tighter than the
# convention sensitivities (widening the amplitude prior moves C95 by ~1.6%),
# so there is little point going further. Runtime ~13 min, dominated by the
# bootstrap resampling rather than by the simulation.
NT_TABLE1 = 1_000_000
NT_NOISE = 200_000
out = {}

# ---------- 1. flagship scale table: r=1.45, base=1 ----------
print(f'=== flagship (r=1.45 ladder + cap rung), {NT_TABLE1//1000}k holdout ===')
out['scale_table'] = []
for nmax in (60, 125, 260, 540, 1150, 2400):
    ladder, shots = flagship_schedule(nmax)
    r = evaluate_schedule(ladder, shots, NT_TABLE1, seed=880000 + nmax,
                          return_errors=True, chunk=1500)
    ci68 = boot_ci_constant(r['errors'], r['nq'], 68)
    ci95 = boot_ci_constant(r['errors'], r['nq'], 95)
    ci99 = boot_ci_constant(r['errors'], r['nq'], 99)
    del r['errors']
    out['scale_table'].append((nmax, ladder, shots, r, ci95, ci99, ci68))
    print(f"nmax={r['nmax']:5d} q={r['nq']:6d} eps95={r['p95']:.2e}  "
          f"C68={r['C68']:.2f} [{ci68[0]:.2f},{ci68[1]:.2f}]  "
          f"C95={r['C95']:.2f} [{ci95[0]:.2f},{ci95[1]:.2f}]  "
          f"C99={r['C99']:.2f} [{ci99[0]:.2f},{ci99[1]:.2f}]  "
          f"C95par={r['C95par']:.3f}", flush=True)

# ---------- 2. what the cap rung buys (plain ladder comparison) ----------
# Sec. IV analyses the PLAIN geometric ladder; Sec. VI uses it plus one rung.
# This quantifies the difference, and runs the three controls that show the
# rung is neither a tuned constant nor a disguised shot increase.
print('\n=== cap rung vs plain geometric ladder, 50k holdout ===')
out['plain_vs_rung'] = {}
for nmax in (125, 540):
    pl = geom_ladder(nmax, 1.45)
    for name, (l, sh) in (('plain', (pl, canonical_shots(len(pl)))),
                          ('flagship', flagship_schedule(nmax))):
        r = evaluate_schedule(l, sh, 50000, seed=991177, chunk=2000)
        out['plain_vs_rung'][(nmax, name)] = r
        print(f"  nmax={r['nmax']:5d} {name:>9}: Ntot={r['nq']:7d} C68={r['C68']:.3f} "
              f"C95={r['C95']:.3f} C99={r['C99']:.2f} par={r['C95par']:.3f}", flush=True)

print('  -- control: rung placement is a shallow dial, not a tuned constant')
pl = geom_ladder(125, 1.45)
for d in (1.1, 1.3, 1.6, 2.0, 2.5):
    l, sh = flagship_schedule(125, cap_divisor=d)
    if len(l) == len(pl):
        continue
    r = evaluate_schedule(l, sh, 50000, seed=88123, chunk=2000)
    print(f"     divisor {d:.1f} (rung {sorted(set(l)-set(pl))[0]:4d}): "
          f"C95={r['C95']:.3f} C99={r['C99']:.2f}", flush=True)

print('  -- control: not a disguised shot increase (uniform scaling, matched Ntot)')
bs = canonical_shots(len(pl))
nq_base = int(np.sum(np.asarray(pl) * np.asarray(bs)) + bs[0])
fl, fs = flagship_schedule(125)
nq_fl = int(np.sum(np.asarray(fl) * np.asarray(fs)) + fs[0])
sf = nq_fl / nq_base
ru = evaluate_schedule(pl, [max(1, int(round(sf * b))) for b in bs], 50000,
                       seed=991177, chunk=2000)
print(f"     uniform x{sf:.2f}: Ntot={ru['nq']} C95={ru['C95']:.3f} "
      f"C99={ru['C99']:.2f} par={ru['C95par']:.3f}", flush=True)

# ---------- 3. depolarizing-noise study ----------
print(f"\n=== noise study: eta per oracle call, matched vs mismatched ({NT_NOISE//1000}k) ===")
out['noise'] = []
for nmax in (182, 540):
    ladder, shots = flagship_schedule(nmax)
    for eta in (0.0, 1e-5, 1e-4, 3e-4, 1e-3):
        ra = evaluate_schedule(ladder, shots, NT_NOISE, chunk=2000, seed=77000 + nmax, eta=eta)
        rm = evaluate_schedule(ladder, shots, NT_NOISE, chunk=2000, seed=77000 + nmax, eta=eta,
                               model_eta=0.0)
        out['noise'].append((nmax, eta, ra, rm))
        print(f"nmax={nmax:4d} eta={eta:7.0e}: aware eps95={ra['p95']:.2e} "
              f"C95={ra['C95']:6.2f} | unaware eps95={rm['p95']:.2e} "
              f"C95={rm['C95']:6.2f}", flush=True)

with open(os.path.join(HERE, 'paper_numbers.pkl'), 'wb') as h:
    pickle.dump(out, h)
print('\nsaved paper_numbers.pkl')
