"""T-MLP-u verification — 1D linear advection of a Gaussian and a step,
periodic, single round-trip.

Compares first_order, minmod_tvd_1d, and T-MLP-u wrapping superbee, MC,
UMIST, and minmod base limiters.

Goals:
  - All variants run finite, conservative.
  - T-MLP-u outputs satisfy the local maximum principle:
      no overshoot beyond the initial maximum,
      no undershoot below the initial minimum (modulo round-off).
  - T-MLP-u with a more compressive base limiter (superbee, MC) gives a
    sharper step profile than minmod_tvd_1d at the same grid.
"""
from __future__ import annotations
import numpy as np

from _pkgshim import setup_paths
setup_paths()

from mesh import build_structured_1d
from equations import Advection
from reconstruction import TMLPU
from solver import solve


def _run(label, U0, mesh, eq, *, recon, t_end, cfl=0.5, integrator='ssp_rk2'):
    res = solve(mesh, eq, U0,
                reconstruction=recon, flux='upwind',
                integrator=integrator,
                cfl=cfl, t_end=t_end, max_steps=200_000)
    U_f = res['U_final'][0]
    return U_f, res['n_steps']


def _gaussian(N, L=1.0):
    mesh = build_structured_1d(N, L=L, periodic=True)
    x = mesh.cell_centers[:, 0]
    sigma = 0.06
    U0 = np.exp(-((x - 0.3) ** 2) / (2 * sigma ** 2))[None, :]
    return mesh, U0


def _step(N, L=1.0):
    mesh = build_structured_1d(N, L=L, periodic=True)
    x = mesh.cell_centers[:, 0]
    U0 = np.where((x > 0.2) & (x < 0.4), 1.0, 0.0)[None, :]
    return mesh, U0


def main():
    print("T-MLP-u verification — 1D advection (Gaussian + step), 1 round-trip")
    print("=" * 78)
    eq = Advection(velocity=[1.0])

    fails = []

    # -- (A) Gaussian: smooth, all variants should preserve well -----------
    print("\n[A] Gaussian smooth profile, N=200, t_end=1.0 (one period):")
    mesh, U0 = _gaussian(200)
    init_max = float(np.max(U0));  init_min = float(np.min(U0))

    cases = [
        ('first_order',                     'first_order'),
        ('minmod_tvd_1d',                   'minmod_tvd_1d'),
        ('TMLPU(superbee)',                 TMLPU(tvd='superbee')),
        ('TMLPU(minmod)',                   TMLPU(tvd='minmod')),
        ('TMLPU(mc)',                       TMLPU(tvd='mc')),
        ('TMLPU(umist)',                    TMLPU(tvd='umist')),
        ('TMLPU(van_leer)',                 TMLPU(tvd='van_leer')),
    ]
    for label, recon in cases:
        U_f, n = _run(label, U0, mesh, eq, recon=recon, t_end=1.0)
        m0 = float(np.sum(U0));  mf = float(np.sum(U_f))
        m_drift = abs(mf - m0) / max(abs(m0), 1e-30)
        corr = float(np.corrcoef(U_f, U0[0])[0, 1])
        over = float(np.max(U_f) - init_max)
        under = float(init_min - np.min(U_f))
        flag = ' '
        if not np.isfinite(U_f).all():
            flag = 'NaN'; fails.append((label, 'NaN'))
        elif m_drift > 1e-10:
            flag = '!mass'; fails.append((label, f'mass drift {m_drift:.2e}'))
        elif label.startswith('TMLPU') and (over > 1e-10 or under > 1e-10):
            flag = '!LMP'
            fails.append((label, f'LMP violation over={over:.3e} under={under:.3e}'))
        print(f"  [{label:22s}] n={n:4d}  drift={m_drift:.1e}  "
              f"corr={corr:+.3f}  over={over:+.2e}  under={under:+.2e}  {flag}")

    # -- (B) Step profile: discontinuity, sharpness ------------------------
    print("\n[B] Square step profile, N=200, t_end=1.0:")
    mesh, U0 = _step(200)
    init_max = float(np.max(U0));  init_min = float(np.min(U0))
    sharpness = {}
    for label, recon in cases:
        U_f, n = _run(label, U0, mesh, eq, recon=recon, t_end=1.0)
        max_grad = float(np.max(np.abs(np.diff(U_f))))
        over = float(np.max(U_f) - init_max)
        under = float(init_min - np.min(U_f))
        flag = ' '
        if not np.isfinite(U_f).all():
            flag = 'NaN'; fails.append((label, 'NaN step'))
        elif label.startswith('TMLPU') and (over > 1e-10 or under > 1e-10):
            flag = '!LMP'
            fails.append((label, f'step LMP over={over:.3e} under={under:.3e}'))
        sharpness[label] = max_grad
        print(f"  [{label:22s}] n={n:4d}  max|Δ|={max_grad:.3f}  "
              f"over={over:+.2e}  under={under:+.2e}  {flag}")

    # T-MLP-u(superbee) should be ≥ minmod_tvd_1d in sharpness on a step.
    if 'TMLPU(superbee)' in sharpness and 'minmod_tvd_1d' in sharpness:
        ratio = sharpness['TMLPU(superbee)'] / max(sharpness['minmod_tvd_1d'], 1e-30)
        print(f"\n  sharpness(TMLPU superbee) / sharpness(minmod_tvd_1d) = {ratio:.2f}")
        if ratio < 0.95:
            fails.append(('TMLPU(superbee)',
                          f'less sharp than minmod ({ratio:.2f})'))

    print("\n" + "=" * 78)
    if fails:
        print(f"FAIL — {len(fails)} cases:")
        for r, why in fails:
            print(f"  {r}: {why}")
        return 1
    print("PASS — T-MLP-u: finite, conservative, LMP-bounded, "
          "≥ minmod sharpness on a step.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
