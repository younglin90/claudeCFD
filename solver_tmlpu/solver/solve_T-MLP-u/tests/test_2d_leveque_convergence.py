"""LeVeque rigid-rotation convergence study — T-MLP-u FINAL config.

Runs the paper FINAL T-MLP-u config (3-tier adaptive: cicsam_co38 sharp /
van_leer moderate / minmod very-smooth, vertex2 LSQ k=3, IDW p=6, virt-UU,
TVB M=64, thresholds 0.10 / 0.05, SSP-RK3, CFL=0.4, 2-pt face Gauss) at
several criss-cross resolutions and records:

  - TOTAL L1 error (vs analytic = initial after 1 period)
  - per-shape L1 (slot, cone, hump)
  - field range (min, max) and mass drift
  - convergence rate = -slope(log L1 vs log h), where h = 1/N

A log-log plot ``results/leveque_convergence.png`` is saved together with
a TSV table ``results/leveque_convergence.tsv`` (overwrite per run).

Each N runs in its own subprocess via ProcessPoolExecutor; wall-time is
the max of the N values, which is dominated by the largest N.

NOTE on N=200: at criss-cross 4·200² = 160 000 triangles single-case
wall is ~3-5 hours in Python, exceeding ralphex iteration budget.  The
default sweep is N ∈ {25, 50, 100}; the helper ``run_convergence`` accepts
an explicit ``n_values`` argument so a longer offline run can extend the
plot to N ∈ {25, 50, 100, 200} when wall time permits.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from _pkgshim import setup_paths
setup_paths()
from test_2d_leveque_rotation import _run_case_worker, phi0, _shape_masks, _l1


# T-MLP-u FINAL config (paper baseline — iter 26 frozen).
def _tmlpu_final_kwargs():
    return dict(
        tvd='cicsam_co38',
        tvd_smooth='van_leer',
        tvd_smooth2='minmod',
        mlp_bound=True,
        extremum_relax=True,
        tvb_M=64.0,
        smoothness_threshold=0.10,
        smoothness_threshold2=0.05,
        virtual_uu_gradient=True,
        stencil='vertex2',
        order=3,
        idw_p=6,
    )


def _run_N_worker(N):
    """Worker — runs T-MLP-u FINAL at one resolution.  Returns metrics dict."""
    setup_paths()
    from mesh import criss_cross_box

    mesh = criss_cross_box(N, L=1.0)
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    V = mesh.cell_volumes
    U0_field = phi0(x, y)
    init_max = float(np.max(U0_field))
    init_min = float(np.min(U0_field))
    mass0 = float(np.sum(U0_field * V))

    case = _run_case_worker(dict(
        case_id=f'N{N}',
        N=N,
        recon=_tmlpu_final_kwargs(),
        flux='upwind',
        integrator='forward_euler',
        n_face_quad=2,
        cfl=0.4,
        t_end=1.0,
        face_velocity_mode='analytic',
        label=f'N={N}: T-MLP-u FINAL',
    ))
    out = case['U_final']
    n_steps = int(case['n_steps'])
    wall = float(case['wall_s'])
    if n_steps < 0:
        return dict(N=N, n_cells=mesh.n_cells, n_steps=n_steps,
                    wall_s=wall, diverged=True,
                    error='FloatingPointError in _run_case_worker')

    masks = _shape_masks(x, y)
    metrics = dict(
        N=N,
        n_cells=int(mesh.n_cells),
        h=1.0 / N,
        n_steps=int(n_steps),
        wall_s=float(wall),
        diverged=False,
        L1_total=_l1(out, U0_field, V),
        L1_slot=_l1(out, U0_field, V, masks['slot']),
        L1_cone=_l1(out, U0_field, V, masks['cone']),
        L1_hump=_l1(out, U0_field, V, masks['hump']),
        phi_min=float(np.min(out)),
        phi_max=float(np.max(out)),
        over=float(np.max(out) - init_max),
        under=float(init_min - np.min(out)),
        drift=float(abs(np.sum(out * V) - mass0) / max(mass0, 1e-30)),
    )
    return metrics


def _convergence_rate(h_arr, L1_arr):
    """Least-squares slope of log L1 vs log h.  Convergence rate = slope.
    Returns NaN if fewer than 2 finite points."""
    h = np.asarray(h_arr, dtype=float)
    L1 = np.asarray(L1_arr, dtype=float)
    mask = np.isfinite(h) & np.isfinite(L1) & (L1 > 0)
    if mask.sum() < 2:
        return float('nan')
    p = np.polyfit(np.log(h[mask]), np.log(L1[mask]), 1)
    return float(p[0])


def _pairwise_rates(results):
    """Rates for adjacent N pairs: p = log(e_coarse/e_fine)/log(N_fine/N_coarse)."""
    finite = [r for r in results if not r['diverged']]
    out = []
    for coarse, fine in zip(finite[:-1], finite[1:]):
        row = {'N0': coarse['N'], 'N1': fine['N']}
        denom = np.log(float(fine['N']) / float(coarse['N']))
        for key in ('L1_total', 'L1_slot', 'L1_cone', 'L1_hump'):
            e0 = coarse[key]
            e1 = fine[key]
            if e0 > 0.0 and e1 > 0.0 and denom > 0.0:
                row[key] = float(np.log(e0 / e1) / denom)
            else:
                row[key] = float('nan')
        out.append(row)
    return out


def run_convergence(n_values=(25, 50, 100), out_dir=None):
    """Run T-MLP-u FINAL at the requested resolutions in parallel.

    Returns the list of metrics dicts (sorted by N).  Saves
    ``leveque_convergence.png`` and ``leveque_convergence.tsv`` in
    ``out_dir`` (defaults to repo ``results/``).
    """
    if out_dir is None:
        # repo_root/results/
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, '..', '..', '..'))
        out_dir = os.path.join(repo_root, 'results')
    os.makedirs(out_dir, exist_ok=True)

    n_values = sorted(set(int(N) for N in n_values))
    print(f"\n=== LeVeque convergence study — T-MLP-u FINAL config ===")
    print(f"  N values: {n_values}")
    print(f"  config: cicsam_co38 sharp / van_leer mod / minmod v.smooth, "
          f"vertex2 k=3, IDW p=6, virt-UU, TVB M=64, thresholds 0.10/0.05, "
          f"SSP-RK3, CFL=0.4, 2-pt GQ")

    n_workers = min(len(n_values), os.cpu_count() or 4)
    print(f"  launching {len(n_values)} cases on {n_workers} processes")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_run_N_worker, N): N for N in n_values}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r['diverged']:
                print(f"  [N={r['N']:4d}]  DIVERGED ({r['error']})  "
                      f"wall={r['wall_s']:.1f} s")
            else:
                print(f"  [N={r['N']:4d}]  cells={r['n_cells']:6d}  "
                      f"steps={r['n_steps']:5d}  wall={r['wall_s']:7.1f} s  "
                      f"L1_total={r['L1_total']:.5f}  "
                      f"range=[{r['phi_min']:+.4f},{r['phi_max']:+.4f}]  "
                      f"drift={r['drift']:.2e}")
    print(f"  parallel wall = {time.time() - t0:.1f} s")
    results.sort(key=lambda d: d['N'])

    # ─── Convergence rates (log L1 vs log h, h = 1/N) ─────────────────────
    h_arr = [r['h'] for r in results if not r['diverged']]
    rates = {}
    for key in ('L1_total', 'L1_slot', 'L1_cone', 'L1_hump'):
        L1_arr = [r[key] for r in results if not r['diverged']]
        rates[key] = _convergence_rate(h_arr, L1_arr)
    pair_rates = _pairwise_rates(results)

    # ─── TSV table ────────────────────────────────────────────────────────
    tsv_path = os.path.join(out_dir, 'leveque_convergence.tsv')
    with open(tsv_path, 'w') as f:
        f.write('N\tn_cells\th\tn_steps\twall_s\tTOTAL\tslot\tcone\t'
                'hump\trange_min\trange_max\tover\tunder\tdrift\n')
        for r in results:
            if r['diverged']:
                continue
            f.write(f"{r['N']}\t{r['n_cells']}\t{r['h']:.6e}\t{r['n_steps']}\t"
                    f"{r['wall_s']:.2f}\t{r['L1_total']:.6e}\t"
                    f"{r['L1_slot']:.6e}\t{r['L1_cone']:.6e}\t"
                    f"{r['L1_hump']:.6e}\t{r['phi_min']:.6e}\t"
                    f"{r['phi_max']:.6e}\t{r['over']:.6e}\t"
                    f"{r['under']:.6e}\t{r['drift']:.6e}\n")
        f.write('\n# adjacent pair rates p = log(e_N0/e_N1)/log(N1/N0)\n')
        for row in pair_rates:
            f.write(f"# rate_{row['N0']}_{row['N1']}\t"
                    f"TOTAL={row['L1_total']:.3f}\t"
                    f"slot={row['L1_slot']:.3f}\t"
                    f"cone={row['L1_cone']:.3f}\t"
                    f"hump={row['L1_hump']:.3f}\n")
        f.write('\n# convergence rates (L1 ~ h^p)\n')
        for key, p in rates.items():
            f.write(f"# rate_{key}\t{p:.3f}\n")
    print(f"  TSV saved: {tsv_path}")

    # ─── Console summary ──────────────────────────────────────────────────
    print("\n  ──── L1 error vs N (T-MLP-u FINAL) ────")
    print("  N      cells     steps   wall(s)    TOTAL       slot       cone       hump")
    for r in results:
        if r['diverged']:
            print(f"  {r['N']:<6d}                   DIVERGED ({r.get('error','')})")
            continue
        print(f"  {r['N']:<6d} {r['n_cells']:<9d} {r['n_steps']:<7d} "
              f"{r['wall_s']:7.1f}  {r['L1_total']:.5e} {r['L1_slot']:.5e} "
              f"{r['L1_cone']:.5e} {r['L1_hump']:.5e}")
    print("\n  ──── convergence rate (slope of log L1 vs log h) ────")
    for key, p in rates.items():
        print(f"  {key:12s}  rate = {p:.3f}")
    if pair_rates:
        print("\n  ──── adjacent pair rates p = log(e_N0/e_N1)/log(N1/N0) ────")
        for row in pair_rates:
            print(f"  N={row['N0']:d}->{row['N1']:d}  "
                  f"TOTAL={row['L1_total']:.3f}  slot={row['L1_slot']:.3f}  "
                  f"cone={row['L1_cone']:.3f}  hump={row['L1_hump']:.3f}")

    # ─── Log-log plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.0))
    finite = [r for r in results if not r['diverged']]
    if finite:
        h = np.array([r['h'] for r in finite])
        for key, color, marker in [
            ('L1_total', 'tab:blue', 'o'),
            ('L1_slot', 'tab:red', 's'),
            ('L1_cone', 'tab:green', '^'),
            ('L1_hump', 'tab:purple', 'D'),
        ]:
            L1 = np.array([r[key] for r in finite])
            label = f"{key.replace('L1_','')} (rate {rates[key]:.2f})"
            ax.loglog(h, L1, marker=marker, color=color, label=label, lw=1.4)
        # reference slopes 1 and 2
        h_ref = np.array([h.min(), h.max()])
        L1_ref_pivot = max(rates['L1_total'], 0.1) if False else \
                       float([r['L1_total'] for r in finite][-1])  # last finest
        # pin reference lines through the finest-N total point
        h_fine = h[-1]; L1_fine = L1_ref_pivot
        ax.loglog(h_ref, L1_fine * (h_ref / h_fine) ** 1.0, 'k--', lw=0.8,
                  alpha=0.5, label='slope 1 (ref)')
        ax.loglog(h_ref, L1_fine * (h_ref / h_fine) ** 2.0, 'k:', lw=0.8,
                  alpha=0.5, label='slope 2 (ref)')
    ax.set_xlabel('h = 1/N')
    ax.set_ylabel('L1 error')
    ax.set_title('LeVeque rotation — T-MLP-u FINAL convergence\n'
                 '3-tier (cicsam_co38 / van_leer / minmod) + LMP, vertex2 k=3, RK3, CFL=0.4',
                 fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    png_path = os.path.join(out_dir, 'leveque_convergence.png')
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {png_path}")

    return results, rates


def main():
    # Default sweep — 3 points sufficient for slope; keeps wall under ~40 min.
    # Override via CLI: python test_2d_leveque_convergence.py 25 50 100 200
    if len(sys.argv) > 1:
        n_values = [int(a) for a in sys.argv[1:]]
    else:
        n_values = [25, 50, 100]
    run_convergence(n_values)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
