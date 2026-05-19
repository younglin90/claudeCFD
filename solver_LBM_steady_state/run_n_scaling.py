"""SCMK-LBM N-scaling study : grid-independence of outer iteration count.

Theory : baseline LBM steady convergence requires O(N^2) steps (diffusion-time scaling).
         SCMK Phase-4 should require O(1) -- O(log N) outer iterations.
         Speedup grows linearly (or faster) with N.

Cases : Kolmogorov + Channel, swept across N in {32, 48, 64, 96, 128}.
"""

import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_periodic import KolmogorovCase, build_spectral_schur
from lbm_channel import ChannelCase
from solver_scmk import solve_scmk, solve_baseline_periodic


def run_at_N(case, label, tol=1e-9, max_baseline=400000, **scmk_kw):
    print(f"  N={case.N} ", end="", flush=True)
    t0 = time.perf_counter()
    f_b, hist_b = solve_baseline_periodic(case, max_steps=max_baseline, tol=tol,
                                           check_every=1000, verbose=False)
    wall_b = time.perf_counter() - t0
    lbe_b = hist_b[-1][2]; res_b = hist_b[-1][1]; iter_b = hist_b[-1][0]
    converged_b = res_b < tol

    case2 = type(case)(*case_args(case))
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    t0 = time.perf_counter()
    f_s, hist_s = solve_scmk(case2, S_inv, tol=tol, verbose=False, **scmk_kw)
    wall_s = time.perf_counter() - t0
    lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]; iter_s = hist_s[-1][0]
    converged_s = res_s < tol

    print(f"baseline {iter_b} step ({'C' if converged_b else 'X'}, {lbe_b} LBE) | "
          f"SCMK {iter_s} outer ({'C' if converged_s else 'X'}, {lbe_s} LBE) | "
          f"LBE x{lbe_b/max(lbe_s,1):.1f} wall x{wall_b/max(wall_s,1e-9):.1f}")

    return {"N": case.N, "label": label,
            "baseline_iter": int(iter_b), "baseline_lbe": int(lbe_b),
            "baseline_wall": float(wall_b), "baseline_converged": bool(converged_b),
            "scmk_outer": int(iter_s), "scmk_lbe": int(lbe_s),
            "scmk_wall": float(wall_s), "scmk_converged": bool(converged_s),
            "speedup_lbe": float(lbe_b / max(lbe_s, 1)),
            "speedup_wall": float(wall_b / max(wall_s, 1e-9))}


def case_args(case):
    if isinstance(case, KolmogorovCase):
        return (case.N, case.nu, case.F0, case.kf)
    if isinstance(case, ChannelCase):
        return (case.N, case.nu, case.F0)
    raise NotImplementedError


def main():
    out = "results_scaling"; os.makedirs(out, exist_ok=True)
    Ns = [32, 48, 64, 96, 128]
    tol = 1e-8

    scmk_kw = dict(max_outer=200, krylov_max=10, krylov_tol=1e-3,
                   line_search_max=5, kinetic_substeps=15)

    print("Kolmogorov sweep :")
    kolmo = []
    for N in Ns:
        c = KolmogorovCase(N=N, nu=0.05, F0=2e-4, kf=1)
        kolmo.append(run_at_N(c, "kolmogorov", tol=tol, **scmk_kw))

    print("\nChannel sweep :")
    channel = []
    for N in Ns:
        c = ChannelCase(N=N, nu=0.05, F0=1e-6)
        channel.append(run_at_N(c, "channel", tol=tol, **scmk_kw))

    summary = {"tol": tol, "Ns": Ns, "kolmogorov": kolmo, "channel": channel}
    with open(f"{out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Plot scaling ----
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax_row, group, name in zip(axs, (kolmo, channel), ("Kolmogorov", "Channel")):
        Nv = [r["N"] for r in group]
        b_lbe = [r["baseline_lbe"] for r in group]
        s_lbe = [r["scmk_lbe"] for r in group]
        b_wall = [r["baseline_wall"] for r in group]
        s_wall = [r["scmk_wall"] for r in group]
        s_iter = [r["scmk_outer"] for r in group]
        b_iter = [r["baseline_iter"] for r in group]

        ax = ax_row[0]
        ax.loglog(Nv, b_lbe, "bo-", label="Baseline LBE", lw=2)
        ax.loglog(Nv, s_lbe, "rs-", label="SCMK LBE", lw=2)
        # reference slopes
        ax.loglog(Nv, [b_lbe[0] * (N / Nv[0]) ** 2 for N in Nv], "b--", alpha=0.4, lw=1, label=r"O($N^2$)")
        ax.loglog(Nv, [s_lbe[0] for _ in Nv], "r--", alpha=0.4, lw=1, label="O(1)")
        ax.set_xlabel("N (grid size)"); ax.set_ylabel("LBE evaluations")
        ax.set_title(f"{name} : LBE-call N-scaling")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)

        ax = ax_row[1]
        speedup = [b / s for b, s in zip(b_lbe, s_lbe)]
        ax.loglog(Nv, speedup, "go-", lw=2)
        ax.loglog(Nv, [speedup[0] * (N / Nv[0]) for N in Nv], "g--", alpha=0.4, lw=1, label="O(N) ref")
        ax.set_xlabel("N"); ax.set_ylabel("Speedup (LBE)")
        ax.set_title(f"{name} : SCMK speedup vs N")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}/scaling.png", dpi=120)
    plt.close(fig)
    print(f"\nPlot : {out}/scaling.png")

    print("\n========== SCALING SUMMARY ==========")
    print(f"{'Case':<14}{'N':>6}{'baseline iter':>15}{'SCMK outer':>13}{'LBE speedup':>13}{'wall speedup':>13}")
    for r in kolmo + channel:
        print(f"{r['label']:<14}{r['N']:>6}{r['baseline_iter']:>15}{r['scmk_outer']:>13}{r['speedup_lbe']:>12.1f}x{r['speedup_wall']:>12.1f}x")


if __name__ == "__main__":
    main()
