"""Section 64: numerical value of the stability constant C_B of theorem L3-F (no-correction configuration,
constants of section 62) on the compatible pulse test, and the measured growth rate of a perturbation.

 C_B = (kappa_K + 1 + 2 Ma^2 + 2 Gamma_max u_max / c_min) ||w_x||_inf
       + 2 u_max (||p_x||/K_min + ||rho_x||/rho_min) + 1 ;  position term 2 sqrt(2 Kbar), Kbar := 1/T (free weight)
 kappa_K = max_k max(n_k, 1 + Gamma_k) + 1,  Ma^2 = rho_max u_max^2 / K_min.
Every quantity is a zone/node value of the solution, none involves the occupancy or the CFL number.
The measured rate is max_t log(N(t)/N(0))/t for a 1e-7 velocity perturbation in the full energy norm,
for interface occupancies 1e-2 ... 1e-6.  The perturbation is smooth (1e-7 sin 2 pi x): a grid-scale one
switches minmod/limiter branches, which the linearised bound does not describe."""
import importlib.util
import os
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EPS_P = 1e-7   # above the solver noise floor (~5e-8 absolute in this norm: Newton tolerance, limiter bisection)
PNG = "results/lp_imex_moment/cb_constant.png"


def load():
    spec = importlib.util.spec_from_file_location("vp", os.path.join(HERE, "verify_pulse_interface.py"))
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    vp.fs.PARABOLIC[0] = "--linear" not in os.sys.argv
    return vp


def terms(vp, st, N):
    fs, PH = vp.fs, vp.PH
    h = 1.0 / N
    Z = st.zones
    l = np.array([z[2] - z[1] for z in Z])
    rho = np.array([z[3] for z in Z]) / l
    p = np.array([fs.zone_p(z, PH[z[0]]) for z in Z])
    K = np.empty(len(Z))
    for i, z in enumerate(Z):
        pf, Ef, l0 = fs._isentrope(z, PH[z[0]])
        K[i] = -l0 * (pf(l0 * (1 + 1e-7)) - pf(l0 * (1 - 1e-7))) / (2e-7 * l0)
    c = np.sqrt(K / rho)
    w = st.w
    full = np.array([st.isface[b] and st.isface[(b + 1) % len(Z)] for b in range(len(Z))])
    same = np.array([Z[i][0] == Z[(i + 1) % len(Z)][0] for i in range(len(Z))])
    # gradients inside each material (one zone step, same material only)
    dx = np.maximum(l, h)
    wx = np.abs(np.roll(w, -1) - w) / l                       # zone strain rate
    px = np.where(same, np.abs(np.roll(p, -1) - p) / dx, 0.0)
    rx = np.where(same, np.abs(np.roll(rho, -1) - rho) / dx, 0.0)
    kK = max(max(ph.n, 1 + ph.G) for ph in PH) + 1
    Gm = max(ph.G for ph in PH)
    umax, Kmin, rmin, rmax, cmin = np.abs(w).max(), K.min(), rho.min(), rho.max(), c.min()
    Ma2 = rmax * umax ** 2 / Kmin
    wxi = wx[full].max()
    Kbar = 1.0 / vp.T
    parts = {
        "AVF kappa_K|w_x|": kK * wxi,
        "remap (iii)": 2 * umax * (px.max() / Kmin + rx.max() / rmin),
        "velocity repartition (iv)": (1 + 2 * Ma2) * wxi,
        "G heat (v)": 2 * Gm * umax / cmin * wxi,
        "position 2 sqrt(2 Kbar)": 2 * np.sqrt(2 * Kbar),
        "Strang/O(tau) 1": 1.0,
    }
    return parts, dict(kappa_K=kK, Ma2=Ma2, wx=wxi, umax=umax, cmin=cmin, Kmin=Kmin, rmin=rmin)


def growth(args):
    occ, N = args
    vp = load()
    vp.X1, vp.X2 = 0.3 + occ / N, 0.7 + occ / N
    st0 = vp.fs.build(N, vp.PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [vp.X1, vp.X2])
    tau, nst = vp.tau_of(N)
    a = st0
    b = vp.fs.build(N, vp.PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [vp.X1, vp.X2])
    b.w = b.w + EPS_P * np.sin(2 * np.pi * np.array([z[1] for z in b.zones]))
    h = 1.0 / N

    def nrm(sa, sb):
        # perturbation energy norm: kinetic (node masses) + acoustic (l/K dp^2); structures must match
        if len(sa.zones) != len(sb.zones):
            return np.nan
        m = np.array([z[3] for z in sa.zones])
        M = 0.5 * (m + np.roll(m, 1))
        e2 = 0.5 * np.sum(M * (sa.w - sb.w) ** 2)
        for za, zb in zip(sa.zones, sb.zones):
            ph = vp.PH[za[0]]
            pf, Ef, l0 = vp.fs._isentrope(za, ph)
            K = -l0 * (pf(l0 * (1 + 1e-7)) - pf(l0 * (1 - 1e-7))) / (2e-7 * l0)
            e2 += 0.5 * (za[2] - za[1]) / K * (vp.fs.zone_p(za, ph) - vp.fs.zone_p(zb, ph)) ** 2
        return np.sqrt(e2)
    n0 = nrm(a, b)
    rates, cb = [], []
    for n in range(nst):
        a, _ = vp.fs.step(a, tau, vp.PH, N)
        b, _ = vp.fs.step(b, tau, vp.PH, N)
        t = (n + 1) * tau
        r = nrm(a, b)
        if np.isfinite(r):
            rates.append(np.log(r / n0) / t)
        if n % max(1, nst // 6) == 0:
            cb.append(sum(terms(vp, a, N)[0].values()))
    return occ, np.array(rates), max(cb)


def main():
    vp = load()
    N = 200
    st = vp.fs.build(N, vp.PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [vp.X1, vp.X2])
    parts, info = terms(vp, st, N)
    print("material/solution data:", {k: float(f"{v:.4g}") for k, v in info.items()})
    for k, v in parts.items():
        print(f"  {k:28s} {v:10.4f}")
    CB = sum(parts.values())
    print(f"  C_B (t=0) = {CB:.3f}   e^(C_B T) = {np.exp(CB * vp.T):.4f}  (T = {vp.T})")
    with Pool(4) as pool:
        res = pool.map(growth, [(o, N) for o in (1e-2, 1e-4, 1e-6)])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(range(len(parts)), list(parts.values()), color="tab:blue")
    ax[0].set_xticks(range(len(parts)))
    ax[0].set_xticklabels(list(parts), rotation=30, ha="right", fontsize=7)
    ax[0].set_ylabel("contribution to C_B (1/time)")
    ax[0].set_title(f"C_B = {CB:.2f} on the pulse test (occupancy-free terms)")
    for occ, r, cbmax in res:
        print(f"occupancy {occ:.0e}: max measured rate {r.max():.3f}  (late-time rate {r[-1]:.3f}),  max_t C_B {cbmax:.3f}")
        ax[1].plot(np.linspace(vp.T / len(r), vp.T, len(r)), r, label=f"measured rate, occupancy {occ:.0e}")
    ax[1].axhline(max(c for _, _, c in res), color="k", ls="--", label="C_B (max over t)")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("log(N(t)/N(0)) / t")
    ax[1].legend(fontsize=7)
    ax[1].set_title("measured perturbation growth rate vs bound")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
