"""Theorem L3-F, item (v), traced on the actual operator (section 62): linear acoustic one-step map of the
interface-fitted scheme at a rest state (remap is the identity to first order at rest), with the real zone
geometry (interface occupancy theta) and the real curvature weights of the force correction
(_curv_weights).  Reports sup_{n<=NPOW} ||G^n|| in the energy norm N^2 = sum M w^2 + sum (l/K) p^2 for
correction off / lagged / secant (raw stencil) / fullonly / fullsym, versus occupancy and acoustic CFL."""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fs", os.path.join(HERE, "fitted_scheme.py"))
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
PNG = "results/lp_imex_moment/l3f_band.png"
PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
N = 40
NPOW = 4000
MODES = ("off", "lagged", "secant", "fullonly", "fullsym")


def operator(theta, lam, mode):
    h = 1.0 / N
    X1, X2 = 0.3 + theta * h, 0.7 + theta * h
    st = fs.build(N, PH, lambda x: 0 if X1 <= x < X2 else 1, lambda k, x: (1.0 + 0 * x) if k == 0 else 0.2 + 0 * x,
                  lambda x: 10.0 + 0 * x, lambda x: 0 * x, [X1, X2])
    Z = st.zones
    B = len(Z)
    m = np.array([z[3] for z in Z])
    l = np.array([z[2] - z[1] for z in Z])
    M = 0.5 * (m + np.roll(m, 1))
    K = np.empty(B)
    for i, z in enumerate(Z):
        pf, Ef, l0 = fs._isentrope(z, PH[z[0]])
        e = 1e-6 * l0
        K[i] = -l0 * (pf(l0 + e) - pf(l0 - e)) / (2 * e)      # isentropic bulk modulus of the zone
    # curvature part of the force correction: corr = Cc p,  corr_z = (l_z^2/12) w . (p_nb - p_z)
    Cc = np.zeros((B, B))
    full = np.abs(l - h) < 1e-9 * h
    if mode != "off":
        for i in range(B):
            w8 = fs._curv_weights(Z, i, full if mode == "fullsym" else None)
            if w8 is None or (mode == "fullonly" and not (full[i] and all(full[w8[0]]))):
                continue
            nbs, cw = w8
            Cc[i, nbs] += cw * l[i] ** 2 / 12
            Cc[i, i] -= cw.sum() * l[i] ** 2 / 12
    if mode == "fullsym":
        H = l / K
        A = Cc - np.diag(np.diag(Cc))
        A = 0.5 * (A + (A.T * H[None, :]) / H[:, None])
        Cc = A - np.diag(A.sum(axis=1))
    cmax = np.sqrt(np.max(K / (m / l)))
    tau = lam * h / cmax
    Dw = np.zeros((B, B))     # zone z: w_{z+1} - w_z
    for z in range(B):
        Dw[z, (z + 1) % B] += 1
        Dw[z, z] -= 1
    Dp = -Dw.T                # node b: p_b - p_{b-1}
    I = np.eye(B)
    O = np.zeros((B, B))
    # unknowns (w, p); implicit midpoint on  M w' = -Dp pf,  p' = -(K/l) Dw w,  pf = p_mid - corr
    Aw_p = -np.diag(1 / M) @ Dp
    Ap_w = -np.diag(K / l) @ Dw
    S_mid = I if mode in ("off", "lagged") else I - Cc
    S_old = -Cc if mode == "lagged" else O
    L = np.block([[I, -0.5 * tau * Aw_p @ S_mid], [-0.5 * tau * Ap_w, I]])
    R = np.block([[I, 0.5 * tau * Aw_p @ S_mid + tau * Aw_p @ S_old], [0.5 * tau * Ap_w, I]])
    G = np.linalg.solve(L, R)
    Wh = np.sqrt(np.concatenate([M, l / K]))
    Gs = Wh[:, None] * G / Wh[None, :]                    # G in the energy norm
    return Gs


def sup_power(Gs):
    P = np.eye(len(Gs))
    best = 1.0
    for n in range(NPOW):
        P = Gs @ P
        if n % 20 == 19 or n < 20:
            best = max(best, np.linalg.norm(P, 2))
            if best > 1e12:
                break
    return best, max(abs(np.linalg.eigvals(Gs)))


def main():
    thetas = [0.9, 0.5, 1e-1, 1e-3, 1e-5, 1e-7]
    lams = [0.5, 2.0, 9.6]
    rows = {}
    for mode in MODES:
        for lam in lams:
            for th in thetas:
                sp_, rho = sup_power(operator(th, lam, mode))
                rows[(mode, lam, th)] = (sp_, rho)
                print(f"{mode:6s} CFL={lam:4.1f} occupancy={th:.0e}: sup_n<= {NPOW} ||G^n|| = {sp_:.4e}  "
                      f"spectral radius = {rho:.10f}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, mk in zip(MODES, "s^ov*"):
        for lam, ls in zip(lams, (":", "--", "-")):
            ax[0].loglog(thetas, [rows[(mode, lam, t)][0] for t in thetas], mk + ls, label=f"{mode}, CFL {lam}")
            ax[1].semilogx(thetas, [rows[(mode, lam, t)][1] - 1 for t in thetas], mk + ls, label=f"{mode}, CFL {lam}")
    ax[0].set_xlabel("interface occupancy theta")
    ax[0].set_ylabel(f"sup over n <= {NPOW} of ||G^n|| (energy norm)")
    ax[0].set_title("linear acoustic map, real geometry and correction weights")
    ax[1].set_xlabel("interface occupancy theta")
    ax[1].set_ylabel("spectral radius - 1")
    ax[1].set_yscale("symlog", linthresh=1e-6)
    ax[0].set_ylim(0.9, 1e13)
    ax[1].set_title("spectral radius of the one-step map")
    ax[0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
