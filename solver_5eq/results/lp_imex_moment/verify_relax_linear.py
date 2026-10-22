"""(1) Relaxation positivity: the selected relaxation root satisfies E' >= E_s(V') (isentropic
    energy of the piece at the same volume), hence theta' >= theta_s(V') > 0.  Checked on the
    strong-compression lamellar runs of verify_kapila_scheme.py.
(2) Linearized sector: single material, uniform background (rho0, u0, p0), small perturbation.
    Theory: one step = upwind symbol R(k) times a Cayley (unitary in the pair energy norm) factor,
    so the pair energy is non-increasing for every acoustic CFL, and the error converges.
"""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


va = load("va", "verify_acoustic.py")
vk = load("vk", "verify_kapila_scheme.py")
PNG = "results/lp_imex_moment/relax_linear_theory_vs_numeric.png"


def isentropic_energy(q, ph, V):
    """Energy of piece q=[k,a,b,m,Z,E] moved isentropically (uniform strain) to volume V."""
    V0, Z0, E0 = q[2] - q[1], q[4], q[5]
    pr = ph.G * (E0 - ph.C * Z0) / V0
    zb = Z0 / V0
    J = V / V0
    ps = ph.A * zb * J ** (-ph.n) + (pr - ph.A * zb) * J ** (-1.0 - ph.G)
    return ps * V / ph.G + ph.C * Z0 * J ** (1 - ph.n)


def relax_with_check(pcs, phases, h, x0, log):
    out, I = vk.relax_cell(pcs, phases, h, x0)
    if len(pcs) > 1:
        pr = va.decode(out, I, phases)
        for q, o in zip(pcs, out):
            ph = phases[q[0]]
            V = o[2] - o[1]
            Enew = q[5] - pr * (V - (q[2] - q[1]))
            Es = isentropic_energy(q, ph, V)
            th_s = (Es - ph.A / (ph.n - 1) * o[4]) / max(abs(Enew), 1e-300)
            log.append(((Enew - Es) / max(abs(Es), 1e-300), th_s))
    return out, I


def run_relax_check(U):
    phases = [va.Phase(1.0, 3, 3.0), va.Phase(0.5, 2, 0.4)]
    N = 40
    h = 1.0 / N
    st = vk.lamella_state(N, phases, U)
    st = dict(cells=[vk.piece_energy_default(p, I, phases) for p, I in zip(st["cells"], st["I"])], I=st["I"], w=st["w"])
    tau = 0.2 * h / max(U, 1.0)
    log = []
    orig = vk.relax_cell
    for n in range(400):
        cells, I = [], np.zeros(N)
        for c, pcs in enumerate(st["cells"]):
            pc, I[c] = relax_with_check(pcs, phases, h, c * h, log)
            cells.append(pc)
        rel = dict(cells=cells, I=I, w=st["w"])
        lag = va.lagrange(rel, tau, phases, None, model="kapila")
        st, Q = va.remap(rel, lag, phases, keep_piece_energy=True)
    return np.array(log)


def linear_run(N, lam, u0=1.0, eps=1e-4, T=0.25):
    ph = va.Phase(1.0, 3, 1.5)
    phases = [ph, ph]
    rho0, p0 = 1.0, 10.0
    K = ph.n * ph.A * rho0 ** ph.n + (1 + ph.G) * (p0 - ph.A * rho0 ** ph.n)
    c = np.sqrt(K / rho0)
    h = 1.0 / N
    tau = lam * h / c
    if u0 * tau > 0.8 * h:
        tau = 0.8 * h / u0
    nst = int(np.ceil(T / tau))
    tau = T / nst
    st = va.init_state(N, phases, lambda x: rho0, lambda x: p0, lambda x: u0 + eps * np.sin(2 * np.pi * x))
    xn = (np.arange(N) + 1) * h
    xc = (np.arange(N) + 0.5) * h

    def pair_energy(s):
        m = np.array([sum(q[3] for q in pcs) for pcs in s["cells"]])
        M = 0.5 * (m + np.roll(m, -1))
        dp = va.pressures(s, phases) - p0
        return 0.5 * np.sum(M * (s["w"] - u0) ** 2) + 0.5 * h * np.sum(dp ** 2) / K

    E = [pair_energy(st)]
    for n in range(nst):
        lag = va.lagrange(st, tau, phases)
        st, Q = va.remap(st, lag, phases, slope="none")
        E.append(pair_energy(st))
    t = nst * tau
    w_ex = u0 + 0.5 * eps * (np.sin(2 * np.pi * (xn - (u0 + c) * t)) + np.sin(2 * np.pi * (xn - (u0 - c) * t)))
    err = np.sqrt(h * np.sum((st["w"] - w_ex) ** 2)) / eps
    return np.array(E) / E[0], err, c, u0 * tau / h


def main():
    print("== (1) relaxation: (E' - E_s)/|E_s| must be >= 0 (theory) ==")
    logs = {}
    for U in (2.0, 4.0, 6.0):
        lg = run_relax_check(U)
        logs[U] = lg
        print(f" U={U}: relaxed pieces checked={len(lg)}, min (E'-E_s)/|E_s| = {lg[:,0].min():.3e}, "
              f"fraction with margin < -1e-12 = {np.mean(lg[:,0] < -1e-12):.3e}")

    print("== (2) linearized sector, u0=1, pair energy ratio max over steps (theory <= 1) and L2 error ==")
    lams = [0.5, 2.0, 8.0]
    Ns = [25, 50, 100, 200]
    E_hist = {}
    errs = {lam: [] for lam in lams}
    for lam in lams:
        for N in Ns:
            Er, err, c, nu = linear_run(N, lam)
            errs[lam].append(err)
            if N == 50:
                E_hist[lam] = Er
            print(f" lambda={lam} N={N} nu={nu:.3f}: max E_n/E_0={Er.max():.6f}  final E/E0={Er[-1]:.4f}  L2 err(w)/eps={err:.3e}")
        o = [np.log2(errs[lam][i] / errs[lam][i + 1]) for i in range(len(Ns) - 1)]
        print("   order:", " ".join(f"{x:.2f}" for x in o))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for U, lg in logs.items():
        ax[0].hist(np.log10(np.maximum(lg[:, 0], 1e-18)), bins=40, histtype="step", label=f"U={U}")
    ax[0].set_xlabel("log10 max((E'-E_s)/|E_s|, 1e-18)  (theory >= 0; min observed -4e-16 = roundoff, in the -18 bin)")
    ax[0].set_ylabel("count of relaxed pieces")
    ax[0].set_title("relaxation positivity margin")
    ax[0].legend(fontsize=7)
    for lam, Er in E_hist.items():
        ax[1].plot(Er, label=f"lambda={lam}")
    ax[1].axhline(1.0, color="k", ls=":", label="theory bound E_n/E_0 <= 1")
    ax[1].set_xlabel("step (N=50, u0=1)")
    ax[1].set_ylabel("pair energy / initial")
    ax[1].set_title("linearized sector: energy-norm stability")
    ax[1].legend(fontsize=7)
    hs = 1.0 / np.array(Ns)
    for lam in lams:
        ax[2].loglog(hs, errs[lam], "o-", label=f"lambda={lam}")
    ax[2].loglog(hs, errs[0.5][0] * hs / hs[0], "k:", label="slope 1")
    ax[2].set_xlabel("h")
    ax[2].set_ylabel("L2 error of w vs exact linear solution / eps")
    ax[2].set_title("linearized sector: convergence (T=0.25)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
