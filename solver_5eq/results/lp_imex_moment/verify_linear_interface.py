"""Linearized two-material sector (variable coefficients): acoustic pulse crossing a moving
material interface (water-like | gas-like) on a uniform background u0, p0.
Theory (section 33): the linearized AVF stage is unitary in the variable-coefficient pair norm
  ||(w, dp)||^2 = 1/2 sum M_f w'^2 + 1/2 sum_i L_i dp_i^2 / K_i,   K_i = -L dp/dL (cell isentrope);
the remap is a positive averaging. Measured: max_n E_n / E_0 for both closures and several lambda.
"""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


va = load("va", "verify_acoustic.py")
vk = load("vk", "verify_kapila_scheme.py")
PNG = "results/lp_imex_moment/linear_interface_energy.png"


def cell_K(pcs, I, phases, model, h):
    p0 = va.decode([q[:5] for q in pcs], I, phases)
    if model == "kapila":
        pf, _, _ = va.isentrope_kapila(h, [q[:5] for q in pcs], phases, p0)
    else:
        pf, _ = va.isentrope(h, [q[:5] for q in pcs], phases, p0)
    e = 1e-6 * h
    return -h * (pf(h + e) - pf(h - e)) / (2 * e)


def run(model, lam, N=100, eps=1e-4, u0=0.5, p0=10.0, nst=None):
    ph = [va.Phase(1.0, 3, 1.5), va.Phase(0.5, 2, 0.4)]
    h = 1.0 / N
    mat = lambda x: 0 if 0.3 <= x < 0.7 else 1
    st0 = va.init_state(N, ph, lambda x: 1.0 if mat(x) == 0 else 0.2, lambda x: p0, lambda x: u0, mat_fun=mat)
    pulse = lambda x: eps * np.exp(-((x - 0.15) / 0.04) ** 2)
    st = va.init_state(N, ph, lambda x: 1.0 if mat(x) == 0 else 0.2, lambda x: p0, lambda x: u0 + pulse(x), mat_fun=mat)
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (p0 - 0.02)) / 0.2)
    tau = min(lam * h / cmax, 0.8 * h / u0)
    nst = nst or int(np.ceil(0.4 / tau))
    if model == "kapila":
        st = dict(cells=[vk.piece_energy_default(p, I, ph) for p, I in zip(st["cells"], st["I"])], I=st["I"], w=st["w"])
        bg = dict(cells=[vk.piece_energy_default(p, I, ph) for p, I in zip(st0["cells"], st0["I"])], I=st0["I"], w=st0["w"])
    else:
        bg = st0

    def energy(s, b):
        if model == "kapila":
            cs = [vk.relax_cell(pcs, ph, h, c * h) for c, pcs in enumerate(s["cells"])]
            cb = [vk.relax_cell(pcs, ph, h, c * h) for c, pcs in enumerate(b["cells"])]
            p = np.array([va.decode(c_[0], c_[1], ph) for c_ in cs])
            pb = np.array([va.decode(c_[0], c_[1], ph) for c_ in cb])
            K = np.array([cell_K(c_[0], c_[1], ph, model, h) for c_ in cb])
        else:
            p, pb = va.pressures(s, ph), va.pressures(b, ph)
            K = np.array([cell_K(pcs, I, ph, model, h) for pcs, I in zip(b["cells"], b["I"])])
        m = np.array([sum(q[3] for q in pcs) for pcs in b["cells"]])
        M = 0.5 * (m + np.roll(m, -1))
        return 0.5 * np.sum(M * (s["w"] - b["w"]) ** 2) + 0.5 * h * np.sum((p - pb) ** 2 / K)

    E = [energy(st, bg)]
    for n in range(nst):
        if model == "kapila":
            st, Q, _ = vk.step_kapila(st, ph, tau)
            bg, Qb, _ = vk.step_kapila(bg, ph, tau)
        else:
            st, Q = va.remap(st, va.lagrange(st, tau, ph), ph)
            bg, Qb = va.remap(bg, va.lagrange(bg, tau, ph), ph)
        E.append(energy(st, bg))
    return np.array(E) / E[0]


def main():
    hist = {}
    for model in ("allaire", "kapila"):
        for lam in (0.5, 2.0, 8.0):
            Er = run(model, lam)
            hist[(model, lam)] = Er
            print(f"{model:8s} lambda={lam}: max E_n/E_0 = {Er.max():.6f}  final = {Er[-1]:.4f}  steps={len(Er)-1}")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for (model, lam), Er in hist.items():
        ax.plot(np.linspace(0, 1, len(Er)), Er, "-" if model == "kapila" else "--", label=f"{model}, lambda={lam}")
    ax.axhline(1.0, color="k", ls=":", label="bound 1")
    ax.set_xlabel("t / T")
    ax.set_ylabel("perturbation pair energy / initial")
    ax.set_title("linearized two-material sector (pulse crosses moving interfaces)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
