"""Aligned distance between two finite-amplitude two-material solutions whose interfaces differ
by dX (section 40-5):
    D^2 = kappa * sum_j |X_U,j - X_V,j|^2
        + sum_k int |rho_U(x) - rho_V(A(x))|^2 dx        (intensive density, aligned by labels)
        + pair norm of (w, p) differences.
A is the piecewise-linear map sending U's interfaces to V's.  Also reported: the plain
(non-aligned) L2 density distance, which is dominated by sqrt(|dX|) * jump (I4 structure).
Target: growth constant independent of the minimum occupancy.
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
vn = load("vn", "verify_nonlinear_lipschitz.py")
vt = load("vt", "verify_two_phase.py")
PNG = "results/lp_imex_moment/aligned_norm_occupancy.png"
PH = vn.PH


def make_state(N, x1, x2):
    h = 1.0 / N
    cells, I = [], np.zeros(N)
    for c in range(N):
        a, b = c * h, (c + 1) * h
        cuts = sorted({a, b} | {x for x in (x1, x2) if a < x < b})
        pcs = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            k = 0 if x1 <= 0.5 * (lo + hi) < x2 else 1
            xm = 0.5 * (lo + hi)
            rho = (1.0 + 0.2 * np.sin(2 * np.pi * xm)) if k == 0 else (0.2 + 0.03 * np.cos(2 * np.pi * xm))
            pcs.append([k, lo, hi, rho * (hi - lo), rho ** PH[k].n * (hi - lo)])
        xc = (c + 0.5) * h
        p = 10.0 * (1 + 0.05 * np.cos(2 * np.pi * xc))
        I[c] = p * sum((q[2] - q[1]) / PH[q[0]].G for q in pcs) + sum(PH[q[0]].C * q[4] for q in pcs)
        cells.append(pcs)
    w = np.array([0.5 + 0.3 * np.sin(2 * np.pi * (j + 1) * h) for j in range(N)])
    return dict(cells=[vk.piece_energy_default(p, I_, PH) for p, I_ in zip(cells, I)], I=I, w=w)


def relaxed(st, h):
    cells, I = [], []
    for c, pcs in enumerate(st["cells"]):
        pc, Ic = vk.relax_cell(pcs, PH, h, c * h)
        cells.append(pc)
        I.append(Ic)
    return cells, np.array(I)


def sampler(cells):
    xs, ks, rs = [], [], []
    for pcs in cells:
        for q in pcs:
            if q[2] > q[1]:
                xs.append(q[2]); ks.append(q[0]); rs.append(q[3] / (q[2] - q[1]))
    xs, ks, rs = np.array(xs), np.array(ks), np.array(rs)

    def f(x):
        i = np.searchsorted(xs, x, side="right")
        i = np.minimum(i, len(xs) - 1)
        return ks[i], rs[i]
    return f


def interfaces(cells):
    out = []
    for pcs_a, pcs_b in zip(cells, cells[1:] + cells[:1]):
        for a, b in zip(pcs_a[:-1], pcs_a[1:]):
            if a[0] != b[0]:
                out.append(a[2])
        if pcs_a[-1][0] != pcs_b[0][0]:
            out.append(pcs_a[-1][2] % 1.0)
    return sorted(out)


def aligned_distance(U, V, h, K, kappa=1.0, nfine=20000):
    cu, Iu = relaxed(U, h)
    cv, Iv = relaxed(V, h)
    XU, XV = interfaces(cu), interfaces(cv)
    assert len(XU) == 2 and len(XV) == 2, (XU, XV)
    x = (np.arange(nfine) + 0.5) / nfine
    A = np.interp(x, [0.0, XU[0], XU[1], 1.0], [0.0, XV[0], XV[1], 1.0])
    su, sv = sampler(cu), sampler(cv)
    ku, ru = su(x)
    kv, rv = sv(A)
    good = ku == kv
    dens = np.sqrt(np.sum((ru - rv)[good] ** 2) / nfine)
    xf = (np.arange(2_000_000) + 0.5) / 2_000_000          # resolves dX = 1e-5
    kuf, ruf = su(xf)
    kvf, rvf = sv(xf)
    plain = np.sqrt(np.mean((ruf - rvf) ** 2))
    pu = np.array([va.decode(p, I, PH) for p, I in zip(cu, Iu)])
    pv = np.array([va.decode(p, I, PH) for p, I in zip(cv, Iv)])
    m = np.array([sum(q[3] for q in pcs) for pcs in cu])
    M = 0.5 * (m + np.roll(m, -1))
    pair = np.sqrt(np.sum(M * (U["w"] - V["w"]) ** 2) + h * np.sum((pu - pv) ** 2 / K))
    dX = np.sqrt(sum((a - b) ** 2 for a, b in zip(XU, XV)))
    return np.sqrt(kappa * dX ** 2 + dens ** 2 + pair ** 2), dX, dens, plain, np.mean(good)


def run(shift, dXrel=1e-3, N=100, lam=2.0, T=0.06):
    h = 1.0 / N
    x1, x2 = 0.3 + shift * h, 0.7 + shift * h
    U = make_state(N, x1, x2)
    V = make_state(N, x1 + dXrel * h, x2 + dXrel * h)
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (10.5 - 0.02)) / 0.2)
    tau = min(lam * h / cmax, 0.5 * h / 0.8)
    nst = int(np.ceil(T / tau))
    tau = T / nst
    K = np.ones(N)
    for c, pcs in enumerate(relaxed(U, h)[0]):
        pc, Ic = pcs, None
    cu, Iu = relaxed(U, h)
    for c in range(N):
        pf, _, _ = va.isentrope_kapila(h, cu[c], PH, va.decode(cu[c], Iu[c], PH))
        e = 1e-6 * h
        K[c] = -h * (pf(h + e) - pf(h - e)) / (2 * e)
    D0, X0, d0, p0, _ = aligned_distance(U, V, h, K)
    D, Pl = [1.0], [1.0]
    occ = vn.min_occupancy(dict(cells=cu), h)
    for n in range(nst):
        U, _, _ = vk.step_kapila(U, PH, tau)
        V, _, _ = vk.step_kapila(V, PH, tau)
        Dn, Xn, dn, pn, gfrac = aligned_distance(U, V, h, K)
        D.append(Dn / D0)
        Pl.append(pn / p0)
        occ = min(occ, vn.min_occupancy(dict(cells=relaxed(U, h)[0]), h))
    return np.array(D), np.array(Pl), occ, tau * np.arange(nst + 1)


def main():
    res = {}
    for shift in (0.5, 1e-2, 1e-4, 1e-6):
        D, Pl, occ, t = run(shift)
        res[shift] = (D, Pl, occ, t)
        C = np.log(D.max()) / t[-1] if D.max() > 1 else 0.0
        print(f"shift={shift:.0e}: min occupancy={occ:.2e}  aligned max D_n/D_0={D.max():.5f} final={D[-1]:.5f} "
              f"implied C={C:.3f}  | plain L2 density ratio max={Pl.max():.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for shift, (D, Pl, occ, t) in res.items():
        ax[0].plot(t, D, label=f"min occupancy {occ:.0e}")
        ax[1].plot(t, Pl, label=f"min occupancy {occ:.0e}")
    ax[0].set_title("aligned distance (positions + aligned intensive + (w,p))")
    ax[1].set_title("plain L2 density distance (not aligned, 2e6 samples)")
    for a_ in ax:
        a_.set_xlabel("time")
        a_.set_ylabel("ratio to initial")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
