"""Section 75: quick feasibility tests of the four open items of section 74.
 T1 (O1) impulse response of the REAL scheme at the interface: does the remap-coupled scheme ring (grid-scale
        trapped mode) or stay bounded as N grows?  S1 = sum_n |trace_n| normalised by the continuum value 1.
 T2 (O2) the same metric over a fine scan of nu = u tau / h (rational and irrational-like values).
 T3 (O3) Richardson orders of the compatible pulse for a 4x stronger background gradient (variable coefficients).
 T4 (O4) trapped eigenmode of the dissipation-free frozen-theta interface: location, mode shape, dependence on
        impedance ratio and theta (null vector of the 5x5 matrix of verify_interface_ukl.py).
Run: python3 results/lp_imex_moment/verify_open_items.py [t1|t2|t3|t4|all]"""
import importlib.util
import os
import sys
from multiprocessing import Pool

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


PNG = "results/lp_imex_moment/open_items.png"
tm = load("tm", "verify_trapped_mode.py")
fs, PH, CG, P0 = tm.fs, tm.PH, tm.CG, tm.P0
ZL, ZR = 0.2 * CG, np.sqrt(25.5 * 1.0)          # impedances left (gas-like) and right (water-like) of interface 1


def iface(st):
    return [b for b in range(len(st.zones)) if (not st.isface[b]) and st.zones[b][0] == 0][0]


def impulse(args):
    N, u0, theta, eps, W, cfl = args
    h = 1.0 / N
    tau = cfl * h / CG
    st = tm.make(N, u0, theta, 0)
    b = iface(st)
    m = np.array([z[3] for z in st.zones])
    MI = 0.5 * (m[b - 1] + m[b])
    st.w[b] += eps
    tr = []
    for n in range(W):
        st, _ = fs.step(st, tau, PH, N)
        b = iface(st)
        pz = lambda i: fs.zone_p(st.zones[i % len(st.zones)], PH[st.zones[i % len(st.zones)][0]])
        tr.append((st.w[b] - u0, pz(b - 1) - P0, pz(b) - P0))
    tr = np.array(tr)
    norm = (ZL + ZR) * tau / (MI * eps)
    s1 = np.abs(tr[:, 0]).sum() * norm
    half = W // 2
    tail = np.abs(tr[half:, 0]).sum() / max(np.abs(tr[:half, 0]).sum(), 1e-300)
    sp = max(np.abs(tr[:, 1]).sum(), np.abs(tr[:, 2]).sum()) * tau / (MI * eps) * 1.0
    return s1, tail, sp, u0 * tau / h


def t1(pool):
    Ns = [100, 200, 400, 800]
    cfg = [(0.0, 0.5), (0.3, 0.5), (1.0, 0.5), (3.0, 0.5), (1.0, 0.05)]
    res = {}
    jobs = [(N, u0, th, 1e-4, max(30, int(0.35 * N)), 1.0) for (u0, th) in cfg for N in Ns]
    out = pool.map(impulse, jobs)
    k = 0
    for (u0, th) in cfg:
        res[(u0, th)] = out[k:k + len(Ns)]
        k += len(Ns)
        print(f"T1 u0={u0} theta={th} (nu={res[(u0, th)][0][3]:.3f}): S1_w for N={Ns}: "
              f"{np.round([r[0] for r in res[(u0, th)]], 3)}  tail ratio {np.round([r[1] for r in res[(u0, th)]], 3)}",
              flush=True)
    a = impulse((200, 1.0, 0.5, 1e-3, 70, 1.0))
    b = impulse((200, 1.0, 0.5, 1e-5, 70, 1.0))
    print(f"T1 linearity (N=200, u0=1): S1 for eps 1e-3 / 1e-4 / 1e-5: {a[0]:.4f} {res[(1.0, 0.5)][1][0]:.4f} {b[0]:.4f}",
          flush=True)
    return Ns, res


def t2(pool):
    u0s = np.concatenate([np.linspace(0.1, 6.0, 24), [CG * 0.125, CG * 0.25, CG / 3.0, CG * 0.5, CG * (np.sqrt(5) - 1) / 2 * 0.3]])
    u0s = np.sort(u0s)
    jobs = [(200, u, 0.37, 1e-4, 70, 1.0) for u in u0s]
    out = pool.map(impulse, jobs)
    nu = np.array([o[3] for o in out])
    s1 = np.array([o[0] for o in out])
    print("T2 nu scan (N=200, theta=0.37): nu -> S1_w")
    print("   " + ", ".join(f"{a:.3f}:{b:.3f}" for a, b in zip(nu, s1)), flush=True)
    return nu, s1


def t3():
    vp = load("vp", "verify_pulse_interface.py")
    vp.fs.PARABOLIC[0] = True
    rows = {}
    for eps in (0.02, 0.08):
        vp.EPS = eps
        sol = {}
        for N in (100, 200, 400, 800):
            sol[N] = vp.run_fitted(N)[:3]
        d = []
        for N in (100, 200, 400):
            rho, p, w = sol[N]
            rf, pf, wf = sol[2 * N]
            h = 1.0 / N
            e = [np.sqrt(h * np.sum((rho - rf.reshape(-1, 2).mean(axis=1)) ** 2)),
                 np.sqrt(h * np.sum((p - pf.reshape(-1, 2).mean(axis=1)) ** 2)),
                 np.sqrt(h * np.sum((w - wf[::2]) ** 2))]
            d.append(e)
        d = np.array(d)
        o = np.log2(d[:-1] / d[1:])
        rows[eps] = (d, o)
        print(f"T3 EPS={eps}: d_N (rho,p,w) for N=100,200,400: {np.array2string(d, precision=3)}\n"
              f"   orders d100/d200, d200/d400 (rho,p,w): {np.array2string(o, precision=2)}", flush=True)
    return rows


def matrix(z, theta, h, tau, rL, cL, rR, cR):
    ukl = load("ukl", "verify_interface_ukl.py")
    sig = (z - 1) / (z + 1)
    s2 = 2 * sig / tau
    KL, KR = rL * cL ** 2, rR * cR ** 2
    kL = ukl.roots(sig, cL * tau / h)[1]
    kR = ukl.roots(sig, cR * tau / h)[0]
    pL = -(kL - 1) * tau * KL / (2 * sig * h)
    pR = -(kR - 1) * tau * KR / (2 * sig * h)
    M1L = rL * h * (1 + theta) / 2
    MI = (rL * theta * h + rR * (1 - theta) * h) / 2
    M1R = rR * h * (2 - theta) / 2
    CL, CR = theta * h / KL, (1 - theta) * h / KR
    A = np.zeros((5, 5), complex)
    A[0, 0], A[0, 2] = s2 * M1L / kL - pL / kL ** 2, 1.0
    A[1, 2], A[1, 1], A[1, 0] = s2 * CL, 1.0, -1.0 / kL
    A[2, 1], A[2, 2], A[2, 3] = s2 * MI, -1.0, 1.0
    A[3, 3], A[3, 4], A[3, 1] = s2 * CR, kR, -1.0
    A[4, 4], A[4, 3] = s2 * M1R * kR + pR * kR, -1.0
    return A


def t4():
    """Peaks of the interface forced response on |z| = e^{a tau} (a tau = 1e-4): location and mode shape."""
    N, Lam = 400, 2.0
    h = 1.0 / N
    out = []
    edge = 2 * np.arctan(Lam) / np.pi
    for name, rL, cL, rR, cR in (("Z_R/Z_L=1", 1, 1, 1, 1), ("Z_R/Z_L=10", 1, 1, 10, 1), ("Z_R/Z_L=0.01", 1, 1, 0.01, 1)):
        tau = Lam * h / max(cL, cR)
        ZS = rL * cL + rR * cR
        for th in (0.1, 0.3, 0.5, 0.7, 0.9):
            best = (0.0, None, None)
            for phi in np.linspace(0.02, np.pi * 0.998, 6000):
                z = np.exp(1e-4 + 1j * phi)
                A = matrix(z, th, h, tau, rL, cL, rR, cR)
                bvec = np.zeros(5, complex)
                bvec[2] = 1.0
                x = np.linalg.solve(A, bvec)
                val = max(abs(x[1]) * ZS, abs(x[2]), abs(x[3]))
                if val > best[0]:
                    best = (val, phi, x)
            x = np.abs(best[2])
            out.append((name, th, best[1] / np.pi, best[0], x / x.max()))
            print(f"T4 {name} theta={th}: peak K={best[0]:.1f} at arg z/pi={best[1] / np.pi:.4f} (band edge {edge:.4f}); "
                  f"|aL,w0,p-1,p0,aR| = {np.round(x / x.max(), 3)}", flush=True)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    if which in ("t1", "all"):
        with Pool(16) as pool:
            Ns, res = t1(pool)
        for (u0, th), r in res.items():
            ax[0].semilogx(Ns, [x[0] for x in r], "o-", label=f"u0={u0} (nu={r[0][3]:.3f}), theta={th}")
        ax[0].set_xlabel("N")
        ax[0].set_ylabel("S1 = sum|w_trace| / continuum value")
        ax[0].set_title("T1: interface impulse response, real scheme")
        ax[0].legend(fontsize=6)
    if which in ("t2", "all"):
        with Pool(16) as pool:
            nu, s1 = t2(pool)
        ax[1].plot(nu, s1, "o-")
        ax[1].set_xlabel("nu = u tau / h")
        ax[1].set_ylabel("S1")
        ax[1].set_title("T2: scan over nu (N=200)")
    if which in ("t3", "all"):
        rows = t3()
        for eps, (d, o) in rows.items():
            for j, nm in enumerate(("rho", "p", "w")):
                ax[2].plot([200, 400], o[:, j], "o-", label=f"{nm}, EPS={eps}")
        ax[2].axhline(2, color="k", ls="--")
        ax[2].set_xlabel("N (pair N/2,N)")
        ax[2].set_ylabel("observed L2 order")
        ax[2].set_title("T3: stronger background gradient")
        ax[2].legend(fontsize=6)
    if which in ("t4", "all"):
        out = t4()
        for name in ("Z_R/Z_L=1", "Z_R/Z_L=10", "Z_R/Z_L=0.01"):
            sel = [o for o in out if o[0] == name]
            ax[3].plot([o[1] for o in sel], [o[2] for o in sel], "o-", label=name)
        ax[3].axhline(2 * np.arctan(2.0) / np.pi, color="k", ls="--", label="band edge (material with larger c)")
        ax[3].set_xlabel("theta")
        ax[3].set_ylabel("trapped eigenvalue arg z / pi")
        ax[3].set_title("T4: trapped mode of the frozen interface")
        ax[3].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
