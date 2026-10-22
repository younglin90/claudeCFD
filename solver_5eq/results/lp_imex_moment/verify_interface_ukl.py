"""Section 74: discrete uniform Kreiss-Lopatinskii (trace) check of the interface problem, with the exact
half-line radiation condition (no truncated box).

Model: two materials (rho, c), staggered nodes/zones, implicit midpoint in time (the Lagrange step of the
interface-fitted scheme linearised about a uniform state).  The interface node sits inside a cell of length h;
the cut cell is split into a left piece of length theta*h and a right piece of length (1-theta)*h, all other
zones have length h.  Homogeneous solutions w^n = z^n w, p^n = z^n p; bulk roots kappa from
    kappa + 1/kappa - 2 = (2 sigma / lambda_c)^2,   sigma = (z-1)/(z+1),   lambda_c = c tau / h,
left material keeps the root with |kappa|>1 (decay as b -> -inf), right material |kappa|<1.
Unknowns (a_L, w_0, p_-1, p_0, a_R); a unit force F on the interface node.  Reported:
  K_tr = sup over |z| = exp(a tau) of max(|w_0| (Z_L+Z_R), |p_-1|, |p_0|)/|F|
(by the maximum principle the sup over |z| >= exp(a tau) is attained on that circle, the transfer function
being analytic outside it).  Continuum transmission value: w_0 (Z_L+Z_R) = F, i.e. determinant 1/Z_L+1/Z_R.
Strong stability (GKS Theorem 5.1 / Gustafsson 1975 Theorem 2.1 hypothesis) <=> K_tr bounded uniformly in h."""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/interface_ukl.png"


def roots(sig, lam):
    q = 4 * sig ** 2 / lam ** 2
    t = 2 + q
    r = np.sqrt(t * t - 4 + 0j)
    k1, k2 = (t + r) / 2, (t - r) / 2
    return (k2, k1) if abs(k2) < abs(k1) else (k1, k2)      # (|kappa|<1, |kappa|>1)


def solve(z, theta, h, tau, rL, cL, rR, cR, F=1.0):
    sig = (z - 1) / (z + 1)
    s2 = 2 * sig / tau
    KL, KR = rL * cL ** 2, rR * cR ** 2
    kL = roots(sig, cL * tau / h)[1]                          # |kappa_L| > 1
    kR = roots(sig, cR * tau / h)[0]                          # |kappa_R| < 1
    pL = -(kL - 1) * tau * KL / (2 * sig * h)                  # p amplitude per unit w amplitude
    pR = -(kR - 1) * tau * KR / (2 * sig * h)
    M1L = rL * h * (1 + theta) / 2
    MI = (rL * theta * h + rR * (1 - theta) * h) / 2
    M1R = rR * h * (2 - theta) / 2
    CL, CR = theta * h / KL, (1 - theta) * h / KR
    # unknowns x = [aL, w0, pm1, p0, aR]
    A = np.zeros((5, 5), complex)
    b = np.zeros(5, complex)
    # node -1: s2 M1L w_-1 - (p_-2 - p_-1) = 0, w_-1 = aL/kL, p_-2 = pL aL / kL^2
    A[0, 0] = s2 * M1L / kL - pL / kL ** 2
    A[0, 2] = 1.0
    # zone -1: s2 CL p_-1 + (w0 - w_-1) = 0
    A[1, 2] = s2 * CL
    A[1, 1] = 1.0
    A[1, 0] = -1.0 / kL
    # node 0: s2 MI w0 - (p_-1 - p_0) = F
    A[2, 1] = s2 * MI
    A[2, 2] = -1.0
    A[2, 3] = 1.0
    b[2] = F
    # zone 0: s2 CR p_0 + (w_1 - w0) = 0, w_1 = aR kR
    A[3, 3] = s2 * CR
    A[3, 4] = kR
    A[3, 1] = -1.0
    # node 1: s2 M1R w_1 - (p_0 - p_1) = 0, p_1 = pR aR kR
    A[4, 4] = s2 * M1R * kR + pR * kR
    A[4, 3] = -1.0
    x = np.linalg.solve(A, b)
    return x[1], x[2], x[3]


def ktr(N, theta, a, rL, cL, rR, cR, Lam=2.0, ny=6000):
    h = 1.0 / N
    tau = Lam * h / max(cL, cR)
    ys = np.concatenate([np.linspace(-np.pi / tau * 0.9999, np.pi / tau * 0.9999, ny), np.linspace(-60, 60, 241)])
    ZS = rL * cL + rR * cR
    best = 0.0
    for y in ys:
        z = np.exp((a + 1j * y) * tau)
        w0, pm, p0 = solve(z, theta, h, tau, rL, cL, rR, cR)
        best = max(best, max(abs(w0) * ZS, abs(pm), abs(p0)))
    return best, None


def main():
    Ns = [50, 100, 200, 400, 800, 1600, 3200]
    cases = [("Z_R/Z_L=1", 1.0, 1.0, 1.0, 1.0), ("Z_R/Z_L=10", 1.0, 1.0, 10.0, 1.0),
             ("Z_R/Z_L=0.01", 1.0, 1.0, 0.01, 1.0), ("c_R=5, Z_R/Z_L=0.05", 1.0, 1.0, 0.01, 5.0)]
    thetas = [1e-9, 0.1, 0.5, 0.9, 1 - 1e-9]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for (name, rL, cL, rR, cR) in cases:
        worst = []
        for N in Ns:
            worst.append(max(ktr(N, th, 1.0, rL, cL, rR, cR, ny=800)[0] for th in thetas))
        print(f"{name}: sup_(theta,z) K_tr for N={Ns}: {np.round(worst, 4)}", flush=True)
        ax[0].semilogx(Ns, worst, "o-", label=name)
    ax[0].set_xlabel("N = 1/h")
    ax[0].set_ylabel("K_tr  (sup over theta and |z| = e^{a tau}, a = 1)")
    ax[0].set_title("trace amplification vs resolution")
    ax[0].legend(fontsize=7)
    name, rL, cL, rR, cR = cases[1]
    for th in thetas:
        v = [ktr(N, th, 1.0, rL, cL, rR, cR, ny=800)[0] for N in Ns]
        ax[1].semilogx(Ns, v, "o-", label=f"theta={th:g}")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("K_tr")
    ax[1].set_title(f"{name}: occupancy theta from 1e-9 to 1-1e-9")
    ax[1].legend(fontsize=7)
    for N in (100, 800, 3200):
        h = 1.0 / N
        tau = 2.0 * h / max(cL, cR)
        yy = np.linspace(0.0, 400.0, 400)
        val = [abs(solve(np.exp((1.0 + 1j * y) * tau), 0.37, h, tau, rL, cL, rR, cR)[0]) * (rL * cL + rR * cR)
               for y in yy]
        ax[2].plot(yy, val, label=f"N={N}")
    ax[2].axhline(1.0, color="k", ls="--", label="continuum: w_0 (Z_L+Z_R) = F")
    ax[2].set_xlabel("Im s")
    ax[2].set_ylabel("|w_0| (Z_L+Z_R) / |F|")
    ax[2].set_title("interface transfer function vs continuum limit (Re s = 1)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
