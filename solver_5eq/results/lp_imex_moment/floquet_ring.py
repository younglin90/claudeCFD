"""Section 77 (D): Floquet multipliers of the reduced linear model for rational nu = p/q.
One period = q steps (Lagrange midpoint + first-order donor-cell remap); the interface moves p cells, the state is
shifted back by p cells so the period map is time-invariant (section 74.2).  The localized eigenvalue with the
largest modulus (excluding the neutral multiplier near 1) gives the ring damping per step delta_F = -ln|mu|/q,
kappa_F = delta_F / nu, to be compared with the adiabatic estimate (derive_ring_damping.py) and the fitted 2.5."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rlm", os.path.join(HERE, "reduced_linear_model.py"))
rlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rlm)
oi = rlm.oi


def period_map(p, q, theta0, N=200):
    h = 1.0 / N
    tau = h / oi.CG
    nu = p / q
    s = -rlm.NF // 2
    X = theta0
    xs, zl, zr, mat = rlm.geometry(X, s)
    n = len(xs) + len(zl)
    Pi = np.eye(n)
    for k in range(q):
        G, M = rlm.lagrange(h, tau, zl, zr, mat)
        R, X2, xs2, zl2, zr2, mat2 = rlm.remap(xs, zl, zr, mat, X, nu, s)
        Pi = R @ (G @ Pi)
        X, xs, zl, zr, mat = X2, xs2, zl2, zr2, mat2
    nn = len(xs)
    nzn = len(zl)
    # shift back by p cells: new index i holds the state of old index i - p (nodes and zones separately)
    S = np.zeros_like(Pi)
    for i in range(nn):
        if 0 <= i - p < nn:
            S[i - p, i] = 1.0
    for z in range(nzn):
        if 0 <= z - p < nzn:
            S[nn + z - p, nn + z] = 1.0
    Phi = S @ Pi
    return Phi, nn, nzn, nu, np.argmin(np.abs(rlm.geometry(theta0, s)[0] - theta0))


def main():
    print("nu=p/q, q steps per period: localized eigenvalue with largest modulus: |mu|, delta_F/step, kappa_F=delta_F/nu")
    for (p, q) in ((1, 4), (1, 5), (1, 6), (1, 8), (1, 10), (1, 12), (1, 16), (2, 9)):
        Phi, nn, nzn, nu, iI = period_map(p, q, 0.37)
        ev, vec = np.linalg.eig(Phi)
        best = None
        for k in range(len(ev)):
            mu = ev[k]
            if abs(mu - 1) < 0.02 or abs(mu) < 1e-6:
                continue
            v = vec[:, k]
            e = np.abs(v) ** 2
            idx = np.arange(len(e))
            near = (np.abs(idx - iI) < 12) | ((idx >= nn) & (np.abs(idx - nn - iI) < 12))
            frac = e[near].sum() / e.sum()
            if frac > 0.9 and (best is None or abs(mu) > abs(best[0])):
                best = (mu, frac)
        if best is None:
            print(f"  nu={p}/{q}: no localized eigenvalue found")
            continue
        mu, frac = best
        d = -np.log(abs(mu)) / q
        print(f"  nu={p}/{q}={nu:.4f}: |mu|={abs(mu):.4f} arg/pi={np.angle(mu) / np.pi:+.3f} localized {frac:.2f}  "
              f"delta_F={d:.4f}  kappa_F={d / nu:.2f}", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
