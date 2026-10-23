"""Section 77 (A): residue of the trapped pole and the ring amplitude A(theta), compared with the real scheme.

Frozen-theta Lagrange interface (implicit midpoint, staggered), parameters of the real pulse test:
  left  (gas-like)  rho_L = 0.2, K_L = 14.012 ; right (water-like) rho_R = 1.0, K_R = 25.5 ; tau = h / C_G.
The operator is unitary in the energy norm W = sum M w^2 + sum (l/K) p^2, hence normal.  For a unit-circle
eigenvalue z_t = e^{i phi_t} with W-normalised eigenvector v, the interface-node velocity after an initial
velocity impulse eps at the interface node is
    w_I(n) = eps M_I sum_k z_k^n |v_k^w(I)|^2  ->  ring term 2 eps M_I |v^w(I)|^2 cos(n phi_t)   (conjugate pair).
The measured normalised trace is w_I (Z_L+Z_R) tau / (M_I eps), so the ring has amplitude
    R = 2 (Z_L+Z_R) tau |v^w(I)|^2        and its mean absolute value is  A = (2/pi) R = (4/pi)(Z_L+Z_R) tau |v^w(I)|^2.
|v^w(I)|^2 = |w_0|^2 / E with E the W-energy of the (unnormalised) mode, whose bulk tails are geometric series
(closed form).  The trapped eigenvalue is a zero of the 5x5 determinant of verify_interface_ukl.py."""
import importlib.util
import os

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("oi", os.path.join(HERE, "verify_open_items.py"))
oi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oi)

RHO_L, K_L, RHO_R, K_R = 0.2, 14.012, 1.0, 25.5
C_L, C_R = np.sqrt(K_L / RHO_L), np.sqrt(K_R / RHO_R)
Z_L, Z_R = RHO_L * C_L, RHO_R * C_R


def equilibrate(A):
    for _ in range(3):
        A = A / np.maximum(np.abs(A).max(axis=1, keepdims=True), 1e-300)
        A = A / np.maximum(np.abs(A).max(axis=0, keepdims=True), 1e-300)
    return A


def mode_data(theta, N, tau_over_h_c=None):
    h = 1.0 / N
    tau = h / oi.CG
    edge = 2 * np.arctan(max(C_L, C_R) * tau / h)
    best = None

    def smin(phi):
        A = oi.matrix(np.exp(1j * phi), theta, h, tau, RHO_L, C_L, RHO_R, C_R)
        s = np.linalg.svd(equilibrate(A), compute_uv=False)
        return s[-1] / s[0]
    grid = np.linspace(edge * 1.001, np.pi * 0.995, 4000)
    vals = np.array([smin(p) for p in grid])
    cand = [i for i in range(1, len(grid) - 1) if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]]
    cand.sort(key=lambda i: vals[i])
    for i in cand[:3]:
        r = minimize_scalar(smin, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded",
                            options={"xatol": 1e-14})
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    phi = best[1]
    z = np.exp(1j * phi)
    A = oi.matrix(z, theta, h, tau, RHO_L, C_L, RHO_R, C_R)
    u, s, vt = np.linalg.svd(A)
    x = vt[-1].conj()
    # energy of the mode (geometric tails)
    sig = (z - 1) / (z + 1)
    ukl = oi.load("ukl", "verify_interface_ukl.py")
    kL = ukl.roots(sig, C_L * tau / h)[1]
    kR = ukl.roots(sig, C_R * tau / h)[0]
    aL, w0, pm1, p0, aR = x
    pLc = -(kL - 1) * tau * K_L / (2 * sig * h)
    pRc = -(kR - 1) * tau * K_R / (2 * sig * h)
    qL, qR = abs(kL) ** -2, abs(kR) ** 2
    M1L = RHO_L * h * (1 + theta) / 2
    MI = (RHO_L * theta * h + RHO_R * (1 - theta) * h) / 2
    M1R = RHO_R * h * (2 - theta) / 2
    E = 0.0
    E += RHO_L * h * abs(aL) ** 2 * qL ** 2 / (1 - qL)               # nodes b <= -2
    E += M1L * abs(aL) ** 2 * qL                                      # node -1
    E += (h / K_L) * abs(pLc * aL) ** 2 * qL ** 2 / (1 - qL)         # zones z <= -2
    E += (theta * h / K_L) * abs(pm1) ** 2                            # zone -1
    E += MI * abs(w0) ** 2                                            # node 0
    E += ((1 - theta) * h / K_R) * abs(p0) ** 2                       # zone 0
    E += M1R * abs(aR * kR) ** 2                                      # node 1
    E += RHO_R * h * abs(aR) ** 2 * qR ** 2 / (1 - qR)                # nodes b >= 2
    E += (h / K_R) * abs(pRc * aR) ** 2 * qR / (1 - qR)               # zones z >= 1
    R = 2 * (Z_L + Z_R) * tau * abs(w0) ** 2 / E
    return phi, best[0], R, 2 * R / np.pi, (kL, kR), (aL, w0, pm1, p0, aR), tau, h


def main():
    meas = {0.05: 0.188, 0.1: 0.205, 0.3: 0.320, 0.5: 0.763, 0.7: 1.759, 0.9: 2.929}
    print("theta  arg z_t/pi  sigma_min  ring amp R   A_pred = 2R/pi   A_meas(real scheme, N=400)   ratio")
    for N in (400,):
        for th in meas:
            phi, sm, R, A, kap, x, tau, h = mode_data(th, N)
            print(f"{th:5.2f}  {phi / np.pi:9.5f}  {sm:9.2e}  {R:10.4f}   {A:12.4f}   {meas[th]:12.3f}   {A / meas[th]:6.3f}")
    print("N-independence of A_pred at theta=0.5:", [round(mode_data(0.5, N)[3], 4) for N in (100, 200, 400, 800)])


if __name__ == "__main__":
    main()
