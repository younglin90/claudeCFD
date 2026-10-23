"""Section 77 (C): reduced linear model of the real scheme at the interface = frozen-geometry Lagrange midpoint
step (exact linear acoustics on the cut cell geometry) followed by the first-order donor-cell remap derived in
derive_ring_damping.py, on a long window.  Impulse response of the interface-node velocity, normalised as in
verify_ringdown_model.series, compared with the real scheme."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rm = load("rm", "verify_ringdown_model.py")
oi = rm.oi
RHO_L, K_L, RHO_R, K_R = 0.2, 14.012, 1.0, 25.5
NF = 240


def geometry(X, s):
    faces = list(range(s, s + NF))
    if abs(X - round(X)) < 1e-9:
        X += 1e-9
    xs = np.array(sorted(faces + [X]))
    zl, zr = xs[:-1], xs[1:]
    mat = np.where(0.5 * (zl + zr) < X, 0, 1)
    return xs, zl, zr, mat


def lagrange(h, tau, zl, zr, mat):
    nz = len(zl)
    rho = np.where(mat == 0, RHO_L, RHO_R)
    K = np.where(mat == 0, K_L, K_R)
    m = rho * (zr - zl) * h
    C = (zr - zl) * h / K
    M = np.zeros(nz + 1)
    M[:-1] += 0.5 * m
    M[1:] += 0.5 * m
    n = nz + 1 + nz
    A = np.zeros((n, n))
    for b in range(nz + 1):
        if b >= 1:
            A[b, nz + 1 + b - 1] += 1.0 / M[b]
        if b <= nz - 1:
            A[b, nz + 1 + b] -= 1.0 / M[b]
    for z in range(nz):
        A[nz + 1 + z, z + 1] -= 1.0 / C[z]
        A[nz + 1 + z, z] += 1.0 / C[z]
    I = np.eye(n)
    return np.linalg.solve(I - 0.5 * tau * A, I + 0.5 * tau * A), M


def remap(xs, zl, zr, mat, X, nu, s):
    X2 = X + nu
    xs2, zl2, zr2, mat2 = geometry(X2, s)
    # zone pressure
    ov = np.clip(np.minimum(zr[None, :] + nu, zr2[:, None]) - np.maximum(zl[None, :] + nu, zl2[:, None]), 0, None)
    ov = ov * (mat[None, :] == mat2[:, None])
    Op = ov / (zr2 - zl2)[:, None]
    # nodal velocity in mass coordinate
    def mu_old(x):
        return np.where(x <= X, RHO_L * (x - xs[0]), RHO_L * (X - xs[0]) + RHO_R * (x - X))
    mz = np.where(mat == 0, RHO_L, RHO_R) * (zr - zl)
    mz2 = np.where(mat2 == 0, RHO_L, RHO_R) * (zr2 - zl2)
    def duals(xs_, mz_, mu):
        ml = np.concatenate([[0.0], mz_])
        mr = np.concatenate([mz_, [0.0]])
        return mu - ml / 2, mu + mr / 2
    lo0, hi0 = duals(xs, mz, mu_old(xs))
    lo1, hi1 = duals(xs2, mz2, mu_old(xs2 - nu))
    ovn = np.clip(np.minimum(hi0[None, :], hi1[:, None]) - np.maximum(lo0[None, :], lo1[:, None]), 0, None)
    Ow = ovn / (hi1 - lo1)[:, None]
    nzn, nn = len(zl), len(xs)
    R = np.zeros((nn + nzn, nn + nzn))
    R[:nn, :nn] = Ow
    R[nn:, nn:] = Op
    return R, X2, xs2, zl2, zr2, mat2


def run(N, u0, theta0, W, eps=1.0):
    h = 1.0 / N
    tau = h / oi.CG
    nu = u0 * tau / h
    s = -NF // 2
    X = theta0
    xs, zl, zr, mat = geometry(X, s)
    nz = len(zl)
    iI = int(np.argmin(np.abs(xs - X)))
    G, M = lagrange(h, tau, zl, zr, mat)
    MI = M[iI]
    x = np.zeros(nz + 1 + nz)
    x[iI] = eps
    norm = (oi.ZL + oi.ZR) * tau / (MI * eps)
    out = []
    for n in range(W):
        G, M = lagrange(h, tau, zl, zr, mat)
        x = G @ x
        R, X2, xs2, zl2, zr2, mat2 = remap(xs, zl, zr, mat, X, nu, s)
        x = R @ x
        X, xs, zl, zr, mat = X2, xs2, zl2, zr2, mat2
        iI = int(np.argmin(np.abs(xs - X)))
        out.append(x[iI] * norm)
    return np.array(out), nu


def main():
    N, W, th0 = 200, 70, 0.37
    print("u0, nu: S1 real scheme | S1 reduced linear model | sums for n>=10: real / model")
    res = []
    for u0 in (0.3, 0.6, 1.0, 1.6):
        meas, nu = rm.series((N, u0, th0, W, 1e-4))
        mod, _ = run(N, u0, th0, W)
        a, b = np.abs(meas), np.abs(mod)
        corr = np.corrcoef(a[3:], b[3:])[0, 1]
        print(f"  u0={u0} nu={nu:.4f}: S1 {a.sum():.3f} | {b.sum():.3f} (ratio {b.sum() / a.sum():.2f}) ;"
              f" n>=10: {a[10:].sum():.3f} / {b[10:].sum():.3f} ; corr {corr:.3f}", flush=True)
        res.append((u0, nu, meas, mod))
    np.save("/tmp/claude-1000/oi/reduced_model.npy", np.array(res, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
