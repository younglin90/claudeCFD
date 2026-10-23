"""Section 78: sparse (banded) version of the reduced linear model of section 77 -- exact Lagrange midpoint step on
the cut-cell geometry followed by the first-order donor-cell remap -- so that long windows (hundreds of steps,
thousands of faces) are cheap.  Impulse response of the interface-node velocity, normalised like the real
scheme (verify_ringdown_model.series): trace_n * (Z_L+Z_R) tau / (M_I eps).  S1(W) = sum_{n<W} |trace_n|.
Parameters (rho_L, K_L, rho_R, K_R, c_L tau / h) are free, so other material pairs can be tested."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DEFAULT = dict(rho_L=0.2, K_L=14.012, rho_R=1.0, K_R=25.5, lam_L=0.9943)


def geometry(X, s, nf):
    faces = np.arange(s, s + nf, dtype=float)
    if abs(X - round(X)) < 1e-9:
        X += 1e-9
    xs = np.sort(np.concatenate([faces, [X]]))
    return xs, xs[:-1], xs[1:], (0.5 * (xs[:-1] + xs[1:]) >= X).astype(int)


def props(mat, P):
    rho = np.where(mat == 0, P["rho_L"], P["rho_R"])
    K = np.where(mat == 0, P["K_L"], P["K_R"])
    return rho, K


def lagrange_ops(tau_h, zl, zr, mat, P):
    """(I - tau A/2), (I + tau A/2) in units h = 1 (tau_h = tau/h in cells per unit of c...)."""
    nz = len(zl)
    rho, K = props(mat, P)
    m = rho * (zr - zl)
    C = (zr - zl) / K
    M = np.zeros(nz + 1)
    M[:-1] += 0.5 * m
    M[1:] += 0.5 * m
    nn = nz + 1
    rows, cols, vals = [], [], []
    b = np.arange(1, nn)
    rows += [b]; cols += [nn + b - 1]; vals += [1.0 / M[b]]
    b = np.arange(0, nn - 1)
    rows += [b]; cols += [nn + b]; vals += [-1.0 / M[b]]
    z = np.arange(nz)
    rows += [nn + z]; cols += [z + 1]; vals += [-1.0 / C]
    rows += [nn + z]; cols += [z]; vals += [1.0 / C]
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nn + nz, nn + nz))
    I = sp.identity(nn + nz, format="csc")
    return (I - 0.5 * tau_h * A).tocsc(), (I + 0.5 * tau_h * A).tocsr(), M


def remap_op(zl, zr, mat, xs, X, nu, s, nf, P):
    X2 = X + nu
    xs2, zl2, zr2, mat2 = geometry(X2, s, nf)
    nz, nz2, nn, nn2 = len(zl), len(zl2), len(xs), len(xs2)
    rows, cols, vals = [], [], []
    k = np.arange(nz2)
    for d in range(-3, 4):
        i = k + d
        ok = (i >= 0) & (i < nz)
        kk, ii = k[ok], i[ok]
        ov = np.clip(np.minimum(zr[ii] + nu, zr2[kk]) - np.maximum(zl[ii] + nu, zl2[kk]), 0, None)
        ov = ov * (mat[ii] == mat2[kk]) / (zr2[kk] - zl2[kk])
        keep = ov > 0
        rows.append(nn2 + kk[keep]); cols.append(nn + ii[keep]); vals.append(ov[keep])
    rho_old = np.where(mat == 0, P["rho_L"], P["rho_R"])
    rho_new = np.where(mat2 == 0, P["rho_L"], P["rho_R"])
    mu_old = lambda x: np.where(x <= X, P["rho_L"] * (x - xs[0]), P["rho_L"] * (X - xs[0]) + P["rho_R"] * (x - X))
    mz, mz2 = rho_old * (zr - zl), rho_new * (zr2 - zl2)
    ml, mr = np.concatenate([[0.0], mz]), np.concatenate([mz, [0.0]])
    lo0, hi0 = mu_old(xs) - ml / 2, mu_old(xs) + mr / 2
    ml2, mr2 = np.concatenate([[0.0], mz2]), np.concatenate([mz2, [0.0]])
    lo1, hi1 = mu_old(xs2 - nu) - ml2 / 2, mu_old(xs2 - nu) + mr2 / 2
    j = np.arange(nn2)
    for d in range(-3, 4):
        i = j + d
        ok = (i >= 0) & (i < nn)
        jj, ii = j[ok], i[ok]
        ov = np.clip(np.minimum(hi0[ii], hi1[jj]) - np.maximum(lo0[ii], lo1[jj]), 0, None) / (hi1[jj] - lo1[jj])
        keep = ov > 0
        rows.append(jj[keep]); cols.append(ii[keep]); vals.append(ov[keep])
    R = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nn2 + nz2, nn + nz))
    return R, X2, xs2, zl2, zr2, mat2


def run(nu, theta0, W, P=None, nf=None, tau_over_h=None):
    """Interface-node trace series (normalised) for an impulse at the interface node.  c_L tau/h = lam_L."""
    P = dict(DEFAULT if P is None else P)
    c_L = np.sqrt(P["K_L"] / P["rho_L"])
    tau_h = (P["lam_L"] / c_L) if tau_over_h is None else tau_over_h        # tau / h in units where h = 1
    if nf is None:
        nf = 2 * W + 80
    s = -nf // 2
    X = theta0
    xs, zl, zr, mat = geometry(X, s, nf)
    nz = len(zl)
    nn = nz + 1
    iI = int(np.argmin(np.abs(xs - X)))
    Lm, Rm, M = lagrange_ops(tau_h, zl, zr, mat, P)
    MI = M[iI]
    Z_L, Z_R = np.sqrt(P["K_L"] * P["rho_L"]), np.sqrt(P["K_R"] * P["rho_R"])
    x = np.zeros(nn + nz)
    x[iI] = 1.0
    norm = (Z_L + Z_R) * tau_h / MI
    out = np.zeros(W)
    for n in range(W):
        Lm, Rm, M = lagrange_ops(tau_h, zl, zr, mat, P)
        x = spla.splu(Lm).solve(Rm @ x)
        R, X2, xs, zl2, zr2, mat2 = remap_op(zl, zr, mat, xs, X, nu, s, nf, P)
        x = R @ x
        X, zl, zr, mat = X2, zl2, zr2, mat2
        iI = int(np.argmin(np.abs(xs - X)))
        out[n] = x[iI] * norm
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    tr = run(0.0358, 0.37, 70)
    print(f"sparse model: S1(W=70, nu=0.0358, theta0=0.37) = {np.abs(tr).sum():.3f} (dense model 6.865, real scheme 7.180); "
          f"{time.time() - t0:.1f}s")
