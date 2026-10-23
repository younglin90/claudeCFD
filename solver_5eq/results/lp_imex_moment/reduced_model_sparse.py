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


def remap_op(zl, zr, mat, xs, X, nu, s, nf, P, order=1):
    X2 = X + nu
    xs2, zl2, zr2, mat2 = geometry(X2, s, nf)
    nz, nz2, nn, nn2 = len(zl), len(zl2), len(xs), len(xs2)
    rows, cols, vals = [], [], []
    srows, scols, svals = [], [], []        # slope-correction part  (new <- old slope entries)
    k = np.arange(nz2)
    for d in range(-3, 4):
        i = k + d
        ok = (i >= 0) & (i < nz)
        kk, ii = k[ok], i[ok]
        lo_ = np.maximum(zl[ii] + nu, zl2[kk])
        hi_ = np.minimum(zr[ii] + nu, zr2[kk])
        ov = np.clip(hi_ - lo_, 0, None)
        ov = ov * (mat[ii] == mat2[kk]) / (zr2[kk] - zl2[kk])
        keep = ov > 0
        rows.append(nn2 + kk[keep]); cols.append(nn + ii[keep]); vals.append(ov[keep])
        if order == 2:
            cen = 0.5 * (zl[ii] + zr[ii])
            arm = (0.5 * (lo_ + hi_) - nu - cen)
            srows.append(kk[keep]); scols.append(ii[keep]); svals.append((ov * arm)[keep])
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
    if order == 2:
        # zone slopes (same-material neighbours, central difference over centre distances), node slopes in mass coord.
        cz = 0.5 * (zl + zr)
        i = np.arange(1, nz - 1)
        okz = (mat[i - 1] == mat[i]) & (mat[i + 1] == mat[i])
        iz = i[okz]
        Sz = sp.csr_matrix((np.concatenate([1.0 / (cz[iz + 1] - cz[iz - 1]), -1.0 / (cz[iz + 1] - cz[iz - 1])]),
                            (np.concatenate([iz, iz]), np.concatenate([iz + 1, iz - 1]))), shape=(nz, nz))
        mc = 0.5 * (lo0 + hi0)
        sides = (xs < X)
        j0 = np.arange(1, nn - 1)
        okn = (sides[j0 - 1] == sides[j0]) & (sides[j0 + 1] == sides[j0]) & (np.abs(xs[j0] - X) > 1e-12)
        jn = j0[okn]
        Sn = sp.csr_matrix((np.concatenate([1.0 / (mc[jn + 1] - mc[jn - 1]), -1.0 / (mc[jn + 1] - mc[jn - 1])]),
                            (np.concatenate([jn, jn]), np.concatenate([jn + 1, jn - 1]))), shape=(nn, nn))
        # zone part: W_z[k,i] = ov*arm  -> R_zone += W_z @ Sz
        Wz = sp.csr_matrix((np.concatenate(svals), (np.concatenate(srows), np.concatenate(scols))), shape=(nz2, nz))
        # node part: W_n[j,i] = ov*(mid-m_i)/width
        nrows, ncols, nvals = [], [], []
        j = np.arange(nn2)
        for d in range(-3, 4):
            ii = j + d
            ok = (ii >= 0) & (ii < nn)
            jj, ii = j[ok], ii[ok]
            lo_ = np.maximum(lo0[ii], lo1[jj]); hi_ = np.minimum(hi0[ii], hi1[jj])
            ov = np.clip(hi_ - lo_, 0, None) / (hi1[jj] - lo1[jj])
            keep = ov > 0
            nrows.append(jj[keep]); ncols.append(ii[keep]); nvals.append((ov * (0.5 * (lo_ + hi_) - mc[ii]))[keep])
        Wn = sp.csr_matrix((np.concatenate(nvals), (np.concatenate(nrows), np.concatenate(ncols))), shape=(nn2, nn))
        corr = sp.bmat([[Wn @ Sn, None], [None, Wz @ Sz]], format="csr")
        R = R + corr
    return R, X2, xs2, zl2, zr2, mat2


def run(nu, theta0, W, P=None, nf=None, tau_over_h=None, order=1):
    """nu: scalar (constant interface Courant number u tau / h) or an array of length W (time dependent, may change sign)."""
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
        nu_n = nu if np.isscalar(nu) else float(nu[n])
        R, X2, xs, zl2, zr2, mat2 = remap_op(zl, zr, mat, xs, X, nu_n, s, nf, P, order)
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


def run_channels(nu, theta0, W, P=None, nf=None, tau_over_h=None, order=1):
    """3x3 impulse-response kernel T[n, i, j]: impulse (unit W-energy) in channel j at step 0, output channel i after step n,
    channels = (interface-node velocity, pressure of the left cut piece, pressure of the right cut piece); each
    channel scaled by the square root of its energy weight (M_I, l/K) so entries are amplitudes in the energy norm."""
    P = dict(DEFAULT if P is None else P)
    c_L = np.sqrt(P["K_L"] / P["rho_L"])
    tau_h = (P["lam_L"] / c_L) if tau_over_h is None else tau_over_h
    if nf is None:
        nf = 2 * W + 80
    s = -nf // 2
    X = theta0
    xs, zl, zr, mat = geometry(X, s, nf)
    nz = len(zl)
    nn = nz + 1
    def chan(xs, zl, zr, mat, X):
        iI = int(np.argmin(np.abs(xs - X)))
        Lm, Rm, M = lagrange_ops(tau_h, zl, zr, mat, P)
        rho, K = props(mat, P)
        w_w = M[iI]
        w_l = (zr[iI - 1] - zl[iI - 1]) / K[iI - 1]
        w_r = (zr[iI] - zl[iI]) / K[iI]
        return iI, np.array([iI, nn + iI - 1, nn + iI]), np.sqrt(np.array([w_w, w_l, w_r]))
    iI, idx, sc = chan(xs, zl, zr, mat, X)
    T = np.zeros((W, 3, 3))
    X0 = X
    for j in range(3):
        x = np.zeros(nn + nz)
        x[idx[j]] = 1.0 / sc[j]
        xs_, zl_, zr_, mat_, X_ = xs, zl, zr, mat, X0
        for n in range(W):
            Lm, Rm, M = lagrange_ops(tau_h, zl_, zr_, mat_, P)
            x = spla.splu(Lm).solve(Rm @ x)
            R, X2, xs_, zl2, zr2, mat2 = remap_op(zl_, zr_, mat_, xs_, X_, nu, s, nf, P, order)
            x = R @ x
            X_, zl_, zr_, mat_ = X2, zl2, zr2, mat2
            iI2, idx2, sc2 = chan(xs_, zl_, zr_, mat_, X_)
            T[n, :, j] = x[idx2] * sc2
    return T
