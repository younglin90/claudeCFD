"""Interface-fitted staggered LP-IMEX (section 45).

Zones = material pieces in global order; boundaries = Eulerian faces + material interfaces.
Each zone: (k, a, b, m, Z, E) with its own pressure p = Gamma (E - C Z)/l (no relaxation).
Each boundary carries a velocity; interface boundaries are Lagrangian (never remapped).

Step:
  1. AVF Lagrange stage on the zone chain (each zone on its own single-material isentrope).
  2. Remap: faces return to the Eulerian grid, interfaces stay; zone contents re-partitioned
     with a limited linear density and pressure reconstruction (mass, Z, E exact).
  3. Nodal momentum re-partition in the Lagrangian mass coordinate (material does not move
     during the remap): minmod-linear velocity per dual zone, new velocity = mass average;
     the kinetic-energy change is returned to the zones as heat.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_GX, _GW = np.polynomial.legendre.leggauss(4)


class Phase:
    def __init__(self, A, n, G):
        self.A, self.n, self.G = A, n, G
        self.C = A * (1.0 / (n - 1) - 1.0 / G)


def zint(rb, s, xc, lo, hi, n):
    xm, xr = 0.5 * (lo + hi), 0.5 * (hi - lo)
    r = rb + s * (xm + xr * _GX - xc)
    return xr * np.sum(_GW * r ** n)


def zone_p(z, ph):
    return ph.G * (z[5] - ph.C * z[4]) / (z[2] - z[1])


class State:
    """zones: list [k, a, b, m, Z, E] in increasing position (periodic, positions may exceed 1 for the
    last zone); bnd: list of (position, is_face, face_index); w: boundary velocities.
    Boundary b is the LEFT boundary of zone b."""

    def __init__(self, zones, isface, w):
        self.zones = zones
        self.isface = isface
        self.w = np.asarray(w, float)


# ------------------------------------------------------------------ construction
def build(N, phases, mat_of, rho_of, p_of, u_of, interfaces):
    """Exact-ish initial zones on N cells with the given interface positions in (0,1)."""
    h = 1.0 / N
    cuts = sorted(set([c * h for c in range(N)] + [x for x in interfaces]))
    zones, isface, w = [], [], []
    GX, GW = np.polynomial.legendre.leggauss(8)
    for i, a in enumerate(cuts):
        b = cuts[i + 1] if i + 1 < len(cuts) else 1.0 + cuts[0]
        xm = 0.5 * (a + b)
        k = mat_of(xm % 1.0)
        ph = phases[k]
        xs = xm + 0.5 * (b - a) * GX
        m = 0.5 * (b - a) * np.sum(GW * rho_of(k, xs % 1.0))
        Z = 0.5 * (b - a) * np.sum(GW * rho_of(k, xs % 1.0) ** ph.n)
        p = 0.5 * np.sum(GW * p_of(xs % 1.0))
        zones.append([k, a, b, m, Z, p * (b - a) / ph.G + ph.C * Z])
        isface.append(abs(a / h - round(a / h)) < 1e-12)
        w.append(u_of(a % 1.0))
    return State(zones, isface, w)


# ------------------------------------------------------------------ Lagrange (AVF)
def _isentrope(z, ph):
    l0, Z0 = z[2] - z[1], z[4]
    zb = Z0 / l0
    p0 = zone_p(z, ph)

    def p(L):
        J = L / l0
        return ph.A * zb * J ** (-ph.n) + (p0 - ph.A * zb) * J ** (-1.0 - ph.G)

    def E(L):
        J = L / l0
        return p(L) * L / ph.G + ph.C * Z0 * J ** (1 - ph.n)
    return p, E, l0


_LAST = {}                # last force correction (diagnostics)


def _keep_positive(D, dD, L0):
    """Newton update D - dD, halved until every zone length stays positive (sub-1e-6 h zones can be
    overshot by a full step; the converged solution is unchanged)."""
    for _ in range(60):
        Dn = D - dD
        if np.all(L0 + np.roll(Dn, -1) - Dn > 0):
            return Dn
        dD = 0.5 * dD
    return D


def lagrange(st, tau, phases):
    Zs = st.zones
    B = len(Zs)
    m = np.array([z[3] for z in Zs])
    M = 0.5 * (m + np.roll(m, 1))            # boundary b between zone b-1 and b
    funcs = [_isentrope(z, phases[z[0]]) for z in Zs]
    L0 = np.array([f[2] for f in funcs])
    w = st.w

    def pt_of(Lz):
        pt, dp = np.empty(B), np.empty(B)
        for i, (pf, Ef, l0) in enumerate(funcs):
            dL = Lz[i] - l0
            if abs(dL) > 1e-7 * l0:
                pt[i] = -(Ef(Lz[i]) - Ef(l0)) / dL
                dp[i] = (pf(Lz[i]) - pt[i]) / dL
            else:
                pt[i] = (pf(l0) + 4 * pf(l0 + 0.5 * dL) + pf(Lz[i])) / 6.0
                e = 1e-7 * l0
                dp[i] = 0.5 * (pf(l0 + 0.5 * dL + e) - pf(l0 + 0.5 * dL - e)) / (2 * e)
        return pt, dp

    corr = np.zeros(B)
    Cimp = None                               # CORR_IMPLICIT: curvature part corr_curv = Cimp @ p_secant
    if FORCE_CORR[0]:
        # zone pressure is a zone average; the dual-zone momentum balance needs p at the zone's mass
        # midpoint: p_avg - p(x_mass_mid) = c_p l^2/12 - s_p s_rho l^2/(8 rho_bar)  (explicit, frozen)
        srr, spp = _slopes(Zs, phases, True)
        # band of zones within two zones of a material interface: there neighbouring zones have
        # unequal lengths (interface-adjacent zone of length xi*h next to full zones), so the O(h^2)
        # terms do not cancel in the nodal force; elsewhere no correction (no explicit stiffness)
        band = np.zeros(B, bool)
        for b in range(B):
            if not st.isface[b]:
                for d in (-2, -1, 0, 1):
                    band[(b + d) % B] = True
        if CORR_ALL[0]:
            band[:] = True
        full = np.array([st.isface[b] and st.isface[(b + 1) % B] for b in range(B)])
        cs_, cap_, mg_ = np.zeros(B), np.zeros(B), np.full(B, np.inf)
        rr, cc, vv = [], [], []
        for i, z in enumerate(Zs):
            if not band[i]:
                continue
            l = z[2] - z[1]
            rbar = z[3] / l
            cs_[i] = -spp[i][0] * srr[i][0] * l * l / (8 * rbar)
            corr[i] = spp[i][1] * l * l / 12 + cs_[i]
            # never let the correction exceed a small fraction of the thermal pressure margin
            ph = phases[z[0]]
            mg_[i] = 0.1 * (zone_p(z, ph) - ph.A * z[4] / l)
            corr[i] = float(np.clip(corr[i], -mg_[i], mg_[i]))
            if CORR_IMPLICIT[0]:
                # the curvature part c_p l^2/12 is a second difference of p (an O(1) fraction of the
                # acoustic operator at the grid scale): lagged it amplifies grid modes, so it is
                # re-evaluated from the secant (midpoint) pressures inside the Newton loop
                corr[i] = float(np.clip(cs_[i], -mg_[i], mg_[i]))
                sym = CORR_STENCIL[0] == "fullsym"
                if not PARABOLIC[0] or (spp[i][0] == 0.0 and not sym):
                    continue
                if CORR_GATE[0] and spp[i][1] == 0.0:
                    continue          # _slopes dropped the curvature (validity check): no curvature term
                w8 = _curv_weights(Zs, i, full if CORR_STENCIL[0] == "fullsym" else None)
                if w8 is None or (CORR_STENCIL[0] == "fullonly" and not (full[i] and all(full[w8[0]]))):
                    continue
                # monotone-parabola cap on c_p (times l^2/12); fullsym: none, so the operator depends on the
                # geometry only (a slope-dependent row set/cap switches with grid noise)
                cap_[i] = np.inf if sym else abs(spp[i][0]) / l * l * l / 12
                rr += [i] * 3
                cc += list(w8[0]) + [i]
                vv += list(w8[1] * l * l / 12) + [-w8[1].sum() * l * l / 12]
        if CORR_IMPLICIT[0] and rr:
            Cimp = sp.csr_matrix((vv, (rr, cc)), shape=(B, B))
            if CORR_STENCIL[0] == "fullsym":
                # symmetrise in the l/K inner product: diag(l/K)(I - C) symmetric positive (energy-stable)
                lz0 = np.array([z[2] - z[1] for z in Zs])
                K = np.array([-l0 * (pf(l0 * (1 + 1e-7)) - pf(l0 * (1 - 1e-7))) / (2e-7 * l0) for pf, Ef, l0 in funcs])
                Hw = lz0 / K
                # graph-Laplacian form corr_i = sum_j a_ij (p_j - p_i): symmetrise the edge weights
                # H_i a_ij <-> H_j a_ji and rebuild the diagonal, so rows still annihilate constants
                A = (Cimp - sp.diags(Cimp.diagonal())).tocsr()
                A = 0.5 * (A + sp.diags(1 / Hw) @ A.T @ sp.diags(Hw))
                Cimp = (A - sp.diags(np.asarray(A.sum(axis=1)).ravel())).tocsr()
                cap_[:] = np.inf
            rows_c = np.nonzero(Cimp.getnnz(axis=1))[0]
    D = _keep_positive(np.zeros(B), -tau * w, L0)    # initial guess tau*w, shortened if it crushes a zone
    for it in range(80):
        Lz = L0 + np.roll(D, -1) - D            # zone b between boundary b and b+1
        if np.any(Lz <= 0):
            raise RuntimeError("zone collapsed")
        pt, dp = pt_of(Lz)
        if Cimp is not None:
            cv = np.clip(Cimp @ pt, -cap_, cap_)
            corr[rows_c] = np.clip(cv[rows_c] + cs_[rows_c], -mg_[rows_c], mg_[rows_c])
        pt = pt - corr
        R = M * (D - tau * w) + 0.5 * tau ** 2 * (pt - np.roll(pt, 1))
        if np.max(np.abs(R) / np.maximum(M, 1e-300)) < 1e-14 * max(1.0, np.max(np.abs(D))):
            break
        c = 0.5 * tau ** 2
        idx = np.arange(B)
        if Cimp is not None:
            # J = diag(M) - c Dw^T (I - C) diag(dp) Dw  (zone length L = L0 + Dw D; force -(Dw^T p_force))
            Dw = sp.csr_matrix((np.concatenate([np.ones(B), -np.ones(B)]),
                                (np.concatenate([idx, idx]), np.concatenate([(idx + 1) % B, idx]))), shape=(B, B))
            Jm = (sp.diags(M) - c * Dw.T @ (sp.identity(B) - Cimp) @ sp.diags(dp) @ Dw).tocsc()
            D = _keep_positive(D, spla.spsolve(Jm, R), L0)
            continue
        rows = np.concatenate([idx, idx, idx])
        cols = np.concatenate([(idx + 1) % B, idx, (idx - 1) % B])
        vals = np.concatenate([c * dp, M - c * dp - c * np.roll(dp, 1), c * np.roll(dp, 1)])
        Jm = sp.csc_matrix((vals, (rows, cols)), shape=(B, B))
        D = _keep_positive(D, spla.spsolve(Jm, R), L0)
    Lz = L0 + np.roll(D, -1) - D
    wn = 2 * D / tau - w
    newzones = []
    for z, (pf, Ef, l0), L, Db in zip(Zs, funcs, Lz, D):
        ph = phases[z[0]]
        J = L / l0
        a2 = z[1] + Db
        # work of the correction force is returned to the zone as heat (total energy exact):
        # the force used p - corr, so the zone receives + corr * dL
        newzones.append([z[0], a2, a2 + L, z[3], z[4] * J ** (1 - ph.n), Ef(L) + corr[len(newzones)] * (L - l0)])
    _LAST['corr'] = corr
    return newzones, wn, M, it


# ------------------------------------------------------------------ remap helpers
FORCE_CORR = [False]  # dual-boundary pressure correction in the Lagrange force, section 57
VCURV = [True]        # one-sided velocity curvature at interface nodes (only with PARABOLIC)
CORR_ALL = [False]    # apply the force correction in every zone (stable only for moderate acoustic CFL)
PARABOLIC = [False]   # curvature-corrected (monotone parabolic) reconstruction, section 56
CORR_IMPLICIT = [False]  # curvature part of the force correction at the secant pressure (section 62)
# stencil of the implicit curvature correction (section 62): "raw" = as _slopes (unstable for some
# occupancies: diag(l/K) C not symmetric), "fullonly" = only zones whose central stencil is full zones,
# "fullsym" = full-zone stencils, then symmetrised in the l/K inner product (energy-stable by construction)
CORR_STENCIL = ["raw"]
CORR_GATE = [False]   # implicit curvature term only where _slopes kept the parabola (as the lagged form)


def rec_int(rb, s, c, xc, l, lo, hi):
    """Integral over [lo, hi] of rb + s (x-xc) + c ((x-xc)^2 - l^2/12)."""
    u, v = lo - xc, hi - xc
    return (hi - lo) * rb + s * (v * v - u * u) / 2 + c * ((v ** 3 - u ** 3) / 3 - l * l / 12 * (hi - lo))


def rec_powint(rb, s, c, xc, l, lo, hi, n):
    xm, xr = 0.5 * (lo + hi), 0.5 * (hi - lo)
    x = xm + xr * _GX
    r = rb + s * (x - xc) + c * ((x - xc) ** 2 - l * l / 12)
    return xr * np.sum(_GW * r ** n)


def _curv_fit(rb_g, xc_g, l, nb):
    """Curvature of the mean-preserving parabola matching the averages of two neighbour zones."""
    A = np.zeros((2, 2))
    y = np.zeros(2)
    for i, (av, a, b) in enumerate(nb):
        al, be = a - xc_g, b - xc_g
        A[i, 0] = 0.5 * (al + be)
        A[i, 1] = (be ** 3 - al ** 3) / (3 * (be - al)) - l * l / 12
        y[i] = av - rb_g
    try:
        return np.linalg.solve(A, y)[1]
    except np.linalg.LinAlgError:
        return 0.0


def _curv_weights(zones, g, full=None):
    """Neighbours and linear weights with curvature = w . (p_nb - p_g), as in _slopes/_curv_fit.
    With `full`, stencil members must also be full (face-to-face) zones."""
    G = len(zones)
    z = zones[g]
    k, l = z[0], z[2] - z[1]
    xc = 0.5 * (z[1] + z[2])
    same = lambda j: zones[j % G][0] == k and (full is None or full[j % G])
    if same(g - 1) and same(g + 1):
        nbs = [g - 1, g + 1]
    elif same(g + 1) and same(g + 2):
        nbs = [g + 1, g + 2]
    elif same(g - 1) and same(g - 2):
        nbs = [g - 1, g - 2]
    else:
        return None
    A = np.zeros((2, 2))
    for r, j in enumerate(nbs):
        zj = zones[j % G]
        cj = 0.5 * (zj[1] + zj[2])
        off = (((cj - xc + 0.5) % 1.0) - 0.5) + xc - cj
        al, be = zj[1] + off - xc, zj[2] + off - xc
        A[r, 0] = 0.5 * (al + be)
        A[r, 1] = (be ** 3 - al ** 3) / (3 * (be - al)) - l * l / 12
    try:
        return np.array([j % G for j in nbs]), np.linalg.inv(A)[1]
    except np.linalg.LinAlgError:
        return None


def _minmod(a, b):
    return np.sign(a) * min(abs(a), abs(b)) if a * b > 0 else 0.0


def _slopes(zones, phases, recon):
    """Limited linear density slope s_rho and pressure slope s_p per Lagrangian zone."""
    G = len(zones)
    sr, sp = np.zeros(G), np.zeros(G)
    if not recon:
        return np.zeros((G, 2)), np.zeros((G, 2))
    rb = np.array([z[3] / (z[2] - z[1]) for z in zones])
    xc = np.array([0.5 * (z[1] + z[2]) for z in zones])
    pz = np.array([zone_p(z, phases[z[0]]) for z in zones])

    def dist(i, j):
        return (xc[j] - xc[i]) % 1.0 if j != i else 1.0
    for g in range(G):
        gl, gr = (g - 1) % G, (g + 1) % G
        z = zones[g]
        ph = phases[z[0]]
        l = z[2] - z[1]
        sl, srr = zones[gl][0] == z[0], zones[gr][0] == z[0]
        dl = (rb[g] - rb[gl]) / dist(gl, g) if sl else None
        dr = (rb[gr] - rb[g]) / dist(g, gr) if srr else None
        if dl is not None and dr is not None:
            s0 = _minmod(dl, dr)
        elif dl is not None or dr is not None:
            s0 = dl if dr is None else dr
        else:
            s0 = 0.0
        # pressure is continuous across interfaces but its gradient jumps (impedance): same-material
        # neighbours only, one-sided at an interface
        pl_ = (pz[g] - pz[gl]) / dist(gl, g) if sl else None
        pr_ = (pz[gr] - pz[g]) / dist(g, gr) if srr else None
        if pl_ is not None and pr_ is not None:
            p0s = _minmod(pl_, pr_)
        elif pl_ is not None or pr_ is not None:
            p0s = pl_ if pr_ is None else pr_
        else:
            p0s = 0.0

        def ok(phi_r, phi_p):
            s = phi_r * s0
            if rb[g] - abs(s) * l / 2 <= 0.05 * rb[g]:
                return False
            X = z[4] - zint(rb[g], s, 0.0, -l / 2, l / 2, ph.n)
            if X < 0:
                return False
            need = ph.A * ((rb[g] + abs(s) * l / 2) ** ph.n + X / l)
            return need < 0.98 * (pz[g] - abs(phi_p * p0s) * l / 2)
        phi = 1.0
        if not ok(phi, phi):
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if ok(mid, mid):
                    lo = mid
                else:
                    hi = mid
            phi = lo
        sr[g], sp[g] = phi * s0, phi * p0s
    cr, cp = np.zeros(G), np.zeros(G)
    if PARABOLIC[0]:
        def unwrap(j, ref):
            zj = zones[j]
            cj = 0.5 * (zj[1] + zj[2])
            off = (((cj - ref + 0.5) % 1.0) - 0.5) + ref - cj
            return zj[1] + off, zj[2] + off
        for g in range(G):
            z = zones[g]
            l = z[2] - z[1]
            k = z[0]
            same = lambda j: zones[j % G][0] == k
            if same(g - 1) and same(g + 1):
                nbs = [g - 1, g + 1]
            elif same(g + 1) and same(g + 2):
                nbs = [g + 1, g + 2]
            elif same(g - 1) and same(g - 2):
                nbs = [g - 1, g - 2]
            else:
                continue
            for arr_s, arr_c, val in ((sr, cr, rb), (sp, cp, pz)):
                if arr_s[g] == 0.0:
                    continue
                nb = [(val[j % G],) + unwrap(j % G, xc[g]) for j in nbs]
                c = _curv_fit(val[g], xc[g], l, nb)
                cmax = abs(arr_s[g]) / l          # keeps the parabola monotone inside the zone
                arr_c[g] = float(np.clip(c, -cmax, cmax))
        for g in range(G):
            z = zones[g]
            ph = phases[z[0]]
            l = z[2] - z[1]
            ends = [rb[g] + sgn * sr[g] * l / 2 + cr[g] * l * l / 6 for sgn in (-1, 1)]
            pends = [pz[g] + sgn * sp[g] * l / 2 + cp[g] * l * l / 6 for sgn in (-1, 1)]
            X = z[4] - rec_powint(rb[g], sr[g], cr[g], 0.0, l, -l / 2, l / 2, ph.n)
            need = ph.A * (max(ends) ** ph.n + X / l)
            if min(ends) <= 0.05 * rb[g] or X < 0 or need >= 0.98 * min(pends):
                cr[g] = cp[g] = 0.0
    return np.stack([sr, cr], 1), np.stack([sp, cp], 1)


def _fragment(z, ph, s_r, s_p, lo, hi):
    """Content (m, Z, E) of the sub-interval [lo, hi] of Lagrangian zone z; s_r, s_p = (slope, curvature)."""
    a, b, m, Z = z[1], z[2], z[3], z[4]
    l = b - a
    rb, xc = m / l, 0.5 * (a + b)
    p = zone_p(z, ph)
    lam = (hi - lo) / l
    (s1, c1), (s2, c2) = s_r, s_p
    if s1 == 0.0 and c1 == 0.0:
        mf, Zf = lam * m, lam * Z
    else:
        X = Z - rec_powint(rb, s1, c1, xc, l, a, b, ph.n)
        mf = rec_int(rb, s1, c1, xc, l, lo, hi)
        Zf = rec_powint(rb, s1, c1, xc, l, lo, hi, ph.n) + lam * X
    pint = rec_int(p, s2, c2, xc, l, lo, hi)
    return mf, Zf, pint / ph.G + ph.C * Zf


def _mass_at(z, s_r, x):
    """Mass of zone z from its left end up to position x."""
    a = z[1]
    l = z[2] - z[1]
    rb, xc = z[3] / l, 0.5 * (z[1] + z[2])
    return rec_int(rb, s_r[0], s_r[1], xc, l, a, x)


# ------------------------------------------------------------------ remap
def remap(st, lz, wL, M, phases, N, recon=True, vslope=True):
    h = 1.0 / N
    B = len(lz)
    sr, sp = _slopes(lz, phases, recon)
    # new boundaries: Eulerian faces + interface positions (Lagrangian, unchanged by the remap)
    ifc = [lz[b][1] for b in range(B) if not st.isface[b]]
    faces = [c * h for c in range(N)]
    x0 = lz[0][1]
    span_lo, span_hi = x0, x0 + 1.0
    cand = []
    for f in faces:
        for sh in (-1.0, 0.0, 1.0):
            if span_lo - 1e-14 <= f + sh < span_hi - 1e-14:
                cand.append((f + sh, True))
    for x in ifc:
        cand.append((x, False))
    cand.sort()
    # merge interface points that coincide with a face (keep the interface)
    newb = []
    for x, isf in cand:
        if newb and abs(x - newb[-1][0]) < 1e-13:
            if not isf:
                newb[-1] = (x, False)
            continue
        newb.append((x, isf))
    def contents(lzz, sr_, sp_):
        out = []
        j = 0
        for i, (xa, isf) in enumerate(newb):
            xb = newb[i + 1][0] if i + 1 < len(newb) else newb[0][0] + 1.0
            while lzz[j % B][2] + (j // B) * 1.0 <= xa + 1e-15:
                j += 1
            mtot = Ztot = Etot = 0.0
            jj = j
            k = None
            while True:
                z = lzz[jj % B]
                off = (jj // B) * 1.0
                a, b = z[1] + off, z[2] + off
                lo, hi = max(a, xa), min(b, xb)
                if hi > lo:
                    zz = [z[0], a, b, z[3], z[4], z[5]]
                    mf, Zf, Ef = _fragment(zz, phases[z[0]], sr_[jj % B], sp_[jj % B], lo, hi)
                    mtot += mf
                    Ztot += Zf
                    Etot += Ef
                    k = z[0]
                if b >= xb - 1e-15:
                    break
                jj += 1
            out.append([k, xa, xb, mtot, Ztot, Etot])
        return out
    newzones = contents(lz, sr, sp)
    if False:
     j = 0
     for i, (xa, isf) in enumerate(newb):
        xb = newb[i + 1][0] if i + 1 < len(newb) else newb[0][0] + 1.0
        while lz[j % B][2] + (j // B) * 1.0 <= xa + 1e-15:
            j += 1
        mtot = Ztot = Etot = 0.0
        jj = j
        k = None
        while True:
            z = lz[jj % B]
            off = (jj // B) * 1.0
            a, b = z[1] + off, z[2] + off
            lo, hi = max(a, xa), min(b, xb)
            if hi > lo:
                zz = [z[0], a, b, z[3], z[4], z[5]]
                mf, Zf, Ef = _fragment(zz, phases[z[0]], sr[jj % B], sp[jj % B], lo, hi)
                mtot += mf
                Ztot += Zf
                Etot += Ef
                k = z[0]
            if b >= xb - 1e-15:
                break
            jj += 1
        newzones.append([k, xa, xb, mtot, Ztot, Etot])
    # normalise positions to start in [0, 1)
    shift = np.floor(newzones[0][1])
    for z in newzones:
        z[1] -= shift
        z[2] -= shift
    isface = [isf for _, isf in newb]

    # ---------------- nodal momentum re-partition in the mass coordinate ----------------
    # anchor: first interface boundary (present in both partitions)
    # old dual zones (Lagrangian): node b owns [mu_b - m_{b-1}/2, mu_b + m_b/2]
    mo = np.array([z[3] for z in lz])
    muo = np.concatenate([[0.0], np.cumsum(mo)])[:-1]          # mass coordinate of old boundary b
    Mtot = mo.sum()
    # mass coordinate of new boundaries relative to the same material origin (old boundary 0 at x0)
    mn = np.array([z[3] for z in newzones])
    # position of old boundary 0 is lz[0][1]; locate it in the new partition
    x_anchor = newzones[0][1] + ((lz[0][1] - shift - newzones[0][1]) % 1.0)
    if x_anchor >= newzones[-1][2] - 1e-15:                    # anchor rounded up to the period end: wrap to the start
        x_anchor -= 1.0
    jz = 0
    while not (newzones[jz][1] - 1e-15 <= x_anchor < newzones[jz][2] - 1e-15):
        jz += 1
    zN = newzones[jz]
    # mass of the new zone from its left end to the anchor, measured with the old fields
    acc = 0.0
    for jj in range(B):
        z = lz[jj]
        a, b = z[1] - shift, z[2] - shift
        for off in (-1.0, 0.0, 1.0):
            lo, hi = max(a + off, zN[1]), min(b + off, x_anchor)
            if hi > lo:
                acc += _mass_at([z[0], a + off, b + off, z[3]], sr[jj], hi) - _mass_at([z[0], a + off, b + off, z[3]], sr[jj], lo)
    mun = np.concatenate([[0.0], np.cumsum(mn)])[:-1]
    mun = (mun - (mun[jz] + acc)) % Mtot                         # new boundary mass coordinates
    # old velocity field per old dual zone, linear in mu about the dual-zone centroid
    dual_lo = (muo - 0.5 * np.roll(mo, 1)) % Mtot
    Mo = 0.5 * (mo + np.roll(mo, 1))
    # velocity slopes in mu about the node; interface nodes get one-sided (kinked) profiles because
    # u_x jumps across a contact; delta keeps the dual-zone mean equal to the node velocity
    sL, sR, dlt = np.zeros(B), np.zeros(B), np.zeros(B)
    if vslope:
        for b in range(B):
            bl, br, bll, brr = (b - 1) % B, (b + 1) % B, (b - 2) % B, (b + 2) % B
            mL, mR = mo[(b - 1) % B], mo[b]
            dl = (wL[b] - wL[bl]) / mL
            dr = (wL[br] - wL[b]) / mR
            if st.isface[b]:
                left_ok = st.isface[bl] or True
                sL[b] = sR[b] = _minmod(dl, dr)
            else:
                dll = (wL[bl] - wL[bll]) / mo[(b - 2) % B]
                drr = (wL[brr] - wL[br]) / mo[(b + 1) % B]
                sL[b] = _minmod(dl, dll) if st.isface[bl] else dl
                sR[b] = _minmod(dr, drr) if st.isface[br] else dr
            dlt[b] = -(sR[b] * mR ** 2 / 8 - sL[b] * mL ** 2 / 8) / Mo[b]
    cL, cR = np.zeros(B), np.zeros(B)
    if vslope and PARABOLIC[0] and VCURV[0]:
        # one-sided parabolic (in mu) velocity profiles at interface nodes (u_x jumps across a
        # contact); monotone-limited
        for b in range(B):
            bl, br, bll, brr = (b - 1) % B, (b + 1) % B, (b - 2) % B, (b + 2) % B
            mL1, mR1, mL2, mR2 = mo[bl], mo[b], mo[bll], mo[br]
            dl = (wL[b] - wL[bl]) / mL1
            dr = (wL[br] - wL[b]) / mR1
            if st.isface[b]:
                # face nodes keep the minmod-linear profile (a central slope + curvature here was
                # measured anti-diffusive: unstable at acoustic CFL ~5, section 57)
                continue
            else:
                dll = (wL[bl] - wL[bll]) / mL2
                drr = (wL[brr] - wL[br]) / mR2
                if st.isface[bl] and dl * dll > 0:
                    c = (dl - dll) / (mL1 + mL2)
                    s0 = dl + c * mL1
                    if s0 * dl > 0:
                        cm = abs(s0) / mL1
                        sL[b], cL[b] = s0, float(np.clip(c, -cm, cm))
                if st.isface[br] and dr * drr > 0:
                    c = (drr - dr) / (mR1 + mR2)
                    s0 = dr - c * mR1
                    if s0 * dr > 0:
                        cm = abs(s0) / mR1
                        sR[b], cR[b] = s0, float(np.clip(c, -cm, cm))
            aL, bR = 0.5 * mL1, 0.5 * mR1
            left = -sL[b] * aL ** 2 / 2 + cL[b] * aL ** 3 / 3
            right = sR[b] * bR ** 2 / 2 + cR[b] * bR ** 3 / 3
            dlt[b] = -(left + right) / Mo[b]

    # extended (periodic) list of old dual intervals sorted in mu for local search
    order = np.argsort(dual_lo)
    ext_lo = np.concatenate([dual_lo[order] - Mtot, dual_lo[order], dual_lo[order] + Mtot])
    ext_b = np.concatenate([order, order, order])
    ext_off = np.concatenate([np.full(B, -Mtot), np.zeros(B), np.full(B, Mtot)])

    def old_avg(mu_a, L):
        """old field on the mass interval [mu_a, mu_a + L] (periodic): mass, momentum, fragments."""
        out_m = out_p = out_k = 0.0
        frags = []
        i0 = max(np.searchsorted(ext_lo, mu_a, side="right") - 1, 0)
        i = i0
        while i < len(ext_lo) and ext_lo[i] < mu_a + L:
            b = ext_b[i]
            s_lo = ext_lo[i]
            lo, hi = max(s_lo, mu_a), min(s_lo + Mo[b], mu_a + L)
            if hi > lo:
                node = s_lo + 0.5 * mo[(b - 1) % B]
                a1, b1 = lo - node, hi - node

                def seg(s_, c_, u, v):
                    return s_ * (v * v - u * u) / 2 + c_ * (v ** 3 - u ** 3) / 3
                if b1 <= 0:
                    tot = seg(sL[b], cL[b], a1, b1)
                elif a1 >= 0:
                    tot = seg(sR[b], cR[b], a1, b1)
                else:
                    tot = seg(sL[b], cL[b], a1, 0.0) + seg(sR[b], cR[b], 0.0, b1)
                wbar = wL[b] + dlt[b] + tot / (b1 - a1)
                dm = hi - lo
                out_m += dm
                out_p += dm * wbar
                out_k += dm * wbar ** 2
                frags.append((b, dm, wbar, lo, hi))
            i += 1
        return out_m, out_p, out_k, frags

    mnz = mn
    Bn = len(newzones)
    nd_lo = (mun - 0.5 * np.roll(mnz, 1)) % Mtot
    Mn = 0.5 * (mnz + np.roll(mnz, 1))

    def repartition():
        wn_ = np.zeros(Bn)
        heat_ = np.zeros(Bn)
        split_ = np.zeros(B)
        for bnew in range(Bn):
            mm, pp, kk, frags = old_avg(nd_lo[bnew], Mn[bnew])
            wn_[bnew] = pp / mm
            heat_[bnew] = 0.5 * kk - 0.5 * mm * wn_[bnew] ** 2          # mixing gap >= 0
            for (b, dm, wbar, lo, hi) in frags:
                split_[b] += 0.5 * dm * wbar ** 2
        return wn_, heat_, split_ - 0.5 * Mo * wL ** 2                    # splitting gap G_b >= 0

    wn, heat_new, G_old = repartition()
    # limit the velocity slopes so that the splitting gap G_b (scales as phi_b^2) taken from the two
    # adjacent Lagrangian zones (mass-half shares) is at most 1/4 of each zone's thermal margin
    margin = np.empty(B)
    for zi_ in range(B):
        z = lz[zi_]
        ph = phases[z[0]]
        l = z[2] - z[1]
        rb = z[3] / l
        s1, c1 = sr[zi_]
        X = z[4] - rec_powint(rb, s1, c1, 0.0, l, -l / 2, l / 2, ph.n)
        need = ph.A * ((rb + abs(s1) * l / 2 + c1 * l * l / 6) ** ph.n + X / l)
        margin[zi_] = max(zone_p(z, ph) - need / 0.98, 0.0) * l / ph.G
    phi = np.ones(B)
    for b in range(B):
        if G_old[b] > 0:
            fl = 0.5 * mo[(b - 1) % B] / Mo[b]
            fr = 0.5 * mo[b] / Mo[b]
            cap = min(0.25 * margin[(b - 1) % B] / (G_old[b] * fl), 0.25 * margin[b] / (G_old[b] * fr))
            phi[b] = min(1.0, np.sqrt(cap))
    if np.any(phi < 1.0):
        sL *= phi
        sR *= phi
        cL *= phi
        cR *= phi
        dlt *= phi
        wn, heat_new, G_old = repartition()
    # subtract the splitting gaps from the Lagrangian zones BEFORE the energy re-partition, then
    # recompute the (limited) pressure slopes; densities, masses and Z are unchanged
    lz2 = [list(z) for z in lz]
    for b in range(B):
        if G_old[b] != 0:
            lz2[(b - 1) % B][5] -= G_old[b] * 0.5 * mo[(b - 1) % B] / Mo[b]
            lz2[b][5] -= G_old[b] * 0.5 * mo[b] / Mo[b]
    if np.any(G_old != 0):
        sp2 = sp.copy()
        for zi_ in range(B):
            z = lz2[zi_]
            ph = phases[z[0]]
            l = z[2] - z[1]
            rb = z[3] / l
            s1, c1 = sr[zi_]
            X = z[4] - rec_powint(rb, s1, c1, 0.0, l, -l / 2, l / 2, ph.n)
            need = ph.A * ((rb + abs(s1) * l / 2 + c1 * l * l / 6) ** ph.n + X / l)
            room = zone_p(z, ph) - need / 0.98
            if abs(sp2[zi_][0]) * l / 2 + abs(sp2[zi_][1]) * l * l / 6 > room:
                sp2[zi_] = np.array([np.sign(sp2[zi_][0]) * max(room, 0.0) * 2 / l, 0.0])
        ez = contents(lz2, sr, sp2)
        for zn, q in zip(newzones, ez):
            zn[5] = q[5]
    for bnew in range(Bn):
        newzones[(bnew - 1) % Bn][5] += 0.5 * heat_new[bnew]
        newzones[bnew][5] += 0.5 * heat_new[bnew]
    if False:
        # distribute -G_old over the new zones by mass overlap in mu (fragments are already local)
        zone_lo = np.concatenate([mun - Mtot, mun, mun + Mtot])
        zone_id = np.concatenate([np.arange(Bn)] * 3)
        zorder = np.argsort(zone_lo)
        zl, zi = zone_lo[zorder], zone_id[zorder]
        for (b, lo, hi) in frag_log:
            if G_old[b] == 0:
                continue
            j0 = max(np.searchsorted(zl, lo, side="right") - 1, 0)
            j = j0
            while j < len(zl) and zl[j] < hi:
                zn = zi[j]
                a_, b_ = max(zl[j], lo), min(zl[j] + mnz[zn], hi)
                if b_ > a_:
                    newzones[zn][5] -= G_old[b] * (b_ - a_) / Mo[b]
                j += 1
    return State(newzones, isface, wn), heat_new.sum() - G_old.sum()


def step(st, tau, phases, N, recon=True, vslope=True):
    lz, wL, M, it = lagrange(st, tau, phases)
    new, q = remap(st, lz, wL, M, phases, N, recon, vslope)
    return new, it


def totals(st):
    m = np.array([z[3] for z in st.zones])
    M = 0.5 * (m + np.roll(m, 1))
    return m.sum(), np.sum(M * st.w), sum(z[5] for z in st.zones) + np.sum(0.5 * M * st.w ** 2)


def min_theta(st, phases):
    worst = np.inf
    for z in st.zones:
        ph = phases[z[0]]
        l = z[2] - z[1]
        p = zone_p(z, ph)
        worst = min(worst, (p * l - ph.A * z[4]) / (p * l))
    return worst
