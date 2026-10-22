"""Minimal oracle for the piecewise-moment LP-IMEX transport (L0 closure).

Verifies, in the equilibrium sector (uniform u0, p0, periodic 1D):
  T3  : uniform pressure/velocity preserved exactly with independent Z moments,
        and broken when Z is replaced by the single-density closure V*rho_bar^n.
  CV  : first-order L1 convergence of mass and Z contents with exact interface
        positions, against the bound of Theorem CV (section 23 of
        docs/paper/mg_endpoint_arxiv_20261020/lp_imex_piecewise_audit_ko_20261022.md).

State per cell: ordered pieces (k, a, b, m, Z, E) with absolute positions.
Step = exact translation by delta = nu*h, proportional cut at cell faces,
additive union of adjacent same-material pieces inside a cell.
"""
import sys
from fractions import Fraction

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/cv_t3_verification.png"

# phase k: (A, n, Gamma)
EOS_NEG = [(1.0, 3, 1.5), (0.5, 2, 0.4)]   # C1 = -1/6,  C2 = -3/4
EOS_POS = [(1.0, 3, 3.0), (0.5, 2, 0.4)]   # C1 = +1/6
P0 = 10.0
U0 = 1.0
MAT1 = (0.2, 0.55)                           # material 1 interval, rest is material 2


def C_of(eos):
    A, n, G = eos
    return A * (1.0 / (n - 1) - 1.0 / G)


def rho_field(k, x):
    if k == 0:
        return 1.0 + 0.5 * np.sin(2 * np.pi * x)
    return 0.2 + 0.05 * np.cos(2 * np.pi * x)


GX, GW = np.polynomial.legendre.leggauss(10)
TOL = 1e-10  # float snapping of interface positions to cell faces


def integrate(fun, a, b):
    xm, xr = 0.5 * (a + b), 0.5 * (b - a)
    return xr * np.sum(GW * fun(xm + xr * GX))


def mat_of(x):
    return 0 if MAT1[0] <= (x % 1.0) < MAT1[1] else 1


def material_breaks(a, b, shift=0.0):
    """Material sub-intervals of [a,b) for interfaces translated by shift."""
    ifc = [(s + shift) % 1.0 for s in MAT1]
    pts = sorted({a, b} | {s + j for s in ifc for j in (0, 1, -1) if a + TOL < s + j < b - TOL})
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def project(N, eos, t, profile=True):
    """Exact projection Pi U(t): pieces = exact interfaces cut by cells."""
    h = 1.0 / N
    cells = []
    for c in range(N):
        pcs = []
        for (a, b) in material_breaks(c * h, (c + 1) * h, U0 * t):
            k = mat_of(0.5 * (a + b) - U0 * t)
            n = eos[k][1]
            m = integrate(lambda x: rho_field(k, x - U0 * t), a, b)
            Z = integrate(lambda x: rho_field(k, x - U0 * t) ** n, a, b)
            pcs.append([k, a, b, m, Z, 0.0])
        cells.append(pcs)
    set_energy(cells, eos, [P0] * N)
    return cells


def set_energy(cells, eos, p):
    for c, pcs in enumerate(cells):
        for pc in pcs:
            k, a, b, m, Z = pc[:5]
            A, n, G = eos[k]
            pc[5] = p[c] * (b - a) / G + C_of(eos[k]) * Z + 0.5 * m * U0 ** 2


def step(cells, N, nu):
    h = 1.0 / N
    d = nu * h
    new = [[] for _ in range(N)]
    for c, pcs in enumerate(cells):
        for k, a, b, m, Z, E in pcs:
            a2, b2 = a + d, b + d
            L = b - a
            face = (c + 1) * h
            if abs(a2 - face) < TOL:
                a2 = face
            if abs(b2 - face) < TOL:
                b2 = face
            parts = [(a2, min(b2, face)), (max(a2, face), b2)]
            parts = [(pa, pb) for pa, pb in parts if pb - pa > 0]
            for j, (pa, pb) in enumerate(parts):
                lam = (pb - pa) / (b2 - a2)
                j = 0 if pa < face else 1
                tc = c if j == 0 else c + 1
                off = 0.0
                if tc == N:
                    tc, off = 0, 1.0
                new[tc].append([k, pa - off, pb - off, lam * m, lam * Z, lam * E])
    for c in range(N):
        pcs = sorted(new[c], key=lambda q: q[1])
        merged = []
        for q in pcs:
            if merged and merged[-1][0] == q[0] and abs(merged[-1][2] - q[1]) < TOL:
                last = merged[-1]
                last[2] = q[2]
                for i in (3, 4, 5):
                    last[i] += q[i]
            else:
                merged.append(list(q))
        new[c] = merged
    return new


def decode(cells, eos, use_Z=True):
    """Cell pressure from total energy; use_Z=False replaces Z by V*rho_bar^n."""
    out = []
    for pcs in cells:
        E = sum(q[5] for q in pcs)
        m = sum(q[3] for q in pcs)
        I = E - 0.5 * m * U0 ** 2
        S = sum((q[2] - q[1]) / eos[q[0]][2] for q in pcs)
        if use_Z:
            CZ = sum(C_of(eos[q[0]]) * q[4] for q in pcs)
        else:
            CZ = 0.0
            for k in (0, 1):
                V = sum(q[2] - q[1] for q in pcs if q[0] == k)
                mk = sum(q[3] for q in pcs if q[0] == k)
                if V > 0:
                    CZ += C_of(eos[k]) * V * (mk / V) ** eos[k][1]
        out.append((I - CZ) / S)
    return np.array(out)


def l1_error(num, ex):
    em = ez = 0.0
    for pn, pe in zip(num, ex):
        assert len(pn) == len(pe), "partition mismatch"
        for qn, qe in zip(pn, pe):
            assert qn[0] == qe[0] and abs(qn[1] - qe[1]) < 1e-9 and abs(qn[2] - qe[2]) < 1e-9
            em += abs(qn[3] - qe[3])
            ez += abs(qn[4] - qe[4])
    return em, ez


def cv_bound(eos, T, nu, h):
    """Theorem CV bound: T*u0*[(1-nu)h/2 * ||g''||_1 + c0*N_I*h*||g'||_inf], summed over fields."""
    xs = np.linspace(0, 1, 4001)
    tot = 0.0
    for k in (0, 1):
        n = eos[k][1]
        for g in (lambda x: rho_field(k, x), lambda x: rho_field(k, x) ** n):
            y = g(xs)
            d1 = np.gradient(y, xs)
            d2 = np.gradient(d1, xs)
            tot += (1 - nu) * h / 2 * np.trapezoid(np.abs(d2), xs) + 6 * 2 * h * np.max(np.abs(d1))
    return T * U0 * tot


def run_cv(eos, nu, T, Ns):
    rows = []
    for N in Ns:
        h = 1.0 / N
        nsteps = int(round(T / (nu * h / U0)))
        cells = project(N, eos, 0.0)
        min_occ = 1.0
        pmax = 0.0
        for _ in range(nsteps):
            cells = step(cells, N, nu)
            for pcs in cells:
                for q in pcs:
                    occ = (q[2] - q[1]) / h
                    if 0 < occ < min_occ:
                        min_occ = occ
            pmax = max(pmax, np.max(np.abs(decode(cells, eos) - P0)) / P0)
        Tn = nsteps * nu * h / U0
        em, ez = l1_error(cells, project(N, eos, Tn))
        rows.append((N, em, ez, cv_bound(eos, Tn, nu, h), min_occ, pmax))
    return rows


def pe_history(eos, N, nu, nsteps):
    cells = project(N, eos, 0.0)
    withZ, noZ = [], []
    for _ in range(nsteps):
        cells = step(cells, N, nu)
        withZ.append(np.max(np.abs(decode(cells, eos, True) - P0)) / P0)
        noZ.append(np.max(np.abs(decode(cells, eos, False) - P0)) / P0)
    return np.array(withZ), np.array(noZ)


def exact_rational_t3(N=16, nsteps=40):
    """Piecewise-constant rational densities, rational nu: pressure must be exactly P0."""
    F = Fraction
    eos = [(F(1), 3, F(3, 2)), (F(1, 2), 2, F(2, 5))]
    C = [A * (F(1, n - 1) - 1 / G) for (A, n, G) in eos]
    h, nu, p0, u0 = F(1, N), F(1, 3), F(10), F(1)
    dens = [F(7, 5), F(11, 10), F(1, 4), F(1, 5)]
    cells = []
    for c in range(N):
        a, b = c * h, (c + 1) * h
        k = 0 if 3 <= c < 9 else 1
        rho = dens[2 * k + (c % 2)]
        A, n, G = eos[k]
        cells.append([[k, a, b, rho * h, rho ** n * h, p0 * h / G + C[k] * rho ** n * h + rho * h * u0 ** 2 / 2]])
    worst = F(0)
    for _ in range(nsteps):
        d = nu * h
        new = [[] for _ in range(N)]
        for c, pcs in enumerate(cells):
            for k, a, b, m, Z, E in pcs:
                a2, b2, face = a + d, b + d, (c + 1) * h
                for j, (pa, pb) in enumerate([(a2, min(b2, face)), (max(a2, face), b2)]):
                    if pb <= pa:
                        continue
                    lam = (pb - pa) / (b - a)
                    tc, off = (c, 0) if j == 0 else ((c + 1) % N, 1 if c + 1 == N else 0)
                    new[tc].append([k, pa - off, pb - off, lam * m, lam * Z, lam * E])
        cells = []
        for pcs in new:
            pcs.sort(key=lambda q: q[1])
            mg = []
            for q in pcs:
                if mg and mg[-1][0] == q[0] and mg[-1][2] == q[1]:
                    mg[-1][2] = q[2]
                    for i in (3, 4, 5):
                        mg[-1][i] += q[i]
                else:
                    mg.append(list(q))
            cells.append(mg)
        for pcs in cells:
            m = sum(q[3] for q in pcs)
            I = sum(q[5] for q in pcs) - m * u0 ** 2 / 2
            S = sum((q[2] - q[1]) / eos[q[0]][2] for q in pcs)
            p = (I - sum(C[q[0]] * q[4] for q in pcs)) / S
            worst = max(worst, abs(p - p0))
    return worst


def main():
    T, nu = 0.5, 0.5
    Ns = [25, 50, 100, 200, 400]
    print("== exact rational T3 ==")
    w = exact_rational_t3()
    print(f"max |p - p0| over all cells and steps (Fraction) = {w}")
    results = {}
    for name, eos in (("C<0", EOS_NEG), ("C1>0", EOS_POS)):
        rows = run_cv(eos, nu, T, Ns)
        results[name] = rows
        print(f"== CV + T3, {name}, nu={nu}, T={T} ==")
        print(" N      L1(m)        L1(Z)        bound        min_occ      max|dp|/p0")
        for N, em, ez, bd, mo, pm in rows:
            print(f"{N:4d} {em:12.4e} {ez:12.4e} {bd:12.4e} {mo:12.4e} {pm:12.4e}")
        r = [np.log2(rows[i][1] / rows[i + 1][1]) for i in range(len(rows) - 1)]
        rz = [np.log2(rows[i][2] / rows[i + 1][2]) for i in range(len(rows) - 1)]
        print("  order m:", " ".join(f"{x:.3f}" for x in r))
        print("  order Z:", " ".join(f"{x:.3f}" for x in rz))
        print("  error <= bound:", all(e1 + e2 <= b for _, e1, e2, b, _, _ in rows))
    rows_nu = run_cv(EOS_NEG, 0.2, T, Ns[:4])
    print("== CV, C<0, nu=0.2 ==")
    for N, em, ez, bd, mo, pm in rows_nu:
        print(f"{N:4d} {em:12.4e} {ez:12.4e} {bd:12.4e} {mo:12.4e} {pm:12.4e}")
    wz, nz = pe_history(EOS_NEG, 100, nu, 200)
    print(f"== T3 ablation N=100: max dp/p0 with Z = {wz.max():.3e}, Z:=V*rho_bar^n = {nz.max():.3e}")
    global MAT1
    MAT1 = (0.2 + 1e-5, 0.55 + 3e-6)
    rows_sl = run_cv(EOS_NEG, nu, T, Ns[:4])
    print("== CV sliver interfaces (min occupancy ~1e-4), C<0, nu=0.5 ==")
    for N, em, ez, bd, mo, pm in rows_sl:
        print(f"{N:4d} {em:12.4e} {ez:12.4e} {bd:12.4e} {mo:12.4e} {pm:12.4e}")
    MAT1 = (0.2, 0.55)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, mk in (("C<0", "o"), ("C1>0", "s")):
        rows = results[name]
        h = [1.0 / r[0] for r in rows]
        ax[0].loglog(h, [r[1] + r[2] for r in rows], mk + "-", label=f"L1 error (m+Z), {name}")
    rows = results["C<0"]
    h = np.array([1.0 / r[0] for r in rows])
    ax[0].loglog(h, [r[3] for r in rows], "k--", label="Theorem CV bound")
    ax[0].loglog(h, rows[0][1] * 2 * h / h[0], ":", color="gray", label="slope 1")
    ax[0].set_xlabel("h")
    ax[0].set_ylabel("L1 error at T=0.5")
    ax[0].set_title("CV: first-order L1, exact interfaces (nu=0.5)")
    ax[0].legend(fontsize=8)
    ax[1].semilogy(np.maximum(wz, 1e-17), label="independent Z (T3)")
    ax[1].semilogy(nz, label="Z := V rho_bar^n (no moment)")
    ax[1].set_xlabel("step (N=100, nu=0.5)")
    ax[1].set_ylabel("max |p - p0| / p0")
    ax[1].set_title("T3: pressure equilibrium, C != 0")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    sys.exit(main())
