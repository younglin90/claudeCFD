"""Oracle for the staggered LP-IMEX step: Hamiltonian AVF Lagrange stage + piece remap
+ nodal momentum remap with Jensen heat (sections 14, 17, 21, 24 of
docs/paper/mg_endpoint_arxiv_20261020/lp_imex_piecewise_audit_ko_20261022.md).

Tests (theory vs numerical):
  A  linear acoustic standing wave: measured cos(omega*tau) vs Cayley/midpoint theory,
     for acoustic CFL lambda = c*tau/h from 0.5 to 16 (large time steps, H1)
  B  shock tube, single phase power-law MG: profiles vs exact Riemann solution,
     total-energy / momentum / mass conservation, sign of nodal heat Q (Theorem Q')
  C  uniform (u0, p0) with variable-density two-phase pieces: pressure stays p0 (T3 + acoustic)
  D  mixture isentrope: closed form vs ODE integration; H_c failure (pressure -> negative)
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

PNG = "results/lp_imex_moment/acoustic_theory_vs_numeric.png"
TOL = 1e-10


class Phase:
    def __init__(self, A, n, G):
        self.A, self.n, self.G = A, n, G
        self.C = A * (1.0 / (n - 1) - 1.0 / G)


def isentrope(L0, pcs, phases, p0):
    """Model (Allaire, frozen alpha) isentrope of one cell: returns p(L), E(L)."""
    S0 = sum((q[2] - q[1]) / phases[q[0]].G for q in pcs)
    Gm = L0 / S0
    terms = {}
    for q in pcs:
        terms[q[0]] = terms.get(q[0], 0.0) + q[4]
    terms = [(phases[k].C, phases[k].n, Z0) for k, Z0 in terms.items()]

    def p_of_J(J):
        acc = p0
        for C, n, Z0 in terms:
            a = Gm - n + 1
            F = np.log(J) if abs(a) < 1e-12 else (J ** a - 1.0) / a
            acc += C * (n - 1) * Z0 * F / S0
        return acc * J ** (-(1.0 + Gm))

    def I_of_J(J):
        return J * S0 * p_of_J(J) + sum(C * Z0 * J ** (1 - n) for C, n, Z0 in terms)

    return (lambda L: p_of_J(L / L0)), (lambda L: I_of_J(L / L0))


def isentrope_kapila(L0, pcs, phases, p0):
    """Piecewise-Kapila cell isentrope: every piece (one material, moments l, m, Z)
    compresses uniformly on its own single-material isentrope
        p_r(J) = A zbar J^-n + (p0 - A zbar) J^-(1+Gamma),   zbar = Z/l,
    and all pieces share one pressure. Returns p(L), E(L), J_r(L)."""
    data = []
    for k, a, b, m, Z in pcs:
        ph = phases[k]
        l = b - a
        data.append((ph, l, Z / l, Z))

    def p_piece(d, J):
        ph, l, zb, Z = d
        return ph.A * zb * J ** (-ph.n) + (p0 - ph.A * zb) * J ** (-1.0 - ph.G)

    def J_piece(d, p):
        return np.exp(brentq(lambda y: p_piece(d, np.exp(y)) - p, -40.0, 40.0, xtol=1e-15))

    if len(data) == 1:
        d = data[0]
        Jr = lambda L: [L / d[1]]
        p_of_L = lambda L: p_piece(d, L / d[1])
    else:
        def p_of_L(L):
            g = lambda y: sum(d[1] * J_piece(d, np.exp(y)) for d in data) - L
            lo, hi = np.log(p0) - 1.0, np.log(p0) + 1.0
            while g(lo) < 0:
                lo -= 2.0
            while g(hi) > 0:
                hi += 2.0
            return np.exp(brentq(g, lo, hi, xtol=1e-15))
        Jr = lambda L: [J_piece(d, p_of_L(L)) for d in data] if abs(L - L0) > 1e-15 * L0 else [1.0] * len(data)

    def E_of_L(L):
        p = p_of_L(L)
        return sum(p * d[1] * J / d[0].G + d[0].C * d[3] * J ** (1 - d[0].n) for d, J in zip(data, Jr(L)))

    return p_of_L, E_of_L, Jr


def decode(pcs, I, phases):
    S = sum((q[2] - q[1]) / phases[q[0]].G for q in pcs)
    return (I - sum(phases[q[0]].C * q[4] for q in pcs)) / S


def lagrange(state, tau, phases, visc=None, model="allaire"):
    """AVF Lagrange stage on the staggered ring; Newton on node displacements.

    visc=(c1, c2): implicit compression-only artificial viscosity
      q_i = rho_i (c1 c_i s + c2 s^2),  s = max(-dL_i, 0)/tau,
    acting as an extra cell pressure; its work -q_i dL_i >= 0 is added to the cell
    internal energy, so total energy stays exact and the heat is non-negative.
    """
    cells, I, w = state["cells"], state["I"], state["w"]
    N = len(cells)
    h = 1.0 / N
    m = np.array([sum(q[3] for q in pcs) for pcs in cells])
    M = 0.5 * (m + np.roll(m, -1))
    pf, Ef, Jrf = [], [], []
    for i, pcs in enumerate(cells):
        if model == "kapila":
            p_i, E_i, J_i = isentrope_kapila(h, pcs, phases, decode(pcs, I[i], phases))
        else:
            p_i, E_i = isentrope(h, pcs, phases, decode(pcs, I[i], phases))
            J_i = (lambda nn: (lambda L: [L / h] * nn))(len(pcs))
        Jrf.append(J_i)
        pf.append(p_i)
        Ef.append(E_i)

    def ptilde(i, L):
        dL = L - h
        if abs(dL) > 1e-6 * h:
            pt = -(Ef[i](L) - Ef[i](h)) / dL
            d = (pf[i](L) - pt) / dL
        else:
            pt = (pf[i](h) + 4 * pf[i](h + 0.5 * dL) + pf[i](L)) / 6.0
            e = 1e-6 * h
            d = 0.5 * (pf[i](h + 0.5 * dL + e) - pf[i](h + 0.5 * dL - e)) / (2 * e)
        return pt, d

    rho_c = m / h
    if visc is not None:
        e_ = 1e-6 * h
        csnd = np.array([np.sqrt(max(-h * (pf[i](h + e_) - pf[i](h - e_)) / (2 * e_) / rho_c[i], 0.0)) for i in range(N)])

    def qv(L):
        if visc is None:
            return np.zeros(N), np.zeros(N)
        s_ = np.maximum(h - L, 0.0) / tau
        q = rho_c * (visc[0] * csnd * s_ + visc[1] * s_ ** 2)
        dq = np.where(h - L > 0, -rho_c * (visc[0] * csnd + 2 * visc[1] * s_) / tau, 0.0)
        return q, dq

    D = tau * w.copy()
    it = 0
    for it in range(60):
        L = h + D - np.roll(D, 1)
        if np.any(L <= 0):
            raise RuntimeError("Lagrangian cell collapsed")
        pt = np.empty(N)
        dp = np.empty(N)
        for i in range(N):
            pt[i], dp[i] = ptilde(i, L[i])
        q, dq = qv(L)
        pt = pt + q
        dp = dp + dq
        R = M * (D - tau * w) + 0.5 * tau ** 2 * (np.roll(pt, -1) - pt)
        if np.max(np.abs(R)) < 1e-14 * max(1.0, np.max(M * h)):
            break
        Jm = np.diag(M)
        for j in range(N):
            jp, jm = (j + 1) % N, (j - 1) % N
            c = 0.5 * tau ** 2
            Jm[j, jp] += c * dp[jp]
            Jm[j, j] += -c * dp[jp] - c * dp[j]
            Jm[j, jm] += c * dp[j]
        D = D - np.linalg.solve(Jm, R)
    L = h + D - np.roll(D, 1)
    wn = 2 * D / tau - w
    Jc = L / h
    newI = np.array([Ef[i](L[i]) for i in range(N)])
    pL = np.array([pf[i](L[i]) for i in range(N)])
    Jr = [Jrf[i](L[i]) for i in range(N)]
    heat = np.zeros(N)
    if visc is not None:
        q, _ = qv(L)
        heat = -q * (L - h)
        newI = newI + heat
        for i, pcs in enumerate(cells):
            S = sum((q_[2] - q_[1]) * jr / phases[q_[0]].G for q_, jr in zip(pcs, Jr[i]))
            pL[i] += heat[i] / S
    return dict(D=D, w=wn, J=Jc, Jr=Jr, I=newI, p=pL, M=M, m=m, iters=it, heat=heat)


def remap(state, lag, phases, slope="minmod", keep_piece_energy=False):
    cells = state["cells"]
    N = len(cells)
    h = 1.0 / N
    D, J, pL = lag["D"], lag["J"], lag["p"]
    new = [[] for _ in range(N)]
    F = np.zeros(N)  # mass flux through node j (right face of cell j), + rightward
    for i, pcs in enumerate(cells):
        XL = i * h + D[(i - 1) % N]
        pos = XL
        for (k, a, b, mq, Z), jr in zip(pcs, lag["Jr"][i]):
            ph = phases[k]
            a2, b2 = pos, pos + (b - a) * jr
            pos = b2
            Z2 = Z * jr ** (1 - ph.n)
            e2 = pL[i] * (b2 - a2) / ph.G + ph.C * Z2
            c0, c1 = int(np.floor(a2 / h + TOL)), int(np.floor(b2 / h - TOL))
            for c in range(c0, c1 + 1):
                lo, hi = max(a2, c * h), min(b2, (c + 1) * h)
                if hi - lo <= 0.0:
                    continue
                lam = (hi - lo) / (b2 - a2)
                tc, off = c % N, (c // N) * 1.0
                new[tc].append([k, lo - off, hi - off, lam * mq, lam * Z2, lam * e2])
                if c == i + 1:
                    F[i] += lam * mq
                elif c == i - 1:
                    F[(i - 1) % N] -= lam * mq
    Inew = np.zeros(N)
    out = []
    for c in range(N):
        pcs = sorted(new[c], key=lambda q: q[1])
        mg = []
        for q in pcs:
            if mg and mg[-1][0] == q[0] and abs(mg[-1][2] - q[1]) < TOL:
                mg[-1][2] = q[2]
                for t in (3, 4, 5):
                    mg[-1][t] += q[t]
            else:
                mg.append(list(q))
        Inew[c] = sum(q[5] for q in mg)
        out.append(mg if keep_piece_energy else [q[:5] for q in mg])

    # nodal (dual-cell) momentum remap
    w, M = lag["w"], lag["M"]
    Fd = 0.5 * (np.roll(F, 1) + F)      # through center of cell j: between node j-1 and j
    Fr = np.roll(Fd, -1)                # dual cell j right face = center of cell j+1
    Fl = Fd                             # dual cell j left face = center of cell j
    Mn = M - Fr + Fl
    dw_r = np.roll(w, -1) - w
    dw_l = w - np.roll(w, 1)
    sig = np.where(dw_r * dw_l > 0, np.sign(dw_r) * np.minimum(abs(dw_r), abs(dw_l)), 0.0)
    if slope == "none":
        sig[:] = 0.0
    out_r = np.maximum(Fr, 0)
    out_l = np.maximum(-Fl, 0)
    in_l = np.maximum(Fl, 0)
    in_r = np.maximum(-Fr, 0)
    Mrem = M - out_r - out_l
    if np.any(Mrem <= 0):
        raise RuntimeError("dual CFL violated")
    diverging = (out_r > 0) & (out_l > 0)
    sig[diverging] = 0.0
    nu_o_r, nu_o_l = out_r / M, out_l / M
    # limiter factor of Theorem Q (ii) for monotone dual cells
    fac = np.ones(N)
    Mp = Mn
    for j in range(N):
        if out_r[j] > 0 and in_l[j] > 0:
            nu_i = in_l[j] / M[(j - 1) % N]
            fac[j] = min(1.0, (1 + nu_i - nu_o_r[j]) * np.sqrt(in_l[j] * M[j] / (out_r[j] * Mp[j])))
        elif out_l[j] > 0 and in_r[j] > 0:
            nu_i = in_r[j] / M[(j + 1) % N]
            fac[j] = min(1.0, (1 + nu_i - nu_o_l[j]) * np.sqrt(in_r[j] * M[j] / (out_l[j] * Mp[j])))
    sig = sig * fac
    w_or = w + 0.5 * (1 - nu_o_r) * sig     # right outflow fragment mean
    w_ol = w - 0.5 * (1 - nu_o_l) * sig     # left outflow fragment mean
    Prem = M * w - out_r * w_or - out_l * w_ol
    w_rem = Prem / Mrem
    # inflow fragments are the neighbours' outflows
    w_il = np.roll(w_or, 1)
    w_ir = np.roll(w_ol, -1)
    P = Prem + in_l * w_il + in_r * w_ir
    wn = P / Mn
    ke_frag = 0.5 * Mrem * w_rem ** 2 + 0.5 * in_l * w_il ** 2 + 0.5 * in_r * w_ir ** 2
    Qmix = ke_frag - 0.5 * Mn * wn ** 2
    Gsplit = 0.5 * Mrem * w_rem ** 2 + 0.5 * out_r * w_or ** 2 + 0.5 * out_l * w_ol ** 2 - 0.5 * M * w ** 2
    Q = Qmix - Gsplit
    heatc = 0.5 * Q + 0.5 * np.roll(Q, 1)
    Inew += heatc
    if keep_piece_energy:
        for c, pcs in enumerate(out):
            S = sum((q[2] - q[1]) / phases[q[0]].G for q in pcs)
            for q in pcs:
                q[5] += heatc[c] * (q[2] - q[1]) / phases[q[0]].G / S
    return dict(cells=out, I=Inew, w=wn), Q


def totals(state, phases):
    cells, I, w = state["cells"], state["I"], state["w"]
    m = np.array([sum(q[3] for q in pcs) for pcs in cells])
    M = 0.5 * (m + np.roll(m, -1))
    return m.sum(), np.sum(M * w), I.sum() + np.sum(0.5 * M * w ** 2)


def pressures(state, phases):
    return np.array([decode(pcs, I, phases) for pcs, I in zip(state["cells"], state["I"])])


def init_state(N, phases, rho_fun, p_fun, u_fun, mat_fun=lambda x: 0):
    h = 1.0 / N
    cells, I = [], np.zeros(N)
    for c in range(N):
        xc = (c + 0.5) * h
        k = mat_fun(xc)
        ph = phases[k]
        rho, p = rho_fun(xc), p_fun(xc)
        cells.append([[k, c * h, (c + 1) * h, rho * h, rho ** ph.n * h]])
        I[c] = p * h / ph.G + ph.C * rho ** ph.n * h
    w = np.array([u_fun((j + 1) * h) for j in range(N)])
    return dict(cells=cells, I=I, w=w)


def run(state, phases, tau, nsteps, slope="minmod", hook=None, visc=None, model="allaire"):
    Qmin, Qscale, it_max = 0.0, 0.0, 0
    for n in range(nsteps):
        lag = lagrange(state, tau, phases, visc, model)
        if lag["heat"].min() < -1e-14:
            raise RuntimeError("negative viscous heat")
        it_max = max(it_max, lag["iters"])
        state, Q = remap(state, lag, phases, slope)
        Qmin = min(Qmin, Q.min())
        Qscale = max(Qscale, np.abs(Q).max())
        if hook:
            hook(n, state)
    return state, Qmin, Qscale, it_max


# ---------------- exact Riemann solver, single power-law MG phase ----------------
def riemann_exact(ph, WL, WR, xs, t, x0=0.5):
    A, n, G, C = ph.A, ph.n, ph.G, ph.C

    def e(r, p):
        return p / (G * r) + C * r ** (n - 1)

    def c_isen(r, ra, pa):
        return np.sqrt(n * A * r ** (n - 1) + (1 + G) * (pa - A * ra ** n) * r ** G / ra ** (1 + G))

    def p_isen(r, ra, pa):
        return A * r ** n + (pa - A * ra ** n) * (r / ra) ** (1 + G)

    def wave(ps, rK, pK):
        if ps > pK:
            rs = brentq(lambda r: e(r, ps) - e(rK, pK) - 0.5 * (pK + ps) * (1 / rK - 1 / r), rK, 50 * rK)
            return np.sqrt((ps - pK) * (1 / rK - 1 / rs)), rs
        rs = brentq(lambda r: p_isen(r, rK, pK) - ps, 1e-8, rK)
        return -quad(lambda r: c_isen(r, rK, pK) / r, rs, rK)[0], rs

    rL, uL, pL = WL
    rR, uR, pR = WR
    ps = brentq(lambda p: wave(p, rL, pL)[0] + wave(p, rR, pR)[0] + uR - uL, 0.5 * min(pL, pR), 5 * max(pL, pR))
    fL, rsL = wave(ps, rL, pL)
    fR, rsR = wave(ps, rR, pR)
    us = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    out = np.zeros((len(xs), 3))
    for i, x in enumerate(xs):
        xi = (x - x0) / t
        if xi < us:
            if ps <= pL:  # left rarefaction
                head, tail = uL - c_isen(rL, rL, pL), us - c_isen(rsL, rL, pL)
                if xi < head:
                    out[i] = (rL, uL, pL)
                elif xi > tail:
                    out[i] = (rsL, us, ps)
                else:
                    def g(r):
                        u = uL + quad(lambda s: c_isen(s, rL, pL) / s, r, rL)[0]
                        return u - c_isen(r, rL, pL) - xi
                    r = brentq(g, rsL, rL)
                    out[i] = (r, xi + c_isen(r, rL, pL), p_isen(r, rL, pL))
            else:
                s = uL - np.sqrt((ps - pL) / (1 / rL - 1 / rsL)) / rL
                out[i] = (rL, uL, pL) if xi < s else (rsL, us, ps)
        else:
            if ps > pR:  # right shock
                s = uR + np.sqrt((ps - pR) / (1 / rR - 1 / rsR)) / rR
                out[i] = (rsR, us, ps) if xi < s else (rR, uR, pR)
            else:
                head, tail = uR + c_isen(rR, rR, pR), us + c_isen(rsR, rR, pR)
                if xi > head:
                    out[i] = (rR, uR, pR)
                elif xi < tail:
                    out[i] = (rsR, us, ps)
                else:
                    def g(r):
                        u = uR - quad(lambda s: c_isen(s, rR, pR) / s, r, rR)[0]
                        return u + c_isen(r, rR, pR) - xi
                    r = brentq(g, rsR, rR)
                    out[i] = (r, xi - c_isen(r, rR, pR), p_isen(r, rR, pR))
    return out, ps, us


def main():
    ph1 = Phase(1.0, 3, 1.5)
    phases = [ph1, Phase(0.5, 2, 0.4)]
    rho0, p0 = 1.0, 10.0
    K = ph1.n * ph1.A * rho0 ** ph1.n + (1 + ph1.G) * (p0 - ph1.A * rho0 ** ph1.n)
    c0 = np.sqrt(K / rho0)

    # ---------------- A: acoustic dispersion vs Cayley ----------------
    N, eps = 40, 1e-5
    h = 1.0 / N
    wh = 2 * c0 / h * np.sin(np.pi * h)
    lams = [0.5, 1, 2, 4, 8, 16]
    meas, theo, edrift = [], [], []
    xn = (np.arange(N) + 1) * h
    for lam in lams:
        tau = lam * h / c0
        st = init_state(N, phases, lambda x: rho0, lambda x: p0, lambda x: eps * np.sin(2 * np.pi * x))
        e0 = totals(st, phases)[2]
        a = []
        st, Qmin, _, _ = run(st, phases, tau, 60, hook=lambda n, s: a.append(np.sum(s["w"] * np.sin(2 * np.pi * xn))))
        a = np.array(a)
        sel = [i for i in range(1, len(a) - 1) if abs(a[i]) > 0.3 * np.max(np.abs(a))]
        meas.append(np.median([(a[i + 1] + a[i - 1]) / (2 * a[i]) for i in sel]))
        theo.append((1 - (wh * tau / 2) ** 2) / (1 + (wh * tau / 2) ** 2))
        edrift.append(abs(totals(st, phases)[2] - e0) / e0)
    print("== A: acoustic standing wave, cos(omega*tau): measured vs Cayley theory ==")
    for lam, mm, tt, ed in zip(lams, meas, theo, edrift):
        print(f" lambda={lam:5.1f}  measured={mm:+.6f}  theory={tt:+.6f}  |diff|={abs(mm-tt):.2e}  energy drift={ed:.2e}")

    # ---------------- B: shock tube vs exact Riemann ----------------
    N = 200
    h = 1.0 / N
    WL, WR = (1.0, 0.0, 20.0), (1.0, 0.0, 10.0)
    cL = np.sqrt(ph1.n * ph1.A + (1 + ph1.G) * (WL[2] - ph1.A))
    Tend = 0.04
    resB = {}
    for lam in (0.8, 3.0):
        tau = lam * h / cL
        nst = int(np.ceil(Tend / tau))
        tau = Tend / nst
        # left half [0, 0.5) high pressure, right half low; second jump at x=0 lies outside the window
        st = init_state(N, phases, lambda x: 1.0, lambda x: 20.0 if x < 0.5 else 10.0, lambda x: 0.0)
        tot0 = totals(st, phases)
        for slope in ("minmod",):
            s2, Qmin, Qs, itm = run(st, phases, tau, nst, slope)
            tot1 = totals(s2, phases)
            resB[lam] = (s2, Qmin, Qs, itm, tot0, tot1, tau)
    xs = (np.arange(N) + 0.5) / N
    exact, ps, us = riemann_exact(ph1, WL, WR, xs, Tend)
    print("== B: shock tube (p 20 | 10), T=0.04, exact p*=%.5f u*=%.5f ==" % (ps, us))
    win = (xs > 0.29) & (xs < 0.75)
    for lam, (s2, Qmin, Qs, itm, t0, t1, tau) in resB.items():
        rho = np.array([sum(q[3] for q in pcs) for pcs in s2["cells"]]) / h
        p = pressures(s2, phases)
        l1r = np.sum(np.abs(rho - exact[:, 0])[win]) * h
        l1p = np.sum(np.abs(p - exact[:, 2])[win]) * h
        print(f" lambda={lam}: dmass={abs(t1[0]-t0[0]):.1e} dmom={abs(t1[1]-t0[1]):.1e} "
              f"dE/E={abs(t1[2]-t0[2])/t0[2]:.1e}  minQ/maxQ={Qmin/max(Qs,1e-300):.2e}  newton<= {itm}  "
              f"L1 rho={l1r:.3e} L1 p={l1p:.3e}")

    # ---------------- C: uniform u0, p0 with two-phase variable pieces ----------------
    N, u0 = 50, 0.7
    h = 1.0 / N
    st = init_state(N, phases, lambda x: 1.0 + 0.3 * np.sin(2 * np.pi * x) if 0.2 <= x < 0.6 else 0.2 + 0.05 * np.cos(2 * np.pi * x),
                    lambda x: p0, lambda x: u0, mat_fun=lambda x: 0 if 0.2 <= x < 0.6 else 1)
    worst = [0.0]
    st, Qmin, Qs, itm = run(st, phases, 0.4 * h / u0, 100,
                            hook=lambda n, s: worst.__setitem__(0, max(worst[0], np.max(np.abs(pressures(s, phases) - p0)) / p0)))
    print(f"== C: uniform u0={u0}, p0={p0}, two phases, 100 steps: max|dp|/p0={worst[0]:.2e}, max|w-u0|={np.max(np.abs(st['w']-u0)):.2e}")

    # ---------------- D: mixture isentrope, H_c failure ----------------
    phD = [Phase(1.0, 3, 3.0), Phase(0.5, 2, 0.4)]
    pcs = [[0, 0.0, 0.5, 0.5, 0.5], [1, 0.5, 1.0, 0.1, 0.02]]
    pD = 1.5
    p_of_L, _ = isentrope(1.0, pcs, phD, pD)
    S0 = 0.5 / 3.0 + 0.5 / 0.4
    Z0 = {0: 0.5, 1: 0.02}

    def ode(J, y):
        return [-(1 + 1.0 / S0) * y[0] / J + sum(phD[k].C * (phD[k].n - 1) * Z0[k] * J ** (-phD[k].n - 1) for k in (0, 1)) / S0]
    Js = np.linspace(1.0, 0.01, 400)
    sol = solve_ivp(ode, (1.0, 0.01), [pD], t_eval=Js, rtol=1e-11, atol=1e-12)
    pc = np.array([p_of_L(J) for J in Js])
    Jstar = brentq(lambda J: p_of_L(J), 0.01, 1.0) if p_of_L(0.01) < 0 else np.nan
    print(f"== D: mixture isentrope closed form vs ODE: max rel diff={np.max(np.abs(pc-sol.y[0])/np.abs(sol.y[0]).max()):.2e}; "
          f"Gamma_m={1/S0:.3f}, pressure crosses 0 at J*={Jstar:.4f}")

    NB = 200
    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax[0, 0].plot(lams, theo, "k-", label="theory: Cayley (1-(w_h tau/2)^2)/(1+(w_h tau/2)^2)")
    ax[0, 0].plot(lams, meas, "o", label="measured (2-step recurrence)")
    ax[0, 0].plot(lams, np.cos(2 * np.pi * c0 * np.array(lams) * (1.0 / 40) / c0), "--", color="gray", label="exact PDE cos(omega tau)")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("acoustic CFL  lambda = c tau / h")
    ax[0, 0].set_ylabel("cos(omega tau)")
    ax[0, 0].set_title("A: acoustic phase per step, N=40")
    ax[0, 0].legend(fontsize=7)
    s2 = resB[3.0][0]
    s1 = resB[0.8][0]
    for s, lab, mk in ((s1, "numerical lambda=0.8", "."), (s2, "numerical lambda=3", "x")):
        rho = np.array([sum(q[3] for q in pcs) for pcs in s["cells"]]) * NB
        ax[0, 1].plot(xs, rho, mk, ms=4, label=lab)
        ax[0, 2].plot(xs, pressures(s, phases), mk, ms=4, label=lab)
        ax[1, 0].plot((np.arange(NB) + 1) / NB, s["w"], mk, ms=4, label=lab)
    ax[0, 1].plot(xs, exact[:, 0], "k-", lw=1.5, label="exact Riemann")
    ax[0, 2].plot(xs, exact[:, 2], "k-", lw=1.5, label="exact Riemann")
    ax[1, 0].plot(xs, exact[:, 1], "k-", lw=1.5, label="exact Riemann")
    for a_, t_ in ((ax[0, 1], "B: density"), (ax[0, 2], "B: pressure"), (ax[1, 0], "B: velocity")):
        a_.set_xlim(0.29, 0.75)
        a_.set_title(t_ + " at T=0.04")
        a_.legend(fontsize=7)
    names = ["mass", "momentum", "total energy"]
    for lam, mk in ((0.8, "o"), (3.0, "s")):
        _, Qmin, Qs, _, t0, t1, _ = resB[lam]
        rel = [abs(t1[i] - t0[i]) / max(abs(t0[i]), 1e-300) if i != 1 else abs(t1[1] - t0[1]) for i in range(3)]
        ax[1, 1].semilogy(names, np.maximum(rel, 1e-17), mk, ms=8, label=f"lambda={lam}")
    ax[1, 1].axhline(1e-12, color="gray", ls=":")
    ax[1, 1].set_title("B: conservation error (theory: 0)")
    ax[1, 1].legend(fontsize=7)
    ax[1, 2].plot(Js, pc, "k-", label="closed-form mixture isentrope")
    ax[1, 2].plot(Js[::20], sol.y[0][::20], "o", ms=4, label="ODE integration")
    ax[1, 2].axhline(0, color="r", lw=0.8)
    if np.isfinite(Jstar):
        ax[1, 2].axvline(Jstar, color="r", ls=":", label=f"p=0 at J*={Jstar:.3f}")
    ax[1, 2].set_xlabel("J = L / L0 (compression to the left)")
    ax[1, 2].set_title("D: H_c failure, phases (n=3,G=3)+(n=2,G=0.4)")
    ax[1, 2].set_ylim(-20, 40)
    ax[1, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
