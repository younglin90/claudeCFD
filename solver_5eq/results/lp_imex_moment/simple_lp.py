"""Simple explicit multi-material scheme for 3D use: Allaire 5-equation model on a fixed Eulerian grid with an
acoustic-relaxation face solver and upwind (by the face velocity u*) transport of ALL variables.

State per cell: m_k = alpha_k rho_k (k = 0, 1), momentum q, total energy E (all conserved) and alpha = alpha_0
(advected with the same face velocity, non-conservative form).  EOS per phase (power-law Mie-Gruneisen, same as the
interface-fitted oracle):  e_k(rho_k, p) = p/G_k + C_k rho_k^{n_k}   (energy per volume), C_k = A_k (1/(n_k-1) - 1/G_k).
Pressure-equilibrium closure is EXPLICIT (the EOS is linear in p):
    p = (E - 1/2 q^2/rho - sum_k alpha_k C_k rho_k^{n_k}) / (sum_k alpha_k / G_k),   rho_k = m_k / alpha_k.
Face solver (one scalar a per face, a = kappa max(rho c)_L,R):
    u* = (u_L+u_R)/2 - (p_R-p_L)/(2a),   p* = (p_L+p_R)/2 - a (u_R-u_L)/2.
Fluxes (s = upwind state chosen by the sign of u*):
    F_m_k = u* m_k^s,  F_q = u* q^s + p*,  F_E = u* E^s + p* u*,
    alpha_t = -[u*_r alpha^s_r - u*_l alpha^s_l - alpha (u*_r - u*_l)]/dx.
Order 1: forward Euler, piecewise-constant states.  Order 2: minmod reconstruction of the primitives
(alpha, rho_0, rho_1, u, p) + SSP-RK2.  No implicit solve, no iteration, no geometry."""
import numpy as np


class Phase:
    def __init__(self, A, n, G):
        self.A, self.n, self.G = A, n, G
        self.C = A * (1.0 / (n - 1) - 1.0 / G)


PH = [Phase(1.0, 3, 1.5), Phase(0.5, 2, 0.4)]       # 0 = water-like, 1 = gas-like (same as the oracle)
RREF = [1.0, 0.2]
EPS = 1e-9
KAPPA = 1.5


def cons(a, r0, r1, u, p, PH=PH):
    m0, m1 = a * r0, (1 - a) * r1
    rho = m0 + m1
    E = a * (p / PH[0].G + PH[0].C * r0 ** PH[0].n) + (1 - a) * (p / PH[1].G + PH[1].C * r1 ** PH[1].n) + 0.5 * rho * u * u
    return m0, m1, rho * u, E, a


def prim(m0, m1, q, E, a, PH=PH):
    a1 = 1 - a
    r0 = np.where(a > EPS, m0 / np.maximum(a, EPS), RREF[0])
    r1 = np.where(a1 > EPS, m1 / np.maximum(a1, EPS), RREF[1])
    rho = m0 + m1
    u = q / rho
    cold = a * PH[0].C * r0 ** PH[0].n + a1 * PH[1].C * r1 ** PH[1].n
    p = (E - 0.5 * q * u - cold) / (a / PH[0].G + a1 / PH[1].G)
    return a, r0, r1, u, p


def rho_c(a, r0, r1, p, PH=PH):
    """mixture density and rho*c (Wood): 1/K = sum alpha_k / K_k, K_k = n A r^n + (1+G)(p - A r^n)."""
    k0 = PH[0].n * PH[0].A * r0 ** PH[0].n + (1 + PH[0].G) * (p - PH[0].A * r0 ** PH[0].n)
    k1 = PH[1].n * PH[1].A * r1 ** PH[1].n + (1 + PH[1].G) * (p - PH[1].A * r1 ** PH[1].n)
    K = 1.0 / (a / np.maximum(k0, 1e-12) + (1 - a) / np.maximum(k1, 1e-12))
    rho = a * r0 + (1 - a) * r1
    return rho, np.sqrt(rho * K), np.sqrt(K / rho)


def face_flux(WL, WR, PH=PH):
    """WL, WR: primitives (a, r0, r1, u, p) of the states left and right of the faces.  Returns the fluxes of
    (m0, m1, q, E), the face velocity u* and the upwind alpha."""
    aL, r0L, r1L, uL, pL = WL
    aR, r0R, r1R, uR, pR = WR
    _, zL, _ = rho_c(aL, r0L, r1L, pL, PH)
    _, zR, _ = rho_c(aR, r0R, r1R, pR, PH)
    a_f = KAPPA * np.maximum(zL, zR)
    us = 0.5 * (uL + uR) - (pR - pL) / (2 * a_f)
    ps = 0.5 * (pL + pR) - 0.5 * a_f * (uR - uL)
    up = us > 0
    sel = lambda x, y: np.where(up, x, y)
    a_s, r0_s, r1_s, u_s, p_s = (sel(WL[i], WR[i]) for i in range(5))
    m0, m1, q, E, _ = cons(a_s, r0_s, r1_s, u_s, p_s, PH)
    return us * m0, us * m1, us * q + ps, us * E + ps * us, us, a_s


def minmod(x, y):
    return np.where(x * y > 0, np.sign(x) * np.minimum(np.abs(x), np.abs(y)), 0.0)


THINC_BETA = 1.6          # THINC sharpness for the volume fraction (0 disables, minmod is then used for alpha too)
THINC_EPS = 1e-6


def thinc(a, axis=0):
    """THINC (tanh) reconstruction of alpha: right-face value of cell j (left state of face j+1/2) and left-face value of
    cell j (right state of face j-1/2); linear-limited (minmod) where alpha is not monotone or the cell is pure."""
    am, ap = np.roll(a, 1, axis=axis), np.roll(a, -1, axis=axis)
    s = minmod(a - am, ap - a)
    aL, aR = a + 0.5 * s, a - 0.5 * s                       # minmod values (left state of j+1/2, right state of j-1/2)
    amin, amax = np.minimum(am, ap), np.maximum(am, ap)
    use = (a > THINC_EPS) & (a < 1 - THINC_EPS) & ((ap - a) * (a - am) > 0) & (amax - amin > THINC_EPS)
    beta = THINC_BETA
    th = np.sign(ap - am)
    C = np.where(use, (a - amin) / np.maximum(amax - amin, 1e-300), 0.5)
    B = np.exp(th * beta * (2 * C - 1))
    A = (B / np.cosh(beta) - 1) / np.tanh(beta)
    fl = 0.5 * (1 + th * (np.tanh(beta) + A) / (1 + A * np.tanh(beta)))      # normalised value at the right face
    fr = 0.5 * (1 + th * A)                                                    # normalised value at the left face
    aL = np.where(use, amin + (amax - amin) * fl, aL)
    aR = np.where(use, amin + (amax - amin) * fr, aR)
    return aL, aR


def faces(W, order, axis=0):
    """left/right primitive states at the faces j+1/2 (periodic)."""
    if order == 1:
        return W, tuple(np.roll(w, -1, axis=axis) for w in W)
    L, R = [], []
    for i, w in enumerate(W):
        if i == 0 and THINC_BETA > 0:
            wl, wr = thinc(w, axis)
        else:
            s = minmod(w - np.roll(w, 1, axis=axis), np.roll(w, -1, axis=axis) - w)
            wl, wr = w + 0.5 * s, w - 0.5 * s
        L.append(wl)
        R.append(np.roll(wr, -1, axis=axis))
    return tuple(L), tuple(R)


def rhs(U, dx, order, PH=PH):
    m0, m1, q, E, a = U
    W = prim(m0, m1, q, E, a, PH)
    WL, WR = faces(W, order)
    F0, F1, Fq, FE, us, a_s = face_flux(WL, WR, PH)
    d = lambda F: -(F - np.roll(F, 1)) / dx
    da = -(us * a_s - np.roll(us * a_s, 1) - a * (us - np.roll(us, 1))) / dx
    return d(F0), d(F1), d(Fq), d(FE), da


def dt_cfl(U, dx, cfl, PH=PH):
    m0, m1, q, E, a = U
    _, r0, r1, u, p = prim(m0, m1, q, E, a, PH)
    _, _, c = rho_c(a, r0, r1, p, PH)
    return cfl * dx / np.max(np.abs(u) + c)


def step(U, dt, dx, order=1, PH=PH):
    if order == 1:
        R = rhs(U, dx, 1, PH)
        V = tuple(u + dt * r for u, r in zip(U, R))
        return V[:4] + (np.clip(V[4], 0.0, 1.0),)
    R1 = rhs(U, dx, 2, PH)
    V = tuple(u + dt * r for u, r in zip(U, R1))
    V = V[:4] + (np.clip(V[4], 0.0, 1.0),)
    R2 = rhs(V, dx, 2, PH)
    Wn = tuple(0.5 * u + 0.5 * (v + dt * r) for u, v, r in zip(U, V, R2))
    return Wn[:4] + (np.clip(Wn[4], 0.0, 1.0),)


def run(U, dx, t_end, order=1, cfl=None, PH=PH):
    cfl = (0.4 if order == 1 else 0.3) if cfl is None else cfl
    t = 0.0
    n = 0
    while t < t_end - 1e-15:
        dt = min(dt_cfl(U, dx, cfl, PH), t_end - t)
        U = step(U, dt, dx, order, PH)
        t += dt
        n += 1
    return U, n


def totals(U, dx):
    m0, m1, q, E, a = U
    return np.array([m0.sum() * dx, m1.sum() * dx, q.sum() * dx, E.sum() * dx])
