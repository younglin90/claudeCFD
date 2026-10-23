"""Dimension-independent (1D/2D/3D) version of simple_lp.py.  State U = (m0, m1, q[0..d-1], E, a) as arrays of shape
(n_1, ..., n_d); periodic.  Unsplit finite volume: for every axis the face flux is the 1D one of simple_lp.py with the
face-normal velocity (the tangential momenta are transported with u* like any other conserved variable); the
alpha equation gets the sum of the divergence terms of all axes.  Order 1: forward Euler; order 2: minmod
reconstruction of the primitives (alpha by THINC) + SSP-RK2."""
import numpy as np

import simple_lp as sl
from simple_lp import PH, EPS, KAPPA, RREF, minmod, thinc, rho_c

THINC = True


def cons(a, r0, r1, vel, p):
    m0, m1 = a * r0, (1 - a) * r1
    rho = m0 + m1
    k2 = sum(v * v for v in vel)
    E = a * (p / PH[0].G + PH[0].C * r0 ** PH[0].n) + (1 - a) * (p / PH[1].G + PH[1].C * r1 ** PH[1].n) + 0.5 * rho * k2
    return (m0, m1) + tuple(rho * v for v in vel) + (E, a)


def prim(U):
    d = len(U) - 4
    m0, m1, E, a = U[0], U[1], U[2 + d], U[3 + d]
    q = U[2:2 + d]
    a1 = 1 - a
    r0 = np.where(a > EPS, m0 / np.maximum(a, EPS), RREF[0])
    r1 = np.where(a1 > EPS, m1 / np.maximum(a1, EPS), RREF[1])
    rho = m0 + m1
    vel = tuple(c / rho for c in q)
    cold = a * PH[0].C * r0 ** PH[0].n + a1 * PH[1].C * r1 ** PH[1].n
    p = (E - 0.5 * sum(c * v for c, v in zip(q, vel)) - cold) / (a / PH[0].G + a1 / PH[1].G)
    return (a, r0, r1) + vel + (p,)


def face_states(W, order, axis):
    if order == 1:
        return W, tuple(np.roll(w, -1, axis=axis) for w in W)
    L, R = [], []
    for i, w in enumerate(W):
        if i == 0 and THINC and sl.THINC_BETA > 0:
            wl, wr = thinc(w, axis)
        else:
            s = minmod(w - np.roll(w, 1, axis=axis), np.roll(w, -1, axis=axis) - w)
            wl, wr = w + 0.5 * s, w - 0.5 * s
        L.append(wl)
        R.append(np.roll(wr, -1, axis=axis))
    return tuple(L), tuple(R)


def face_flux(WL, WR, axis, d):
    aL, r0L, r1L, pL = WL[0], WL[1], WL[2], WL[-1]
    aR, r0R, r1R, pR = WR[0], WR[1], WR[2], WR[-1]
    uL, uR = WL[3 + axis], WR[3 + axis]
    zL = rho_c(aL, r0L, r1L, pL)[1]
    zR = rho_c(aR, r0R, r1R, pR)[1]
    a_f = KAPPA * np.maximum(zL, zR)
    us = 0.5 * (uL + uR) - (pR - pL) / (2 * a_f)
    ps = 0.5 * (pL + pR) - 0.5 * a_f * (uR - uL)
    up = us > 0
    S = tuple(np.where(up, WL[i], WR[i]) for i in range(len(WL)))
    Us = cons(S[0], S[1], S[2], S[3:3 + d], S[-1])
    F = [us * Us[0], us * Us[1]]
    for k in range(d):
        F.append(us * Us[2 + k] + (ps if k == axis else 0.0))
    F.append(us * Us[2 + d] + ps * us)
    return F, us, S[0]


def rhs(U, dx, order):
    d = len(U) - 4
    W = prim(U)
    a = U[-1]
    R = [np.zeros_like(u) for u in U]
    for ax in range(d):
        WL, WR = face_states(W, order, ax)
        F, us, a_s = face_flux(WL, WR, ax, d)
        for i in range(d + 3):
            R[i] -= (F[i] - np.roll(F[i], 1, axis=ax)) / dx[ax]
        R[-1] -= (us * a_s - np.roll(us * a_s, 1, axis=ax) - a * (us - np.roll(us, 1, axis=ax))) / dx[ax]
    return R


def dt_cfl(U, dx, cfl):
    d = len(U) - 4
    W = prim(U)
    c = rho_c(W[0], W[1], W[2], W[-1])[2]
    s = sum((np.abs(W[3 + k]) + c) / dx[k] for k in range(d))
    return cfl / np.max(s)


def _clip(V):
    return V[:-1] + (np.clip(V[-1], 0.0, 1.0),)


def step(U, dt, dx, order=1):
    if order == 1:
        R = rhs(U, dx, 1)
        return _clip(tuple(u + dt * r for u, r in zip(U, R)))
    R1 = rhs(U, dx, 2)
    V = _clip(tuple(u + dt * r for u, r in zip(U, R1)))
    R2 = rhs(V, dx, 2)
    return _clip(tuple(0.5 * u + 0.5 * (v + dt * r) for u, v, r in zip(U, V, R2)))


def run(U, dx, t_end, order=1, cfl=None, max_steps=10 ** 9):
    d = len(U) - 4
    cfl = (0.4 if order == 1 else 0.3) / (1.0 if d == 1 else 1.0) if cfl is None else cfl
    if d > 1:
        cfl = cfl * (0.5 if d == 2 else 0.4)
    t, n = 0.0, 0
    while t < t_end - 1e-15 and n < max_steps:
        dt = min(dt_cfl(U, dx, cfl), t_end - t)
        U = step(U, dt, dx, order)
        t += dt
        n += 1
    return U, n


def totals(U, dx):
    cell = np.prod(dx)
    return np.array([u.sum() * cell for u in U[:-1]])
