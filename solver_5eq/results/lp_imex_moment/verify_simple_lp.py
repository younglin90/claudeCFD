"""Section 89 tests of simple_lp.py (1D):  (A) smooth acoustic pulse, convergence by successive differences;
(B) gas | water | gas shock tube vs the exact two-material Riemann solution (as section 83);
(C) shock hitting the interface vs the exact interface Riemann solution (as section 88);
(D) interface smearing of a translating contact."""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simple_lp as sl


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pulse(args):
    N, order = args
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    s = np.where(np.abs((x - 0.3) / 0.12) < 1, np.cos(0.5 * np.pi * (x - 0.3) / 0.12) ** 6, 0.0)
    eps = 0.02
    c0 = 5.0498
    a = np.ones(N)
    r0 = 1.0 * (1 + eps * s)
    r1 = np.full(N, 0.2)
    u = c0 * eps * s
    p = np.full(N, 10.0) + 25.5 * eps * s
    U = sl.cons(a, r0, r1, u, p)
    U, n = sl.run(U, dx, 0.04, order=order)
    W = sl.prim(*U)
    return N, order, W[1], W[4], W[3]


def test_pulse():
    Ns = [100, 200, 400, 800, 1600, 3200]
    with Pool(12) as pool:
        res = pool.map(pulse, [(N, o) for N in Ns for o in (1, 2)])
    sol = {(r[0], r[1]): r[2:] for r in res}
    print("(A) smooth water pulse: orders of successive differences d_N = ||U_N - U_2N||_L1 (rho, p, u)")
    for o in (1, 2):
        d = []
        for N in Ns[:-1]:
            h = 1.0 / N
            a, b = sol[(N, o)], sol[(2 * N, o)]
            d.append([h * np.sum(np.abs(a[0] - b[0].reshape(-1, 2).mean(axis=1))), h * np.sum(np.abs(a[1] - b[1].reshape(-1, 2).mean(axis=1))),
                      h * np.sum(np.abs(a[2] - b[2].reshape(-1, 2).mean(axis=1)))])
        d = np.array(d)
        print(f"  order {o}: orders per doubling:", [np.round(x, 2).tolist() for x in np.log2(d[:-1] / d[1:])])


def tube_setup(N, PG):
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    inside = (x > 0.25) & (x < 0.75)
    a = np.where(inside, 1.0, 0.0)
    r0 = np.full(N, 1.0)
    r1 = np.full(N, 0.2)
    u = np.zeros(N)
    p = np.where(inside, 10.0, PG)
    return dx, sl.cons(a, r0, r1, u, p)


def tube(args):
    N, order, PG, T = args
    dx, U = tube_setup(N, PG)
    t0 = sl.totals(U, dx)
    t_start = time.time()
    U, n = sl.run(U, dx, T, order=order)
    wall = time.time() - t_start
    W = sl.prim(*U)
    t1 = sl.totals(U, dx)
    rho = U[0] + U[1]
    drift = np.array([(t1[0] - t0[0]) / t0[0], (t1[1] - t0[1]) / t0[1], (t1[2] - t0[2]) / (t0[3] / 10.0), (t1[3] - t0[3]) / t0[3]])
    return N, order, rho, W[4], W[3], drift, n, wall, W[0]


def exact_tube(N, PG, T):
    tp = load("tp", "verify_two_phase.py")
    fs = load("fs0", "fitted_scheme.py")
    PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
    h = 1.0 / N
    ns = 6
    xs = np.array([(c + (k + 0.5) / ns) * h for c in range(N) for k in range(ns)])
    xq = np.where(xs <= 0.5, xs, 1.0 - xs)
    sol, ps, us = tp.riemann2(PH[1], PH[0], (0.2, 0.0, PG), (1.0, 0.0, 10.0), xq, T, x0=0.25)
    sgn = np.where(xs <= 0.5, 1.0, -1.0)
    return (sol[:, 0].reshape(N, ns).mean(axis=1), sol[:, 2].reshape(N, ns).mean(axis=1),
            (sol[:, 1] * sgn).reshape(N, ns).mean(axis=1), ps, us)


def test_tube(PG, T, Ns):
    with Pool(len(Ns) * 2 + len(Ns)) as pool:
        res = pool.map(tube, [(N, o, PG, T) for N in Ns for o in (1, 2)])
        ex = pool.starmap(exact_tube, [(N, PG, T) for N in Ns])
    print(f"(B) shock tube P_G={PG}, T={T}: window L1 errors (rho, p, u) over the whole periodic domain")
    err = {}
    for r in res:
        N, o = r[0], r[1]
        er, ep, eu, ps, us = ex[Ns.index(N)]
        h = 1.0 / N
        e = (h * np.sum(np.abs(r[2] - er)), h * np.sum(np.abs(r[3] - ep)), h * np.sum(np.abs(r[4] - eu)))
        err[(N, o)] = e
        print(f"  order {o} N={N}: rho={e[0]:.3e} p={e[1]:.3e} u={e[2]:.3e} | drift (m0,m1,q/(E/10),E) max {np.max(np.abs(r[5])):.1e}; {r[6]} steps, {r[7]:.2f}s")
    for o in (1, 2):
        print(f"  order {o} mean order:", np.round(np.mean([np.log2(np.array(err[(a, o)]) / np.array(err[(b, o)])) for a, b in zip(Ns[:-1], Ns[1:])], axis=0), 2).tolist())
    return res, ex


def shock_interface(args):
    N, order, p2 = args
    si = load("si", "verify_shock_interface.py")
    tp, PH0, r2, u2, sshock = si.states(p2)
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    water = (x > si.XI) & (x < 0.95)
    a = np.where(water, 1.0, 0.0)
    slab = x < si.XS0
    r0 = np.full(N, 1.0)
    r1 = np.where(slab, r2, 0.2)
    u = np.where(slab & ~water, u2, 0.0)
    p = np.where(water, 10.0, np.where(slab, p2, 10.0))
    T = (si.XI - si.XS0) / sshock + si.POST
    U = sl.cons(a, r0, r1, u, p)
    t0 = sl.totals(U, dx)
    U, n = sl.run(U, dx, T, order=order)
    W = sl.prim(*U)
    t1 = sl.totals(U, dx)
    rho = U[0] + U[1]
    return N, order, rho, W[4], W[3], T, (t1 - t0)


def exact_si(N, p2):
    si = load("si", "verify_shock_interface.py")
    tp, PH, r2, u2, sshock = si.states(p2)
    h = 1.0 / N
    ns = 6
    xs = np.array([(c + (k + 0.5) / ns) * h for c in range(N) for k in range(ns)])
    t_hit = (si.XI - si.XS0) / sshock
    sol, ps, us = tp.riemann2(PH[1], PH[0], (r2, u2, p2), (1.0, 0.0, 10.0), xs, si.POST, x0=si.XI)
    return (sol[:, 0].reshape(N, ns).mean(axis=1), sol[:, 2].reshape(N, ns).mean(axis=1), sol[:, 1].reshape(N, ns).mean(axis=1), ps, us)


def test_shock_interface(p2=30.0, Ns=(200, 400, 800, 1600)):
    with Pool(len(Ns) * 3) as pool:
        res = pool.map(shock_interface, [(N, o, p2) for N in Ns for o in (1, 2)])
        ex = pool.starmap(exact_si, [(N, p2) for N in Ns])
    print(f"(C) shock (p2={p2}) hits the interface: window L1 errors on x in (0.55, 0.68), exact p*={ex[0][3]:.3f}, u*={ex[0][4]:.3f}")
    err = {}
    for r in res:
        N, o = r[0], r[1]
        er, ep, eu, _, _ = ex[Ns.index(N)]
        h = 1.0 / N
        xc = (np.arange(N) + 0.5) / N
        win = (xc > 0.55) & (xc < 0.68)
        e = (h * np.sum(np.abs(r[2] - er)[win]), h * np.sum(np.abs(r[3] - ep)[win]), h * np.sum(np.abs(r[4] - eu)[win]))
        err[(N, o)] = e
        print(f"  order {o} N={N}: rho={e[0]:.3e} p={e[1]:.3e} u={e[2]:.3e} | drift m0,m1,q,E (abs/rel): {np.round(r[6], 15).tolist()}")
    for o in (1, 2):
        print(f"  order {o} mean order:", np.round(np.mean([np.log2(np.array(err[(a, o)]) / np.array(err[(b, o)])) for a, b in zip(Ns[:-1], Ns[1:])], axis=0), 2).tolist())
    return res, ex


def thickness(args):
    N, order = args
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    a = np.where((x > 0.3) & (x < 0.7), 1.0, 0.0)
    U = sl.cons(a, np.full(N, 1.0), np.full(N, 0.2), np.full(N, 0.7), np.full(N, 10.0))
    out = []
    for T in (0.5, 1.0, 2.0):
        U, n = sl.run(U, dx, T if T == 0.5 else T, order=order) if False else sl.run(U, dx, 0.5, order=order)
        out.append(np.sum((U[4] > 0.01) & (U[4] < 0.99)))
    return N, order, out


def test_thickness():
    with Pool(8) as pool:
        res = pool.map(thickness, [(N, o) for N in (200, 400, 800) for o in (1, 2)])
    print("(D) number of interface cells (0.01 < alpha < 0.99) of a contact translating at u=0.7 after t = 0.5, 1.0, 1.5 (two interfaces)")
    for N, o, out in res:
        print(f"  order {o} N={N}: {out}  (in units of h)")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"):
        test_pulse()
    if which in ("B", "all"):
        test_tube(40.0, 0.012, [100, 200, 400, 800, 1600])
        test_tube(150.0, 0.005, [200, 400, 800])
    if which in ("C", "all"):
        test_shock_interface()
    if which in ("D", "all"):
        test_thickness()
