"""Section 77 (B): ring damping kappa from the remap, and the full ring-down prediction of S1.

For the near-Nyquist, sign-alternating trapped mode the minmod slopes of the reconstruction vanish, so the real
scheme's remap acts as first-order donor cell on the (w, p) perturbation:
  - zone pressure: p'_new = sum_i |old zone i (shifted by nu h) cap new piece| p_i / |new piece|   (length weights)
  - nodal velocity: mass-coordinate donor cell: w'_b = sum_i |dual_i cap dual'_b| w_i / |dual'_b|,
    node b owns [mu_b - m_{b-1}/2, mu_b + m_b/2]; new mass coordinates mu_new(x') = mu_old(x' - nu h).
The interface (material boundary) is Lagrangian and moves by nu h per step, the Eulerian grid is fixed, theta -> theta+nu.
One step maps the trapped-mode amplitude by a_{n+1} = z_t(theta_n) c(theta_n) a_n with
    c = < v_t(theta'), R v_t(theta) >_{W'}       (W-normalised eigenvectors of the frozen Lagrange operator).
The interface-node trace is trace_n = 2 (Z_L+Z_R) tau Re[ b_n v_I(theta_n) ], b_0 = conj(v_I(theta_0)) (impulse at the
interface node), b_{n+1} = z_t c b_n.  Compared with the real scheme (verify_ringdown_model.series)."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dra = load("dra", "derive_ring_amplitude.py")
oi = dra.oi
RHO_L, K_L, RHO_R, K_R = dra.RHO_L, dra.K_L, dra.RHO_R, dra.K_R
KW = 40                      # window half width in cells


def grid(cell0, X):
    """nodes (positions), zones (intervals, materials) of the window with faces cell0 .. cell0+2KW+1 and the interface X."""
    faces = list(range(cell0, cell0 + 2 * KW + 2))
    xs = sorted(faces + [X])
    zones = [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    mat = [0 if 0.5 * (a + b) < X else 1 for a, b in zones]            # 0 = gas (left), 1 = water (right)
    return xs, zones, mat


def mass(zones, mat):
    return np.array([(b - a) * (RHO_L if m == 0 else RHO_R) for (a, b), m in zip(zones, mat)])


def mode_vector(theta, c, xs, zones, mat, modecache):
    """W-normalised trapped eigenvector of the frozen operator on the window grid."""
    phi, sm, R, A, (kL, kR), (aL, w0, pm1, p0, aR), tau, h, z, pLc, pRc = modecache[theta]
    X = c + theta
    w = np.zeros(len(xs), complex)
    for i, x in enumerate(xs):
        if abs(x - X) < 1e-12:
            w[i] = w0
        elif x < X:
            b = -(c - round(x) + 1)
            w[i] = aL * kL ** b
        else:
            b = round(x) - c
            w[i] = aR * kR ** b
    p = np.zeros(len(zones), complex)
    for i, ((a, b2), m) in enumerate(zip(zones, mat)):
        if m == 0 and abs(b2 - X) < 1e-12:
            p[i] = pm1
        elif m == 1 and abs(a - X) < 1e-12:
            p[i] = p0
        elif m == 0:
            j = round(a)
            zz = -(c - j + 1)
            p[i] = pLc * aL * kL ** zz
        else:
            j = round(a)
            zz = j - c
            p[i] = pRc * aR * kR ** zz
    mz = mass(zones, mat)
    Mn = np.zeros(len(xs))
    Mn[1:-1] = 0.5 * (mz[:-1] + mz[1:])
    Mn[0], Mn[-1] = 0.5 * mz[0], 0.5 * mz[-1]
    Cz = np.array([(b2 - a) / (K_L if m == 0 else K_R) for (a, b2), m in zip(zones, mat)])
    nrm = np.sqrt(np.sum(Mn * np.abs(w) ** 2) + np.sum(Cz * np.abs(p) ** 2))
    return w / nrm, p / nrm, Mn, Cz


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def remap_apply(w, p, c, theta, nu, xs, zones, mat, xs2, zones2, mat2):
    """first-order donor-cell remap of the perturbation (w, p) from the geometry (c, theta) after a shift nu."""
    X = c + theta
    # zone pressure
    p2 = np.zeros(len(zones2), complex)
    for k, ((a2, b2), m2) in enumerate(zip(zones2, mat2)):
        s = 0.0
        for (a, b), m, pv in zip(zones, mat, p):
            if m != m2:
                continue
            s += overlap(a + nu, b + nu, a2, b2) * pv
        p2[k] = s / (b2 - a2)
    # nodal velocity in mass coordinate
    def mu_old(x):
        return RHO_L * (x - xs[0]) if x <= X else RHO_L * (X - xs[0]) + RHO_R * (x - X)
    mz = mass(zones, mat)
    mz2 = mass(zones2, mat2)
    dual_old = []
    for i, x in enumerate(xs):
        ml = mz[i - 1] if i > 0 else 0.0
        mr = mz[i] if i < len(zones) else 0.0
        dual_old.append((mu_old(x) - ml / 2, mu_old(x) + mr / 2))
    w2 = np.zeros(len(xs2), complex)
    for i, x in enumerate(xs2):
        ml = mz2[i - 1] if i > 0 else 0.0
        mr = mz2[i] if i < len(zones2) else 0.0
        lo, hi = mu_old(x - nu) - ml / 2, mu_old(x - nu) + mr / 2
        tot = 0.0
        for (d0, d1), wv in zip(dual_old, w):
            tot += overlap(d0, d1, lo, hi) * wv
        w2[i] = tot / max(hi - lo, 1e-300)
    return w2, p2


def prep(thetas, N=400):
    cache = {}
    for th in thetas:
        r = dra.mode_data(th, N)
        phi, sm, R, A, kap, x, tau, h = r
        z = np.exp(1j * phi)
        sig = (z - 1) / (z + 1)
        kL, kR = kap
        pLc = -(kL - 1) * tau * K_L / (2 * sig * h)
        pRc = -(kR - 1) * tau * K_R / (2 * sig * h)
        cache[th] = (phi, sm, R, A, (kL, kR), tuple(x), tau, h, z, pLc, pRc)
    return cache


def amp_factor(theta, nu, cache_fn, c0=0):
    """c(theta -> theta') and the new (theta', c') ; cache_fn(theta) gives mode data (cached by rounding)."""
    cell0 = c0 - KW
    xs, zones, mat = grid(cell0, c0 + theta)
    cache = cache_fn(theta)
    w, p, Mn, Cz = mode_vector(theta, c0, xs, zones, mat, cache)
    X2 = c0 + theta + nu
    c2 = int(np.floor(X2))
    th2 = X2 - c2
    xs2, zones2, mat2 = grid(cell0, X2)
    w2, p2 = remap_apply(w, p, c0, theta, nu, xs, zones, mat, xs2, zones2, mat2)
    cache2 = cache_fn(th2)
    v2w, v2p, Mn2, Cz2 = mode_vector(th2, c2, xs2, zones2, mat2, cache2)
    cc = np.sum(Mn2 * np.conj(v2w) * w2) + np.sum(Cz2 * np.conj(v2p) * p2)
    return cc, th2, c2, cache[theta][8], cache2[th2][8]


def mode_worker(th):
    return th, prep([th], 200)[th]


def predict(nu, theta0, W, N, memo, rmod):
    """predicted normalised interface trace series for an impulse at the interface node (ring part only)."""
    ths = [(theta0 + n * nu) % 1.0 for n in range(W + 2)]
    def cache_fn(theta):
        return {theta: memo[round(theta, 9)]}
    c = 0
    th = theta0
    # mode vector at theta0 for b_0 and v_I
    cell0 = -KW
    xs, zones, mat = grid(cell0, th)
    w, p, Mn, Cz = mode_vector(th, 0, xs, zones, mat, cache_fn(th))
    iI = [i for i, x in enumerate(xs) if abs(x - th) < 1e-12][0]
    b = np.conj(w[iI])
    CG = oi.CG
    out = []
    for n in range(W):
        cc, th2, c2, z1, _ = amp_factor(th, nu, cache_fn)
        b = b * z1 * cc
        xs2, zones2, mat2 = grid(cell0, c2 + th2)
        w2, p2, Mn2, Cz2 = mode_vector(th2, c2, xs2, zones2, mat2, cache_fn(th2))
        iI2 = [i for i, x in enumerate(xs2) if abs(x - (c2 + th2)) < 1e-12][0]
        out.append(2 * (dra.Z_L + dra.Z_R) / CG * (b * w2[iI2]).real)
        th = th2
    return np.array(out)


def main():
    from multiprocessing import Pool
    N, W, theta0 = 200, 70, 0.37
    nus = {0.3: None, 0.6: None, 1.0: None, 1.6: None}
    CGv = oi.CG
    thetas = set()
    for u0 in nus:
        nu = u0 / CGv
        for n in range(W + 3):
            thetas.add(round((theta0 + n * nu) % 1.0, 9))
    thetas = sorted(thetas)
    print(f"computing {len(thetas)} frozen trapped modes ...", flush=True)
    with Pool(32) as pool:
        res = pool.map(mode_worker, thetas)
    memo = {round(t, 9): v for t, v in res}
    # exact theta keys used inside amp_factor are floats; map by rounding in cache_fn
    rm = load("rm", "verify_ringdown_model.py")
    print("u0, nu: S1 measured (real scheme, N=200, W=70) | ring part predicted by the amplitude map | S_c = difference")
    rows = []
    for u0 in nus:
        nu = u0 / CGv
        meas, nu_m = rm.series((N, u0, theta0, W, 1e-4))
        pred = predict(nu, theta0, W, N, memo, rm)
        s_meas, s_pred = np.abs(meas).sum(), np.abs(pred).sum()
        corr = np.corrcoef(np.abs(meas[10:]), np.abs(pred[10:]))[0, 1]
        print(f"  u0={u0} nu={nu:.4f}: S1_meas={s_meas:.3f}  S1_pred(ring)={s_pred:.3f}  diff={s_meas - s_pred:.3f}  "
              f"corr(|trace|, |pred|) over steps 10..{W} = {corr:.3f}", flush=True)
        rows.append((u0, nu, meas, pred))
    np.save("/tmp/claude-1000/oi/ringdown_pred.npy", np.array([(r[0], r[1], r[2], r[3]) for r in rows], dtype=object), allow_pickle=True)


if __name__ == "__main__" and not os.environ.get("KAPPA_AVG"):
    os.chdir(os.path.join(HERE, "..", ".."))
    main()


def kappa_avg():
    """theta-average of the single-step adiabatic damping: delta_ad = -<ln|c|>_theta, kappa_ad = delta_ad / nu."""
    from multiprocessing import Pool
    thg = [round(0.025 + 0.05 * k, 9) for k in range(20)]
    out = {}
    for nu in (0.0358, 0.0717, 0.1195):
        need = set()
        for th in thg:
            need.add(round(th, 9))
            need.add(round((th + nu) % 1.0, 9))
        need = sorted(need)
        with Pool(32) as pool:
            res = pool.map(mode_worker, need)
        memo = {round(t, 9): v for t, v in res}
        cache_fn = lambda theta: {theta: memo[round(theta, 9)]}
        lc = []
        for th in thg:
            cc, th2, c2, z1, z2 = amp_factor(th, nu, cache_fn)
            lc.append(-np.log(abs(cc)))
        lc = np.array(lc)
        out[nu] = (lc.mean(), lc.mean() / nu, lc.min() / nu, lc.max() / nu)
        print(f"nu={nu}: theta-average delta_ad = {lc.mean():.4f}/step, kappa_ad = {lc.mean() / nu:.2f} "
              f"(range over theta {lc.min() / nu:.2f} .. {lc.max() / nu:.2f})", flush=True)
    return out


if __name__ == "__main__" and os.environ.get("KAPPA_AVG"):
    kappa_avg()
