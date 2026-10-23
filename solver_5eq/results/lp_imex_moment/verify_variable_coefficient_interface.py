"""Section 82 (O3, frozen-coefficient check): interface impulse response S1 of the real scheme on a NON-uniform
background (a smooth acoustic pulse centred on / near the interface, amplitude EPS) versus the uniform background.
Perturbation = velocity impulse 1e-4 on the interface node; trace = difference of the interface-node velocity of the
perturbed and unperturbed runs, normalised as in verify_ringdown_model.series.  Ratio S1(pulse)/S1(uniform) ~ 1 supports
that the constants of the uniform-background analysis carry over to smooth variable coefficients."""
import importlib.util
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def job(args):
    N, u0, theta, eps_bg, centre, W = args[:6]
    eps = args[6] if len(args) > 6 else 1e-4
    recon = args[7] if len(args) > 7 else True
    vp = load("vp", "verify_pulse_interface.py")
    oi = load("oi", "verify_open_items.py")
    fs = vp.fs
    h = 1.0 / N
    tau = h / oi.CG
    X1, X2 = 0.3 + theta * h, 0.7 + theta * h
    vp.U0, vp.EPS, vp.X1, vp.X2 = u0, eps_bg, X1, X2          # vp.mat_of reads the module-level X1, X2
    c0 = X1 + centre

    def bump(x):
        s = (x - c0) / 0.08
        return np.where(np.abs(s) < 1, np.cos(0.5 * np.pi * s) ** 6, 0.0)
    vp.bump = bump
    def mk():
        return fs.build(N, vp.PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [X1, X2])
    a, b = mk(), mk()
    ia = oi.iface(b)
    m = np.array([z[3] for z in b.zones])
    MI = 0.5 * (m[ia - 1] + m[ia])
    b.w[ia] += eps
    norm = (oi.ZL + oi.ZR) * tau / (MI * eps)
    tr = []
    for n in range(W):
        a, _ = fs.step(a, tau, vp.PH, N, recon=recon, vslope=recon)
        b, _ = fs.step(b, tau, vp.PH, N, recon=recon, vslope=recon)
        tr.append((b.w[oi.iface(b)] - a.w[oi.iface(a)]) * norm)
    return args, float(np.abs(tr).sum())


def main():
    N, W = 200, 70
    cases = []
    for u0 in (0.3, 1.0):
        for eps_bg, centre in ((0.0, 0.0), (0.02, 0.0), (0.02, 0.06), (0.08, 0.0), (0.08, -0.06)):
            cases.append((N, u0, 0.5, eps_bg, centre, W))
    with Pool(len(cases)) as pool:
        res = pool.map(job, cases)
    base = {}
    for (N_, u0, th, eb, ce, W_), s1 in res:
        if eb == 0.0:
            base[u0] = s1
    print("S1 of the interface impulse response on a non-uniform background (N=200, W=70, theta=0.5):")
    for (N_, u0, th, eb, ce, W_), s1 in res:
        print(f"  u0={u0} (nu={u0 / 8.4183:.3f}) background pulse EPS={eb} centre offset {ce:+.2f}: S1 = {s1:.3f}  ratio to uniform {s1 / base[u0]:.3f}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
