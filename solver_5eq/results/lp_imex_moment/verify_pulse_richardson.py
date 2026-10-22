"""Section 51: asymptotic order of the interface-fitted scheme on the compatible pulse test by
successive differences d_N = ||U_N - U_2N|| (no reference solution needed): order = log2(d_N/d_2N)."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp", os.path.join(HERE, "verify_pulse_interface.py"))
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)


def main():
    import sys
    if "--parabolic" in sys.argv:
        vp.fs.PARABOLIC[0] = True
        print("PARABOLIC reconstruction on", flush=True)
    if "--forcecorr" in sys.argv:
        vp.fs.FORCE_CORR[0] = True
        print("FORCE_CORR on", flush=True)
    if "--corrall" in sys.argv:
        vp.fs.CORR_ALL[0] = True
        print("CORR_ALL on", flush=True)
    if "--corrimpl" in sys.argv:
        vp.fs.CORR_IMPLICIT[0] = True
        print("CORR_IMPLICIT on", flush=True)
    if "--gate" in sys.argv:
        vp.fs.CORR_GATE[0] = True
        print("CORR_GATE on", flush=True)
    for a in sys.argv:
        if a.startswith("--stencil="):
            vp.fs.CORR_STENCIL[0] = a.split("=", 1)[1]
            print("CORR_STENCIL", vp.fs.CORR_STENCIL[0], flush=True)
    if "--novcurv" in sys.argv:
        vp.fs.VCURV[0] = False
    Ns = (200, 400, 800, 1600, 3200) if "--to3200" in sys.argv else (200, 400, 800, 1600)
    sol = {}
    for N in Ns:
        rho, p, w, dE = vp.run_fitted(N)
        sol[N] = (rho, p, w)
        print(f"N={N} done, dE/E={dE:.1e}", flush=True)
    d = []
    for N in Ns[:-1]:
        h = 1.0 / N
        r = 2
        a, b = sol[N], sol[2 * N]
        dp = np.sqrt(h * np.sum((a[1] - b[1].reshape(-1, r).mean(axis=1)) ** 2))
        dw = np.sqrt(h * np.sum((a[2] - b[2][::r]) ** 2))
        dr = np.sqrt(h * np.sum((a[0] - b[0].reshape(-1, r).mean(axis=1)) ** 2))
        d.append((N, dr, dp, dw))
        x = (np.arange(N) + 0.5) * h
        near = np.zeros(N, bool)
        for q in ((vp.X1 + vp.U0 * vp.T) % 1.0, (vp.X2 + vp.U0 * vp.T) % 1.0):
            near |= np.abs(((x - q + 0.5) % 1.0) - 0.5) < 2.5 * h
        dinf = np.max(np.abs(a[0] - b[0].reshape(-1, r).mean(axis=1))[near])
        print(f"d_{N} = ||U_{N} - U_{2*N}||: rho={dr:.3e} p={dp:.3e} w={dw:.3e}  rho Linf near interface={dinf:.3e}", flush=True)
    for i in range(len(d) - 1):
        print(f"order from d_{d[i][0]}/d_{d[i+1][0]}: " + " ".join(f"{np.log2(d[i][j]/d[i+1][j]):.2f}" for j in (1, 2, 3)), flush=True)


if __name__ == "__main__":
    main()
