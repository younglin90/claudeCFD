"""Section 76 (task 1): does the order recover for strong background gradients at fine grids?
Usage: verify_gradient_asymptotics.py EPS N  -> saves rho, p, w of the interface-fitted scheme (PARABOLIC on,
no force correction) for the compatible pulse with amplitude EPS to /tmp/claude-1000/oi/grad_<EPS>_<N>.npz;
verify_gradient_asymptotics.py EPS summary -> successive-difference orders from the saved files."""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/tmp/claude-1000/oi"


def main():
    eps, what = float(sys.argv[1]), sys.argv[2]
    spec = importlib.util.spec_from_file_location("vp", os.path.join(HERE, "verify_pulse_interface.py"))
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    vp.fs.PARABOLIC[0] = True
    vp.EPS = eps
    os.makedirs(OUT, exist_ok=True)
    if what != "summary":
        N = int(what)
        t0 = time.time()
        rho, p, w, dE = vp.run_fitted(N)
        np.savez(f"{OUT}/grad_{eps}_{N}.npz", rho=rho, p=p, w=w)
        print(f"EPS={eps} N={N} done in {time.time() - t0:.0f}s dE/E={dE}", flush=True)
        return
    Ns = sorted(int(f.split("_")[2][:-4]) for f in os.listdir(OUT) if f.startswith(f"grad_{eps}_") and f.endswith(".npz"))
    sol = {N: np.load(f"{OUT}/grad_{eps}_{N}.npz") for N in Ns}
    d = {}
    for N in Ns:
        if 2 * N in sol:
            h = 1.0 / N
            a, b = sol[N], sol[2 * N]
            d[N] = [np.sqrt(h * np.sum((a["rho"] - b["rho"].reshape(-1, 2).mean(axis=1)) ** 2)),
                    np.sqrt(h * np.sum((a["p"] - b["p"].reshape(-1, 2).mean(axis=1)) ** 2)),
                    np.sqrt(h * np.sum((a["w"] - b["w"][::2]) ** 2))]
    ks = sorted(d)
    print(f"EPS={eps}: d_N (rho,p,w):", {k: np.round(d[k], 5).tolist() for k in ks})
    for k1, k2 in zip(ks[:-1], ks[1:]):
        print(f"  order d{k1}/d{k2} (rho,p,w): {np.round(np.log2(np.array(d[k1]) / np.array(d[k2])), 2)}")


if __name__ == "__main__":
    main()
