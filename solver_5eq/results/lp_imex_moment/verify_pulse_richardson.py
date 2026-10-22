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
    Ns = (200, 400, 800, 1600, 3200)
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
        print(f"d_{N} = ||U_{N} - U_{2*N}||: rho={dr:.3e} p={dp:.3e} w={dw:.3e}", flush=True)
    for i in range(len(d) - 1):
        print(f"order from d_{d[i][0]}/d_{d[i+1][0]}: " + " ".join(f"{np.log2(d[i][j]/d[i+1][j]):.2f}" for j in (1, 2, 3)), flush=True)


if __name__ == "__main__":
    main()
