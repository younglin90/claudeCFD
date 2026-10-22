"""Section 56: pointwise (L-infinity) successive differences of the density in the cells next to
the material interfaces (where an uncancelled inflow-fragment reconstruction error could
accumulate), vs the global L2 differences. Solutions are cached under the scratch folder."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp", os.path.join(HERE, "verify_pulse_interface.py"))
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)
CACHE = sys.argv[1] if len(sys.argv) > 1 else "."


def get(N):
    fn = os.path.join(CACHE, f"pulse_fitted_{N}.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        return d["rho"], d["p"], d["w"]
    rho, p, w, dE = vp.run_fitted(N)
    np.savez(fn, rho=rho, p=p, w=w)
    return rho, p, w


def main():
    Ns = (200, 400, 800, 1600)
    sol = {N: get(N) for N in Ns}
    xi = [(vp.X1 + vp.U0 * vp.T) % 1.0, (vp.X2 + vp.U0 * vp.T) % 1.0]
    prev = None
    for N in Ns[:-1]:
        h = 1.0 / N
        a, b = sol[N][0], sol[2 * N][0].reshape(-1, 2).mean(axis=1)
        x = (np.arange(N) + 0.5) * h
        near = np.zeros(N, bool)
        for q in xi:
            near |= np.abs(((x - q + 0.5) % 1.0) - 0.5) < 2.5 * h
        dinf_near = np.max(np.abs(a - b)[near])
        dinf_far = np.max(np.abs(a - b)[~near])
        dl2_near = np.sqrt(h * np.sum((a - b)[near] ** 2))
        cur = (dinf_near, dinf_far, dl2_near)
        print(f"N={N}: Linf near-interface={dinf_near:.3e} Linf far={dinf_far:.3e} L2 near={dl2_near:.3e}"
              + ("" if prev is None else "  orders " + " ".join(f"{np.log2(p_/c_):.2f}" for p_, c_ in zip(prev, cur))), flush=True)
        prev = cur


if __name__ == "__main__":
    main()
