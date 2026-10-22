"""Section 56: Richardson successive differences of (a) the interface POSITIONS and (b) the L1
density error, interface-fitted scheme on the compatible pulse test.  Mixed-cell averages of a
discontinuous density respond to a position error dX as jump*dX/h, so L-inf ~ h, L2 ~ h^1.5 and
L1 ~ h^2 for a second-order dX; the user's target norm is L1 plus the interface position."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp", os.path.join(HERE, "verify_pulse_interface.py"))
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)
fs = vp.fs


def run(N):
    h = 1.0 / N
    st = fs.build(N, vp.PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [vp.X1, vp.X2])
    tau, nst = vp.tau_of(N)
    for n in range(nst):
        st, _ = fs.step(st, tau, vp.PH, N)
    ifc = sorted((st.zones[b][1] % 1.0) for b in range(len(st.zones)) if not st.isface[b])
    rho = np.zeros(N)
    for z in st.zones:
        c = int(np.floor((0.5 * (z[1] + z[2])) / h + 1e-12)) % N
        rho[c] += z[3] / h
    return np.array(ifc), rho


def main():
    Ns = (200, 400, 800, 1600)
    sol = {}
    for N in Ns:
        sol[N] = run(N)
        print(f"N={N}: interfaces {sol[N][0]}", flush=True)
    prev = None
    for N in Ns[:-1]:
        h = 1.0 / N
        dX = np.max(np.abs(sol[N][0] - sol[2 * N][0]))
        a, b = sol[N][1], sol[2 * N][1].reshape(-1, 2).mean(axis=1)
        d1 = h * np.sum(np.abs(a - b))
        dinf = np.max(np.abs(a - b))
        cur = (dX, d1, dinf)
        print(f"d_{N}: |dX|max={dX:.3e}  L1 rho={d1:.3e}  Linf rho={dinf:.3e}"
              + ("" if prev is None else "  orders " + " ".join(f"{np.log2(p_ / c_):.2f}" for p_, c_ in zip(prev, cur))), flush=True)
        prev = cur


if __name__ == "__main__":
    main()
