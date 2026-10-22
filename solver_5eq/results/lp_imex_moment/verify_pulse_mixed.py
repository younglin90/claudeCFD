"""Mixed-cell piecewise-Kapila scheme on the compatible pulse test (section 46), errors measured
against the interface-fitted N=1600 solution used as an accurate reference."""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vp", os.path.join(HERE, "verify_pulse_interface.py"))
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)


def main():
    R = vp.run_fitted(1600)
    prev = None
    for N in (100, 200, 400):
        r = 1600 // N
        rho, p, w, dE = vp.run_mixed(N)
        rf, pf, wf, _ = vp.run_fitted(N)
        h = 1.0 / N
        pr = R[1].reshape(-1, r).mean(axis=1)
        em = np.sqrt(h * np.sum((p - pr) ** 2))
        ef = np.sqrt(h * np.sum((pf - pr) ** 2))
        print(f"N={N}: L2 p error mixed={em:.3e} fitted={ef:.3e}  dE/E mixed={dE:.1e}"
              + ("" if prev is None else f"  orders mixed={np.log2(prev[0] / em):.2f} fitted={np.log2(prev[1] / ef):.2f}"), flush=True)
        prev = (em, ef)


if __name__ == "__main__":
    main()
