"""Test APEC at different resolutions to find where it stabilizes."""
import sys, time
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
from pressure_eq import run

for N in [51, 101, 201, 501]:
    t0 = time.time()
    x, U, T, p, t_h, pe_h, en_h, div = run('APEC', N=N, t_end=0.02, CFL=0.3, verbose=False)
    elapsed = time.time() - t0
    import numpy as np
    pe_max = float(np.max(pe_h))
    n_steps = len(t_h) - 1
    print(f"N={N:4d}: t_final={t_h[-1]:.5f}  PE_max={pe_max:.3e}  "
          f"steps={n_steps}  div={div}  wall={elapsed:.1f}s")
print("Done.")
