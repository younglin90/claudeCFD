"""Verify APEC stability at N=501 for t=0.01 (one flow-through)."""
import sys, time
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import run

t0 = time.time()
x, U, T, p, t_h, pe_h, en_h, div = run(
    'APEC', N=501, t_end=0.01, CFL=0.3, verbose=True)
elapsed = time.time() - t0
print(f"\nN=501 APEC: t_final={t_h[-1]:.5f}  PE_max={np.max(pe_h):.3e}  "
      f"PE_final={pe_h[-1]:.3e}  diverged={div}  wall={elapsed:.1f}s")

# Also run FC for comparison
t0 = time.time()
x, U, T, p, t_h_fc, pe_h_fc, en_h_fc, div_fc = run(
    'FC', N=501, t_end=0.01, CFL=0.3, verbose=False)
elapsed_fc = time.time() - t0
print(f"N=501 FC:   t_final={t_h_fc[-1]:.5f}  PE_max={np.max(pe_h_fc):.3e}  "
      f"PE_final={pe_h_fc[-1]:.3e}  diverged={div_fc}  wall={elapsed_fc:.1f}s")

print(f"\nPE ratio FC/APEC at final time = {pe_h_fc[-1]/pe_h[-1]:.2f}")
