"""Run FC and APEC to t=0.09s to find when FC diverges (paper's key result)."""
import sys, time
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from apec_1d import run

t0 = time.time()
print("Running FC to t=0.09s...")
x, U, T, p, t_h_fc, pe_h_fc, en_h_fc, div_fc = run(
    'FC', N=501, t_end=0.09, CFL=0.3, verbose=True)
elapsed_fc = time.time() - t0
print(f"\nFC result: t_final={t_h_fc[-1]:.5f}  "
      f"PE_max={np.max(pe_h_fc):.3e}  PE_final={pe_h_fc[-1]:.3e}  "
      f"diverged={div_fc}  wall={elapsed_fc:.0f}s")

t0 = time.time()
print("\nRunning APEC to t=0.09s...")
x, U, T, p, t_h_ap, pe_h_ap, en_h_ap, div_ap = run(
    'APEC', N=501, t_end=0.09, CFL=0.3, verbose=True)
elapsed_ap = time.time() - t0
print(f"\nAPEC result: t_final={t_h_ap[-1]:.5f}  "
      f"PE_max={np.max(pe_h_ap):.3e}  PE_final={pe_h_ap[-1]:.3e}  "
      f"diverged={div_ap}  wall={elapsed_ap:.0f}s")

print(f"\n--- Summary ---")
print(f"FC:   t_final={t_h_fc[-1]:.5f}  diverged={div_fc}")
print(f"APEC: t_final={t_h_ap[-1]:.5f}  diverged={div_ap}")
