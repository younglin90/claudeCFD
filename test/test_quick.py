"""Quick test: N=51, t_end=0.02 for all 3 schemes."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
from pressure_eq import run
import numpy as np

N = 51
t_end = 0.02

for sch in ['FC', 'APEC', 'PEqC']:
    x, U, T, p, t_h, pe_h, en_h, div = run(sch, N=N, t_end=t_end, CFL=0.3, verbose=False)
    pe_final = float(np.max(pe_h[-5:]))
    en_final = float(np.max(en_h[-5:])) if sch != 'PEqC' else float('nan')
    steps = len(t_h) - 1
    print(f"{sch:6s}: steps={steps:4d}  PE_final={pe_final:.3e}  "
          f"Enerr_final={en_final:.3e}  diverged={div}")
print("Done.")
