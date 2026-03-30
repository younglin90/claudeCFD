"""Timing test: how fast is the solver per step?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from four_eq import ic_droplet, nasg_prim, rhs_iec, rkstep

N = 100
dx = 1.0 / N
x  = np.linspace(dx/2, 1.0 - dx/2, N)
P0, T0, u0 = 101325.0, 297.0, 5.0
r0, r1, u_ic, rhoE = ic_droplet(x, P0, T0, u0, eps_factor=2.0)
U = [r0.copy(), r1.copy(), (r0+r1)*u_ic, rhoE.copy()]

rho = r0 + r1
Y0  = r0 / rho
_, _, _, _, c2 = nasg_prim(Y0, rho, U[2], U[3])
lam = float(np.max(np.abs(u_ic) + np.sqrt(c2)))
dt  = 0.5 * dx / (lam + 1e-10)
print(f"dt={dt:.3e}  steps needed: {int(0.2/dt)}")

def rhs_fn(U, dx):
    return rhs_iec(U, dx, 'water', 'air', iec=True, use_char=True, weno_order=5)

# Time 100 steps
n_warmup = 10
n_test   = 100

for _ in range(n_warmup):
    U, _ = rkstep(U, rhs_fn, dx, dt)

t0 = time.time()
for _ in range(n_test):
    U, _ = rkstep(U, rhs_fn, dx, dt)
elapsed = time.time() - t0

steps_per_sec = n_test / elapsed
total_steps = int(0.2 / dt)
print(f"Steps/sec: {steps_per_sec:.0f}")
print(f"Total steps needed: {total_steps}")
print(f"Est. time for t=0.2: {total_steps/steps_per_sec:.1f} s")
print(f"Est. time for full validation (4 configs): {4*total_steps/steps_per_sec:.1f} s")
