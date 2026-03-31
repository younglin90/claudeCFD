"""
Quick test for Case A with reduced time domain.
"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from solver.eos.ideal import IdealGasEOS
from solver.utils import cons_to_prim, prim_to_cons
from solver.solve import run_1d

print("Starting Case A quick test...")

# Physical setup
eos1 = IdealGasEOS(gamma=1.4, M=28.0)
eos2 = IdealGasEOS(gamma=1.66, M=4.0)
eos_list = [eos1, eos2]

# Domain and mesh (smaller for testing)
L = 1.0
nx = 101  # Reduced from 501
x = np.linspace(0, L, nx, endpoint=False) + L / (2*nx)
dx = L / nx

# Initial condition
x_c, r_c, w1, w2, k = 0.5, 0.25, 0.6, 0.2, 20
u0 = 1.0
p0 = 0.9
t_end = 1.0  # Reduced from 8.0

r = np.abs(x - x_c)
rhoY1 = (w1/2) * (1 - np.tanh(k * (r - r_c)))
rhoY2 = (w2/2) * (1 + np.tanh(k * (r - r_c)))
rho = rhoY1 + rhoY2
Y1 = rhoY1 / rho
Y2 = rhoY2 / rho

R_mix = Y1 * eos1.R_s + Y2 * eos2.R_s
T_arr = p0 / (rho * R_mix)

c_v1 = eos1.R_s / (eos1.gamma - 1)
c_v2 = eos2.R_s / (eos2.gamma - 1)
e_arr = Y1 * c_v1 * T_arr + Y2 * c_v2 * T_arr
E_arr = e_arr + 0.5 * u0**2

U0 = np.zeros((nx, 4))
U0[:, 0] = rho
U0[:, 1] = rho * u0
U0[:, 2] = rho * E_arr
U0[:, 3] = rhoY1

print(f"Initial conditions set: nx={nx}, t_end={t_end}")
print(f"  ρ ∈ [{rho.min():.6f}, {rho.max():.6f}]")
print(f"  T ∈ [{T_arr.min():.6f}, {T_arr.max():.6f}] K")

# Run simulation
print(f"\nRunning simulation...")
case_params = {
    'eos_list': eos_list,
    'x_cells': x,
    'U_init': U0,
    't_end': t_end,
    'CFL': 0.5,
    'bc_left': 'periodic',
    'bc_right': 'periodic',
    'output_times': [0.0, 0.5, 1.0],
    'verbose': True,
}

result = run_1d(case_params)
print(f"✓ Completed: {result['n_steps']} steps, t_final = {result['t_final']:.4f}")

# Check results
U_final = result['U_final']
prim_final = np.array([cons_to_prim(U_final[i], eos_list) for i in range(nx)])
p_final = prim_final[:, 0]

pe_error = np.sqrt(np.mean((p_final / p0 - 1)**2))
print(f"\nFinal pressure L₂ error: {pe_error:.2e}")
print(f"PASS threshold: 1e-4")
print(f"Result: {'✓ PASS' if pe_error < 1e-4 else '✗ FAIL'}")
