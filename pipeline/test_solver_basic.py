#!/usr/bin/env python3
"""Quick test of solver module"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.eos.ideal import IdealGasEOS
from solver.solve import run_1d

# Simple test: 1D ideal gas
print("Testing basic solver...")

eos = IdealGasEOS(gamma=1.4, M=28.97)
print(f"EOS created: gamma={eos.gamma}, M={eos.M}, c_v={eos.c_v}")
print(f"R_s = {eos.R_s}")

# Create initial condition
n_cells = 10
x_cells = np.linspace(0.05, 0.95, n_cells)
p0, rho0, u0, T0 = 1e5, 1.0, 0.0, 300.0

e0 = eos.internal_energy(T0)
E0 = e0 + 0.5*u0**2

U = np.zeros((n_cells, 2))
U[:, 0] = rho0
U[:, 1] = rho0 * u0  # rho*u
U_dummy = np.zeros((n_cells, 3))  # dummy for energy
U_dummy[:, 0] = rho0
U_dummy[:, 1] = rho0 * u0
U_dummy[:, 2] = rho0 * E0

print(f"Initial U shape: {U_dummy.shape}")
print(f"Initial U[0]: {U_dummy[0]}")

# Try to run a very short simulation
params = {
    'eos_list': [eos],
    'x_cells': x_cells,
    'U_init': U_dummy,
    't_end': 0.001,
    'CFL': 0.1,
    'bc_left': 'transmissive',
    'bc_right': 'transmissive',
    'output_times': [0.001],
    'verbose': False
}

try:
    result = run_1d(params)
    if result:
        print(f"SUCCESS: Solver ran. Result keys: {result.keys()}")
        print(f"Final time: {result.get('t_final')}")
        print(f"Final U shape: {result['U_final'].shape}")
        print(f"Number of steps: {result.get('n_steps')}")
    else:
        print("FAILED: Solver returned None")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
