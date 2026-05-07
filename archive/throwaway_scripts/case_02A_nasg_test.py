"""Case 02-A Test A: 2-phase Water(NASG) - Air(Ideal) PE advection.
Uses Iter 43 baseline (SSP2 + Richardson + default opts).
Spec: N=10, periodic, dt=0.01, 100 steps, t_end=1.0s (one full advection).
PASS: max|(p-p₀)/p₀| < 1e-2, max|u-u₀| < 1e-2.
"""
import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
os.makedirs('results/case_02', exist_ok=True)

# Standard 02-A: ph1 = Air (Ideal, phase 1), ph2 = Water (NASG, phase 2)
# Match existing ablation_02A_nasg.py convention.
ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}                              # Air
ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
       'b': 6.61e-4, 'eta': -1.177788e6}                                    # Water NASG

N = 10; L = 1.0; dx = L/N
x = (np.arange(N) + 0.5)*dx

# IC: ph1=air everywhere except [0.4, 0.6] = water (so a1=α_air)
# Spec says "water in [0.4, 0.6], α_water=1" → since ph2=water, α₂=1 there → α₁=ε there.
p0 = 1e5; u0 = 1.0; T0 = 300.0
a1 = np.where((x >= 0.4) & (x < 0.6), 1e-6, 1.0 - 1e-6)  # α_air

# Phase densities from EOS
# NASG: ρ = (p + p∞)/((γ-1)·kv·T) · 1/(1 - b·ρ_unbound) → solve. For p+pinf >> 0:
# Standard NASG: ρ(p,T) = (p+p∞)/((γ-1)·kv·T + b·(p+p∞))
# Equivalently: ρ⁻¹ = (γ-1)·kv·T/(p+p∞) + b
def rho_nasg(p, T, eos):
    inv_rho = (eos['gamma']-1.0)*eos['kv']*T/(p + eos['pinf']) + eos['b']
    return 1.0/inv_rho

def rho_ideal(p, T, eos):
    return p/((eos['gamma']-1.0)*eos['kv']*T)

rho1 = rho_ideal(p0, T0, ph1)        # air
rho2 = rho_nasg(p0, T0, ph2)         # water NASG
print(f"  ρ_air (Ideal, ph1)   = {rho1:.4f} kg/m³  (typical ~1.16)")
print(f"  ρ_water (NASG, ph2)  = {rho2:.3f} kg/m³  (typical ~1000)")

a1r1 = a1*rho1; a2r2 = (1.0 - a1)*rho2
rho = a1r1 + a2r2; ru = rho*u0

# Internal energy per phase (NASG: ρe = (p + γp∞)/(γ-1) + ρ·η + ... — use kv·T)
# For NASG: e = kv·T + η + (p+p∞)/((γ-1)ρ) · b·ρ... simpler form:
# ρe = αρ·kv·T + αρ·η + (p+p∞)/((γ-1)) for SG; NASG adds b·p term
# Standard mixture: ρE = α₁(p+γ₁p∞₁)/(γ₁-1) + α₂(p+γ₂p∞₂)/(γ₂-1) + ½ρu²  (SG/NASG simplification w/o η,b energy)
# More accurate NASG: e_k = (p+γp∞)/((γ-1)ρ) + η, hence αρe = α[(p+γp∞)/(γ-1) + ρη]
# NASG energy density: ρ·e = (p+γP∞)·(1-b·ρ)/(γ-1) + ρ·η  (CRITICAL: (1-bρ) factor!)
# Ideal energy density: ρ·e = p/(γ-1)
e1_v = (p0 + ph1['gamma']*ph1['pinf'])/(ph1['gamma']-1)                    # air ideal
br2 = ph2['b']*rho2
e2_v = (p0 + ph2['gamma']*ph2['pinf'])*(1.0 - br2)/(ph2['gamma']-1) + rho2*ph2['eta']   # water NASG
rE = a1*e1_v + (1-a1)*e2_v + 0.5*rho*u0*u0

# Per memory project_25th_nasg_ibp_attempts: NASG works with
#   acoustic_method='imex_5n' (5N coupled NK direct), acoustic CFL=0.2, N=10.
# This bypasses SSP2 dispatch (uses 5N NK at outer step level).
# Try unified path: IM1 + SSP2 + Richardson (case 07's solver) on 02-A NASG
# Acoustic CFL=10 → ~1300 steps (vs 80 for cfl=200), but IM1 is L-stable for NASG
out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=1.0,
                 cfl=0.4, use_material_cfl=False,
                 time_integrator='ssp222',
                 acoustic_method='im1',
                 acid_interface=True,
                 primitive_recon='none',
                 max_steps=60000,
                 bc_l='periodic', bc_r='periodic',
                 print_interval=5000)
t_f, a1r1f, a2r2f, ruf, rEf, a1f = out

# Use solver's cons_to_prim for general EOS (NASG-aware)
p_f, u_f, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1, ph2)
rho_f = a1r1f + a2r2f

# Metrics
err_p_rel = np.max(np.abs((p_f - p0)/p0))
err_u = np.max(np.abs(u_f - u0))
finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))
alpha_ok = (np.min(a1f) >= -1e-6) and (np.max(a1f) <= 1.0+1e-6)

# Energy conservation
rE_init = float(np.sum(a1*e1_v + (1-a1)*e2_v + 0.5*rho*u0*u0)*dx)
rE_final = float(np.sum(rEf)*dx)
dE_rel = abs(rE_final - rE_init)/abs(rE_init)

PASS = (err_p_rel < 1e-2) and (err_u < 1e-2) and finite and alpha_ok and (dE_rel < 1e-2)
status = 'PASS' if PASS else 'FAIL'

print(f"\n=== Case 02-A Test A (Water NASG / Air Ideal) — {status} ===")
print(f"  t_final = {float(t_f):.4f} s  (target 1.0)")
print(f"  max|(p-p₀)/p₀| = {err_p_rel:.3e}  (target < 1e-2)")
print(f"  max|u-u₀|      = {err_u:.3e}  (target < 1e-2)")
print(f"  |ΔE/E|         = {dE_rel:.3e}  (target < 1e-2)")
print(f"  α range        = [{float(np.min(a1f)):.3e}, {float(np.max(a1f)):.3e}]")
print(f"  finite         = {finite}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(x, a1f, 'b-o'); axes[0,0].set_title('α₁ (water vol frac)')
axes[0,0].set_ylim(-0.1, 1.1); axes[0,0].grid(True)
axes[0,1].plot(x, u_f, 'b-o'); axes[0,1].axhline(u0, color='r', ls='--', label='u₀=1.0')
axes[0,1].set_title(f'u (max err = {err_u:.2e})'); axes[0,1].legend(); axes[0,1].grid(True)
axes[1,0].plot(x, (p_f-p0)/p0, 'b-o'); axes[1,0].axhline(0, color='r', ls='--')
axes[1,0].set_title(f'(p-p₀)/p₀ (max = {err_p_rel:.2e})'); axes[1,0].grid(True)
axes[1,1].plot(x, rho_f, 'b-o')
axes[1,1].set_title(f'ρ_mix (water ρ₁={rho1:.1f}, air ρ₂={rho2:.3f})'); axes[1,1].grid(True)
fig.suptitle(f'Case 02-A NASG Water / Ideal Air — Iter 43 (SSP2+Richardson) — {status}')
plt.tight_layout()
plt.savefig('results/case_02/case_02A_nasg_iter43.png', dpi=120); plt.close()
print(f"  PNG: results/case_02/case_02A_nasg_iter43.png")
