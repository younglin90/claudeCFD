"""Case 02-B Test B — 3-species (Air/Helium/SF6) advection at u=100 m/s.
Periodic, p₀=1e5, u₀=100, t_end=0.01s (one period), N=100.
PASS: err_p < 1e-10, err_u < 1e-10.
"""
import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.kapila_k import solve_kapila_K
from solver.He2024.eos_general import IdealEOS
os.makedirs('results/case_02', exist_ok=True)

# 3 species per spec Table
eos_air    = IdealEOS(gamma=1.4,  kv=717.5)
eos_He     = IdealEOS(gamma=1.667, kv=3116.0)
eos_SF6    = IdealEOS(gamma=1.094, kv=665.0)
eos_list = [eos_air, eos_He, eos_SF6]

N = 100; L = 1.0; dx = L/N
x = (np.arange(N) + 0.5)*dx

p0 = 1e5; u0 = 100.0; T0 = 300.0

# 4-region: Region 4 (x≥0.75) reconnects to Region 1 (Air) for periodic
# α_air = 1 in [0, 0.25] ∪ [0.75, 1]
# α_He  = 1 in [0.25, 0.5]
# α_SF6 = 1 in [0.5, 0.75]
eps = 1e-6
a_air = np.where((x < 0.25) | (x >= 0.75), 1.0 - 2*eps, eps)
a_He  = np.where((x >= 0.25) & (x < 0.5), 1.0 - 2*eps, eps)
a_SF6 = np.where((x >= 0.5) & (x < 0.75), 1.0 - 2*eps, eps)
# Renormalize so sum = 1
s = a_air + a_He + a_SF6
a_air /= s; a_He /= s; a_SF6 /= s
a_list = [a_air, a_He, a_SF6]

rho_air = p0/((eos_air.gamma-1)*eos_air.kv*T0)
rho_He  = p0/((eos_He.gamma-1)*eos_He.kv*T0)
rho_SF6 = p0/((eos_SF6.gamma-1)*eos_SF6.kv*T0)
print(f"  ρ_air = {rho_air:.4f}, ρ_He = {rho_He:.4f}, ρ_SF6 = {rho_SF6:.4f}")

ar_air = a_air*rho_air; ar_He = a_He*rho_He; ar_SF6 = a_SF6*rho_SF6
ar_list = [ar_air, ar_He, ar_SF6]

rho = ar_air + ar_He + ar_SF6
ru = rho*u0

# Internal energy density per phase (ideal gas)
e_air_v  = p0/(eos_air.gamma-1)
e_He_v   = p0/(eos_He.gamma-1)
e_SF6_v  = p0/(eos_SF6.gamma-1)
rE = a_air*e_air_v + a_He*e_He_v + a_SF6*e_SF6_v + 0.5*rho*u0*u0

# Run K=3 explicit (acoustic CFL 0.4 per spec, t_end=0.01 = exactly one period)
out = solve_kapila_K(eos_list, ar_list, ru, rE, a_list, dx=dx, t_end=0.01,
                     cfl=0.4, max_steps=5000,
                     bc_l='periodic', bc_r='periodic',
                     print_interval=200)
t_f, ar_list_f, ru_f, rE_f, a_list_f = out
ar_air_f, ar_He_f, ar_SF6_f = ar_list_f
a_air_f, a_He_f, a_SF6_f = a_list_f

rho_f = ar_air_f + ar_He_f + ar_SF6_f
u_f = ru_f/rho_f
ke = 0.5*ru_f*u_f
# Pressure: ρe = α_k·(p+γ_k·p∞_k)/(γ_k-1) + ke. For ideal: ρe = Σα_k·p/(γ_k-1)
ie = rE_f - ke
Gi = a_air_f/(eos_air.gamma-1) + a_He_f/(eos_He.gamma-1) + a_SF6_f/(eos_SF6.gamma-1)
p_f = ie/Gi

err_p = float(np.max(np.abs(p_f - p0))/p0)
err_u = float(np.max(np.abs(u_f - u0)))
sum_a = a_air_f + a_He_f + a_SF6_f
err_sum = float(np.max(np.abs(sum_a - 1.0)))
finite = bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f)))

PASS = (err_p < 1e-10) and (err_u < 1e-10) and (err_sum < 1e-12) and finite
status = 'PASS' if PASS else 'FAIL'

print(f"\n=== Case 02-B Test B (3-species K=3, u=100 m/s) — {status} ===")
print(f"  t_final  = {float(t_f):.4f} s  (target 0.01)")
print(f"  err_p    = {err_p:.3e}  (target < 1e-10)")
print(f"  err_u    = {err_u:.3e}  (target < 1e-10)")
print(f"  Σα-1     = {err_sum:.3e}  (target < 1e-12)")
print(f"  finite   = {finite}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(x, a_air_f, 'b-', label='Air')
axes[0,0].plot(x, a_He_f, 'g-', label='He')
axes[0,0].plot(x, a_SF6_f, 'r-', label='SF6')
axes[0,0].legend(); axes[0,0].set_title('α_k after one period'); axes[0,0].grid(True)
axes[0,1].plot(x, u_f - u0, 'b-')
axes[0,1].set_title(f'u - u₀ (max = {err_u:.2e})'); axes[0,1].grid(True)
axes[1,0].plot(x, (p_f-p0)/p0, 'b-')
axes[1,0].set_title(f'(p-p₀)/p₀ (max = {err_p:.2e})'); axes[1,0].grid(True)
axes[1,1].plot(x, rho_f, 'b-')
axes[1,1].set_title('ρ_mix'); axes[1,1].grid(True)
fig.suptitle(f'02-B 3-species K=3 advection u=100 m/s — {status}')
plt.tight_layout()
plt.savefig('results/case_02/case_02B_iter43.png', dpi=120); plt.close()
print(f"  PNG: results/case_02/case_02B_iter43.png")
