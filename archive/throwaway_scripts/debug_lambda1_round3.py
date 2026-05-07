"""
Debug: λ₁ 계산 중간값 추적 (α=0.5)
"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.He2024.eos_general import NASGEOS, IdealEOS

# NASG EOS
eos_water = NASGEOS(
    gamma=1.187, 
    pinf=7.028e8, 
    b=6.61e-4, 
    kv=3610,
    eta=-1.177788e6
)
eos_air = IdealEOS(gamma=1.4, kv=717.5)

# 혼합상 조건
a1 = 0.5
a2 = 0.5
p0 = 1e5
T0 = 300
rho1 = 998.0
rho2 = 1.225
rho = a1 * rho1 + a2 * rho2

_EPS = 1e-30

print("="*70)
print("Debug: λ₁ 계산 (He & Zhao 2025 Eq. 53), α=0.5")
print("="*70)
print(f"a1={a1}, a2={a2}, p={p0}, T={T0}")
print(f"ρ1={rho1}, ρ2={rho2}, ρ_mix={rho:.2f}")
print()

# EOS 도함수 계산
dpdT1 = eos_water.dpdT_rho(rho1, T0)
dpdrho1_T = eos_water.dpdrho_T(rho1, T0)
dpdT2 = eos_air.dpdT_rho(rho2, T0)
dpdrho2_T = eos_air.dpdrho_T(rho2, T0)

print(f"(∂p/∂T)_ρ|1 = {dpdT1:.3e} Pa/K")
print(f"(∂p/∂ρ)_T|1 = {dpdrho1_T:.3e} Pa·m³/kg")
print(f"(∂p/∂T)_ρ|2 = {dpdT2:.3e} Pa/K")
print(f"(∂p/∂ρ)_T|2 = {dpdrho2_T:.3e} Pa·m³/kg")
print()

# κ_{T,k}, β_k 계산
kappa_T1 = 1.0 / np.maximum(rho1 * dpdrho1_T, _EPS)
kappa_T2 = 1.0 / np.maximum(rho2 * dpdrho2_T, _EPS)
beta1 = dpdT1 / np.maximum(rho1 * dpdrho1_T, _EPS)
beta2 = dpdT2 / np.maximum(rho2 * dpdrho2_T, _EPS)

print(f"κ_T1 = 1/(ρ1·(∂p/∂ρ)_T|1) = {kappa_T1:.3e} m³/kg/Pa")
print(f"κ_T2 = 1/(ρ2·(∂p/∂ρ)_T|2) = {kappa_T2:.3e} m³/kg/Pa")
print(f"β1 = (∂p/∂T)_ρ|1 / (ρ1·(∂p/∂ρ)_T|1) = {beta1:.6f} 1/K")
print(f"β2 = (∂p/∂T)_ρ|2 / (ρ2·(∂p/∂ρ)_T|2) = {beta2:.6f} 1/K")
print()

# Cp_k 계산
cv1 = eos_water.cv(rho1, T0)
cv2 = eos_air.cv(rho2, T0)
Cp1 = cv1 + T0 * dpdT1 ** 2 / np.maximum(rho1 ** 2 * dpdrho1_T, _EPS)
Cp2 = cv2 + T0 * dpdT2 ** 2 / np.maximum(rho2 ** 2 * dpdrho2_T, _EPS)

print(f"cv1 = {cv1:.3e} J/kg/K")
print(f"cv2 = {cv2:.3e} J/kg/K")
print(f"Cp1 = {Cp1:.3e} J/kg/K")
print(f"Cp2 = {Cp2:.3e} J/kg/K")
print()

# 혼합량
nu = 1.0 / np.maximum(rho, _EPS)
Y1 = a1 * rho1 / np.maximum(rho, _EPS)
Y2 = a2 * rho2 / np.maximum(rho, _EPS)
kappa_T = a1 * kappa_T1 + a2 * kappa_T2
beta = a1 * beta1 + a2 * beta2
C_P = Y1 * Cp1 + Y2 * Cp2

print(f"ν = 1/ρ = {nu:.3e} m³/kg")
print(f"Y1 = α1·ρ1/ρ = {Y1:.6f}")
print(f"Y2 = α2·ρ2/ρ = {Y2:.6f}")
print(f"κ_T = α1·κ_T1 + α2·κ_T2 = {kappa_T:.3e}")
print(f"β = α1·β1 + α2·β2 = {beta:.6f} 1/K")
print(f"C_P = Y1·Cp1 + Y2·Cp2 = {C_P:.3e} J/kg/K")
print()

# λ₁ 분자/분모 계산
T_nu_beta = T0 * nu * beta
numerator = kappa_T1 * C_P - T_nu_beta * beta1
denominator = kappa_T * C_P - T_nu_beta * beta

print(f"T·ν·β = {T_nu_beta:.3e}")
print(f"κ_T1·C_P = {kappa_T1 * C_P:.3e}")
print(f"T·ν·β·β1 = {T_nu_beta * beta1:.3e}")
print(f"Numerator = κ_T1·C_P - T·ν·β·β1 = {numerator:.3e}")
print()
print(f"κ_T·C_P = {kappa_T * C_P:.3e}")
print(f"T·ν·β² = {T_nu_beta * beta:.3e}")
print(f"Denominator = κ_T·C_P - T·ν·β² = {denominator:.3e}")
print()

# λ₁
lambda1 = numerator / np.where(np.abs(denominator) > _EPS, denominator, _EPS * np.sign(denominator + _EPS))
print(f"λ₁ = numerator / denominator = {lambda1:.6f}")
print()

# 부호 진단
if denominator > 0:
    print(f"Denominator > 0, 정상 계산")
elif denominator == 0:
    print(f"Denominator = 0, singularity!")
else:
    print(f"Denominator < 0, 부호 반전")

print()
print("="*70)
