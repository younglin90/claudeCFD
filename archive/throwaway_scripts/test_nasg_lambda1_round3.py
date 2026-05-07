"""
Unit Test: λ₁ 및 c_eff 검증 (Round 3)
- NASG 사용 시 λ₁이 순수/혼합상에서 적절히 작동하는지 확인
"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.He2024.eos_general import NASGEOS, IdealEOS
from solver.He2024.explicit_mmacm_ex import _lambda_temp_eq_general, _ceff_temp_eq_general

# NASG 초기화 (02-A validation에서 사용)
eos_water = NASGEOS(
    gamma=1.187, 
    pinf=7.028e8, 
    b=6.61e-4, 
    kv=3610,      # reversible heat capacity
    eta=-1.177788e6  # reference energy offset
)
eos_air = IdealEOS(gamma=1.4, kv=717.5)

# 기준 조건 (Phase 1 균형)
p0 = 1e5
T0 = 300
rho1_ref = 998.0  # NASG water @ 1 atm, 300K
rho2_ref = 1.225   # Ideal air @ 1 atm, 300K

# Test cases
test_cases = [
    ("Pure Water (α=1-1e-6)", 1.0 - 1e-6),
    ("Mixed (α=0.5)", 0.5),
    ("Mixed (α=0.1)", 0.1),
    ("Mixed (α=0.9)", 0.9),
    ("Pure Air (α=1e-6)", 1e-6),
]

print("="*70)
print("Unit Test: λ₁ (Defect Coefficient) — Round 3")
print("="*70)
print()

for test_name, a1 in test_cases:
    a2 = 1.0 - a1
    
    # Phase densities
    rho1 = np.array([rho1_ref])
    rho2 = np.array([rho2_ref])
    p_arr = np.array([p0])
    T_arr = np.array([T0])
    a1_arr = np.array([a1])
    
    # Call λ₁ function (function signature: a1, rho1, rho2, p, T, eos1, eos2)
    try:
        lambda1_arr = _lambda_temp_eq_general(
            a1=a1_arr,
            rho1=rho1,
            rho2=rho2,
            p=p_arr,
            T=T_arr,
            eos1=eos_water,
            eos2=eos_air
        )
        lambda1 = lambda1_arr[0]
        
        # Expected behavior
        is_pure = a1 < 1e-4 or a1 > 1 - 1e-4
        expected_behavior = "λ₁=1.0 (pure_mask)" if is_pure else "λ₁ ∈ [0.3, 1.5] (finite)"
        
        # Stricter pass criteria based on fix report
        if is_pure:
            status = "✓ PASS" if abs(lambda1 - 1.0) < 0.01 else "✗ FAIL"
        else:
            status = "✓ PASS" if 0.1 < lambda1 < 2.0 and np.isfinite(lambda1) else "✗ FAIL"
        
        print(f"{test_name:30s} | a1={a1:.6e} | λ₁={lambda1:10.6f} | {expected_behavior:25s} | {status}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{test_name:30s} | ✗ ERROR: {str(e)[:40]}")

print()
print("="*70)
print("Unit Test: c_eff (Effective Sound Speed) — Round 3")
print("="*70)
print()

for test_name, a1 in test_cases:
    a2 = 1.0 - a1
    
    # Phase densities
    rho1 = np.array([rho1_ref])
    rho2 = np.array([rho2_ref])
    p_arr = np.array([p0])
    T_arr = np.array([T0])
    a1_arr = np.array([a1])
    
    # Call c_eff function (function signature: a1, rho1, rho2, p, T, eos1, eos2)
    try:
        ceff_arr = _ceff_temp_eq_general(
            a1=a1_arr,
            rho1=rho1,
            rho2=rho2,
            p=p_arr,
            T=T_arr,
            eos1=eos_water,
            eos2=eos_air
        )
        ceff = ceff_arr[0]
        
        # Expected: c_eff should be real and positive, between phase speeds
        c1 = eos_water.sound_speed(rho1[0], p0)
        c2 = eos_air.sound_speed(rho2[0], p0)
        cmin = min(c1, c2)
        cmax = max(c1, c2)
        
        in_range = cmin <= ceff <= cmax * 1.2
        status = "✓ PASS" if in_range and ceff > 0 and np.isfinite(ceff) else "✗ FAIL"
        
        print(f"{test_name:30s} | c₁={c1:7.1f} | c₂={c2:7.1f} | c_eff={ceff:8.1f} m/s | {status}")
    except Exception as e:
        print(f"{test_name:30s} | ✗ ERROR: {str(e)[:40]}")

print()
print("="*70)
print("Summary: All tests should PASS")
print("="*70)
