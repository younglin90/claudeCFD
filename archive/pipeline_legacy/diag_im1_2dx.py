"""IM1의 순수 2Δx mode 감쇠 검증.
p = p0 + (-1)^i * ε 초기화 후 IM1 한 단계 돌려 실제 damping factor 측정.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from solver.He2024.explicit_mmacm_ex import _peluchon_acoustic_im1, cons_to_prim

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
N = 100; L = 1.0; dx = L/N
x = np.linspace(dx/2, L-dx/2, N)

# 순수 2Δx 모드 생성
p0 = 1e5; eps = 100.0
sign = (-1.0) ** np.arange(N)  # +1, -1, +1, ...
p_init = p0 + eps * sign
u_init = np.zeros(N)  # u uniform = 0

T0 = 293.0
rho_water = (p_init + 6e8) / (3.4 * 474.2 * T0)
rho_air = p_init / (0.4 * 717.5 * T0)
a_air_frac = 1e-6 * np.ones(N)
a1r1 = a_air_frac * rho_air
a2r2 = (1 - a_air_frac) * rho_water
rho = a1r1 + a2r2
ru = rho * u_init
rho_e0 = a_air_frac * p_init / 0.4 + (1-a_air_frac) * (p_init + 4.4*6e8) / 3.4
rE = rho_e0

# 초기 2Δx 진폭
def extract_2dx_amp(p):
    return np.abs(np.mean(p * sign))

amp_init = extract_2dx_amp(p_init)
print(f"Initial 2dx amplitude: {amp_init:.3f}")
print(f"Expected damping per IM1 step = 1/(1+2*CFL_acoustic)")

# IM1 한 스텝씩 여러 dt로 테스트
c_water = np.sqrt(4.4 * (p0 + 6e8) / rho_water[0])
print(f"c_water = {c_water:.1f}, dx = {dx:.4f}")

print("\nEmpirical IM1 2Δx damping per step:")
print(f"{'CFL_acoustic':<15s} {'dt':<12s} {'amp after 1 IM1':<18s} {'damping':<10s} {'analytic 1/(1+2CFL)':<20s}")
for cfl in [0.4, 0.2, 0.1, 0.05, 0.01]:
    dt = cfl * dx / c_water
    a1r1_new, a2r2_new, ru_new, rE_new = _peluchon_acoustic_im1(
        a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy(), a_air_frac.copy(),
        ph1, ph2, dx, dt, 'transmissive', 'transmissive')
    p_new, u_new, _, _, _, _, _, _ = cons_to_prim(a1r1_new, a2r2_new, ru_new, rE_new, a_air_frac, ph1, ph2)
    amp_new = extract_2dx_amp(p_new)
    damp = amp_new / amp_init
    analytic = 1.0 / (1.0 + 2*cfl)
    print(f"  {cfl:<15.3f} {dt:<12.3e} {amp_new:<18.3f} {damp:<10.4f} {analytic:<20.4f}")

# 여러 스텝 누적
print("\nIM1 5-step accumulation (CFL=0.4):")
dt = 0.4 * dx / c_water
a_c, b_c, r_c, e_c = a1r1.copy(), a2r2.copy(), ru.copy(), rE.copy()
for step in range(10):
    a_c, b_c, r_c, e_c = _peluchon_acoustic_im1(a_c, b_c, r_c, e_c, a_air_frac, ph1, ph2, dx, dt,
                                                  'transmissive', 'transmissive')
    p_c, u_c, _, _, _, _, _, _ = cons_to_prim(a_c, b_c, r_c, e_c, a_air_frac, ph1, ph2)
    amp = extract_2dx_amp(p_c)
    print(f"  step {step+1}: amp={amp:.4e}, damp={amp/amp_init:.4e}")
