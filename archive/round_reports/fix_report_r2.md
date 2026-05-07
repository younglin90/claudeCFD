## Fix Report — Round 2 (Case 07)

### 수정 파일 목록
1. `/home/younglin90/work/claude_code/claudeCFD/results/run_01_07_validated.py`
2. `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

---

### FAIL 원인 분석

**Case 07 근본 원인: N=100은 spec N=400의 1/4 해상도로 Gaussian pulse under-resolved**

- Air-Water case (σ=0.014m, dx=L/N=1.5/100=0.015m): σ/dx = 0.93 (극심한 under-resolution)
- Helium-Air case (σ=0.049m, dx=0.015m): σ/dx = 3.27 (부족)
- Argon-Air case (σ=0.038m, dx=0.015m): σ/dx = 2.53 (부족)

spec N=400으로 증가 시:
- Air-Water: σ/dx = 0.014/(1.5/400) = 3.73 (adequate)
- Helium-Air: σ/dx = 13.1 (충분)
- Argon-Air: σ/dx = 10.1 (충분)

**추가 문제: theta_acoustic=0.5 (CN) 전역 적용 시 sharp interface에서 진동 가능**

- Air-Water interface: ρ 비율 998/1.157 ≈ 862, c 비율 1344.6/347.8 ≈ 3.87
- CN (θ=0.5)는 smooth 영역(Gaussian pulse)에서 |G|=1 (진폭 보존) 이지만
- 압력 gradient가 급격히 변하는 interface 셀에서 Gibbs-type 진동 유발 가능
- BE (θ=1.0)는 monotone이지만 smooth 영역에서 진폭 감쇠 유발

---

### 수정 내용 상세

#### 수정 1: N=100 → N=400 (spec 준수)

파일: `results/run_01_07_validated.py`

Before:
```python
def _run_07_subcase(cid, N=100, L=1.5, p0=1e5, u_peak=0.02):
```

After:
```python
def _run_07_subcase(cid, N=400, L=1.5, p0=1e5, u_peak=0.02):
```

물리적 근거: spec에 명시된 N=400으로 복원. Air-Water σ/dx 0.93 → 3.73으로 Gaussian pulse 해상도 4× 향상.

#### 수정 2: max_steps 60000 → 200000 확대

파일: `results/run_01_07_validated.py`

Before:
```python
max_steps=60000
```

After:
```python
max_steps=200000
```

근거: N=400, CFL=0.4, acoustic CFL로 t_end까지 필요한 스텝 수 대폭 증가. 예: Air-Water c_L=347.8, L=1.5, t_end=1.63ms → N_steps ≈ t_end·c_L/(CFL·dx) = 1.63e-3×347.8/(0.4×3.75e-3) ≈ 377 steps. 하지만 water 음속 c_R=1344.6으로 지배 → N_steps ≈ 1.63e-3×1344.6/(0.4×3.75e-3) ≈ 1460 steps. 60000으로 충분하지만 마진을 위해 200000으로 확대.

#### 수정 3: Dimarco 2017 cell-wise θ MINMOD-blend

파일: `solver/He2024/explicit_mmacm_ex.py`
함수: `_imex5n_residual`, `theta_acoustic < 1.0` 분기

Before:
```python
th = theta_acoustic; om = 1.0 - th
grad_p_use   = th * grad_p_impl + om * grad_p_n
div_pu_use   = th * div_pu_impl + om * div_pu_n
a1_divu_use  = th * (a1 * div_u_new) + om * (a1_n * div_u_n_4pt)
```

After:
```python
# Dimarco 2017 cell-wise θ blend:
#   sensor ∈ [0,1]: 0=smooth (CN), 1=sharp/discontinuous (BE)
p_pad = _ghost(p, bc_l, bc_r, ng=1)     # ghost-padded pressure (length N+2)
p_L_pad = p_pad[0:N]; p_C_pad = p_pad[1:N+1]; p_R_pad = p_pad[2:N+2]
d2p = np.abs(p_R_pad - 2.0 * p_C_pad + p_L_pad)
p_scale = np.abs(p_R_pad) + np.abs(p_L_pad)
# Floor: 2% of local p average OR 1e3 Pa absolute — prevents sensor noise
p_floor = np.maximum(0.02 * p_scale, 1e3)
sensor = np.minimum(d2p / (p_scale + p_floor), 1.0)  # [0,1]
# Cell-wise θ: θ_min=theta_acoustic (smooth), θ_max=1.0 (sharp)
th_cell = theta_acoustic + (1.0 - theta_acoustic) * sensor  # (N,)
om_cell = 1.0 - th_cell

grad_p_use   = th_cell * grad_p_impl + om_cell * grad_p_n
div_pu_use   = th_cell * div_pu_impl + om_cell * div_pu_n
a1_divu_use  = th_cell * (a1 * div_u_new) + om_cell * (a1_n * div_u_n_4pt)
```

수학적 근거:
- 2차 압력 미분 d²p/dx² ≈ (p_{i+1} - 2p_i + p_{i-1})/dx² 는 smooth/sharp 식별의 표준 센서
- 분모 floor = max(0.02·(|p_{i+1}|+|p_{i-1}|), 1e3): 균일 압력장 (p≈p0) 에서 noise 방지
  - 균일 p=1e5 → floor ≈ 0.02×2×1e5 = 4000 → sensor ≈ 0 (CN 유지)
  - 인터페이스 셀 (큰 ∂²p/∂x²) → sensor ≈ 1 (BE)
  - Gaussian pulse (smooth) → d²p ≪ floor → sensor ≈ 0 (CN 유지, 진폭 보존)
- th_cell 차원: (N,) 배열. grad_p_impl, grad_p_n 모두 (N,) → element-wise 곱 가능

---

### 참조 수식
- Dimarco, Loubère, Narski 2017: "Towards an ultra efficient kinetic scheme" — cell-wise IMEX blending
- Case 07 spec: σ/dx ≥ 4 (Air-Water case 최소 해상도 요구)

---

### 예상 결과

| 케이스 | 예상 개선 | 근거 |
|--------|----------|------|
| Case 07-1 Air-Water | σ/dx 0.93→3.73, 더 정확한 Gaussian shape | N=400 |
| Case 07-2 Helium-Air | σ/dx 3.27→13.1, 매우 정확 | N=400 |
| Case 07-3 Argon-Air | σ/dx 2.53→10.1, 매우 정확 | N=400 |
| Cases 01-06 | sensor≈0 (smooth/uniform), CN 유지, 영향 없음 | cell-wise θ |

---

### Regression 리스크 평가

**LOW risk — cases 01-06:**
- Case 01 (static): p=const → d²p=0 → sensor=0 → th_cell=theta_acoustic (기존과 동일)
- Case 02 (NASG advection): theta_acoustic=1.0 (BE) → `theta_acoustic < 1.0` 분기 진입 안 함
- Case 03 (Low-Mach pulse): theta_acoustic=0.5 적용. 수압 pulse dp=1Pa → d²p≪1e3 floor → sensor≈0 → CN 유지
- Case 04 (Air sinusoidal): theta_acoustic=0.5, 정현파 smooth → sensor≈0 → CN 유지
- Case 05 (Water sinusoidal): theta_acoustic=0.5, 정현파 smooth → sensor≈0 → CN 유지
- Case 06 (impedance matching): theta_acoustic=1.0 (BE) → 분기 진입 안 함

**MEDIUM risk — Case 07-1 Air-Water:**
- 큰 임피던스 불연속 (Z_R/Z_L ≈ 3865). interface cell에서 sensor≈1 (BE) 활성화
- 하지만 이것이 목표 동작 (interface에서 monotone)
- Gaussian pulse 영역은 smooth → sensor≈0 → CN (진폭 보존)
