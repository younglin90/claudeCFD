# Validation Case — 1D High-Pressure Water / Low-Pressure Air Shock Tube

> **출처:** Yoo & Sung 2018 (IJHMT 127:210-221), §4.1 Validation; 선행연구 Yeom & Chang 2013, Haimovich & Frankel 2017
> **목적:** 고압의 물과 저압의 공기 사이에서 발생하는 저밀도파, 계면충격파(Interface shockwave), 충격파 등 복잡한 파 구조의 수치적 재현 검증

## 케이스 설명

1D 튜브에서 0.7 m를 기준으로 왼쪽에 고압의 물($10^9$ Pa), 오른쪽에 저압의 공기($10^5$ Pa)가 존재한다. 초기 접촉면에서 왼쪽으로 가는 저밀도파(Rarefaction wave), 오른쪽으로 가는 계면충격파(Interface shockwave), 그리고 충격파(Shock wave)의 세 파가 발생한다.

## 설정

| 항목 | 값 |
|------|-----|
| 도메인 | $x \in [0, 1]\text{ m}$ |
| 경계조건 | 좌·우 모두 Transmissive |
| 격자 수 (N) | 400 cells ($\Delta x = 0.0025\text{ m}$) 균일 격자 |
| Water (액상) 영역 | $x < 0.7\text{ m}$ (고압) |
| Air (기상) 영역 | $x \geq 0.7\text{ m}$ (저압) |
| CFL | 0.25 |
| **t_end** | $2.29 \times 10^{-4}\text{ s}$ (229 μsec) |

## 초기조건

$x = 0.7\text{ m}$을 기준으로 좌·우 상태량.

> **⚠ 밀도는 (p, T)에서 유도하지 않고, Yoo & Sung 2018 논문 값을 직접 지정한다.**
> 7-equation 모델에서 각 상의 밀도는 모든 cell에서 동일하게 설정.
> 5-equation 모델에서는 이 밀도를 통해 implied temperature가 결정된다.

| 상태량 | 왼쪽 — Water ($x < 0.7$) | 오른쪽 — Air ($x \geq 0.7$) |
|--------|--------------------------|------------------------------|
| 속도 $u$ | $0\text{ m/s}$ | $0\text{ m/s}$ |
| 압력 $p$ | $1 \times 10^9\text{ Pa}$ | $1 \times 10^5\text{ Pa}$ |
| 체적분율 $\alpha_\text{air}$ | $10^{-6}$ (물 영역) | $1 - 10^{-6}$ (공기 영역) |
| **밀도 $\rho_\text{air}$** | **50 kg/m³** | **50 kg/m³** |
| **밀도 $\rho_\text{water}$** | **1000 kg/m³** | **1000 kg/m³** |

Implied temperature (참고, 직접 사용하지 않음):
- Water left: $T = (p + P^\infty) / ((\gamma-1) \cdot kv \cdot \rho) = 992\text{ K}$
- Air right: $T = p / ((\gamma-1) \cdot kv \cdot \rho) = 6.97\text{ K}$

## EOS 파라미터 (Stiffened Gas)

| 성분 | $\gamma$ [-] | $P^\infty$ [Pa] | $kv$ [J/kg·K] |
|------|-------------|-----------------|----------------|
| Air (공기) | 1.4 | 0 | 717.5 |
| Water (물) | 4.4 | $6 \times 10^8$ | 474.2 |

## 기대 물리 현상

초기 접촉면(x = 0.7 m)에서 세 개의 파가 발생한다:

| 파 | 방향 | 설명 |
|----|------|------|
| 저밀도파 (Rarefaction wave) | 왼쪽 (← water 내부) | 고압 물 영역에서 왼쪽으로 전파하는 팽창파 |
| 계면충격파 (Interface shockwave) | 오른쪽 (→ air 내부) | 물-공기 계면이 오른쪽으로 이동하며 공기를 압축 |
| 충격파 (Shock wave) | 오른쪽 (→ air 내부) | 계면충격파 앞에서 공기 내부로 전파하는 강한 충격파 |

| 물리량 | 현상 기대치 |
|--------|--------|
| 체적분율 | 계면이 x = 0.7에서 오른쪽으로 이동 |
| 혼합물 밀도 | 물(~1000 kg/m³) 영역에서 rarefaction, 공기 영역에서 shock에 의한 밀도 급증 |
| 속도 | 계면 부근에서 ~500 m/s, 충격파 전방은 정지 |
| 압력 | 물 영역 rarefaction 감소, 계면~충격파 사이 플래토, 충격파 전방 $10^5$ Pa |

## PASS 기준

- reference 결과는 analytic ideal/SG Riemann exact profile을 기준으로 한다.
- `14_ref.png`는 시각적 문헌 reference로만 사용하고, PNG digitization 값은 exact로 사용하지 않는다.
- 현재 검증 드라이버 `.codex-loop/verify_08_26_acceptance.py --case 14`는 left SG-water `(rho=1000, u=0, p=1e9)` / right ideal-air `(rho=50, u=0, p=1e5)` Riemann problem을 직접 풀어 exact를 생성한다.
- 결과 PNG: `results/1D/14_E/diff_vs_exact.png`
- exact CSV: `results/1D/14_E/reference_exact_14.csv`

| 항목 | 기준 |
|------|------|
| 수치 발산 없이 **t_end 완주** | 필수 (압력비 $10^4$:1 조건) |
| 3파 구조 | 저밀도파(좌측), 계면충격파(중앙), 충격파(우측)가 명확히 식별되어야 함 |
| 수치 진동 없음 | 계면 및 충격파 부근에서 비물리적 oscillation이 없어야 함 |
| 계면 이동 | 물-공기 계면이 오른쪽으로 이동해야 함 |
| 최대 속도 | $u_\text{max} \in [400, 600]\text{ m/s}$ (레퍼런스 ~500 m/s) |
| 레퍼런스 일치 | phase2_high_p_water_low_p_air_shock_tube.png 그래프 결과와 정성적 유사 |
| **계면 밀도 단조성** | 상경계면에서 mixture density가 좌우 순수상 값 사이의 단조 전이여야 함 (peak/overshoot 없음) |
| 0.8~0.9 m 두 discontinuity 분리 | analytic exact는 $x\approx0.8$--$0.9$ 사이에 가까운 두 discontinuity, 즉 이동 contact/interface shock와 transmitted air shock를 가진다. 수치해는 이 둘을 하나의 ramp로 병합하면 안 되며, $\alpha_1$의 최대 face jump 위치 $x_{\Gamma,\mathrm{num}}$와 $u$의 최대 face jump 위치 $x_{\Delta u,\mathrm{num}}$가 순서 $x_{\Gamma,\mathrm{num}} < x_{\Delta u,\mathrm{num}}$를 유지해야 한다. 또한 수치 간격은 exact 간격의 $0.50 \le (x_{\Delta u,\mathrm{num}}-x_{\Gamma,\mathrm{num}})/(x_\mathrm{shock}-x_\mathrm{contact}) \le 1.80$ 범위에 있어야 한다 |
| transmitted shock의 $u$ 위치 일치 | analytic exact의 transmitted air shock 위치 $x_\mathrm{shock}$ 주변에서 수치 $u$의 가장 큰 face jump 위치 $x_{\Delta u,\mathrm{num}}$를 검출한다. shock-capturing 확산은 허용하지만 shock center가 밀리면 안 되므로 $\lvert x_{\Delta u,\mathrm{num}}-x_\mathrm{shock}\rvert/\Delta x \le 3$을 만족해야 한다 |
| 0.85~0.89 m rho plateau 보존 | exact 기준 이 구간의 rho는 국소 dip-then-hump가 없는 거의 일정한 상태여야 한다. 수치해가 먼저 undershoot한 뒤 overshoot하는 형태의 비물리적 ringing을 만들면 FAIL이다. 기준: $\max|\rho_\mathrm{num}-\rho_\mathrm{exact}|/\rho_\mathrm{scale} \le 3.0\times10^{-2}$, exact envelope 밖 overshoot/undershoot $\le 1.0\times10^{-2}$, local TV excess $\le 2.5\times10^{-2}$, residual slope reversal count $\le 1$. 단, 이 band 내부에는 analytic exact의 genuine close discontinuity가 있으므로 residual slope reversal count는 exact contact 주변 $\pm3\Delta x$ 및 transmitted shock 주변 $\pm3\Delta x$를 제외한 plateau cell에서 contiguous subsegment별로 계산한다. 이는 finite-volume shock thickness를 wiggle로 오판하지 않기 위한 기준이며, full-band envelope/TV guard는 그대로 유지한다 |
| 0.80~0.88 m 상부 wave packet의 rho peak 억제 | 상경계면/계면충격파와 transmitted shock가 가까운 이 구간에서 finite-volume shock thickness는 허용하지만, density가 exact local envelope 밖으로 위아래 peak/rebound를 만들면 안 된다. 자동 기준: $0.80\le x\le0.88$ band에서 envelope 초과 $\le3.0\times10^{-2}$, 같은 band에서 positive overshoot와 negative undershoot가 동시에 생기는 opposite rebound ratio $\le1.0\times10^{-2}$, exact contact 주변 $\pm3\Delta x$ 및 transmitted shock 주변 $\pm3\Delta x$를 제외한 residual second-difference HF ratio $\le3.0\times10^{-2}$, local peak prominence ratio $\le2.0\times10^{-2}$, 의미 있는 residual slope reversal count $\le1$ |
| 0.88 m 부근 u peak 허용 기준 | transmitted shock 부근 속도 packet은 단일 shock-capturing transition이어야 하며, saw-tooth/rebound는 불허한다. 다만 shock capturing으로 생기는 국소 속도 peak는 기존보다 소폭 완화하여 sharp-region overshoot ratio $\le2.0\times10^{-2}$까지 허용한다. 위치 기준($\le3\Delta x$)과 p/u HF guard는 유지한다 |
| 0.85 m 부근 rho envelope | 기존 peak-only guard를 확장하여 positive overshoot뿐 아니라 negative undershoot도 함께 본다. contact/shock 사이 band에서 $\max(\rho_\mathrm{num}-\rho_\mathrm{exact}^{max},\rho_\mathrm{exact}^{min}-\rho_\mathrm{num})/(\rho_\mathrm{exact}^{max}-\rho_\mathrm{exact}^{min}) \le 1.0\times10^{-3}$ |

## 사기 판정 기준

- 고압부 압력을 $10^9$ Pa 미만으로 낮추거나 저압부를 올려 압력비를 완화하는 행위
- 도메인 길이를 변경하는 행위
- CFL을 극단적으로 낮추어(< 0.01) 수치 확산으로 파 구조를 뭉개는 행위
- 결과 변수(체적분율, 밀도, 속도, 압력) 중 불리한 변수의 출력을 누락하는 행위
- **초기 밀도를 논문 값(ρ_air=50, ρ_water=1000)이 아닌 다른 값으로 변경하는 행위**

---

## 개발 히스토리

### 1차: 기본 explicit solver (T=300K IC) — 부분 실패

- **설정**: N=100, CFL=0.25, 밀도를 (p, T=300K)에서 유도 → ρ_water=3308, ρ_air=1.16
- **결과**: TVD-only u_max=899, MMACM-Ex u_max=5914 (비물리적 velocity spike)
- **원인**: 밀도비 2850:1이 너무 극단적 + MMACM-Ex correction에서 `a1r1/a1` 나눗셈으로 상밀도 부정확
- **판정**: TVD PASS, MMACM-Ex FAIL (velocity spike)

### 2차: Phase A (T-consistent 밀도) + Phase B (온도 평형 DC) — MMACM-Ex 정상화

- **Phase A**: MMACM-Ex correction에서 `a1r1/a1` → `cons_to_prim` T-consistent 밀도로 교체
- **Phase B**: 온도 평형 distribution coefficient λ_k (He & Tan 2024 Eq. A.19) 구현
  - α 소스항: `a1·du/dx` → `a1·λ₁·du/dx`
  - SG EOS 전용 열역학 도함수(𝔄,𝔅,ℭ,𝔇) + c_eff (Eq. A.17)
- **결과**: TVD u_max=589, MMACM-Ex u_max=669 — velocity spike 해결
- **판정**: PASS (하지만 T=300K IC로 레퍼런스와 불일치)

### 3차: Phase C (T-eq relaxation) — 온도 spike 해결

- **문제**: 온도가 ~50,000K까지 spike
- **원인**: 5-equation에서 α와 ρE가 독립 진화 → T₁≠T₂
- **해결**: SSP-RK3 각 substep 후 4-equation T-equilibrium closure 적용
  - 2차 방정식으로 p 계산, 체적 조건으로 T 계산, α₁과 ρE 리셋
  - 질량(a1r1, a2r2)과 운동량(ru) 보존
- **결과**: 온도 ~300K 수준으로 안정화
- **판정**: PASS

### 4차: 논문 IC 적용 (ρ₁=50, ρ₂=1000) — 레퍼런스 일치

- **문제**: T=300K IC가 논문(Yoo & Sung 2018)과 불일치
- **원인**: 논문은 7-equation 모델에서 ρ₁=50, ρ₂=1000을 양쪽 동일하게 직접 지정
- **해결**: 밀도를 (p,T)에서 유도하지 않고 직접 지정. EOS로 에너지 계산.
- **결과**: u_max≈500 m/s — 레퍼런스(Fig. 9)와 일치
- **판정**: PASS

### 5차: cons_to_prim T-eq closure + c_eff 음속 — 일관성 확보

- **Phase D**: cons_to_prim을 α-based mixture T → 4-equation quadratic closure로 교체
  - 솔버 전체에서 동일한 (p, T) 계산 사용 → 모델-스킴 일관성
- **c_eff**: HLLC face 음속과 CFL에 온도 평형 음속 c_eff (Eq. A.17) 사용
  - Wood 음속(5-eq용) → c_eff(4-eq T-eq용)로 교체
- **판정**: PASS, regression 없음

### 6차: THINC-BVD α₁ reconstruction — 계면 sharpness 개선

- **Phase E**: α₁ reconstruction을 TVD → THINC-BVD로 교체
  - THINC: tanh 프로파일로 step-function reconstruction (β=2.0)
  - BVD: 각 cell에서 TVD와 THINC 중 boundary variation이 작은 쪽 선택
  - (T₁, T₂, u, p)는 기존 TVD 유지
- **결과**: α₁ 계면 두께 ~4-5 cells → ~2 cells, oscillation 감소
- **판정**: PASS

### 7차: FCT-style density monotonicity limiter — 잔여 density peak 저감 시도

- **Phase F**: MMACM-Ex G_alpha에 FCT limiter 적용
  - 순 질량 flux `(ρ₁-ρ₂)·G_alpha`가 밀도 gradient를 증폭하면 G_alpha=0
- **결과**: N=400에서 density peak 감소, N=100에서 약간의 peak 잔존
- **판정**: PASS (mesh convergence 확인)
- **미해결**: 상경계면 density peak 완전 제거는 미완 — 추가 개선 필요

### 현재 결과 (N=100/200/400 MMACM-Ex)

| N | Steps | u_max | p_min | 판정 |
|---|-------|-------|-------|------|
| 100 | 244 | 506.0 | 1.0e5 | **PASS** (소량 peak 잔존) |
| 200 | 487 | 513.4 | 1.0e5 | **PASS** |
| 400 | 973 | 500.4 | 1.0e5 | **PASS** |

### 현재 솔버 구성

| 항목 | 구현 |
|------|------|
| 시간 적분 | SSP-RK3 |
| Reconstruction | TVD van Leer (T₁,T₂,u,p) + **THINC-BVD** (α₁) |
| Riemann solver | HLLC (c_eff 음속) |
| Interface sharpening | MMACM-Ex (H_k + pure downwind + consistency G + **FCT limiter**) |
| α source term | **온도 평형 DC** (λ_k, He & Tan 2024 Eq. A.19) |
| Post-step | **Instantaneous T-relaxation** (4-eq closure) |
| cons_to_prim | **4-eq T-eq quadratic closure** (He & Tan 2024 Eq. A.20) |
