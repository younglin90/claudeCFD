# Compressible flow in a Noble-Abel Stiffened-Gas fluid

> **출처:** Matei Ioan Radulescu, *arXiv* 2004.08750v1 (2020), physics.flu-dyn.
> **관련 실패:** IM1 block-tridiag (u,p) 가 NASG (1-bρ ≈ 0.3 water) factor 미반영 → 02-A advection 발산. 이 논문이 NASG 닫힌 형 sound speed, Riemann 불변량 제공.

---

## 1. 핵심 수식

### NASG EOS (Eq. 1, 단상)

$$
e(p, v) = \frac{(p + p_\infty)(v - b)}{\gamma - 1} + \eta
$$

> **의미:** Stiffened gas (P∞) + Noble-Abel covolume (b). v=1/ρ.

### NASG Sound Speed (Eq. 9) ★ 핵심

$$
c^2 = \gamma \frac{p + p_\infty}{\rho (1 - \rho b)}
$$

> **의미:** 일반 SG c² = γ(p+P∞)/ρ 에 `1/(1-ρb)` factor 추가.
> Water (ρ=1054, b=6.61e-4): 1/(1-0.696) = **3.29×** higher than SG 가정. IM1 행렬이 SG ρc² 가정 시 NASG 에서 stiffness 3× 과소평가 → 발산.

### NASG Isentropic Exponent (Eq. 12)

$$
\gamma_s \equiv \left(\frac{\partial \ln p}{\partial \ln \rho}\right)_s = \frac{\rho c^2}{p} = \gamma \frac{1 + p_\infty/p}{1 - \rho b}
$$

### NASG Riemann Invariants (Eq. 20) ★ 특성선 분석

$$
J_\pm = \frac{2}{\gamma-1}\sqrt{\gamma (p + p_\infty)(v - b)} \pm u
$$

> **의미:** SG 의 J_± = 2c/(γ-1) ± u 일반화. (v-b) factor 가 covolume 보정.

### NASG Isentrope (Eq. 11)

$$
(p + p_\infty)(v - b)^\gamma = \text{const}
$$

> **의미:** SG `(p+p∞) v^γ = const` 의 (v-b) 일반화.

---

## 2. 방법론

### 핵심 기여
NASG EOS 에 대한 **분석적 닫힌 형 표현** 도출:
1. Sound speed (Eq. 9) — covolume 1/(1-ρb) factor
2. Isentropic exponent (Eq. 12)
3. Riemann invariants J_± (Eq. 20) — 특성선 IMEX 의 implicit 행렬 derivation 에 직접 사용
4. Shock jump conditions
5. Riemann problem (shock tube) 풀이

### 기존 방법 대비 차이점

| 항목 | SG (perfect gas, 기존 IM1 가정) | NASG (이 논문) |
|------|-------------------------------|----------------|
| Sound speed | c² = γ(p+P∞)/ρ | c² = γ(p+P∞)/[ρ(1-ρb)] |
| Riemann invariant | 2c/(γ-1) ± u | (2/(γ-1))√[γ(p+P∞)(v-b)] ± u |
| Isentrope | (p+P∞)v^γ = const | (p+P∞)(v-b)^γ = const |
| 압력 stiffness | γ(p+P∞) | γ(p+P∞)/(1-ρb) — water 3.29× |

### IM1 적용 방향

IM1 block-tridiag (u,p) 행렬은 wave system:
$$
\begin{aligned}
\partial_t u + (1/\rho) \partial_x p &= 0 \\
\partial_t p + \rho c^2 \partial_x u &= 0
\end{aligned}
$$

NASG 에서 `ρc² = γ(p+P∞)/(1-ρb)` 대입 시 행렬 계수 자동 보정.
현재 `_peluchon_acoustic_im1` (L3765+) 의 Wood mixture sound speed 가 이미 EOS.sound_speed_sq 호출 → 이론상 NASG 처리.

**의심**: 행렬 RHS rE 보정 (`rE += -dt·∂(p̄ū)/∂x`) 에서 NASG 에너지식 (1-ρb) factor 미반영 가능. ρe = (p+γP∞)(1-ρb)/(γ-1) + ρη → SSP2 stage 누적 시 mass-energy 불일치.

---

## 3. 검증 및 시뮬레이션 설정

이 논문은 분석 도구 derivation 이 주 목적, 본격 numerical case 는 sequel 에서.

### Riemann Problem (§5)
- Water shock tube illustrating
- p₂ > p₁, u₁ = u₂ = 0
- mechanical equilibration: p₃ = p₄, u₃ = u₄
- 닫힌 형 식 (33) 으로 star pressure 계산:
  - SG 와 동일 형태 + `(v-b)` correction

---

## 4. claudeCFD 적용 메모

### 적용 가능 위치 1: `_peluchon_acoustic_im1` (L3765+)
- 현재 구조: `c_mix_s` 가 EOS.sound_speed_sq 호출 → NASG 처리됨
- **확인 필요**: `a_cell = ρ·c_mix` 가 ρ_star (보존변수에서 추출) 와 c_mix (EOS) 일관성
- ACID interface (acid_interface=True) 가 NASG 에서 작동하는지 확인

### 적용 가능 위치 2: SSP2 stage RHS rE (L4400+)
- IM1 후 `rE_new = rE_star - dt·∂(p̄ū)/∂x` — 압력-속도 product 만 갱신
- **NASG 에서**: ρe = (p+γP∞)(1-ρb)/(γ-1) + ρη 이므로 (1-ρb) factor 가 conservative 변환에서 자동 처리되지만 SSP2 stage 누적 시 drift 가능

### 수정 방향 (Iter 62 새 시도)
1. IM1 함수 안에서 NASG 검출 (b > 0) 시 `c² = γ(p+P∞)/(ρ(1-ρb))` 명시 (현재 EOS 호출이 정확한지 검증)
2. ACID interface (acid_interface=True) 강제 활성화 — interface 에서 cell-i ψ 만 사용 → NASG b·ρ 일관성
3. SSP2 sub-step 수 늘려 (substep=2) drift 감소
4. ITER 62 핵심: NASG Riemann invariant J_+ 을 IM1 RHS 에 사용 — SG-style 단순 (u, p) 대신 (J_+, J_-) characteristic form

### 주의사항
- water 에서 ρb ≈ 0.696, 1-ρb ≈ 0.304 → c² 가 SG 가정 대비 3.29× 큼
- IM1 σ = dt/dx 의 a_cell·σ stability bound 가 NASG 에서 √3.29 ≈ 1.81× 작아져야 함 → cfl 을 SG 대비 0.55× 로 낮춰야 함
- Iter 60-61 cfl=0.5 도 NaN → cfl 0.2 또는 substep 으로 회복 가능
