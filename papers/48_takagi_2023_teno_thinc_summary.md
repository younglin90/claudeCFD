# High-Order Low-Dissipation Shock-Resolving TENO-THINC Schemes

> **출처:** Takagi, Wakimura, Fu, Xiao, *Communications in Computational Physics* **34**(4) (2023) 1043-1078. DOI: 10.4208/cicp.oa-2023-0061
> **관련 실패:** Case 10-1 Air-Water transmission — **TENO + THINC 결합** 으로 shock/interface sharp, smooth low-diss 동시 달성

---

## 1. 핵심 수식

### TENO-THINC hybrid 개념 (Fig. 1 원리)

$$
\hat{f}_{i+1/2} = \begin{cases} 
\hat{f}^{\text{TENO}}_{i+1/2} & \text{if all } \delta_k = 1 \text{ (smooth)} \\
\hat{f}^{\text{THINC}}_{i+1/2} & \text{if any } \delta_k = 0 \text{ (discontinuity)}
\end{cases}
$$

여기서 `δ_k ∈ {0, 1}` 는 TENO 의 stencil-selection indicator (0 = discard, 1 = use).

### δ function (TENO discontinuity indicator)

$$
\delta_k = \begin{cases} 1 & \text{if } \chi_k \geq C_T \\ 0 & \text{otherwise} \end{cases}
$$

**→ TENO 가 discard 판정 한 지점 = discontinuity 위치** (별도 sensor 불필요!)

### THINC reconstruction (Xiao-Shyue 2014)

$$
\tilde{q}(x) = \bar{q}_{\min} + \frac{\bar{q}_{\max} - \bar{q}_{\min}}{2} \left(1 + \tanh(\beta \xi)\right)
$$

- `β`: steepness parameter (~2-3)
- ξ: normalized position in cell
- Sub-cell resolution 으로 discontinuity 해상

### 6-point & 8-point TENO-THINC (본 논문 기여)

- TENO6-THINC: 6-point stencil + THINC blend
- TENO8-THINC: 8-point stencil + THINC blend
- 기존 TENO5-THINC (Fu 2016, 2019) 보다 고차, 정확도 향상

---

## 2. 방법론

### 알고리즘 개요

1. K-point TENO candidate stencil 구성
2. 각 stencil smoothness indicator `χ_k` 계산
3. Binary δ function 으로 discard 판정
4. **모든 stencil smooth (δ_k = 1)**: TENO linear optimal (low-diss)
5. **일부 discard (δ_k = 0)**: THINC 로 교체 (sub-cell discontinuity)

### 핵심 아이디어 — BVD 원리 활용

**BVD (Boundary Variation Diminishing)**:
- Face 좌우 reconstructed 값 차이 최소화
- Shock 에서는 THINC (작은 차이), smooth 에서는 TENO (작은 차이)
- TENO-THINC 는 이 BVD 원리에 δ function 을 결합

### 기존 방법 대비 차이점

| 항목 | WENO5 | TENO5 | **TENO5-THINC (Fu 2019)** | **TENO6/8-THINC (본 논문)** |
|------|-------|-------|---------------------------|----------------------------|
| Smooth 영역 | Partial diss | Zero diss | Zero diss | **Zero diss + 고차** |
| Shock 영역 | ENO-smooth | ENO | **Sub-cell THINC** | **Sub-cell THINC** |
| Contact discontinuity | Smeared | Smeared | **1-cell sharp** | **1-cell sharp + 고차 부근** |
| 차수 | 5 | 5 | 5 | **6, 8** |
| 감지 sensor | βₖ | χₖ (cutoff) | χₖ (δ 자동) | χₖ (δ 자동) |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | TENO5-JS | TENO5-THINC | TENO6/8-THINC |
|---|--------|----------|-------------|---------------|
| 4.1 | Contact discontinuity 단독 advection | 5-10 cell smearing | **1-cell** | 1-cell + 주변 고차 |
| 4.2 | Sod shock tube | OK | Sharp | Sharp |
| 4.3 | Shu-Osher | Good | Good + sharp | **Best** |
| 4.4 | 2D Riemann | OK | OK | **Very sharp** |
| 4.5 | Rayleigh-Taylor | Diss | Low diss | **Lowest diss** |

### 주요 결과

- Contact discontinuity: 5 cells (WENO/TENO) → **1 cell** (TENO-THINC)
- Long-time advection: TENO 도 서서히 diffuse but TENO-THINC 는 유지
- Shock 안정성: 전혀 손실 없이 sharp interface 동시 달성

---

## 4. claudeCFD 적용 메모

### 현재 vs Takagi 2023

**현재 우리 (Round 22)**:
- `_tvd_reconstruct` (van Leer): 2nd order, contact 에서 5-cell smearing
- `_weno5_reconstruct` (Jiang-Shu): 5th order, smooth 영역 partial diss
- `_thinc_bvd_reconstruct` (Deng 2018): α 전용, sub-cell sharp interface

**Takagi 권장 (TENO-THINC)**:
- All primitives 에 TENO-THINC blend
- `δ` indicator 로 자동 TENO/THINC 선택
- α 외에도 ρ_k 에 THINC 적용 (contact)

### 적용 위치
`solver/He2024/explicit_mmacm_ex.py::_advective_rhs_imex` L3905+ reconstruction

### 구현 복잡도
- TENO5 reconstruction: 이미 WENO5 구조 재사용 가능 (binary δ weights만 변경)
- THINC-BVD: 이미 구현되어 있음 (α 전용)
- TENO-THINC blend: δ function 으로 자동 선택

### Case 10-1 예상 효과

**현재 (Chamarthi 변수별 WENO5)**: trans = 2.019 (+1.0%)

**Takagi TENO-THINC 적용 시 예상**:
- **ρ₁, ρ₂**: TENO-THINC → contact 1-cell sharp (Air-Water 계면)
- **u, p**: TENO 만 (contact 연속) → smooth low-diss
- 예상 trans: **1.995-2.00** (exact 1.999 거의 완벽)

### 우선순위

1. **우선**: 현재 Chamarthi 기반 (u/p WENO5, ρ TVD) → 9/9 PASS 유지
2. **Round 23 후보**: TENO5 구현 (WENO5 → TENO5 교체, Case 06/07/08 amplitude 개선)
3. **Round 24 후보**: TENO-THINC (Case 10-1 sharp interface + low-diss 동시)

## 참고 — δ function 자동 감지의 우수성

Chamarthi 2025 (논문 46) 는 별도 contact detector 개발이 복잡하다고 지적.
Takagi 2023 TENO-THINC 는 **TENO 자체의 δ 를 재사용** → 별도 sensor 없음.
이것이 구현 관점에서 가장 간결하고 우리 solver 에 이식하기 쉬운 접근.
