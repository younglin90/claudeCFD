# A low-dissipation finite-volume method based on a new TENO shock-capturing scheme

> **출처:** Fu, L., *Computer Physics Communications* **235** (2019) 25-39. DOI: 10.1016/j.cpc.2018.10.009
> **관련 실패:** Case 06/07/08 amplitude 감쇠 (WENO-JS smooth mode 과도 소산)

---

## 1. 핵심 수식

### TENO-FV framework (Fu et al. 2016, 2018)

$$
\hat{f}_{i+1/2} = F(f_{i-r}, \dots, f_i, \dots, f_{i+s})
$$

**WENO vs TENO 핵심 차이**:

$$
\text{WENO}: \quad w_k = \frac{\alpha_k}{\sum_j \alpha_j}, \quad \alpha_k = d_k / (\varepsilon + \beta_k)^2
$$

$$
\text{TENO}: \quad w_k = \begin{cases} d_k & \text{if } \chi_k \geq C_T \\ 0 & \text{otherwise} \end{cases}
$$

- **WENO**: 모든 stencil 을 nonlinear weighted → smooth 영역 에서도 부분적 dissipation
- **TENO**: **Binary selection** (큰 stencil smooth 면 그것만, shock 감지 시 small 로 완전 전환)
- `χ_k` = scale separation indicator, `C_T` = cutoff (typical 1e-5)

### Optimal linear weight (smooth region)

Large stencil 5-point TENO5:
$$
w_{\text{large}} = d_{\text{large}} \quad \text{when smooth, zero dissipation}
$$

Small stencil candidate (3-point each), ENO-like selection when non-smooth.

### "Low-dissipation" Riemann flux (smooth) vs "Dissipative" (shock)

$$
\hat{f}^{\text{LD}}_{i+1/2} = \text{central scheme} \quad (\chi_{\text{both sides}} = 1)
$$

$$
\hat{f}^{\text{D}}_{i+1/2} = \text{HLLC/HLL} \quad (\text{any shock detected})
$$

---

## 2. 방법론

### 알고리즘 개요

1. K-point stencil (K=5,7,8) 분할: 3개 small 3-point + 1개 large stencil
2. 각 stencil smoothness indicator 계산
3. Large smooth 이면 → large 만 사용 (optimal linear, **zero nonlinear dissipation**)
4. Large non-smooth → small stencil 중 ENO-selection
5. Riemann flux 도 smoothness 따라 LD/D 선택

### 핵심 아이디어: 이중 selection

- **Reconstruction stage**: TENO 가 stencil 결정 (smooth → linear, shock → ENO)
- **Flux stage**: Smoothness indicator 로 LD/D Riemann 선택

**→ 수치 dissipation 2단계 모두 최소화**

### WENO 대비 우위

| 항목 | WENO-JS (Jiang-Shu 1996) | TENO (Fu 2016-2019) |
|------|--------------------------|----------------------|
| Smooth 영역 weighting | Weighted avg (ε 기반 partial) | **Binary** (linear optimal) |
| 저마하 smooth acoustic | 부분 dissipation | **Zero** dissipation |
| Shock robustness | OK | OK (small stencil ENO) |
| Critical point accuracy | 5th order loss | **5th order 유지** |
| 구현 복잡도 | Medium | Medium-High |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | WENO-JS | TENO5 |
|---|--------|---------|--------|
| 6.1 | Sine advection (linear) | 5th order (critical point 3) | **Full 5th** |
| 6.2 | Sod shock tube | OK | OK + sharp |
| 6.3 | Shu-Osher (high-freq) | Smearing | **Sharp** |
| 6.4 | Double Mach Reflection | OK | Low diss |
| 6.5 | Rayleigh-Taylor | Dissipative | Low diss 섬세한 구조 |
| 6.6 | 2D Riemann | OK | 향상된 해상도 |

### 주요 결과

- Shu-Osher: TENO 가 small-scale 구조 해상도 **2x** 우수
- Linear advection: TENO 는 critical point 에서 full 5th-order (WENO-JS 는 3rd 로 저하)
- Shock 포착: WENO 와 동등 (non-oscillatory)

---

## 4. claudeCFD 적용 메모

### 현재 구현
- WENO5-JS (Jiang-Shu 1996) — Round 22 추가됨
- Case 06 u/p 에 WENO5 적용 시 dissipation 부분 감소 but **여전히 smooth 영역 partial dissipation** 존재

### TENO 이식 시 예상 효과

**저마하 smooth acoustic (Case 06, 07, 08)**:
- WENO5-JS 의 nonlinear weight 는 smooth 영역에서도 `w_k ≠ d_k` → partial dissipation
- TENO5 는 smooth 감지 시 `w_k = d_k` → **zero nonlinear diss**
- 예상 개선:
  - Case 06 amplitude -(5-10)% 감소
  - Case 07 -5% → -2% ~ -3%
  - Case 08 -4.4% → -2% ~ -3%

### 구현 난이도
- TENO5 stencil selection 로직: medium (smoothness cutoff C_T tuning 필요)
- Dual-level Riemann (LD/D) 선택: HIGH (현 HLLC/SLAU2 dispatch 와 별개 layer 추가)

### 권장 Round 23 시도
1. **TENO5 reconstruction** 만 단독 구현 (Riemann 교체는 다음 단계)
2. 동일 `primitive_recon='teno5'` 옵션
3. 10-1 유지 확인 + 06/07/08 개선 여부 측정

---

## 참고
Takagi 2023 (TENO-THINC) 는 이 Fu 2019 TENO 위에 THINC 결합 — Case 10-1 stiff interface 에 추가 이득 가능 (별도 요약 참조).
