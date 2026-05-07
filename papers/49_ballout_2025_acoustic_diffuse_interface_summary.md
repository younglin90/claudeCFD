# Acoustic Propagation/Refraction Through Diffuse Interface Models

> **출처:** Ballout, Marino, Ntoukas, Rubio, Ferrer, *arXiv:2504.01727v2* (2025), DOI: 10.1016/j.jcp.2025.114478 (JCP accepted)
> **관련 실패:** Case 10-1 Air-Water acoustic transmission — **정확한 phase-wise sound speed interpolation**

---

## 1. 핵심 수식

### Weak compressibility formulation (Eq. 5)

$$
\partial_t p + \rho c_s^2 \nabla \cdot \boldsymbol{u} = 0
$$

- `c_s` 는 phase-wise speed of sound (air vs water)
- isentropic acoustic wave, 저마하 (c_s >> u_advect)

### Diffuse interface linear interpolation (Eq. 6)

$$
(.)= (.)_1 \cdot c + (.)_2 \cdot (1-c)
$$

- `c ∈ [0, 1]`: concentration (Cahn-Hilliard phase field)
- 밀도 `ρ`, 점성 `η`, 음속 `c_s` 모두 선형 보간
- **→ Wood c 와 다름!** Ballout 는 선형 평균 사용 (Volume-averaged)

### Non-conservative flux for c_s (Eq. 17-20)

DGSEM 에서 `ρ·c_s²` 를 non-conservative 계수로 처리:

$$
\hat{F}_{p} = \frac{1}{2} \rho c_s^2 \left(u_n^L + u_n^R\right) + \frac{1}{2} \lambda (p^R - p^L)
$$

`λ = c_s` (entropy-stable 선택), 음속 따른 upwinding

---

## 2. 방법론

### 알고리즘 개요

1. **Navier-Stokes/Cahn-Hilliard (iNS/CH)** 결합 시스템:
   - c 전송: Cahn-Hilliard 확산
   - u 전송: iNS (momentum)
   - p 전송: weak compressibility (acoustic wave)
2. **DGSEM** 고차 (p=2~8) spectral element
3. **Entropy-stable** numerical flux

### 핵심 아이디어

- **기존 iNS/CH**: p 를 incompressibility 강제용 → non-physical pressure
- **Modified**: p 방정식 `∂_t p + ρc_s²∇·u = 0` → 실제 acoustic wave propagation
- **Phase-wise c_s**: air c_s=347, water c_s=1480 → 선형 보간 (Wood 과 다름)

### 기존 방법 대비 차이점

| 항목 | 완전 compressible (우리) | Ballout (weak compress) |
|------|------------------------|-------------------------|
| 모델 | 5-eq Kapila + EOS | iNS/CH + weak comp |
| Mach 범위 | All-Mach (0~100) | 저마하 only (c>>u) |
| p equation | p = EOS(ρ, e) | p wave-like ∂_t p + ρc²∇u=0 |
| 계면 | VOF/THINC 샤프 | Cahn-Hilliard 확산 |
| Acoustic transmission | via Riemann solver | via c_s²∇·u 직접 |

### Snell's law 재현

2D air-water 계면: transmitted angle θ_t 측정
- 수치 θ_t vs exact (Snell) <1° 오차 (spectral order 8)
- **critical angle 13° 까지 정확**

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | 도메인 | order | 결과 |
|---|--------|--------|-------|------|
| 4.1 | 1D transmission/reflection | [0, 10] air-water | p=2-8 | **Spectral convergence** 재현율 exp order |
| 4.2 | 2D Snell's law | 2D wedge | p=4-8 | θ_t error < 1° (13° critical) |
| 4.3 | 3D spherical wave → flat interface | 3D | p=4 | Qualitative air-water transmission |

### 주요 결과

- 1D: exponential convergence in transmitted amplitude as DG order increases
- 2D: Snell's law 만족, critical angle 재현
- Diffuse interface width 가 줄어들수록 정확도 증가 (물리적 sharp limit)

---

## 4. claudeCFD 적용 메모

### 직접 이식 불가능
- Ballout 는 **low-Mach incompressible** approach (c >> u 가정)
- 우리 5-eq Kapila 는 **fully compressible all-Mach** (shock 포함)
- DGSEM spectral 과 우리 FVM 아키텍처 다름

### 간접 insight
1. **c_s 선형 보간 vs Wood 공식**: Ballout 는 선형 `c·c_1 + (1-c)·c_2` 사용 → DG spectral 에서 잘 작동. 우리 Wood 기반 cell c_mix 는 cell-center 사용하지만 face reconstruction 에서 TVD/WENO 로 적절 처리 중.
2. **Weak compressibility 용 pressure wave eq**: 우리 IM1 block-tridiag 에 이미 비슷한 구조 (`∂_t p + ρc²∇u=0` implicit). Peluchon 2017 구조와 유사.
3. **Diffuse interface width 영향**: 5-eq 에서 α transition region 이 `ε=1e-6` 으로 거의 sharp. Ballout 처럼 `ε` 가 transmission error 의 dominant 원인 주장 — 우리에게는 덜 해당.

### Case 10-1 특화 insight
- Ballout 의 "transmitted wave spectral convergence" 는 DG order 덕분
- 우리 2nd-order TVD / 5th-order WENO5 (Round 22) 는 이미 차수 증대 효과 확인됨 (trans 0.91→2.019)
- 추가 개선은 DG/spectral 이 아닌 **characteristic-based recon** (Chamarthi 방향) 이 현실적

### 결론
Ballout 2025 는 다른 architecture (DG-iNS/CH) 라 직접 이식 대상 아님. 하지만 **"phase-wise c_s + weak compressibility"** 개념은 우리 IM1 block-tridiag 의 `a_imp = ρ·c` 와 본질적으로 동일한 원리 — 이미 구현되어 있음. 추가 개선 방향은 **Chamarthi 2025 variable-specific reconstruction** 이 더 직접적.
