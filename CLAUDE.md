# CLAUDE.md — APEC 1D 프로젝트 작업 기록

## 프로젝트 개요

**논문**: Terashima, Ly, Ihme, "Approximately Pressure-Equilibrium-Preserving scheme", JCP 524, 2025
**목표**: 1D CH4/N2 초임계 계면 이송 문제에서 FC-NPE vs APEC vs PEqC 방법 비교 재현

---

## 물리 배경

### Abgrall 문제
보존형 다성분 유동에서 비선형 실기체 EOS를 사용할 때 인터페이스에서 가짜 압력 진동이 발생한다.
- 보존변수 `ρE`에서 `T` → `p` 역산 시 EOS 비일관성 발생
- 균일 압력 초기조건이 이송 후 압력 진동으로 오염됨

### SRK EOS
Soave-Redlich-Kwong 실기체 상태방정식 (CH4/N2 혼합물):
```
p = Ru*T/(v-b) - aα/(v*(v+b))
```
- 액체 상태 CH4: `ρe < 0` (음수) — 절대 양수 하한 적용 금지
- `ε_s = (∂ρe/∂ρ_s)_p`: 인터페이스에서 O(10^6) 크기, 부호 음수

---

## 구현 파일 구조

| 파일 | 설명 |
|------|------|
| `apec_1d.py` | 메인 시뮬레이션 (EOS + 수치기법 + 실행) |
| `debug_coarse.py` | 다해상도 PE 비교 (N=51~501) |
| `debug_eps.py` | ε_s 값 및 플럭스 분석 |
| `debug_apec_early.py` | 초기 20 스텝 PE 비교 |
| `test_divergence.py` | FC vs APEC 발산 시점 비교 (N=501) |
| `test_divergence2.py` | 장시간 발산 비교 (N=101, 20000 스텝) |
| `test_sharp_interface.py` | 인터페이스 날카로움(k) 별 APEC 성능 |

---

## 핵심 발견: MUSCL-일관 PE 소산

### FC-NPE의 근본 문제
MUSCL-LLF 방법에서 질량 플럭스와 에너지 플럭스의 소산 항이 **불일치**:

```
질량 소산: -0.5*λ*(r1R - r1L)      ← MUSCL 점프 (limiter-clipped, 작음)
에너지 소산: -0.5*λ*(ρE_{m+1} - ρE_m)  ← 셀 중심 점프 (큼)
```

인터페이스에서 에너지가 질량보다 과소산(over-dissipated) → 가짜 압력 진동.

### APEC의 수정
에너지 소산에 동일한 MUSCL 점프를 사용:

```python
drhoE_pep = (eps0_h*(r1R - r1L)
             + eps1_h*(r2R - r2L)
             + 0.5*u_h**2*((r1R + r2R) - (r1L + r2L))
             + rho_h*u_h*(uR - uL))
FE = FE_cen - 0.5*lam*drhoE_pep
```

추가로 centered flux의 `ρe_{m+1/2}` PE-consistent 보정 (논문 Eq. 40):
```python
corr = 0.5*(eps0p - eps0)*0.5*dr1 + 0.5*(eps1p - eps1)*0.5*dr2
rhoe_h = 0.5*(rhoe + rhoe_{m+1}) - corr
```

---

## 실패한 접근법 (시행착오)

1. **Centered-only APEC (FC 소산 유지)**: corr 항이 N=501에서 rhoe의 3.4×10⁻⁹ — 완전히 무시 가능, FC와 동일한 결과
2. **셀 중심 PE-consistent 소산** (`ε_s_h*(r1_{m+1}-r1_m)`): FC보다 불안정 (step 759에서 발산, k=15)
3. **Quasi-conservative** (`ε_s * F_s 합산`): `ε_s*ρ_s ≠ ρe` (SRK EOS에서 5배 오차) → 22배 더 나쁜 PE

---

## 수치 결과

### PE 개선율 (k=15, 초기 1스텝)

| N | FC PE | APEC PE | 개선율 |
|---|-------|---------|--------|
| 51 | 5.56e-02 | 9.12e-03 | **6×** |
| 101 | 1.43e-02 | 1.26e-03 | **11×** |
| 201 | 3.73e-03 | 1.64e-04 | **23×** |
| 501 | 6.04e-04 | 1.10e-05 | **55×** |

### 발산 시점 비교 (N=101, k=15, CFL=0.3)

| 방법 | 발산 시각 |
|------|-----------|
| PEqC | t = 4.85ms |
| FC-NPE | t = 6.48ms |
| **APEC** | **t = 45.97ms (7× 더 오래)** |

### 에너지 보존
FC, APEC 모두 총에너지 기계 정밀도(10⁻¹³) 수준으로 보존.

---

## 한계

- APEC도 장시간에서 결국 발산 (근사적 방법)
- 논문은 Split-form/KEEP 계열 방법 사용으로 추정 — LLF로 완전 재현 불가
- 인터페이스가 충분히 날카롭지 않으면(k 작음) centered correction은 무시 가능
- 개선율이 논문의 4× 주장보다 훨씬 큼(55×) — MUSCL 일관 접근이 더 공격적

---

## 주요 파라미터

```python
# 초기조건
r1_inf = 400.0  # CH4 밀도 [kg/m3]
r2_inf = 100.0  # N2 밀도 [kg/m3]
p_inf  = 5e6    # 압력 [Pa]
u      = 100.0  # 속도 [m/s]
k      = 15.0   # tanh 인터페이스 날카로움 (기본값)
xc, rc = 0.5, 0.25  # 블롭 중심, 반경

# 수치 설정
N   = 101~501   # 격자 수
CFL = 0.3       # Courant 수
# 시간적분: SSP-RK3
# 공간: MUSCL-LLF (minmod 제한자)
```

---

## 실행 방법

```bash
# 메인 시뮬레이션 및 플롯 생성
py apec_1d.py

# 다해상도 비교
py debug_coarse.py

# 발산 시점 비교 (N=101, 장시간)
py test_divergence2.py

# 인터페이스 날카로움별 테스트
py test_sharp_interface.py
```

---

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main 브랜치)
```
