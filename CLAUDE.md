# CLAUDE.md — 프로젝트 작업 기록

## 프로젝트 개요

압축성 다상유동 Finite Volume Method (FVM) CFD 검증 해석 툴.

**핵심 목표**
- pressure, velocity numerical oscillation 또는 checkerboard 현상, 그리고 Abgrall 문제(인터페이스 가짜 압력 진동) 들이 발생하지 않는 강건한 수치 기법 개발
- 특정 EOS에 종속되지 않는 General EOS 프레임워크 적용
- 각 화학종(Y_i)에 독립적인 EOS 적용 가능, 혼합물 열역학 물성치는 혼합물 이론으로 산출

**언어**: Python 전용 (NumPy/SciPy 허용, C extension 금지)

---

## Primitive & Conservative Variables

### Primitive variables
| 차원 | 변수 벡터 |
|------|-----------|
| 1D | `{p, u, T, Y_1, Y_2, ...}` |
| 2D | `{p, u, v, T, Y_1, Y_2, ...}` |
| 3D | `{p, u, v, w, T, Y_1, Y_2, ...}` |

### Conservative variables (1D 기준)
```
U = [ρ, ρu, ρE, ρY_1, ρY_2, ...]
```

### 혼합 물성치
```
ρ   = Σ ρ_i = Σ (Y_i * ρ)
1/ρ = Σ (Y_i / ρ_i)          # 체적분율 혼합 규칙
e   = Σ (Y_i * e_i(T, p))    # 비내부에너지
```

---

## 지배방정식 (1D Euler, Conservative Form)
```
∂U/∂t + ∂F/∂x = 0

U = [ρ, ρu, ρE, ρY_k]^T
F = [ρu, ρu²+p, (ρE+p)u, ρY_k·u]^T

E = e + u²/2
```

다성분: k = 1, ..., N-1  (마지막 종은 Y_N = 1 - ΣY_k 로 결정)

---

## 물리 배경

### Abgrall 문제

보존형 다성분 유동에서 비선형 실기체 EOS 사용 시 인터페이스에서 가짜 압력 진동 발생.

**근본 원인**
- 보존변수 `ρE` → `e` → `T` → `p` 역산 시 EOS 비일관성
- 균일 압력 초기조건이 이송 후 압력 진동으로 오염
- 보존변수의 선형 합산과 비선형 EOS 사이의 불일치

**판별 기준**: 균일 압력·속도 초기조건에서 이송 후 압력이 기계적 정밀도 이상으로 변동하면 실패

---

### EOS 종류

#### 1. Ideal Gas EOS
```
p = ρ R_s T = ρ (γ-1) e
e = c_v T
c_v = R_s / (γ-1)
```

#### 2. NASG (Noble-Abel Stiffened Gas) EOS
```
p = (γ-1) ρ (e - q) / (1 - b ρ) - p_∞
e = c_v T + q + p_∞/ρ    (근사)
c_v, γ, p_∞, b, q : 물질 상수
```

#### 3. SRK (Soave-Redlich-Kwong) EOS
```
p = R_u T / (v - b)  -  a·α(T) / [v(v + b)]

α(T)  = [1 + m(1 - √(T/T_c))]²
m     = 0.48 + 1.574ω - 0.176ω²
a     = 0.42748 R_u² T_c² / p_c
b     = 0.08664 R_u T_c / p_c
```
적용 대상: CH₄/N₂ 혼합물  
역산(T→p): `scipy.optimize.brentq` 사용

---

## 적용 수치 기법

### 시간 차분

| 조건 | 기법 | 비고 |
|------|------|------|
| 초음속 (Ma > 1) | Forward Euler (명시) | CFL ≤ 0.8 |
| 아음속 (Ma < 1) | Backward Euler (암시) | Newton 반복, 수렴 판정 `‖ΔU‖/‖U‖ < 1e-8` |

### Flux 기법: APEC (Approximate Pressure-Equilibrium-preserving with Conservation)
docs/APEC_flux.md 참고
---

### Jacobian 계산 (Backward Euler용)
- **Phase 1 (현재)**: 수치 미분 (Finite Difference)
```python
  dF/dU ≈ [F(U + ε·e_j) - F(U)] / ε,  ε = 1e-7 * |U_j|
```
- **Phase 2 (검증 완료 후)**: 이론적 해석 Jacobian으로 교체

### High-order 기법
MUSCL 등 고차 기법은 **현재 미적용** (검증 단계 완료 후 추가 예정)

---

## 구현 파일 구조
```
claudeCFD/
├── CLAUDE.md
├── solver/
│   ├── solve.py         # 메인 솔버 (EOS + APEC flux + time integration)
│   ├── eos/
│   │   ├── ideal.py     # Ideal gas EOS
│   │   ├── nasg.py      # NASG EOS
│   │   └── srk.py       # SRK EOS
│   ├── flux.py          # APEC flux 계산
│   ├── jacobian.py      # 수치/해석 Jacobian
│   └── utils.py         # 보존↔원시변수 변환, 혼합 물성치
├── validation/
│   ├── 1D               # 1D 검증 문제 모음
│   ├── 2D               # 2D 검증 문제 모음
│   ├── 3D               # 3D 검증 문제 모음
└── results/
    └── figures/         # 검증 결과 플롯
```

---

## 작업 플로우
```
코드 수정 → pytest validation/ → 전체 통과 시 커밋 → 반복
```

검증 전체 통과 후 → 해석 Jacobian 교체 → 재검증

---

## 주의
백업 폴더는 건드리지 않고, 읽지도 않는다.
(백업_* 폴더)

---

## GitHub
```
https://github.com/younglin90/claudeCFD.git  (main 브랜치)
```