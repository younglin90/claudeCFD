# CLAUDE.md — 프로젝트 작업 기록

## 프로젝트 개요

압축성 2상 유동 1D CFD 솔버.  
**Demou, Scapin, Pelanti, Brandt (JCP 448, 2022)** 의 4-equation pressure-based model을 1D로 구현·검증하는 것이 현재 목표.  
전체 스킴 명세: `docs/DENNER_SCHEME.md` 참조.

**핵심 목표**
- 4-equation model (α₁, T, u, p): pressure-based, diffuse interface
- Mode A (현재): Explicit RK3 + Fractional-Step (Helmholtz 압력 implicit)
- Mode B (차후): Fully-Coupled Implicit BDF2 (음향 CFL 제거)
- EOS: NASG (Noble-Abel Stiffened Gas), 각 상별 독립 적용

**언어**: Python 전용 (NumPy/SciPy 허용, C extension 금지)

---

## 솔버 구분

| 솔버 | 경로 | 기법 | 상태 |
|------|------|------|------|
| **Demou 2022** (★ 주 개발 대상) | `solver/demou2022_1d/` | RK3+Helmholtz, (α₁,T,u,p) | **개발 중** |
| Denner 2018 (구 버전) | `solver/denner2018_1d.py` | PISO+ACID, (p,u,T,ψ) | 참고용 유지 |
| APEC (보존형) | `solver/solve.py` | APEC flux, 보존변수 | 참고용 유지 |

---

## 주요 수식 요약

> 상세 수식은 `docs/DENNER_SCHEME.md` 참조.

### 원시변수

| 변수 | 의미 |
|------|------|
| α₁ | Phase 1 체적분율 |
| T | 혼합 온도 |
| u | 혼합 속도 |
| p | 압력 |

### NASG EOS (핵심 수식)

```
ρₖ(p,T) = (p + p∞ₖ) / [κᵥₖ T (γₖ-1) + bₖ(p + p∞ₖ)]
cₖ²     = γₖ(p + p∞ₖ) / [ρₖ(1 - bₖρₖ)]
Γₖ      = (γₖ-1) / (1 - bₖρₖ)
hₖ      = κₚₖ T + bₖ p + ηₖ        (κₚₖ = γₖ κᵥₖ)
```

이상기체 극한: p∞=0, b=0, η=0 → p = ρ(γ-1)κᵥ T

**표준 물성치**

| 물질 | γ | p∞ [Pa] | b [m³/kg] | κᵥ [J/kg/K] | η [J/kg] |
|------|---|---------|-----------|-------------|----------|
| Air  | 1.4 | 0 | 0 | 717.5 | 0 |
| Water(liquid) | 1.187 | 7.028×10⁸ | 6.61×10⁻⁴ | 3610 | −1.177788×10⁶ |
| Water(vapor)  | 1.467 | 0 | 0 | 955 | 2.077616×10⁶ |

### Mode A 알고리즘 (RK3 부분단계, 서브스텝 m마다)

```
1. 상태 갱신:  ρₖ, cₖ, Γₖ, hₖ, φₖ, ζₖ, c_mix, S^(2), S^(3)
2. α₁ 이송:   CICSAM + S^(3) 발산 보정
3. T  이송:   van Leer + S^(3) 발산 보정
4. u* 예측:   van Leer 대류, 압력 기울기 제외
5. Helmholtz:  삼중대각 → Thomas algorithm → p^{m+1}
6. u 보정:     u^{m+1} = u* - γ̃ₘΔt/ρ · ∇p^{m+1}
7. 상태 재계산 (다음 서브스텝 준비)
```

---

## 파일 구조

```
claudeCFD/
├── CLAUDE.md
├── docs/
│   ├── DENNER_SCHEME.md      # ★ 스킴 전체 명세 (수식, 알고리즘)
│   └── APEC_flux.md          # APEC flux 참고
├── solver/
│   ├── demou2022_1d/         # ★ 주 솔버 패키지
│   │   ├── __init__.py
│   │   ├── config.py         # EOS 파라미터, 격자, 모드 설정
│   │   ├── eos.py            # NASG EOS
│   │   ├── source_terms.py   # S^(2), S^(3), c_mix
│   │   ├── grid.py           # 1D 격자, ghost cells
│   │   ├── cicsam.py         # CICSAM (1D Hyper-C)
│   │   ├── flux_limiter.py   # van Leer
│   │   ├── spatial.py        # gradient, divergence, Laplacian
│   │   ├── helmholtz.py      # Thomas algorithm
│   │   ├── rk3.py            # RK3 드라이버 (Mode A)
│   │   ├── boundary.py       # BC (periodic, transmissive, wall)
│   │   ├── timestepping.py   # CFL 기반 dt
│   │   ├── io_utils.py       # 결과 저장
│   │   └── run.py            # 진입점
│   ├── denner2018_1d.py      # 구 솔버 (참고용)
│   ├── solve.py              # APEC 솔버 (참고용)
│   └── eos/                  # APEC용 EOS 클래스
├── validation/
│   └── 1D/                   # 검증 케이스 명세 (*.md)
└── results/                  # 검증 결과 출력
```

---

## 검증 계획 (Denner 2018 솔버)

### Phase 1 (현재): 1D Smooth Interface Advection — Water/Air

**솔버**: `solver/demou2022_1d/`  
**스킴**: Mode A (RK3 + Helmholtz)

#### 케이스 설명

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m, periodic BC (좌우 모두) |
| 격자 수 | **N = 10** (초기 검증용, 빠른 확인 목적) |
| dx | 0.1 m |
| Water(NASG, phase 1) 영역 | x ∈ [0.4, 0.6] |
| Air(Ideal Gas, phase 2) 영역 | x ∉ [0.4, 0.6] |
| 초기 속도 | u = 1.0 m/s (전 도메인 균일) |
| 초기 압력 | p₀ = 1×10⁵ Pa (전 도메인 균일) |
| 초기 온도 | T₀ = 300 K (전 도메인 균일) |
| CFL | 0.5 (물의 음속 기준 자동 결정) |
| 종료 시간 | t_end = 1.0 s (1 flow-through 주기) |

#### 초기 α₁ 프로파일 (smooth tanh)

```
α₁(x) = 0.5 * [tanh((x - 0.4)/δ) - tanh((x - 0.6)/δ)]

δ = 0.5 * dx  (인터페이스 두께)
```

α₁ = 1 → pure water, α₁ = 0 → pure air

#### EOS 파라미터

| 상 | γ | p∞ [Pa] | b [m³/kg] | κᵥ [J/kg/K] | η [J/kg] |
|----|---|---------|-----------|-------------|----------|
| Water (phase 1) | 1.187 | 7.028×10⁸ | 6.61×10⁻⁴ | 3610 | −1.177788×10⁶ |
| Air (phase 2) | 1.4 | 0 | 0 | 717.5 | 0 |

#### 이론해 (Exact Solution)

균일 속도 u₀=1 m/s로 이송 → t=1.0 s 후 α₁ 프로파일이 초기 상태로 복원(주기 BC).

| 물리량 | 이론값 |
|--------|--------|
| p | p₀ = 1×10⁵ Pa (전 도메인 균일 유지) |
| u | u₀ = 1.0 m/s (전 도메인 균일 유지) |
| T | T₀ = 300 K (전 도메인 균일 유지) |
| α₁(t=1.0) | α₁(t=0) 복원 (1 주기 후 원위치) |

#### PASS 기준

| 검증 항목 | 기준 |
|-----------|------|
| 수치 발산 없이 t=1.0 s 완료 | 필수 |
| 압력 편차 max\|p-p₀\|/p₀ | < 1×10⁻³ |
| 속도 편차 max\|u-u₀\| | < 1×10⁻³ m/s |
| α₁ 범위 | 0 ≤ α₁ ≤ 1 유지 |

> **N=10은 수치 정확도보다 알고리즘 안정성 확인이 목적.**  
> PASS 후 격자를 늘려 정확도·수렴성 검증 진행.

---

### Phase 2 이후: 미정

Phase 1 통과 후 다음 검증 케이스를 별도로 결정.

---

## 작업 플로우

```
코드 수정 → 검증 케이스 실행 → PASS 확인 → 커밋 → 반복
```

---

## 주의사항

- 백업 폴더(`백업_*`)는 읽지도, 수정하지도 않는다.
- `denner2018_1d.py` 내 EOS는 NASG 전용 인라인 함수(`_rho`, `_a2`, `_h`)로 구현되어 있으며 `solver/eos/` 의 클래스와 별도임.
- CFL은 반드시 `_cfl_dt()` 함수를 통해 계산 (물의 음속 기준 자동 결정).
- `dt_fixed` 파라미터 사용 시 물의 음속 CFL 조건을 수동으로 만족시켜야 함.

---

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main 브랜치)
```