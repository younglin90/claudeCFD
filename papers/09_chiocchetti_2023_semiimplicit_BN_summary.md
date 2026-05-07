# An Exactly Curl-Free Staggered Semi-Implicit Finite Volume Scheme for a First Order Hyperbolic Model of Viscous Two-Phase Flows with Surface Tension

> **출처:** S. Chiocchetti, M. Dumbser, *Journal of Scientific Computing* 94(1) (2023). DOI: 10.1007/s10915-022-02077-2
> **관련 실패:** 5-equation fully coupled에서 acoustic CFL 제한. Semi-implicit staggered FV로 음향파만 implicit 처리.

---

## 1. 핵심 수식

### 지배방정식

**1차 쌍곡형 BN/GPR 모델** (Baer-Nunziato + Godunov-Peshkov-Romenski):
- 2상 유동에 점도(viscosity) + 표면장력(surface tension) 포함
- 모든 파동 전파 메커니즘이 1차 쌍곡형으로 통합

### Semi-Implicit 시간 차분 구조

**Acoustic (implicit):**
- 압력파 전파: $\partial p / \partial t$, $\nabla \cdot u$ 관련 항
- 대칭 양정치(SPD) 선형 시스템 → **켤레구배법(CG)** 으로 효율적 풀이

**Non-acoustic (explicit):**
- 대류 속도, 체적분율 이송, 점성/표면장력 비음향 기여

### Pressure System

- 단일 스칼라장인 압력 $p$에 대한 Poisson-like 방정식
- SPD 구조 → CG 수렴 보장

---

## 2. 방법론

### Staggered Mesh 구조

- **스칼라 변수** (α, ρ, p, E): 주격자(primal) 셀 중심
- **벡터 변수** (u): 이중격자(dual) 셀 중심 = 주격자 면
- **정확한 curl-free 보존:** 이산 Schwarz 항등식 만족 → 이산 curl 연산자가 정확히 0

### Implicit/Explicit Splitting

| 처리 | 항 |
|------|-----|
| **Implicit** | 압력파 (acoustic), $\nabla p$, $\nabla \cdot u$ |
| **Explicit** | 대류, 체적분율, 점도, 표면장력 비음향 기여 |

### CFL 제한 완화

- Explicit CFL: $\Delta t \propto \Delta x / (|u| + c)$ → 음속 $c$에 의해 지배
- **Semi-implicit:** 음속 제거 → $\Delta t \propto \Delta x / |u|$ (유속 기준만)
- 저 Mach에서 효율 극대화

### 강성 대수 소스항

- **반해석적 시간 적분(semi-analytical):** 점성 이완 소스항을 해석적으로 처리
- 점성-소성 고체의 변형 이완 극한까지 안정적 계산

---

## 3. 검증 및 시뮬레이션 설정

### 검증 방향

- 2상 유동(two-phase flow) 시뮬레이션
- 점도 효과 포함
- 표면장력 효과 포함
- 저 Mach 수 영역 검증

> **참고:** PDF 변환 제한으로 구체적인 수치 결과는 원본 논문 참조 필요.

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/denner_1d/solver_5eq.py` — 시간 적분 구조
- **수정 방향:** Staggered semi-implicit 구조 도입 시:
  1. Pressure Poisson 방정식을 SPD로 구성 → CG solver 사용
  2. 비음향 항은 explicit → 기존 upwind/CICSAM 재사용
  3. Curl-free 조건은 1D에서 자동 만족
- **주의사항:**
  1. 1D에서는 staggered grid의 이점이 제한적 (체커보드 문제가 MWI로 해결 가능)
  2. 표면장력/점도 모델은 현재 claudeCFD 범위 밖
  3. Re & Abgrall (2022)과 유사한 acoustic-implicit splitting이나, GPR/SHTC 모델 기반으로 더 일반적
