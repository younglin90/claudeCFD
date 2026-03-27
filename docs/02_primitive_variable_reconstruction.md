# Group 2: 온도 T 기반 원시변수 재구성 (IEC 만족 스킴)
# (Primitive Variable Reconstruction with Temperature — IEC-Satisfying)

## 핵심 아이디어

ENO/WENO/TENO 재구성 시 밀도 `ρ`를 기반으로 하면 문제가 된다:
- 계면에서 `ρ`는 불연속이므로, `ρ`를 재구성하면 계면 양쪽에서 다른 스텐실이 선택됨
- 이 스텐실 불일치가 계면 인터페이스 플럭스에 비물리적 비대칭을 만들어 압력 진동 유발

**해결책**: 재구성 기저 변수를 `W = [T, Y, u, v, w, P]^T`로 바꾼다.
- `T`, `P`, `u`는 계면에서 모두 **연속** → 재구성 스텐실이 양쪽에서 동일하게 선택됨
- Interface Equilibrium Condition (IEC) 자동 만족 → **기계 정밀도 수준의 압력 진동 억제**

---

## Paper A: Collis, Bezgin, Mirjalili, Mani (2026) — 4방정식 모델

**파일**: `[적용해볼것] 4eq.pdf` (동일 내용: `ssrn-5372486.pdf`)
**제목**: "A robust four-equation model for compressible multi-phase multi-component flows satisfying interface equilibrium and phase-immiscibility conditions"
**저널**: JCP, 2026 (DOI: 10.1016/j.jcp.2026.114827)

### 방법: IEC-satisfying ENO + HLLC

재구성 변수를 `W = [T, Y^c_p, u, v, w, P]^T`로 선택하고, 특성 공간 투영(characteristic-space projection)을 통해 ENO 보간:

```
알고리즘:
1. 셀 중심에서 W = [T, Y, u, v, w, P] 계산
2. 계면 평균 상태에서 좌 고유벡터 행렬 S^{-1} 계산
3. 특성 공간 투영: W̃ = S^{-1} · W
4. W̃에 ENO/WENO/TENO 보간 → W̃^L, W̃^R
5. 역투영: W^L = S · W̃^L, W^R = S · W̃^R
6. EOS로 밀도 복원: ρ = ρ(T, Y, P)
7. 보존 변수 재구성: U^L, U^R ← 원시변수에서 변환
8. HLLC Riemann solver로 인터페이스 플럭스 계산
```

### IEC가 만족되는 이유

`P, T, u`가 공간 균일할 때, 모든 셀에서 `W̃`가 일정 → ENO 보간 결과 `W̃^L = W̃^R = W̃` → HLLC가 trivial flux 반환 → `∂P/∂t = 0` ✅

반면, `ρ`를 재구성 변수로 쓰면 계면에서 `ρ`가 불연속이므로 `W̃^L ≠ W̃^R` → 허위 플럭스 발생 ❌

### 4방정식 시스템 (Phase-Immiscible)

```
∂(αρ_k)/∂t + ∇·(αρ_k u) = 0         (각 상 연속 방정식)
∂(ρu)/∂t   + ∇·(ρu⊗u + pI) = 점성+표면장력
∂(ρE)/∂t   + ∇·((ρE+p)u) = 점성 일
```

인터페이스 정규화 (CDI — Conservative Diffuse Interface):
```
psi_p = ε · ln((ϕ_p + δ) / (1 - ϕ_p + δ))  [부호있는 거리 변환]
```
CDI 정규화 플럭스도 IEC 만족하도록 2차 KEEP 중심 스킴으로 이산화 (Section 3.2.1에서 해석적으로 증명).

### EOS

**NASG (Noble-Abel Stiffened Gas)**: 이상기체 + stiffened gas를 통합, 액체-기체 모두 적용 가능.

닫힌 형태 압력 역산 (Amagat 혼합 규칙 사용):
```
P = (a2 + sqrt(a2² + 4·a1·a3)) / (2·a1)
```

### 장단점

| | |
|---|---|
| ✅ | IEC 기계 정밀도 만족 — 가장 우수한 압력 진동 억제 |
| ✅ | 완전 보존형 (에너지 보존 유지) |
| ✅ | 구현 단순: ENO 재구성 변수만 교체 (`ρ → T`) |
| ✅ | HLLC와 직접 호환 |
| ✅ | CDI 인터페이스 정규화 포함 (무한 smearing 방지) |
| ⚠ | **NASG EOS 사용** — cubic EOS (PR/SRK) 아님 |
| ⚠ | 전임계(transcritical) 테스트 케이스 없음 |
| ⚠ | Cubic EOS 적용 시 EOS 역산 루틴 수정 필요 |

---

## Paper B: Xu, Sun, Guo (2026) — CDHD 방법

**파일**: `1-s2.0-S0021999126000033-main.pdf`
**제목**: "A Central Differential flux with high-Order dissipation for robust simulations of transcritical flows"
**저널**: JCP 550 (2026) 114653

### 방법: CDHD (Central Differential flux with High-Order Dissipation)

총에너지 방정식을 완전히 버리고, **원시변수 준선형 시스템**만 사용:

```
∂V/∂t + B(V)·∂V/∂x = 0,    V = [ρ, u, p]^T
B = [u, ρ, 0; 0, u, 1/ρ; 0, ρc², u]
```

CDHD 플럭스 = 중심항 + 고차 소산항:
```
CDHD = central_v + diss_v

중심항: -B(V_i)·(V⁻_{i+1/2} - V⁺_{i-1/2})/Δx
소산항: DOTRS 방식의 path-conservative upwind fluctuation
        H±_{i+1/2} = (∫₀¹ B±(Ψ(s)) ds)·(V_R - V_L)
        [3점 Gauss-Legendre 적분; 매끄러운 영역에서 O(Δx⁴)]
```

충격파 감지 시 보존형 WENO-5+Roe 플럭스로 전환 (PVRS sensor):
```
충격파 감지: P*_{i+1/2} / P_i > 1 + ε  AND  s^L_{i+1/2} < 0
```

### 핵심: 압력을 에너지에서 역산하지 않음

원시변수 포뮬레이션이므로 `p = f(ρ, E)`를 계면에서 계산할 필요가 없음 → Abgrall 진동 원천 차단.

### EOS

**Peng-Robinson**: `p = R̂T/(v̂-b) - aα/(v̂(v̂+b)+b(v̂-b))`

전임계 질소(N₂) 시뮬레이션 검증 완료 (Widom line 횡단 포함).

### 장단점

| | |
|---|---|
| ✅ | **Peng-Robinson EOS + 전임계 흐름 직접 검증** |
| ✅ | 압력 진동 근본 제거 (에너지 역산 없음) |
| ✅ | 충격파에서는 보존형으로 자동 전환 |
| ❌ | **비보존형** — 에너지 보존 오류 O(Δx⁴) (매끄러운 영역), 충격파 서열 O(Δx²) 가능 |
| ❌ | 완전 보존을 우선순위 1로 요구하는 경우 부적합 |
| ⚠ | 하이브리드 충격파/비충격파 감지 로직 구현 필요 |

---

## 코드에 적용 시 비교

| 항목 | Collis 4eq (T 재구성) | Xu CDHD |
|------|----------------------|---------|
| 보존성 | ✅ 완전 보존 | ❌ 비보존 |
| 실제유체 EOS | △ NASG (cubic 확장 필요) | ✅ PR EOS |
| 전임계 검증 | ❌ | ✅ |
| 구현 난이도 | 낮음 (재구성 변수 교체) | 중간 (하이브리드 flux) |
| 기존 코드 수정 | 소 | 중 |

---

## 사용자 코드 적용 권장

- **완전 보존 + 기존 FVM 코드 최소 수정**: **Collis 4eq (T 재구성)** 추천
  - ENO/WENO 재구성 루프에서 `[ρ, u, p] → [T, Y, u, P]`로 변경
  - NASG EOS 역산 추가 (닫힌 형태 공식 존재)
  - Cubic EOS 필요 시 나중에 APEC (Group 3)으로 업그레이드

- **전임계 실제유체 + 일부 비보존 허용**: **Xu CDHD** 검토
  - 단, 완전 보존 요구사항 충돌 주의
