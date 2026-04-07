# Group 3: PEP 플럭스 호환 조건 (압력 평형 보존 스킴)
# (Pressure-Equilibrium-Preserving via Flux Compatibility Condition)

## 핵심 아이디어

보존형 시스템에서 내부에너지 플럭스와 종 질량 플럭스 사이에 **EOS로부터 유도되는 호환 조건(compatibility condition)** 을 만족하는 반점값(half-point values)을 구성하면, 압력이 균일할 때 `∂p/∂t = 0`이 대수적으로 보장된다.

**완전 보존 + 비보존 방정식 없이 Abgrall 조건 만족**이 목표.

---

## Paper A: Fujiwara, Tamaki, Kawai (Tohoku Univ.) — 완전 보존 PEP

**파일**: `[적용해볼것] PEP.pdf`
**제목**: "Fully conservative and pressure-equilibrium preserving scheme for compressible multi-component flows"

### 방법

**PEP 호환 조건** (핵심 해석 결과):

균일한 `p = p₀`, `u = u₀` 상태에서 `∂p/∂t = 0`이 만족되려면 내부에너지 반점값이 반드시:

```
ρe|_{m+1/2} - ρe|_{m-1/2} = Σ_i β_i|_m · (ρY_i|_{m+1/2} - ρY_i|_{m-1/2})

여기서: β_i = (∂ρe/∂ρY_i)_{p, ρY_{j≠i}}  ← EOS로부터 계산
```

를 만족해야 한다. 이 조건을 만족하는 **반점값 공식** (N성분 이상기체):

```
ρY_i|_{m+1/2} = (φ⁻·ρY_i|_m + φ⁺·ρY_i|_{m+1}) / 2

여기서:
  φ⁺ = (M|_{m+1}/ρ|_{m+1}) × (ρ|_m/M|_m)   [몰질량 비 기반 가중치]
  φ⁻ = 1/φ⁺

G|_{m+1/2} = (G|_m + G|_{m+1}) / 2
  여기서 G = 1/(γ_mix - 1)  [혼합 내부에너지 함수]

ρe|_{m+1/2} = G|_{m+1/2} · p₀ + Σ_i β_i · ρY_i|_{m+1/2}
```

### 핵심 특징

- **완전 보존형** — 추가 수송 방정식 없음 (비보존 항 없음)
- Riemann solver 불필요 — split-form 중심 차분법 (central difference)
- N성분 이상기체에 일반 적용 가능
- 2차 또는 고차 중심차분법과 호환
- Appendix B에 임의 EOS 확장 가능성 언급 (단, 테스트 케이스는 이상기체만)

### 장단점

| | |
|---|---|
| ✅ | **완전 보존 + 기계 정밀도 PEP** (이상기체) |
| ✅ | 추가 방정식 없음 |
| ✅ | 기존 split-form 코드에 반점값 공식만 교체하면 됨 |
| ✅ | 원리가 명확하고 구현 간단 |
| ❌ | **이상기체 전용** — cubic EOS(PR/SRK) 직접 적용 불가 |
| ❌ | 전임계/실제유체 테스트 없음 |

---

## Paper B: Terashima, Ly, Ihme (Hokkaido/Stanford) — APEC (실제유체 확장)

**파일**: `[적용해볼것] PEP2.pdf`
**제목**: "Approximately pressure-equilibrium-preserving scheme for fully conservative simulations of compressible multi-species and real-fluid interfacial flows"
**저널**: JCP 524 (2025) 113701

### 방법: APEC (Approximately Pressure-Equilibrium-Preserving scheme for Energetics Consistency)

Fujiwara의 정확한 PEP 조건은 이상기체에서만 성립한다 (몰질량 비 공식이 cubic EOS에서는 정확히 성립 안 함). Terashima et al.은 SRK cubic EOS에서 **O(Δx²) 오류를 1/12 수준으로 감소시키는 근사 PEP** 반점값을 유도:

**APEC 수정 반점값**:

```
ρY_i|_{m+1/2} = [ρY_i|_m + ρY_i|_{m+1}]/2
              + 보정항(∂ε_i/∂x 포함, 2차 오류 감소)

ρe|_{m+1/2}  = [ρe|_m + ρe|_{m+1}]/2
              + Σ_i ε_i × (ρY_i 반점값 보정량)

여기서 ε_i = (∂ρe/∂ρY_i)_{ρ_{j≠i}, p}  ← SRK EOS로부터 해석적 계산
```

**PEP 오류 분석**:
```
표준 스킴:  f_PE 오류 ∝ (1/3)·Δx²  (선두항)
APEC:       f_PE 오류 ∝ (1/12)·Δx²  (4배 감소)
```

**SRK EOS에서 ε_i 해석적 표현** (Eqs. 48–51):
```
SRK EOS: p = RT/(v̂ - b) - a(T)/(v̂(v̂ + b))

ε_i = Cv_i·T + [... 출발 함수 항 (departure function) ...]
    → 닫힌 형태로 계산 가능
```

**대안 준보존형 PEqC** (완전 보존 포기 시):
```
∂ρE/∂t + ∂((ρE+p)u)/∂x = ∂ρeu/∂x - Σ_i ε_i·∂ρY_i u/∂x
[RHS 보정항이 PEP를 정확히 만족, 에너지 보존은 희생]
```

### 핵심 특징

- **SRK cubic EOS 직접 지원** — 전임계 CH₄/N₂ 계면 이류 테스트 포함
- **완전 보존형** (APEC 버전)
- 기존 split-form/Kinetic Energy Preserving (KEP) 코드에 최소 수정
- 3rd-order TVD RK 시간 적분과 호환

### 장단점

| | |
|---|---|
| ✅ | **실제유체(SRK EOS) + 전임계 흐름 검증** |
| ✅ | **완전 보존형** |
| ✅ | 기존 split-form FVM 코드에 소규모 수정으로 적용 |
| ✅ | ε_i를 SRK에서 해석적으로 계산 — EOS 역산 불필요 |
| ⚠ | 완전 PEP 아닌 근사 (오류 계수 1/12; 실용적으로 충분) |
| ⚠ | PR EOS는 직접 검증 안 됨 (SRK와 구조 유사하므로 확장 가능) |

---

## Paper C: Wang, Wehrfritz, Hawkes (UNSW/Turku) — 물리적 일관성 분석

**파일**: `[적용해볼것] PEP3.pdf`
**제목**: "Physically consistent formulations of split convective terms for turbulent compressible multi-component flows"
**저널**: JCP

### 방법

기존 split 형태(KG, KEEP, KEEPPE, KEEPPE-R)가 **다성분 흐름(가변 γ)에서 PEP를 보장하지 못한다**는 것을 분석하고, 동시에 다음을 만족하는 새로운 물리 일관성 종 이류 형태 제안:

1. **KEP** (운동에너지 보존)
2. **EP** (엔트로피 보존)
3. **PEP** (압력 평형 보존)
4. **UMP** (균일 질량분율 보존: `∂Y_α/∂t = 0` if `Y_α = const`)
5. **TEP** (온도 평형 보존: `∂T/∂t = 0` if `T = uniform`)

**가변 γ에서 PEP 핵심 조건** (Eq. 65):
```
∂(1/(γ-1))/∂t + u_j·∂(1/(γ-1))/∂x_j = 0
```
→ 이것이 모든 종 `X_α` 이류 방정식과 일관되게 이산화되어야 함

**Fujiwara PE-F 스킴 플럭스** (고차 확장, Eqs. 68–71):
```
Ŷ_{α,j}|_{m+1/2} = Σ_{l,k} a_{L,l} · (φ⁻ρY_α|_{m-k} + φ⁺ρY_α|_{m-k+l})/2
                             × (u_j|_{m-k} + u_j|_{m-k+l})/2
```

**새 기여 — UMP/TEP-consistent 종 이류** (Section 3.3.3):
균일 질량분율 보존을 위한 종 플럭스 일관성 조건 (Theorem 2, Eqs. 73–81):
```
ρ̃(ρ, u_j, Y_α0) = Y_α0 · ρ̃(ρ, u_j, 1)  [균일 Y_α0에 대해]
```

### 핵심 특징

- DNS/LES 대상 고차 유한차분 split-form
- 이상기체 (가변 γ) — 실제유체 EOS 미적용
- PEP3 스킴들의 계층적 분석 프레임워크 제공 (코드 설계 참고용)

### 장단점

| | |
|---|---|
| ✅ | PEP + KEP + EP + UMP + TEP 동시 만족하는 통합 프레임워크 |
| ✅ | DNS/LES 고차 코드에 적합 |
| ✅ | 기존 스킴들의 한계를 체계적으로 분석 — 설계 참고용으로 가치 높음 |
| ❌ | **이상기체 전용** |
| ❌ | 실제유체 / 전임계 적용 없음 |

---

## 세 방법 비교

| 항목 | Fujiwara PEP | Terashima APEC | Wang PEP3 |
|------|-------------|----------------|-----------|
| EOS | 이상기체 | SRK (실제유체) | 이상기체 |
| PEP 정확도 | 기계 정밀도 | ≈PEP (오류 1/12·Δx²) | ≈PEP (분석 프레임워크) |
| 완전 보존 | ✅ | ✅ | ✅ |
| 전임계 검증 | ❌ | ✅ | ❌ |
| DNS/LES 적합 | ✅ | ✅ | ✅ (특화) |
| 구현 난이도 | 하 | 중 | 중-상 |

---

## 사용자 코드 적용 권장

**⭐ 최우선 권장**: **Terashima APEC (PEP2.pdf)**

이유:
1. SRK cubic EOS 지원 → 전임계 흐름에 즉시 적용 가능
2. 완전 보존형 → 우선순위 1 충족
3. 기존 split-form FVM 코드에 내부에너지 반점값 보정항(ε_i)만 추가 → 최소 수정
4. ε_i를 SRK EOS로부터 해석적으로 계산 가능 → Newton 반복 불필요

**구현 순서 제안**:
1. Fujiwara PEP (이상기체)로 먼저 구현하여 IEC 검증
2. EOS를 SRK로 교체하며 ε_i 계산 루틴 추가 → APEC으로 업그레이드
3. PR EOS가 필요하면 SRK와 동일 구조이므로 계수만 변경
