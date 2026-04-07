# Group 5: 엔트로피 안정 (ES) 스킴
# (Entropy-Stable Schemes for Multi-Component Flows)

## 핵심 아이디어

**엔트로피 보존/안정(entropy-conserving/stable)** 수치 플럭스는 열역학적으로 일관된 이산화를 보장한다. 계면에서 비물리적 엔트로피 생성이 없으면 압력 불균형도 자동으로 억제된다.

핵심 도구:
- **엔트로피 변수** `v = ∂s/∂U` 를 통해 수치 플럭스를 엔트로피 일관되게 설계
- **엔트로피 보존 플럭스** `F^EC`: `(v_L - v_R)·F^EC = 0` 만족
- **엔트로피 안정 플럭스** `F^ES = F^EC + 소산항`: `(v_L - v_R)·F^ES ≤ 0` 만족

---

## Paper A: Gouasmi, Murman, Duraisamy (2022) — 저-Mach ES 스킴

**파일**: `1-s2.0-S0021999122000985-am.pdf`
**제목**: "Entropy-Stable Schemes in the Low-Mach-Number Regime: Flux-Preconditioning, Entropy Breakdowns, and Entropy Transfers"
**저널**: JCP (2022)

### 방법

⚠ **이 논문은 Abgrall 문제(다성분 계면 진동)를 직접 다루지 않음.**
단일 성분 압축성 유동에서 **저-Mach 수치 부정확성** 문제를 ES 스킴과 flux preconditioning 조합으로 해결.

**엔트로피 안정 플럭스 + 전처리**:
```
표준 Roe: f* = ½(f_L + f_R) - ½|A|(u_R - u_L)
전처리 Roe: f* = ½(f_L + f_R) - ½P⁻¹|PA|(u_R - u_L)
```

**저-Mach 스케일된 엔트로피 변수** (Eq. 33):
```
v = [γ - s/(γ-1) - M²_r·ρk/p, M²_r·ρu^T/p, -ρ/p]^T
```

**Entropy Production Breakdowns (EPBs)**:
모드별 엔트로피 생성 분해로 특정 수치 아티팩트(음향 모드 허위 발생) 진단.

### 장단점

| | |
|---|---|
| ✅ | 저-Mach 정확도 개선 분석 프레임워크로 유용 |
| ✅ | EPB 진단 도구 — 수치 아티팩트 원인 파악에 유용 |
| ❌ | **단일 성분 전용** — 다성분/Abgrall 문제 미적용 |
| ❌ | 전임계/실제유체 없음 |
| ❌ | FVM 1차 정확도, Backward Euler 시간 적분 |

### 코드 적용성

간접 참조용. ES 스킴 설계 원리 이해와 저-Mach 보정에 참고.

---

## Paper B: Regener Roig, Crivellini, Colombo (2026) — 다성분 EC/ES DG

**파일**: `1-s2.0-S0021999126001580-main.pdf`
**제목**: "Efficient entropy-conserving/stable discontinuous Galerkin solution of the multicomponent compressible Euler equations"
**저널**: JCP 556 (2026) 114808

### 방법: EC/ES DG + DEEB 보정 + 엔트로피 투영

다성분 압축성 Euler 방정식을 모달 DG로 풀면서 동시에:
1. **완전 보존형** (에너지 보존)
2. **엔트로피 보존/안정** (EC/ES 플럭스)
3. **압력 평형 보존** (PEP) — 엔트로피 투영을 통한 간접 만족

**다성분 Euler 보존 변수**:
```
q = [ρ₁,...,ρ_N, ρu, ρe_t]^T   (N종 혼합 이상기체, Dalton 법칙)
p = Σ_k (γ_k - 1)·ρ_k·e_k = Σ_k ρ_k·r_k·T
```

**엔트로피 변수** (Eq. 19):
```
v = (1/T)[h_k - T·s_k - |u|²/2, ..., u, -1]^T
```

**엔트로피 보존 플럭스 (Gouasmi et al.)** (Eq. 41):
```
F̂^EC_{1,k} = ρ^{ln}_k · u_n           [로그 평균 부분밀도]
F̂^EC_2     = (arithmetic mean of total pressure)
F̂^EC_3     = (log-mean temperature 기반)
```
여기서 `ρ^{ln}_k = (ρ_{k,L} - ρ_{k,R}) / (ln ρ_{k,L} - ln ρ_{k,R})`

**엔트로피 안정 Rusanov 소산** (Eq. 44):
```
F̂^RU = F̂^EC - ½·λ_max·(∂q/∂v)(w̃)·[[w]]
```

**핵심 기여: DEEB 보정** (Direct Enforcement of Entropy Balance, Eq. 38):

DG에서 엔트로피 보존을 보장하려면 과적분(over-integration)이 필요한데 이는 비용이 크다. 대신 요소별 **보정 계수 α_K**를 추가:

```
Σ_K α_K · (∫_K Φ_h v*_h dΩ) / (∫_K v*_h·v*_h dΩ) = 0

α_K (Eq. 40) = 요소 내 엔트로피 포텐셜 플럭스 균형의 이산 잔차
```

이 보정은 엔트로피 보존을 요소 수준에서 강제하면서 과적분 비용을 피함.

**엔트로피 투영**: DOF를 보존 변수로 진전시키지만, 공간 잔차 계산 시 **엔트로피 변수 `v*(q_h)`의 L²-투영**을 DG 다항식 공간에 적용하여 PEP를 간접적으로 달성.

**방향성 충격파 감지 + 인공 확산** (Eqs. 48–57): 압력 구배 방향으로 정렬된 국소 인공 확산 추가.

### 장단점

| | |
|---|---|
| ✅ | **완전 보존형 DG** (모달, 비정렬 메쉬 가능) |
| ✅ | 엔트로피 보존/안정 + PEP 동시 만족 |
| ✅ | DEEB 보정으로 과적분 없이 엔트로피 안정 달성 |
| ✅ | 다성분 Euler 방정식 전용 설계 |
| ⚠ | 이상기체 혼합 (Dalton 법칙) — cubic EOS 아님 |
| ❌ | 실제유체/전임계 없음 |
| ❌ | DG 한정 — FVM 파트에 직접 적용 어려움 |
| ❌ | 구현 복잡 (엔트로피 변수 계산, DEEB 루프, 로그 평균 플럭스) |

---

## 두 논문 비교

| 항목 | Gouasmi (2022) | Regener Roig (2026) |
|------|---------------|---------------------|
| 방법 | ES + 저-Mach preconditioning | EC/ES DG + DEEB + 엔트로피 투영 |
| 다성분 | ❌ 단일 성분 | ✅ N성분 |
| DG/FVM | FVM (1차) | DG (고차) |
| 실제유체 | ❌ | ❌ |
| PEP 만족 | N/A | ✅ (간접, 엔트로피 투영) |
| 주 기여 | EPB 진단 도구 | DEEB 보정, EC/ES DG 다성분 |

---

## 사용자 코드 적용 권장

**DG 파트에서 엔트로피 안정성이 필요한 경우**: Regener Roig 방법 검토 가치 있음.

단, 사용자 우선순위 기준으로:
- **실제유체 지원**: ❌ (이상기체 전용)
- **구현 난이도**: 높음 (로그 평균 플럭스, DEEB 루프)
- **FVM과의 호환**: 제한적

→ **이 그룹은 2차 선택지**. 기본 PEP 구현(Group 3 APEC) 후 엔트로피 안정성이 필요한 경우 추가로 통합 고려.

**추천 조합**: APEC (Group 3) + Rusanov 엔트로피 소산항 추가 → 실질적으로 ES 효과 획득
