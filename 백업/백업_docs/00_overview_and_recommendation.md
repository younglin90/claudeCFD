# Abgrall 문제 회피 방법 개요 및 코드 적용 권장사항

## Abgrall 문제란?

다성분 압축성 유동에서 **보존형 수치 스킴**은 다음 문제를 겪는다:

- 밀도 불연속(물질 계면)이 존재할 때, 보존 에너지 `E = ρe + ½ρu²`를 통해 압력을 역산(`p = f(ρ, E, 조성)`)하면 **수치적으로 균일했던 압력 장에 허위 진동(spurious oscillation)이 발생**한다.
- 유속 `u`와 압력 `p`가 공간적으로 균일하더라도, 밀도 불연속이 있는 계면을 지나며 이산화 과정에서 생기는 수치 불일치가 진동을 만든다.

## 논문 그룹 구성

| 그룹 | 핵심 전략 | 파일 |
|------|-----------|------|
| [Group 1](./01_nonconsevative_reformulation.md) | **비보존형 에너지/체적분율 방정식** 사용 | `01_nonconsevative_reformulation.md` |
| [Group 2](./02_primitive_variable_reconstruction.md) | **온도 T 기반 원시변수 재구성** (IEC 만족) | `02_primitive_variable_reconstruction.md` |
| [Group 3](./03_PEP_flux_compatibility.md) | **PEP 플럭스 호환 조건** (반점값 호환) | `03_PEP_flux_compatibility.md` |
| [Group 4](./04_pressure_based_ACID.md) | **압력 기반 / ACID / All-Mach** 포뮬레이션 | `04_pressure_based_ACID.md` |
| [Group 5](./05_entropy_stable.md) | **엔트로피 안정 (ES) 스킴** | `05_entropy_stable.md` |
| [Group 6](./06_artificial_diffusivity_kinetic.md) | **인공 확산 / 운동론적 플럭스** | `06_artificial_diffusivity_kinetic.md` |
| [Group 7](./07_DG_overintegration.md) | **DG 과적분 / L2-투영** | `07_DG_overintegration.md` |
| [Group 8](./08_diffuse_interface_analogue.md) | **확산 계면 스칼라 유사 문제** (참고) | `08_diffuse_interface_analogue.md` |

---

## 사용자 코드에 대한 권장사항

### 코드 환경 요약
- **Solver 형태**: FVM + FEM/DG 하이브리드
- **다성분 목적**: 실제유체 / 전임계(real gas / transcritical) 흐름
- **우선순위**:
  1. 완전 보존성 유지 (Fully conservative)
  2. 압력 진동 억제 성능
  3. 기존 코드 수정 최소화
  4. 구현 난이도 낮을 것

---

### ✅ 1순위 권장: APEC (PEP2.pdf — Terashima, Ly, Ihme)

**파일**: `[적용해볼것] PEP2.pdf`
**방법명**: Approximately Pressure-Equilibrium-Preserving scheme for Energetics Consistency

**왜 적합한가?**
- ✅ **완전 보존형** — 에너지, 질량, 운동량 모두 보존
- ✅ **실제유체(SRK cubic EOS) 직접 적용** — 전임계 CH₄/N₂ 테스트 케이스 포함
- ✅ **기존 FVM/split-form 코드에 최소 수정으로 추가 가능** — 내부에너지 반점값에 EOS 미분 기반 보정항 ε_i 추가
- ✅ **FVM과 FDM 모두 적용 가능** — 하이브리드 solver에 적합
- ⚠ 정확한 PEP 조건이 아닌 근사(O(Δx²) 오류가 1/12로 감소)이나 실용적으로 충분

**핵심 구현 변경점**:
```
기존: ρY_i|_{m+1/2} = (ρY_i|_m + ρY_i|_{m+1}) / 2
      ρe|_{m+1/2}  = (ρe|_m + ρe|_{m+1}) / 2

APEC: ρY_i|_{m+1/2} += 보정항(∂ε_i/∂x 포함)
      ρe|_{m+1/2}   += Σ ε_i × (ρY_i 반점값 보정량)
      여기서 ε_i = (∂ρe/∂ρY_i)_{ρ_{j≠i}, p} — SRK EOS로부터 해석적 계산
```

---

### ✅ 2순위 권장: L2-투영 과적분 (2410.13810v2 — Ching & Johnson) — DG 파트용

**파일**: `2410.13810v2.pdf`
**방법명**: DG with L2-projection overintegration of primitive variables

**왜 적합한가?**
- ✅ **완전 보존형 DG** — 에너지 보존 유지
- ✅ **Peng-Robinson EOS로 실제유체/전임계 테스트 완료** (N₂, n-C₁₂H₂₆)
- ✅ **기존 DG 코드에 quadrature rule 교체 + L2-projection 추가만 필요**
- ✅ FVM 파트에는 영향 없음 (DG 파트만 수정)
- ⚠ 압력 진동을 완전 제거하지는 않음 (bounded, non-growing 수준으로 억제)

**핵심 구현 변경점**:
```
기존(colocated):  F = F(y(x_k)) at n_b nodes
APEC overint:     z = [v_1,...,v_d, P, C_1,...,C_ns]  ← 중간변수
                  Pi(z)를 V^p_h에 L2-투영
                  F = F(Pi(z)) at n_c > n_b quadrature nodes
```

---

### ✅ 3순위 권장 (단순성 우선 시): 온도 T 기반 재구성 (4eq / ssrn-5372486 — Collis et al.)

**파일**: `[적용해볼것] 4eq.pdf`, `ssrn-5372486.pdf`
**방법명**: IEC-satisfying ENO reconstruction with W = [T, Y, u, v, w, P]

**왜 적합한가?**
- ✅ **완전 보존형 + 기계 정밀도 수준의 IEC 만족** (가장 우수한 진동 억제)
- ✅ **구현 단순** — ENO/WENO 재구성 변수를 ρ → T로 교체
- ✅ HLLC Riemann solver와 직접 호환
- ⚠ NASG EOS 사용 (cubic EOS 미적용; 전임계 흐름에 PR/SRK 필요 시 EOS 확장 필요)
- ⚠ real gas (cubic EOS) 테스트 케이스 없음

---

### 비교 요약표

| 방법 | 완전보존 | 실제유체 | 코드 수정량 | 구현 난이도 | 진동 억제 |
|------|---------|---------|------------|------------|-----------|
| **APEC** (PEP2) | ✅ | ✅ SRK | 소 (내부에너지 반점값 보정) | 중 | ★★★★ |
| **L2-투영 DG** (2410) | ✅ | ✅ PR | 중 (DG quadrature 교체) | 중 | ★★★☆ |
| **T-재구성 4eq** (ssrn) | ✅ | △ NASG | 소 (재구성변수 교체) | 하 | ★★★★★ |
| PEP Fujiwara (PEP) | ✅ | ✗ 이상기체 | 중 | 중 | ★★★★ |
| ACID/Denner | ✅ | △ | 대 (solver 구조 변경) | 상 | ★★★★ |
| 비보존 에너지 eq. | ✗ | ✅ | 중 | 중 | ★★★★ |
| AMD (Jain LAD) | ✅ | △ | 소 (항 추가) | 하 | ★★☆☆ |
| 운동론적 λ | ✅ | ✗ | 소 | 하 | ★★★☆ |

---

### 결론

> **실제유체(real gas) + 완전 보존 + 하이브리드 FVM+DG solver**에는 다음 조합을 권장:
>
> - **FVM 파트**: APEC (PEP2.pdf, Terashima et al.) — SRK EOS 지원, 최소 코드 수정
> - **DG 파트**: L2-투영 과적분 (2410.13810v2, Ching & Johnson) — PR EOS 지원, DG 내 quadrature 수정
>
> 전임계 흐름 재현이 최우선이고 cubic EOS 적용 일정이 나중이라면, **먼저 T-재구성 4eq 방법으로 IEC 검증 후 APEC으로 업그레이드**하는 순서도 유효하다.
