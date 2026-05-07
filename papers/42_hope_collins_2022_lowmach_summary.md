# Artificial Diffusion for Convective and Acoustic Low Mach Number Flows (Hope-Collins & di Mare 2022)

> **출처**: Hope-Collins J., di Mare L., *JCP* (2023) DOI: 10.1016/j.jcp.2022.111858. arXiv 2112.08977.
> **관련 실패**: **IMEX 저마하 acoustic amplitude 감쇠**. Round 2에서 `chi = (1-M)²` (convective limit scaling)을 acoustic regime에 잘못 적용한 원인 설명.

---

## 1. 핵심 수식

### 1.1 세 가지 asymptotic limit (Section 1)

**Convective limit** (M→0, ∂_t ≈ u·∇, no acoustic):
- Diffusion scaling: **ε⁰ (O(1))** — full convective diffusion
- 예: Roe convective correction

**Acoustic limit** (oscillatory, no net flow):
- Diffusion scaling: **ε¹ or ε²** — reduced
- 현재 `chi=(1-M)²` 가 이 scaling (ε²) — **하지만 convective limit용이 아닌 acoustic용**

**Mixed convective-acoustic** (일반적):
- Three diffusion scales 모두 필요

### 1.2 수정된 Euler + artificial diffusion

$$
\partial_t U + \partial_x F(U) = \partial_x (D(U) \partial_x U)
$$

D: artificial diffusion matrix. Low-Mach asymptotic:
$$
D \sim \epsilon^\alpha \cdot D_0
$$
여기서 α는 limit에 따라 0, 1, 2.

### 1.3 SLAU2 chi analysis (본 논문의 context)

현재 코드의 `chi = (1-M̂)²`는 **convective limit scaling** (α=2 for velocity).
→ Acoustic limit (순수 음향파)에서는 **over-dissipation**.

Bharate 2025 `z = min(1, max(M_L, M_R))` = **acoustic limit friendly** (velocity jump 자체를 scale).

---

## 2. 방법론

### 2.1 Design guidelines (Section 3)

저마하 scheme 설계시:
1. **알고자 하는 limit 판별** (convective / acoustic / mixed)
2. Diffusion scaling 적절히 선택
3. **Adaptive switching** (Section 3.4) — **우리 approach**

### 2.2 Adaptive schemes (Section 3.4)

Flow regime에 따라 diffusion scaling 전환:
- Detector: local pressure perturbation, Mach number, ∇p/p
- Switch: different diffusion weights

**우리 계획 = Hope-Collins adaptive scheme 방향**:
- Acoustic region → Bharate z-factor (acoustic limit scaling)
- Shock region → SLAU2 χ (convective limit scaling)
- Uniform → minimal diffusion

### 2.3 기존 방법 대비 차이점

| Scheme | Limit | Diffusion scaling |
|--------|:---:|:---|
| Roe (original) | all | ε⁰ (over-diffuse acoustic) |
| Weiss-Smith preconditioning | convective | ε⁰ |
| SLAU2 chi=(1-M)² | mixed | ε² for velocity |
| Thornber 2008 z-factor | acoustic | scales velocity jump by M |
| **제안 (Hope-Collins)** | **adaptive** | regime별 전환 |

---

## 3. 검증 및 시뮬레이션 설정

### 3.1 Roe scheme 응용 (Section 5)

- 각 scaling으로 Roe 구현 → Gresho vortex, acoustic wave propagation 테스트
- Acoustic wave: over-dissipation 제거 입증

### 3.2 주요 결과

- Acoustic regime: convective scaling → O(1) amplitude 손실. Acoustic scaling → preserve
- Mixed: neither alone sufficient → adaptive essential

---

## 4. claudeCFD 적용 메모

### 직접 적용 위치

**`solver/He2024/explicit_mmacm_ex.py` L3763-3786 (SLAU2 block)**:

현재 (convective limit bias):
```python
chi = (1 - M_hat)**2
u_face = V_avg - (chi / (rho_avg * c_avg)) * (pR - pL)
```

**Hope-Collins adaptive**:
```python
# Detect regime
dp_rel = |p_R - p_L| / max(p_L, p_R)
du_rel = |u_R - u_L| / c_avg

is_acoustic = (du_rel > dp_rel * M_hat * 10)  # acoustic-dominated
is_convective = (not is_acoustic) and M_hat > 0.1
# Apply scaling
if is_acoustic:
    # Bharate z-factor (acoustic limit)
    z = min(1.0, max(M_L, M_R))
    # pressure-preserving velocity scaling
elif is_convective:
    # Standard SLAU2 (convective limit)
    chi = (1 - M_hat)**2
else:
    # Mixed: minimum of both
    ...
```

### 수정 방향

1. **Low-Mach acoustic wave에 대한 이론적 타당성 확보** (referees 필수 요구사항)
2. Bharate z-factor 단독 대신 **Hope-Collins adaptive**로 **convective + acoustic** 둘 다 최적화
3. 논문 Section 3.4 adaptive schemes 인용 → 우리 기여가 **regime-switching** 명시
