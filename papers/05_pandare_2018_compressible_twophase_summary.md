# A Finite-Volume Method for Compressible Viscous Multiphase Flows

> **출처:** Aditya K. Pandare, Hong Luo, AIAA 2018-1814, 2018 AIAA Aerospace Sciences Meeting.
> **관련 실패:** 대밀도비 압축성 다상 유동에서 density-based solver의 implicit 접근법

---

## 1. 핵심 수식

### 6-equation single-pressure system

$$
\frac{\partial \alpha_k \rho_k}{\partial t} + \nabla \cdot (\alpha_k \rho_k \mathbf{u}_k) = 0
$$

$$
\frac{\partial \alpha_k \rho_k \mathbf{u}_k}{\partial t} + \nabla \cdot (\alpha_k \rho_k \mathbf{u}_k \otimes \mathbf{u}_k + \alpha_k p \mathbf{I}) = \text{drag} + \text{virtual mass}
$$

$$
\frac{\partial \alpha_k \rho_k E_k}{\partial t} + \nabla \cdot (\alpha_k (\rho_k E_k + p) \mathbf{u}_k) = \text{source terms}
$$

$$
\frac{\partial \alpha_1}{\partial t} + \mathbf{u}_{int} \cdot \nabla \alpha_1 = \mu(p_1 - p_2)
$$

> **의미:** Stratified flow model (6-equation, single-pressure). 각 상(k=1,2)별 보존 방정식.

### AUSM+-up with volume fraction coupling (AUSM+-upf)

> **의미:** 표준 AUSM+-up에 volume fraction coupling 항 추가. 강한 α-p 불연속 상호작용에서 안정성 확보.

### Primitive variable transformation for implicit solve

$$
\frac{\partial \mathbf{W}}{\partial t} = \left(\frac{\partial \mathbf{Q}}{\partial \mathbf{W}}\right)^{-1} \mathbf{RHS}
$$

> V = {α₁, u₁, v₁, w₁, p, u₂, v₂, w₂, T₁, T₂} — primitive variable로 변환 후 implicit 풀기. ∂Q/∂V 변환 행렬 사용.

---

## 2. 방법론

- **Density-based** FVM (pressure-based가 아님)
- Primitive variable transformation으로 implicit time stepping
- AUSM+-upf flux: void fraction coupling으로 strong shock-interface 안정화
- Virtual mass force: Cvm=0.5, 대밀도비에서 안정성 개선
- Low-Mach preconditioning: pressure diffusion term M_{k,p}

### 기존 방법 대비 차이점

| 항목 | 기존 density-based | 본 논문 |
|------|-------------------|---------|
| Flux at interface | Exact Riemann solver | **AUSM+-upf** (10× 저렴) |
| Low Mach 처리 | 별도 | pressure diffusion term |
| Implicit method | Conservative variables | **Primitive variables** |

---

## 3. 검증

- Water faucet problem (밀도비 ~1000:1)
- Ransom problem (shock-interface)
- Air-water shock tube
- Viscous: 원형 실린더 주위 다상 유동

---

## 4. claudeCFD 적용 메모

- **직접 적용 어려움:** Density-based 방식으로, 우리 pressure-based ACID 솔버와 근본적으로 다름.
- **참고할 점:**
  - Primitive variable transformation (Q→W)은 우리 ΔQ formulation과 유사한 발상
  - Virtual mass force가 대밀도비 안정화에 도움 — 우리 PTC의 M matrix와 유사 역할
  - AUSM+-up의 pressure diffusion이 low-Mach 보정 제공
- **한계:** Two-fluid model (각 상 별도 속도)이므로 우리 one-fluid VOF와 다른 구조.
