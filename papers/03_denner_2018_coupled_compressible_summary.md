# Fully-coupled pressure-based algorithm for compressible flows: linearisation and iterative solution strategies

> **출처:** Fabian Denner, *Computers & Fluids* 175 (2018) 53-65. DOI: 10.1016/j.compfluid.2018.07.005
> **관련 실패:** 압축성 fully coupled 솔버에서 Newton vs fixed-coefficient linearisation 선택, single-loop vs dual-loop 전략

---

## 1. 핵심 수식

### 지배방정식 (보존형, 원시변수)

$$
\frac{\partial \rho}{\partial t} + \frac{\partial \rho u_i}{\partial x_i} = 0 \quad \text{(continuity → solve for } p\text{)}
$$

$$
\frac{\partial \rho u_j}{\partial t} + \frac{\partial \rho u_i u_j}{\partial x_i} = -\frac{\partial p}{\partial x_j} \quad \text{(momentum → solve for } u\text{)}
$$

$$
\frac{\partial \rho h}{\partial t} + \frac{\partial \rho u_i h}{\partial x_i} = \frac{\partial p}{\partial t} \quad \text{(energy → solve for } h\text{)}
$$

> **의미:** 보존형이지만 원시변수 (p, u, h)로 풀음. ρ는 EOS에서 계산.

### Fixed-coefficient linearisation (Eq. 1)

$$
\alpha^{(n+1)} \phi^{(n+1)} \approx \alpha^{(n)} \phi^{(n+1)}
$$

> **의미:** 계수 α를 이전 iteration 값으로 고정. 구현 간단하지만 수렴 느림.

### Newton linearisation (Eq. 2)

$$
\alpha^{(n+1)} \phi^{(n+1)} \approx \alpha^{(n)} \phi^{(n+1)} + \alpha^{(n+1)} \phi^{(n)} - \alpha^{(n)} \phi^{(n)}
$$

> **의미:** product rule로 cross-coupling 항 도입. 수렴 빠르고 안정적.

### 밀도를 pressure의 함수로 implicit 처리 (Eq. 16)

$$
\rho^{(n+1)}_P = \frac{p^{(n+1)}_P}{(\gamma - 1) c_v T_P}
$$

> **의미:** 연속 방정식에서 ρ를 p에 대해 implicit. T는 가장 최근 값 사용 (dual-loop에서는 outer loop에서 갱신).

### Continuity advection — Newton linearisation (Eq. 21)

$$
\tilde{\rho}^{(n+1)}_f \vartheta^{(n+1)}_f \approx \tilde{\rho}^{(n)}_f \vartheta^{(n+1)}_f + \tilde{\rho}^{(n+1)}_f \vartheta^{(n)}_f - \tilde{\rho}^{(n)}_f \vartheta^{(n)}_f
$$

> **의미:** 연속 방정식 advection에서 ρ_f·ϑ_f를 Newton linearise → 저Mach에서 elliptic, 고Mach에서 hyperbolic 자연 전환.

### 5N×5N 선형계 (Eq. 19)

$$
\begin{pmatrix} A^u_{\rho u} & A^v_{\rho u} & A^w_{\rho u} & A^p_{\rho u} & 0 \\ \cdots & \cdots & \cdots & \cdots & \cdots \\ A^u_{\rho} & A^v_{\rho} & A^w_{\rho} & A^p_{\rho} & 0 \\ A^u_{\rho h} & A^v_{\rho h} & A^w_{\rho h} & A^p_{\rho h} & A^h_{\rho h} \end{pmatrix} \cdot \phi = b
$$

> **의미:** (u, v, w, p, h) 5변수 동시 풀기. Block-Jacobi + BiCGSTAB.

---

## 2. 방법론

### 알고리즘 개요 — Single-loop vs Dual-loop

**Single-loop:**
1. T를 (p, h)에서 갱신: T = (h - u²/2) / c_p
2. ρ = p / ((γ-1)c_v T) 갱신
3. A·φ=b 조립 및 풀기
4. 수렴 확인 → 미수렴이면 반복

**Dual-loop:**
1. **Inner loop** (barotropic): T 고정, ρ = f(p, T_frozen) 사용, A·φ=b 풀기
2. Inner 수렴 후 T 갱신: T = (h - u²/2) / c_p
3. **Outer loop**: 새 T로 inner loop 재수행
4. Inner+Outer 동시 수렴 확인

### 핵심 아이디어

- **Newton linearisation이 결정적**: transient term에 Newton linearisation 적용 시 수렴이 크게 개선.
  - Fixed-coefficient: 운동량 transient = ρ^(n)·u^(n+1)
  - Newton: 운동량 transient = ρ^(n)·u^(n+1) + ρ^(n+1)·u^(n) - ρ^(n)·u^(n)
- **Advection linearisation**: Newton(ρ)·Newton(ϑ) = "full Newton" → 최고 성능이지만 안정성 이슈 가능
- **Single-loop이 dual-loop보다 빠름** (대부분의 경우): Newton linearisation 적용 시 underrelaxation 불필요.
- **Dual-loop의 장점**: fixed-coefficient에서 안정성 확보. Newton에서는 불필요.

### 기존 방법 대비 차이점

| 항목 | Segregated (SIMPLE/PISO) | Coupled (본 논문) |
|------|--------------------------|-------------------|
| 방정식 풀기 | 순차적 (p→u→h) | **동시** (p,u,h) |
| Underrelaxation | 필요 | **불필요** (Newton시) |
| p-u coupling | 약함 | **강함** (MWI implicit) |
| 수렴 속도 | 느림 | **빠름** (Newton quadratic) |
| 메모리 | 적음 | 많음 |

### Linearisation 조합별 성능 (Table 3)

| Case | Continuity advection | Transient (mom/energy) | Advection (mom/energy) | 성능 |
|------|---------------------|----------------------|----------------------|------|
| A | fixed-coeff | fixed-coeff | fixed-coeff | 기준 (느림) |
| D | Newton | Newton | fixed-coeff | **2× 빠름** |
| G | Newton | Newton | full-Newton | **최고 성능** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 목록

| # | 케이스명 | Mach | 격자 | CFL | 비고 |
|---|---------|------|------|-----|------|
| 1 | Acoustic wave propagation | ~0 | 200 | 0.1~1.0 | 저Mach 안정성 |
| 2 | Sod shock tube | 0~1+ | 200 | 0.1~0.5 | 충격파 포착 |
| 3 | Forward-facing step | 3.0 | 12000 | 0.5 | 초음속 |
| 4 | Supersonic cone | 1.96 | 3840 | 0.5 | 원뿔 충격파 |

### 주요 결과

- Newton linearisation이 모든 Mach 영역에서 고정계수 대비 1.5~2× 빠름
- Single-loop + Newton이 dual-loop보다 빠르고 간단
- CFL=0.5에서도 안정적 (dual-loop: CFL=0.1 필요한 경우 있음)
- Acoustic wave: machine precision 수준 정확도
- Shock tube: 충격파/접촉면/팽창파 정확 포착

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/solver_1d.py` — Newton system assembly
- **수정 방향:**
  1. **Single-loop 채택**: 현재 barotropic inner/outer loop → single-loop으로 전환 가능. T를 iteration마다 (p,h)에서 갱신.
  2. **Newton linearisation for transient**: ρ^(n+1)·u^(n+1) ≈ ρ^(n)·u^(n+1) + ρ^(n+1)·u^(n) - ρ^(n)·u^(n). 이것이 cross-coupling 항을 자연스럽게 도입.
  3. **Newton linearisation for continuity advection**: ρ_f·ϑ_f를 Newton linearise → p가 연속 방정식에 더 강하게 결합.
  4. **Underrelaxation 제거**: Newton이 충분하면 별도 relaxation 불필요.
- **주의사항:**
  - 이 논문은 **단상 압축성** (multiphase 아님). Species/VOF 방정식 없음.
  - 하지만 linearisation 전략은 multiphase에도 직접 적용 가능.
  - **핵심 교훈**: Newton linearisation of transient terms가 coupled solver의 수렴/안정성에 결정적. 이는 우리의 α·ΔY/Δt 항을 Newton으로 처리하는 것의 이론적 근거.
  - Dual-loop (barotropic)은 fixed-coefficient 시 필요. Newton을 쓰면 single-loop이 더 좋음.
