# Preconditioning a Newton-Krylov Solver for All-Speed Melt Pool Flow Physics

> **출처:** B. Weston, R. Nourgaliev, J.-P. Delplanque, A.T. Barker, *Journal of Computational Physics* 397 (2019) 108847. DOI: 10.1016/j.jcp.2019.07.045
> **관련 실패:** 5-equation primitive Newton에서 α/ζ conditioning κ=10²¹ → Newton step 과대. Block Schur preconditioning으로 해결 가능.

---

## 1. 핵심 수식

### 지배방정식

압축성 Navier-Stokes (상변화 포함):
$$
\frac{\partial U}{\partial t} + \nabla \cdot (G - D) = S
$$

보존 변수: $U = (\rho, \rho v, \rho e)^T$

### Primitive Variable 공식화 [P, v, T]

보존 변수 대신 **원시 변수** $W = (P, v, T)$ 사용:
$$
\delta U = \frac{\partial U}{\partial W} \delta W
$$

> **의미:** 저 Mach에서 보존 변수 기반 조건수가 극도로 나쁨. 원시 변수로 변환 시 조건수 개선.

### Block Schur Complement Preconditioner (vP-vT 분할)

3×3 블록 행렬:
$$
M = \begin{bmatrix} M_{vv} & M_{vP} & M_{vT} \\ M_{Pv} & M_{PP} & 0 \\ M_{Tv} & 0 & M_{TT} \end{bmatrix}
$$

> **핵심 관찰:** $M_{PT} \approx 0$, $M_{TP} \approx 0$ (압력-온도 약결합)

**vP Schur 여인수:**
$$
S_{vP} = M_{PP} - M_{Pv} M_{vv}^{-1} M_{vP}
$$

**vT Schur 여인수:**
$$
S_{vT} = M_{TT} - M_{Tv} M_{vv}^{-1} M_{vT}
$$

> **의미:** 3×3 완전 결합 시스템을 2개의 2×2 부분계 순서로 분해. SIMPLE/Uzawa 투영법과 유사.

### 해결 절차 (예측-보정)

1. **전진 대입:** $x_v^* = M_{vv}^{-1} b_v$
2. **Schur 풀이:** $x_P = S_{vP}^{-1}(b_P - M_{Pv} x_v^*)$, $x_T = S_{vT}^{-1}(b_T - M_{Tv} x_v^*)$
3. **역진 대입:** $x_v = M_{vv}^{-1}(b_v - M_{vP} x_P - M_{vT} x_T)$

### 시간 이산화

- **BDF2** (L-안정, 2차)
- **ESDIRK** (명시적-대각 암시적 Runge-Kutta)
- 완전 암시적 MOL (Method of Lines) 형식

---

## 2. 방법론

### Newton-Krylov 알고리즘 구조

1. **비선형 시스템:** $F(x) = 0$
2. **Newton:** $J_k \delta x_k = -F(x_k)$, 수렴: $\|F\|_2 < \text{tol}_N \|F_0\|_2$ ($\text{tol}_N \geq 10^{-5}$)
3. **선형 해석:** FGMRES (Flexible GMRES)
4. **Jacobian-Free:** $J\kappa \approx [F(x + h\kappa) - F(x)]/h$ (Frechet 미분)
5. **부정확 Newton:** $\eta_k = \gamma_N (\|F_k\| / \|F_{k-1}\|)^\alpha$ ($\alpha=1.26$, $\gamma_N=0.9$)

### Block 근사 인수분해: 3×3 → 2×2 감소

**3가지 전략:**

| 전략 | $M_{vv}^{-1}$ 근사 | Schur 근사 | 특징 |
|------|---------------------|-----------|------|
| #1 대각 | $D_{vv} = \text{diag}(M_{vv})$ | 명시적 형성 | 저비용, 약함 |
| #2 반복 | AMG V-cycle | 행렬-무료 | 중간 |
| **#3 선행정확** | AMG V-cycle | 명시적 Schur로 precondition | **최고 성능** |

### All-Speed Conditioning 문제 해결

**문제:** 저 Mach ($M < 10^{-2}$)에서 음향 시간스케일 ≫ 유속 시간스케일 → 조건수 폭증

**해결:**
1. **원시 변수 [P,v,T]** 선택 → 보존 변수 대비 조건수 개선
2. **vP Schur**로 강한 속도-압력 결합 명시적 처리 (음향파 주 원인)
3. **vT 독립 처리** → 온도 조건수 양호

**결과:** Mach $10^{-6}$까지 수렴 가능. 반복 횟수 Mach/CFL 무관.

### JFNK vs 명시적 Jacobian

- **선형 해석:** JFNK (행렬-무료, Frechet 미분)
- **Preconditioning:** 근사 명시적 Jacobian (1차 FD) + vP-vT 블록

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 목록

| # | 케이스명 | Mach | Re | 격자 | 특징 |
|---|---------|------|-----|------|------|
| 1 | Lid-driven cavity | $10^{-2}$~$10^{-4}$ | 1,000 | 512×512 | 시간스텝/Mach 연구 |
| 2 | Rayleigh-Benard 용융 | $10^{-3}$ | - | 400×800 | 상변화, 부력 |
| 3 | 3D 레이저 용융풀 | $10^{-3}$ | - | 1M~10M | Marangoni, 증발 |

### 주요 결과

| 지표 | 결과 |
|------|------|
| Mach 수렴 범위 | $10^{-2}$ ~ $10^{-6}$ (cavity) |
| CFL 독립성 | 음향 CFL 0.1 ~ 10,300 안정 |
| 약한 확장성 | 18.5M DoF (288 프로세서) 검증 |
| vP-vT vs AMG(전체) | AMG는 CFL>10에서 발산, vP-vT 안정 |
| vP-vT vs SOR | SOR 대비 3배 빠름 |
| vP-vT vs Block GS | Block GS는 Mach<$10^{-3}$에서 발산 |

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/denner_1d/solver_5eq.py` — Newton 선형 solver preconditioning
- **수정 방향:** 현재 5eq primitive 변수 [p,u,T,α₁]의 4×4 블록 시스템에서:
  1. $M_{vP}$-$M_{vT}$ 분할로 α/ζ conditioning 완화
  2. Schur complement $S_{vP} = M_{PP} - M_{Pv}M_{vv}^{-1}M_{vP}$로 속도-압력 결합 처리
  3. α₁ 방정식은 독립 블록으로 분리 가능
- **주의사항:**
  1. 1D 문제에서 $M_{vv}$가 3중대각 → LU로 정확한 $M_{vv}^{-1}$ 가능
  2. Schur complement 명시적 형성 시 fill-in 주의
  3. 우리의 α/ζ ≈ $2 \times 10^8$은 Weston의 Mach $10^{-6}$ 조건수와 유사 → vP-vT 적용 가능성 높음
