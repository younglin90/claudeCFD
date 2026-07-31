# 2. Methods

본 장은 제안하는 정상상태 격자 볼츠만 가속 솔버(이하 **proposed solver**)를 기술한다. 핵심 설계 원칙은 다음 두 가지다. (i) 표준 LBM의 collision–streaming 연산자 $L$ 을 수정하지 않고 그 **native fixed-point residual** $R(f) = L(f) - f$ 만을 직접 줄인다. (ii) 단일 safeguarded Newton–Krylov 템플릿을 모든 검증 사례에 적용하되, 솔버의 내부 파라미터는 사례 이름이 아닌 **측정 가능한 문제 속성**(격자 크기 $N$, 고체 마스크 $\chi$의 유무와 공극률, Reynolds 수 $Re$)에 의해서만 적응한다. 모든 단계는 동일한 LBE 평가(LBE-call)로 비용을 측정하며, 해석적 목표해(analytic target)나 평형분포 lifting을 사용하지 않는다.

§2.1에서 D2Q9 BGK 정식화와 거시 모멘트 연산자를, §2.2에서 정상상태 residual 정식화를, §2.3에서 secant 고정점 부트스트랩을, §2.4에서 Fourier-moment AP-Schur 전처리기를, §2.5에서 safeguarded Newton–Krylov 코어를, §2.6에서 kinetic Picard 후처리와 질량 보존 투영을, §2.7에서 통합 솔버 템플릿과 수렴 판정을, §2.8에서 의사코드와 계산 복잡도를 기술한다.

---

## 2.1 D2Q9 lattice Boltzmann 정식화

2차원 등온 비압축성 극한의 유동은 D2Q9 격자에서 분포함수 $f_i(\mathbf{x}, t)$, $i = 0, \dots, 8$ 의 진화로 기술된다. 이산 속도 $\mathbf{c}_i$ 와 가중치 $w_i$ 는 표준값을 사용한다:

$$
\mathbf{c}_0 = (0,0), \quad
\mathbf{c}_{1\text{--}4} = (\pm 1, 0), (0, \pm 1), \quad
\mathbf{c}_{5\text{--}8} = (\pm 1, \pm 1),
$$
$$
w_0 = \tfrac{4}{9}, \quad w_{1\text{--}4} = \tfrac{1}{9}, \quad w_{5\text{--}8} = \tfrac{1}{36}, \quad c_s^2 = \tfrac{1}{3}.
$$

거시 밀도와 운동량은 분포함수의 모멘트로 정의된다:

$$
\rho = \sum_i f_i, \qquad \rho\,\mathbf{u} = \sum_i \mathbf{c}_i f_i .
$$

BGK 충돌 후 streaming 으로 구성되는 한 시간 스텝 연산자 $L$ 은

$$
L(f)_i(\mathbf{x}) = f_i(\mathbf{x} - \mathbf{c}_i) - \omega\big[f_i(\mathbf{x} - \mathbf{c}_i) - f_i^{\mathrm{eq}}(\mathbf{x} - \mathbf{c}_i)\big] + S_i,
$$

이며, 평형분포는

$$
f_i^{\mathrm{eq}} = w_i \rho \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2 c_s^4} - \frac{\mathbf{u}\cdot\mathbf{u}}{2 c_s^2} \right],
$$

$\omega = 1/\tau$ 는 완화율, $\tau = 0.5 + \nu / c_s^2$ 는 점성 $\nu$ 에 대응하는 완화시간, $S_i$ 는 Guo et al. 외력항(강제 유동 사례)이다. 경계조건은 사례별로 (i) 주기, (ii) 정지/이동 벽의 half-way bounce-back, (iii) voxel 고체 마스크 $\chi$ 의 fluid–solid link bounce-back 로 부여하며, $L$ 의 일부로 흡수된다. **본 연구의 가속 알고리즘은 $L$ 을 black-box 로 호출하므로 collision 모델·경계조건·외력에 독립적이다.**

거시 투영 $\mathsf{M}: \mathbb{R}^9 \to \mathbb{R}^3$ 과 lifting $\mathsf{T}: \mathbb{R}^3 \to \mathbb{R}^9$ 를

$$
\mathsf{M} =
\begin{bmatrix}
1 & \cdots & 1 \\
c_{x,0} & \cdots & c_{x,8} \\
c_{y,0} & \cdots & c_{y,8}
\end{bmatrix},
\qquad
\mathsf{T}_{i,:} = \big[\, w_i, \; 3 w_i c_{x,i}, \; 3 w_i c_{y,i} \,\big]
$$

로 정의하면, 보존 모멘트 $(\rho, \rho u_x, \rho u_y)$ 에 대해 Galerkin 일관성 $\mathsf{M}\mathsf{T} = \mathsf{I}_3$ 이 성립한다.

---

## 2.2 정상상태 fixed-point residual 정식화

정상상태 해 $f^\star$ 는 시간 전진 연산자 $L$ 의 고정점이다:

$$
R(f^\star) = 0, \qquad R(f) := L(f) - f .
$$

표준 LBM은 단순 Picard 반복 $f^{n+1} = L(f^n)$ 으로 이 고정점에 수렴하지만, 저 Mach·고 Reynolds 수 영역에서 음향 모드의 stiffness로 인해 수렴이 매우 느리다. 제안 솔버는 $R(f)=0$ 을 **수정 방정식 없이** 직접 푸는 safeguarded Newton–Krylov 가속기로, 수렴해가 원래 Picard 고정점과 동일함을 보장한다(§2.7의 stopping 판정 참조).

residual의 정량 척도로 RMS norm을 사용한다:

$$
\|R(f)\|_{\mathrm{rms}} = \left( \frac{1}{9 N^2} \sum_{i,\mathbf{x}} R_i(\mathbf{x})^2 \right)^{1/2}.
$$

추가로, 물리적 정상상태 판정을 위해 거시 속도 변화율

$$
\eta_u(f) = \frac{\| \mathbf{u}(L(f)) - \mathbf{u}(f) \|_2}{\max(\|\mathbf{u}(L(f))\|_2,\ \varepsilon_0)}
$$

을 함께 평가한다($\varepsilon_0 = 10^{-30}$).

---

## 2.3 Secant 고정점 부트스트랩

Newton 단계 이전에, 비용이 낮은 **Type-II Anderson(secant) 부트스트랩**을 적용하여 초기장을 정상상태의 끌개 근방으로 이동시킨다. 깊이 $m$ 의 이력 $\{f^{(j)}, g^{(j)} = L(f^{(j)}), r^{(j)} = g^{(j)} - f^{(j)}\}_{j=k-m}^{k}$ 에 대해 차분 행렬

$$
\Delta r_j = r^{(j+1)} - r^{(j)}, \qquad \Delta g_j = g^{(j+1)} - g^{(j)}
$$

을 구성하고, 최소자승

$$
\boldsymbol{\gamma} = \arg\min_{\boldsymbol{\gamma}} \big\| r^{(k)} - \Delta r\, \boldsymbol{\gamma} \big\|_2
$$

의 해로 후보

$$
f^{(k+1)} = g^{(k)} - \Delta g\, \boldsymbol{\gamma}
$$

를 생성한다. residual-monotone safeguard로 $\|r(f^{(k+1)})\|_{\mathrm{rms}} < \|r^{(k)}\|_{\mathrm{rms}}$ 일 때만 후보를 수용하고, 아니면 단순 Picard 스텝 $f^{(k+1)} = g^{(k)}$ 로 후퇴한다. 부트스트랩은 잔차가 tolerance 미만이 되거나 $\max(15, N/2)$ 회에 도달하면 종료한다.

---

## 2.4 Fourier-moment AP-Schur 전처리기

Newton–Krylov 단계의 핵심 효율 layer는 **거시 부분공간에 작용하는 Schur complement 전처리기**이다. 균일 기저상태 $(\bar\rho = 1, \bar{\mathbf{u}} = 0)$ 에서 선형화한 LBE 스텝은

$$
L'(\mathbf{k}) = A(\mathbf{k})\, C(\omega), \qquad
A(\mathbf{k}) = \mathrm{diag}\!\big(e^{-i \mathbf{k}\cdot\mathbf{c}_i}\big), \quad
C(\omega) = (1-\omega)\mathsf{I} + \omega\, \mathsf{T}\mathsf{M},
$$

이며, residual Jacobian은 $J(\mathbf{k}) = \mathsf{I} - L'(\mathbf{k})$ 이다. 여기서 streaming 연산자 $A(\mathbf{k})$ 가 D2Q9 주기 격자에서 **Fourier 모드별로 정확히 대각화**된다는 점이 핵심이다. 거시 부분공간의 Galerkin Schur complement는

$$
S_U^{\mathrm{G}}(\mathbf{k}) = \mathsf{M}\, J(\mathbf{k})\, \mathsf{T} = \mathsf{I}_3 - \mathsf{M} A(\mathbf{k}) \mathsf{T},
$$

로 각 파수 $\mathbf{k} = 2\pi(m,n)/N$ 에서 $3\times3$ 복소행렬로 닫힌다.

운동학적 영공간(kinetic null-space)의 기여를 보정하기 위해 **asymptotic-preserving (AP) 보정항**을 추가한다:

$$
S_U^{\mathrm{AP}}(\mathbf{k}) = S_U^{\mathrm{G}}(\mathbf{k}) - \kappa(\omega)\,\big[\, \mathsf{M} A(\mathbf{k})^2 \mathsf{T} - (\mathsf{M} A(\mathbf{k}) \mathsf{T})^2 \,\big],
$$

여기서 보정계수는 magnitude-clipping을 적용한

$$
\kappa(\omega) = \tfrac{1}{2}\,\mathrm{sign}(\xi)\,\min(0.5,\ |\xi|), \qquad \xi = \frac{1-\omega}{\omega}
$$

로, $\omega \to 2$ ($\nu \to 0$, 고 Reynolds) 극한에서 보정항이 발산하지 않도록 유계화한다.

각 모드별 $3\times3$ 블록은 spectrum-적응형 Tikhonov 정칙화로 조건수를 균등 상한한다:

$$
\tilde S_U(\mathbf{k}) = S_U^{\mathrm{AP}}(\mathbf{k}) + \eta\, \mathsf{I}_3, \qquad
\eta = \frac{1}{50}\max_{\mathbf{k}} \sigma_{\max}\!\big(S_U^{\mathrm{AP}}(\mathbf{k})\big),
$$

전처리기는 $S_{\mathrm{inv}}(\mathbf{k}) = \tilde S_U(\mathbf{k})^{-1}$ 로 **사전 계산 1회**에 모든 모드에 대해 구한다($\mathcal{O}(N^2)$ 메모리, $3\times3$ 역행렬). 평균 모드 $\mathbf{k} = 0$ 은 질량 보존에 해당하므로 밀도 평균 성분에는 Newton 보정을 가하지 않고 운동량 평균 성분만 통과시킨다:

$$
S_{\mathrm{inv}}(0) = \mathrm{diag}(0,\ 1,\ 1).
$$

전처리기 작용은 FFT 2회와 모드별 $3\times3$ 곱으로 $\mathcal{O}(N^2 \log N)$ 비용에 수행된다:

$$
M^{-1} R_f = \mathsf{T}\; \mathcal{F}^{-1}\!\Big[\, S_{\mathrm{inv}}(\mathbf{k})\; \mathcal{F}\big[\mathsf{M} R_f\big] \,\Big].
$$

**이 전처리기는 LBM 격자의 streaming spectrum과 보존 모멘트 구조에 직접 의존하므로 일반 PDE용 전처리기와 구별된다.**

---

## 2.5 Safeguarded Newton–Krylov 코어 (Safe-NN)

부트스트랩 이후, residual-monotone safeguard로 보호되는 Nesterov-가속 Newton–Krylov 반복(Safe-NN)을 적용한다. 외부 반복 $k$ 에서:

**(1) Nesterov lookahead.** 직전 두 반복점을 외삽하여

$$
y_k = f_k + \beta_k (f_k - f_{k-1})
$$

를 구성한다. 모멘텀 계수 $\beta_k$ 는 동적 상한 $\beta_{\max}$ 하에서 잔차 감소 시 점증($\beta \leftarrow \min(\beta_{\max}, \beta + 0.15)$), 증가 시 감쇠($\beta \leftarrow 0.7\beta$)한다.

**(2) Residual-monotone safeguard.** 외삽 강도가 큰 경우($\beta_k > 0.3$)에만 추가 LBE 평가로 $R(y_k)$ 를 사전 검사하고, 완화된 단조성 조건

$$
\| R(y_k) \|_2 \le (1 + \varepsilon_{\mathrm{eff}})\, \| R(f_k) \|_2, \qquad \varepsilon_{\mathrm{eff}} = \varepsilon_0' + 0.2\,\beta_k
$$

을 만족할 때만 외삽점을 수용한다. 위반 또는 비유한값 발생 시 $y_k = f_k$, $\beta_k \leftarrow 0.7\beta_k$ 로 후퇴한다.

**(3) Preconditioned JFNK 보정.** 수용된 $y_k$ 에서 inexact Newton 보정 $\delta f$ 를 Jacobian-free Krylov로 푼다:

$$
J(y_k)\, \delta f = -R(y_k),
$$

여기서 Jacobian–vector 곱은 1차 유한차분으로 근사한다:

$$
J(y_k)\, v \approx \frac{R(y_k + \epsilon v) - R(y_k)}{\epsilon}, \qquad
\epsilon = \frac{10^{-7}(1 + \|y_k\|_2)}{\|v\|_2}.
$$

선형계는 §2.4의 AP-Schur 전처리기를 우측 전처리기로 하는 GMRES(maxiter=1, restart=$2 k_{\max}$, forcing tol $= 10^{-3}$)로 근사 해결한다.

**(4) K-annealed kinetic 후처리.** 중간 상태 $y_k + \delta f$ 에 $K_{\mathrm{eff}}$ 회의 BGK 완화를 적용하여 Newton 보정 후 남은 고주파 운동학적 잔차를 LBM 자체 동역학으로 감쇠시킨다:

$$
f_{k+1} = L^{K_{\mathrm{eff}}}\big(y_k + \alpha\, \delta f\big).
$$

$K_{\mathrm{eff}}$ 는 잔차가 작고 단조 감소할 때 절반으로 줄여($K_{\mathrm{eff}} \leftarrow \max(K_{\min}, K/2)$) smooth regime에서 잉여 LBE를 제거한다. 최종 상태가 비유한값이거나 단조 조건을 위반하면 해당 가속 단계를 폐기하고 baseline Picard 완화로 후퇴(failure recovery)하여 NaN 발산을 차단한다.

---

## 2.6 Kinetic Picard 후처리와 질량 보존 투영

Newton 수렴 후, 물리적 정상상태 판정($\eta_u$, §2.2)이 만족될 때까지 추가 Picard 완화를 적용한다. voxel 마스크 사례에서는 numba njit 커널로 가속된 in-place 완화를 사용하며, 완화 비용은 모두 LBE-call로 계상한다.

마스크/벽 경계에서 누적될 수 있는 질량 drift를 제거하기 위해, collision invariant인 밀도 평균을 보존하는 투영을 적용한다:

$$
\hat f = f \odot \chi + w\, \frac{\big(\textstyle\sum f^{(0)} - \sum (f \odot \chi)\big)}{\sum \chi}\, \chi,
$$

여기서 $\chi$ 는 fluid 마스크(주기/벽 사례에서는 1), $f^{(0)}$ 는 초기장, $w = (w_i)$ 는 가중치 벡터이다. 이 투영은 거시 운동량장을 변화시키지 않으면서 전체 질량을 초기값으로 복원한다.

---

## 2.7 통합 솔버 템플릿과 수렴 판정

제안 솔버는 단일 함수 `solve_proposed_single(case, tol)` 로 모든 사례에 동일한 알고리즘 골격(secant 부트스트랩 → Safe-NN Newton–Krylov → kinetic Picard 후처리 → 질량 투영)을 적용한다. 솔버 내부 파라미터(Krylov 차원 $k_{\max}$, kinetic 스텝 $K_{\mathrm{eff}}$, 모멘텀 상한 $\beta_{\max}$)는 **측정 가능한 문제 속성**에만 의존하여 적응한다:

| 문제 속성 | 적응 동작 |
|---|---|
| 격자 크기 $N \ge 64$ (large grid) | 낮은 kinetic 스텝, 높은 Krylov 차원 |
| 고체 마스크 $\chi$ 존재 + 공극률 $<0.45$ | tight tolerance, 낮은 kinetic 스텝 |
| Reynolds 수 $Re \ge 10^3$ (stiff) | PLBE 형태 전처리 warm-start + cavity 커널 |
| 강제 주기 유동 (Kolmogorov 형) | secant 부트스트랩 단독 |

수렴 판정은 두 기준의 결합이다: (i) native residual $\|R(f)\|_{\mathrm{rms}} < \mathrm{tol}$, 또는 (ii) 거시 속도 변화율 $\eta_u < 10^{-6}$ (논문 검증의 표준 정지 규칙). 두 척도를 모두 보고하여 "stopping rule만 만족하고 잔차가 큰" 거짓 수렴을 배제한다.

**정상상태 해 보존.** 알고리즘의 모든 가속 단계(부트스트랩, Newton 보정, kinetic 후처리, 질량 투영)는 원래 고정점 $R(f^\star)=0$ 을 불변점으로 갖는다. 부트스트랩과 Newton은 $R$ 의 영점을 변경하지 않고, kinetic 후처리는 $L$ 자체이며, 질량 투영은 collision invariant만 보정한다. 따라서 수렴해는 표준 LBM의 정상해와 동일하며 수정 방정식을 풀지 않는다.

---

## 2.8 의사코드와 계산 복잡도

**Algorithm 1. Proposed single-template steady-state solver**

```
입력: case (LBM 연산자 L, 초기장 f0, 속성 N, χ, Re), tolerance tol
출력: 정상상태 해 f*

1.  f ← secant_bootstrap(case)                       # §2.3, Type-II Anderson
2.  S_inv ← build_AP_Schur(N, ω)                      # §2.4, 1회 사전계산
3.  work ← case with initial f
4.  f ← Safe_NN(work, S_inv):                         # §2.5
        for k = 0,1,...:
            R ← L(f) - f
            if ||R||_rms < tol: break
            y ← f + β(f - f_prev)                     # Nesterov lookahead
            if β > 0.3 and ||R(y)|| > (1+ε_eff)||R||: # safeguard
                y ← f;  β ← 0.7β
            δf ← GMRES(J(y)·δf = -R(y),  M⁻¹=AP-Schur) # JFNK
            f ← L^{K_eff}(y + α·δf)                    # K-annealed relax
            if not finite or 비단조: f ← L(f)          # failure recovery
5.  f ← picard_polish_until(f, η_u < 1e-6)            # §2.6
6.  f ← restore_mass(f)                                # §2.6
7.  return f
```

**계산 복잡도(외부 반복 1회당).**

| 연산 | 비용 |
|---|---|
| residual / JVP 평가 | $\mathcal{O}(N^2)$ per LBE-call (numba njit) |
| AP-Schur 전처리 작용 | $\mathcal{O}(N^2 \log N)$ (FFT 2회 + 모드별 $3\times3$) |
| GMRES inner (restart $2k_{\max}$) | $k_{\max}$ JVP = $2k_{\max}$ LBE-call |
| kinetic 후처리 | $K_{\mathrm{eff}}$ LBE-call |

**메모리.** baseline LBM 분포함수($9N^2$) 위에 추가로 (i) 직전 반복점 $f_{k-1}$, (ii) secant/Newton 작업공간, (iii) Krylov 부분공간($\mathcal{O}(k_{\max} \cdot 9N^2)$), (iv) AP-Schur 블록 inverse table($N^2 \times 3 \times 3$ 복소수)을 요한다. 합계는 baseline의 약 2–3배로, 동일 depth Anderson acceleration과 동급이다.

**병렬화.** collision·streaming·residual 평가는 노드 단위 local 연산이므로 표준 domain-decomposition 병렬화가 그대로 적용되며, AP-Schur의 핵심 비용은 분산 FFT이다. 본 실험은 단일 노드 CPU에서 numba njit 커널과 NumPy FFT/SciPy GMRES로 수행했고, 효율은 구현 중립적인 **LBE-call 수**로 보고한다(wall-clock은 보조 지표).
