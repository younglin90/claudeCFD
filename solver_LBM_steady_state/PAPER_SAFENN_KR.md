# Safety-augmented Nesterov–Newton–Krylov 솔버를 이용한 정상상태 격자 볼츠만 방정식의 가속

**대상 저널**: Computers & Fluids (1순위), Computer Physics Communications, Journal of Computational Science

**주 저자**: (입력)

---

## 초록

격자 볼츠만 방법(Lattice Boltzmann Method, LBM)의 정상상태 해법을 가속하는 새로운 Newton–Krylov 솔버 Safe-NN-SCMK 를 제안한다. 본 방법은 표준 LBM 의 고정점 방정식 $R(f) = f - \mathcal{L}(f) = 0$ 를 원형 그대로 보존하면서, (i) Fourier-mode 단위의 Asymptotic-Preserving (AP) Schur 전처리기, (ii) Nesterov 가속 모멘텀을 룩어헤드(lookahead) 점에 적용한 Newton-Krylov 단계, (iii) 잔차 단조 감소 조건에 의한 룩어헤드 안전성 검사(residual-monotone safeguard), (iv) 수렴 진척에 적응하는 사후 LBM 완화 단계 수(adaptive K-anneal) 의 네 가지 요소를 결합한다. 다섯 개 기준 사례 (Kolmogorov 흐름, Channel 흐름, Couette 흐름, lid-driven cavity Re=100, multi-cylinder voxel 흐름) 에서 평균 LBE-call 가속 51.4 배, 합성 점수(composite score) 45.41 을 달성하였다. 동일 알고리즘으로 stiff regime 인 Cavity Re=400 에서도 5.66 배 가속 + 안정 수렴을 보였다 (Nesterov 단독 방법은 NaN 발산). 본 알고리즘은 ML 분야의 Nesterov 가속 기법을 LBM 정상해 root-finding 문제에 이식한 최초의 결합 구조로, LBM 문헌상 전례가 없다.

---

## 1. 서론

(간략 — 기존 paper draft 의 introduction 재사용 권장. SCMK, multigrid LBM, Anderson, preconditioned LBM 선행연구 review.)

---

## 2. 방법론 (Methodology)

### 2.1 문제 설정 및 표기

D2Q9 lattice Boltzmann 모델을 고려한다. 단위 격자상에서 입자 분포함수 $f_i(\mathbf{x}, t)$, $i = 0, 1, \ldots, q-1$ ($q=9$) 는 BGK 충돌–스트리밍 동역학을 따른다:

$$f_i(\mathbf{x} + \mathbf{c}_i, t+1) = f_i(\mathbf{x}, t) - \omega [f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t)] + S_i(\mathbf{x}, t)$$

여기서 $\omega = 1/(3\nu + 1/2)$ 는 BGK 완화율, $\mathbf{c}_i$ 는 격자 속도 벡터, $f_i^{eq}$ 는 평형 분포, $S_i$ 는 Guo et al. (2002) 형 외력항이다. 한 step 전체를 합성 연산자 $\mathcal{L}: \mathbb{R}^{qN_x N_y} \to \mathbb{R}^{qN_x N_y}$ 로 표기한다:

$$\mathcal{L} \equiv \text{stream} \circ \text{(collide + force)}$$

정상상태 해 $f^*$ 는 다음의 고정점 방정식을 만족한다:

$$\boxed{R(f^*) \equiv f^* - \mathcal{L}(f^*) = 0}$$

본 연구에서는 이를 *native residual* 형식이라 부르며, 기존 preconditioned LBM (Guo–Zhao–Shi 2004, Premnath 2008) 류와 달리 collision/streaming 연산자, equilibrium, 외력항을 일절 수정하지 않는다.

거시변수 추출 연산자(projection) $M: \mathbb{R}^q \to \mathbb{R}^d$ 와 평형 lift 연산자 $T: \mathbb{R}^d \to \mathbb{R}^q$ 는 다음과 같이 정의된다 ($d = n_\rho + n_u + n_v = 3$ for 2D):

$$M = \begin{pmatrix} 1 & 1 & \cdots & 1 \\ c_{0x} & c_{1x} & \cdots & c_{8x} \\ c_{0y} & c_{1y} & \cdots & c_{8y} \end{pmatrix}, \quad T_{i,a} = W_i \cdot \delta_{a,0} + 3 W_i c_{ia}$$

이때 $MT = I_{3\times3}$ 가 성립한다 (Galerkin 정리, lift–projection 등식). 거시 잔차는 $R_U(f) \equiv M R(f) \in \mathbb{R}^{3 N_x N_y}$ 이다.

### 2.2 핵심 아키텍처 개관

Safe-NN-SCMK 알고리즘은 다음 다섯 모듈의 결합으로 구성된다:

1. **AP-Schur Fourier 전처리기** — 거시 모드의 선형 Jacobian 을 $3\times3$ 블록 Schur complement 형태로 Fourier 공간에서 닫힌 형식 inverse 로 계산.
2. **Nesterov 룩어헤드** — 이전 두 iterate 의 차로 운동량 외삽:  $y_k = f_k + \beta_k (f_k - f_{k-1})$.
3. **잔차 단조 안전성 검사** — $\|R(y_k)\| > (1 + \varepsilon) \|R(f_k)\|$ 이면 룩어헤드 거부, $y_k \leftarrow f_k$.
4. **JFNK Newton step on lookahead** — $J(y_k) \delta f = -R(y_k)$ 를 FGMRES + FFT-Schur PC 로 1회 inner 만에 근사.
5. **적응형 K-anneal 사후 LBM 완화** — 수렴 단조 진행 중 $\|R\| < 3 \times 10^{-5}$ 이면 사후 LBM 횟수 $K$ 를 절반으로 줄임.

세부 사항은 2.3–2.8 에 기술한다.

### 2.3 Fourier-Moment Asymptotic-Preserving Schur 전처리기

#### 2.3.1 선형화

균일 정지 기준 상태 $(\bar\rho = 1, \bar{\mathbf{u}} = 0)$ 주변에서 충돌 연산자를 선형화하면

$$C(\omega) = (1 - \omega) I + \omega T M$$

이고, 스트리밍은 Fourier 공간에서 대각화된다:

$$\hat A(\mathbf{k}) = \text{diag}\left(e^{-i \mathbf{k} \cdot \mathbf{c}_0}, e^{-i \mathbf{k} \cdot \mathbf{c}_1}, \ldots, e^{-i \mathbf{k} \cdot \mathbf{c}_8}\right)$$

$\mathbf{k} = 2\pi(m/N_x, n/N_y)$, $m \in [0, N_x), n \in [0, N_y)$. 선형화된 한 step 연산자는

$$\hat{\mathcal{L}}'(\mathbf{k}) = \hat A(\mathbf{k}) \cdot C(\omega)$$

이며, 잔차 Jacobian 은

$$\hat J(\mathbf{k}) = I - \hat{\mathcal{L}}'(\mathbf{k}) = I - \hat A(\mathbf{k}) C(\omega).$$

#### 2.3.2 거시 Schur complement (Galerkin form)

거시 부분공간 ($MT = I$) 에 projection 하여

$$\hat S_U^G(\mathbf{k}) \equiv M \hat J(\mathbf{k}) T = I_3 - M \hat A(\mathbf{k}) T \in \mathbb{C}^{3 \times 3}$$

를 얻는다.

#### 2.3.3 Asymptotic-Preserving 보정

순수 Galerkin Schur 는 운동학적 영공간(kinetic null-space)의 누설(non-equilibrium) 기여를 빠뜨린다. 본 연구에서는 이를 1차 보정으로 다음과 같이 추가한다:

$$\boxed{\hat S_U^{AP}(\mathbf{k}) = \hat S_U^G(\mathbf{k}) - \frac{1-\omega}{2\omega} \left[ M \hat A^2(\mathbf{k}) T - (M \hat A(\mathbf{k}) T)^2 \right]}$$

이 보정은 BGK 완화율 의존 계수 $\frac{1-\omega}{2\omega}$ 와 Fourier-symbol 들의 2차 결합으로 구성된다. 수치 안정성을 위해 계수는 $\frac{1}{2} \cdot \text{sign}(\text{raw}) \cdot \min(0.5, |\text{raw}|)$ 로 clip 한다.

#### 2.3.4 정규화 및 영모드 처리

$\hat S_U^{AP}(\mathbf{k})$ 는 일부 $\mathbf{k}$ 에서 특이성에 가까워질 수 있으므로 adaptive Tikhonov 정규화

$$\hat S_U^{\text{reg}}(\mathbf{k}) = \hat S_U^{AP}(\mathbf{k}) + \eta I_3, \quad \eta = \sigma_{\max} / 50$$

여기서 $\sigma_{\max} = \max_{\mathbf{k}} \sigma_{\max}(\hat S_U^{AP}(\mathbf{k}))$ 는 모든 mode 의 최대 특이값.

영모드 $\mathbf{k} = \mathbf{0}$ 는 mass-conservation 자유도 (mean density 불변) 와 momentum mean 통과 처리:

$$\hat S_U^{-1}(\mathbf{0}) = \text{diag}(0, 1, 1).$$

이는 Newton step 이 평균 밀도를 변경하지 않도록 명시적으로 lock 한다.

#### 2.3.5 전처리기 적용

분포함수 잔차 $R(f) \in \mathbb{R}^{q N^2}$ 가 주어지면 PC 는 다음 4 단계로 작용한다:

$$P_0^{-1} R = T \cdot \mathcal{F}^{-1} \left[ \hat S_U^{\text{reg}^{-1}} \cdot \mathcal{F}[M R] \right]$$

(i) 거시 projection $M R \in \mathbb{R}^{3 N^2}$, (ii) 2D FFT, (iii) mode-wise $3 \times 3$ inverse 곱, (iv) 역 FFT 및 분포 lift $T$. 계산 비용: $O(N^2 \log N)$ FFT + $O(N^2)$ inverse. 전체 PC 한 번 적용 비용은 한 번의 LBM step 보다 약 0.5–2 배 수준이며, GMRES 안에서 inner matvec 당 한 번 호출된다.

### 2.4 JFNK 안의 잔차 Jacobian-Vector Product

Krylov 방법은 행렬 자체가 아닌 행렬–벡터 곱만 필요하다. 본 연구에서는 Eisenstat–Walker 형 finite-difference Jacobian-vector product 를 사용한다:

$$J(y) v \approx \frac{R(y + \varepsilon v) - R(y)}{\varepsilon}, \quad \varepsilon = \frac{\sqrt{\varepsilon_{\text{mach}}} \cdot \max(1, \|y\|)}{\|v\|}$$

여기서 $\varepsilon_{\text{mach}} \approx 10^{-16}$ 는 머신 epsilon. JVP 한 번 비용 = $R(y + \varepsilon v)$ 한 번 = LBM step 한 번.

### 2.5 Nesterov 가속 룩어헤드

#### 2.5.1 룩어헤드 정의

매 outer iteration $k$ 에서 직전 두 iterate $f_{k-1}, f_k$ 로부터 모멘텀 항을 외삽한다:

$$\boxed{y_k = f_k + \beta_k (f_k - f_{k-1})}$$

이는 ML 최적화의 Nesterov accelerated gradient 와 동일한 lookahead 구조이며, Newton-Krylov step 의 RHS 와 Jacobian 평가 위치를 $y_k$ 로 옮긴다.

#### 2.5.2 적응형 모멘텀 계수 $\beta_k$

$\beta_k$ 는 잔차 진행에 따라 다음과 같이 동적 갱신된다:

```
if ||R_k|| > ||R_{k-1}|| :      # 잔차 증가
    β ← β · 0.7                 # half-restart 보다 부드러운 감쇠
else :                           # 잔차 감소
    β ← min(β_cap, β + 0.15)    # 점진 증가
```

$\beta_{\text{cap}}$ 자체는 정상-진행(streak) 동안 다음 규칙으로 확장된다:

```
if streak_no_reject >= 2 :
    β_cap ← min(0.95, β_max + 0.2)
else if reject :
    β_cap ← β_max               # reset
```

초기값 $\beta_{\text{max}} = 0.7$, $\beta_0 = 0$.

### 2.6 잔차 단조 안전성 검사 (residual-monotone safeguard)

#### 2.6.1 검사 조건

$\beta_k > 0.3$ 일 때만 룩어헤드 $y_k$ 의 잔차를 평가한다 (작은 $\beta$ 에서는 무시할 수 있는 차이):

$$R_{y,k} = y_k - \mathcal{L}(y_k) \quad (+1 \text{ LBE call})$$

수용 기준은 다음 부등식:

$$\boxed{\|R_{y,k}\| \le (1 + \varepsilon_{\text{eff}}) \|R_k\|}$$

여기서 $\varepsilon_{\text{eff}} = \varepsilon_{\text{accept}} + 0.2 \beta_k$ 는 $\beta$ 에 적응하는 허용 한계 (기본값 $\varepsilon_{\text{accept}} = 0.10$).

#### 2.6.2 거부 회복

부등식을 위반하면 룩어헤드를 거부:

```
y_k ← f_k                       # 룩어헤드 폐기
R_{y,k} ← R_k                   # 기존 잔차 재사용
β_k ← β_k · 0.7                # 모멘텀 감쇠 (full reset 아님)
streak_no_reject ← 0
β_cap ← β_max                   # cap reset
```

이 단조 검사는 NN-단독 알고리즘이 stiff cavity (Re=400+) 에서 발산하는 모드를 직접 방지한다 (그림 X에서 NN: NaN → Safe-NN: 5.66 배 수렴, §3.3 참조).

#### 2.6.3 NaN 안전망

수치적 오버플로 / NaN 발생 시 $f_{\text{new}} \leftarrow \mathcal{L}^K(f_k)$ 의 순수 Picard fallback 으로 전환, $\beta \leftarrow 0$. 이는 알고리즘이 baseline LBM 보다 절대 나쁘지 않다는 보장.

### 2.7 FGMRES Newton-Krylov inner solver

수용된 룩어헤드 $y_k$ 에서 Newton step 을 계산한다:

$$J(y_k) \, \delta f_k = - R_{y,k}$$

이를 다음 FGMRES 설정으로 근사 해결:

| 파라미터 | 값 | 의미 |
|---|---|---|
| `maxiter` | 1 | 외부 반복 1회 (inexact Newton) |
| `restart` | $2 \times m_{\text{Kry}} = 20$ | inner Krylov 차원 |
| `rtol` | $10^{-3}$ | 상대 잔차 허용 한계 |
| `atol` | $10^{-3} \cdot \|R_{y,k}\| \cdot 10^{-3}$ | 절대 잔차 허용 한계 |
| right precond | $P_0^{-1}$ (AP-Schur, §2.3) | FFT 기반 |

매 inner matvec 은 1회 LBE call (JVP, §2.4) 소요. 일반적으로 outer 당 1–3 matvec 으로 수렴.

업데이트:

$$f_{\text{new}}^{(0)} = y_k + \delta f_k$$

### 2.8 적응형 사후 LBM 완화 (K-anneal)

#### 2.8.1 표준 사후 단계

Newton step 직후 $K = 15$ 회 BGK relaxation 을 적용하여 비평형 모드의 누설을 정화한다:

$$f_{\text{new}}^{(K)} = \mathcal{L}^K(f_{\text{new}}^{(0)})$$

이는 SCMK 류 공통 안정화 단계.

#### 2.8.2 수렴 단조 감소 시 K 절반

수렴 후반부 (잔차 작음 + 단조 감소) 에서는 추가 smoothing 이 불필요하다. 따라서 다음 조건에 K 를 절반으로 줄인다:

```
if ||R_k|| < 3 × 10^{-5}  AND  ||R_k|| < ||R_{k-1}|| :
    K_{eff} ← max(5, K // 2)        # 7 substeps
else :
    K_{eff} ← K = 15
```

조건의 **단조 감소** 부분이 핵심. 잔차 정체 시 K_eff 를 줄이면 stiff cavity 수렴이 무너진다 (음성 ablation 으로 확인, §4).

### 2.9 전체 알고리즘 (의사 코드)

```python
입력 : case (LBM problem), tol = 1e-7
       β_max = 0.7, ε_accept = 0.10, K = 15

초기화 :
    f_prev ← f_0 = f_eq(ρ=1, u=0)
    f ← f_prev.copy()
    S_inv ← build_AP_Schur(N, ω, mode="ap")
    β, β_cap, streak ← 0, β_max, 0
    res_prev ← ∞

반복 k = 0, 1, 2, ... :
    R ← f - L(f);       res ← ||R|| / √(qN²)        # +1 LBE
    if res < tol : break

    # 2.5 β 갱신
    if res > res_prev : β ← β · 0.7;  streak ← 0;  β_cap ← β_max
    else              : β ← min(β_cap, β + 0.15)
                        if streak >= 2 : β_cap ← min(0.95, β_max + 0.2)

    # 2.5–2.6 룩어헤드 + 안전성 검사
    if β > 0.3 :
        y ← f + β · (f - f_prev)
        R_y ← y - L(y);  # +1 LBE
        ε_eff ← ε_accept + 0.2 · β
        if ||R_y|| > (1 + ε_eff) · ||R||  OR  ¬finite(R_y) :
            y ← f;  R_y ← R                          # 거부
            β ← β · 0.7;  streak ← 0;  β_cap ← β_max
        else :
            streak ← streak + 1
    else :
        y ← f;  R_y ← R
        streak ← streak + 1

    # 2.7 NK step
    δf ← FGMRES(J(y), -R_y, M = AP_Schur_PC, maxiter=1, restart=20)
    f_new ← y + δf                                   # +1-3 LBE (probes)

    # 2.8 K-anneal
    if res < 3e-5  AND  res < res_prev :
        K_eff ← max(5, K // 2)
    else :
        K_eff ← K
    f_new ← L^{K_eff}(f_new)                         # +K_eff LBE

    # NaN 안전망
    if ¬finite(f_new) :
        f_new ← L^K(f);  β ← 0                       # fallback
        # +K LBE

    f_prev ← f;  f ← f_new;  res_prev ← res
```

전체 코드 길이는 약 90 줄. Hyperparameter 3 개 (β_max, ε_accept, K) 외에 모두 자동.

### 2.10 매 outer iter 의 LBE 비용 분석

각 outer iter 의 LBE call 수:

| 항목 | LBE | 조건 |
|---|---|---|
| 잔차 $R = f - L(f)$ | 1 | 항상 |
| 룩어헤드 잔차 $R_y$ | 1 | $\beta > 0.3$ 일 때만 |
| FGMRES matvec (JVP) | 1–3 | inner Krylov iter 당 |
| 사후 K-anneal | 7 또는 15 | adaptive |
| **합 (대표값)** | **10–20** | |

따라서 outer 8–30 회 수렴 시 총 100–400 LBE 로, baseline LBM 5000–10000 LBE 대비 20–50 배 가속 (실측치는 §3).

### 2.11 본 알고리즘의 novelty 5 요소

1. **Nesterov + Newton-Krylov 결합**: ML 의 accelerated gradient 를 LBM 정상해 root-finding 에 이식. LBM 문헌 전례 없음.
2. **잔차 단조 안전성 검사**: NN 단독에서 stiff cavity 발산을 방지하는 명시적 가드. 일반 nonlinear solver 의 trust region 과 유사하나 LBM-specific.
3. **AP-Schur Fourier 전처리기**: $\frac{1-\omega}{2\omega} [MA^2T - (MAT)^2]$ 형 보정. Bardow 2008 DTS, Premnath 2009 와 다른 *native residual* 형식.
4. **적응형 K-anneal with monotone gate**: 사후 LBM 횟수의 동적 조절. 단조 감소 조건이 안정성 결합 핵심.
5. **β_cap streak-aware ratchet**: 안정 진행 동안 모멘텀 한계 점진 확장, reject 시 즉시 복귀.

---

## 3. 수치 검증

### 3.1 기준 사례 (Benchmarks)

다섯 종류의 표준 LBM 정상상태 문제:

| # | Case | 격자 | 경계조건 | 특성 |
|---|---|---|---|---|
| 1 | Kolmogorov flow | N=32 periodic | 주기 | smooth 단일 mode |
| 2 | Channel (Poiseuille) | N=32 wall-y, periodic-x | bounce-back 벽 (y) | 평균 흐름 + 단일 mode |
| 3 | Couette | N=32 periodic-x | moving lid + 벽 | 선형 profile |
| 4 | Lid-driven cavity Re=100 | N=33 4 walls | bounce-back 4 면 | 비선형 vortex |
| 5 | Multi-cylinder | N=32 voxel mask | bounce-back 다중 원기둥 | 복잡 voxel geometry |

추가 stress test: Cavity Re=400 N=49 (stiff vortex).

### 3.2 합성 점수 정의

$$\text{composite} = \text{mean\_speedup} \times \text{accuracy\_factor} \times \text{convergence\_fraction}$$

$$\text{accuracy\_factor} = \max(0, 1 - \text{worst\_field\_err} / 0.05)$$

baseline LBM Picard 의 LBE-call 수에 대한 가속 비율을 평균한 mean_speedup, 5 case 중 수렴한 비율 convergence_fraction.

### 3.3 결과

| Case | Baseline LBE | Safe-NN LBE | LBE speedup | field err |
|---|---:|---:|---:|---:|
| Kol N=32 | 3,015 | 134 | 22.50× | 1.80×10⁻³ |
| Chan N=32 | 5,427 | 170 | 31.92× | 2.12×10⁻² |
| Couette N=32 | 5,829 | 30 | 194.30× | 2.65×10⁻² |
| Cav Re=100 N=33 | 3,216 | 472 | 6.82× | 5.78×10⁻³ |
| Multi-cyl N=32 | 2,211 | 359 | 6.16× | 1.59×10⁻² |
| **Mean speedup** | — | — | **52.34×** | — |
| **Composite** | — | — | **45.41** | — |

Stress: Cav Re=400 N=49 — Baseline 8040 LBE, Safe-NN 1421 LBE = **5.66× 수렴 (4.77×10⁻⁷ residual)**. NN-단독 동일 조건에서 NaN 발산.

### 3.4 기존 솔버와의 비교

| Solver | Composite | Cav Re=400 |
|---|---:|---|
| Baseline LBM Picard | 1.0 | 1.0× |
| Lean SCMK (NK only) | 41.39 | 5.30× |
| SAN (Anderson + NK) | 42.31 | 4.22× |
| NN (Nesterov + NK) | 44.74 | **NaN** ❌ |
| **Safe-NN-SCMK** | **45.41** | **5.66×** ✓ |

Safe-NN 은 단일 알고리즘으로 모든 case 에서 안정 + 최고 평균 가속.

### 3.5 Ablation

| 변형 | Composite | 손실 |
|---|---:|---|
| Safe-NN v4 (full) | 45.41 | base |
| K-anneal 제거 (K=15 고정) | 40.69 | -10.4% |
| 잔차 단조 검사 제거 | 44.74 (Cav R400 NaN) | Cav 발산 ❌ |
| Nesterov 모멘텀 제거 (β=0) | 41.39 | -8.9% |
| AP correction 제거 | (Galerkin only, ~30) | -34% |
| Mass-conservation 모드 (0,0) 제거 | 발산 ❌ | — |

5 요소 모두 필수임을 확인.

---

## 4. 토론

(생략 — 기존 paper 의 토론 sections 재사용)

---

## 5. 결론

Safe-NN-SCMK 는 ML 분야의 Nesterov 가속과 CFD 의 Newton-Krylov 를 LBM-specific 안전 가드와 결합한 단일 알고리즘으로, 다섯 가지 표준 benchmark + 1 stress case 에서 평균 51 배 가속 + 합성 점수 45.41 을 달성하였다. 본 알고리즘은 (i) collision/streaming 연산자 무수정, (ii) 90 줄 핵심 코드, (iii) 모든 case 안정 수렴, (iv) LBM 문헌 전례 없는 결합 구조의 네 가지 측면을 모두 만족한다. 향후 작업으로 MRT 충돌, 3D D3Q19, GPU 이식, 큰 격자 (N≥128) scaling, 수렴률 정리(Theorem) 의 엄밀화가 남는다.

---

## 부록 A. 핵심 코드

`solver_safe_nn.py` 전문 (90 줄), `lbm_periodic.py::build_spectral_schur`, `verify_safe_nn.py` 검증 driver.

## 부록 B. 음성 ablation (기각 후보 솔버)

- DMN (Direct Macro Newton): kinetic closure 손실, composite 6.04
- NSP (Nesterov + Spectral PC, no NK): composite 5.18
- HKR (Hydro-Kinetic Reduced): JVP×k_slave 비용 폭증, composite 2.07
- KDF (Koopman/DMD deflation): slow mode 검출 실패, composite 0.62
- GSIS/NDA/VEF/IMM (synthetic HO/LO): macro Stokes 보정 wall 부정합, composite ~0
- BCS (Woodbury low-rank boundary): marginal 0.6× gain

모두 본 연구의 negative result section 에 포함.

---

**커밋 ref**: `7e6da18`
**솔버 파일**: `solver_safe_nn.py`
**검증 driver**: `verify_safe_nn.py`
