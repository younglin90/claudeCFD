# Segregated Implicit Solver 명세서

**기반 논문**: Denner, Xiao & van Wachem, *"Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation"*, J. Comput. Phys. 367 (2018) 192–234.

---

## 1. 개요

1D 전속도 영역(비압축성~압축성) 다상 압축성 유한체적 솔버.
분리형(segregated) 시간 전진: **VOF 명시적 이송** → **3N Newton 암시적 (p, u, T)** 순차 수행.

| 항목 | 내용 |
|------|------|
| 지배방정식 | 혼합물 연속, 운동량, 에너지 (보존형 Euler) |
| 시간 이산 | Backward Euler (1차) |
| 공간 이산 | 1차 풍상(upwind) + MWI 면속도 |
| 계면 처리 | ACID (Acoustically-Conservative Interface Discretisation) |
| VOF 이송 | 명시적 CICSAM (Hyper-C) + 자동 서브스텝 |
| 비선형 솔버 | Full Newton (매 반복마다 ρ̃, θ 갱신) |
| 선형 솔버 | scipy spsolve (직접법) |
| 상태방정식 | NASG, Stiffened Gas, Ideal Gas (EOS 클래스 인터페이스) |

---

## 2. 지배방정식

### 2.1 연속방정식 (압력방정식)

$$
\frac{\rho^{n+1} - \rho^n}{\Delta t} + \frac{1}{\Delta x}\left(\tilde{\rho}_R\,\vartheta_R - \tilde{\rho}_L\,\vartheta_L\right) = 0
$$

- $\rho^{n+1} = \rho(p^{n+1}, T^{n+1}, \psi_{\rm new})$: 새 시간 밀도 (EOS forward)
- $\rho^n = \rho(p^n, T^n, \psi_{\rm new})$: 구 시간 밀도 (**ACID**: 구 (p,T)에 신 ψ 적용)
- $\tilde{\rho}_f$: ACID 면 밀도 (§4.1)
- $\vartheta_f$: MWI 면속도 (§4.2)

### 2.2 운동량방정식

$$
\frac{(\rho u)^{n+1} - (\rho u)^n}{\Delta t} + \frac{1}{\Delta x}\left(\tilde{\rho}_R\,\vartheta_R\,\tilde{u}_R - \tilde{\rho}_L\,\vartheta_L\,\tilde{u}_L\right) + \frac{p_R^{\rm face} - p_L^{\rm face}}{\Delta x} = 0
$$

- $\tilde{u}_f$: 풍상 면속도
- $p_f^{\rm face} = \frac{1}{2}(p_i + p_{\rm neighbor})$: 산술평균 면 압력

### 2.3 에너지방정식

**(p, u, T) 모드 — 기본 설정:**

$$
\frac{(\rho h_t)^{n+1} - (\rho h_t)^n}{\Delta t} + \frac{1}{\Delta x}\left(\tilde{\rho}_R\,\vartheta_R\,\tilde{h}_{t,R} - \tilde{\rho}_L\,\vartheta_L\,\tilde{h}_{t,L}\right) = \frac{p^{n+1} - p^n}{\Delta t}
$$

여기서 비총엔탈피:

$$
h_t = c_p T + b\,p + \eta + \tfrac{1}{2}u^2
$$

- $c_p = \gamma k_v$ (NASG), $b$ = covolume, $\eta$ = 에너지 상수
- 혼합물: $c_{p,\rm mix} = \psi\,c_{p,1} + (1-\psi)\,c_{p,2}$ (체적분율 가중)

**(p, u, h) 모드 — 대안:**

$$
\frac{(\rho h)^{n+1} - (\rho h)^n}{\Delta t} + \frac{1}{\Delta x}\left(\tilde{\rho}_R\,\vartheta_R\,\tilde{h}_R - \tilde{\rho}_L\,\vartheta_L\,\tilde{h}_L\right) = \frac{p^{n+1} - p^n}{\Delta t}
$$

### 2.4 VOF 이송방정식

$$
\frac{\psi^{n+1} - \psi^n}{\Delta t} + \frac{1}{\Delta x}\left(\vartheta_R\,\psi_{f,R} - \vartheta_L\,\psi_{f,L}\right) - (\psi + K)\,\frac{\vartheta_R - \vartheta_L}{\Delta x} = 0
$$

- $\psi_{f}$: CICSAM (Hyper-C) 면값 (§5)
- $K$: 압축성 보정 인자 (§5.2)
- **명시적(explicit) 시간 전진**, 자동 서브스텝

---

## 3. 시간 전진 알고리즘

하나의 시간 스텝에서 두 단계를 순차 수행:

```
┌─────────────────────────────────────────────────┐
│  Step A: VOF 명시적 이송                          │
│    psi_new = CICSAM(psi_n, u_n, dt)             │
│    (자동 서브스텝, K factor, 압축항 옵션)           │
├─────────────────────────────────────────────────┤
│  Step B: 3N Newton 암시적 (p, u, T)              │
│    psi = psi_new (고정)                           │
│    rho_old = rho(p_n, T_n, psi_new)  ← ACID     │
│    h_old   = h(p_n, T_n, psi_new)               │
│                                                  │
│    for k = 0, 1, 2, ..., max_newton:             │
│      1. 혼합물 물성 계산 (rho_k, zeta_k, phi_k)   │
│      2. ACID 면밀도 rho_tilde 계산                │
│      3. MWI d_hat, 면속도 theta_k 계산            │
│      4. 3N 행렬 조립: A·x = b                     │
│      5. 잔차: r = b - A·x_k                      │
│      6. 선형 풀이: dx = solve(A, r)               │
│      7. 감쇠: omega, 갱신 p_k, u_k, T_k          │
│      8. 수렴 판정                                  │
│    end for                                        │
└─────────────────────────────────────────────────┘
```

### 3.1 시간 스텝 결정

음향 CFL 조건:

$$
\Delta t = \text{CFL} \cdot \frac{\Delta x}{\max_i\left(|u_i| + c_{\rm mix,i}\right)}
$$

여기서 $c_{\rm mix}$는 Wood의 혼합음속:

$$
\frac{1}{\rho\,c_{\rm Wood}^2} = \frac{\psi}{\rho_1 c_1^2} + \frac{1-\psi}{\rho_2 c_2^2}
$$

---

## 4. 공간 이산화

### 4.1 ACID 면밀도 (Acoustically-Conservative Interface Discretisation)

**핵심 원리**: 셀 $i$의 이산 스텐실 내에서 모든 체적분율이 $\psi_i$와 같다고 가정. 이웃 셀의 (p, T)에 현재 셀의 ψ를 적용하여 면밀도를 계산.

**풍상 선택**:
- $\vartheta_R \ge 0$ → 면 R은 셀 $i$의 (p, T) 사용 (풍상 = 셀 i)
- $\vartheta_R < 0$ → 면 R은 셀 $i_R$의 (p, T) 사용 (풍상 = 셀 iR)

**체적분율 혼합** (volume fraction mode):

$$
\tilde{\rho}_{f} = \psi_i \cdot \rho_1(p_{\rm up}, T_{\rm up}) + (1-\psi_i) \cdot \rho_2(p_{\rm up}, T_{\rm up})
$$

여기서 하첨자 "up"은 풍상 셀의 열역학 상태를 의미.

**질량분율 혼합** (mass fraction mode):

$$
\frac{1}{\tilde{\rho}_{f}} = \frac{Y_i}{\rho_1(p_{\rm up}, T_{\rm up})} + \frac{1-Y_i}{\rho_2(p_{\rm up}, T_{\rm up})}
$$

**ACID의 핵심 성질**: 균일장 ($p$, $T$ 일정, $u$ 일정)에서 체적분율에 관계없이 $\tilde{\rho}_f \cdot \vartheta = \rho \cdot u$ (정확), 따라서 잔차 = 0 → Abgrall 보존 만족.

### 4.2 MWI 면속도 (Momentum-Weighted Interpolation)

Denner 2018, Eq. 20:

$$
\vartheta_f = \bar{u}_f - \hat{d}_f \cdot \frac{p_R - p_L}{\Delta x} + \hat{d}_f \cdot \frac{\rho^*_{f,\rm old}}{\Delta t}\left(\vartheta_{f,\rm old} - \bar{u}_{f,\rm old}\right)
$$

각 항:

| 기호 | 정의 |
|------|------|
| $\bar{u}_f$ | 산술평균 면속도: $\frac{1}{2}(u_L + u_R)$ |
| $\hat{d}_f$ | Denner MWI 계수 (아래 참조) |
| $\rho^*_f$ | 조화평균 면밀도 |
| 3번째 항 | 과도(transient) 보정 — 이전 스텝 면속도 이력 사용 |

**MWI 계수** (Denner 2018):

$$
\hat{d}_f = \frac{\Delta x / e_L + \Delta x / e_R}{\Delta x / e_L + \Delta x / e_R + 2\rho^*_f / \Delta t}
$$

여기서 운동량 대각: $e_P = \rho_P / \Delta t$.

**조화평균 면밀도**:

$$
\rho^*_f = \frac{2\,\rho_L\,\rho_R}{\rho_L + \rho_R}
$$

대밀도비 환경에서 산술평균 대비 수치 안정성 우수 (Denner 2018 §3.2).

---

## 5. VOF 이송 상세

### 5.1 CICSAM (Hyper-C) 면값

각 면에서 Donor(풍상), Acceptor(하풍), UpUpwind(풍상의 풍상) 셀을 식별.

정규화 변수:

$$
\tilde{\psi}_D = \frac{\psi_D - \psi_{UU}}{\psi_A - \psi_{UU}}
$$

Hyper-C 면값:

$$
\tilde{\psi}_f = \begin{cases}
\min\!\left(\dfrac{\tilde{\psi}_D}{\text{Co}_f},\;1\right) & \text{if } 0 \le \tilde{\psi}_D \le 1 \\[6pt]
\tilde{\psi}_D & \text{otherwise (upwind fallback)}
\end{cases}
$$

역정규화:

$$
\psi_f = \psi_{UU} + \tilde{\psi}_f \cdot (\psi_A - \psi_{UU})
$$

최종 클립: $\psi_f \in [0, 1]$.

### 5.2 K Factor (압축성 보정)

Wood의 혼합 공식에서 유도 (Denner 2018 Eq. 11):

$$
K = \frac{\rho_b\,a_b^2 - \rho_a\,a_a^2}{\dfrac{\rho_a\,a_a^2}{1-\psi} + \dfrac{\rho_b\,a_b^2}{\psi}}
$$

등가 표현 (코드 구현):

$$
K_k = \psi_k \left(\frac{Z_k}{Z_{\rm mix}} - 1\right), \qquad Z_k = \rho_k c_k^2, \qquad Z_{\rm mix} = \sum_k \psi_k Z_k
$$

성질: $\sum_k K_k = 0$ (체적분율 합 보존).

비압축 유동($\nabla \cdot u = 0$)에서는 K항이 자동으로 사라짐.

### 5.3 압축항 (Anti-diffusion)

CICSAM의 수치 확산을 줄이기 위한 반확산항:

$$
\psi^{n+1} \mathrel{-}= \Delta t \cdot \nabla \cdot \left(C_k \,|u|\,\psi(1-\psi)\,\hat{n}\right)
$$

- $\hat{n} = \text{sign}(\nabla \psi)$: 계면 법선 방향
- $C_k$: 압축 계수 (기본 1.0)

**Zalesak FCT Limiter** (1979)로 $\psi \in [0,1]$ 강제:

$$
P^+ = \sum_{f:\,F_f>0} F_f, \quad P^- = \sum_{f:\,F_f<0} |F_f|
$$
$$
Q^+ = (1-\psi_i)/\Delta t, \quad Q^- = \psi_i/\Delta t
$$
$$
R^+ = \min(1,\,Q^+/P^+), \quad R^- = \min(1,\,Q^-/P^-)
$$
$$
C_f = \begin{cases} \min(R^+_L,\,R^-_R) & F_f > 0 \\ \min(R^-_L,\,R^+_R) & F_f < 0 \end{cases}
$$

### 5.4 자동 서브스텝

Courant 수 $\text{Co} = \max_f |u_f| \Delta t / \Delta x > 1$이면, 서브스텝 수 $N_{\rm sub} = \lceil \text{Co} \rceil$로 분할하여 VOF 안정성 확보.

---

## 6. Newton 선형화 (3N 시스템)

미지수 벡터: $\mathbf{x} = [p_0, \ldots, p_{N-1},\; u_0, \ldots, u_{N-1},\; T_0, \ldots, T_{N-1}]^T$ (또는 $h$ 대신 $T$)

행렬 형태: $\mathbf{A}\,\mathbf{x} = \mathbf{b}$, Newton 갱신: $\mathbf{A}\,\delta\mathbf{x} = \mathbf{b} - \mathbf{A}\,\mathbf{x}_k$

### 6.1 Full Newton ρ̃ (Denner 2018 Eq. 25, 29, 30)

**매 Newton 반복마다** ACID 면밀도 $\tilde{\rho}$, MWI 면속도 $\vartheta$를 현재 반복값 $(p_k, u_k, T_k)$에서 재계산. Picard (ρ̃ 고정) 대비 수렴 속도 우수, 극한 밀도비(1000:1)에서도 안정.

### 6.2 EOS 미분계수

혼합물 밀도의 원시변수 미분:

$$
\zeta_{\rm mix} = \frac{\partial \rho_{\rm mix}}{\partial p}\bigg|_T = \psi\,\zeta_1 + (1-\psi)\,\zeta_2
$$

$$
\phi_{\rm mix} = \frac{\partial \rho_{\rm mix}}{\partial T}\bigg|_p = \psi\,\phi_1 + (1-\psi)\,\phi_2
$$

면밀도 미분 (ACID, 풍상 p,T 사용):

$$
\zeta_f = \psi_i \cdot \zeta_1(p_{\rm up}, T_{\rm up}) + (1-\psi_i) \cdot \zeta_2(p_{\rm up}, T_{\rm up})
$$

$$
\phi_f = \psi_i \cdot \phi_1(p_{\rm up}, T_{\rm up}) + (1-\psi_i) \cdot \phi_2(p_{\rm up}, T_{\rm up})
$$

### 6.3 연속방정식 Jacobian

셀 $i$의 연속방정식에 대한 행렬 기여:

**시간항** ($\rho^{n+1}$ Newton 선형화):

$$
A[r_p, c_p^i] \mathrel{+}= \zeta_i / \Delta t, \qquad A[r_p, c_T^i] \mathrel{+}= \phi_i / \Delta t
$$
$$
b[r_p] \mathrel{+}= \rho_{\rm old} / \Delta t + (\zeta_i p_k - \rho_k)/\Delta t \quad (+\; \phi_i T_k / \Delta t \text{ for T-mode})
$$

**공간항 Term 1** — $\tilde{\rho}_k \cdot \vartheta^{n+1}$ (면밀도 고정, 면속도 암시적):

MWI 분해: $\vartheta_f = \frac{1}{2}(u_L + u_R) - \hat{d}_f (p_R - p_L)/\Delta x$

- 속도 산술평균 부분: $A[r_p, c_u^i] \mathrel{+}= \tilde{\rho}_R / (2\Delta x)$, $A[r_p, c_u^{iR}] \mathrel{+}= \tilde{\rho}_R / (2\Delta x)$
- 압력 라플라시안 부분: $A[r_p, c_p^i] \mathrel{+}= \tilde{\rho}_R \hat{d}_R / \Delta x^2$, $A[r_p, c_p^{iR}] \mathrel{-}= \tilde{\rho}_R \hat{d}_R / \Delta x^2$

좌측 면도 동일 구조, 부호 반대.

**공간항 Term 2** — $\tilde{\rho}^{n+1} \cdot \vartheta_k$ (면밀도 Newton, 면속도 고정):

$$
A[r_p, c_p^{\rm up}] \mathrel{+}= \zeta_f \cdot \vartheta_k / \Delta x
$$
$$
A[r_p, c_T^{\rm up}] \mathrel{+}= \phi_f \cdot \vartheta_k / \Delta x
$$

### 6.4 운동량방정식 Jacobian

Newton 곱: $(\rho u)^{n+1} \approx \rho_k u + \zeta \,\delta p \cdot u_k + \phi \,\delta T \cdot u_k$

**시간항**:

$$
A[r_u, c_u^i] \mathrel{+}= \rho_i / \Delta t, \qquad A[r_u, c_p^i] \mathrel{+}= \zeta_i u_i / \Delta t
$$

**대류항**: $\tilde{\rho}_f \vartheta_f u_f$ → 3개 기여:

1. 면밀도·면속도 고정, 풍상 $u$ 암시적
2. 면밀도 고정, MWI $\vartheta$ 암시적 (d̂ 압력 라플라시안 × $u_k$)
3. 면밀도 Newton ($\zeta_f \cdot \vartheta_k \cdot u_{\rm up} / \Delta x$)

**압력 구배**:

$$
A[r_u, c_p^{iR}] \mathrel{+}= 1/(2\Delta x), \qquad A[r_u, c_p^{iL}] \mathrel{-}= 1/(2\Delta x)
$$

### 6.5 에너지방정식 Jacobian (puT 모드)

Newton 곱 규칙: $d(\rho h_t)/dT = \rho_k c_p + h_{t,k} \phi$ (T-계수, 항상 양수)

**시간항**:

$$
A[r_T, c_T^i] \mathrel{+}= (\rho_k c_p + h_{t,k} \phi_i) / \Delta t
$$
$$
A[r_T, c_p^i] \mathrel{+}= (\rho_k b_{\rm mix} + h_{t,k} \zeta_i - 1) / \Delta t
$$
$$
A[r_T, c_u^i] \mathrel{+}= \rho_k u_k / \Delta t
$$

**대류항**: 풍상 $c_p T$를 암시적, 나머지 ACID 엔탈피 보정은 $\mathbf{b}$에 지연(deferred):

$$
\text{acid\_corr} = \left(H_{\rm acid} - \tilde{\rho}_f \cdot h_{t,\rm up}\right) \cdot \vartheta_f / \Delta x
$$

여기서 $H_{\rm acid} = \psi_i \rho_1 h_{t,1} + (1-\psi_i) \rho_2 h_{t,2}$ (ACID 가정 엔탈피).

### 6.6 감쇠 (Line Search)

적응적 감쇠 계수:

$$
\omega = \min\!\left(1,\;\frac{0.5\,p_{\rm ref}}{\max|\delta p|},\;\frac{c_{\max}}{\max|\delta u|},\;\frac{0.5\,T_{\rm ref}}{\max|\delta T|}\right)
$$

### 6.7 수렴 판정

$$
\max\!\left(\frac{\max|\omega\,\delta p|}{p_{\rm ref}},\;\frac{\max|\omega\,\delta u|}{u_{\rm ref}},\;\frac{\max|\omega\,\delta T|}{T_{\rm ref}}\right) < \epsilon_{\rm Newton}
$$

기본 $\epsilon_{\rm Newton} = 10^{-6}$.

---

## 7. 선형 시스템 풀이

### 7.1 행 스케일링

각 블록의 대표값으로 행 정규화:

- 압력 블록: $1/p_{\rm ref}$
- 속도 블록: $1/u_{\rm ref}$  
- 온도 블록: $1/T_{\rm ref}$ (또는 $h_{\rm ref}$)

### 7.2 직접 풀이

`scipy.sparse.linalg.spsolve` 사용. BiCGSTAB + Block-Jacobi 프리컨디셔너도 구현되어 있으나, 직접법이 기본.

---

## 8. 상태방정식 (EOS)

### 8.1 EOS 클래스 인터페이스

모든 EOS는 다음 10개 메서드를 제공:

| 메서드 | 수식 | 설명 |
|--------|------|------|
| `rho(p, T)` | $\rho(p, T)$ | 밀도 |
| `h(p, T)` | $h(p, T)$ | 비엔탈피 |
| `c(p, T)` | $c(p, T)$ | 음속 |
| `cp(p, T)` | $c_p$ | 정압 비열 |
| `dh_dp(p, T)` | $\partial h / \partial p$ | 엔탈피의 압력 미분 |
| `drho_dp(p, T)` | $\zeta = \partial \rho / \partial p$ | 밀도의 압력 미분 |
| `drho_dT(p, T)` | $\phi = \partial \rho / \partial T$ | 밀도의 온도 미분 |
| `e_vol(p, T)` | $\rho e$ | 체적 내부에너지 밀도 |
| `de_vol_dp(p, T)` | $\partial(\rho e)/\partial p$ | |
| `de_vol_dT(p, T)` | $\partial(\rho e)/\partial T$ | |

### 8.2 NASG (Noble-Abel Stiffened Gas)

$$
\rho = \frac{p + p_\infty}{k_v T (\gamma - 1) + b(p + p_\infty)}
$$

$$
h = \gamma k_v T + bp + \eta
$$

$$
c = \sqrt{\frac{\gamma(p + p_\infty)}{\rho(1 - b\rho)}}
$$

파라미터: $\gamma$, $p_\infty$, $b$, $k_v$, $\eta$

### 8.3 Stiffened Gas (Denner convention)

$$
\rho = \frac{p + \gamma \Pi}{(\gamma - 1) c_v T}
$$

$$
h = c_p T + q, \qquad c_p = \gamma c_v
$$

파라미터: $\gamma$, $\Pi$, $c_v$, $q$

### 8.4 EOS 역산 (Inversion)

보존변수 $(\rho, \rho e)$로부터 $(p, T)$ 복원이 필요한 경우 (5eq 보존 솔버 등), 2×2 Newton 반복:

$$
\begin{cases}
f_1 = \psi \rho_1(p,T) + (1-\psi)\rho_2(p,T) - \rho_{\rm target} = 0 \\
f_2 = \psi (\rho e)_1(p,T) + (1-\psi)(\rho e)_2(p,T) - (\rho e)_{\rm target} = 0
\end{cases}
$$

수렴 조건: 상대 잔차 $< 10^{-10}$, 최대 50회.

---

## 9. 경계조건

### 9.1 Ghost Cell 방식

2층 고스트 셀 사용 (CICSAM에 UU 셀 필요).

| 경계 유형 | 스칼라 (p, T, ψ) | 속도 (u) |
|-----------|------------------|----------|
| `periodic` | 반대편 내부셀 복사 | 반대편 내부셀 복사 |
| `transmissive` | 최근접 내부셀 복사 | 최근접 내부셀 복사 |
| `wall` | 최근접 내부셀 복사 | 부호 반전 (no-slip) |

---

## 10. 발산 판정

시간 전진 루프에서 매 스텝 6가지 기준 확인:

1. NaN/Inf 발생
2. 압력 범위 이탈: $p < p_{\rm floor}$ 또는 $p > 10^{15}$ Pa
3. 온도 범위 이탈: $T < T_{\rm floor}$ 또는 $T > 10^{10}$ K
4. 속도 폭발: $|u| > 10 \cdot c_{\rm max}$
5. 음속 폭발: $c > 10^8$ m/s
6. 시간 스텝 붕괴: $\Delta t < 10^{-20}$ s

---

## 11. 설정 옵션 요약

| 키 | 기본값 | 설명 |
|----|--------|------|
| `variable_set` | `'puT'` | `'puh'` 또는 `'puT'` |
| `vof_type` | `'volume'` | `'volume'` 또는 `'mass'` |
| `use_K` | `False` | 압축성 VOF K factor 활성화 |
| `use_compress` | `False` | 반확산 압축항 활성화 |
| `coupled` | `False` | `True`면 4N coupled (미완성) |
| `five_eq` | `False` | `True`면 5-equation 솔버 (미완성) |
| `CFL` | 0.5 | 음향 CFL 수 |
| `max_newton` | 50 | Newton 최대 반복수 |
| `newton_tol` | $10^{-6}$ | Newton 수렴 허용오차 |
| `bc_left`, `bc_right` | — | `'periodic'`, `'transmissive'`, `'wall'` |

---

## 12. 검증 이력 요약

### Phase 1: Abgrall Advection (균일 압력·속도장 계면 이송)

11개 설정 조합 모두 PASS. 대표 결과:

| 설정 | err(p) | err(u) | err(E) |
|------|--------|--------|--------|
| seg vol + puT | 3.2e-15 | 1.3e-13 | 1.3e-14 |
| seg vol + K + comp | 1.2e-15 | 7.8e-14 | 6.6e-15 |

### Phase 2: Gas-Liquid Shock Tube (Denner 2018 Fig. 26)

- Air (1 GPa) vs Water (10 kPa), N=200, CFL=0.5
- 117 스텝으로 $t_{\rm end} = 2.4 \times 10^{-4}$ s 완주
- 3파 구조 (팽창파, 접촉면, 충격파) 명확 식별
- Denner 2018 Fig. 26과 정성적 일치

---

## 13. 제한사항

1. **VOF 명시적**: 체적분율은 명시적으로 이송되어, 음향 CFL 제한을 받음
2. **1차 공간 정확도**: 풍상 스킴 사용, 2차 확장 미구현
3. **1D 전용**: 다차원 확장 미구현
4. **BDF1**: 시간 1차 정확도, BDF2 미구현
5. **Coupled 4N/5eq**: 완전 결합 암시적 솔버는 α/ζ 조건수 문제로 미완성

---

## 14. 파일 구조

```
solver/denner_1d/
├── main.py              # run() 진입점, 시간 루프
├── solver_a.py          # step() 디스패치, Newton 루프
├── assembly.py          # 3N/4N 행렬 조립, 선형 풀이
├── flux/
│   └── mwi.py           # ACID 면밀도, 조화평균, MWI d_hat
├── vof_cn.py            # VOF 명시적 이송, K factor, 압축항
├── interface/
│   └── cicsam.py        # CICSAM Hyper-C 면값
├── boundary.py          # Ghost cell 경계조건
├── timestepping.py      # 음향 CFL 시간 스텝
├── eos/
│   ├── eos_class.py     # EOS 클래스 (NASG, SG), create_eos()
│   ├── invert.py        # (ρ, ρe) → (p, T) 역산
│   └── base.py          # 혼합물 물성 계산 유틸리티
├── solver_5eq.py        # 5-equation 솔버 (미완성)
└── assembly_5eq.py      # 5-equation 조립 (미완성)
```
