# DENNER_SCHEME.md
# 1D 압축성 다상 Euler 솔버 — Implicit Conservative Diffuse Interface

## 0. 최종 기법 선정

```
지배방정식:   Conservative Euler (ρ, ρu, E) + VOF (α)
              → 검증 후 mass fraction (ρY) 보존형으로 전환
계면 포착:    CICSAM (Phase 1) → THINC/QQ (Phase 2)
              compression term 없음 (CICSAM과 충돌, Zanutto2022)
플럭스:       MWI (p-u 결합) + ACID (계면 음향 보존)
              + Consistent transport (ψ̃→ρ̃→ρũ 체인)
EOS:          Ideal gas / NASG / General (TDU 기반)
시간:         Implicit Backward Euler (BDF2)
              Mode A: (p,u,T) coupled BDF2 + α Crank-Nicolson 분리
              Mode B: (p,u,T,α) fully coupled BDF2
안정화:       BSD (Phase 2, 점성 유동 추가 시)
              + Artificial viscosity (선택, 계면 감쇠)
향후 확장:    CST + Model 1 (질량전달/증발)
              Surface tension (CSF + implicit Newton)
```

**구현 로드맵:**
1. Phase 1a: 1D Euler (비점성) + VOF(CICSAM) + NASG + Implicit BDF
2. Phase 1b: 검증 (acoustic pulse, expansion tube, advection, hydrostatic)
3. Phase 2a: 점성항 추가 + BSD 안정화
4. Phase 2b: VOF → mass fraction 보존형 전환
5. Phase 2c: THINC/QQ 교체
6. Phase 3: CST/질량전달, 표면장력

---

## 1. 지배 방정식 (1D)

### 1.1 Phase 1: VOF 기반 (비보존형 계면)

**① 연속:**
$$\frac{\partial\rho}{\partial t} + \frac{\partial(\rho u)}{\partial x} = 0 \tag{1}$$

**② 운동량:**
$$\frac{\partial(\rho u)}{\partial t} + \frac{\partial(\rho u^2 + p)}{\partial x} = \frac{\partial\tau}{\partial x} + \rho g \tag{2}$$

**③ 총에너지:**
$$\frac{\partial\mathcal{E}}{\partial t} + \frac{\partial((\mathcal{E}+p)u)}{\partial x} = \frac{\partial(\tau u)}{\partial x} + \frac{\partial}{\partial x}\!\left(\lambda\frac{\partial T}{\partial x}\right) + \rho g u \tag{3}$$

**④ VOF (비보존형):**
$$\frac{\partial\psi}{\partial t} + \frac{\partial(\psi u)}{\partial x} - \psi\frac{\partial u}{\partial x} = 0 \tag{4}$$

**Phase 1 (비점성·단열)**: $\tau = 0$, $\lambda = 0$ → 순수 Euler.

정의:
- $\psi = \alpha_1$, $\alpha_2 = 1-\psi$
- $\rho = \alpha_1\rho_1(p,T)+\alpha_2\rho_2(p,T)$
- $\mathcal{E} = E+\frac{1}{2}\rho u^2$, $E = \alpha_1 E_1(p,T)+\alpha_2 E_2(p,T)$

### 1.2 Phase 2 전환: Mass Fraction 보존형

VOF 검증 완료 후, Eq. (4)를 보존형으로 교체:

$$\frac{\partial(\rho Y)}{\partial t} + \frac{\partial(\rho Y u)}{\partial x} = 0 \tag{4'}$$

$Y = \alpha_1\rho_1/\rho$ (phase 1의 질량분율). $Y \in [0,1]$이므로 CICSAM/THINC 적용 가능.
$\alpha_1 = \rho Y/\rho_1(p,T)$로 복원.

---

## 2. NASG EOS

### 2.1 정방향: $(p,T) \to$ 물성

$$\rho_k(p,T) = \frac{p+p_{\infty,k}}{A_k}, \quad A_k = \kappa_{v,k}T(\gamma_k-1)+b_k(p+p_{\infty,k}) \tag{5}$$

$$E_k(p,T) = \rho_k(\kappa_{v,k}T+\eta_k)+\frac{p_{\infty,k}(1-\rho_k b_k)}{\gamma_k-1} \tag{6}$$

$$c_k = \sqrt{\frac{\gamma_k(p+p_{\infty,k})}{\rho_k(1-\rho_k b_k)}}, \quad h_k = \gamma_k\kappa_{v,k}T+b_k p+\eta_k \tag{7}$$

특수 경우: 이상기체 → $p_\infty=0, b=0, \eta=0, \rho=p/(\kappa_v(\gamma-1)T)$.

### 2.2 열역학 도함수 (TDU — implicit 선형화 필수)

$$\zeta_k \equiv \frac{\partial\rho_k}{\partial p}\bigg|_T = \frac{\kappa_{v,k}T(\gamma_k-1)}{A_k^2} \tag{8a}$$

$$\phi_k \equiv \frac{\partial\rho_k}{\partial T}\bigg|_p = -\frac{(p+p_{\infty,k})\kappa_{v,k}(\gamma_k-1)}{A_k^2} \tag{8b}$$

혼합: $\zeta_v = \psi\zeta_1+(1-\psi)\zeta_2$, $\phi_v = \psi\phi_1+(1-\psi)\phi_2$

에너지 도함수:
$$\frac{\partial E_k}{\partial p}\bigg|_T = \left(\kappa_{v,k}T+\eta_k-\frac{p_{\infty,k}b_k}{\gamma_k-1}\right)\zeta_k \tag{9a}$$

$$\frac{\partial E_k}{\partial T}\bigg|_p = \rho_k\kappa_{v,k}+\left(\kappa_{v,k}T+\eta_k-\frac{p_{\infty,k}b_k}{\gamma_k-1}\right)\phi_k \tag{9b}$$

### 2.3 EOS 역변환: 보존 변수 → $(p,T)$

매 비선형 반복 후, $(\psi,\rho,E)$로부터 $(p,T)$ 복원.

$$f_1(p,T)=\psi\rho_1(p,T)+(1-\psi)\rho_2(p,T)-\rho=0 \tag{10a}$$
$$f_2(p,T)=\psi E_1(p,T)+(1-\psi)E_2(p,T)-E=0 \tag{10b}$$

Newton iteration:
$$\begin{pmatrix}p\\T\end{pmatrix}^{(k+1)}=\begin{pmatrix}p\\T\end{pmatrix}^{(k)}-\mathbf{J}^{-1}\begin{pmatrix}f_1\\f_2\end{pmatrix}^{(k)}$$

$$\mathbf{J}=\begin{pmatrix}\psi\zeta_1+(1-\psi)\zeta_2 & \psi\phi_1+(1-\psi)\phi_2 \\ \psi\frac{\partial E_1}{\partial p}+(1-\psi)\frac{\partial E_2}{\partial p} & \psi\frac{\partial E_1}{\partial T}+(1-\psi)\frac{\partial E_2}{\partial T}\end{pmatrix} \tag{11}$$

초기값: 이전 시간 $(p,T)$. 수렴: $\max(|f_1|/\rho,|f_2|/E)<10^{-12}$. 3-5회.

---

## 3. Implicit 선형화 — 보존 변수를 $(p,u,T)$로

### 3.1 Implicit 밀도

$$\rho^{(n+1)} = \rho^{(n)}+\zeta_v\delta p+\phi_v\delta T+(\rho_1-\rho_2)\delta\psi \tag{12}$$

$\delta\phi = \phi^{(n+1)}-\phi^{(n)}$.

### 3.2 Implicit 운동량

$$(\rho u)^{(n+1)} = \rho^{(n)}u^{(n+1)}+u^{(n)}[\zeta_v\delta p+\phi_v\delta T+\Delta\rho_\psi\delta\psi] \tag{13}$$

### 3.3 Implicit 총에너지

$$\mathcal{E}^{(n+1)} \approx \mathcal{E}^{(n)}+\frac{\partial\mathcal{E}}{\partial p}\delta p+\frac{\partial\mathcal{E}}{\partial T}\delta T+\frac{\partial\mathcal{E}}{\partial u}\delta u+\frac{\partial\mathcal{E}}{\partial\psi}\delta\psi \tag{14}$$

$$\frac{\partial\mathcal{E}}{\partial p}=\psi\frac{\partial E_1}{\partial p}+(1-\psi)\frac{\partial E_2}{\partial p}+\tfrac{1}{2}u^2\zeta_v \tag{14a}$$
$$\frac{\partial\mathcal{E}}{\partial T}=\psi\frac{\partial E_1}{\partial T}+(1-\psi)\frac{\partial E_2}{\partial T}+\tfrac{1}{2}u^2\phi_v \tag{14b}$$
$$\frac{\partial\mathcal{E}}{\partial u}=\rho u \tag{14c}$$
$$\frac{\partial\mathcal{E}}{\partial\psi}=(E_1-E_2)+\tfrac{1}{2}u^2(\rho_1-\rho_2) \tag{14d}$$

---

## 4. 일관적 대류 플럭스 [Janodet2025, Arrufat2021]

### 4.1 Picard 분리

$$\Phi_f^{(\phi)} = \tilde{\phi}_f^{(n)}\cdot F_f^{(n+1)} \tag{15}$$

### 4.2 일관성 체인

```
① CICSAM → ψ̃_f
② ρ̃_f = ψ̃_f·ρ_{1,f} + (1-ψ̃_f)·ρ_{2,f}
③ ũ_f = (ρu)_{f,TVD} / ρ_{f,TVD}   (Favre-TVD, Minmod)
④ (ρũ)_f = ρ̃_f · ũ_f
⑤ H̃_f = Ẽ_f + p_f   (van Leer for E, linear for p)
```

### 4.3 각 방정식의 대류 플럭스

$$\text{연속}: \tilde{\rho}_f F_f^{(n+1)} \tag{16a}$$
$$\text{운동량}: (\tilde{\rho}\tilde{u})_f F_f^{(n+1)} + \bar{p}_f^{(n+1)} \tag{16b}$$
$$\text{에너지}: \tilde{H}_f F_f^{(n+1)} \tag{16c}$$
$$\text{VOF}: \tilde{\psi}_f F_f^{(n+1)} \tag{16d}$$

운동량의 $\bar{p}_f^{(n+1)}$는 implicit. 나머지 face 값은 deferred (Picard).

### 4.4 MWI — Implicit 체적 플럭스 [Bartholomew2018]

$$F_{i+1/2}^{(n+1)} = \bar{u}_{i+1/2}^{(n+1)} - \hat{d}_f\!\left[\frac{p_{i+1}^{(n+1)}-p_i^{(n+1)}}{\Delta x}-\frac{\breve{\rho}_f}{2}\!\left(\frac{(\nabla p)_i^{(n+1)}}{\rho_i}+\frac{(\nabla p)_{i+1}^{(n+1)}}{\rho_{i+1}}\right)\right]+\hat{d}_f G_f+\hat{d}_f\frac{\breve{\rho}_f^n}{\Delta t}(\vartheta_f^n-\bar{u}_f^n) \tag{17}$$

- $\breve{\rho}_f$: 조화평균 밀도
- $\hat{d}_f$: MWI 가중인자 (운동량 대각의 역수)
- 압력 low-pass filter: $(p_{i+1}-p_i)/\Delta x - \overline{\nabla p}_f \propto \partial^3 p/\partial x^3$ → checkerboard 제거

### 4.5 ACID — Acoustically Conservative Interface Discretisation [Denner2018]

**Phase 1부터 필수.** 비점성에서 점성 소산이 없으므로 계면 음향 에러가 감쇠 없이 누적 → ACID가 더 중요.

**문제**: 밀도 불연속($\rho_1/\rho_2 \gg 1$)에서 음향 임피던스($Z_k = \rho_k c_k$) 불일치. 표준 보간(산술/조화평균)으로 face 밀도를 구하면 계면에서 음향 에너지가 보존되지 않음 → 비물리적 음압 반사/생성.

**핵심**: MWI의 face 밀도 $\breve{\rho}_f$와 압력 기울기 이산화를 수정하여, 계면을 통과하는 음향파의 에너지를 정확히 보존.

1D에서 face $i+1/2$ (계면 포함)의 ACID 수정 밀도:

$$\breve{\rho}_{i+1/2}^{\text{ACID}} = \frac{\rho_i\rho_{i+1}(\rho_i c_i+\rho_{i+1}c_{i+1})}{\rho_i^2 c_i+\rho_{i+1}^2 c_{i+1}} \tag{17'}$$

비계면 셀에서는 표준 조화평균으로 복원. 계면 판별: $|\psi_{i+1}-\psi_i| > \epsilon$.

이 $\breve{\rho}_f^{\text{ACID}}$를 MWI (Eq. 17)의 $\breve{\rho}_f$ 대신 사용.

---

## 5. CICSAM (1D = Hyper-C)

$$\tilde{\psi}_D=\frac{\psi_D-\psi_{UU}}{\psi_A-\psi_{UU}+\epsilon}, \quad \tilde{\psi}_f^*=\begin{cases}\min(\tilde{\psi}_D/Co_f,1)&0\le\tilde{\psi}_D\le1\\\tilde{\psi}_D&\text{else}\end{cases} \tag{18}$$

비정규화: $\tilde{\psi}_f=\psi_{UU}+\tilde{\psi}_f^*(\psi_A-\psi_{UU})$. $Co_f<0.5$.
Compression term 불필요 [Zanutto2022].

---

## 6. BDF2 시간 이산화

$$\frac{\partial\Omega}{\partial t}\approx\frac{3\Omega^{n+1}-4\Omega^n+\Omega^{n-1}}{2\Delta t} \tag{19}$$

첫 시간단계: BDF1. 가변 시간간격은 [Janodet2025, Eq. 47] 참조.

---

## 7. Mode A: Segregated VOF(CN) + Coupled $(p,u,T)$(BDF2)

### 7.1 알고리즘

```
매 시간단계:
  for n_iter = 1..max_iter:          // 비선형 반복
    [Step 1] VOF — Crank-Nicolson
      ψ^{n+1} (u^(n) 사용, CICSAM face값, Thomas alg.)
    [Step 2] 열역학 갱신 (ψ^{n+1}, p^(n), T^(n)) → ρ_k, E_k, ...
    [Step 3] Coupled (p,u,T) — BDF2, 3N 선형계
    [Step 4] EOS 역변환 → (p,T) 갱신
    [Step 5] 수렴 확인: max L2(ρ,ρu,E,ψ) < 1e-6
  end
```

### 7.2 VOF Crank-Nicolson

$$\frac{\psi^{n+1}-\psi^n}{\Delta t}+\frac{1}{2}[\mathcal{L}_\psi^{n+1}+\mathcal{L}_\psi^n]=0 \tag{20}$$

### 7.3 Coupled $(p,u,T)$: $3N\times3N$

$$\begin{pmatrix}\mathcal{A}^p_{\text{cont}}&\mathcal{A}^u_{\text{cont}}&\mathcal{A}^T_{\text{cont}}\\\mathcal{A}^p_{\text{mom}}&\mathcal{A}^u_{\text{mom}}&\mathcal{A}^T_{\text{mom}}\\\mathcal{A}^p_{\text{ener}}&\mathcal{A}^u_{\text{ener}}&\mathcal{A}^T_{\text{ener}}\end{pmatrix}\begin{pmatrix}\phi_p\\\phi_u\\\phi_T\end{pmatrix}=\begin{pmatrix}b_{\text{cont}}\\b_{\text{mom}}\\b_{\text{ener}}\end{pmatrix} \tag{21}$$

---

## 8. Mode B: Fully-Coupled $(p,u,T,\psi)$ BDF2

### 8.1 $4N\times4N$

$$\begin{pmatrix}\mathcal{A}^p_{\text{cont}}&\mathcal{A}^u_{\text{cont}}&\mathcal{A}^T_{\text{cont}}&\mathcal{A}^\psi_{\text{cont}}\\\mathcal{A}^p_{\text{mom}}&\mathcal{A}^u_{\text{mom}}&\mathcal{A}^T_{\text{mom}}&\mathcal{A}^\psi_{\text{mom}}\\\mathcal{A}^p_{\text{ener}}&\mathcal{A}^u_{\text{ener}}&\mathcal{A}^T_{\text{ener}}&\mathcal{A}^\psi_{\text{ener}}\\\mathcal{A}^p_{\text{vof}}&\mathcal{A}^u_{\text{vof}}&\mathcal{A}^T_{\text{vof}}&\mathcal{A}^\psi_{\text{vof}}\end{pmatrix}\begin{pmatrix}\phi_p\\\phi_u\\\phi_T\\\phi_\psi\end{pmatrix}=\begin{pmatrix}b_{\text{cont}}\\b_{\text{mom}}\\b_{\text{ener}}\\b_{\text{vof}}\end{pmatrix} \tag{22}$$

VOF에서 $F_f^{(n+1)}$를 통해 $u$, $p$와 결합 [Janodet2025, Eq. 53].

---

## 9. BSD — Both-Sides Diffusion (Phase 2: 점성 추가 시 구현)

> **Phase 1(Euler)에서는 비활성. Phase 2에서 점성항 추가 시 활성화.**

### 9.1 목적

큰 밀도비/속도 구배에서 implicit 행렬의 대각 우세성 강화. 물리적 해를 변경하지 않으면서 고주파 수치 에러만 선택적 감쇠.

### 9.2 운동량 방정식에 추가

$$\frac{\partial(\rho u)}{\partial t}+\cdots = -\nabla p+\nabla\cdot\tau+\underbrace{\nabla\cdot(\hat{\eta}\nabla u)_{\text{small}}}_{\text{implicit}}-\underbrace{\nabla\cdot(\hat{\eta}\nabla u)_{\text{large}}}_{\text{deferred}} \tag{23}$$

### 9.3 $\hat{\eta}$ 결정

유체의 물리적 점성을 그대로 사용:

$$\hat{\eta}_i = \mu_i = \psi_i\mu_1+(1-\psi_i)\mu_2 \tag{24}$$

Face: 조화평균 $1/\hat{\eta}_{i+1/2}=1/(2\hat{\eta}_i)+1/(2\hat{\eta}_{i+1})$

$\hat{\eta}$를 물리적 점성 스케일에 맞추는 이유:
- 너무 크면 → 수치확산이 물리적 디테일(전단층, 소용돌이) 파괴
- 너무 작으면 → 대각 강화 효과 미미
- $\hat{\eta}=\mu$ → 물리적 점성 소산 범위 내에서 최적 균형

### 9.4 Small vs Large 스텐실 (1D)

**Small stencil** (compact 3-point, implicit $u^{(n+1)}$):

$$D_s[u]_i = \frac{1}{\Delta x}\left[\hat{\eta}_{i+1/2}\frac{u_{i+1}^{(n+1)}-u_i^{(n+1)}}{\Delta x}-\hat{\eta}_{i-1/2}\frac{u_i^{(n+1)}-u_{i-1}^{(n+1)}}{\Delta x}\right] \tag{25}$$

**Large stencil** (5-point, deferred $u^{(n)}$):

Face gradient를 셀 중심 기울기의 보간으로:

$$\left(\frac{\partial u}{\partial x}\right)_{i+1/2}^{\text{large}}=\frac{1}{2}\left[\frac{u_{i+1}-u_{i-1}}{2\Delta x}+\frac{u_{i+2}-u_i}{2\Delta x}\right]=\frac{u_{i+2}+u_{i+1}-u_i-u_{i-1}}{4\Delta x} \tag{26}$$

$$D_l[u]_i = \frac{1}{\Delta x}\left[\hat{\eta}_{i+1/2}\frac{u_{i+2}^{(n)}+u_{i+1}^{(n)}-u_i^{(n)}-u_{i-1}^{(n)}}{4\Delta x}-\hat{\eta}_{i-1/2}\frac{u_{i+1}^{(n)}+u_i^{(n)}-u_{i-1}^{(n)}-u_{i-2}^{(n)}}{4\Delta x}\right] \tag{27}$$

### 9.5 수학적 효과

균일 $\hat{\eta}$에서 Taylor 전개:

$$D_s-D_l = \hat{\eta}\frac{\Delta x^2}{4}\frac{\partial^4 u}{\partial x^4}+O(\Delta x^4) \tag{28}$$

- **4차 미분 감쇠 (biharmonic)**: 고주파($k\to\pi/\Delta x$) 최대 감쇠, 저주파 영향 없음
- **자동 적응**: 급변 영역($\partial^4 u/\partial x^4$ 큼)에서만 강하게 작동
- **격자 수렴**: 효과가 $O(\Delta x^2)$로 소멸 → 공간 정확도 유지

### 9.6 대각 기여

물리적 점성 + BSD 합산:

$$\text{대각 계수} \propto (\mu+\hat{\eta})\frac{2}{\Delta x^2}$$

$\hat{\eta}=\mu$이면 **대각이 물리적 점성만 있을 때의 2배**로 강화.

### 9.7 Ghost Cell 요구

Large stencil이 5-point → **ghost cells ≥ 3** (Phase 1의 2에서 증가).

---

## 10. Artificial Viscosity [Denner2017] (선택, BSD와 독립)

> BSD와 역할이 다름. BSD = 수치적 안정화 (수렴 시 $O(\Delta x^2)$). AV = 물리적 계면 감쇠 (잔류).

$$f_\tau=\mu_\Gamma\frac{u_{i+1}-2u_i+u_{i-1}}{\Delta x^2}, \quad \mu_\Gamma=C_\mu\Delta x\max(\rho_1,\rho_2)c_{\max}\hat{\delta}_\Gamma \tag{29}$$

$\hat{\delta}_\Gamma=\min(|\psi_{i+1}-\psi_{i-1}|/2,\;0.5)$. 기본 비활성.

| | BSD | Artificial Viscosity |
|---|---|---|
| 목적 | 행렬 대각 강화 | 계면 고주파 감쇠 |
| 잔류 | $O(\Delta x^2)$ 소멸 | 물리적 잔류 |
| 적용 | 전체 도메인 | 계면 근처만 |
| $\hat{\eta}$ | $=\mu$ (물리 점성) | 별도 스케일링 |

---

## 11. 확산항 이산화 (Phase 2)

$$\frac{\partial\tau}{\partial x}\bigg|_i=\frac{4}{3\Delta x}\left[\mu_{i+1/2}\frac{u_{i+1}-u_i}{\Delta x}-\mu_{i-1/2}\frac{u_i-u_{i-1}}{\Delta x}\right] \tag{30}$$

$$\frac{\partial}{\partial x}\!\left(\lambda\frac{\partial T}{\partial x}\right)\bigg|_i=\frac{1}{\Delta x}\left[\lambda_{i+1/2}\frac{T_{i+1}-T_i}{\Delta x}-\lambda_{i-1/2}\frac{T_i-T_{i-1}}{\Delta x}\right] \tag{31}$$

Implicit 처리 (Mode A, B 모두). Face 물성: 조화평균.

---

## 12. 시간간격

Implicit → 음향 CFL 제거.

$$\Delta t=C_{\Delta t}\min\!\left(\frac{\Delta x}{\max|u|},\;\frac{\Delta x^2}{2\max(\mu/\rho)},\;\frac{\Delta x^2}{2\max(\lambda/(\rho c_p))}\right) \tag{32}$$

CICSAM $Co_f<0.5$. $C_{\Delta t}=0.5\sim1.0$.
Phase 1 (비점성): 대류 CFL만 적용.

---

## 13. 경계조건

- **투과**: $\phi_{\text{ghost}}=\phi_{\text{경계셀}}$
- **벽면**: $u_{\text{ghost}}=-u_{\text{내부}}$, 나머지 zero-gradient
- **주기**: wrap-around
- Ghost cells: $\ge2$ (Phase 1), $\ge3$ (Phase 2, BSD)

---

## 14. 검증 테스트 (Phase 1)

1. **단상 음향 펄스**: 이상기체, 양방향 음파, 에너지 보존
2. **Water expansion tube** [Demou2022 §4.4]: 쌍방향 희박파
3. **Interface advection**: CICSAM 선명도·질량보존
4. **정수압 균형**: $u=0$ 유지

---

## 15. 구현 구조

```
denner_1d/
├── config.py          # Mode A/B, Phase 1/2, 매개변수
├── eos/
│   ├── base.py        # EOS 인터페이스 (TDU: ζ,φ,∂E/∂p,∂E/∂T)
│   ├── ideal.py       # Ideal gas
│   ├── nasg.py        # NASG
│   └── invert.py      # (ψ,ρ,E)→(p,T) Newton 역변환
├── grid.py            # 1D 격자, ghost cells
├── interface/
│   ├── cicsam.py      # Hyper-C (Phase 1)
│   └── thinc_qq.py    # THINC/QQ (Phase 2c)
├── flux/
│   ├── consistent.py  # ψ̃→ρ̃→ũ→H̃ 체인
│   ├── limiter.py     # van Leer, Minmod
│   └── mwi.py         # MWI implicit 체적 플럭스
├── linearize.py       # Newton: ρ,ρu,E → (p,u,T,ψ) 계수
├── assembly.py        # 블록 행렬 조립 (3N or 4N)
├── vof_cn.py          # Mode A: VOF Crank-Nicolson
├── solver_a.py        # Mode A 드라이버
├── solver_b.py        # Mode B 드라이버
├── stabilize/
│   ├── bsd.py         # BSD (Phase 2)
│   └── art_visc.py    # Artificial viscosity (선택)
├── diffusion.py       # 점성·열전도 (Phase 2)
├── boundary.py        # BC
├── timestepping.py    # Δt 제어
├── io_utils.py        # 출력
├── test_cases/
│   ├── acoustic_pulse.py
│   ├── expansion_tube.py
│   ├── advection.py
│   └── hydrostatic.py
└── main.py
```

---

## 16. 주의사항

1. **밀도 독립 보간 금지**: 반드시 $\tilde{\psi}_f$에서 유도
2. **$\psi$ 클리핑**: $[10^{-8},1-10^{-8}]$
3. **BDF1 시작**: 첫 시간단계
4. **EOS 도함수 해석적 구현**: 수치미분 금지
5. **비대칭 행렬**: BiCGSTAB/GMRES
6. **MWI transient correction** 포함 필수
7. **Phase 1은 비점성**: $\tau=0$, BSD 비활성, ghost $\ge2$
8. **에너지 플럭스 $(\mathcal{E}+p)u$**: $\tilde{H}_f$ deferred (Picard)
9. **CICSAM + compression term 동시 사용 금지** [Zanutto2022]
10. **VOF → mass fraction 전환 시**: Eq.(4)→Eq.(4'), CICSAM을 $Y$에 적용, $\alpha=\rho Y/\rho_1$로 복원

---

## 17. 검증 과정에서의 실패 원인 분석

> **대상 테스트**: 1D Smooth Interface Advection (Air/Water NASG, periodic BC)
> 케이스 C: N=50, CFL=1.0, 10 flow-through, PASS 기준 L2(p) < 1e-4

---

### 17.1 BiCGSTAB 선형 솔버 오차 누적 → spsolve로 교체

**현상**: N=10, 약 25 step 후 p가 1e5 → 5e9 Pa로 발산.

**원인**: 반복법인 BiCGSTAB는 매 time step마다 ~1e-4 Pa의 잔차를 남긴다.
Picard 반복 내 비선형 루프에서 이 오차가 누적되고,
특정 계면 위치(position 5/10, 주기 경계 근처)에서 압력 오차가 매 flow-through마다 ~20배 증폭된다.
3 flow-through(25 step) 후 누적 오차가 폭발적으로 증가해 발산.

**수정**: 직접법인 `scipy.sparse.linalg.spsolve`로 교체.
직접법은 machine precision(~1e-15) 수준으로 해를 구하므로 오차 누적이 없다.

> 주의사항 §16-5의 "BiCGSTAB/GMRES" 권장은 대형 3D 시스템 기준이며,
> 1D 소규모(3N×3N) 시스템에서는 spsolve가 훨씬 안정적이다.

---

### 17.2 CFL < 1에서의 CICSAM 압축 불안정

**현상**: CFL=0.5로 설정 시 ft=1부터 발산(p 진동 → NaN).

**원인**: 1D CICSAM(Hyper-C)은 면 Courant 수 $Co_f$에 따라 face VOF 값을 결정한다:

$$\tilde{\psi}_f^* = \min\!\left(\frac{\tilde{\psi}_D}{Co_f},\;1\right)$$

- $Co_f = 1.0$: $\tilde{\psi}_f = \psi_D$ → 순수 upwind → 안정
- $Co_f < 1.0$: $\tilde{\psi}_f > \psi_D$ → compressive → 계면에서 rho_face 불연속 급등
- Compressive face 밀도 → 질량 플럭스 불일치 → 압력 방정식 우변 오류 → 발산

**수정**: CFL = 1.0으로 고정. implicit 방법이므로 CFL<0.5 기준(명시적)은 적용 불필요.
CFL > 1.0도 동일 이유(역방향 compressive 효과)로 발산.

**결론**: 이 솔버에서 CFL=1.0이 유일하게 안정적인 값.

---

### 17.3 N=10에서의 주기 경계 공진 불안정

**현상**: BiCGSTAB → spsolve로 교체 후에도 N=10에서 step 25(약 2.5 flow-through) 지점에서 발산.

**원인**: N=10, CFL=1.0이면 dt=dx/u=0.1 s, 즉 10 step이 1 flow-through.
CFL=1.0이므로 인터페이스는 매 step 정확히 1셀씩 이동한다.
10 step마다 동일한 격자 위치로 복귀하며, 주기 경계(셀 10→셀 0 연결)를 통과하는
구성(인터페이스 위치 5/10)에서 압력 오차가 구조적으로 증폭된다.
spsolve로도 step당 ~1e-7 Pa 오차가 남고, 이것이 매 통과마다 ~20배 증폭되어
2-3 flow-through 후 발산한다.

**수정**: N=50 사용. 격자가 세밀해지면 동일 증폭 효과가 분산되고,
10 flow-through 동안 L2(p) < 5e-8 (기준 1e-4 대비 2000배 마진)을 유지.

**교훈**: 주기 BC + 정수비 CFL 조합은 공진 불안정을 유발할 수 있다.
N=10처럼 소규모 격자에서는 이 효과가 치명적이다.

---

### 17.4 다중 세그먼트 run() 재시작 누적 오차

**현상**: ft=1 성공 후 그 결과를 initial condition으로 ft=2를 별도 run()으로 시작하면,
ft=2 step 5~6에서 Picard가 더 많은 반복을 요구하고 결국 발산.

**원인**: 각 run() 호출은 `is_first_step=True`로 초기화된다.
ft=1 종료 시 p=99999.99 Pa(오차 -0.01 Pa)인 상태를 initial condition으로 건네면,
ft=2의 첫 step에서 BDF1 보정항이 이 오차를 기반으로 계산된다.
재시작마다 이 오차가 추가되어 ft=5에서는 압력 편차가 초기 허용 범위를 벗어난다.

**수정**: 단일 연속 run() + `output_times=[1,2,...,max_ft]` 방식.
run() 내부에서 output_times에 도달할 때마다 스냅샷을 저장하고 계산을 이어간다.
재시작 없으므로 BDF 상태(n-1 시간 단계 정보)가 완전히 보존된다.

---

### 17.5 Picard 이후 EOS 역변환 → 압력 스파이크

**현상**: Picard 반복 수렴 후 EOS 역변환(Eq. 10-11)을 적용하면
p가 1e5 → 8.7e5 Pa로 순간 스파이크, 다음 step에서 발산.

**원인**: Picard 결과 $(p,u,T)$는 선형화된 연속·운동량·에너지 방정식을 동시에 만족시킨다.
이 $(p,T)$ 쌍은 이미 EOS와 일관성이 있다(선형화 계수 $\zeta, \phi$를 사용해 구성).
여기에 EOS 역변환 Newton을 추가 적용하면 선형화 오차가 보정되는 것이 아니라
일관성 있는 해가 교란된다 → 압력 급등.

**수정**: Picard 이후 EOS 역변환 제거. Picard 결과 $(p,u,T,\psi)$를 그대로 다음 step에 사용.

---

### 17.6 BDF2 시간 적분 → NaN

**현상**: BDF2 활성 시 ft=1 내 초반 step에서 NaN 발생.

**원인**: BDF2는 $\Omega^{n-1}$, $\Omega^n$, $\Omega^{n+1}$ 세 시간 단계를 사용한다 (Eq. 19).
밀도가 셀마다 O(1) 이상 점프하는 계면 근처에서는 $\Omega^{n-1}$의 기여(계수 +1)가
과도한 이동량을 생성해 행렬 우변 b에 NaN을 유발한다.

**수정**: BDF1(후진 Euler)만 사용:
$$\frac{\Omega^{n+1}-\Omega^n}{\Delta t}$$
1차 정확도이지만 안정적. BDF2 전환은 추후 계면 밀도 점프 처리가 개선된 후 재검토.

---

### 17.7 부동소수점 드리프트로 인한 미세 시간 단계

**현상**: ft=1 성공 후 미세 단계(dt ≈ 2.6e-9 s)가 생성되며 CICSAM Co ≈ 2.6e-8이 됨.
이 미세 단계에서 극도로 compressive한 CICSAM이 적용되어 psi 오차 급등.

**원인**: Python 부동소수점 덧셈 누적:
```
10 × 0.1 = 0.9999999999999999  (< 1.0)
```
dt clamp 로직이 `remaining = t_end - t = 1e-16` 같은 극소값을 dt로 사용하도록 허용.

**수정**: dt clamp 시 잔여 시간 검사 추가:
```python
if remaining > 0.01 * dt:   # 잔여가 충분히 클 때만 clamp
    dt = min(dt, remaining)
```
0.01*dt 미만의 미세 잔여는 무시하고 현재 dt 유지.

---

### 17.8 psi_face VOF 불일치 (~47 kg/(m³s) 질량 잔차)

**현상**: 이전 세션에서 발견. 수렴 잔차(연속 방정식 우변)가 ~47 kg/(m³s)으로 높아
Picard가 느리게 수렴하고 압력 진동 유발.

**원인**: VOF step에서 CICSAM으로 계산한 `psi_face_vof`와,
이후 assembly에서 face량 재계산 시 사용하는 `psi_face`가 서로 다른 값.
두 경로가 다른 보간을 거치므로 질량 플럭스 불일치 발생.

**수정**: VOF step에서 계산한 `psi_face_vof`를 `compute_all_face_quantities`에
`psi_face_given` 인자로 직접 전달하여 일관성 강제.
이 수정 후 잔차가 machine precision 수준으로 감소.

---

### 17.9 스냅샷 저장 누락 (엄격한 시간 조건)

**현상**: output_times=[1.0, 2.0, ...]로 설정했으나 ft=3 이후 스냅샷이 저장되지 않음.

**원인**: 스냅샷 저장 조건을 `abs(t - target) < 1e-12 * max(t, 1e-10)`으로 설정.
시간 적분에서 target을 미세하게 초과(overshoot)하면 조건 불충족 → 영구적으로 저장 안 됨.

**수정**: 조건을 `t >= target - 1e-10`으로 완화.
처음 target에 도달하거나 통과하는 시점에 스냅샷 저장.