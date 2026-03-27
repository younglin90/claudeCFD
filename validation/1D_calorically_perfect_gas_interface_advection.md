# APEC Validation Case 3.1 — Calorically Perfect Gas: 1-D Smooth Interface Advection

> **출처:** Terashima, Ly & Ihme, *Journal of Computational Physics* 524 (2025) 113701, §3.1  
> **목적:** Calorically perfect gas 조건에서 APEC의 압력 평형 보존(PE-preserving) 및 에너지 보존 성능 검증

---

## 1. 지배방정식 및 상태방정식

### 1.1 혼합 상태방정식

$$\rho e = \frac{p}{\bar{\gamma} - 1}$$

### 1.2 혼합 비열비

$$\frac{1}{\bar{\gamma} - 1} = \bar{M} \sum_{i=1}^{N} \frac{1}{\gamma_i - 1} \frac{Y_i}{M_i}$$

### 1.3 APEC 편미분 계수 (half-point 구성에 필요)

$$\epsilon_i = \left(\frac{\partial \rho e}{\partial \rho_i}\right)_{\rho_{j \neq i},\, p}
= \frac{p \bar{M}^2}{\rho^2} \frac{1}{M_i}
\left(
\frac{1}{\gamma_i - 1} \sum_{k=1}^{N} \frac{\rho Y_k}{M_k}
- \sum_{k=1}^{N} \frac{1}{\gamma_k - 1} \frac{\rho Y_k}{M_k}
\right)$$

---

## 2. 물성 (2-성분계: Species 1 / Species 2)

| 항목 | Species 1 | Species 2 |
|------|-----------|-----------|
| 비열비 $\gamma_i$ | 1.4 | 1.66 |
| 분자량 $M_i$ | 28 (N₂ 유사) | 4 (He 유사) |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | **501 points** (uniform) |
| 격자 간격 $\Delta x$ | $1/500 = 0.002$ |
| 차원 | 1-D |

> **수렴성 검토 시 사용된 추가 격자:** 251점 (최조밀 기준 정규화)

---

## 4. 초기 조건

$$\begin{pmatrix} (\rho Y_1)_0 \\ (\rho Y_2)_0 \\ u_0 \\ p_0 \end{pmatrix}
=
\begin{pmatrix}
\dfrac{w_1}{2}\left(1 - \tanh(k(r - r_c))\right) \\[8pt]
\dfrac{w_2}{2}\left(1 + \tanh(k(r - r_c))\right) \\[8pt]
1.0 \\[4pt]
0.9
\end{pmatrix}$$

여기서 $r = |x - x_c|$ (파 중심으로부터의 거리)

### 초기 조건 파라미터

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| $x_c$ | 0.5 | 파 중심 위치 |
| $r_c$ | 0.25 | 파 중심~계면 거리 |
| $w_1$ | 0.6 | Species 1 밀도 스케일 |
| $w_2$ | 0.2 | Species 2 밀도 스케일 |
| $k$ | 20 | 계면 두께 제어 계수 (클수록 급준) |

---

## 5. 경계 조건

| 항목 | 값 |
|------|-----|
| 경계 조건 타입 | **Periodic** (양단) |

---

## 6. 출력 변수 및 결과 그래프

### 6.1 공간 분포 (Fig. 1) — $t = 8.0$

| 그래프 | 변수 | 비고 |
|--------|------|------|
| Fig. 1a | $\rho_1$ (Species 1 밀도) | FC-NPE에서 oscillation 발생 |
| Fig. 1b | $\rho_2$ (Species 2 밀도) | |
| Fig. 1c | $u$ (속도) | FC-NPE에서 spurious oscillation |
| Fig. 1d | $p$ (압력) | FC-NPE에서 발산 → $t=9.0$ blow-up |

**관찰:** FC-NPE는 $t \approx 9.0$에서 발산. APEC 및 Fujiwara는 oscillation 없이 안정적.

---

### 6.2 에러 시계열 (Fig. 2)

#### 에너지 보존 오차
$$E_{cons}(t) = \sum_t \left\{ \frac{\sum_{m=1}^{N_g} E_m(t)}{\sum_{m=1}^{N_g} E_{0,m}} - 1 \right\}$$

#### 압력 평형 오차 (PE error)
$$E_{PE}(t) = \sqrt{ \frac{1}{N_g} \sum_{m=1}^{N_g} \left( \frac{p_m(t)}{p_0} - 1 \right)^2 }$$

| 스킴 | 에너지 보존 | PE 오차 크기 | 발산 여부 |
|------|------------|-------------|----------|
| FC-NPE | 만족 | $t \approx 4.0$부터 발산 | **발산** |
| APEC | 만족 | $O(10^{-5} \sim 10^{-4})$ | 안정 |
| Fujiwara | 만족 | APEC보다 작음 | 안정 |

---

### 6.3 PE 오차 공간 분포 (Fig. 3) — $t = 0$

Leading-order PE-preserving 오차 (중심차분으로 평가):

$$e_\text{APEC}(x) = \sum_{i=1}^{N} \left\{ \frac{1}{12} \left.\frac{\partial \epsilon_i}{\partial x}\right|_m \left.\frac{\partial^2 \rho Y_i}{\partial x^2}\right|_m - \frac{1}{12} \left.\frac{\partial^2 \epsilon_i}{\partial x^2}\right|_m \left.\frac{\partial \rho Y_i}{\partial x}\right|_m \right\} \Delta x^2$$

$$e_\text{FC-NPE}(x) = \sum_{i=1}^{N} \left\{ \frac{1}{3} \left.\frac{\partial \epsilon_i}{\partial x}\right|_m \left.\frac{\partial^2 \rho Y_i}{\partial x^2}\right|_m + \frac{1}{6} \left.\frac{\partial^2 \epsilon_i}{\partial x^2}\right|_m \left.\frac{\partial \rho Y_i}{\partial x}\right|_m \right\} \Delta x^2$$

**관찰:** 오차는 계면 부근에서 최대. APEC 오차 분포는 대칭 (오차식의 대칭성과 일치).

---

### 6.4 오차 norm 시계열 (Fig. 4)

$$\|e\| = \sqrt{ \frac{1}{N_g} \sum_{m=1}^{N_g} \{e(x)\}^2 \bigg|_m }$$

| 스킴 | 시간 거동 |
|------|---------|
| APEC | 거의 상수 → $\epsilon_i$, $\rho Y_i$ 프로파일 안정적 유지 |
| FC-NPE | $t \approx 4.0$부터 발산 |

---

### 6.5 격자 수렴성 (Fig. 5) — $t = 20.0$

$$\text{error} = \sqrt{ \frac{1}{N_g} \sum_{m=1}^{N_g} f_{PE}^2 \bigg|_m }$$

| 결과 | 내용 |
|------|------|
| 수렴 차수 | **2차 정확도** ($O(\Delta x^2)$) |
| 기준 격자 | 251 points (coarsest) |
| 이론과의 일치 | Eq.(32), Eq.(63)과 일치 |