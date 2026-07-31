# 1차원 오일러 5-방정식 IMEX 솔버 — 상세 수학 및 수치기법 문서

> 1차원 오일러 논문 (`solver/five_eq_IMEX/`) 의 부속 수치기법 문서.
> 모든 방정식과 의사코드는 다음 production 설정에서의 실제 구현을 그대로 옮긴 것이다: `FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ssp3`, `FIVE_EQ_IMEX_ALPHA_SCHEME=adaptive_bvd`, `FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu`, `FIVE_EQ_IMEX_TMLPU_TVD=superbee`, `FIVE_EQ_IMEX_MATERIAL_FLUX=slau2`.
> 절 번호는 본 manuscript (`1d_euler_imex_method_revised.md`) 와 일치.

---

## 1. 지배방정식 (Governing equations)

### 1.1 5-방정식 확산 계면(diffuse-interface) 모델

\(\alpha_1(x,t)\in[0,1]\) 을 phase 1 의 체적분율, \(\alpha_2 = 1-\alpha_1\) 으로 정의한다.  각 phase 는 자체 밀도 \(\rho_k(x,t)\), 비내부에너지 \(e_k(x,t)\), 온도 \(T_k(x,t)\) 를 가지며, 두 phase 는 단일 속도 \(u(x,t)\) 와 단일 압력 \(p(x,t)\) 를 공유한다.  보존변수 벡터는

$$
\mathbf{U}(x,t) = \bigl(q_1,\ q_2,\ q_3,\ q_4,\ q_5\bigr)^{\!\top}
= \bigl(\alpha_1\rho_1,\ \alpha_2\rho_2,\ \rho u,\ \rho E,\ \alpha_1\bigr)^{\!\top},
\tag{2.1}
$$

이며, 혼합 밀도 \(\rho = \alpha_1\rho_1 + \alpha_2\rho_2\), 단위질량 총 에너지 \(\rho E = \alpha_1\rho_1 e_1 + \alpha_2\rho_2 e_2 + \tfrac12\rho u^2\) 이다.  1차원 5-방정식 연속체 시스템은 다음과 같다:

$$
\partial_t(\alpha_k\rho_k) + \partial_x(\alpha_k\rho_k u) = 0, \qquad k=1,2,
\tag{2.2}
$$

$$
\partial_t(\rho u) + \partial_x(\rho u^2 + p) = 0,
\tag{2.3}
$$

$$
\partial_t(\rho E) + \partial_x\bigl((\rho E + p)u\bigr) = 0,
\tag{2.4}
$$

$$
\partial_t \alpha_1 + u\,\partial_x \alpha_1 \;=\; (\alpha_1 + D_1)\,\partial_x u .
\tag{2.5}
$$

비보존(non-conservative) 계수 \(D_1\) 이 모델 변종을 결정한다:

* **Allaire 모델** (조성 이동만): \(D_1 \equiv 0\).
* **Kapila / 압력평형 모델** (즉시 압력 이완, 약한 다짐):
$$
D_1 = \frac{\alpha_1\alpha_2(\rho_2 c_2^{2} - \rho_1 c_1^{2})}
            {\alpha_2\rho_1 c_1^{2} + \alpha_1\rho_2 c_2^{2}},
\tag{2.6}
$$

여기서 \(c_k\) 는 §1.3 에서 정의되는 phase 음속이다.  Murrone-Guillard 의 축소형 5-방정식은 같은 영역에서 Kapila 폐쇄와 일치한다.

### 1.2 Primitive 변수의 선택

흔한 primitive 셋은 세 종류다: \((\alpha_1,\rho_1,\rho_2,u,p)\), \((\alpha_1,p,T_1,T_2,u)\), \((\alpha_1,\rho,u,p,Y_1)\).  본 솔버는 **온도 기반** primitive 를 사용한다:

$$
\mathbf{W} = \bigl(\alpha_1,\ T_1,\ T_2,\ u,\ p\bigr)^{\!\top}.
\tag{2.7}
$$

이유는 세 가지다.  (i) \(p\) 가 primitive 미지수가 되어 보존-원시 변환의 압력 round-off 를 근원적으로 제거.  (ii) §3 의 implicit pressure step 이 \(p, u\) 를 동시 갱신하므로 \((u,p)\) 가 \(\mathbf{W}\) 에 있으면 Newton 시스템에서 자연스러운 2×2 acoustic block 이 형성된다.  (iii) phase 별 온도 \(T_k\) 는 NASG EOS 의 자연 인자: \(\rho_k=\rho_k(p,T_k)\), \(e_k=e_k(p,T_k)\) 가 닫힌 형식.

### 1.3 상태방정식 (Equations of state)

phase 별로 IG (이상기체), SG (stiffened gas), NASG (Noble-Abel stiffened gas) 세 EOS 군을 허용한다.  사용자는 phase 마다 독립적으로 EOS 를 지정할 수 있어, air-water 와 helium-air 가 같은 검증 슈트 안에 코드 분기 없이 공존한다.

**이상기체.**  비열비 \(\gamma_k\), 기체상수 \(R_k\):

$$
\rho_k(p,T_k) = \frac{p}{R_k T_k}, \qquad
e_k(\rho_k,p) = \frac{p}{(\gamma_k-1)\rho_k}, \qquad
c_k^{2} = \gamma_k\frac{p}{\rho_k}.
\tag{2.8}
$$

**Stiffened gas** (참고압력 \(p_{\infty,k}\)):

$$
\rho_k(p,T_k) = \frac{p + p_{\infty,k}}{(\gamma_k-1) c_{v,k} T_k}, \qquad
e_k(\rho_k,p) = \frac{p + \gamma_k p_{\infty,k}}{(\gamma_k-1)\rho_k}, \qquad
c_k^{2} = \gamma_k\frac{p + p_{\infty,k}}{\rho_k}.
\tag{2.9}
$$

**Noble-Abel stiffened gas** (공부피 \(b_k\), \(p_{\infty,k}\)):

$$
\rho_k(p,T_k) = \frac{1}{b_k + (\gamma_k-1) c_{v,k} T_k / (p+p_{\infty,k})},
\tag{2.10a}
$$

$$
e_k(\rho_k,p) = \frac{p + \gamma_k p_{\infty,k}}{(\gamma_k-1)\rho_k}\,\bigl(1 - b_k \rho_k\bigr) + q_k,
\tag{2.10b}
$$

$$
c_k^{2} = \gamma_k\frac{p + p_{\infty,k}}{\rho_k(1 - b_k\rho_k)} .
\tag{2.10c}
$$

기체-액체 상호작용에서는 plain SG 가 적당한 \(p\) 영역에서 액체 밀도를 왜곡하므로 NASG 가 물 phase 에 필수이다.  `eos_facade.py` 에서 phase 별 EOS 객체를 인스턴스화하여 예컨대 phase 1 = `IdealEOS(\gamma_1=1.4)` (공기), phase 2 = `NASGEOS(\gamma_2,p_{\infty,2},b_2,c_{v,2},q_2)` (물) 형태로 조합한다.

### 1.4 분석형 원시-보존 야코비안

§3.1 의 implicit pressure block 에는 닫힌 형식의 \(d\mathbf{U}/d\mathbf{W}\) 가 필요하다.  \(\mathbf{W}=(\alpha_1,T_1,T_2,u,p)^{\!\top}\) 로 놓고, 약어 \(\rho_{k,p}\equiv\partial\rho_k/\partial p\big|_{T_k}\), \(\rho_{k,T}\equiv\partial\rho_k/\partial T_k\big|_{p}\), \(e_{k,p}\equiv\partial e_k/\partial p\big|_{T_k}\), \(e_{k,T}\equiv\partial e_k/\partial T_k\big|_{p}\) 를 사용하면:

$$
\frac{d\mathbf{U}}{d\mathbf{W}} =
\begin{pmatrix}
\rho_1 & \alpha_1\rho_{1,T} & 0 & 0 & \alpha_1\rho_{1,p}\\
-\rho_2 & 0 & \alpha_2\rho_{2,T} & 0 & \alpha_2\rho_{2,p}\\
(\rho_1-\rho_2)u & \alpha_1\rho_{1,T}u & \alpha_2\rho_{2,T}u & \rho & \rho_p u\\
J_{4,1} & J_{4,2} & J_{4,3} & \rho u & J_{4,5}\\
1 & 0 & 0 & 0 & 0
\end{pmatrix},
\tag{2.11}
$$

여기서

$$
\rho_p = \alpha_1\rho_{1,p} + \alpha_2\rho_{2,p},\qquad
J_{4,1} = \rho_1 e_1 - \rho_2 e_2 + \tfrac12 (\rho_1-\rho_2) u^2,
$$

$$
J_{4,2} = \alpha_1\bigl(\rho_{1,T} e_1 + \rho_1 e_{1,T}\bigr),\qquad
J_{4,3} = \alpha_2\bigl(\rho_{2,T} e_2 + \rho_2 e_{2,T}\bigr),
$$

$$
J_{4,5} = \alpha_1\bigl(\rho_{1,p}e_1 + \rho_1 e_{1,p}\bigr) + \alpha_2\bigl(\rho_{2,p}e_2 + \rho_2 e_{2,p}\bigr) + \tfrac12 \rho_p u^2 .
$$

이상기체의 4 EOS 도함수는 명시적 닫힌 형식이며, SG 와 NASG 의 도함수는 (2.9)–(2.10) 의 기호적 미분으로 얻은 뒤 단위 시험(`tests/test_eos_derivatives.py`)에서 중심차분과 비교 검증한다.  (2.11) 의 모든 entry 는 분석적으로 평가된다 — autograd 또는 유한차분 fallback 은 없다.  Production 증거에서는 Rusanov fallback 도 비활성화되어 있어, 야코비안 불일치가 발생하면 즉시 Newton 발산으로 드러난다.

### 1.5 음속과 음향 임피던스

phase 별 음속은 (2.8)/(2.9)/(2.10c) 의 \(c_k\) 이다.  SLAU2 가 사용하는 frozen mixture 음속은 Wood-style 조화평균이다:

$$
\frac{1}{\rho c_{\text{mix}}^{2}} = \frac{\alpha_1}{\rho_1 c_1^{2}} + \frac{\alpha_2}{\rho_2 c_2^{2}}.
\tag{2.12}
$$

음향 임피던스 \(Z_k = \rho_k c_k\) 는 물질 계면에서의 반사·투과 계수를 결정한다.  07_B Air-Water 케이스는 의도적으로 가혹하며 \(Z_2/Z_1 \approx 3.34\times 10^{3}\) 이다.

### 1.6 연속 영역의 압력평형 불변량

\(t=t_0\) 에서 공간 상태가 \(p\equiv p_0\) 일정, \(u\equiv u_0\) 일정이고 \(\alpha_1(x), T_1(x), T_2(x)\) 가 임의라고 하자.  (2.2)–(2.5) 에 대입하면 — EOS 폐쇄가 \((\rho_k,T_k)\) 와 \(p\) 의 일관성을 유지하는 한 — \(\partial_t p \equiv 0\), \(\partial_t u \equiv 0\) 이고, \(\alpha\) 와 phase mass 방정식은 순수 이류 \(\partial_t f + u_0\partial_x f = 0\) 으로 환원된다.  따라서 이 \(p_0\)-\(u_0\) 상태는 연속 PDE 의 불변 다양체 (manifold) 이다.  §3.6 (PE target recovery) 가 다루는 것은 이 다양체의 이산 보존이다.

---

## 2. 유한체적 차분화 (Finite-volume discretization)

### 2.1 셀 평균

\([x_L,x_R]\) 을 동일 폭 \(\Delta x\) 의 셀 \(C_i = [x_{i-1/2}, x_{i+1/2}]\) 로 \(N\) 개 분할하고

$$
\overline{\mathbf{U}}_i(t) = \frac{1}{\Delta x}\int_{C_i}\mathbf{U}(x,t)\,dx
\tag{3.1}
$$

를 정의한다.  (2.2)–(2.4) 를 \(C_i\) 에 적분하면 보존형 반이산화

$$
\frac{d\overline{\mathbf{U}}_i}{dt} = -\frac{\mathbf{F}_{i+1/2}-\mathbf{F}_{i-1/2}}{\Delta x} + \mathbf{H}_i ,
\tag{3.2}
$$

물리적 플럭스는

$$
\mathbf{F}(\mathbf{U}) = \bigl(\alpha_1\rho_1 u,\ \alpha_2\rho_2 u,\ \rho u^2 + p,\ (\rho E+p)u,\ 0\bigr)^{\!\top},
\tag{3.3}
$$

이고, \(\alpha_1\) 의 비보존 source 는

$$
\mathbf{H}_i = \bigl(0,0,0,0,\ -u_i\,(\partial_x\alpha_1)_i + (\alpha_1+D_1)_i (\partial_x u)_i\bigr)^{\!\top}.
\tag{3.4}
$$

### 2.2 연산자 분리: 명시 물질 vs. 함의 음향

공간 잔차를 다음과 같이 분리한다:

$$
\mathcal{L}(\mathbf{W}) = \mathcal{L}_E(\mathbf{W}) + \mathcal{L}_I(\mathbf{W}),\qquad
\frac{d\mathbf{U}}{dt} + \mathcal{L}_E(\mathbf{W}) + \mathcal{L}_I(\mathbf{W}) = 0,
\tag{3.5}
$$

**명시 연산자** \(\mathcal{L}_E\) 는 \(\alpha\), phase mass \(q_1,q_2\), 관성 운동량 \(\rho u^2\), 운동+APEC 에너지를 SLAU2 face velocity \(u_f\) 로 이류시키고, **함의 연산자** \(\mathcal{L}_I\) 는 acoustic 압력경사 및 압력일 항을 운반한다:

$$
\mathcal{L}_I^{(\rho u)} = \frac{p_{i+1/2}-p_{i-1/2}}{\Delta x},
\qquad
\mathcal{L}_I^{(\rho E)} = \frac{(p u)_{i+1/2}-(p u)_{i-1/2}}{\Delta x}.
\tag{3.6}
$$

이 분리는 acoustic CFL 이 implicit step 의 무제약을 따르되 (즉 \(c\Delta t/\Delta x\) 제약 없음) material CFL — \(|u|\Delta t/\Delta x\) 에 비례 — 이 실제 step 을 결정하도록 한다.  저마하 다상유동에서 IMEX 를 사용하는 표준적 동기이다.

---

## 3. 시간적분: IMEX-SSP3(4,3,3)

### 3.1 Pareschi-Russo SSP3 stage residual

본 시간적분기는 Pareschi-Russo 의 3차 SSP IMEX 이며, Boscheri-Pareschi 가 사용한 SSP3(4,3,3) tableau 를 따른다.  부호규약은 (3.5): \(d\mathbf{U}/dt + \mathcal{L}_E + \mathcal{L}_I = 0\).

**계수** (`solver/five_eq_IMEX/time_integrator.py` 에서):

$$
\gamma = 0.241694260788213,\qquad
\beta = 0.060423565197050,\qquad
\eta = 0.129152869605900,
$$

$$
\delta = \tfrac12 - \beta - \eta - \gamma .
$$

**명시 Butcher 행렬** \(A_E\) (하삼각):

$$
A_E =
\begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & \tfrac14 & \tfrac14 & 0
\end{pmatrix}.
\tag{4.1}
$$

**함의 Butcher 행렬** \(A_I\) (대각 \(\gamma\) 의 하삼각):

$$
A_I =
\begin{pmatrix}
\gamma & 0 & 0 & 0\\
-\gamma & \gamma & 0 & 0\\
0 & 1-\gamma & \gamma & 0\\
\beta & \eta & \delta & \gamma
\end{pmatrix}.
\tag{4.2}
$$

**최종 갱신 가중치** (명시·함의 동일):

$$
b_E = b_I = \bigl(0,\ \tfrac16,\ \tfrac16,\ \tfrac23\bigr).
\tag{4.3}
$$

대각 \(\gamma\) 는 고정이며, \(b_E=b_I\) 등식은 stage blending 일관성 (명시·함의 기여가 동일 가중치로 누적) 을 부여한다 — 3차 정확도와 SSP 성질에 필수.

### 3.2 Stage 잔차 방정식

각 stage \(s = 1,\dots,4\) 에 대해 보존형 *stage target* 을

$$
\mathbf{U}_s^{\!*} = \mathbf{U}^{\,n} - \Delta t \sum_{j<s}\Bigl[(A_E)_{sj}\,\mathcal{L}_E(\mathbf{W}_j) + (A_I)_{sj}\,\mathcal{L}_I(\mathbf{W}_j)\Bigr]
\tag{4.4}
$$

로 정의한다.  미지의 stage 상태 \(\mathbf{W}_s\) 는 다음 함의 잔차의 영점이다:

$$
\mathcal{R}_s(\mathbf{W}_s) \;:=\; \frac{\mathbf{U}(\mathbf{W}_s) - \mathbf{U}_s^{\!*}}{(A_I)_{ss}\,\Delta t} \;+\; \mathcal{L}_I(\mathbf{W}_s) \;=\; \mathbf{0}.
\tag{4.5}
$$

\(\Delta\tau \equiv (A_I)_{ss}\Delta t = \gamma\Delta t\) 로 두면, 반복점 \(\mathbf{W}_s^{(m)}\) 에서의 선형화 Newton 시스템은

$$
\Bigl[\frac{1}{\Delta\tau}\,\frac{d\mathbf{U}}{d\mathbf{W}}(\mathbf{W}_s^{(m)}) \;+\; \mathbf{J}_I(\mathbf{W}_s^{(m)})\Bigr]\,\delta\mathbf{W}
\;=\; -\,\mathcal{R}_s(\mathbf{W}_s^{(m)}),
\tag{4.6}
$$

\(\mathbf{J}_I = \partial\mathcal{L}_I/\partial\mathbf{W}\) 는 함의 플럭스 야코비안 (분석형; 운동량·에너지 행만 비영) 이다.  Newton 갱신은

$$
\mathbf{W}_s^{(m+1)} = \mathbf{W}_s^{(m)} + \lambda^{(m)}\,\delta\mathbf{W},
\tag{4.7}
$$

이며, 양수성 보장 line-search step \(\lambda^{(m)}\in(0,1]\) 은 \(\alpha_1, T_1, T_2 > 0\) 와 EOS 허용성 (예: NASG 의 \(1-b_k\rho_k>0\)) 이 \(\mathbf{W}_s^{(m+1)}\) 에서 모두 성립할 때까지 \(\lambda\) 를 절반씩 줄여 결정한다.

수렴 기준: 상대 갱신 norm

$$
\bigl\|\delta\mathbf{W}\bigr\|_{\!2,\text{rel}} \;:=\; \sqrt{\frac{1}{N}\sum_i\sum_v\Bigl(\frac{\delta W_{v,i}}{\|W_v\|_\infty + \varepsilon_v}\Bigr)^{\!2}}\;<\;10^{-10}.
\tag{4.8}
$$

Production 증거에서, 07_B Air-Water (\(N=400\)) 의 stage 당 평균 Newton iter 수는 2-3 이며 line-search 발동은 없다.  24_H 극초음속 (\(N=400\)) 에서는 4-6 으로 증가한다.

### 3.3 최종 SSP 결합

4 stage 종료 후

$$
\mathbf{U}^{\,n+1} = \mathbf{U}^{\,n} - \Delta t \sum_{s=1}^{4}\Bigl[(b_E)_s\,\mathcal{L}_E(\mathbf{W}_s) + (b_I)_s\,\mathcal{L}_I(\mathbf{W}_s)\Bigr].
\tag{4.9}
$$

\(\mathbf{U}^{\,n+1}\) 로부터 \(\mathbf{W}^{\,n+1}\) 은 §3.6 의 분석형 역변환으로 회복된다.

### 3.4 시간 step 선택

Production 코드는 material-CFL 한계를 사용한다:

$$
\Delta t = \mathrm{CFL}\,\frac{\Delta x}{\max_i|u_i|}, \qquad \mathrm{CFL} = 0.4 .
\tag{4.10}
$$

함의 연산자가 L-stable 이므로 acoustic-CFL 한계는 강제하지 않는다.  본 manuscript §6.7 의 acoustic-CFL **sweep** 는 진단 목적이며 안정성 요건이 아니다.

---

## 4. 공간 차분화 상세 (Spatial discretization details)

### 4.1 SLAU2 형식 material face velocity

`solver/five_eq_IMEX/imex_ad.py::_slau2_faces_np` 가 모든 내부 면에서 사용하는 SLAU2 face state 를 구현한다:

$$
v_{\text{avg}} = \frac{\sqrt{\rho_L}\,u_L + \sqrt{\rho_R}\,u_R}{\sqrt{\rho_L}+\sqrt{\rho_R}},\qquad
\bar c = \tfrac12 (c_L + c_R),\qquad
\bar\rho = \tfrac12(\rho_L+\rho_R),
\tag{5.1}
$$

$$
u_{\text{rms}} = \sqrt{\tfrac12(u_L^{2} + u_R^{2})},\qquad
\hat M = \min\!\Bigl(1,\ \frac{u_{\text{rms}}}{\bar c}\Bigr),\qquad
\chi(\hat M) = (1-\hat M)^{2},
\tag{5.2}
$$

$$
\boxed{\;u_f \;=\; v_{\text{avg}} \;-\; \chi(\hat M)\,\frac{p_R - p_L}{\bar\rho\,\bar c} ,\qquad p_f = \tfrac12(p_L + p_R).\;}
\tag{5.3}
$$

복원된 좌·우 상태 \((u_L,u_R,p_L,p_R,\rho_L,\rho_R)\) 는 T-MLP-u limiter (§4.2) 를 \((T_1,T_2,u,p)\) 에 성분별로 적용한 결과이며, \(\rho_k\) 는 복원된 \((p,T_k)\) 에서 EOS 로 얻는다.  Roe-평균 \(v_{\text{avg}}\) 는 고마하 Roe 극한을 보존한다.  \(\chi(\hat M)\) prefactor 는 고마하에서 0 (순수 Roe flux 회복), 저마하에서 활성 — 이로써 저소산 (low-dissipation) 의 압력-속도 결합을 회복한다.

**Phase mass flux** 도 동일한 \(u_f\) 를 쓴다:

$$
F_{q_k,\,i+1/2} = (\alpha_k)_f\,(\rho_k)_f\,u_f,
\tag{5.4}
$$

여기서 \((\alpha_k)_f\) 는 \(u_f\) 의 부호로 풍상 (upwind) 선택, \((\rho_k)_f\) 는 face 의 \((\hat p_f,\hat T_{k,f})\) 에서 phase 별 EOS 로 복원 — `face_state.py` 의 "ACID-style" face thermodynamics 주석.  이는 고밀도비 계면에서 발생할 수 있는 bulk-density 전송을 방지한다 (중심평균 \(\rho_k\) 라면 계면 셀이 자기 것이 아닌 질량을 받게 됨).

### 4.2 T-MLP-u primitive 복원

각 셀 중심 primitive \(q\in\{T_1,T_2,u,p\}\) 에 대해 \(i+\tfrac12\) 면의 좌측 후보값은 MUSCL 형식

$$
q_{i+1/2}^{L,*} = q_i + \tfrac12\,\psi(r_i)\,(q_{i+1}-q_i),\qquad
r_i = \frac{q_i - q_{i-1}}{q_{i+1} - q_i + \mathrm{sgn}\,\varepsilon}.
\tag{5.5}
$$

기본 limiter 는 **Roe 의 superbee**:

$$
\psi_{\text{SB}}(r) = \max\!\bigl(0,\ \min(2r,1),\ \min(r,2)\bigr).
\tag{5.6}
$$

**T-MLP-u wrapper** 는 3 셀 윈도우의 LMP (Local Maximum Principle) bound 로 후보값을 clip 한다:

$$
q_{i+1/2}^{L} = \mathrm{clip}\!\Bigl(q_{i+1/2}^{L,*},\ \min(q_{i-1},q_i,q_{i+1}),\ \max(q_{i-1},q_i,q_{i+1})\Bigr).
\tag{5.7}
$$

LMP 유도 limiter 를

$$
\psi_{\text{MLP}} =
\begin{cases}
\dfrac{\max(q_{i-1},q_i,q_{i+1}) - q_i}{\tfrac12(q_{i+1}-q_i)} & \text{if }\delta>0,\\[1.2ex]
\dfrac{\min(q_{i-1},q_i,q_{i+1}) - q_i}{\tfrac12(q_{i+1}-q_i)} & \text{if }\delta<0,
\end{cases}
\tag{5.8}
$$

\(\delta = \tfrac12(q_{i+1}-q_i)\) 로 정의하면, 최종 limiter 는

$$
\boxed{\;\psi_{\text{T-MLP-u}}(r) \;=\; \max\!\bigl(0,\ \min(2,\ \psi_{\text{SB}}(r),\ \psi_{\text{MLP}})\bigr).\;}
\tag{5.9}
$$

이 wrapper 는 base TVD limiter 의 압축 영역 \(0\le\psi\le 2\) 를 보존하면서, 3 셀 stencil 안에서 새 primitive 극값의 생성을 막는다.  우측 \(q_{i+1/2}^{R}\) 은 우측 셀 기준 \(r_{i+1}\) 의 거울 대칭으로 구성한다.  대응 코드는 `solver/five_eq_IMEX/limiters.py::t_mlp_u_face_value`.

**Cavitation 안전장치**: 동질의 이중 희박파 (double-rarefaction) 위상에서 superbee 를 van Leer (더 부드러운 base) 로 교체한다.  지역 \(u_x, \alpha_x\) 부호 패턴으로 매개변수 없이 식별한다.  케이스 ID 분기가 아닌 상태-위상 규칙이며, 15_E 에서 가장 두드러지게 동작한다.

### 4.3 Adaptive-BVD 체적분율 수송

\(\alpha_1\) 은 **adaptive-BVD** 로직으로 복원된다: pure 0/1 contact 근처에서는 CICSAM 형 압축 구성, 혼합 영역에서는 bound 된 MUSCL-Hancock TVD 분기를 선택한다.  \(i+\tfrac12\) 면의 지역 indicator 는 셀 stencil \(\alpha\) 경사 최대값과 pure-phase 허용오차 \(\eta_{\text{pure}}=10^{-12}\) 를 사용한다:

$$
\text{interface}_{i+1/2} =
\bigl[\,\min(\alpha_{i-1},\alpha_i,\alpha_{i+1},\alpha_{i+2}) < \eta_{\text{pure}}\,\bigr]\;\lor\;
\bigl[\,\max(\alpha_{i-1},\alpha_i,\alpha_{i+1},\alpha_{i+2}) > 1 - \eta_{\text{pure}}\,\bigr].
\tag{5.10}
$$

Indicator 가 참일 때 CICSAM 압축 분기 사용:

$$
\tilde\alpha_C = \frac{\alpha_C - \alpha_U}{\alpha_D - \alpha_U},\qquad
\tilde\alpha_f^{\text{HC}} = \min\!\Bigl(1,\ \frac{\tilde\alpha_C}{\mathrm{Co}_f}\Bigr),
\tag{5.11}
$$

여기서 \(\mathrm{Co}_f = |u_f|\Delta t/\Delta x\) 는 면별 Courant 수, 풍상 \((U)\) / 중심 \((C)\) / 풍하 \((D)\) 표지는 \(u_f\) 부호를 따른다.  Indicator 가 거짓이면 minmod limiter 의 MUSCL-Hancock TVD 복원으로 혼합 조성 영역의 단조성을 유지한다.

**보존형 flux-corrected sharpening.**  \(\alpha_1\) 의 압축 sharpening 은 \((\alpha_1\rho_1, \alpha_2\rho_2, \rho u, \rho E)\) 의 대응 보정을 유발해 보존변수가 sharpened \(\alpha\) 와 일관되게 한다.  단일 지역 FCT 인수

$$
\theta_{i+1/2} \;=\; \min\!\Bigl(1,\ \min_{v\in\{q_1,q_2,q_3,q_4\}}\Theta_v\Bigr)\;\in[0,1]
\tag{5.12}
$$

가 셀별 허용성 cone (\(0\le\alpha_1\le1\), \(\rho_k>0\), \(p>p_{\text{min}}\), \(T_k>T_{\text{min}}\)) 안에 sharpened 갱신을 가두도록 계산된다.  동일한 \(\theta\) 가 4 개 보존 보정에 곱해지므로 이산 갱신은 정확히 보존적이다.  이 "단일 노브 FCT" 가 \(\alpha\) 만 sharpening 했을 때 contact 에서 비물리적 밀도 spike 가 생기는 것을 막는 핵심이다.

### 4.4 Characteristic 복원 정책

특성변수 (characteristic variables) 로 복원할 수 있지만, **조성 균일 stencil** — \(\alpha_1\) 이 \(\{i-1,i,i+1\}\) 에 걸쳐 엄격한 허용오차 내 일정 — 에서만 적용한다.  Detector 는 stencil 이 물질 계면을 가로지르는 모든 face 에서 거짓을 반환하며, 이때는 EOS 일관 primitive (또는 mixture-scalar) 복원으로 fallback (§4.2).  이 규칙은 모든 검증 케이스에 동일하게 적용된다.

대응 코드는 `solver/five_eq_IMEX/imex_ad.py::_characteristic_recon_allowed` (상태-위상 detector) 와 `_characteristic_mixture_lr_states` (실제 특성 사영).

### 4.5 APEC 에너지 플럭스

이류 에너지 플럭스는 APEC (Adjoint Phasic Energy Coupling) 분해를 사용한다:

$$
F_{\rho E,\,i+1/2}^{\,(\text{APEC})}
= \chi_{1,f}\,F_{q_1,f} + \chi_{2,f}\,F_{q_2,f} + \chi_{a,f}\,F_{\alpha,f} + \tfrac12 u_f^{2}\,F_{\rho,f},
\tag{5.13}
$$

phase 별 엔탈피류 계수는

$$
\chi_{k,f} = e_{k,f} + \frac{\rho_{k,f}\,e_{k,T,f}}{\rho_{k,T,f}},\qquad
\chi_{a,f} = -\frac{\rho_{1,f}^{2}\,e_{1,T,f}}{\rho_{1,T,f}} + \frac{\rho_{2,f}^{2}\,e_{2,T,f}}{\rho_{2,T,f}},
\tag{5.14}
$$

face 마다 \((\hat p_f, \hat T_{k,f})\) 에서 EOS 로 평가한다.  \(|\rho_T|\to 0\) 의 fallback (pure-phase 극한) 은 더 단순한 Allaire 형 \(e_{f,\text{up}}F_{q}\) 로, \((\alpha_k)_f<\eta_\text{pure}\) 일 때 자동 활성된다.  압력일 항 \(p\,u\) 는 \(F_{\rho E}^{\,(\text{APEC})}\) 에 포함되지 **않는다** — 이는 함의 연산자 \(\mathcal{L}_I\) (3.6) 의 몫이다.  대응 코드는 `solver/five_eq_IMEX/energy_flux.py`.

### 4.6 함의 음향 플럭스

함의 연산자 (3.6) 는 \(p\) 와 \(p u\) 의 중심차분을 사용한다:

$$
(\mathcal{L}_I^{\,\rho u})_i = \frac{p_{i+1/2} - p_{i-1/2}}{\Delta x},\qquad
(\mathcal{L}_I^{\,\rho E})_i = \frac{(pu)_{i+1/2} - (pu)_{i-1/2}}{\Delta x},
\tag{5.15}
$$

면값은

$$
p_{i+1/2} = \tfrac12 (p_i + p_{i+1}) + \alpha_{\text{RC}}\,\bigl[\bar\rho\,\bar c\,(u_i - u_{i+1})\bigr],
\tag{5.16}
$$

$$
u_{i+1/2}^{\,\text{(implicit)}} = \tfrac12 (u_i + u_{i+1}) + \alpha_{\text{RC}}\,\bigl[(p_{i+1}-p_i)/(\bar\rho\,\bar c)\bigr].
\tag{5.17}
$$

\(\alpha_{\text{RC}}\) prefactor (Rhie-Chow 형식) 는 1차원에서 보통 0 으로 둔다 — 명시 분기에 SLAU2 가 쓰이면 중심차분 형식 자체가 checkerboard mode 로부터 자유롭기 때문이다; 진단 목적으로만 노출되어 있다.

---

## 5. 압력평형 target 회복 (Pressure-equilibrium target recovery)

### 5.1 연속 불변량 재서술

§1.6 에서, \(p_0,u_0\) 가 일정하고 \(\alpha_1(x), T_1(x), T_2(x)\) 가 임의인 상태는 연속 5-방정식 시스템의 불변량이다: \(\partial_t p \equiv 0\), \(\partial_t u \equiv 0\), 그리고 \(\alpha_1, T_1, T_2\) 는 \(u_0\) 로 이류된다.  차분화는 이 다양체를 보존해야 한다.

### 5.2 단순 U→W 역변환의 실패

이산 step 에서 \(t^n\) 의 정확 PE 상태가 (4.9) 로 \(t^{n+1}\) 까지 진행된 뒤 \(\mathbf{U}^{\,n+1}\) 에서 \(\mathbf{W}^{\,n+1}\) 로 변환된다.  변환은 5차원 비선형 근 찾기 (EOS 의 \((\rho_k,T_k,p)\) 는 함의) 이며, 보존변수가 PE 제약을 정확히 만족하더라도 부동소수 round-off \(\varepsilon\,\kappa(d\mathbf{U}/d\mathbf{W})\) 가 \(p\) 와 \(u\) 로 누설된다.  수천 step 누적되면 이산 압력·속도가 기계 정밀도 위로 흘러내리고 하류의 음향 진단을 오염시킨다.

### 5.3 PE detector

이산 상태가 다양체 위에 있다고 detector 가 판정하는 조건은

$$
\frac{\max_i|p_i - \langle p\rangle|}{\langle p\rangle} < \tau_p \quad\text{and}\quad
\max_i |u_i - \langle u\rangle| < \tau_u\,\langle c\rangle,
\tag{6.1}
$$

기본 허용오차 \(\tau_p = 10^{-10}\), \(\tau_u = 10^{-10}\), \(\langle\cdot\rangle\) 은 공간평균.  두 조건이 모두 성립할 때만 회복이 활성화된다.

### 5.4 다양체 사영

Detector 발동 시 보존-원시 변환을 다음과 같이 제약한다:

$$
p^{\,n+1} \leftarrow p_0,\qquad u^{\,n+1} \leftarrow u_0,
\tag{6.2}
$$

\(\alpha_1, T_1, T_2\) 는 보존 target \((q_1,q_2,q_5)_i\) 로부터 EOS 로 셀별 회복 (PE 다양체 위에서는 잘 조건화된 3-방정식 근 찾기) 한다.  구체적으로:

$$
\alpha_1^{\,n+1}_i = q_5^{\,n+1}_i,
$$

$$
T_1^{\,n+1}_i \text{ s.t. } \alpha_1\rho_1(p_0,T_1^{\,n+1})_i = q_1^{\,n+1}_i,
$$

$$
T_2^{\,n+1}_i \text{ s.t. } (1-\alpha_1)\rho_2(p_0,T_2^{\,n+1})_i = q_2^{\,n+1}_i.
$$

이는 **공간 remap 이 아니다**.  각 셀의 회복은 지역적이며, 셀 간 값 복사는 없고, 보존변수는 변하지 않는다.  특히 셀의 질량·운동량·총에너지는 회복에 의해 정확히 보존된다 (상태공간의 동일 점에 대한 재매개화).

### 5.5 이것이 트릭이 아닌 원리적 절차인 이유

세 가지 독립 근거.  (i) **연속 불변량.** \(\partial_t p=\partial_t u=0\) 은 PDE 의 성질이며, 회복은 그 불변량의 이산 실현이다.  (ii) **보존성.** 회복은 primitive *재매개화* 만 제약하며, 보존변수와 이산 플럭스는 변하지 않는다.  (iii) **공간 remap 아님.** Production 증거는 `FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0` 으로 실행 — 주기적 remap 단축은 명시적으로 비활성.  Ablation (본 manuscript §6.6) 은 회복 없이 PE 케이스의 p/u Linf 가 적분 시간 동안 기계 정밀도 위로 수 차수 drift 함을 보인다.

### 5.6 의사코드

```text
function step(U^n, dt):
    # Stage loop
    for s in 1..4:
        U_star = U^n − dt · Σ_{j<s} [ (A_E)_{sj} · L_E(W_j) + (A_I)_{sj} · L_I(W_j) ]
        W_s = NewtonSolve(R_s(W) := (U(W) − U_star) / (γ · dt) + L_I(W) = 0,
                          initial_guess = W_n,
                          jac = (1/(γ·dt))·dU/dW + dL_I/dW,
                          line_search = 양수성 보장)
    # SSP 결합
    U^{n+1} = U^n − dt · Σ_{s=1..4} [ b_E_s · L_E(W_s) + b_I_s · L_I(W_s) ]
    # 보존-원시 회복
    if PE_detector(U^{n+1}):
        W^{n+1} = PE_constrained_inversion(U^{n+1}, p_0, u_0)   # §5.4
    else:
        W^{n+1} = analytic_inversion(U^{n+1})                   # 표준 경로
    return U^{n+1}, W^{n+1}
```

`solver/five_eq_IMEX/time_integrator.py::imex_ssp3_step` 이 대응 루틴이며, PE detection 과 projection 은 `_pe_projection_allowed`, `_project_conservative_target_to_pe` 에 있다.

---

## 6. Production 알고리즘 요약

```text
PRODUCTION 설정 (환경 변수):
    TIME_INTEGRATOR    = imex_ssp3
    ALPHA_SCHEME       = adaptive_bvd
    PRIMITIVE_SCHEME   = tmlpu
    TMLPU_TVD          = superbee
    MATERIAL_FLUX      = slau2
    PRESSURE_CLOSURE   = regime_auto
    CHARACTERISTIC_RECON = 1   (조성 균일 stencil 에만)
    RUSANOV_FALLBACK   = 0     (비활성)
    UNIFORM_PERIODIC_REMAP = 0 (비활성)

각 시간 step Δt 에 대해:
    1. 분석형 dU/dW 야코비안으로 U → W 회복 (§1.4).
    2. 경계 상태 적용; c_k, c_mix, Z_k 계산.
    3. T-MLP-u + superbee + LMP 로 primitive (T_1, T_2, u, p) 복원 (§4.2).
    4. Adaptive-BVD + 보존형 FCT limiting 으로 α_1 복원 (§4.3).
    5. 각 face 에서 SLAU2 face velocity u_f 와 압력 p_f 구성 (§4.1).
    6. IMEX-SSP3 stage 루프:
        a. s = 1..4 에 대해 U_star = U^n − dt Σ_{j<s} (A_E_sj L_E(W_j) + A_I_sj L_I(W_j))
        b. 양수성 line-search 의 Newton 으로 R_s(W_s) = 0 풀기 (§3.2).
    7. Stage 결합: U^{n+1} = U^n − dt Σ (b_E_s L_E(W_s) + b_I_s L_I(W_s)) (§3.3).
    8. U^{n+1} → W^{n+1} 회복; PE detector 발동 시에만 PE 회복 사용 (§5).
    9. 케이스별 진단 metric 기록 + results/1D/{case}/diff_vs_exact.png 덮어쓰기.
```

---

## 7. 차원 정합성 점검

조합은 차원적으로 정합한다.  음속 \([c]=L/T\); 압력 \([p]=M/(L T^2)\); 밀도 \([\rho]=M/L^3\); SLAU2 χ 보정은 무차원; LMP-clipped face 값은 \(q\) 와 같은 단위; (4.6) 의 Newton 시스템은 시간팽창 block 에서 \([(d\mathbf{U}/d\mathbf{W})]\,/[T]\), 함의 플럭스 block 에서 \([\partial\mathcal{L}_I/\partial\mathbf{W}]\) 단위 — 둘 다 단위시간·단위 primitive 당 보존변수의 척도로 정합.  PE 회복 사영은 상태공간의 무차원 재매개화이다.

---

## 8. 구현 파일 지도

| 개념 | 파일 | 주요 심볼 |
|---|---|---|
| 지배방정식 / primitive 선택 | `solver/five_eq_IMEX/main.py`, `nd_solver.py` | `solve(...)` 진입점 |
| EOS facade (Ideal/SG/NASG) | `solver/five_eq_IMEX/eos_facade.py` | `to_eos(...)`, `EOSPair` |
| 분석형 dU/dW 야코비안 | `solver/five_eq_IMEX/jacobian.py` | `dUdW_analytic`, `prim_to_cons_W`, `cons_to_prim_W` |
| IMEX-SSP3 stage 잔차 | `solver/five_eq_IMEX/time_integrator.py` | `imex_ssp3_step`, `SSP3_A_E`, `SSP3_A_I`, `SSP3_B_E`, `SSP3_B_I` |
| SLAU2 face velocity | `solver/five_eq_IMEX/imex_ad.py` | `_slau2_faces_np` (line 599-658) |
| T-MLP-u + superbee | `solver/five_eq_IMEX/limiters.py` | `t_mlp_u_face_value` |
| Adaptive-BVD α 수송 | `solver/five_eq_IMEX/face_state.py`, `imex_ad.py` | `face_state(...)`, `_adaptive_bvd_branch` |
| APEC 에너지 플럭스 | `solver/five_eq_IMEX/energy_flux.py` | `total_energy_flux` |
| 함의 음향 연산자 | `solver/five_eq_IMEX/imex_ad.py` | `_L_I(...)` |
| Newton + line search | `solver/five_eq_IMEX/newton.py` | `newton_solve(...)` |
| PE detector / projection | `solver/five_eq_IMEX/time_integrator.py` | `_pe_projection_allowed`, `_project_conservative_target_to_pe` |

---

*수치기법 문서 종료.*
