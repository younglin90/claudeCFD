# T-MLP-u Reviewer Assessment for Mach 3 Forward-Facing Step

## 0. Reviewer Position

이 문서는 T-MLP-u reconstruction/limiter를 JCP, CMAME, Computers & Fluids, IJNMF 수준의 수치해석 논문 관점에서 검토하기 위한 reviewer-style assessment이다. 목적은 T-MLP-u를 홍보하는 것이 아니라, 동일한 governing equations, flux, time integration, mesh, CFL, boundary condition, gradient reconstruction, residual tolerance, post-processing 조건에서 MLP-u1 대비 어떤 수치적 장점이 검증 가능한지를 명확히 하는 것이다.

현재 검증 문제는 Mach 3 forward-facing step이다. 실제 계산 결과가 주어지지 않은 항목은 모두 **확인 필요**로 표시한다. 허위 수치, 조작된 성능 우위, 검증되지 않은 보편적 우월성 주장은 금지한다.

---

## 1. Theoretical Novelty

T-MLP-u의 핵심은 기존 MLP-u1의 vertex-based local maximum principle을 유지하면서, face-normal projection과 transverse correction을 통해 skewed/non-orthogonal face에서 face/vertex increment를 더 일관되게 구성하려는 시도이다.

| 구성 요소 | MLP-u1 대비 차이 | novelty 수준 | 논문 주장 가능성 | reviewer risk |
|---|---|---:|---|---|
| \(t^*\) face-normal projection | face center \(m_f\)를 \(c_L \to c_R\) line 위의 normal-equivalent point \(f_0\)로 투영 | 중간 | skewed face에서 normal jump와 transverse correction을 분리한다는 점은 주장 가능 | \(t^*\notin[0,1]\), \(d_{LR}\cdot n_f\approx0\)에서 불안정하면 공격받음 |
| \(f_0=c_L+t^*d_{LR}\) 기준 correction | vertex/face increment를 \(f_0\) 기준 normal + transverse 형태로 분해 | 중간 | 비정렬 격자에서 face-centered reconstruction을 정교화했다는 framing 가능 | 단순 geometric rearrangement로 보일 수 있음 |
| \(\nabla\phi_{f,corr}\) 비직교 보정 | interpolated gradient의 \(e_{LR}\) 방향 성분을 cell-center jump와 일치시키려 함 | 중간-높음 | skewness/non-orthogonality correction이 MLP limiter와 결합된 점은 기여 가능 | sign convention, \(\beta\) 정의가 엄밀하지 않으면 ad hoc으로 보임 |
| vertex MLP bound + TVD ratio \(r\) 결합 | multidimensional vertex bound와 1D TVD ratio를 동시에 사용 | 낮음-중간 | shock 근방에서 boundedness와 compressiveness를 조절한다는 claim 가능 | 1D \(r\)이 회전/비정렬 격자에서 directional bias를 유발할 수 있음 |
| \(\psi=\max(0,\min(\alpha r,\alpha,\psi_{TVD}))\) | TVD-style limiter를 vertex bound와 coupled | 낮음 | 기존 TVD limiter의 변형으로 설명 가능 | boundedness 보장이 불명확하면 핵심 약점 |
| pathological grid 인식 | \(|d_{LR}\cdot n_f|\to0\)를 명시적으로 위험 조건으로 분류 | 낮음 | robust implementation requirement로는 중요 | fallback이 수학적으로 정의되지 않으면 완성도 부족 |
| smooth region accuracy 보존 | \(|\Delta\phi_{V_i}|\)가 scale 대비 작으면 limiter 비활성 | 중간 | limiter over-activation 감소 가능성 주장 가능 | fixed tolerance이면 scale invariance 위반 |
| shock/discontinuity boundedness | vertex min/max bound로 local DMP를 목표 | 중간 | overshoot/undershoot 감소를 검증하면 설득 가능 | overly dissipative limiter라는 공격 가능 |

**Reviewer 판단:** T-MLP-u의 novelty는 단일 요소가 아니라, MLP-u1의 vertex-bound framework 안에서 \(t^*\), \(f_0\), corrected face gradient, vertex constraint를 하나의 face-consistent reconstruction으로 묶은 점에 있다. 다만 \(\psi\) 식과 \(r\)-coupling은 기존 TVD logic의 변형으로 보일 수 있으므로, 논문에서는 “새로운 limiter class”보다는 “skewness-aware transverse reconstruction embedded in MLP-u” 정도의 제한된 claim이 더 안전하다.

---

## 2. Mathematical Consistency Check

### 2.1 Core Formulation

주어진 T-MLP-u increment는 다음과 같다.

\[
\Delta\phi_{V_i}
=t^*(\phi_R-\phi_L)
+\nabla\phi_{f,corr}\cdot(V_i-f_0),
\qquad
\Delta\phi_f
=t^*(\phi_R-\phi_L)
+\nabla\phi_{f,corr}\cdot(m_f-f_0).
\]

boundedness를 보장하려면 모든 constrained vertex에 대해

\[
\phi_{V_i}^{\min}\le \phi_L+\psi_L\Delta\phi_{V_i}\le \phi_{V_i}^{\max}
\]

가 성립해야 한다.

### 2.2 Critical Issues and Recommended Fixes

| 이슈 | 위험 | 권장 수정 | 확인 지표 |
|---|---|---|---|
| \(\alpha\) 분모에 \(r\) 재삽입 | \(\alpha\)가 이미 bound ratio 역할인데 \(r\)로 다시 나누면 \(r\approx0\)에서 비정상 증폭/과도 clipping 가능 | bounded candidate를 먼저 \(\psi^{bound}=\max(0,\min(1,B/\Delta\phi))\)로 정의하고, TVD factor는 별도 \(\psi=\min(\psi^{bound},\psi^{TVD})\)로 결합 | bound violation, limiter activation ratio |
| \(\psi=\max(0,\min(\alpha r,\alpha,\psi_{TVD}))\) | \(r<0\)이면 \(\alpha r<0\)로 limiter가 0이 되며 first-order화 가능. \(\alpha<0\) 처리도 불명확 | \(B/\Delta\phi\ge0\)인 bounded ratio만 사용하고 \(r\le0\)에서는 compressive factor 비활성 | smooth region order loss |
| \(den\approx0\) fixed \(10^{-8}\) | scale invariance 위반. \(\phi\) scaling에 따라 limiter behavior 변화 | \(\epsilon_{den}=C_\epsilon\epsilon_{mach}(|\phi_L|+|\phi_R|+|\nabla\phi_L||d_{LR}|+1)\) | affine scaling test |
| \(\Delta\phi_{V_i}\approx0\) fixed criterion | small physical variation과 roundoff를 구분하지 못함 | \(\epsilon_{\Delta}=C_\epsilon\epsilon_{mach}(|\phi_L|+|\phi_R|+\phi_V^{max}-\phi_V^{min}+1)\) | constant/linear preservation |
| \(\psi_{TVD}=2\) downwind | scalar advection에서는 compressive할 수 있으나 compressible shock에서는 pressure/velocity oscillation 및 carbuncle risk | Mach shock 문제에서는 기본 \(\psi_{\max}\le1\) 또는 shock sensor/entropy stable flux와 함께 검증 필요 | post-shock oscillation, pressure positivity |
| \(\beta=\min(1,\hat e_{LR}\cdot n_f/\theta_{\min})\) | \(n_f\) orientation이 뒤집히면 \(\beta<0\) 가능 | \(\beta=\mathrm{clip}(\max(0,\hat e_{LR}\cdot n_f)/\theta_{\min},0,1)\) | left/right symmetry test |
| \(t^*\notin[0,1]\) | extrapolated gradient interpolation으로 overshoot 가능 | clamping은 linear consistency를 해칠 수 있으므로, 먼저 mesh-quality diagnostic을 기록하고 LSQ-consistent fallback 또는 bounded face projection을 정의 | \(t^*\) histogram, fail cell map |
| \(|d_{LR}\cdot n_f|\to0\) | division blow-up | denominator에 arbitrary fallback 금지. mesh-quality threshold와 LSQ face reconstruction을 논문에 명시 | pathological mesh robustness |
| \(\phi_{V_i}^{min/max}\) vs \(\phi_{V,min/max}\) | 표기 혼동으로 bound set이 달라질 수 있음 | 모든 vertex \(V_i\)에 대해 \(\phi_{V_i}^{min},\phi_{V_i}^{max}\)로 통일 | implementation audit |
| \(\psi_L=\min\) over all pairs | smooth extrema에서 excessive clipping 가능 | smooth extrema preservation test 및 limiter activation map 필요. 단, DMP를 유지하려면 min 구조 자체는 유지 가능 | smooth hump/cone peak loss |

### 2.3 Safer Limiter Form

T-MLP-u의 핵심 increment 구조는 유지하되, reviewer 공격을 줄이려면 limiter는 다음처럼 분리하는 편이 더 명확하다.

\[
B_{V_i}=
\begin{cases}
\phi_{V_i}^{\max}-\phi_L, & \Delta\phi_{V_i}>0,\\
\phi_{V_i}^{\min}-\phi_L, & \Delta\phi_{V_i}<0.
\end{cases}
\]

\[
\psi_{V_i}^{bound}
=\max\left(0,\min\left(1,\frac{B_{V_i}}{\Delta\phi_{V_i}}\right)\right).
\]

TVD ratio를 유지할 경우에는

\[
\psi_{V_i}
=\min\left(\psi_{V_i}^{bound},\psi^{TVD}(r)\right),
\qquad
0\le \psi^{TVD}(r)\le \psi_{\max}.
\]

compressible shock-capturing에서는 \(\psi_{\max}=1\)을 기본값으로 두고, \(\psi_{\max}>1\)은 scalar advection 또는 별도 shock-robust flux와 함께 검증된 경우에만 제한적으로 주장하는 것이 안전하다.

---

## 3. Mach 3 Forward-Facing Step Verification Design

Mach 3 step 하나만으로 “보편적 우월성”을 주장할 수는 없다. 그러나 strong shock, reflected shock, slip line, expansion fan, wall interaction이 동시에 존재하므로, T-MLP-u가 MLP-u1 대비 shock-capturing robustness를 개선했다는 제한된 주장은 가능하다.

### 3.1 Required Metric Table

| 지표 | 측정 방법 | T-MLP-u 우위 조건 | 현재 상태 |
|---|---|---|---|
| density contour shock resolution | 동일 contour level에서 shock thickness cell count 측정 | shock transition width가 같거나 더 작고 oscillation이 작음 | 확인 필요 |
| Mach stem 위치 | reference 또는 고해상도 solution 대비 \(x,y\) 위치 오차 | 위치 오차가 MLP-u1보다 작거나 동등 | 확인 필요 |
| triple point 위치 | density gradient 또는 schlieren peak intersection 추적 | reference 대비 오차 감소 | 확인 필요 |
| reflected shock 위치 | shock line fitting 후 distance error | 평균/최대 distance error 감소 | 확인 필요 |
| slip line 선명도 | \(|\nabla\rho|\), vorticity, entropy gradient along slip line | shear layer가 덜 확산되고 비물리 oscillation 없음 | 확인 필요 |
| expansion fan 보존성 | fan region pressure/density smoothness와 angular spread | fan이 과도하게 smeared되지 않고 oscillation 없음 | 확인 필요 |
| post-shock oscillation | shock-normal line probe에서 \(p,\rho\) peak-to-peak amplitude | MLP-u1 대비 amplitude 감소 | 확인 필요 |
| overshoot/undershoot | \(\rho_{\min},p_{\min}\), local extrema beyond reference states | 음압/음밀도 없음, overshoot 감소 | 확인 필요 |
| bound violation | primitive/conserved variable bound check | global violation count와 magnitude 감소 | 확인 필요 |
| total variation 증가율 | \(TV(\rho),TV(p),TV(u),TV(v)\) 시간 변화 | shock sharpness 유지하면서 nonphysical TV growth 감소 | 확인 필요 |
| residual convergence | residual history, final residual, stagnation | 동일 tolerance까지 iteration/time 감소 또는 안정성 개선 | 확인 필요 |
| CPU time | same hardware wall time 또는 normalized cost | overhead가 정확도/안정성 개선 대비 합리적 | 확인 필요 |
| iteration count | final time까지 step count | 동일 CFL에서 같아야 함. adaptive dt면 dt collapse 없어야 함 | 확인 필요 |
| limiter activation ratio | \(\psi<1\), \(\psi\approx0\), \(\psi=1\) cell/face fraction | shock 근방 집중, smooth 영역 과활성화 감소 | 확인 필요 |
| \(\psi\) distribution | shock ROI와 smooth ROI의 histogram | shock: bounded activation, smooth: \(\psi\approx1\) | 확인 필요 |
| grid refinement | \(120\times40\), \(240\times80\), \(480\times160\) | refined grid에서 feature convergence와 oscillation 감소 | 확인 필요 |
| \(L_1/L_2\) error | high-resolution reference 대비 norm | MLP-u1 대비 error 감소 또는 robust feature preservation | 확인 필요 |

### 3.2 Suggested Quantitative Summary Table

| Scheme | Grid | Flux | CFL | \(\rho_{\min}\) | \(p_{\min}\) | Shock width | Mach stem error | Triple point error | Slip-line diffusion | Oscillation amp. | TV growth | CPU normalized | PASS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP-u1 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 1.00 | 확인 필요 |
| T-MLP-u | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 |

T-MLP-u가 우수하다고 주장하려면 최소한 다음 경향이 필요하다.

1. 동일 contour/post-processing에서 shock feature가 더 sharp하거나 동등하다.
2. \(p_{\min}>0\), \(\rho_{\min}>0\)이고 post-shock oscillation이 더 작다.
3. slip line과 expansion fan이 더 잘 보존되거나 최소한 MLP-u1보다 나빠지지 않는다.
4. limiter activation이 smooth 영역 전체에 퍼지지 않고 shock/steep gradient 주변에 국소화된다.
5. CPU overhead가 과도하지 않거나, 정확도/robustness 개선으로 정당화된다.

---

## 4. Fair Comparison Protocol

MLP-u1과 T-MLP-u 비교가 reviewer에게 공정하다고 인정되려면 다음 항목을 고정해야 한다.

| 고정 항목 | 요구 사항 |
|---|---|
| same mesh | 동일 grid size, topology, skewness, boundary face treatment |
| same CFL | global CFL 및 adaptive dt rule 동일 |
| same flux function | HLLE, HLLC, Roe, AUSM, SLAU 계열 중 하나로 고정 |
| same equation of state | 동일 \(\gamma\), thermodynamic closure |
| same boundary conditions | inflow/outflow/wall/step corner 조건 동일 |
| same time integration | RK1, RK2, SSP-RK3 중 하나로 고정 |
| same residual tolerance | final time 또는 residual stop criterion 동일 |
| same gradient reconstruction | LSQ/Green-Gauss, stencil, weighting 동일 |
| same shock sensor | 사용한다면 두 scheme 모두 동일하게 적용 |
| same visualization levels | contour levels, colormap, schlieren scaling 동일 |
| same post-processing | ROI, interpolation, filtering, line probe method 동일 |
| same reference solution | 동일 high-resolution reference 사용 |
| same hardware or normalized CPU | same machine 또는 normalized CPU cost 보고 |

### Parameter Tuning That Would Bias the Comparison

다음은 T-MLP-u만 유리하게 만드는 tuning으로 reviewer가 공격할 수 있다.

| Bias source | 왜 문제인가 |
|---|---|
| T-MLP-u에만 shock flattening 적용 | reconstruction 차이가 아니라 sensor 차이일 수 있음 |
| T-MLP-u에만 boundary first-order fallback 적용 | boundary artifact를 숨기는 case-specific treatment로 보임 |
| MLP-u1은 aggressive flux, T-MLP-u는 robust flux 사용 | flux effect와 limiter effect가 분리되지 않음 |
| T-MLP-u만 CFL 낮춤 | 안정성 개선이 scheme이 아니라 time step 때문일 수 있음 |
| contour level을 scheme별로 다르게 설정 | 시각적 sharpness 조작 |
| 실패한 flux/reconstruction 조합 제외 | selective reporting 문제 |
| T-MLP-u 전용 parameter sweep 후 best만 보고 | tuning budget unfairness |

---

## 5. Ablation Study

T-MLP-u의 각 구성 요소가 실제 성능 향상에 기여했는지 확인하려면 다음 ablation이 필요하다.

| Variant | 구성 | 관찰 지표 | 예상 효과 | 결과 상태 |
|---|---|---|---|---|
| A0 | MLP-u1 baseline | 모든 Mach 3 metric | 기준 성능 | 확인 필요 |
| A1 | MLP-u1 + TVD ratio \(r\) only | oscillation, shock width, smooth clipping | 1D TVD 효과만 분리. directional bias 가능 | 확인 필요 |
| A2 | MLP-u1 + \(t^*\) projection only | skewed face shock alignment, feature position | face-normal projection 효과 확인 | 확인 필요 |
| A3 | MLP-u1 + \(\nabla\phi_{f,corr}\) only | non-orthogonal grid robustness, linear preservation | skewness correction 효과 확인 | 확인 필요 |
| A4 | MLP-u1 + vertex-TVD coupled limiter | bound violation, limiter activation ratio | vertex bound와 TVD coupling의 안정성 확인 | 확인 필요 |
| A5 | full T-MLP-u | 전체 지표 | 구성 요소 간 combined effect 확인 | 확인 필요 |

각 variant는 같은 mesh, flux, CFL, time integration, gradient reconstruction, shock sensor 조건에서 실행해야 한다. A5가 A0보다 좋더라도 A2/A3/A4 중 어떤 요소가 기여했는지 보이지 않으면 novelty claim이 약해진다.

---

## 6. Manuscript Framing

### Weak Claim

> We propose a transverse-corrected extension of the MLP-u reconstruction, denoted T-MLP-u, in which the face-normal projected jump and the skewness-corrected transverse gradient are incorporated into the vertex-based limiting process.

이 claim은 안전하다. 성능 우위를 주장하지 않고 formulation을 설명한다.

### Moderate Claim

> For skewed or non-orthogonal grids, the proposed T-MLP-u formulation provides a more face-consistent high-order increment than the original MLP-u1 while retaining a vertex-based local maximum principle. In the Mach 3 forward-facing step test, this can be assessed through shock position, post-shock oscillation, and bound-violation metrics under identical flux and time-integration settings.

조건부이며 검증 지표를 명시하므로 적절하다.

### Strong Claim

> Under identical discretization settings, T-MLP-u reduces post-shock oscillations and bound violations relative to MLP-u1 without degrading the principal shock structures in the Mach 3 forward-facing step problem.

이 claim은 실제 수치 결과가 있을 때만 가능하다. 현재는 **확인 필요**이다.

### Claims to Avoid

| 피해야 할 표현 | 이유 |
|---|---|
| “T-MLP-u is universally superior to MLP-u1.” | 검증 범위를 넘어선 보편 주장 |
| “The method is TVD in multiple dimensions.” | multidimensional TVD의 엄밀 증명 없으면 위험 |
| “The method eliminates carbuncle.” | flux, grid, BC 영향을 분리해야 함 |
| “The method is parameter-free.” | \(\theta_{\min}\), \(\psi_{TVD}\), tolerance가 존재 |
| “Shock resolution is improved.” | shock thickness/position 지표 없이 주장 불가 |

---

## 7. Reviewer Attack Points

| # | 공격점 | 왜 위험한가 | 방어 또는 추가 검증 |
|---:|---|---|---|
| 1 | MLP-u1 대비 novelty가 작다 | geometric rearrangement로 보일 수 있음 | ablation으로 \(t^*\), \(f_0\), \(\nabla\phi_{f,corr}\) 기여 입증 |
| 2 | \(\beta\) 정의가 ad hoc이다 | \(\theta_{\min}=0.3\) 근거 부족 | mesh-quality parameter로 해석하고 sensitivity study 수행 |
| 3 | \(\beta<0\) 가능성 | orientation error 시 anti-diffusive correction | clipped beta 식으로 수정 |
| 4 | \(r\)-based TVD가 multidimensional symmetry를 해친다 | 1D ratio가 grid direction bias 유발 | rotation/reflection test와 limiter histogram 제시 |
| 5 | \(\alpha\)에 \(r\)를 중복 포함 | boundedness 보장이 불명확 | bounded ratio와 TVD factor를 분리한 수정식 사용 |
| 6 | smooth extrema clipping | vertex min/max limiter가 2차 정확도 저하 가능 | smooth vortex/advection, isentropic vortex test 추가 |
| 7 | pathological grid fallback 미정 | \(d_{LR}\cdot n_f\to0\)에서 blow-up 가능 | mesh-quality diagnostic과 LSQ-consistent fallback 명시 |
| 8 | Mach 3 step 하나로 부족 | benchmark-specific 성능일 수 있음 | Double Mach reflection, LeVeque rotation, isentropic vortex 추가 |
| 9 | CPU overhead | full T-MLP-u가 MLP-u1보다 복잡 | normalized CPU vs accuracy table 제시 |
| 10 | \(\psi_{TVD}=2\) shock 안전성 부족 | compressive limiter가 oscillation/carbuncle 유발 가능 | \(\psi_{\max}=1\) baseline과 \(\psi_{\max}=2\) ablation 비교 |
| 11 | shock sensor/fallback fairness | T-MLP-u만 안정화 장치 사용 가능성 | sensor/fallback을 양쪽에 동일 적용 |
| 12 | grid refinement 불명확 | coarse-grid visual improvement일 수 있음 | \(120\times40\), \(240\times80\), \(480\times160\) refinement trend |
| 13 | reference error 정의 불명확 | shock problem에서 norm 비교가 민감 | 동일 reference, 동일 interpolation, feature error 병행 |
| 14 | conservative consistency | primitive reconstruction이 conservation과 충돌 가능 | flux construction과 conserved update audit |

---

## 8. Result Interpretation Template

### Case A: T-MLP-u가 shock resolution, oscillation, bound violation에서 모두 우수

사용 가능한 문장:

> Under identical mesh, flux, CFL, and time-integration settings, T-MLP-u produced a sharper representation of the principal shock structures while reducing post-shock oscillations and global bound violations compared with MLP-u1. This indicates that the transverse-corrected face increment improves the compatibility between the multidimensional vertex bound and the face reconstruction on the Mach 3 step geometry.

한국어 논문 서술:

> 동일한 격자, flux, CFL, 시간적분 조건에서 T-MLP-u는 MLP-u1 대비 주요 shock 구조를 더 선명하게 유지하면서 post-shock oscillation과 bound violation을 동시에 감소시켰다. 이는 \(t^*\)-기반 normal projection과 \(\nabla\phi_{f,corr}\) transverse correction이 Mach 3 step의 비정렬 shock/벽면 상호작용에서 face reconstruction과 vertex bound의 정합성을 개선했음을 시사한다.

주의: 실제 수치 표가 있어야만 사용 가능. 현재는 **확인 필요**.

### Case B: oscillation은 줄었지만 shock이 더 퍼짐

사용 가능한 문장:

> T-MLP-u reduced nonphysical post-shock oscillations and bound violations, but this robustness was accompanied by a measurable increase in shock thickness. Therefore, the present form should be interpreted as a more dissipative but more robust bounded reconstruction rather than a uniformly sharper shock-capturing method.

한국어 논문 서술:

> T-MLP-u는 MLP-u1 대비 비물리적 post-shock oscillation과 bound violation을 줄였지만, shock thickness가 증가하는 trade-off를 보였다. 따라서 현재 formulation은 모든 shock 구조를 더 선명하게 해상하는 방법이라기보다, 강한 shock 문제에서 boundedness와 robustness를 우선하는 reconstruction으로 해석하는 것이 적절하다.

### Case C: T-MLP-u와 MLP-u1 차이가 작음

사용 가능한 문장:

> The difference between T-MLP-u and MLP-u1 was small for the Mach 3 step problem under the present grid and flux settings. This suggests that the dominant error may be controlled by the Riemann flux, grid resolution, or boundary treatment rather than by the limiter formulation alone.

한국어 논문 서술:

> 현재 격자 및 flux 조건에서 Mach 3 step 결과의 T-MLP-u와 MLP-u1 차이는 제한적이었다. 이는 해당 조건에서 지배적인 오차가 limiter 구조보다는 Riemann flux, grid resolution, 또는 boundary treatment에 의해 결정될 가능성을 시사한다. 따라서 T-MLP-u의 우위는 추가적인 skewed-grid benchmark, smooth-extrema preservation test, 그리고 refinement study를 통해 별도로 검증되어야 한다.

---

## 9. Minimum Evidence Required Before Submission

top-tier SCI 투고를 위해서는 최소한 다음 evidence package가 필요하다.

| Evidence | Required content | 상태 |
|---|---|---|
| Theory audit | constant/linear preservation, affine scaling, DMP check | 확인 필요 |
| Fair Mach 3 table | MLP-u1 vs T-MLP-u, identical settings | 확인 필요 |
| Ablation | A0-A5 variant comparison | 확인 필요 |
| Limiter diagnostics | \(\psi\) distribution in shock/smooth ROI | 확인 필요 |
| Robustness | positivity, bound violation, dt collapse absence | 확인 필요 |
| Refinement | at least 3 grid levels | 확인 필요 |
| Cost | normalized CPU and memory overhead | 확인 필요 |
| Visual reproducibility | same contour levels and post-processing | 확인 필요 |

**Reviewer-level conclusion:** 현재 제시된 이론만으로 T-MLP-u의 “보편적 우월성”은 주장할 수 없다. 그러나 skewed/non-orthogonal face에서 face-normal jump와 transverse gradient를 분리해 vertex-based MLP bound에 결합하는 구조는 제한적 novelty가 있으며, Mach 3 step에서 post-shock oscillation, bound violation, limiter activation localization, shock feature error를 MLP-u1보다 개선한다면 “MLP-u1보다 strong-shock/skewed-grid 상황에서 더 robust한 bounded reconstruction”이라는 moderate claim은 가능하다.
