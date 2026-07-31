# Validation Case — 1D Mach-10 Air Shock / Water Interface Interaction

> **출처:** Denner et al., JCP 367 (2018), §7.4.4
> **목적:** Mach 10 강한 충격파와 공기-물 계면 상호작용 검증

---

## 1. 물리 모델

- **유체:** 공기(Air) + 물(Water) — sharp material interface
- **점성:** 비점성(inviscid)
- **상태방정식:** 이상기체(공기) + NASG EOS(물)
- **지배방정식:** 압축성 5-equation / pressure-equilibrium two-material model
- **중요:** 이 문제는 균질 혼합물 shock 문제가 아니다. 초기 계면은 공기/물 sharp interface이며, Fig.23 기준해는 shock-interface interaction Riemann solution이다.

---

## 2. 물성

| 물성 | 공기(Air) | 물(Water) |
|------|----------|----------|
| model [-] | ideal | NASG |
| $\gamma$ [-] | 1.4 | 1.187 |
| $P^\infty$ [Pa] | - | 7.028E8 |
| $b$ [m³/kg] | 0 | 6.61E-4 |
| $k_v$ / $c_v$ | 717.5 | 3610.0 |
| $\eta$ [J/kg] | 0 | -1.177788E6 |
| $\rho_{0}$ [kg/m^3] | 1.157 | 998.0 |
| $c_{0}$ [m/s] | 347.8 | 1567.335 |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 논문 격자 수 | $N = 1000$ ($\Delta x = 10^{-3}$ m) |
| 검증 코드 기본 격자 수 | $N = 400$ ($\Delta x = 2.5\times10^{-3}$ m) |
| 차원 | 1-D |

---

## 4. 초기 조건

- 초기 shock 위치: $x_{s,0}=0.25$ m
- 초기 air-water interface 위치: $x_{\Gamma,0}=0.50$ m
- 충격파 Mach수: $M_s=10$
- 충격파 속도:

$$
u_s = M_s \sqrt{\gamma_{air} p_{II}/\rho_{air,II}}
$$

| 구역 | 위치 | 의미 | 유체 | 속도 | 압력 | 밀도 |
|------|------|------|------|------|------|------|
| Region I | $0 \le x < 0.25$ | 충격파 후방 공기 (post-shock air) | air | 2869.3 m/s | 1.165E7 Pa | 6.614 kg/m^3 |
| Region II | $0.25 \le x < 0.50$ | 충격파 전방 공기 (pre-shock air) | air | 0.0 m/s | 1.0E5 Pa | 1.157 kg/m^3 |
| Region III | $0.50 \le x \le 1.0$ | 정지한 물 (initially quiescent water) | water | 0.0 m/s | 1.0E5 Pa | 998 kg/m^3 |

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 좌/우 경계 조건 | transmissive |
| 논문 time step | $\Delta t = 10^{-8}$ s |
| 검증 코드 time step | adaptive CFL 기반 |
| 논문 reference 출력 시각 | shock가 계면과 상호작용한 뒤 $t_{after}=2.78\times10^{-4}$ s |
| 현재 검증 출력 시각 | **$t_{after}=2.42\times10^{-4}$ s** |
| shock 도달 시각 | $t_{hit}=(x_{\Gamma,0}-x_{s,0})/u_s$ |
| 전역 최종 시각 | **$t_{end}=t_{hit}+2.42\times10^{-4}=3.1386917678\times10^{-4}$ s** |

주의: Denner Fig.23의 caption은 "at $t=2.78\times10^{-4}$ s after the shock wave has interacted with the interface"이다. 따라서 초기 shock 위치 $x_{s,0}=0.25$ m에서 계산을 시작하는 경우, $2.78\times10^{-4}$ s를 전역 최종시각으로 사용하면 reference보다 이른 시각의 해를 비교하게 된다.

2026-05-01 검증에서는 우측 경계 근처에 상압 물 영역이 충분히 남도록 논문 caption 시각보다 이른 $t_{after}=2.42\times10^{-4}$ s를 사용한다. 이때 NASG exact 기준 transmitted shock 위치는 $x\approx0.900$ m이고, 오른쪽에 약 0.10 m의 undisturbed water 구간이 남는다.

---

## 7. 출력 변수 및 결과 비교

- 거리 $x$에 따른 최종 결과와 exact solution을 비교해 `results/1D/25_H/diff_vs_exact.png`로 저장한다.
- exact solution은 이미지 digitization이 아니라 interaction 시각의 two-material Riemann problem을 직접 풀어서 생성한다.
- `validation/1D/25_ref.png`는 시각적 reference로만 사용한다.
- 현재 검증 드라이버 `.codex-loop/verify_08_26_acceptance.py --case 25`는 generated exact를 `results/1D/25_H/reference_exact_25.csv`에 저장한다.

Exact Riemann states:

- left state: Region I post-shock air
- right state: Region III quiescent water
- EOS: ideal air / NASG water
- interaction origin: $(x,t)=(x_{\Gamma,0},t_{hit})$

현재 NASG exact 주요 위치/상태:

| 양 | 값 |
|----|----|
| $p^*$ | $8.61337365\times10^7$ Pa |
| $u^*$ | 52.1406 m/s |
| reflected shock 위치 | $x\approx0.22698$ m |
| contact/interface 위치 | $x\approx0.51262$ m |
| transmitted shock 위치 | $x\approx0.90011$ m |

| 그래프 | 변수 | 관찰 |
|--------|------|------|
| (a) | 밀도 $\rho$ | 물에서의 충격파 압축 |
| (b) | 압력 $p$ | 계면에서 압력 전달 |
| (c) | 속도 $u$ | 물 매체에서의 입자 속도 |

추가 PASS 위치 기준:

- reflected shock 위치: exact 대비 12 cells 이내
- air-water contact/interface 위치: exact 대비 80 cells 이내
- transmitted shock 위치: exact 대비 80 cells 이내

현재 위치 기준은 수치 확산을 고려해 shock/contact의 폭은 직접 제한하지 않고, 각 wave의 대표 gradient peak 위치만 비교한다.

추가 interface 안정성 기준:

- contact 주변 narrow-band($|x-x_\Gamma^{exact}|\le 0.05$ m)에서 pressure/velocity는 Riemann star plateau를 유지해야 한다.
- density는 물리적 air-star/water-star jump를 제외한 추가 TV와 overshoot가 작아야 한다.
- PASS 조건:
  - `interface_instability <= 0.30`
  - `interface_p_linf <= 0.08`
  - `interface_u_linf <= 1.5`
  - `interface_rho_tv_excess <= 0.30`
  - `interface_rho_overshoot <= 0.25`

여기서 `interface_rho_tv_excess`는 contact band의 총변동량에서 exact density jump를 뺀 값을 exact density jump로 나눈 값이다. 즉 단조로운 sharp/smeared contact는 허용하지만, Gibbs-like ringing 또는 nonphysical overshoot/undershoot는 FAIL 처리한다.

---

## 8. 참고사항

- Mach 10은 §7.4.3의 Mach 1.22보다 훨씬 강한 충격 — 극단 조건 테스트
- 물의 고밀도 + 강한 충격 → 높은 압력이 물에 전달 (수중 충격파)
- 수중 폭발(underwater explosion) 물리의 1D 모델 문제
