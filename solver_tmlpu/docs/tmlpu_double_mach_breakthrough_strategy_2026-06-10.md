# T-MLP-u Double Mach Breakthrough Strategy

작성일: 2026-06-10  
상태: 사용자 제안 전략 기록. 계산 재개 없음.  
대상: LeVeque + Mach3 + Double Mach 통합 T-MLP-u reconstruction 연구의 다음 단계

---

## 0. 결론 요약

현재 실패 패턴은 다음과 같이 고정되어 있다.

- Double Mach vortex count가 MLP-u1보다 크지 않음.
- T-MLP-u 후보가 보통 `9 vs 9` 동률, 일부는 `8 vs 9`로 열세.
- checker 지표가 MLP-u1보다 큼.
- primary cluster가 큼.
- shock integrity와 positivity는 대체로 통과.

이 패턴을 종합하면 문제의 본질은 signed-tail correction이 회전(rotation)과 전단(shear)을 충분히 구분하지 못한다는 것이다.

Mach3에서는 rotation과 shear를 모두 보존하는 것이 upper ROI vortex clarity로 읽힌다. 그러나 Double Mach에서는 shear sheet 보존이 connected cluster 병합과 grid-frequency 오염(checker)으로 읽힌다.

따라서 다음 연구의 최우선 방향은 다음이다.

1. P0: face-level 진단 강화
2. A: 속도구배 텐서 기반 rotation/strain 분해 게이트
3. B: correction field 자체의 고주파 필터링
4. C: 장기적으로는 feature-relaxed MLP bound 재정식화

---

## P0. 후보 생성 전 필수 진단

50분짜리 Double Mach candidate를 더 반복하기 전에, v158이 왜 metric을 움직이지 못했는지 먼저 확인해야 한다.

### P0-1. 시간 분해 checker 추적

목표:

- checker가 언제 발생하는지 확인한다.

방법:

- N step마다 checker, primary cluster size, vortex count를 기록한다.
- checker가 초기 shock 형성기에 발생하는지, slip-line 발달 후기 누적으로 발생하는지 분리한다.

해석:

- 초기 shock 형성기 발생: triple point / Mach stem 근방 reconstruction이 원인일 가능성이 높음.
- slip-line 후기 누적: slip-line tangential correction이 원인일 가능성이 높음.

처방이 완전히 달라지므로 이 구분이 중요하다.

### P0-2. 공간 분해 checker map

목표:

- checker 초과분이 어느 공간 영역에서 생기는지 특정한다.

방법:

- cell-local checker indicator를 field로 출력한다.
- TMLP-u - MLP-u1 차분 map과 overlay한다.
- Mach stem 후방, slip line, jet head 중 어느 영역이 문제인지 확인한다.

### P0-3. anti-sheet gate 활성화 진단

기존 v157/v158 anti-sheet가 metric을 움직이지 못한 이유를 분해한다.

기록할 값:

- raw gate count
- final damping count
- damping factor distribution: min, p50, p90
- 활성 face 좌표
- 활성 face와 primary cluster face의 overlap 비율
- signed-tail active face와 anti-sheet active face의 overlap

판단:

- overlap이 0에 가까움: q_pair/contact_gate가 표적을 아예 못 잡은 것.
- overlap이 높은데 metric 불변: face-local damping 자체가 무력하거나 correction 성분이 원인이 아님.

### P0-4. correction field spectrum 진단

목표:

- checker가 해 자체에서 생긴 것인지, signed-tail correction의 grid-frequency 성분이 직접 주입한 것인지 확인한다.

방법:

- `d_ut_signed_L/R` field를 dump한다.
- face-to-face 부호 교번, odd-even pattern, 고주파 correction을 확인한다.

판단:

- correction field가 이미 checkered이면 correction dispersion error가 직접 원인이다.
- 이 경우 B correction-field filter가 우선이다.

### P0-5. vortex counter 동작 검증

목표:

- TMLP-u cluster 내부가 실제 다중 core인지, 아니면 하나의 단일 구조인지 확인한다.

방법:

- connected component threshold sweep.
- Q/local maxima/lambda_ci core count 확인.
- cluster 내부 bridge로 인해 여러 core가 하나로 세어지는지 확인.

판단:

- 다중 core가 bridge로 연결되어 하나로 세어짐: bridge 절단 전략 필요.
- 실제 단일 구조: roll-up 촉진 전략 필요.

---

## A. Rotation/Strain 분해 게이트

### 핵심 가설

v157/v158의 q-ratio + contact gate는 sheet를 압력/접촉 특성으로 추정했다. 그러나 sheet와 vortex core를 구분하는 물리적으로 더 적절한 지표는 속도구배 텐서의 회전/변형률 분해다.

### 구현 아이디어

이미 reconstruction에서 셀별 velocity gradient가 사용 가능하다.

2D velocity gradient:

```text
grad u = [[du/dx, du/dy],
          [dv/dx, dv/dy]]
```

이를 strain tensor와 rotation tensor로 분해한다.

Okubo-Weiss 또는 Q-criterion:

```text
Q = 0.5 * (||Omega||^2 - ||S||^2)
```

정규화:

```text
Q_hat = Q / (||S||^2 + ||Omega||^2 + eps)
```

게이트:

- `Q_hat > 0`: rotation-dominant vortex core. signed-tail correction 유지.
- `Q_hat < 0`: strain/shear-dominant sheet 또는 saddle. correction 감쇠.

이는 ROI-local switching이 아니다. 전 영역에 동일한 물리 feature rule을 적용하는 것이므로 기존 제약과 맞다.

### 기대 효과

- core 사이 saddle 영역에서 correction이 줄어 connecting bridge가 약해진다.
- connected cluster가 분리되어 vortex count 증가 가능.
- sheet에서 anti-diffusive correction이 줄어 checker 감소 가능.
- Mach3 pass 기준 자체가 `Q > 0`, `lambda_ci > 0`를 보기 때문에, gate 물리 지표와 validation 지표가 일치한다.

### 리스크

- roll-up 초기 sheet가 winding되기 전에 너무 감쇠되면 Mach3 hook 형성이 약해질 수 있다.
- 따라서 min factor를 0이 아니라 `0.3 ~ 0.5`에서 시작한다.

### 추천 후보

```text
v159 = v131 + Q_hat rotation/strain gate
```

초기 sweep:

- `f_min = 0.35, 0.50`
- `Q_hat threshold = 0.0`
- shock/compression/pressure-jump face에서는 기존 shock-safe gate 유지

---

## B. Correction Field High-Frequency Filter

### 핵심 가설

checker는 solution field에서 사후적으로 생기는 것이 아니라, signed-tail correction field의 face-to-face grid-frequency 성분이 직접 주입한 것이다.

### 구현 아이디어

signed-tail correction 적용 전 또는 후에 correction field만 필터링한다.

형식:

```text
d_filtered_f = d_f - kappa * (d_f - avg_neighbor_directional(d))
```

여기서 neighbor 평균은 유사 방향 face만 사용한다.

비정렬 삼각 격자에서 유사 방향 face 선택:

```text
abs(n_f dot n_neighbor) > 0.7
```

목표:

- smooth large-scale correction은 유지.
- face-to-face alternating correction만 제거.

### 기대 효과

- checker를 직접 줄인다.
- vortex count에는 중립적일 수 있으므로 A와 조합하는 것이 자연스럽다.

### 리스크

- 너무 강하면 Mach3 clarity 하락.
- `kappa` sweep 필요.

추천:

```text
v160 = v159 + correction high-frequency filter
```

---

## C. Feature-Relaxed MLP Bound 재정식화

### 핵심 가설

현재 구조는 다음 형태다.

```text
MLP-u1 기반 reconstruction
+ additive correction
+ 여러 사후 damping gate
```

이 구조는 correction이 monotonicity/bound 바깥으로 가는 문제를 근본적으로 막지 못한다.

### 구현 아이디어

correction을 더하는 대신 MLP bound 자체를 feature에 비례해 이완한다.

```text
phi_relaxed =
    phi_MLP
    + alpha * f(Q_hat) * (1 - g_shock) * (1 - chi_checker)
```

구성:

- `f(Q_hat)`: A의 rotation gate
- `g_shock`: pressure jump / compression shock gate
- `chi_checker`: local checker indicator

의미:

- sheet/shock/checker 영역에서는 정확히 MLP-u1로 수축.
- vortex core에서만 high-order relaxation 허용.

### 기대 효과

- checker 영역이 MLP-u1로 회귀하므로 baseline보다 checker가 커지는 문제를 구조적으로 차단할 수 있다.
- vortex core는 보존한다.
- 단일 unified reconstruction이며 케이스별 분기가 아니다.

### 리스크

- 구현 변경 폭이 크다.
- 다만 논문화 가능성은 가장 높다.

---

## D. Local Checker Indicator Direct Limiter

### 핵심 가설

checker는 발생 후 자기증폭하므로, 발생 지점에서 즉시 감지하고 correction을 줄이는 feedback이 필요하다.

### 구현 아이디어

cell-local checker indicator:

```text
chi_i = |alternating residual against neighbors|
        / (|grad u|_i * h + eps)
```

조건:

- `chi`가 임계 초과
- pressure jump 낮음, 즉 shock 아님

동작:

- face 양측 reconstruction을 MLP-u1 값으로 blend.

### 기대 효과

- checker 지표 직접 감소.

### 한계

- vortex count를 늘리지는 못한다.
- A 또는 E와 조합해야 한다.

---

## E. Sheet-Breakup Limiter

### 적용 조건

P0-5에서 다음이 확인될 때 사용한다.

- cluster 내부에 실제 다중 vortex core가 있다.
- 하지만 bridge 때문에 connected component가 하나로 묶여 vortex count가 낮게 나온다.

### 구현 아이디어

vorticity distribution의 local shape을 분석한다.

- elongated structure: high aspect ratio, same-sign vorticity bridge
- compact core: low aspect ratio, Q/lambda_ci positive

동작:

- elongated sheet bridge에서 tangential correction 감쇠
- compact core에서는 보존

### 리스크

- stencil이 넓어 비용 증가.
- speckle 생성 방지 필요.
- A의 Okubo-Weiss gate가 더 저렴하므로 A 실패 후 2차 후보로 둔다.

---

## F. Triple Point / Mach Stem Local Protection

### 적용 조건

P0-1에서 checker가 초기 shock 형성기에 생긴다고 확인될 때 사용한다.

### 구현 아이디어

- pressure jump gate 폭을 shock-normal 방향으로 1~2셀 확장.
- shock band 내 signed-tail 완전 차단.
- shock band 후방 1셀까지 ramp-in.
- pressure gradient direction rotation rate를 추가 gate로 사용.

---

## G. 비용/운영 개선

현재 Double Mach quick 한 candidate가 약 50분 걸린다.

따라서 다음 운영 개선이 필요하다.

| 항목 | 내용 | 기대 |
|---|---|---|
| Early proxy | `t_end=0.10~0.12` proxy와 final `t=0.2` metric 상관 확인 | candidate screening 비용 절반 |
| Parameter sweep | A의 `f_min`, B의 `kappa`를 proxy로 sweep | full quick 횟수 감소 |
| Metric snapshot | 시간 분해 checker/vortex/cluster 기록을 probe에 내장 | 별도 진단 run 감소 |
| state 정리 | `state.json` 중복 key 정리 | 재개 안정성 |
| baseline 재검증 | MLP-u1 cache와 current metric code 정합성 확인 | 잘못된 비교 방지 |

---

## 권장 실행 순서

1. 연구 재개 전, 현재 pause 상태 유지 확인.
2. contract test 실행.
3. P0-1 ~ P0-5 진단 추가.
4. v131과 v158을 짧은 proxy로 비교하여 checker 기원과 anti-sheet overlap 확인.
5. `v159 = v131 + Okubo-Weiss / Q_hat rotation-strain gate` 구현.
6. Double Mach proxy 실행.
7. 개선되면 full Double Mach quick `480 x 120`.
8. Double Mach 통과 시 Mach3 quick 재검증.
9. Mach3 통과 시 LeVeque `100 x 100` 재검증.
10. 그래도 실패하면 `v160 = v159 + correction high-frequency filter`.
11. A/B 조합 실패 시 C 구조적 bound relaxation으로 전환.

---

## Codex 실행 메모

현재까지 v132~v158 실패 후보들의 공통점:

- pressure/contact/shear scalar gate 중심
- signed-tail correction의 rotation vs strain 분리 없음
- correction field spectrum 진단 없음

따라서 다음 미탐색 공간은 속도구배 텐서 기반 gate다.

가장 우선 구현할 후보:

```text
tmlpu_v159_v131_qgate_on
```

기본 설계:

- base: `tmlpu_v131_signed_gate_decay_relief_on`
- signed-tail correction에 `Q_hat` gate 추가
- `Q_hat < 0` strain-dominant sheet에서 correction 감쇠
- `Q_hat > 0` vortex core에서 correction 유지
- shock/compression/pressure-jump gate 유지
- scalar branch 변경 금지

첫 파라미터:

```text
qgate_min_factor = 0.40
qgate_transition_lo = -0.05
qgate_transition_hi = 0.05
```

진단 필수:

- qgate active face count
- damped face count
- qgate factor min/p50/p90
- overlap with signed-tail active faces
- optional Double Mach ROI overlap for diagnosis only, not scheme switching

---

## 현재 연구 상태와 연결

현재 pause report:

- `docs/tmlpu_autoresearch_pause_report_2026-06-10.md`

현재 전략 파일:

- `docs/tmlpu_double_mach_breakthrough_strategy_2026-06-10.md`

현재 best candidate:

- `tmlpu_v131_signed_gate_decay_relief_on`

현재 blocker:

- Double Mach reflection quick `480 x 120`

현재 last recorded iteration:

- `85`

현재 recommended next candidate:

- `v159 = v131 + rotation/strain Q-gate`

