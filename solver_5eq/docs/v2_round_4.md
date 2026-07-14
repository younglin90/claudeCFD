# v2 Round 4 — cons_to_prim tolerance 강화 (ACID 의 부분적 시도)

> 일자: 2026-04-28
> 변경 1 개: `state.py::cons_to_prim` wrapper 의 tolerance 1e-9 → 1e-12, max_iter 30 → 60.
> **결과: 폐기 (R3 로 회귀)**.  Air-Water 회귀가 PE-coupling 향상보다 무거움.
> 자유 파라미터: 0개 (수치 정확도 정의).

---

## 1. R4 시도의 motivation

R3 의 가장 큰 잔존 문제는 `S2 Case A` 의 long-time NaN at step ~1000. 진단 결과 cons_to_prim Newton 의 *residual tolerance* 가 1e-9 (absolute on a small residual vector) 라 매 step δp ≈ 1e-14 perturbation 이 random-walk 으로 누적되어 ~1000 step 후 *wrong-root* basin 으로 점프.

가장 단순한 1 변경: tolerance 1e-9 → 1e-12, max_iter 30 → 60. 자유 파라미터 0 (수치 의미만 변경).

---

## 2. 검증 결과 (R3 vs R4)

| Test | R3 | R4 | 평가 |
|---|---|---|---|
| S1 | 1.46e-16 ✅ | 1.46e-16 ✅ | 동일 |
| **S2 Case A** | **NaN @ ~1000** | **NaN @ 1683** | 약간 개선 |
| S2 Case B/C | finite (R1 폭발 회복) | finite | 동일 |
| S3 short ep | 5.4e-10 | 1.0e-9 | 살짝 worse |
| S3 medium ep | 4.5e-4 | **0.10** | 악화 |
| S4 α | 6.5e-3 | 6.5e-3 | 동일 |
| S4 T₁ | 1.4e-9 | **1.2e-11** ↑↑ | 2 decades better |
| S4 T₂ | 8.5e-6 | **8.0e-8** ↑↑ | 2 decades better |
| S4 u | 2.1e-9 | **2.5e-11** ↑↑ | 2 decades better |
| S4 p | 2.3e-3 | **2.7e-5** ↑↑ | 2 decades better |
| **S5 Case A** | **machine ε (2.4e-15)** | machine ε (7.1e-16) | 동일 |
| **S5 Case B (phase 2)** | 2.7e-3 | **3.11** ⬇ | 1100× **악화** |
| **07-1 Air-Water** | **finite t=1.63 ms** | **NaN @ step 432 (0.72 ms)** ⬇ | **회귀** |
| 07-2 Helium-Air | finite t=1.51 (3192) | finite t=1.51 (3192) | 동일 |
| 07-3 Argon-Air | finite t=2.02 (1.20) | finite t=2.02 (1.20) | 동일 |

---

## 3. 진단 — 왜 R4 가 mixed 인가

### 3.1 PE-coupling 정확도 향상 (S4)

R4 의 더 strict 한 cons_to_prim Newton 이 매 step W 의 round-off level 을 ~1e-14 → ~1e-16 로 줄여, S4 의 *PE-coupling* (p, u, T 의 frame-shift 일치) 정확도를 2 자릿수 향상.

### 3.2 Stiff Newton 의 long-time 발산 (Air-Water 회귀)

Air-Water 는 Z 비 3340 — most stiff. cons_to_prim Newton 이 (α=1e-6 → α=1−1e-6 transition 영역) 매 step 더 strict tolerance 충족하려 함. R4 의 tolerance 1e-12 는 종종 max_iter 60 안에도 도달 못 함 → fall-back path → wrong root → NaN.

### 3.3 S5 Case B 의 1100× 악화

R3 (drift 2.7e-3) → R4 (drift 3.11). α-jump + u=1 advection 에서 cons_to_prim 의 strict mode 가 *다른* root 로 진입 → 갑작스런 mass loss.

### 3.4 결론

**R4 의 tolerance 강화는 "고정밀 PE-coupling" 과 "stiff Newton 안정성" 의 trade-off 를 만든다.** 자유 파라미터 0 의 정책 하에서는 한 쪽을 선택해야. R3 의 1e-9 가 Air-Water 같은 stiff problem 에서 더 *robust*, R4 의 1e-12 가 PE-coupling 에서 더 *accurate*.

**더 근본적 해결**: cons_to_prim 의 *random-walk 누적 자체* 를 제거해야 함.
- Face-level PE projection (cell W 가 PE state 일 때 cons_to_prim 우회).
- W-based time integration (cons_to_prim 호출 자체 제거).
- ACID 의 *cell-centered version*: PE-tangent projection 을 매 step 적용 (v1 IMEX 가 시도했으나 정책 L5 위반으로 거부됨).

이런 *구조적* 해결이 R5+ 단계의 후보.

---

## 4. 결정 — R3 로 회귀

R4 의 mixed trade-off 를 정직하게 인정하고 **R3 default** 로 회귀:
- `state.py::cons_to_prim` 의 tolerance 1e-9, max_iter 30 (R3 default).
- R4 는 *시도 결과 폐기* 로 변경 로그에 기록.
- 다음 라운드 R5 에서 *구조적* PE-preservation 접근 시도.

---

## 5. R5 후보 — 구조적 변경

| 후보 | 변경 | 정당성 | 예상 효과 |
|---|---|---|---|
| **R5 진짜 SLAU2 mass flux (Recommended)** | LF blend (R3) 를 SLAU2 mass-flux (Shima-Kitamura 2011 §III) 로 교체. mass flux dissipation 이 *(ρu)_R − (ρu)_L* 기반 → contact discontinuity 자동 보존. p_face 는 central 유지. | Air-Water L2p/A 234k → 100? 까지 줄 가능성. contact discontinuity *spurious dissipation* 제거. | Air-Water 정확도 큰 향상. R3 의 모든 finite 보존. |
| R5' phase-by-phase energy flux (APEC pre-step) | F[3] = α·ρ₁·h₁·u + (1−α)·ρ₂·h₂·u + ½ρu³ + p·u 로 명시 분해. Saurel 2009 §3.2 eq 16-17. | 수학적으로 등가지만 round-off 행보 다름. interface 의 internal energy 와 kinetic energy 분리 진행. | S2 long-time 회귀 일부 해결 가능. |
| R5'' HLLC | Riemann star state 기반 face flux. PE 자동 보존 + Riemann-consistent. | 가장 큰 변경 (~150 줄), 가장 정확. | 모든 sub-case 정확도 큰 향상. |

**추천 R5 = 진짜 SLAU2 mass flux** — Air-Water 정확도가 R3 결과의 핵심 약점이고, SLAU2 가 contact 보존 으로 직접 해결.

---

## 6. 변경 로그 (plan §13 항목 갱신)

| 일자 | R | 변경 1 개 | 결과 | 비고 |
|---|---|---|---|---|
| 2026-04-28 | R4 (시도) | cons_to_prim tol 1e-9→1e-12, max_iter 30→60 | S4 2 decades ↑, S5 B 1100× ↓, 07-1 회귀 NaN @432 | **폐기** — `docs/v2_round_4.md` |
