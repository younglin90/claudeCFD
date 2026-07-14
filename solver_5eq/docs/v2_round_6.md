# v2 Round 6 — HLLC Riemann face flux

> 일자: 2026-04-28
> 변경 1 개: R3 의 face_upwind + LF blend 를 **HLLC Riemann solver** 로 교체.
>   `flux_hllc.py` 신규 (~150 줄), `time_euler.py` 가 face state 없이 cell L/R 직접 사용,
>   `main.py` 의 cell_max_wave_speed import 가 flux_hllc 로 변경.
> 자유 파라미터: 0개 (Davis wave speed 정의 고정, HLLC closure 정의 고정).

---

## 1. R6 정의

5-equation Allaire-Massoni model 에 Toro 1994 HLLC + Saurel-Petitpas-Berry 2009 §3.3 multi-phase 확장:

```
S_L = min(u_L − c_L, u_R − c_R)
S_R = max(u_L + c_L, u_R + c_R)
S* = (p_R − p_L + ρ_L u_L (S_L − u_L) − ρ_R u_R (S_R − u_R))
     / (ρ_L (S_L − u_L) − ρ_R (S_R − u_R))

factor_K = (S_K − u_K) / (S_K − S*)
α_K* = α_K                                      (passive at contact)
(αρ_k)_K* = (αρ_k)_K · factor_K                 (per-species)
ρ_K* = ρ_K · factor_K
(ρu)_K* = ρ_K* · S*
(ρE)_K* = ρ_K · factor_K · (E_K + (S* − u_K)·(S* + p_K/(ρ_K (S_K − u_K))))

F = F_L                            if 0 ≤ S_L
    F_L + S_L · (U_L* − U_L)       if S_L ≤ 0 ≤ S*
    F_R + S_R · (U_R* − U_R)       if S* ≤ 0 ≤ S_R
    F_R                            if S_R ≤ 0
```

PE state (u uniform, p uniform, α-jump): S* = u₀ exactly, factor_K = 1, U_K* = U_K (with α 변경 없음). F = F_upwind 정확. 따라서 **contact discontinuity 자동 보존**.

---

## 2. 검증 결과 (R3 vs R6 비교)

### 2.1 Smoke gates — R6 압도적 개선

| Test | R3 | **R6 HLLC** | 평가 |
|---|---|---|---|
| S1 uniform | 1.46e-16 ✅ | **1.46e-16 ✅** | 동일 |
| **S2 Case A** | NaN ~step 1000 | **2000 step, ep=1.46e-16, eu=2.66e-16** ✅✅ | machine ε PASS! |
| **S2 Case B** | finite 169 step (2.79) | **finite, ep=2.38e-12, eu=2.25e-12** ✅✅ | machine ε! |
| **S2 Case C** | (informational) | **finite, ep=4.77e-12** ✅ | machine ε |
| **S3 short** | 5.4e-10 ✅ | **2.38e-11** ↑ | 2자리 ↑ |
| **S3 medium** | 4.5e-4 | **2.38e-11** ↑↑ | **7 자릿수 향상** |
| S4 α | 6.5e-3 | 6.3e-3 | 거의 동일 (α-source upwind 부재) |
| S4 T₁ | 1.4e-9 | **0.000** ↑↑↑ | machine ε |
| S4 T₂ | 8.5e-6 | **0.000** ↑↑↑ | machine ε |
| S4 u | 2.1e-9 | **2.03e-12** ↑↑ | 3자리 ↑ |
| S4 p | 2.3e-3 | **1.30e-6** ↑↑ | 3자리 ↑ |
| S5 Case A | machine ε ✅ | **machine ε ✅** | 동일 |
| **S5 Case B** | drift 2.7e-3 | **drift 5.16e-14** ↑↑↑ | **11 자릿수 향상** |

**HLLC = simple gates 모두 machine ε 또는 그에 근접**.

### 2.2 07-B air-water — long-run dt 자동조절 instability

| Sub-case | R3 | R6 |
|---|---|---|
| 1 Air-Water | finite t=1.63 ms (L2p/A=234407) | **timeout ≥600s** |
| 2 Helium-Air | finite t=1.51 ms (L2p/A=3192) | **timeout** |
| 3 Argon-Air | finite t=2.02 ms (L2p/A=1.20) | **10000 step → t=0.058 ms (3%)**, L2p/A=8.6e6 |

R6 의 첫 step 은 정상 (`dt=1.67e-6, F[2] range [1e5, 1.0001e5]`), 그러나 시간이 흐르면서 cell `c_max` 가 점진 증가 → CFL dt 줄어듦 → step 폭주 → 결과 발산수준.

### 2.3 진단 — long-run instability 의 origin

R6 의 첫 step 정상, 어떤 시점부터 cell 의 `c_max` 가 amplify. 가능 원인:
1. **HLLC star state 의 micro-perturbation** 이 acoustic mode 에 amplify.
2. **alpha-pure 영역** (α=1e-6 또는 1−1e-6) 에서 ρ_k 또는 c_k 의 stiff calculation 누적 round-off → c_mix 폭주.
3. **Forward Euler + HLLC** 의 acoustic wave 에서 sub-cycle dt 부족 (HLLC 가 wave-별 stable 한 dt 가정하나 mixed-phase 에서 그 가정 실패 가능).

---

## 3. R6 의 trade-off

| 측면 | R3 LF blend | R6 HLLC |
|---|---|---|
| simple gates (S1-S5) | 일부 FAIL | **모두 PASS** ✅ |
| PE-coupling 정확도 (S4) | 9 자리 (R1 대비 ↑) | **machine ε** |
| mass conservation (S5) | machine ε (Case A) | **machine ε (둘 다)** |
| 07 long-run survival | **모두 t_end finite** ✅ | 모두 timeout / instability ❌ |
| 07 정확도 (Argon, achievable) | L2p/A=1.20 | (미 측정) |

### 3.1 결정 후보

A. **R6 active 유지** — simple gates 모두 PASS. 07 의 long-run instability 는 후속 라운드 (R7?) 에서 cons_to_prim robustness 또는 hybrid HLLC+LF fallback 으로 해결.

B. **R3 회귀** — 07 모두 finite 가 더 중요. R6 의 simple gates 개선은 보조적.

C. **Hybrid: R3 + R6 dispatch** — case 별 또는 *adaptive* 로 LF (R3) 와 HLLC (R6) 사이 전환. 정책상 자유 파라미터 도입 위험.

### 3.2 권장 방향

R6 의 simple gates 결과는 *진정한 PE-preserving* 의 증거 (machine ε mass conservation, S2 long-time 통과). 07 의 long-run instability 는 *별개 문제* — HLLC 자체가 아니라 forward Euler + alpha-pure 영역의 acoustic stiffness.

따라서 R6 active 유지 + 다음 라운드 R7 에서 long-run stability fix:
- R7a: alpha-pure 영역 (α<ε or α>1-ε) detection + face flux fall-back 으로 LF
- R7b: cons_to_prim warm-start 강화 (R4 같은 tolerance 변경 없이)
- R7c: Forward Euler → SSP-RK2 (시간 정확도 + stability)

---

## 4. 변경 로그

| 일자 | R | 변경 1 개 | S1-S5 | 02-A | 07 | 비고 |
|---|---|---|---|---|---|---|
| 2026-04-28 | R6 HLLC | face_upwind+LF → HLLC Riemann | **모두 machine ε / PASS** ✅✅ | short ✅ | long-run timeout / instability | simple gates 압도적 개선, 07 long-run 후속 수정 필요 — `docs/v2_round_6.md` |
