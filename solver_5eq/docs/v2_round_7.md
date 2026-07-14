# v2 Round 7 — SSP-RK2 (Heun) 시간 적분 (시도 후 폐기)

> 일자: 2026-04-28
> 변경 1 개: R6 HLLC 위에서 forward Euler → SSP-RK2 (Heun, two stages + ½ averaged on conservative U).
> **결과: 폐기 — R6 forward Euler 로 회귀**.  Long-run instability 가 *더 악화*.
> 자유 파라미터: 0개.

---

## 1. R7 시도의 motivation

R6 HLLC 의 simple gates 는 모두 machine ε 로 PASS, 그러나 07 long-run 에서 dt 자동조절 instability 발현 (Argon-Air 10000 step → t=0.058 ms, L2p/A=8.6e6). 가설: forward Euler + HLLC + acoustic stiff perturbation 의 sub-cycle dt 부족.

가장 단순한 1 변경: SSP-RK2 (Heun, Gottlieb-Shu 2001).
```
U^(1) = U^n + Δt · L(U^n)
U^{n+1} = ½ U^n + ½ (U^(1) + Δt · L(U^(1)))
```
시간 정확도 1차 → 2차, SSP property 에 의해 monotone 보존. 자유 파라미터 0.

---

## 2. R7 측정 결과

### 2.1 Simple smoke gates — R6 와 동일 우수

| Test | R6 | R7 |
|---|---|---|
| S1 | 1.46e-16 ✅ | 1.46e-16 ✅ |
| S2 Case A | machine ε | machine ε |
| S2 Case B | 2.38e-12 | 2.38e-12 |
| S3 short | 2.38e-11 | 3.34e-11 (살짝 worse) |
| S4 T₁/T₂ | 0.000 | 0.000 |
| S4 u | 2.03e-12 | 2.53e-12 |
| S4 p | 1.30e-6 | 2.30e-6 |
| S5 Case A | machine ε | machine ε |
| S5 Case B | 5.16e-14 | 6.90e-14 |

R7 의 simple gates 결과는 R6 와 거의 동일 (round-off 수준). RK2 가 spatial accuracy 의 round-off 를 살짝 amplify 시키나 본질적 변화 없음.

### 2.2 07 long-run — *더 악화*

| sub-case | R6 | R7 |
|---|---|---|
| Argon-Air (10000 step max) | t=0.058 ms, L2p/A=**8.6e6** | t=0.052 ms, L2p/A=**2.7e25** ⬇⬇ |

R7 의 두 stage forward Euler 의 평균 ½(U^n + U^(2)) 가 stage 1 의 instability seed 를 stage 2 에서 amplify → 결과 ~ 1e19 배 악화.

---

## 3. 진단 — long-run instability 의 *진짜* 원인

R7 의 RK2 시간 정확도 향상이 instability 해결 못 한 것은 **spatial discretization 의 다른 mode** 가 발산 source 임을 의미.

가능한 spatial source:
1. **alpha-pure 영역** (α=1e-6, α=1−1e-6) 의 *stiff EOS*: ρ_k 가 cell-by-cell 미세 변동 → c_mix 폭주
2. **HLLC star state 의 micro-perturbation amplification** at α-jump
3. **cons_to_prim Newton 의 wrong-root jump** (acoustic perturbation 이 cons_to_prim 의 conditioning 깨뜨림)
4. **5-eq 모델 자체의 acoustic-contact decoupling** at large impedance ratio (Z_water/Z_air ≈ 3340)

이 중 어떤 것이 dominant 인지 진단 위해서는 더 깊은 분석 필요. 시간 정확도 향상 (RK2) 은 효과 없으니 *공간 이산화 또는 EOS 평가* 측면에서 접근해야 한다.

---

## 4. 결정 — R6 forward Euler 로 회귀

R7 SSP-RK2 폐기. main.py 의 `_step = ssp_rk2_step` → `euler_step`. R6 HLLC active 유지.

R7 은 *시도 결과 폐기* 로 변경 로그에 기록.

---

## 5. v2 진행 종합 (R1 ~ R7)

| 라운드 | 변경 | 결과 | 최종 상태 |
|---|---|---|---|
| R1 | Forward Euler + 1차 upwind | per-step amp 1.16, 07 NaN @3000 | 통합 R6 까지 |
| R2a | p_face + u_face central | PE-coupling 9자리↑, advection ⬇ | R3 base |
| R2.1a | u_face → upwind | 모든 면 worse | **폐기** |
| R3 | + χ(M̂) LF blend | 07 모두 finite, simple gates 일부 FAIL | base for R6 |
| R4 | cons_to_prim tol 1e-12 | mixed, 07-1 회귀 | **폐기** |
| R5 | wave-별 dissipation | 거대 회귀 (Rusanov framework 부정당) | **폐기** |
| **R6** | HLLC Riemann solver | **simple gates 모두 machine ε / PASS**, 07 long-run instability | **active** ✅ |
| R7 | + SSP-RK2 (Heun) | simple gates 동일, 07 더 악화 | **폐기** |

**현재 v2 best (R6 HLLC active)**:
- Simple gates: 모두 machine ε 또는 PASS (R1 의 모든 거대 회귀 해결)
- 07 air-water: long-run instability — Argon 10k step 에 t=58μs (3%) 도달 후 발산수준
- R3 와 비교 trade-off: R6 가 simple gates 우수, R3 가 07 long-run survival.

---

## 6. R8 후보 (미래)

R6 의 long-run instability 해결을 위한 옵션:

| 후보 | 변경 | 정당성 |
|---|---|---|
| **R8a alpha-pure detection** | α<ε 또는 α>1−ε 영역 face flux 를 LF fallback. ε=1e-3 같은 기준은 *수치 정의*. | stiff EOS 영역 의 instability 회피 |
| R8b | mixture frozen sound speed → Wood/Kapila c_mix² | 정확한 mixture acoustic speed |
| R8c | cons_to_prim 강건성 (R4 의 reverse — looser tolerance + outer iteration) | wrong-root 회피 |
| R8d | hybrid R3+R6: face 별 dispatch — alpha-pure 면 LF, 그 외 HLLC | 두 좋은 면의 합 |

각 후보가 *어떤 자유 파라미터 도입을 요구하는지* 신중 검토 필요.

---

## 7. 변경 로그

| 일자 | R | 변경 1 개 | 결과 | 비고 |
|---|---|---|---|---|
| 2026-04-28 | R7 (시도) | Forward Euler → SSP-RK2 (Heun) | simple gates 동일, 07 long-run 더 악화 (L2p/A 1e25) | **폐기** — `docs/v2_round_7.md` |
