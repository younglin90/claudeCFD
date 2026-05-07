# v2 Round 5 — Wave-decomposed dissipation (시도 후 폐기)

> 일자: 2026-04-28
> 변경 1 개: `flux_basic.py` 의 LF dissipation 을 wave-별로 분리.
>   mass (F[0,1,4]) → `|u|` 만 (contact wave speed)
>   momentum/energy (F[2,3]) → `c+|u|` (acoustic wave speed)
> **결과: 폐기 — R3 의 단일 |λ| 로 회귀**.  거의 모든 게이트에서 R3 보다 worse.
> 자유 파라미터: 0개.

---

## 1. R5 시도의 motivation

R3 의 Air-Water L2p/A=234407 정확도 부족의 직접 원인은 *contact discontinuity 에서 LF dissipation 의 mass 항이 ρ jump (~996) × c (~1340) 에 비례*. R5 의 의도는 mass component 의 dissipation 을 **contact wave speed `|u|` 로 줄여** PE state (u=0) 에서 mass dissipation 0 → contact 자동 보존.

이는 HLLC / SLAU2 의 contact-preservation essence 의 가장 단순한 표현.

## 2. R5 측정 결과 — 거대 회귀

| Test | R3 | R5 | 평가 |
|---|---|---|---|
| S1 | 1.46e-16 ✅ | 1.46e-16 ✅ | 동일 |
| S2 Case B | finite (169 step) | **NaN @ 220** | 후퇴 (다른 분기) |
| S3 short ep | 5.4e-10 ✅ | **6.72** ⬇⬇ | **10 자릿수 악화** |
| S3 medium ep | 4.5e-4 | **13.4** | 폭발 |
| S4 | 207 step finite | **NaN @ 174** | 발산 |
| S5 Case A | machine ε | **NaN @ 48** | **거대 회귀** |

S5 Case A 에서 machine ε mass conservation (R3) → step 48 NaN (R5).

## 3. 근본 원인 분석

R5 의 wave-decomposed dissipation 은 **HLLC / SLAU2 의 핵심 idea** 지만, 그것을 **Rusanov-form (LF blend) framework 안에 단순 차감으로** 도입하면 *forward Euler + central velocity advection* 의 von-Neumann instability 가 발현.

수학적 분석:
- mass advection equation: ∂t (αρ) + ∂x (αρ u) = 0
- discrete: αρ_i^{n+1} = αρ_i^n − Δt/Δx · (F_{i+½} − F_{i−½})
- F_{i+½} = ½(αρ u)_L + ½(αρ u)_R − ½ ν · ((αρ)_R − (αρ)_L)
- ν = numerical viscosity. 여기서 R3 ν = (1−χ)·(c+|u|) ~ c (low Mach), R5 ν = (1−χ)·|u| ≈ 0 (low Mach + |u| small).
- forward Euler + central u + ν → 0 → von-Neumann growth factor |g|² ≥ 1 (unstable).

**HLLC 와 차이점**:
- HLLC 는 face flux 자체를 wave 별 contributions 의 sum 으로 정의:
  F_HLLC = F_L + S_L · (U*_L − U_L) + ... (contact wave 의 dissipation 이 *star state* 에 implicit 으로 들어감)
- Rusanov R5 는 *F_central + LF blend* 형식 — wave-decomposed 가 정당하려면 face flux 의 *characteristic decomposition* 이 필요.

**결론**: wave-별 dissipation 은 진짜 Riemann solver (HLLC) 안에서만 정당. Rusanov 형식의 simple subtraction 으로는 다른 instability 도입.

## 4. 결정 — R3 로 회귀

`flux_basic.py` 의 dissipation 을 다시 단일 `|λ| = c + |u|` 로 통일 (R3 default).
R5 는 *시도 결과 폐기* 로 변경 로그에 기록.

검증 후 R3 결과가 정상 복귀:
- S5 Case A: machine ε (drift 2.4e-15)
- 07 Air-Water: finite t=1.63 ms (L2p/A=234407)
- 07 Helium-Air: finite t=1.51 ms (L2p/A=3192)
- 07 Argon-Air: finite t=2.02 ms (L2p/A=1.20)

## 5. R6 후보

R3 의 정확도 부족을 *진짜* 해결하려면 더 큰 변경이 필요.

| 후보 | 변경 | 정당성 | 변경량 |
|---|---|---|---|
| **R6 HLLC Riemann solver (Recommended)** | face flux 자체를 HLLC 로 교체. star state (p*, u*) PE-preserving + contact wave separation 자동. | Toro 1994. wave-별 dissipation 이 characteristic 영역에 정당 살아 있음. | ~150 줄 |
| R6' SLAU2 정확 형식 (Shima-Kitamura 2011) | mass flux 의 SLAU2 정확 형식 (eq 17-22). pressure flux 의 g-function. | 진짜 SLAU2 paper 형식. all-Mach + contact-preserving. | ~80 줄 |
| R6'' R3 + RK2 (SSP) | R3 그대로 + 시간 정확도 SSP-RK2 (Heun). | 시간 정확도 향상이지만 정확도 핵심 약점이 spatial dissipation 이라 효과 작음. | ~30 줄 |

추천: **R6 HLLC** 또는 **R6' SLAU2 정확** (paper 정확 형식).

## 6. 변경 로그

| 일자 | R | 변경 1 개 | 결과 | 비고 |
|---|---|---|---|---|
| 2026-04-28 | R5 (시도) | LF dissipation 을 wave-별 분리 (mass→|u|, others→c+|u|) | 거의 모든 게이트 거대 회귀. S5 A machine ε → NaN. S3 short 10 자릿수 악화. | **폐기** — Rusanov framework 내 wave-decomp 부정당. HLLC 필요. `docs/v2_round_5.md` |
