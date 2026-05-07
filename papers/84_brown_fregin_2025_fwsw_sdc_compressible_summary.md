# Brown, Fregin, Bendall, Melvin, Ruprecht, Shipton 2025 — FWSW-SDC for Compressible Euler

**Paper**: arXiv 2505.15985v1

## 핵심 (R113 실패 분석을 위한 reference)

R113 의 fwsw_sdc 가 INNER level 에서 K×M=4 BE 호출 → damping 4× 증폭으로 실패.
본 논문은 **정식** FWSW-SDC 를 compressible Euler 에 적용한 사례:

### 정식 FWSW-SDC 구조 (Ruprecht-Speck 2016 기반)

```
y₀ = y(t_n)
For m = 0,...,M-1 (collocation node):
    Predictor: y_{m+1}^{[0]} = y_m^{[0]} + Δτ_m · [F_fast(y_{m+1}^{[0]}) implicit
                                                 + F_slow(y_m^{[0]}) explicit]
For k = 0,...,K-1 (correction sweep):
    For m = 0,...,M-1:
        y_{m+1}^{[k+1]} = y_m^{[k+1]}
            + Δτ_m · [F_fast(y_{m+1}^{[k+1]}) − F_fast(y_{m+1}^{[k]})] (implicit correction)
            + Δτ_m · [F_slow(y_m^{[k+1]}) − F_slow(y_m^{[k]})] (explicit correction)
            + ∫ residual (collocation integral)
End
```

핵심:
1. Predictor 가 fast/slow 분리 처리 (**not** Strang outer split)
2. Sweep 의 BE 항은 **residual correction** (`F_fast(y^{[k+1]}) − F_fast(y^{[k]})`)
3. residual → 0 으로 수렴 시 BE damping 도 → 0 (cancel)

### R113 가 잘못한 점
- maker 가 K=M=2 단순 구현에서 매 sweep 마다 **full F_fast** 를 BE 로 풀어버림 → BE damping 가산
- Residual 형태 미사용 → cancel 매커니즘 작동 안 함
- 정식 FWSW-SDC 의 핵심 (Picard-style residual sweep) 미반영

## 검증 결과 (논문)

- Compressible Euler with gravity (LFric/Gusto): arbitrary order in time
- Standard NWP idealised tests: gravity wave, baroclinic wave 통과
- Spurious computational mode 없음 (compatible FE space 사용)

## Round 115 와의 관계

본 논문은 **FWSW-SDC 가 옳게 구현되면 작동** 하지만, R113 같은 단순 구현은 함정. Round 115 plan 은 **더 안전한 OUTER Strang Richardson** 를 채택 — 단순하고 검증된 1306.1169 (Einkemmer-Ostermann) 의 composition 이론에 직접 기반.

향후 라운드에서 정식 FWSW-SDC 를 시도한다면:
- predictor 1번 + sweep K=2 → IM1 호출 = 1 + 2K = 5 회 (여전히 비싸지만 정확)
- residual subtraction 정확히 구현 필수
- 그러나 **현재 plan 의 우선순위는 아님** (R113 실패 직후 동일 함정 회피).
