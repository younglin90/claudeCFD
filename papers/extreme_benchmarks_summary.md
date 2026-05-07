# Extreme Two-Phase 1D Benchmarks for All-Mach Compressible Solvers

> **검색 결과 (2026-04-17):**
> - Jiang et al. 2025, Phys. Fluids 37, 016112 — DOI 10.1063/5.0244527 (UNDEX)
> - Yu, Song, Choi 2023, Phys. Fluids — DOI 10.1063/5.0165384 (modified Saurel 6-eq)
> - Re & Abgrall 2021, IJNMF — DOI 10.1002/fld.5087 (pressure-based BN)
> - Tymen et al. 2020 — DOI 10.2495/cmem-v8-n4-341-354 (Mach 4.25 shock-droplet)
> - **모두 paywall** — abstract/기존 문헌에서 benchmark 조건 재구성
> **기존 보유**: Denner 2018 §7.4, Pelanti-Shyue 2014, Saurel-Abgrall 1999, Shyue 2006

## 극한 Benchmark Case List

### 1. **Saurel-Abgrall 1999 Epoxy-Spinel Shock Tube** (극한 밀도비 + 고압)
- Left: epoxy (ρ=1840, p=2.5e9), Right: spinel (ρ=3622, p=1e5)
- γ_epoxy=2.43, P∞_epoxy=5.3e9; γ_spinel=2.94, P∞_spinel=12.08e9
- **압력비 25000, 밀도비 1:2**, 초강력 solid-solid shock

### 2. **Shyue 2006 Gas-Water Test 2** (고압 기체/액체)
- Left: air (ρ=1, p=1e9), Right: water (ρ=1000, p=1e5)
- γ_air=1.4, γ_water=7.15, P∞_water=3.31e8
- **압력비 10000, 밀도비 1:1000**

### 3. **Chang-Liou 2007 Underwater Explosion** (극한 수중 충격)
- Left: detonated gas (ρ=1250, p=1e9), Right: water (ρ=1000, p=1e5)
- Air-water detonation byproducts
- **압력비 10000, 극한 에너지 밀도**

### 4. **Low-Mach Pressure Wave in Liquid** (저마하 ~1e-3)
- Left: water (p=1.01e5), Right: water (p=1e5)
- p 변화 1% → u ~ O(c_water × 0.01) = ~15 m/s, Mach = 0.01
- **순수 선형 acoustic 영역 검증**

### 5. **Pelanti-Shyue 2014 Air-Water Dodecane Test**
- 3-phase problem with cavitation
- 5-eq with mass transfer (밀도비 1:10000)

## 현재 솔버로 검증 가능한 것:

| Case | 솔버 대응 | 주의 |
|------|----------|------|
| Saurel-Abgrall 1999 | **가능** — solid-solid but SG EOS | γ, P∞ 차이로 초강한 shock |
| Shyue 2006 gas-water | **가능** — air+water 표준 | 밀도비 1:1000 처리 확인 |
| Chang-Liou UNDEX | 부분 가능 — EOS 재매핑 필요 | 고에너지 detonation 초기조건 |
| Low-Mach Pressure Wave | **가능** — Phase 1 변형 | Mach~0.01 all-Mach 검증 |

## 각 케이스의 물리적 도전

1. **Saurel-Abgrall**: P∞~1e10 vs p~1e9 → SG catastrophic cancellation 위험
2. **Shyue**: air/water ρ=1:1000, MMACM-Ex G corrections 부하
3. **Chang-Liou**: 수중 폭발 → shock+rarefaction+shock
4. **Low-Mach**: acoustic CFL 커야 → IMEX의 진가 발휘 영역
