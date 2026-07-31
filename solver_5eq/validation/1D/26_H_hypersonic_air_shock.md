# Phase 4-A2 — Hypersonic Air Shock Tube

> 전속도 영역 상한(Mach ~10 shock) 검증. 극한 압력비에서의 shock-capturing 안정성.

## 목적
고마하 영역에서 solve_IMEX가 강한 shock을 정확히 포착하며 conservation을 유지하는지 확인.

## 물리 설정

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m |
| N | 200 |
| BC | transmissive |
| EOS | Air: ideal (γ=1.4) |
| 상 | 순수 공기 |
| x < 0.5 (좌) | p = 1e9 Pa, ρ = 10 kg/m³ |
| x ≥ 0.5 (우) | p = 1e5 Pa, ρ = 1 kg/m³ |
| u₀ | 0 |
| CFL | 0.25 |
| t_end | 5e-5 s |

## 이론값 (exact Riemann, pure air)
- p* = 1.89e8 Pa
- u* = 12536.6 m/s
- Shock Mach ≈ 10

## PASS 기준
- t_end 완주
- |u_max − u*| / u* < 5%
- L1 오차 p, u, ρ 각 < 10%
- 모든 cell p, ρ > 0

## 스크립트
`pipeline/option_a_extreme_validation.py::case_A2_hypersonic_air`
