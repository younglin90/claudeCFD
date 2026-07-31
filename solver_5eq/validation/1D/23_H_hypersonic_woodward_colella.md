# Phase 6-5: Woodward-Colella 1984 — Two Shock Interaction (Denner 2018 §7.4.2)

> **목적**: 가장 고전적 1D **multi-shock, multi-rarefaction** 상호작용 벤치마크.
> Woodward & Colella 1984 가 제시한 이후 거의 모든 압축성 유동 솔버의 표준 검증.
> Strong shock + strong rarefaction + contact discontinuity 상호작용 → 극단적 정확도 및 안정성 시험.
>
> **출처**:
> - Woodward P. & Colella P. 1984, *J. Comput. Phys.* **54**:115-173
> - Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, §7.4.2 Figs. 20, 21

---

## 물리적 의도

1D 닫힌 도메인 (reflective walls) 내 **3-region 초기조건** → 좌우 2개 강한 충격파 + 중앙 rarefaction → 충격파-충격파 상호작용, 다중 contact, 반사 현상.
단상 유동이지만 **인위적 rarefaction + 충격파 체인 반응** → 수치적으로 매우 까다로움. 저마하와 고마하 혼재.

## 지배방정식

단일상 이상기체 Euler (1D):

## 초기 조건

도메인 [0, 1] m, 3개 영역으로 분할:

| 영역 | 범위 | ρ | u | p |
|------|------|---|---|---|
| Left | 0 ≤ x ≤ 0.1 | 1.0 kg/m³ | 0 m/s | **1000 Pa** |
| Middle | 0.1 < x ≤ 0.9 | 1.0 kg/m³ | 0 m/s | **0.01 Pa** |
| Right | 0.9 < x ≤ 1.0 | 1.0 kg/m³ | 0 m/s | **100 Pa** |

**압력비**:
- p_L/p_M = 10⁵
- p_R/p_M = 10⁴

## EOS

**Ideal gas (Woodward-Colella 단위계, dimensionless)**:
- γ = 1.4
- cv = 720 J/(kg·K)

## 경계조건

- **양쪽 모두 reflective (wall)** — 1D 닫힌 도메인
- 충격파가 벽에 반사되며 중앙으로 이동 → 충격파 간 상호작용

## 이산화

- **도메인**: [0, 1] m
- **격자**: N = 400
  - Woodward-Colella 1984 original: effective N=24768 with AMR
  - 일반적 검증: N=400 이면 충분히 구조 확인
- **CFL**: 0.4 ~ 0.5 (acoustic)
- **t_end**: **0.038 s** (Denner 2018 Fig. 21, 최종 상호작용 시간)
  - 중간 단계 확인: t = 0.016 s (Fig. 20) — 1차 충격파 상호작용 직전

## 이론 해 (수치적 reference)

- **Exact 해 없음** (다중 파 상호작용은 analytic 불가)
- **Reference**: Woodward-Colella 1984 adaptive mesh refinement solution (effective N=24768)
- 또는 고해상도 WENO5 / DG 결과와 정량 비교
- 23_ref.png 그래프에서의 reference result 결과 참고.

## 주요 관찰 대상

1. **Left strong shock** (p_L=1000 Pa → p_M=0.01 Pa): Mach ~ 수십
2. **Right strong shock** (p_R=100 Pa → p_M=0.01 Pa): 
3. **중앙 rarefaction fans**: 양쪽에서 생성
4. **충격파 교차** @ t ≈ 0.028 s: 새로운 state 형성
5. **Contact discontinuity**: ρ jump 존재

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| Density peak 위치 | Woodward-Colella 참조 ± 3 cells | shock position |
| Density peak 값 | Ref. 값 ± 5% | shock strength |
| Positive pressure | p > 0 everywhere | positivity |
| Positive density | ρ > 0 everywhere | positivity |
| Total mass conservation | ΔM/M < 1e-10 | closed domain |
| Entropy condition | global s 증가 | 2nd law |

## 구체 reference 값 (t = 0.038 s)

Woodward-Colella 1984 (N=24768):
- 좌측 shock: x ≈ 0.50, ρ_peak ≈ 5.9
- 우측 shock: x ≈ 0.76, ρ_peak ≈ 10.6
- 중앙 plateau: ρ ≈ 3.0

(세부 값은 Denner 2018 Figs. 20, 21 참조)

## 예상 결과 특이사항

- **고차 TVD 또는 WENO 필수**: 1차 upwind 로는 너무 diffusive
- THINC-BVD 는 α 가 없으므로 이 case에 불필요
- **Reflective BC 정확 구현 필수**: `u = 0 on wall`

## 참고 문헌

- Woodward P., Colella P. 1984, *J. Comput. Phys.* **54**:115-173 (원본)
- Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, Figs. 20, 21
- Toro E.F. 2009 *Riemann Solvers and Numerical Methods* Ch. 4 (참조 해법)
- Titarev V.A., Toro E.F. 2004, *J. Comput. Phys.* **201**:238 (WENO benchmark)
