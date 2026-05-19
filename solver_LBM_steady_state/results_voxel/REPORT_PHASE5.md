# SCMK-LBM Phase-5 — Voxel Mesh Benchmark (혈관-style)

목적: 복잡 voxel mask (혈관-like geometry) 에서 SCMK Phase-4 의 적용성 직접 검증.

## 케이스 설계

| Case | Geometry | fluid_frac |
|---|---|---|
| clean periodic | 균일 box | 1.000 |
| random 5/10/20% | 무작위 산재 단일-voxel solid (sparse occlusion) | 0.95/0.90/0.80 |
| cylinder | 단일 원형 obstacle (vessel cross-section 흉내) | 0.916 |
| multi-cylinder | 8개 원형 obstacle (porous-media 흉내) | 0.845 |

모두 N=48 D2Q9 + Guo body force (`F_x = F0 sin(2π y/N)`) + periodic boundary + bounce-back at fluid-solid interfaces.

ν=0.05, F0=2e-4, ω=1.538, tol=1e-7.

## 결과

| Case | fluid_frac | Baseline LBE | Baseline wall (s) | SCMK LBE | SCMK wall (s) | LBE speedup | Wall speedup | Note |
|---|---|---|---|---|---|---|---|---|
| clean periodic | 1.000 | 7,014 | 6.41 | 561 | 0.71 | **12.5×** | **9.1×** | 깨끗함 |
| random 5% | 0.950 | 2,505 | 2.35 | 353 | 0.47 | 7.1× | 5.0× | |
| random 10% | 0.900 | 2,505 | 2.32 | 221 | 0.33 | 11.3× | 7.0× | |
| random 20% | 0.800 | 80,160 | 78.71 | 529 | 0.87 | (151×) | (90×) | baseline max-iter ✗ 미수렴 |
| cylinder | 0.916 | 7,014 | 6.88 | 925 | 1.37 | 7.6× | 5.0× | curved boundary |
| **multi-cylinder** | **0.845** | 2,004 | 1.96 | 1,409 | 2.07 | **1.4×** | **0.9×** | **vessel-like 가장 가까움** |

## 패턴 분석

### Speedup vs geometry 복잡도

```
LBE speedup
    12.5× ┤ clean
    11.3× ┤ random 10%
     7.6× ┤ cylinder
     7.1× ┤ random 5%
     1.4× ┤ multi-cylinder ★ vessel-like
   --------|----+----+----+----+
           1.0  0.95 0.90 0.85
                       fluid_fraction
```

**관찰**:
- Random scatter (구조 없음) : SCMK 5–12× 안정 (PC 의 spectral 가정 거의 무너지지 않음)
- Single cylinder : 7.6× (curved boundary 가 mode coupling 유발하나 manageable)
- **Multi-cylinder (vessel-like)** : 1.4× — *paper claim 30–1000× 와 거리 큼*
- Random 20% : baseline 이 수렴 못해서 artificial 151× — 실제 fair comparison 아님

### Multi-cylinder 가 어려운 이유

`results_voxel/multi-cylinder.png` 그래프: SCMK 와 baseline 곡선이 거의 평행. SCMK 가 약 30% LBE 절약하지만 wall-clock 동일 (overhead).

원인:
- 8개 obstacle 의 wake → flow 가 *structured spatial pattern* (여러 wake interaction)
- Spectral PC 가 *uniform-mean base* + *periodic* 가정 → 다중 wake / multi-scale geometry 표현 불가
- Newton step 마다 macro correction 이 일부만 잘 가, kinetic substep 이 나머지 처리

## 혈관 (vessel) 적용성 정직 평가

| Vessel scenario | 예상 SCMK Phase-4 speedup |
|---|---|
| 단순 직선 vessel (channel-like) | **30–50×** (Phase-4 channel 결과) |
| 단일 bend / curve | **5–10×** (cylinder-like) |
| 복잡 branching / multi-vessel | **1.5–3×** (multi-cylinder-like) |
| Patient-specific Circle of Willis | **2–5× 추정** (보수적) |

**평가 문서 claim 30–1000× → 단순 vessel 한정**. 임상 vessel 응용은 추가 알고리즘 필요.

## 결론

**Phase-4 단일 라인 fix 만으로는 복잡 vessel geometry 충분치 않음**. Multi-cylinder 의 1.4× wall speedup 이 conservative upper bound.

### Phase-4 가 작동하는 영역 (확인됨)
- ✅ Periodic uniform flow (25–501×)
- ✅ Single-wall channel/Couette (33–183×)
- ✅ Random voxel scatter (5–12×)
- ✅ Single curved obstacle (7×)

### Phase-4 가 부족한 영역
- ⚠ Multi-cylinder / porous-like (1.4×)
- ⚠ Cavity 4-wall corner vortex (2–6×)
- ❌ 실제 혈관 voxel mesh — Phase-5+ multigrid V-cycle 필요

### 다음 단계 — Phase-6 권장

평가 문서 §4–5 의 **proper geometric multigrid V-cycle**:

1. Fine level : LBE substep smoother (현재 composite line search 의 K_post 와 동일 메커니즘, 더 다단계)
2. Mid levels : coarsened mask + LBE smoother (geometry 단계적 단순화)
3. Coarsest level : Phase-4 spectral PC on simplified mask
4. **Kinetic-aware transfer** : macro residual restriction, kinetic drop

이 구조에서 multi-cylinder geometry 가 coarsening 단계에서 *통합* 되어 coarse level 에서 single-blob 처럼 보임 → spectral PC valid.

예상 multi-cylinder 가속 with Phase-6 multigrid : **5–20×** (현재 1.4× → 개선).

## 산출물

```
results_voxel/
├── clean_periodic.png             ← 12.5×
├── random_5pct.png                ← 7.1×
├── random_10pct.png               ← 11.3×
├── random_20pct.png               ← 151× (baseline 미수렴 caveat)
├── cylinder.png                   ← 7.6×
├── multi-cylinder.png             ← 1.4× (vessel-like upper bound)
├── *_mask.png                     ← geometry visualization
├── summary.json
└── REPORT_PHASE5.md               ← 본 문서
```

추가:
```
lbm_voxel.py        — voxel mask + bounce-back LBM operator
run_voxel_suite.py  — driver
```
