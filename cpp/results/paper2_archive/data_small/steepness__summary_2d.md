# gauss_paper_v3 — 2D statistics for paper2 Appendix A

Redesigned replacement for `cpp/results/gauss_paper_v2/` (v2 is **not** modified and its numbers
remain valid for what they measured). Single reproducible run, 2026-07-28.

- source: `bench_2d.cpp` (self-contained, kept here for reproducibility)
- build/run: `bash build_run_2d.sh 100 100 100 prod 64 128`
- program output verbatim: `tables_2d.md`
- raw 5 % subsample: `raw_2d_subsample.csv` (100 000 rows, 38 MB)
- run metadata (md5, timing, cores): `run_log_2d.txt`
- figure: `gauss_paper_v3_2d.png` -> https://tmpfiles.org/dl/whw4zgq5mWDf/gauss_paper_v3_2d.png

---

## 1. What changed relative to v2, and why

| v2 defect | v3 fix |
|---|---|
| beta sampled at **3 discrete points** {0.8, 1.6, 3.2} -> error could not be fitted as a function of steepness | beta ~ **U(0.5, 5.0) continuous**, 100 draws, shared across every sample and both shapes |
| 2D and 3D used different curvature scale, different Q sampling, different error metric -> no 2D-vs-3D comparison | curvature `U(0,2)/H`, Q = 50/50 `U(-0.999,0.999)` + `tanh(U(-4,4))`, conservation-error metric — **all three matched to the 3D benchmark** |
| steep tail under-sampled | full factorial 100 cells x 100 P2 x 100 beta = **1e6 samples per shape**; smax reaches 250 (tri) |
| cell-D error measured as \|D - D_ref\| (ill-posed when the sigmoid saturates) | cell-D error measured as the **conservation error** \|<tanh(beta_hat*P + d)>_exact - Qbar\|; \|D - D_ref\| retained only as a diagnostic CSV column |

## 2. Exact configuration (reproducible)

```
factorial     : 100 cells x 100 P2 x 100 beta per shape = 1 000 000 samples/shape
shapes        : triangle, quadrilateral (random convex non-degenerate, star-polar + rejection,
                identical construction to v2 gen_cell(): radius U(0.25,1.0), min angular gap 0.45,
                strictly convex CCW, min edge 0.12, min area 0.06; H = 4A/perimeter)
seeds         : beta 20260728001 | cells 20260728002 | P2+Q 20260728003
                (per-cell and per-(cell,P2) RNGs are seeded by a splitmix64 hash of
                 (seed, shape, ic, ip) so the sample set is INDEPENDENT of thread scheduling)
beta          : 100 draws ~ U(0.5, 5.0), sorted, SHARED by every sample of both shapes
P2 surface    : A0=cos(th), A1=sin(th) (unit normal), A2..A4 = (2U-1)*gam/H, gam ~ U(0, 2.0)
Q (=2*cbar-1) : 50% U(-0.999,0.999) + 50% tanh(U(-4,4)), clipped to |Q|<=0.999
reference     : 1x = 64x64 Duffy-collapsed GL per fan sub-triangle (4096 pts; tri 4096 / quad 8192)
                2x = 128x128                                        (16384 pts; tri 16384 / quad 32768)
                edge 1x = 64-pt GL, 2x = 128-pt GL
production    : 6-pt Dunavant deg-4 per fan sub-triangle (tri 6 / quad 24 pts), 4-pt GL edge
GAUSS         : probit closed form, c = pi/2, analytic cell moments + closed-form edge moments
metric cell-D : |<tanh(kk*P + d_method)>_2x  -  Qbar|          (CONSERVATION error)
metric face   : |th_method - th_2x|, with D HELD FIXED at atanh(Q) so no cell-D error leaks in
wall          : 295 s total / 277 s sweep, 24 physical cores, 249 MB RSS
```

The beta draws are listed in full at the top of `tables_2d.md`.

## 3. Production code called vs code copied

`#include "cfd/reconstruct_bvd.hpp"` — **the production header is used unmodified.**

| quantity | how obtained | production site |
|---|---|---|
| polygon-fan cell quadrature (points + area-normalised weights) | **CALLED** `cfd::c3_cell_quad_P()` | `reconstruct_bvd.hpp:101` |
| GAUSS analytic cell geometric moments | **CALLED** `cfd::c3_build_gmom()` | `reconstruct_bvd.hpp:1139` |
| GAUSS <P>, <P^2> contraction | **CALLED** `cfd::c3_gmom_moments()` | `reconstruct_bvd.hpp:1158` |
| tanh rational Newton cell-D | **copied verbatim** | `reconstruct_bvd.hpp:1533-1543` |
| GAUSS closed-form cell-D | **copied verbatim** (calls the real `c3_gmom_moments`) | `reconstruct_bvd.hpp:1429-1434` |
| production 4-pt GL edge loop + `sig` lambda | **copied verbatim** | `reconstruct_bvd.hpp:1975, 1994-1995` |
| GAUSS closed-form edge moments F1/F2 | **copied verbatim** | `reconstruct_bvd.hpp:1977-1985` |
| TQ (6-pt Dunavant) / EQ (4-pt GL) tables, GC = pi/2 | **copied verbatim** | `reconstruct_cheng3` locals |

Reason for the copies is unchanged from v2: those blocks live *inside* `reconstruct_cheng3()` as
local lambdas / inline branches, there is no callable entry point, and the swept quantities
(`A[]`, `Q`, `beta`, `D`) are internal state derived from the mesh field. The geometry half of the
cell stage is genuine production code called directly.

## 4. Reference-accuracy verification — PASSED, with margin

Two independent checks.

**(a) The reference solve itself is exact.** The 2x-order Newton residual |<tanh(kk P + d_ref2x)> - Q|:

| shape | median | p99 | max | fraction > 1e-12 |
|---|---|---|---|---|
| tri | 1.94e-15 | 1.02e-14 | 2.05e-14 | 0.000 % |
| quad | 2.72e-15 | 1.47e-14 | 3.39e-14 | 0.000 % |

This also settles a design question: g(d) = <tanh(kk P + d)> is strictly monotone with range
(-1,1), so for any |Q| < 1 a root **always exists**. An earlier draft of this benchmark clamped
|d| <= 40 and that clamp manufactured a fake "infeasibility floor" of up to 0.4 in the steep tail.
The final solver brackets on [-(smax+10), smax+10] (rigorous) with safeguarded Newton/bisection.
**There is no ill-posed subset and no well-posedness filter is applied — all 2 000 000 samples are
used.**

**(b) refconv = reference at 1x vs 2x order** (both *evaluated* with the 2x rule, so nothing but
the reference order differs). Requirement: at least two decades below the GAUSS error it measures.

| stage | shape | worst smax bin | refconv med | GAUSS med | ratio | frac(refconv > 1 % of that sample's GAUSS err) |
|---|---|---|---|---|---|---|
| cell-D | tri | [64, inf) | 3.44e-06 | 8.79e-02 | **2.6e4** | 1.66 % |
| cell-D | tri | [32, 64) | 7.50e-08 | 9.32e-02 | 1.2e6 | 0.19 % |
| cell-D | quad | [32, 64) | 7.20e-11 | 1.03e-01 | 1.4e9 | 0.00 % |
| face | tri | [64, inf) | 1.22e-13 | 1.89e-01 | 1.5e12 | 0.13 % |

Every other bin is between 1e8 and 1e14. Globally the refconv median is 3.8e-15 (cell) and
3.3e-16 (face). **The reference is 4 to 14 decades better than the quantity being measured**, so
reference error contributes nothing to the reported statistics.

Reference-order convergence study that led to 64/128 (20x20x20 pilot, tri, worst bin [64,inf)):

| 1x/2x order | refconv med | ratio GAUSS/refconv | pilot wall |
|---|---|---|---|
| 16/32 | 9.6e-03 | 8.7 | 0.37 s |
| 32/64 | 1.0e-04 | 823 | 0.57 s |
| 48/96 | 4.0e-05 | 2 066 | 1.25 s |
| **64/128** | **1.4e-05** | **6 045** | **2.06 s** |

16/32 (the v2 order) was **rejected**: only one decade of margin in the steep tail.

## 5. Error vs steepness — the main relation

s = beta_hat * P ; smax = max_q |kk * P(x_q)| over the cell (edge steepness sfmax analogously).
GAUSS cell-D conservation error, median per bin:

| smax bin | n (tri) | GAUSS tri | GAUSS quad | PROD tri | PROD quad |
|---|---|---|---|---|---|
| [0.25,0.5) | 3 153 | 8.52e-04 | 8.34e-04 | 7.59e-06 | 2.56e-07 |
| [0.5,1) | 26 131 | 1.89e-03 | 1.86e-03 | 3.41e-05 | 2.40e-06 |
| [1,2) | 82 711 | 5.35e-03 | 6.07e-03 | 3.90e-04 | 3.62e-05 |
| [2,4) | 233 801 | 1.30e-02 | 1.38e-02 | 3.21e-03 | 3.64e-04 |
| [4,8) | 323 247 | 2.16e-02 | 2.12e-02 | 1.09e-02 | 1.66e-03 |
| [8,16) | 195 939 | 3.67e-02 | 4.05e-02 | 3.13e-02 | 5.85e-03 |
| [16,32) | 86 462 | 5.78e-02 | 6.22e-02 | 7.60e-02 | 1.48e-02 |
| [32,64) | 33 931 | 9.32e-02 | 1.03e-01 | 1.28e-01 | 1.06e-02 |
| [64,inf) | 14 618 | 8.79e-02 | — | 2.39e-01 | — |

Power-law fits err = 10^a * x^b on the **pre-saturation** bins (smax < 8, where the sigmoid has not
yet saturated over the whole cell) and on all bins:

| stage | shape | variable | a | b | R^2 |
|---|---|---|---|---|---|
| cell-D GAUSS | tri | smax (pre-sat) | -2.506 | **1.211** | 0.989 |
| cell-D GAUSS | quad | smax (pre-sat) | -2.497 | **1.223** | 0.979 |
| cell-D GAUSS | tri | smax (all) | -2.472 | 0.867 | 0.949 |
| cell-D GAUSS | quad | smax (all) | -2.487 | 0.986 | 0.972 |
| cell-D GAUSS | tri | sigma = kk*sqrt(<P^2>-<P>^2) (pre-sat) | -1.976 | **1.095** | 0.966 |
| cell-D GAUSS | quad | sigma (pre-sat) | -1.966 | **1.163** | 0.980 |
| cell-D GAUSS | tri | beta | -2.266 | 1.197 | 0.945 |
| cell-D GAUSS | quad | beta | -2.498 | 1.376 | 0.965 |
| face GAUSS | tri | sfmax (pre-sat) | -2.569 | **1.772** | 0.997 |
| face GAUSS | quad | sfmax (pre-sat) | -2.746 | **1.734** | 0.997 |
| face GAUSS | tri | beta | -2.227 | 1.552 | 0.964 |
| face GAUSS | quad | beta | -2.922 | 1.833 | 0.988 |
| cell-D PROD | tri | smax (all) | -3.828 | 1.898 | 0.943 |
| cell-D PROD | quad | smax (all) | -5.008 | 2.316 | 0.924 |
| face PROD | tri | sfmax (all) | -5.152 | 2.376 | 0.880 |
| face PROD | quad | sfmax (all) | -6.296 | 2.995 | 0.922 |

Readings:

1. **The GAUSS cell-D error is very close to linear in the steepness**, ~ 3.1e-3 * smax^1.21, and
   the exponent AND prefactor are the **same for triangles and quadrilaterals** (1.211 / 1.223,
   -2.506 / -2.497). The whole shape dependence collapses onto the steepness.
2. Written in the probit theory's own variable sigma = kk*sqrt(<P^2>-<P>^2) the fit is
   ~ 1.06e-2 * sigma^1.10 — again shape-independent. This is the direct empirical confirmation of
   the diagnosis that the closed form's error is a function of the sigmoid argument's spread, i.e.
   the tanh-vs-probit kernel substitution, and not of the cell.
3. **The face closed form degrades faster than the cell closed form** (b ~ 1.75 vs 1.21). The edge
   average has no area averaging to smooth the substitution error.
4. Beyond smax ~ 8 the exponent falls (0.87-0.99 over all bins) purely because the error saturates
   at the bounded value ~0.1-0.3 — an O(1) mismatch cannot grow further.
5. The **production tanh** slopes are ~2x steeper (b ~ 1.9-3.0) with a much smaller prefactor: it
   starts far more accurate and loses accuracy faster. That sets up the crossover in section 7.

## 6. Shape effect with steepness controlled

Quad median / tri median **within the same smax bin** (so steepness is held fixed):

| smax bin | cell GAUSS | cell PROD | face GAUSS | face PROD |
|---|---|---|---|---|
| [0.25,0.5) | 0.978 | 0.034 | 0.736 | 0.079 |
| [0.5,1) | 0.984 | 0.070 | 0.697 | 0.147 |
| [1,2) | 1.136 | 0.093 | 0.674 | 0.138 |
| [2,4) | 1.060 | 0.114 | 0.586 | 0.098 |
| [4,8) | 0.984 | 0.151 | 0.643 | 0.083 |
| [8,16) | 1.105 | 0.187 | 0.682 | 0.058 |
| [16,32) | 1.076 | 0.195 | 0.590 | 0.022 |
| [32,64) | 1.101 | 0.083 | — | — |

- **Cell-D GAUSS: ratio 0.98-1.14 across five decades of steepness — no shape effect.** Cell shape
  enters only through how much steepness it produces, exactly as the earlier hex-vs-tet 3D finding
  (ratio 0.69-1.36, random) suggested. This is now established on 1e6 samples per shape.
- **Face GAUSS: ratio 0.59-0.74 — a genuine but modest ~1.5x shape effect** that does NOT vanish
  under steepness control. It is systematic across every bin. Caveat: the face experiment uses edge
  (v0,v1) of each cell, and at matched sfmax a quad's edge sits at a different offset from the cell
  centroid than a triangle's, so part of this is edge/centroid geometry rather than polygon type.
  Reported as-is; it is not large enough to change any conclusion.
- **Cell/face PROD ratios of 0.02-0.20 are NOT a shape effect** — the production rule puts 6 points
  on a triangle and 24 on a quad (4 fan sub-triangles), so quads simply get a 4x finer rule.

## 7. Production-order THINC/QQ is not exact either — where it wins and where it loses

This is the column the brief asked to be preserved. Global aggregate:

| stage | shape | method | median | RMS | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| cell-D | tri | PROD (6-pt Dunavant tanh) | 9.43e-03 | 9.70e-02 | 9.58e-02 | 4.53e-01 | **1.338** |
| cell-D | tri | GAUSS closed form | 1.82e-02 | 7.60e-02 | 1.23e-01 | 2.93e-01 | 4.57e-01 |
| cell-D | quad | PROD (24-pt) | 4.00e-04 | 1.51e-02 | 6.05e-03 | 3.99e-02 | 6.28e-01 |
| cell-D | quad | GAUSS closed form | 1.24e-02 | 4.28e-02 | 6.28e-02 | 1.74e-01 | 3.99e-01 |
| face | tri | PROD (4-pt GL) | 1.08e-03 | 3.81e-02 | 5.45e-02 | 1.62e-01 | 4.74e-01 |
| face | tri | GAUSS closed form | 2.92e-02 | 8.73e-02 | 1.55e-01 | 2.88e-01 | 3.78e-01 |
| face | quad | PROD (4-pt GL) | 1.39e-05 | 1.23e-02 | 7.93e-03 | 6.24e-02 | 2.58e-01 |
| face | quad | GAUSS closed form | 7.09e-03 | 4.08e-02 | 6.14e-02 | 1.67e-01 | 3.36e-01 |

- **On triangles the production cell-D stage crosses over and becomes WORSE than the closed form at
  smax ~ 16** (see section 5: [8,16) PROD 3.13e-2 < GAUSS 3.67e-2; [16,32) PROD 7.60e-2 > GAUSS
  5.78e-2; [64,inf) PROD 2.39e-1 vs GAUSS 8.79e-2). Its worst case, **1.338**, is a total failure of
  the cell-average constraint — the reconstructed cell average is off by more than half the full
  variable range. Two causes stack: the 6-point deg-4 rule cannot integrate a near-step function,
  and the production Newton clamps |D| <= 0.999999, i.e. |d| <= 7.254, while the true root needs
  |d| ~ smax. The GAUSS closed form has no such clamp, which is why it stays bounded.
- On quadrilaterals (24-pt rule) production always wins at every steepness measured.
- At the **face** stage production 4-pt GL beats the closed form everywhere in the median, by 1-3
  decades. **The face stage is where the GAUSS speed-up is paid for.**

## 8. |Qbar| dependence

GAUSS cell-D median by |Q| bin (tri): 6.31e-2 [0,0.3) -> 3.40e-2 [0.3,0.6) -> 1.71e-2 [0.6,0.85)
-> 1.85e-2 [0.85,0.99) -> **2.01e-3 [0.99,1)**. The closed-form error is largest at mid-range volume
fraction and collapses by 30x in the saturated tail. This confirms the design decision to sample Q
with a 50/50 mixture — a purely tanh(U(-4,4)) draw would have concentrated samples in the cheap
tail and under-reported the error by roughly an order of magnitude.

## 9. Where the production solver actually operates (context for the numbers above)

The sweep deliberately goes to beta = 5 and curvature 2/H, far beyond the scheme's defaults.
Anchoring at the real S1 settings (beta_l = 1.4, beta_s = 0.8), from the raw subsample:

| beta | shape | smax p50 | smax p90 | sigma p50 | cell PROD med | cell GAUSS med | face PROD med | face GAUSS med |
|---|---|---|---|---|---|---|---|---|
| 1.4 | tri | 2.56 | 8.62 | 0.71 | 1.72e-03 | 9.22e-03 | 8.28e-05 | 1.06e-02 |
| 1.4 | quad | 1.57 | 3.10 | 0.51 | 3.45e-05 | 5.58e-03 | 6.06e-07 | 2.19e-03 |
| 0.8 | tri | 1.61 | 5.55 | 0.44 | 3.52e-04 | 4.13e-03 | 8.61e-06 | 4.74e-03 |
| 0.8 | quad | 0.95 | 1.82 | 0.31 | 5.40e-06 | 2.37e-03 | 3.45e-08 | 7.32e-04 |

**At the real operating point the GAUSS closed form carries a ~0.5 % (quad) to ~1 % (tri) error in
both stages, while production tanh carries 1e-5 to 2e-3.** The catastrophic PROD numbers of
section 7 live at smax > 16, which beta = 1.4 reaches only in the p99 tail of triangles.

## 10. The |D - D_ref| diagnostic — honest note

The brief's premise was that |D - D_ref| is pathological because the cell average becomes
insensitive to D in the saturated regime. The data **partly** supports this, and the effect is
smaller than expected:

- |D_gauss - D_ref| > 1e-3 in 67.2 % (tri) / 74.0 % (quad) of samples.
- Of those, only **0.33 % (tri) / 0.36 % (quad)** nonetheless have a conservation error below 1e-3,
  i.e. genuinely decoupled.
- The decoupled fraction **falls** with steepness (tri: 1.6 % at smax in [0.25,0.5) -> 0.0 % above
  32), i.e. the decoupling is a *small-steepness* effect, not a saturation effect.

So the D-metric and the conservation metric mostly agree, and the earlier "|D| blows up in the
saturated regime" impression came largely from a |d| <= 40 clamp in the reference solver
(section 4a), not from the physics. The conservation metric is still the right choice — it is what
the scheme is required to satisfy, it is defined identically in 2D and 3D, and it needs no
well-posedness filter — but it should not be sold as a rescue from a pathology that turns out to
affect < 0.4 % of samples. absD_prod / absD_gauss remain in `raw_2d_subsample.csv` for anyone who
wants the old view.

## 11. Limitations

1. smax is a max-norm over the 2x quadrature points; it is a proxy, not the exact sup of |kk*P| over
   the cell. With 16 384-32 768 points the difference is negligible at the reported precision.
2. In the steepest bin (tri, smax > 64) 1.7 % of samples have a refconv above 1 % of their own GAUSS
   error. Those are samples whose GAUSS error happens to be unusually small; bin medians and p90s
   are unaffected. Going to 128/256 would cost ~4x wall for no change to any conclusion.
3. The face experiment uses one edge per cell (v0,v1), not all edges, and holds D at atanh(Q). That
   isolates the face stage cleanly but means the face and cell samples are not independent draws of
   "a face in a real mesh".
4. Random convex cells are far more extreme than real mesh cells (radius ratio up to 4:1). This is
   deliberate — it is a stress test, and section 9 gives the realistic anchor.
5. The aggregation record stores errors as `float`; all arithmetic is `double`. The `float` store
   limits reported values to ~7 significant digits, irrelevant here.
6. The run shared the machine with a concurrent 3D benchmark, so the 295 s wall is an upper bound;
   it is not a performance measurement and no timing conclusions are drawn from it.
7. The table row labelled `floor (unsolvable)` in the "Global aggregate" table of `tables_2d.md` is
   the **2x reference solve residual**; the name is a leftover from the earlier clamped solver and
   was left in place so that `tables_2d.md` matches `bench_2d.cpp` byte-for-byte.

## 12. Summary for the paper

- The GAUSS/QQ closed-form error in 2D is ~ 3.1e-3 * smax^1.21 (cell-D) and ~ 2.7e-3 * smax^1.77
  (face), with **identical exponent and prefactor for triangles and quadrilaterals**.
- In the probit theory's own variable, ~ 1.06e-2 * sigma^1.10, sigma = beta_hat*sqrt(<P^2>-<P>^2).
- **Cell shape has no independent effect** on the closed-form cell-D error once steepness is
  controlled (ratio 0.98-1.14); the face stage shows a modest systematic 1.5x (tri worse).
- At the production settings beta = 1.4 / 0.8 the closed form costs ~0.2-1 % in both stages.
- The production tanh reference is *not* exact: on triangles it fails outright at smax > 16
  (median 0.24, worst 1.34) because of the 6-point rule plus the |D| <= 0.999999 clamp.
- The reference used here is verified to 4-14 decades of margin; no sample had to be discarded.
