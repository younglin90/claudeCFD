# THINC rho-monotonicity BVD guard — near-miss, trade-off documented (NOT committed)

Date: 2026-07-14. Status: **measured in three forms; the strongest (V-A) passes 19/19 and
fixes the case14 oscillation completely but misses two acceptance bars (case02 corr_rho,
case14 l2_rho) through one and the same mechanism — code reverted byte-identical, doc-only
per the no-regression rule.** Companions: THINC_CONSISTENCY_RESEARCH.md,
denner-pitfalls.md ("case14 THINC band spread").

## Diagnostic that motivated this (Advisor measurement, N=400)

case14 contact band decomposition: THINC OFF — rho slope-reversals 0, TV-excess 0.7% of the
jump (pure monotone smear, benign); THINC ON — reversals 4, TV-excess 44.6% (a genuine
oscillation). The case14 artifact is specifically THINC-INDUCED rho NON-monotonicity
(sharp alpha x smeared T), not extra smearing.

## Guard formulation (parameter-free, case-blind)

At each THINC-active face, the mixture density the candidate face alpha implies at the
upwind cell's (p,T) — `rho_imp = af*rho_a(p_up,T_up) + (1-af)*rho_b(p_up,T_up)`, the same
EOS blend that couples alpha to rho — must lie within [min,max] of the two adjacent CELL
mixture densities; otherwise plain upwind at that face. Bounds are neighbour values: zero
new constants.

## Measured variants (registered config, DENNER_ACID=1; "current" = unguarded THINC ON)

| variant | case14 TV-exc% / band / l2_u / corr_u / l2_rho | case02 corr_rho / front | suite |
|---|---|---|---|
| OFF (no THINC) | 0.99 / 23 / 0.1131* / 0.9665* / 0.039 | 0.9800 / 34 cells | 19/19 |
| current (THINC, no guard) | 44.6 / 42 / 0.1131 / 0.9665 / 0.0312 | 0.9999 / 1 cell exact | 19/19 |
| **V-A reject form** | **0.69 / 23 / 0.1028 / 0.9723 / 0.0382** | **0.9971 / 1 cell, 1-cell offset** | **19/19** |
| V-B clamp form (rho-interval mapped to alpha, intersect + clamp) | 43.9 / 41 (oscillation back) | 0.9999 / exact | — |
| V-C reject with endpoint exemption | 57.5 / 42 (oscillation back) | 0.9999 / exact (0 rejects) | — |

(*case14 metrics under OFF equal current to 3 digits — the unguarded sharpening never helped case14's u/rho metrics.)

V-A side effects, all measured: case13 band 9 (<=10 ok), case30 band 1, case31 band 3
(sharp wins preserved); **case25 interface dramatically CLEANER: ip 0.0121 -> 0.0001,
iu -> 0.034, wave positions 8/1/9 -> 0/0/1 cells** (guard rejects 28 early-transient
activations at the shocked contact); case01 linf 0/0/0; 15/24/33/34 zero activations,
byte-identical. Guard rejects per case (V-A): 02: 82, 13: 1, 14: 27 (of ~70 candidates —
the guard starves the strict indicator and THINC de-facto deactivates on case14), 25: 28,
30: 0, 31: 4. N=800 (temp patch): case14 TV-excess OFF 0.63% vs V-A 0.37% — the
oscillation component vanishes at all N, as predicted.

## The trade-off, precisely (why no variant meets ALL acceptance bars)

case14's genuine violations and case02's spurious rejects live on the SAME instance class:
BVD-clamped ENDPOINT face values, where `blend(alpha_endpoint)` at the upwind (p,T) is
compared against neighbour cells' own EOS densities. At case14's contact the mismatch is
the ~50% smeared-T signal; at case02's uniform-(p,T) contact it is ~1 ulp of EOS roundoff
(82 rejects, 5% of instances -> 1-cell front lag, corr_rho 0.9999 -> 0.9971). Separating
them requires a magnitude threshold = a banned constant. Both constant-free escapes were
measured/analysed and REJECTED: exempting endpoints (V-C) or clamping into the rho-mapped
interval (V-B) removes exactly the case14 signal (oscillation returns); evaluating the
blend at the DOWNWIND state makes the endpoint blend bitwise-equal to the neighbour rho
(analytic), also erasing the signal. The second miss is intrinsic, not numerical: removing
the oscillation removes the (oscillatory) sharpening whose aliasing had flattered l2_rho —
V-A's case14 l2_rho (0.0382) is simply the OFF value (0.039); the 0.031 "current" number
is bought BY the oscillation.

## Verdict and recommendation to the Advisor

V-A is a net scientific improvement on 4 of 5 axes (case14 oscillation gone with BETTER
u-metrics, case25 interface an order cleaner, all other wins intact, 19/19) and its two
acceptance misses are (i) case02 corr_rho 0.9971 vs the 0.999 bar — still far above the
0.98 suite gate and the 0.9800 OFF value, front 1 cell wide at 1-cell offset; (ii) case14
l2_rho equal to the THINC-OFF value rather than the oscillation-flattered 0.031. Per the
standing no-regression rule this stays UNCOMMITTED; if the Advisor judges the case02
0.9971/1-cell trade acceptable against the case14+case25 gains, V-A is complete,
reproducible from this note (reject form, upwind-state blend, neighbour-cell bounds), and
one re-application away. Numbers to re-verify after applying: suite 19/19; case14 TV<=1%,
band 23; case02 corr_rho 0.997, front x=0.798; case25 ip 1e-4.
