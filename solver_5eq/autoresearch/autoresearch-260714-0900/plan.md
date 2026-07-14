# Autoresearch plan — AW interface ringing, parameter-free fix

Goal: 07-B Air-Water passes the strict verifier with a parameter-free,
case-uniform scheme change; no regression elsewhere.

## Scope
- Editable: `solver/five_eq_IMEX/` (research options env-gated; final winner
  must be adoptable as the single default for ALL cases).
- Frozen: `solver/He2024/` (read-only reference), `validation/`, verifier
  thresholds (no gaming the metric).
- Hard constraint: zero user-tuned constants. Derived constants (scheme
  algebra, EOS-derived, topology-gated binary switches) allowed.

## Metric (per iteration)
Primary (minimize): `p_smooth_local_tv_excess` on 07-B Air-Water N=400
(target <= 0.30). Hard gates (any broken = FAIL row, discard):
- amp ratios in [0.85, 1.10] (p) / u gate as verifier reports `amp_ok`
- `peak=True` (<=3 cells), `symmetry=True` (<=0.38), L2p<=0.216, Lip<=0.756,
  corr_p>=0.88, finite, complete
- 02_A with candidate active: pass=True, p_rel_linf < 1e-10
- Default path (env unset): bit-identical golden 2.764863893389702e-15

## Verify commands (WSL, from solver_5eq)
- AW: `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=.codex-loop FIVE_EQ_CASE07_ONLY=Air-Water [CANDIDATE_ENV] python3 .codex-loop/verify_02_07_acceptance.py` (~90 s)
- 02_A: `... python3 results/1D/cases/02_A_PE_advection_unified.py` (~4 s)
- Promotion ladder for an AW-passing candidate: full 07 (3 pairs) -> 04/05/13
  quick set -> full 13-case core -> adopt as default.

## Baseline (iteration 0)
tv=0.5370, L2p=0.09004, amp=1.00/0.97, peak ok, sym ok — aw_wiggle FAIL only.

## Iterations
User opted in: run until solved (`Iterations: unlimited`), eval checkpoint
report to user every ~5 iterations. Results log: `results.tsv` in this dir.

## Hypothesis queue (priority order, updated as learnings land)
1. H10 term-level attribution: instrument the acoustic p-row residual terms
   (u0*dp_dx advection vs beta*div_u compression) at cells 125-145 around the
   packet-birth steps (~1050-1150) to identify which term seeds the packet.
2. H11 (informed by H10) targeted interface-face fix in the seeding term.
3. H12 ACID face thermodynamics ported to the imex_ad acoustic solve.
4. H13 unified acoustic face, full Phase A (material fluxes included).
5. H14 stacking partial wins (e.g., interface_be band + spatial fix).

## Candidate ledger carried in (all discarded)
T1 global BE, T2a interface_be, T2b trbdf2, P1 upwind-interface,
P2 implicit_energy, C1 characteristic, C2 BVD-envelope, C2v2 BVD-jump,
C3 muscl3. See docs/research_aw_ringing_campaign.md.
