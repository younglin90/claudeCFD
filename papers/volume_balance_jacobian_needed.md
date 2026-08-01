# Volume-balance fully-implicit Jacobian papers — unreachable, wanted for context

Surfaced by the round-6 Planner (`solver_4eq_mass` ACID_YADV Phase-2 Stage 1) as a structural
precedent from a different community: reservoir-simulation "volume-balance fully-implicit"
formulations put the **total mixture compressibility** (mathematically the same object as
`docs/YADV_PHASE2_PLAN.md` §1's `D_p + (rho_a-rho_b)*a_p` identity) on the diagonal of a
fully-implicit Newton Jacobian. Not required to implement Stage 1 (the identity is already proved
algebraically and unit-tested), but would strengthen the "barotropic substitution" consistency
argument in Phase-2 §3 with an independent precedent from outside the CFD/VOF literature.

Both fetch-blocked this round (403 from ScienceDirect; expired TLS on the open NTNU mirror) — not
read, not summarized, not relied on for anything already implemented.

1. **Fernandes, B. R. B.; Marcondes, F.; Sepehrnoori, K.** — "Comparison of two volume balance
   fully implicit approaches in conjunction with unstructured grids for compositional reservoir
   simulation." *Applied Mathematical Modelling* **40**(9-10), 2016, 5153-5170.
   DOI: `10.1016/j.apm.2015.09.002`.
   Why wanted: their Jacobian's pressure coefficient is the total fluid compressibility with the
   remaining entries the phase partial volumes — the volume-balance analogue of Phase-2 §1's
   `D_p + (rho_a-rho_b)*a_p` cross-check identity.

2. **Coats, K. H.** — "Compositional and Black Oil Reservoir Simulation." SPE 50990,
   *SPE Reservoir Evaluation & Engineering* **1**(4), 1998.
   Why wanted: the origin of the volume-balance Jacobian formulation cited above; would give the
   original derivation this identity's reservoir-engineering analogue traces back to.
