# PE Operator Priorities

Use this reference when selecting one research-improvement target.

## Current Technical Position

- Time integration is not the primary issue.
- BE1 is the baseline stability gate.
- ARS222 should be recovered only after the spatial operator is pressure-equilibrium invariant.
- Residual-level PE correction is insufficient when face flux/source assembly does not preserve the PE manifold.

## Primary Invariant

For pressure-equilibrium states, require:

```text
p_U dot L_E(U_PE) = 0
p_U dot (L_E + L_I)(U_PE) = 0
```

where `p_U = dp/dU`.

## Priority Order

1. Face/source-level PE-preserving explicit operator.
2. Path/secant APEC energy flux and pure-branch behavior.
3. Path-conservative alpha source discretization.
4. Pressure/acoustic Schur or Helmholtz block.
5. 07-B acoustic interface decomposition and exact comparison checks.

## Required Regression Gates

- `tests/test_uniform_flow.py`
- `tests/test_amplification_matrix.py`
- `tests/test_transport_eigenmode.py`
- `results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01`

07-B is diagnostic until at least one subcase meets strict criteria without breaking 02-A.
