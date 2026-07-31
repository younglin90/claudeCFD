# Accuracy on Skewed / Non-Orthogonal Unstructured Meshes — research + enhancement

Goal: make the unstructured 3D FVM (vertex/node-ring LSQ-P2 + GAUSS-THINC + BVD + MOOD) accurate
on severely skewed / non-orthogonal grids. Data-driven via an MMS harness; backed by primary literature.

## 1. Mesh-quality definitions (per face)
- **Non-orthogonality** θ = arccos( |d̂·n̂| ), d = x_N − x_P (centre-to-centre), n̂ = face unit normal. OpenFOAM `checkMesh` warns > 70°, invalid ≥ 90°.
- **Skewness vector** m = (x_f − x_P) − [S_f·(x_f − x_P) / (S_f·d)] d = x_f − x_f′, where x_f′ = line(P,N)∩face-plane, S_f = A·n̂. m lies in the face plane (S_f·m = 0). OpenFOAM skewness = |m|/fd (fd≈0.2|d|), warn > 4.

## 2. MMS harness (`tests/test_mms_skew.cpp`, `tools/gen_distorted_mesh.py`)
Manufactured smooth field φ = sin(1.7x+.3)·sin(2.1y+.5)·sin(1.3z+.7); cell value = φ(centroid); measure
L2 errors of (cell gradient, convective o2-quad face value, viscous face gradient) vs analytic, +
non-ortho/skew metrics. Distortion = interior-node random perturbation α·h_local (boundary fixed),
α∈{0,.15,.25,.35}; two resolutions (3.4k / 210k tet, h-ratio 3.78) for the order test. Diskin-Thomas
random-perturbation regime (skewness O(1) under refinement = the stress test).

## 3. Findings (order = log(e_coarse/e_fine)/log(3.78))
| quantity | clean L2 | order | distortion (α=.35, non-ortho 88°) | verdict |
|---|---|---|---|---|
| cell P2-LSQ gradient | 1.34e-2 | **1.96 (2nd)** | 1.33e-2 (≈unchanged) | already robust ✓ |
| convective o2-quad face | 5.1e-4 | **3.05 (3rd)** | 5.3e-4 (≈unchanged) | already robust ✓ (P2 at TRUE x_f = skew-exact) |
| viscous face grad — CURRENT (centroid + over-relaxed along d) | 2.68e-2 | **1.10 (1st!)** | 3.05e-2 (+14%) | **the gap** |
| viscous face grad — **P2@face (new)** | 1.31e-2 | **1.98 (2nd)** | 1.29e-2 (≈unchanged) | **fixed**; 2–7× lower error, distortion-robust |
| convective o2-quad **BJ-limited** (current limiter) | 1.80e-3 | **1.46** | 1.81e-3 | BJ over-clips smooth extrema (general, not skew-specific) |

Why the convective recon is already skew-robust: it evaluates the cell P2 polynomial at the TRUE face
centroid x_f (mesh-driven exact moments) — this IS the Ferziger-Perić skewness correction
φ_f = φ_f′ + ∇φ·(x_f − x_f′). The node-ring inverse-distance² WLSQ-P2 is provably linearly-exact on
arbitrary skewed meshes (Syrakos 2017; Diskin-Thomas: quadratic-LSQ recommended for gradients).

## 4. Enhancement implemented — skew/non-ortho-corrected viscous face gradient (P2@face)
`viscous3d.hpp`: `viscous3d_cell_coeffs_o2` (full 9 P2 coeffs of u,v,w,T) + `viscous3d_add_face_flux_p2face`.
The face gradient is the average of each adjacent cell's P2 gradient EVALUATED AT THE TRUE FACE CENTROID:
∇φ_f = ½(∇φ_o(x_f) + ∇φ_n(x_f)), ∇φ_c(x_f) = grad_c + Hess_c·(x_f − x_c). This does NOT difference
cell-centre values along d, so it has NO skewness error (vs the centroid scheme's O(h) skew error). Boundary
faces: owner P2 gradient at x_f with the wall-normal component from the BC ghost (no-slip/adiabatic kept).
Wired in `solver_euler3d.hpp` both RHS paths; DEFAULT ON for unstructured (opt-out `VISC_CENTROID`).
Solver-verified stable (Daru-Tenaud tet, p_min=4.24 vs centroid 4.25; +18% wall for the 2nd-order gain).

Aligns with NASA best practice (Diskin-Thomas 2012): quadratic-LSQ for inviscid/gradients; for high-
aspect-ratio CURVED grids Green-Gauss viscous is preferred — a future option (`viscous3d_cell_gradients`).

## 4b. Enhancement implemented — U2 smooth-extremum limiter spare (accuracy-preserving)
The BJ limiter on the smooth candidate spuriously clips smooth extrema (MMS: 3rd→1.46 order; ≈constant
vs distortion, so general not skew-specific). NOTE: the smooth ratio limiters (Venkatakrishnan/MLP-u2/
Michalak cubic) are MORE dissipative than BJ for UNSTEADY flow (they approach ψ=1 only gradually past
y_t>1, i.e. limit MORE in y∈[1,y_t]); their benefit is steady-state CONVERGENCE, not unsteady accuracy.
The ONLY unsteady-accuracy fix is to DETECT smooth extrema and not limit. Implemented in
`reconstruct3d_unstr.hpp`: spare the BJ limiter (φ=1) when the per-axis curvature (o2 Hessian diagonal)
is sign-coherent over the node-ring (no Gibbs ⇒ smooth); keep BJ where curvature flips sign (genuine
discontinuity). Scale-free (no (Kh)³ h-coupling). DEFAULT ON, opt-out `RECON_BJ_HARD`. MMS: recovers the
unlimited 3rd-order convex error EXACTLY (5.11e-4, vs BJ 1.80e-3). Shock-safe: Daru-Tenaud strong shock
identical to BJ-hard (p_min 3.85 even with NO shock-sensor — u2 keeps BJ at the shock, so the
unlimited-smooth-candidate Langseth divergence is NOT reintroduced). MOOD(PAD) backstops a-posteriori.
Solver impact on octant/LeVeque is MARGINAL (≤0.02%) because the BVD candidate selection already masks
most BJ clipping there — but it removes the limiter as an accuracy bottleneck for smooth-flow cases and is
the provably-correct choice at no stability cost.

## 5. Diagnosed, not yet changed (future)
- **LSQ via normal equations (ATA_inv)** → κ(AᵀA)=κ(A)². Robust at tested distortion (no degradation);
  add Householder-QR only if a measured ill-conditioning shows up on extreme/high-AR stencils.
- **Over-relaxed non-orthogonal split** (Δ=d|S|²/(d·S)) — only needed if an IMPLICIT diffusion path is added
  (our viscous is explicit; P2@face already gives 2nd order without it).

## 6. Key references
Jasak 1996 (PhD, non-ortho corrections); Ferziger-Perić (skewness corr.); Moukalled-Mangani-Darwish 2016
(S=E+T over-relaxed); Syrakos et al. PoF 2017 (GG 0th / LS 1st order on unstructured, min{p,1} law);
Diskin-Thomas AIAA-J 2010/2011/2012 (gradient accuracy on irregular/high-AR grids, quadratic-LSQ + GG-viscous
recommendations); Mavriplis AIAA 2003-3986 (inverse-distance WLSQ); Barth-Frederickson AIAA-90-0013 +
Ollivier-Gooch (k-exact moments); Michalak-Ollivier-Gooch JCP 2009 (accuracy-preserving limiter); Park-Yoon-Kim
JCP 2010 (MLP/MLP-u); Mirtich 1996 / Chin-Lasserre-Sukumar 2015 (exact polyhedral moments); Roy 2005 (MMS).
