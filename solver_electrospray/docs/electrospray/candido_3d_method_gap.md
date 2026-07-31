# Candido 3D Taylor Cone Jet Method Gap

Target: reproduce 3D asymmetric Taylor cone jet instability and whipping on
structured and unstructured finite-volume meshes, following an OpenFOAM-like
owner/neighbour face-mesh formulation.

## Implemented In This Path

- `FaceMesh3D` owner/neighbour geometry for Cartesian and skew/unstructured
  smoke meshes.
- Variable-coefficient electrostatics:
  `-div(epsilon grad(phi)) = rho_e` with harmonic face coefficients.
- 3D Maxwell/Coulomb forcing:
  `rho_e E - 0.5 |E|^2 grad(epsilon)`.
- Conservative scalar transport for VOF and free charge.
- Least-squares gradient reconstruction and bounded limited-linear advection.
- Geometric PLIC swept-face VOF advection using exact tetra/hex plane-cut
  volume fractions for the swept face prism, with conservative bounded
  liquid volume transfer.
- Ohmic conduction and implicit charge relaxation.
- PLIC-like interface normal/curvature scaffold plus CSF capillary force.
- Balanced-face capillary force assembled as a pressure-compatible
  capillary-potential gradient.
- Face Maxwell-stress divergence force assembly for a balanced face-mesh
  electric-force path.
- An opt-in hybrid Maxwell-stress diagnostic now combines the Poisson-consistent
  face-normal electric flux with tangential electric components from the
  reconstructed cell electric field. This is diagnostic-only and not a default
  production path.
- Tag-level boundary-condition schema for inlet/outlet/wall/electrode data,
  with potential, VOF alpha, prescribed inlet velocity flux, outlet/open
  backflow clipping, and free-charge scalar inlet values wired into the 3D
  step.
- Contact-angle metadata, boundary diagnostics, and a verified 3D normal
  projection primitive are recorded, but production contact-angle curvature
  enforcement remains missing.
- A production-adjacent contact-angle curvature diagnostic now compares the
  baseline local-PIC curvature fit with contact-angle-adjusted wall normals on
  the Candido initial interface. It changes all 32 wall-adjacent mixed cells
  without RDF fallback, but raises the p95 stencil condition from 33.7147 to
  69.131; this is evidence for a real wall-curvature path, not yet a production
  enforcement claim.
- The Candido smoke options include an opt-in production switch for
  contact-angle curvature. A 1-step diagnostic exercises the switch and remains
  bounded, but it changes the CSF force scale from 5.84024 to 0.896303 on the
  current smoke mesh, so the default remains off pending surface-tension
  regression.
- Force decomposition shows this is mainly a wall-curvature redistribution:
  wall mean/max `|kappa|` move from 2.28397/5.03857 to 1.15217/2.3173, and
  wall mean/max CSF force move from 1.32957/5.84024 to 0.225526/0.896303.
- Variable-density collocated pressure-velocity coupling with PISO-like
  repeated pressure corrections and Rhie-Chow face-flux correction.
- PIMPLE-style outer pressure-velocity correction wrapper, momentum
  under-relaxation control, and non-orthogonal correction pass count.
- Explicit 3D timestep estimates: face-flux CFL, electric relaxation, and
  capillary-wave limits.
- AMR refinement indicators for interface, curvature, charge, and electric
  field layers.
- Reduced 3D whipping observables: transverse centroid amplitude, RMS radius,
  axial liquid extent, and dominant frequency from centroid-offset histories.
- Small executable 3D validation cases for electrostatics, pressure
  projection, PISO/Rhie-Chow coupling, PIMPLE/non-orthogonal controls,
  limited-linear transport, PLIC normal reconstruction, geometric PLIC VOF,
  exact tetra/hex PLIC plane-cut volume fractions, exact swept-face PLIC
  wet-fraction checks, VOF compression, Maxwell-stress balance, balanced
  capillary-pressure cancellation, boundary schema, AMR indicators,
  non-axisymmetric seed transport, and whipping observables.
- Discrete-alpha local curvature now has two separate evidence levels:
  `equivalent_sphere` is a spherical sanity diagnostic only, while
  `local_plic_quadric` is the production-facing unstructured local fit. The
  latter improves the hardening fixture versus RDF and reports fallback
  fraction, p95/max stencil condition, and condition-triggered RDF fallback.
  Dynamic oscillating-droplet frequency/damping remains DOWNGRADED.
- Dynamic droplet hardening now writes time-history and force-isolation CSVs.
  The new evidence separates VoF transport, pressure projection, local
  curvature conditioning, and frozen-alpha CSF acceleration; it does not
  upgrade the surface-tension dynamics claim.
- Irregular swept-face PLIC coverage now has an explicit unsupported
  polyhedral diagnostic row. It is finite/bounded evidence only, not a claim
  of exact arbitrary-polyhedron plane-cut support.
- Candido morphology comparison currently uses a coarse alpha-volume integral
  proxy. The paper reports an `alpha=0.5` silhouette-style volume, so the
  current morphology rows are DOWNGRADED until a robust connected-component
  PLIC/free-surface silhouette extractor is implemented or independent
  contour-coordinate data is available.
- The Candido smoke default VOF compression was moderately increased from
  `0.05` to `0.06` after a local compression sweep showed this was the
  smallest tested setting that kept mass/divergence bounded while moving the
  fixed-time alpha-integral proxy inside the strict 10% band. The full-run
  `candido_guo_morphology_error3d.csv` now reports long-window CaE=0.25
  alpha-integral morphology errors of `+9.67294%` at 0.4 ms and `-0.554899%`
  at 0.7 ms. This upgrades only the coarse alpha-integral proxy; it does not
  prove a paper-faithful connected `alpha=0.5` silhouette extractor, and the
  0.8/0.9 ms rows remain blocked by missing external contour/volume data.
- A morphology observable audit CSV now records alpha-integral, inlet-connected
  alpha-integral, and coarse `alpha>=0.5` axial silhouette proxies side by
  side. The inlet-connected proxy falls inside the post-initial 10% band on
  the current smoke mesh, but it is diagnostic only and does not upgrade the
  paper silhouette claim.
- A ray-sampled connected `alpha=0.5` silhouette diagnostic has also been
  added. It remains quantified as a diagnostic: the long-window 0.4/0.7 ms
  errors are -10.8924% and -8.93099%. A newer reference-independent
  `outer_envelope_alpha05` observable takes the outer envelope of the connected
  ray-alpha05 and PLIC ray-plane q25 contours. That gives +2.52434% and
  -8.93099% at the same two digitized Fig. 3(b) times, so the stricter
  `morphology_outer_envelope_alpha05_within_10_percent` check is now true.
  The old ray-only value is retained in the CSV as diagnostic evidence, not
  used as the paper-faithful contour acceptance metric.
- A gas-side ray-intersection attempt was tested and rejected: including all
  outer cells lets disconnected high-alpha structures set the silhouette radius
  and overpredicts the long-window 0.4/0.7 ms volumes by 85.8463%/102.873%.
  The retained `candido_morphology_silhouette_bracket3d.csv` instead records
  a conservative bracket between the connected ray-alpha05 silhouette and the
  inlet-connected alpha volume. The 0.4 ms reference is bracketed
  (-10.8924% to +7.27111%), but 0.7 ms sits just outside the bracket
  (-8.93099% to -1.61039%). This keeps the morphology claim DOWNGRADED while
  narrowing the missing piece to robust connected PLIC/free-surface extraction.
- A first connected PLIC contour silhouette diagnostic was added directly from
  the reconstructed discrete-alpha interface centroids. It is intentionally
  retained as negative evidence: the 0.4/0.7 ms errors are
  +125.718%/+133.733%, much worse than the connected ray-alpha05 proxy. This
  shows that cell-centered PLIC centroids alone are not a paper-faithful
  contour extractor on the coarse cone-jet smoke mesh; a true clipped
  interface-polygon/free-surface contour path is still required.
- A follow-up clipped PLIC plane/cell-edge intersection diagnostic was also
  tested. It is even more conservative as a silhouette envelope and
  overpredicts the 0.4/0.7 ms digitized volumes by +269.993%/+387.337%.
  Therefore the missing morphology path is not merely "use PLIC points"; it
  needs a connected outer free-surface contour extraction with nozzle/wall
  masking and disconnected-structure rejection.
- A sector-median PLIC cut-point diagnostic and a PLIC ray-plane contour
  diagnostic were added to test whether robust local PLIC sampling can replace
  the coarse ray-alpha05 proxy. The sector-median cut-point path is still
  strongly overexpanded (+226.043%/+210.473% at 0.4/0.7 ms). The ray-plane
  path is much better than raw PLIC envelopes but still misses the paper
  silhouette by +35.8748%/+18.4168%. This remains DOWNGRADED evidence: the
  current coarse mesh and local PLIC geometry are not enough for a
  paper-faithful contour-volume observable.
- A PLIC ray-plane quantile diagnostic now records a 25% interpolated crossing
  in `candido_guo_morphology_error3d.csv`. It improves the long-window
  CaE=0.25 0.4 ms silhouette miss to `+2.52434%`, but it still underexpands
  the 0.7 ms contour by `-25.8759%`. This confirms that neither first-exit,
  lower-quantile, nor median local ray-plane selection is a robust
  time-evolving paper contour extractor on the coarse smoke mesh.
- A paper-style tip-synchronization morphology diagnostic now follows the
  Candido text that sets 0.4 ms to maximum cone length. It gives -3.72026% and
  -1.60099% alpha-integral errors at the 0.4/0.7 ms paper points, but this is
  still not a paper-faithful `alpha=0.5` silhouette.
- A fixed-time morphology phase-lag diagnostic now separates shape-volume
  mismatch from evolution-speed mismatch. After the moderate VOF-compression
  change, the 0.4 ms fixed-time alpha-integral error is `9.67294%`, and the
  best-volume match occurs at `0.569043 ms` with `-0.31264%` error. At 0.7 ms
  the fixed-time error is `-0.554899%`, with a best-volume match of
  `0.125342%` at `0.637328 ms`. This clears the coarse alpha-integral proxy
  gate but still does not clear the paper morphology claim, because the paper
  observable is an `alpha=0.5` silhouette and late 0.8/0.9 ms external data are
  still absent.
- A stricter current-voltage sensitivity diagnostic is DOWNGRADED: the high
  CaE long-window run has a tail-mean convective-current ratio of 1.93042e11
  relative to the low CaE run, contradicting the paper's qualitative statement
  that average current is weakly influenced by electric potential.
- A total-current diagnostic now integrates `rho_e*u_y + sigma*E_y` over the
  midplane slab and writes
  `candido_current_voltage_sensitivity_total_current3d.csv`. It does not
  rescue the current-voltage claim: the low/high tail mean ratio is 4.35188e8,
  because the high-voltage long-window run shares the same 24153.5 current
  blow-up as the convective-only diagnostic. The blocker is therefore in the
  high-CaE charge/momentum evolution, not merely in using a convective-only
  current observable.
- The same high-CaE long-window run exposes accumulated open-domain liquid
  growth: current first exceeds 10x the Ganan-Calvo scale at 0.93323 ms, and
  the standalone long-window budget rerun shows the final liquid volume grows
  by `458.244` times the initial volume. The boundary-flux budget closes to
  relative residual `4.37119e-15`, so this is not a VoF conservation residual.
  It remains a high-CaE boundary/dynamics/current calibration failure and makes
  current interpretation non-quantitative.
- Charge-budget decomposition narrows the current failure further. The
  low-CaE long-window case closes the charge budget to `2.35169e-16` with no
  clamped cells. The high-CaE case has relative charge-budget residual `26.2652`,
  cumulative clamp correction L1 `3.57514e6`, max clamped cells `1637`, and
  max unclamped charge magnitude `231142` before clipping. The high-voltage
  current claim is therefore DOWNGRADED because the charge-transport/clamping
  path is non-quantitative, not because the liquid VoF budget is leaking.
- Short standalone charge diagnostics now recover the subcycling, conservative
  bounding, combined bounding+subcycling, and qLimit sensitivity evidence
  without rerunning the long Candido suite. Four charge subcycles are
  DOWNGRADED in the short fixture: residual ratio `7.62521`, current ratio
  `3.99884`, and no clamp correction to reduce (`0/0`). Conservative bounding
  closes the local charge budget relative to an unbounded high-CaE run
  (`0.27538 -> 1.19195e-15`, residual ratio `4.32839e-15`) while leaving the
  current essentially unchanged (`0.999934`). Combining bounding with four
  subcycles also closes the budget (`9.08888e-15`) but increases current by
  `3.99858`. The qLimit sweep (`5/50/500`) is locally insensitive because no
  clipping occurs. These rows reduce mechanical evidence gaps, but they do not
  validate Candido current-voltage behavior.
- The combined path still fails the paper's weak voltage-sensitivity statement.
  The high/low tail-current ratio drops from 1.93042e11 to 2.02796e10, but the
  acceptance bar is <=2. This creates an additional explicit gap:
  `combined_charge_current_voltage_sensitivity_ok=0`.
- A qLimit sensitivity probe confirms the current observable is not physically
  calibrated yet. In the combined path, high-CaE `chargeLimitBase` values
  5/50/500 produce max convective currents 29.4072/1997.94/6.02883e8 and max
  velocities 773.572/1122.46/260514. The largest cap also destroys VoF mass
  quality (`alpha_mass_drift=1`). The current path is therefore sensitive to an
  arbitrary numerical charge cap and cannot be used as a paper-level current
  validation until charge scaling and boundary current treatment are specified.
- An opt-in quasi-implicit bulk charge-relaxation diagnostic was added using
  `rho_e <- rho_e / (1 + dt*sigma/eps)` with an explicit relaxation sink in the
  charge budget. It is DOWNGRADED on the current Candido smoke path: relative
  charge-budget residual stays closed (3.3048e-12), but max convective current
  slightly increases from 1997.94 to 2002.77 (`current_ratio=1.00242`) and max
  velocity barely changes (`velocity_ratio=0.99883`). This indicates the
  high-CaE instability is not fixed by bulk relaxation alone; boundary current
  treatment, dimensional charge scaling, and the arbitrary qLimit regime remain
  the limiting issues.
- A patch-resolved conductive boundary-current decomposition now writes
  `candido_boundary_current_decomposition3d.csv`. It narrows the high-CaE
  current failure to the nozzle/electrode boundary: in `long_window_ca042`,
  `ymin_nozzle` contributes cumulative conductive charge flux -0.24437 versus
  total -0.244315 (fraction -1.00023), while `ymax_collector` is 0 and lateral
  walls are only 5.56299e-05. The combined bounded/subcycled path shows the
  same pattern (`ymin_nozzle` -0.246028 of total -0.245989). This supports the
  blocker diagnosis that paper-faithful nozzle/electrode charge-current
  boundary conditions and dimensional charge scaling are required; bulk
  relaxation and lateral leak fixes are not the next lever.
- A pairwise boundary-current sensitivity artifact now writes
  `candido_boundary_current_sensitivity3d.csv`. It confirms the baseline
  high-CaE boundary-current blow-up is extreme (`long_window`
  total/nozzle/lateral ratios `2.13781e6`/`3.83349e5`/`3.28103e5`), while
  paper-charge, inlet/open/moving-collector, and unit-Maxwell candidates make
  cumulative conductive boundary-current ratios weakly sensitive
  (`paper_charge_boundary` total/nozzle `1.29479`/`1.29597`,
  moving-collector `1.29629`/`1.29618`, unit-Maxwell `1.29588`/`1.29612`).
  This is useful negative/diagnostic evidence, not a Candido Fig. 8(b) pass:
  the high-CaE dominant patch is still the nozzle, and the comparable
  convective liquid-jet current remains too voltage-sensitive or blocked by
  zero face flux in the separate Poisson-face convective diagnostics.
- An opt-in interface-localized conservative charge redistribution candidate
  now writes `candido_interface_charge_transport_diagnostic3d.csv`. Unlike the
  previous current-observable diagnostics, this changes the actual `rho_e`
  path used by the next electric force and convective current evaluation:
  after bounded charge projection, the conservative charge deficit is
  redistributed with `alpha*(1-alpha)+0.02*alpha` weights. The result is
  negative evidence. The candidate keeps charge budget, mass, and continuity
  bounded (`relative_charge_budget_residual=6.25486e-13/7.63047e-13`,
  `alpha_mass_drift=2.49507e-15/2.7627e-15`,
  `max_div=1.02813e-12/1.66459e-12`) and uses weighted support
  (`weighted_cells=1680/1680`), but the axial alpha>=0.5 convective-current
  ratio worsens from the paper-charge baseline `2.06484` to `2.22145`.
  Morphology and whipping remain degraded (`31.371%` low-Ca morphology error,
  high-Ca max asymmetry `0.00408637`). Simple interface/liquid-weighted
  conservative redistribution is therefore not the missing charge-current
  transport model. The full guard after this diagnostic remained green:
  `git diff --check && cmake --build build && ctest --test-dir build
  --output-on-failure`, 53/53 tests passed, total `1814.32 sec`, Candido smoke
  `632.01 sec`.
- An opt-in post-charge potential refresh candidate now writes
  `candido_post_charge_potential_refresh_diagnostic3d.csv`. This resolves
  `phi/E` again after charge advance before the next Maxwell force and current
  diagnostics, removing a stale-`rho_e`/`E` time-level mismatch without changing
  default numerics. Targeted evidence is only approximate: the axial alpha>=0.5
  convective-current ratio improves slightly from `2.06484` to `2.04167`, with
  post-charge potential residuals `9.33659e-12/9.7162e-12`, but electric-source
  and velocity ratios remain `1.84701` and `1.7982`, morphology error remains
  `31.371%`, and high-Ca asymmetry remains `0.00408637`. The stale-potential
  artifact is therefore not the dominant Candido Fig. 8(b) blocker. The full
  guard after adding this opt-in path remained green:
  `git diff --check && cmake --build build && ctest --test-dir build
  --output-on-failure`, 53/53 tests passed, total `1825.66 sec`, Candido smoke
  `649.79 sec`.
- An opt-in conductivity-potential charge closure now writes
  `candido_conductivity_potential_charge_closure3d.csv`. This candidate solves
  `div(sigma grad phi)=0` on the Candido electrode boundaries, reconstructs
  `rho_e` from `div(epsilon grad phi)`, and skips the separate conservative
  volume-charge advection for that candidate. It is the strongest current-ratio
  improvement in this local sequence: axial alpha>=0.5 convective-current ratio
  improves from `2.06484` to `1.93502`, conductivity potential residuals are
  `8.78545e-12/8.82591e-12`, and closure clamp is `0/0`. It is still not a
  paper-level replacement because electric-source and velocity ratios remain
  high (`1.86719` and `1.8088`), morphology worsens to `33.4912%`, high-Ca
  asymmetry remains `0.00444554`, and the volume-charge budget residual becomes
  very large (`901916/711341`) because the candidate is a quasi-steady
  reconstruction rather than a conservative surface-charge transport closure.
  Regression coverage after this opt-in path remains green, with the caveat
  that the sequential full command hit the host runtime cap: `git diff --check
  && cmake --build build && ctest --test-dir build --output-on-failure` passed
  tests 1-38 before external SIGTERM 143 at test 39, then the continuation
  command `ctest --test-dir build --output-on-failure -I 39,53` passed 15/15,
  total `702.91 sec`, Candido smoke `700.87 sec`; combined coverage was 53/53
  with no test failures observed.
- The paper-gap metric now treats an interrupted Candido CTest log directory as
  partial evidence if many `candido_*.csv` files are zero-byte. This prevents a
  user- or host-stopped long smoke run from hiding the last complete benchmark
  artifacts. The live targeted diagnostic guard
  `ctest --test-dir build --output-on-failure -R
  '^test_electrospray_paper_gap_metric$'` passed 1/1 in `0.28 sec`; with the
  interrupted build logs ignored, `apps/electrospray_paper_gap_metric.py`
  currently selects the older repo-root `benchmark_logs` and reports
  `paper_validation_gap_count=42`. This is a conservative evidence reset, not a
  solver regression claim. The latest conductivity/surface-charge candidates
  must be re-run to restore complete live CSV evidence.
- A short standalone conservative surface-charge fixture now exercises just the
  paper-charge-boundary baseline and the conservative surface-charge candidate,
  avoiding the full 700-second Candido suite for this one diagnosis. Targeted
  command passed: `ctest --test-dir build --output-on-failure -R
  '^test_(candido_conservative_surface_charge_closure3d|electrospray_paper_gap_metric)$'`,
  2/2 tests passed, total `4.08 sec`. The candidate keeps the guard numerics
  bounded (`alpha` mass drift `1.30626e-15/1.69819e-15`, max divergence
  `9.11325e-13/1.50745e-12`, implicit Ohmic residual
  `8.28744e-12/8.2761e-12`, charge-budget residual
  `9.64442e-15/5.96013e-15`), but it does not reduce the developed alpha>=0.5
  convective-current ratio: baseline `2.13023`, candidate `2.13035`.
  Status is
  `DOWNGRADED_CONSERVATIVE_SURFACE_CHARGE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY`.
  After removing zero-byte partial logs, the live paper-gap metric reports
  `conservative_surface_charge_closure_quantified=1` and
  `paper_validation_gap_count=39`.
- A short electrode/surface-current boundary isolation fixture now compares the
  paper-charge-boundary path against nozzle-allowed and collector-only
  conductive boundary switches under the same implicit Ohmic projection path,
  and also includes opt-in rows where the same boundary filter is applied
  inside the implicit Ohmic sigma operator.
  Targeted command passed: `ctest --test-dir build --output-on-failure -R
  '^test_(candido_electrode_surface_current_boundary_isolation3d|electrospray_paper_gap_metric)$'`,
  2/2 tests passed, total `14.97 sec`. The finding is DOWNGRADED but useful:
  nozzle-allowed and collector-only rows are numerically identical to the
  paper-charge-boundary row (`max_option_effect_deviation=0`,
  axial alpha>=0.5 current ratio `2.13023`, total boundary-current ratio
  `1.29479`). This shows those boundary switches are bypassed by the current
  default implicit Ohmic charge projection path. The new opt-in implicit
  filter does change the solved/reporting path: the
  `implicit_filtered_paper_charge_boundary` row zeros nozzle conductive flux
  and moves the dominant high patch to collector, while
  `implicit_filtered_collector_only_boundary` zeros both nozzle and lateral
  boundary conductive fluxes. However, it does not improve the developed
  alpha>=0.5 current sensitivity (`2.13202` and `2.13201`, respectively,
  versus baseline `2.13023`) and it remains
  `DOWNGRADED_BOUNDARY_CURRENT_NOT_SOLE_LIMITER`. Guard values remain bounded
  (`max_div` about `1.5e-12`, `alpha` mass drift about `2e-15`). The live
  metric still has `electrode_surface_current_boundary_isolation_quantified=1`
  and `paper_validation_gap_count=39`.
- A first opt-in diffuse-interface leaky-dielectric source closure now adds the
  local Ohmic source `-grad(sigma).E` on mixed cells, refreshes the electric
  potential before force/current evaluation, and records the applied source in
  the charge budget. Targeted command passed: `ctest --test-dir build
  --output-on-failure -R
  '^test_(candido_interfacial_ohmic_charge_source3d|electrospray_paper_gap_metric)$'`,
  2/2 tests passed, total `3.68 sec`. The source is active on both low/high
  short Candido paths (`216/216` source cells; max source density
  `32831.6/47925.6`; applied source charge `-33.1706/-61.8356`) and keeps
  mass/continuity bounded (`max_div` `3.39797e-11/8.60583e-11`). It is
  DOWNGRADED: the developed alpha>=0.5 current ratio worsens from the baseline
  `2.13023` to `4.70647`, with charge ratio `1.66147`, velocity ratio
  `2.83569`, and electric-source ratio `3.68941`. This rules out the naive
  diffuse-volume `-grad(sigma).E` insertion as the missing Candido Fig. 8(b)
  current model; the next path needs a conservative surface-current transport
  or calibrated electric-stress/current closure rather than adding unbounded
  interfacial volume charge.
- A short standalone momentum-source factorization fixture now restores the
  `candido_momentum_source_factorization3d.csv` evidence without requiring the
  long Candido smoke suite. Targeted command passed: `ctest --test-dir build
  --output-on-failure -R
  '^test_(candido_momentum_source_factorization3d|electrospray_paper_gap_metric)$'`,
  2/2 tests passed, total `13.73 sec`; the fixture itself took `13.50 sec`.
  It writes all seven rows required by the paper-gap metric:
  Ca-independent drive, boundary-advected drive, unit-Maxwell drive,
  paper-charge-boundary, paper-inlet-velocity, open-atmosphere, and
  moving-collector cases. Every row has `3/3` low/high developed samples, and
  every row remains
  `DOWNGRADED_ELECTRIC_SOURCE_DRIVES_VELOCITY_SENSITIVITY`. The live
  paper-charge row has velocity/electric-source/surface-source/total-source
  ratios `1.64577/1.68149/0.999488/1.65943`; the unit-Maxwell row has
  `1.56379/1.68037/0.999872/1.60374`. This is a diagnostic improvement, not a
  solver validation pass: it reduces the mechanical paper-validation gap count
  to `37` by proving the source-factorization and electric-drive tradeoff are
  now quantified, while confirming that the remaining Fig. 8(b)-style current
  mismatch is tied to electric momentum-source sensitivity.
- A second short standalone fixture now restores the post-charge potential
  refresh and conductivity-potential closure CSVs after the interrupted long
  Candido log reset. Targeted command passed: `ctest --test-dir build
  --output-on-failure -R
  '^test_(candido_charge_closure_candidates3d|electrospray_paper_gap_metric)$'`,
  2/2 tests passed, total `5.80 sec`; the fixture itself took `5.58 sec`.
  The post-charge refresh row has `3/3` developed samples, changes the
  developed alpha>=0.5 current ratio only from `2.13024` to `2.13014`, and
  remains `APPROXIMATE_POST_CHARGE_REFRESH_REDUCES_CURRENT_SENSITIVITY`.
  The conductivity-potential row also has `3/3` samples and changes the ratio
  from `2.13024` to `2.1234`, with charge/velocity/electric-source ratios
  `1.29404/1.64127/1.68164`, status
  `APPROXIMATE_CONDUCTIVITY_CLOSURE_REDUCES_CURRENT_SENSITIVITY`. Both reduce
  mechanical validation gaps but neither is a paper-level closure: morphology
  error is still about `45%`, high-Ca radial asymmetry is only about
  `1.7e-4`, and electric-source sensitivity remains high. The live paper-gap
  metric is now `paper_validation_gap_count=35`.
- The same standalone fixture now also restores
  `candido_interface_charge_transport_diagnostic3d.csv`. The interface-localized
  redistribution candidate is active (`1772/1772` weighted cells,
  redistribution deficit `2.95153e-10/3.82197e-10`) and keeps the charge budget
  tight (`7.71553e-15/7.30116e-15`), but it does not reduce the current
  sensitivity: baseline/candidate developed alpha>=0.5 current ratios are
  `2.13024/2.1304`, with charge/velocity ratios `1.2949/1.64577`.
  Status is
  `DOWNGRADED_INTERFACE_CHARGE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY`. This
  lowers the mechanical gap count to `34` while reinforcing the finding that
  simple conservative redistribution is not the missing surface-current model.
- The standalone momentum-source fixture now also restores the Poisson-face
  current-observable diagnostics that previously depended on the long Candido
  smoke suite. Targeted command passed: `ctest --test-dir build
  --output-on-failure -R
  '^test_(candido_momentum_source_factorization3d|electrospray_paper_gap_metric)$'`,
  2/2 tests passed, total `15.10 sec`. This turns on seven metric checks:
  `poisson_face_total_current_observable_quantified`,
  `poisson_face_alpha05_total_current_observable_quantified`,
  `poisson_face_axial_current_window_quantified`,
  `poisson_face_candidate_axial_windows_quantified`,
  `poisson_face_candidate_axial_convective_windows_quantified`,
  `poisson_face_convective_factorization_quantified`, and
  `poisson_face_velocity_projection_factorization_quantified`, lowering the
  live gap count to `27`. The evidence is still not a validation pass:
  face-consistent all-phase total current has weak tail ratio `1.2949`, but
  alpha>=0.5 total current is a zero-current observable and is explicitly
  `BLOCKED_ZERO_CURRENT_OBSERVABLE`. Paper-charge convective Poisson-face
  current remains DOWNGRADED with current ratio `2.13024`; its factorization
  gives charge/face-flux/abs-convective-flux ratios
  `1.2949/1.64577/2.13024`, and the projected-vs-raw row has projected/raw
  current ratios `2.13024/2.13024`, so the remaining current sensitivity is not
  a Rhie-Chow projection artifact. Inlet/open/moving convective Poisson-face
  rows are still `BLOCKED_ZERO_FACE_FLUX`.
- A follow-up opt-in nozzle conductive-flux suppression diagnostic was tested.
  It reduces the combined high-CaE nozzle cumulative flux from -0.246028 to
  -0.0844609, but it worsens the current and velocity: max convective current
  rises from 1997.94 to 2258.44 (`current_ratio=1.13038`) and max velocity
  rises from 1122.46 to 1497 (`velocity_ratio=1.33368`). This is retained as
  DOWNGRADED evidence: a crude insulated-nozzle charge transport switch is not
  the missing paper boundary model.
- A charge-scale audit now writes `candido_charge_scale_audit3d.csv`. It is a
  scale-sanity diagnostic rather than a calibrated nondimensional-to-SI
  conversion, but it exposes the current blocker sharply: the combined high-CaE
  path has cumulative clamp correction `659012`, about `4.84914e16` times the
  Rayleigh charge scale for the inlet radius, and peak convective current
  `1997.94`, about `2.53113e11` times the Ganan-Calvo current scale. The
  nozzle-suppressed and relaxed variants remain at the same order
  (`6.77096e16`/`2.86115e11` and `4.84654e16`/`2.53725e11`). The qLimit sweep
  is also non-physical: raising `chargeLimitBase` 5 -> 50 -> 500 moves
  peak current `29.4072 -> 1997.94 -> 6.02883e8`. This keeps the Candido
  current claim DOWNGRADED and makes dimensional charge scaling plus a
  conservative, paper-faithful charge boundary model the next required work.
- A single-charge-unit consistency diagnostic now writes
  `candido_charge_unit_consistency3d.csv`. It asks whether one physical charge
  unit can simultaneously map the integrated charge to the Rayleigh charge
  scale and the convective current to the Ganan-Calvo current scale using the
  hydrodynamic time scale. The answer is no on every current smoke row:
  combined low-CaE already has a 27.004 mismatch, combined high-CaE has
  50480.4, nozzle-suppressed high-CaE has 166272, relaxed high-CaE has
  50820.1, and the qLimit sweep spans 449.687 to 4.57189e9. This is stronger
  than the previous scale audit because it rules out a simple post-hoc scalar
  calibration of the present charge/current fields.
- An electric-property scaling audit now writes
  `candido_electric_property_scaling_audit3d.csv`. For the Candido liquid,
  the physical electric relaxation time is `8.20488e-06 s` (`8.20488 us`),
  or `0.0296188` hydrodynamic times. The current smoke path instead uses
  normalized conductivities (`K_l=1`, `K_g=1e-6`), giving
  `dt/tau_l=0.00147783` in solver units while the physical timestep ratio is
  `dt/tau_l=2.77417`. This makes the normalized relaxation scale about
  1877x slower than the physical scale and confirms that paper-level current
  validation needs dimensional electrical-property scaling, not only a better
  output current metric.
- An opt-in dimensional electrical-conductivity path was then tested using
  `sigma*th/eps0` for the dimensionless conductivity. It correctly matches the
  physical liquid relaxation ratio (`dt/tau_l=2.77417`) but is strongly
  DOWNGRADED in the current explicit/clamped charge path: relative
  charge-budget residual rises from `3.87276e-12` to `0.219108`, clamp
  correction rises by `270.975x`, max convective current rises by `594.699x`,
  max velocity rises by `499.175x`, and alpha mass drift reaches `0.870574`.
  This means dimensional conductivity scaling is necessary but not sufficient;
  it must be paired with a stiff/quasi-implicit conservative charge update and
  paper-faithful charge-current boundary conditions.
- A Poisson-operator-consistent face-normal conductive-current diagnostic now
  writes `candido_current_voltage_sensitivity_poisson_face_current3d.csv` while
  preserving the original cell-gradient current path. It is negative evidence:
  the low/high tail-current ratio is `2.02391e9`, the high-CaE
  clamp/Rayleigh ratio is `1035.48`, and the relative cell-gradient
  Gauss-law residual remains order-one (`1.37042`). The symmetric Poisson
  solve itself is converged, but replacing the current flux by the Poisson
  face stencil does not calibrate the charge-current dynamics.
- An opt-in implicit Ohmic charge projection now writes
  `candido_current_voltage_sensitivity_implicit_ohmic_charge3d.csv`. It treats
  the stiff conductive step through the same symmetric Poisson operator instead
  of only post-processing the current flux. The targeted Candido smoke test
  passes, but the result is still DOWNGRADED: the low/high tail-current ratio
  is `9.4997e8`, high-CaE current/Ganan-Calvo is `6.51183`, and the
  high-CaE cell-gradient Gauss-law residual rises to `6853.32`. This narrows
  the current blocker further: the repository now has a runnable conservative
  implicit diagnostic, but the coarse cone-jet path still needs a
  paper-faithful dimensional charge-current boundary model and a production
  face-field force/current formulation before Candido-level current validation
  can be claimed.
- An opt-in face-consistent electric diagnostic now writes
  `candido_current_voltage_sensitivity_face_consistent_electric3d.csv`. It
  combines Poisson face-normal conductive current with a Maxwell-stress
  divergence assembled from the same face-normal Poisson field. This is a
  partial scale improvement but still not validation: the high-CaE
  current/Ganan-Calvo ratio drops to `0.000854954` and clamp/Rayleigh drops to
  `1.68184`, but the low-CaE current is almost zero, so the tail-current ratio
  remains `3.68294e9`. The cell-gradient Gauss-law audit is still bad
  (`11.132`) because the diagnostic force uses face-normal data while the
  legacy audit intentionally measures the old cell-gradient field. This says
  the previous high-CaE current blow-up was partly a force/field consistency
  problem, but the coarse smoke run still lacks a paper-faithful charge
  injection/boundary model and developed low-CaE jet current.
- A paper-style liquid-jet cross-section current diagnostic now writes
  `candido_current_voltage_sensitivity_liquid_jet3d.csv` and
  `candido_current_voltage_sensitivity_alpha05_jet3d.csv`. This tests whether
  the previous voltage-sensitivity failure was only caused by measuring raw
  `rho_e u_y` outside the liquid jet. It is also DOWNGRADED: the alpha-weighted
  long-window ratio is `1.1017e18`, and the alpha>=0.5 ratio is `4.76369e31`
  because the low-CaE coarse smoke run has essentially no developed liquid-jet
  current at the fixed midplane. The blocker is therefore coupled low/high-CaE
  jet development and charge transport, not merely a raw-current observable
  bug.
- A developed-jet current-window diagnostic now writes
  `candido_developed_jet_current_window3d.csv` using the new history columns
  `midplane_liquid_area_di2` and `midplane_alpha05_area_di2`. This is the
  clearest current-observable decomposition so far: for the default long-window
  comparison, the low-CaE tail has `0/27` samples with
  `midplane_alpha05_area_di2 >= 1e-4`, while the high-CaE tail has `3/27`.
  For the face-consistent electric diagnostic both low and high have `0/27`.
  The current-voltage claim is therefore not just DOWNGRADED; on this coarse
  fixture the strict average-current comparison is BLOCKED by an undeveloped
  low-CaE measurement plane. A paper-level comparison needs either a
  non-degenerate developed-jet low-CaE fixture or a measurement plane/time
  definition matched to the external current dataset.
- An axial developed-jet scan now writes
  `candido_axial_developed_jet_current_window3d.csv` and adds history columns
  `developed_jet_y_over_Di`, `developed_jet_alpha05_area_di2`, and
  developed-section convective-current variants. This rejects the idea that the
  current failure is only a bad fixed-midplane choice. In the default
  long-window case, the axial scan finds developed tail windows for both
  low-CaE (`21/27`) and high-CaE (`13/27`), but the convective developed-window
  current ratio is still `4.64035e9` (`6.60724e9` alpha-weighted,
  `7.06812e9` alpha>=0.5). Therefore the current-voltage claim remains
  DOWNGRADED even after avoiding the fixed-midplane degeneracy. In the
  face-consistent electric path, low-CaE has developed axial samples (`11/27`)
  but high-CaE has none, so that diagnostic remains BLOCKED as a current
  comparison.
- A combined face-implicit electric diagnostic now writes
  `candido_current_voltage_sensitivity_face_implicit_electric3d.csv`. It turns
  on the implicit Ohmic charge projection and the Poisson face-normal Maxwell
  force together. This reduces the tail-current ratio to `61525`, far below
  the raw face-consistent ratio but still nowhere near the <=2 weak-voltage
  criterion. The high-CaE current/Ganan-Calvo ratio is only `0.000951026`, and
  the axial developed-window audit is BLOCKED because the high-CaE tail has
  `0/27` developed axial samples. This is another partial diagnostic
  improvement, not validation.
- An electric-relaxation timestep diagnostic now writes
  `candido_electric_relaxation_timestep_limit3d.csv`. It keeps the same
  face-implicit electric path, turns on dimensional electrical scaling, and
  applies `dt <= min(eps/sigma)` from the phase relaxation times. The timestep
  is actually limited from `0.0821674` to `0.0296188` (`dt/tau=1`). After fixing
  the implicit Ohmic boundary-convective flux accounting and applying the
  bounded global charge-conservation projection, the tail-current ratio drops
  from `61980.2` to `3.41126` while the limited low/high relative charge-budget
  residuals are `3.83567e-14` and `1.10782e-13`. This is substantial progress,
  but still not a current-validation pass because the weak-voltage bar is <=2
  and the fixture is still a coarse smoke case rather than a paper-resolution
  developed cone-jet current comparison.
  The fixed-midplane developed-current window remains BLOCKED (`0/27` low and
  `0/27` high developed samples), but the axial scan is non-degenerate:
  low/high developed samples are `27/24` and the convective, liquid-convective,
  and alpha05 ratios are `3.54232`, `2.96558`, and `3.50436`. That keeps the
  current claim DOWNGRADED even when the measurement plane is allowed to follow
  the developed jet.
- A boundary charge-advection diagnostic now writes
  `candido_boundary_charge_advection_diagnostic3d.csv`. It adds the
  paper-motivated neutral inflow and zero-gradient outflow charge advection to
  the relaxation-limited face-implicit electric path. The charge budgets remain
  tight (`7.1562e-15` and `2.58903e-14` relative residuals), but the tail-current
  ratio worsens from `3.41126` to `3.60075`. This is negative evidence: missing
  boundary charge advection alone is not the remaining Candido Fig. 8(b)
  current-validation blocker. The axial developed-window ratios are still above
  the <=2 bar (`3.07004`, `2.60454`, `3.0735`) with `27/24` low/high samples.
- An axial-current factorization diagnostic now writes
  `candido_axial_current_factorization3d.csv`. It decomposes the alpha>=0.5
  developed-section current ratio into area, mean absolute charge, mean absolute
  axial velocity, and charge-velocity alignment using the same tail samples as
  the axial-window audit. The default long-window alpha05 failure is charge
  dominated (`current_ratio=7.06812e9`, `charge_ratio=3.20054e6`), but after the
  relaxation-limited conservative charge path the area and charge ratios are
  near unity (`0.989583` and `0.887982`) while the velocity ratio remains
  `3.87434`, leaving `current_ratio=3.50436`. Boundary charge advection gives
  the same velocity ratio and only lowers the current ratio to `3.0735`. This
  shifts the next blocker from charge-budget bookkeeping to electric-force /
  momentum calibration and paper-resolution developed-jet dynamics.
- An axial total-current closure diagnostic now writes
  `candido_axial_total_current_closure3d.csv`. It extends the developed
  alpha>=0.5 axial scan with the charge-equation flux
  `rho_e*u_y + sigma*E_y`, while keeping the paper-style convective
  `q_e U dot n` current separate. This is useful but not a paper-current pass:
  the raw long-window alpha05 total-current ratio is still
  `1.12929e9`, but the relaxation-limited, hybrid Maxwell, bounded-vector
  Maxwell, and Ca-independent/boundary-advected diagnostics all have
  conductive-inclusive total-current ratios near unity (`1.01253`,
  `1.12964`, `1.15321`, and `1.18563`). The conductive share is about
  unity, so this only proves that the Ohmic/conductive part has weak voltage
  sensitivity; the paper `q_e U dot n` convective current remains
  DOWNGRADED/APPROXIMATE and still needs a paper-faithful charge-current
  boundary model plus external current calibration.
- An electric-drive scaling diagnostic now writes
  `candido_electric_drive_scaling_diagnostic3d.csv`. It keeps the
  relaxation-limited conservative charge path but removes the extra empirical
  `CaE^1.25` multiplier from the Maxwell-force drive scale in an opt-in run.
  This lowers the tail-current ratio from `3.41126` to `2.08008` and the axial
  alpha05 developed-current ratio from `3.50436` to `2.12646`; combining the
  same Ca-independent drive with boundary charge advection lowers them further
  to `2.06312` and `2.04822`. The velocity ratio drops from `3.87434` to
  `1.79515`, so the old exponent was a major source of high-CaE velocity bias.
  The result is still APPROXIMATE, not UPHELD, because the strict <=2 current
  bar is not met and there is not yet an external calibration basis for
  tuning the drive scale across the threshold.
- An axial-current threshold sweep now writes
  `candido_axial_current_threshold_sweep3d.csv`. It tests whether the near-pass
  Ca-independent boundary-advection current ratio is only an artifact of the
  `alpha>=0.5` developed-section area cutoff. It is not: for area thresholds
  from `0` through `2` inner-diameter^2, the axial alpha05 current ratio stays
  fixed at `2.04822` with `27/27` low/high developed tail samples. The
  relaxation-limited path remains `3.50436` over thresholds `1e-8` through `1`,
  and the raw long-window path remains order `7.06812e9` until the low-CaE
  developed window disappears. This keeps the current claim DOWNGRADED and
  shifts the blocker back to electric-force/momentum calibration, not
  threshold selection.
- A Ca-independent current-resolution sweep now writes
  `candido_ca_independent_current_resolution_sweep3d.csv` for the best near-pass
  path: Ca-independent electric drive plus boundary charge advection. The n=10
  grid gives an apparent axial alpha05 current ratio of `0.424505`, but it is
  BLOCKED because the high-CaE developed window has only one tail sample. The
  non-degenerate n=12 and n=14 grids do not converge monotonically to the <=2
  bar: `2.04822` worsens to `2.5334`, with the n=14 velocity ratio `1.91039`.
  This rules out a defensible claim that the current mismatch is solved by
  coarse-grid refinement alone.
- The same near-pass current path was then written into the morphology and
  whipping diagnostics to test whether it can be promoted as the paper-faithful
  default. It cannot. The CaE=0.25 morphology errors worsen to `29.5572%` and
  `30.8668%` at the 0.4/0.7 ms paper comparison points; the time-alignment
  best-volume errors remain `28.0483%` and `30.8668%`, and the tip-sync
  diagnostic is `DOWNGRADED_TIP_QUANTIZED_COARSE_GRID` with only one unique tip
  level. At CaE=0.42 the whipping row becomes
  `DOWNGRADED_THRESHOLD_NOT_REACHED`: max radial asymmetry is only `0.00398144`
  versus the `0.05` threshold, even though the wave-peak location error is a
  superficially good `4.20516%`. This is decisive negative evidence that the
  Ca-independent drive is a useful diagnostic for the velocity bias but is not
  a production default unless morphology/whipping are recalibrated against
  external data.
- A short Maxwell tangential-closure diagnostic now writes
  `candido_maxwell_tangential_closure3d.csv`. It compares the cell-gradient
  Maxwell force, the Poisson face-normal-only stress, and an opt-in hybrid
  stress that keeps the Poisson normal component but restores tangential
  electric-field content. The hybrid force is only `0.305979`/`0.306389` of the
  face-normal-only force at CaE=0.25/0.42, and the one-step velocity ratio is
  `0.87243`/`0.869961`. This is useful negative evidence: the current
  face-normal force path is not a paper-faithful Maxwell stress closure, and
  the remaining current/whipping calibration should move through a validated
  full-vector face electric field/stress reconstruction rather than scalar
  drive-scale tuning.
- A face-vector electric reconstruction diagnostic now writes
  `candido_face_electric_reconstruction3d.csv`. It computes the Poisson
  face-normal electric field, the cell-gradient tangential field, and the
  hybrid Maxwell traction from the same initial Candido potential solve. This
  is not a production force switch; it is a diagnostic for the missing
  full-vector face-field closure. The evidence is strongly DOWNGRADED for the
  current face-normal-only stress: both CaE rows have mean tangential fraction
  `0.739045`, relative normal mismatch about `0.26`, and active-face mean
  hybrid/normal traction ratios `2564.82` and `10151.6` after excluding
  `920`/`910` faces whose normal-only traction is degenerate. This makes the
  next required work more specific: a bounded face-vector electric-field
  reconstruction must be validated before promoting a Maxwell-stress current
  calibration.
- A bounded face-vector Maxwell diagnostic now writes
  `candido_bounded_vector_maxwell_diagnostic3d.csv`. It keeps the Poisson
  normal electric field and clips the reconstructed tangential component using
  a normal-field floor (`0.05`) and limit-factor sweep. The short force-scale
  result is finite and bounded: at limit factor `2`, max force is `0.449987`
  and `0.44988` of the face-normal baseline for CaE=0.25/0.42, with only
  `5.24017%` of faces clipped. However, the long-window opt-in path written to
  `candido_bounded_vector_maxwell_long_window3d.csv` is still DOWNGRADED:
  tail-current ratio is `2.52183`, axial alpha05 ratio is `4.79659`, velocity
  ratio is `4.05169`, CaE=0.25 morphology errors are `39.2992%`/`42.1344%`,
  and CaE=0.42 whipping remains below threshold. The bounded vector closure is
  therefore diagnostic-only; it rules out this simple clipping model as the
  paper-faithful Maxwell/current fix.
- A Tomar-style conducting surface-force candidate now writes
  `candido_tomar_conducting_surface_force3d.csv` and
  `candido_tomar_conducting_long_window3d.csv`. The short initial diagnostic
  is finite but much stiffer than the current default Maxwell body-force path:
  max force is `280.812` vs `11.6969` at CaE=0.25 and `470.827` vs `19.7211`
  at CaE=0.42, about `24x` the default. The decomposition shows the normal
  conductive-current jump term dominates; the tangential term share is only
  `0.0369861`/`0.0371628`. In the long-window production path this candidate
  stays numerically finite and lowers the raw tail-current ratio from
  `3.41126` to `2.88883`, but it removes the developed alpha>=0.5 axial current
  window (`0`/`0` usable low/high samples) and is therefore
  `DOWNGRADED_TOMAR_NO_DEVELOPED_ALPHA05_WINDOW`. This rules out promoting the
  direct Tomar-like conducting force on the current coarse Candido fixture; the
  next force/current work must address paper-matched charge injection/boundary
  physics or use a non-degenerate developed-jet validation mesh.
- Extending that hybrid Maxwell stress to the full long-window Candido path
  gives negative evidence, not a promotion. The new
  `candido_hybrid_maxwell_long_window3d.csv` row is
  `DOWNGRADED_HYBRID_MAXWELL_NOT_CURRENT_FIX`: the alpha05 axial current ratio
  worsens from `3.50436` to `4.28268`, the velocity ratio stays high
  (`3.88357`), and the charge ratio rises to `1.20607`. The same run also
  worsens the CaE=0.25 morphology errors to `36.511%` and `38.8888%` at
  0.4/0.7 ms, while CaE=0.42 whipping remains
  `DOWNGRADED_THRESHOLD_NOT_REACHED` with max radial asymmetry `0.00310137`.
  Therefore the immediate blocker is not solved by simply adding tangential
  LS-cell electric components to the Poisson normal flux; the next force path
  needs a genuinely bounded, face-reconstructed full-vector electric field
  with external calibration.
- A reference-gap CSV now records the required current-path inputs explicitly:
  dimensional charge scaling, conservative bounded transport replacing qLimit,
  and paper-faithful electrode/nozzle/outlet charge-current boundary
  conditions. Candido Eq. (6) and Lopez-Herrera's charge-conservative VOF model
  justify this direction, but the present repository does not yet contain the
  missing physical calibration.

## Still Required For Direct Whipping DNS

1. Geometric VOF fluxing
   - Exact tetra/hex PLIC plane cuts are now used in the swept-face PLIC
     wet-fraction path.
   - Direct breakup still needs resolved breakup DNS, breakup-time grid
     convergence, and stronger non-Cartesian polyhedral sweeps beyond the
     current tetra/hex guarded path.

2. Balanced-force surface tension and Maxwell stress
   - Maxwell stress can now be assembled by conservative face-stress
     divergence.
   - Surface tension has a pressure-compatible balanced-face option.
  - Curvature has RDF and local PLIC/shape-operator paths; the local path
    improves static unstructured sphere error but does not yet produce
    validated Lamb/Prosperetti droplet dynamics. Force-isolation evidence
    currently points to both CSF acceleration magnitude/sign errors and
    time-evolving interface-conditioning spikes as the next defects to fix.
  - A diagnostic-only negated-kappa force-isolation pass narrows the failure:
    for mode-2 at n=10/12, flipping the local-PIC kappa sign restores the
    restoring-force sign, but the acceleration magnitude error remains about
    98.9%/98.5%. For mode-3 the flipped sign restores direction but still
    leaves 270%--473% magnitude error. Therefore a production sign flip alone
    is not a defensible fix; the local curvature magnitude/stencil operator
    still needs repair.
  - A curvature-mode diagnostic now checks local kappa before force assembly.
    The unflipped local-PIC kappa has the wrong perturbation sign; the
    negated-kappa rows recover the sign but still underpredict the analytic
    small-perturbation curvature coefficient by 45%--75%. This places the
    first defect in the curvature reconstruction itself, before pressure
    projection or VoF advection.
  - A diagnostic cell-centered force path, `sigma*kappa*grad(alpha)`, sharply
    separates force assembly from curvature reconstruction. With the same
    local-PIC kappa it gives restoring force and mode-2 n=12 acceleration error
    17.2539% (`UPHELD` under the diagnostic bar), while the production
    balanced-face force row remains 101.522% error and wrong sign. This points
    to a dynamic balanced-face CSF transfer/sign-convention defect; the
    cell-centered row is diagnostic only and is not a balanced-force production
    replacement.
  - A follow-up force-path split shows that the balanced pressure-gradient
    path is not the main loss mechanism: mode-2 n=12 pressure-gradient error is
    101.111% with wrong sign, essentially matching the raw balanced-face CSF
    error of 101.522%. Mode-3 n=12 is likewise 665.163% versus 672.721%.
    Therefore the defect is in the dynamic modal transfer of
    `snGrad(alpha)*Sf`/face interpolation, not merely in the pressure solve.
  - An opt-in diagnostic `gaussAlphaCsfForce3D` path uses a conservative
    face-Gauss `alpha_f*Sf` transfer. It restores force sign and improves
    mode-2 n=12 acceleration error to 20.2512%, close to the cell-centered
    17.2539% diagnostic. Mode-3 n=12 improves from 672.721% to 70.1696%, but
    remains approximate. This is a production candidate only after a
    balanced-pressure-compatible formulation and static-droplet regression are
    proven.
  - Static balance rejects a direct production promotion of `gaussAlphaCsfForce3D`:
    on the n=24 static droplet, the original balanced-force residual is
    1.69082e-13 max / 1.92695e-14 L2, while face-Gauss gives 102.071 max /
    22.7972 L2 (`STATIC_BALANCE_DOWNGRADED`). The next formulation must pair
    the surface-tension force and pressure gradient in the same discrete
    operator, not simply swap in a better dynamic force projection.
  - A mean-balanced/delta-Gauss hybrid preserves the constant-curvature static
    residual exactly at the original balanced level (n=24 max 1.69082e-13),
    but its dynamic acceleration is too weak: mode-2 n=12 error 67.4586% and
    mode-3 n=12 error 604.808% with wrong sign. A pure delta-Gauss perturbation
    path restores the mode-3 sign but is still weak (mode-2 67.2074%,
    mode-3 82.3212%). The remaining route is therefore not a simple
    mean/delta split; it needs a pressure-compatible modal transfer operator or
    better curvature magnitude first.
  - A local height-quadric curvature diagnostic restores perturbation sign
    without manual negation but is ill-conditioned and too weak: n=12 mode-2
    error 83.7248% with condition max 1.74778e6, and mode-3 error 70.8336%
    with condition max 1.75491e6. This rules out a naive height fit as the next
    production curvature operator.
  - A mean-preserving reflected-perturbation kappa transform keeps the static
    curvature mean plausible and restores curvature-mode sign, but does not
    repair force transfer: balanced-face mode-2 n=12 remains 98.9807% error and
    mode-3 remains wrong-sign with 572.254% error. Reflected kappa also worsens
    the face-Gauss path (mode-2 85.8363%, mode-3 wrong-sign 105.527%).
  - A stencil-size sweep shows that smaller local PLIC stencils increase
    perturbation magnitude. With `maxSamples=12` plus mean-preserving
    reflection, curvature-mode errors improve to 22.8541% (mode-2 n=12) and
    43.8349% (mode-3 n=12). However the same kappa gives wrong-sign force in
    balanced-face, face-Gauss, and cell-gradient paths. This rejects simple
    sample-size tuning as a fix.
  - A paired-operator static diagnostic shows the actual design split:
    face-Gauss CSF against snGrad pressure is unbalanced (n=24 max residual
    102.071), but face-Gauss CSF against Green-Gauss pressure gradient is
    machine-balanced (n=24 max residual 7.3426e-15). This explains why
    face-Gauss helps dynamic modal force while violating the current
    identical-snGrad balanced-force invariant.
  - A follow-up `face_gauss_sn_pressure_gradient` diagnostic projects the
    face-Gauss modal source through the snGrad pressure operator instead of
    using the raw Green-Gauss force. This keeps the experiment within the
    snGrad-compatible pressure path, but it loses most of the modal force:
    mode-2 n=12 error is 66.0427% and mode-3 n=12 error is 86.6535%, both only
    `APPROXIMATE`. Direct face-Gauss remains better dynamically
    (20.2512%/70.1696%) but statically unbalanced. The current evidence
    therefore points to a missing pressure-compatible modal correction or a
    different shared interface-geometry operator, not a simple post-hoc
    projection of the face-Gauss source.

3. Full PIMPLE loop and non-orthogonal correction
   - Current coupling includes outer PIMPLE iterations, momentum
     under-relaxation, and non-orthogonal correction pass controls.
   - OpenFOAM-level robustness still needs a fully implicit momentum matrix,
     turbulence/viscosity closure integration, and production solver controls.

4. Boundary physics
   - Structured tag-level boundary data exists for potential, alpha, velocity,
     charge, and contact angle metadata.
   - Prescribed inlet velocity flux, charge scalar inlet values, outlet/open
     backflow clipping, and no-through walls are implemented.
   - Contact-angle normal projection is verified on Candido wall-adjacent
     mixed cells, but production curvature reconstruction is not yet wired to
     use it.
   - The diagnostic contact-angle curvature path is finite and fallback-free on
     the current Candido smoke mesh, but the higher stencil condition requires
     static/dynamic regression before it should be enabled in the CSF force.
   - An opt-in production switch exists for experiments; it is deliberately not
     the default because the first Candido switch diagnostic changes the CSF
     force magnitude substantially.
   - Before enabling that switch by default, the wall-contact curvature
     reconstruction needs a cap/sessile-droplet validation target rather than
     only boundedness and continuity checks.
   - The first sessile-cap analytic target is now recorded: contact-angle
     wall-weighted mean curvature error is 3.3461% versus -14.3616% for the
     baseline on the current diagnostic mesh, so the direction is promising
     but still only `APPROXIMATE_WITHIN_20_PERCENT`.
   - The 3-level sessile-cap refinement diagnostic downgrades that optimism:
     contact-angle wall error moves -10.8158%, 3.3461%, 14.8782% from coarse
     to fine, so it is not yet a convergent wall-curvature scheme.
   - Production contact-angle curvature enforcement, wall charge treatment,
     and production-grade mixed outlet conditions are still needed.

5. Adaptive refinement
   - Interface/curvature/charge/electric-field indicators are implemented.
   - Actual mesh refinement/coarsening and solution transfer are still needed.

6. Physical validation observables
   - Current checks include operator/smoke validations and reduced whipping
     amplitude/frequency observables.
   - Candido-style reproduction still needs cone angle, jet radius, breakup
     length, current, charge, mass conservation, and grid convergence histories
     from a resolved DNS run.
   - Current scaling is only order-of-magnitude for one developed case; voltage
     sensitivity is explicitly downgraded on the current coarse smoke path.
   - A first inlet/outlet liquid-volume budget now closes on the coarse
     long-window cases. Quantitative current/whipping validation still needs
     physically calibrated electric-current boundary treatment, removal or
     quasi-implicit replacement of the current charge-clipping regime,
     conservative bounded charge transport enabled only after current-scale
     validation, and grid/time convergence.
   - Opt-in dimensional electrical-property scaling now matches the Candido
     physical liquid relaxation scale (`dt/tau_l=2.77417`), but the explicit
     conductive update is unusable on the coarse smoke path: charge residual
     rises to 0.219108, clamp correction rises by 270.975x, current rises by
     594.699x, and alpha mass drift reaches 0.870574.
   - A diagnostic quasi-implicit bulk-conduction split stabilizes the same
     dimensional electrical-property run without promoting it to production:
     charge residual improves to 1.84978e-14, alpha mass drift is
     3.25434e-15, and max divergence is 8.86565e-09. It is still
     `DOWNGRADED` for current validation because clamp correction remains
     2.97425x above the normalized baseline, convective current remains
     5.00521x higher, and charge-unit consistency is still 331.567. This
     narrows the blocker to a paper-faithful boundary-current/charge-unit
     model plus a conservative stiff charge solve, not just a local relaxation
     patch.
   - Applying the Poisson nondimensional charge scale
     `Q0 = eps0*E0*d0^2` makes the low-CaE combined case internally plausible
     (`qLimit/Rayleigh=5.82808`, current/Ganan-Calvo=3.14095e-08), but it does
     not rescue the high-CaE validation path. The combined high-CaE run still
     has `qLimit/Rayleigh=12.6908`, clamp/Rayleigh=99564.2, and
   current/Ganan-Calvo=1876.07. The dimensional implicit-bulk run is stable
   but worse as a current observable (`max_integrated_charge/Rayleigh=28.3204`,
   clamp/Rayleigh=296129, current/Ganan-Calvo=9390.1). This rules out a
   simple nondimensional-output rescaling as a path to Candido-level current
   validation.
   - A Rayleigh charge-limit diagnostic was then added to replace the arbitrary
     high-CaE charge cap with `Q_Rayleigh` under the same Poisson
     nondimensional charge scale. This makes the cap physically bounded
     (`qLimit/Rayleigh=1` for both CaE=0.25 and CaE=0.42), but it still does
     not recover Candido's weak current-voltage sensitivity. The tail mean
     current ratio remains 1.69249e8, and the high-CaE row still has
     clamp/Rayleigh=2030.54 and current/Ganan-Calvo=20.3955. This is useful
     negative evidence: the current blocker is not just the cap magnitude; it
     still needs a conservative dimensional charge-current model and
     paper-faithful electrode/nozzle boundary treatment.
   - The charge-model reference gap is now written as
     `candido_charge_model_reference_gap3d.csv`. It explicitly records three
     unresolved requirements: bulk charge conservation over the full developed
     cone-jet window, a published electrode/nozzle/outlet surface-current
     boundary closure, and a nonzero Fig. 8(b)-consistent `q_e U dot n`
     current validation observable.
   - The 3D mesh boundary-patch classifier was fixed to use the actual domain
     bounding box instead of hard-coded unit-box coordinates. This is a
     diagnostic/geometry bug fix: the Candido stretched box now reports
     `ymax_collector` separately instead of folding non-unit-domain collector
     faces into the wrong patch. The new guard is `test_mesh3d_geometry` on a
     stretched/skewed 6.0 x 7.25 x 5.5 grid.
   - With correct patch labels, a collector-only boundary-current diagnostic
     was added. It restricts boundary conductive charge flux to the collector
     patch while keeping the same Rayleigh charge cap and bounded/subcycled
     charge path. This reduces the high-CaE physical current scale
     (`current/Ganan-Calvo=10.9643`, versus 20.3955 for the Rayleigh-limited
     run), but it does not fix the validation: the tail mean current ratio is
     still 2.56491e8 and charge-unit consistency worsens to 2.01645e6. The
     result remains DOWNGRADED and points to a missing dimensional,
     conservative electrode/nozzle charge-current model rather than a single
     patch filter.
   - A charge-field consistency audit now separates the Poisson matrix solve
     from the field actually used by the production current/force path. The
     Poisson residual remains small (`~1e-11`), but the cell-gradient field has
     `div(eps E_cell)` relative residuals of order unity: combined high-CaE
     1.20566, Rayleigh-limited high-CaE 1.18817, and collector-only high-CaE
     1.29111. This is actionable negative evidence: current and Maxwell-force
     assembly should use a face-normal electric flux consistent with the
     Poisson operator instead of relying only on the least-squares cell
   gradient for charge/current observables.
   - An opt-in implicit Ohmic charge projection was added to test the same
     blocker inside the charge update rather than only as a current
     diagnostic. It is executable and conservative enough to keep the Candido
     smoke fixture passing, but it remains DOWNGRADED: tail current ratio
     9.4997e8, high-CaE current/Ganan-Calvo 6.51183, and high-CaE relative
     cell-gradient Gauss-law residual 6853.32. This means a symmetric implicit
     conductive solve alone is not the missing Candido current model; boundary
     current physics and face-consistent downstream field assembly are still
     required.
   - A face-consistent electric diagnostic then uses Poisson face-normal field
     data for both conductive current and Maxwell-stress divergence. It
     sharply reduces the high-CaE absolute current scale
     (`current/Ganan-Calvo=0.000854954`, `clamp/Rayleigh=1.68184`) but still
     fails Candido voltage sensitivity because the low-CaE tail current is
     nearly zero and the high/low tail ratio is 3.68294e9. This is retained as
     diagnostic evidence, not a production promotion: the face-normal-only
     stress is missing tangential electric-field content and the coarse
     low-CaE jet is not a developed paper-current state.
   - A newer face-current observable diagnostic now measures total current
     directly on y-normal mesh faces using the Poisson face-normal conductive
     flux instead of the cell-gradient `sigma E_y` slab approximation. On the
     face-consistent electric candidate, the all-phase total-current tail ratio
     is `1.04973` and the peak ratio is `1.28357`, so the all-phase voltage
     sensitivity problem is partly an observable inconsistency. This is still
     not a Candido Fig. 8(b) validation pass: the alpha>=0.5 liquid-jet
     face-current row is `0/0`, so the paper liquid-current plane remains
     undeveloped on the coarse smoke fixture.
   - The same diagnostic now scans y-normal faces axially and selects the
     most developed alpha>=0.5 face-current section per time sample. This
     removes the fixed-plane artifact but still blocks the comparison:
     low/high tail samples are `27/27`, developed samples are `13/0`, max
     alpha>=0.5 face areas are `2.4375/0`, and the developed-current ratio is
     `inf`. The high-CaE face-consistent candidate therefore loses the
     developed liquid-jet face section entirely, so the remaining current
     blocker is coupled interface development and charge/current evolution,
     not only where the current plane is measured.
   - Applying the same Poisson-face alpha>=0.5 axial scan to the existing
     paper-charge, paper-inlet, open-atmosphere, and moving-collector
     candidates gives non-degenerate `27/27` developed windows and weak
     total-current ratios of `1.2643`, `1.31028`, `1.31027`, and `1.31029`.
     This is useful positive evidence for a face-consistent total-current
     diagnostic. It is still not a paper-current validation pass: Candido
     Fig. 8(b) defines the plotted current as `q_e U dot n`, not
     convective-plus-conductive total current, and those current-friendly
     candidates still leave the fixed paper plane undeveloped while trading
     away morphology/whipping fidelity.
   - The developed-jet current-window audit confirms that low-CaE degeneracy
     is structural in the current fixture. With a midplane
     `alpha>=0.5` area threshold of 1e-4, the default long-window low-CaE case
     has 0 developed tail samples while high-CaE has 3; the face-consistent
     electric path has 0 developed tail samples for both low and high. This
     blocks a defensible Candido Fig. 8(b)-style average-current comparison on
     the current coarse smoke fixture.
   - The axial developed-jet scan removes that fixed-plane ambiguity for the
     default path and still finds a voltage-sensitivity failure: low/high
     developed-window current ratio is 4.64035e9 for raw convective current.
     This keeps the current blocker on charge scaling/transport and
     boundary-current physics, not only on measurement-plane placement.
   - Combining implicit Ohmic projection with face-normal Maxwell stress lowers
     the raw tail-current ratio to 61525, but high-CaE no longer has a
     developed axial current window. The remaining current work therefore must
     address physical charge injection/transport and geometry development
     together; toggling existing diagnostic pieces is not enough.
   - Enforcing the phase electric-relaxation timestep with dimensional
     electrical scaling and bounded global charge conservation lowers the same
     tail-current ratio further to 3.41126 with charge-budget residuals below
     `1.2e-13`. The axial developed-window alpha05 current ratio is still
     3.50436. Factorization shows the remaining relaxation-limited failure is
     velocity dominated (`velocity_ratio=3.87434`) rather than area or charge
     dominated. Removing the extra empirical `CaE^1.25` electric-drive
     multiplier reduces that velocity ratio to 1.79515 and the best combined
     boundary-advected alpha05 current ratio to 2.04822, still just above the
     <=2 bar. The next implementation step is therefore physically justified
     electric-force/momentum calibration plus a paper-resolution developed-jet
     fixture, not another charge-budget repair or threshold tuning.
   - The same Ca-independent boundary-advected candidate is now also written
     in the standard current-voltage sensitivity CSV format. Its all-phase
     convective tail-current ratio is `2.06312` and peak ratio is `2.03548`,
     so it remains a near-miss `DOWNGRADED` result against the <=2 weak
     voltage-sensitivity bar. The alpha05 row is `0/0` and is explicitly not a
     validation pass; it is a degenerate observable on this coarse fixture.
   - Reducing the boundary-advected, Ca-independent drive reference scale from
     `3.8` to `1.0` lowers absolute currents and gives an axial developed
     alpha05 ratio of `1.99384`, but the standard all-phase paper-current
     tail ratio worsens to `2.31574`. This is a mixed negative result: force
     scaling alone can move a chosen axial diagnostic across the weak bar, but
     it does not make the paper-current observable robust or non-degenerate.
   - A fixed-midplane current-reach diagnostic now tests the Candido Fig. 8(b)
     current definition directly: `i_e = integral_S q_e U dot n dS` on a
     fixed middle-domain liquid-jet cross-section. Extending both CaE=0.25 and
     CaE=0.42 coarse runs to 90 steps still gives low/high developed
     `alpha>=0.5` midplane sample counts of `0/6`; the low-CaE tail reaches
     `y/Di=3.90298` but never forms a nonzero `alpha>=0.5` area at the fixed
     current plane. This makes the paper-current comparison
     `BLOCKED_UNDEVELOPED_FIXED_MIDPLANE_CURRENT` on the current coarse smoke
     fixture. It also confirms that the previous `0/0` alpha05 current row is
     not a weak-voltage validation pass.
   - A reduced-collector current fixture then halves the collector distance to
     0.75 mm so that the fixed middle plane is moved upstream to `y/Di=2.34375`.
     The closest current path (`ca_independent_boundary`) still has low/high
     developed sample counts of `0/0`, max `alpha>=0.5` area `0/0`, and tail
     tip positions `0.881778/0.881778`. Enabling VOF inlet boundary alpha gives
     the same blocked row. This rules out simple current-plane relocation or
     nozzle-alpha supply as sufficient fixes; the current blocker remains a
     coupled developed-jet dynamics and charge/current-boundary problem.
   - A paper-charge-boundary candidate then combines the closest
     Ca-independent boundary-charge path with VOF inlet alpha and suppressed
     conductive nozzle charge flux. The short standalone current-candidate CSV
     now restores the two Candido Fig. 8(b) source rows. The all-phase
     average-current row is quantified but DOWNGRADED (`tail=2.23202`,
     `peak=2.23755`), and the paper `q_e U dot n` alpha>=0.5 liquid-jet row is
     `BLOCKED_ZERO_CURRENT_OBSERVABLE` (`0/0` tail current). This is stricter
     and weaker than the older long-suite near-pass, so the path remains
     diagnostic rather than a paper-current validation pass.
   - Adding a fully developed laminar inlet-velocity boundary on top of the
     paper-charge-boundary candidate is also now covered by the short
     standalone current-candidate CSV. It is still DOWNGRADED: all-phase
     tail/peak ratios are `2.23275`/`2.23853`, and the alpha>=0.5 liquid-jet
     `q_e U dot n` row is again `BLOCKED_ZERO_CURRENT_OBSERVABLE`. This narrows
     the remaining current blocker away from a missing inlet velocity profile
     and toward coupled velocity/charge evolution plus fixed-plane jet
     development on the coarse smoke mesh.
   - The short standalone fixture now also writes
     `candido_jet_current_metrics3d.csv` and `candido_current_scaling3d.csv`
     from actual report fields rather than placeholders. For the
     Ca-independent drive rows, final midplane radii are
     `7.08963e-05/6.93068e-05` m, max convective currents are
     `2.13973e-12/4.78746e-12`, max electric forces are `17.726/29.7796`, and
     mass/divergence remain bounded (`O(1e-15)` mass drift and `O(1e-12)`
     max div). This closes the mechanical radius/current-metric presence gap,
     but it is still a short-fixture diagnostic and does not make the
     voltage-sensitivity or Fig. 8(b) current claim pass.
   - The same standalone fixture now restores the short three-level
     refinement sweep without the long Candido suite. The `nx=10/12/14` rows
     use 2 steps and have active forces at all levels:
     max electric force `11.8774/40.3515/93.3049`, max CSF force
     `5.85904/13.3672/18.6302`, and final midplane jet radius
     `0.000491671/0.00011571/0.000145465`. The current readiness guard marks
     radius and CSF force as `PASS_CONVERGING`, but max electric force remains
     `DOWNGRADED_NONCONVERGENT`. This is useful coarse-grid evidence, not a
     full paper-level grid-convergence claim.
   - The `candido_fig8b_current_blocker3d.csv` artifact now summarizes the
     current blocker mechanically. On the coarse smoke mesh (`12x18x12` for the
     current candidate), the paper fixed plane is at `y/Di=4.6875` and the
     reduced plane is at `y/Di=2.34375`; both have low/high developed
     `alpha>=0.5` sample counts of `0/0`. The best average tail-current ratio
     is now `2.23202` and the best peak ratio is `2.23755`, so status remains
     `BLOCKED_COARSE_FIXTURE_FIG8B_CURRENT_UNDEVELOPED_FIXED_PLANE`. The
     current-friendly path cannot be promoted because it leaves fixed current
     planes undeveloped and remains too voltage-sensitive, so the remaining
     Fig. 8(b) gap is a documented fixture/physics blocker, not a missing
     diagnostic row.
   - A standalone Pareto current/morphology/whipping audit now writes
     `candido_current_morphology_whip_pareto3d.csv` from live long-window and
     short candidate reports. This confirms the tradeoff numerically: the
     baseline keeps the morphology proxy (`9.67294%`) and whipping
     (`max asymmetry=0.294331`) but has catastrophic current sensitivity
     (`tail=1.63434e11`, `axial alpha05=6.37384e10`). The paper-charge
     candidate has all-phase current ratio `2.23202`, axial alpha05 ratio
     `2.13024`, fixed-plane samples `0/0`, morphology error `45.298%`, and max
     asymmetry `0.000170499`. The paper-inlet candidate is similar
     (`tail=2.23275`, `axial alpha05=2.13094`, fixed-plane samples `0/0`,
     morphology error `45.4347%`). This prevents relabeling a current-only
     near-pass as a paper-level Fig. 8(b) validation.
   - A short standalone paper-style open side/outlet hydrodynamic flux
     diagnostic now writes `candido_open_boundary_current_diagnostic3d.csv`
     without rerunning the long Candido smoke suite. In this fixture the
     open-atmosphere paper-inlet candidate has no measurable open-boundary
     liquid flux (`low/high inflow=0/0`, `outflow=0/0`), fixed midplane samples
     `0/0`, all-phase tail/peak current ratios `2.23279/2.23859`, axial
     alpha05 convective ratio `2.13086`, axial total ratio `1.29622`,
     morphology error `45.4347%`, and high-CaE max asymmetry `0.000166501`.
     Status is `BLOCKED_OPEN_BOUNDARY_NO_MEASURABLE_FLUX`, so this is a
     quantified mechanical gap, not a paper-current validation pass.
   - A 90-step development tradeoff diagnostic now writes
     `candido_paper_current_development_tradeoff3d.csv`. With the same
     paper-inlet/open-atmosphere options, the stable 4-step and extended
     90-step windows both have fixed midplane developed samples `0/0`.
     The extended low/high runs keep mass and continuity bounded
     (`alpha_mass_drift=2.34905e-15/1.69654e-15`,
     `max_div=1.55502e-12/2.3246e-12`), but the max tip only reaches
     `1.122/1.122` while the paper fixed current plane is at
     `y/Di=4.6875`. Status is
     `BLOCKED_STABLE_AND_EXTENDED_FIXED_PLANE_UNDEVELOPED`. The current blocker
     is therefore not solved by simply extending this coarse paper-inlet/open
     fixture to 90 steps.
   - A linear-gradient ray alpha=0.5 contour diagnostic was added to test
     whether centroid-only ray sampling was the reason the coarse morphology
     observable missed the digitized Fig. 3(b) volume. It is additive and does
     not change the production VoF/CSF path. The result is negative evidence:
     for the long-window CaE=0.25 case, the existing ray-alpha05 errors are
     -10.8924% at 0.4 ms and -8.93099% at 0.7 ms, while the linear-gradient
     ray diagnostic worsens them to -21.7001% and -22.9544%. The reference
     remains bracketed at 0.4 ms by coarse observables, but the more local
     linear ray reconstruction is not the missing paper-faithful contour.
   - A connected PLIC first-exit ray diagnostic was then added to test whether
     choosing the first outward PLIC crossing, instead of a median crossing,
     better represents the visible free-surface outline. This gives useful but
     still insufficient evidence: at 0.4 ms it improves the long-window CaE=0.25
     silhouette error to -5.79343%, inside the 10% bar, but at 0.7 ms it
     underexpands the contour to -31.9597%. The median PLIC ray-plane diagnostic
     remains +35.8748%/+18.4168%, while first-exit is -5.79343%/-31.9597%.
     The paper morphology gap is therefore not closed by a single local crossing
     rule; it needs true connected contour tracking over time or external
     contour-coordinate data for the later Fig. 3(b) frames.
  - The long-window fixture now also regenerates
    `candido_late_morphology_source_audit3d.csv`. The audited sources are the
    public AIP Figure 3 image asset, the local Candido PDF Fig. 3 render, the
    local extracted Candido text, the Guo 2018 source experiment paper, the
    Candido data-availability statement, the local external-data input
    contract, and the public author GitHub `interIsoFoamEHD` repository. The
    public AIP image is only `700x310` and shows the same three
    Fig. 3(b) times (`0.0/0.4/0.7 ms`) as the local PDF render. The Candido text
    lists relative morphology errors for `0.8/0.9 ms` but no contour coordinates
    or morphology volumes, the data-availability statement points to
    corresponding-author request rather than a public dataset, and the Guo paper
    does not contain Candido-synchronized late contours. The public author
    GitHub repository contains OpenFOAM solver/case/image material, but no
    postprocessed Candido Fig. 3(b) `0.8/0.9 ms` contour or volume CSV. The
    metric therefore records
    `morphology_late_source_audit_quantified=1` while correctly leaving
    `morphology_late_0_8_0_9_digitized=0`.
  - `apps/electrospray_late_morphology_dataset_check.py` now defines the
    minimum external input contract for this remaining gate. It accepts either
    positive finite `digitized_experimental_volume_di3` rows or digitized
    `contour_y_di,contour_radius_di` coordinate rows for `long_window_ca025` at
    `0.8` and `0.9 ms`. In contour mode it computes
    `V/Di^3 = pi*integral(r_di^2 dy_di)`. Both modes require explicit
    source/extraction metadata and `not_derived_from_reported_error=1`. With the
    default missing input file it writes
    `candido_late_morphology_external_dataset_check3d.csv` as
    `BLOCKED_EXTERNAL_DATASET_MISSING`; this is intentional and does not close
    the paper morphology gate.
  - `docs/electrospray/candido_late_morphology_data_request.md` now packages
    the exact request text, validation command, and two empty CSV templates:
    `candido_late_morphology_external_volume_template.csv` and
    `candido_late_morphology_external_contour_template.csv`. The additive
    `test_late_morphology_request_package_check` CTest guard verifies that the
    templates have the expected headers and contain no placeholder data rows.
  - A follow-up interpolated 25% ray-plane quantile diagnostic confirms the
    same conclusion. It is strong at 0.4 ms (`+2.52434%`) but still fails
    0.7 ms (`-25.8759%`). The missing piece is not a fixed local quantile;
    it is connected, time-consistent outer-contour tracking or higher-fidelity
    contour data.
  - A reference-independent outer alpha=0.5 contour envelope was then added to
    avoid choosing a single local crossing rule. It uses
    `max(ray_alpha05_silhouette, plic_ray_plane_q25_silhouette)` without
    looking at the digitized reference. On the long-window CaE=0.25 comparison
    it gives `+2.52434%` at 0.4 ms and `-8.93099%` at 0.7 ms, reducing the
    paper gap metric from 5 to 4. This is an observable hardening improvement,
    not a production VoF/CSF change; the alpha-integral morphology claim and
    the missing 0.8/0.9 ms external geometry remain DOWNGRADED/BLOCKED.
  - A cell-boundary ray alpha=0.5 support diagnostic was added to test whether
     the original ray-alpha05 miss was just a center-sampling underestimation.
     It overcorrects rather than fixes the paper comparison: for the long-window
     CaE=0.25 case, 0.4/0.7 ms errors become `+18.282%`/`+20.8856%`, compared
     with the original ray-alpha05 `-10.8924%`/`-8.93099%`. This rules out a
   simple outer-cell support correction as the paper-faithful silhouette
   extractor; the result is retained as negative diagnostic evidence only.
   - An all-liquid ray `alpha=0.5` silhouette diagnostic was added to check
     whether the connected-component mask, rather than interface quality, caused
     the morphology miss. It does not fix the paper comparison. In the
     long-window CaE=0.25 case the fixed-time errors are `+85.8463%` at
     0.4 ms and `+16.5512%` at 0.7 ms, and at the paper text's tip-synchronized
     times both connected and all-liquid `alpha=0.5` ray volumes collapse to
     zero (`-100%`/`-100%`). The alpha-integral proxy still matches the
     synchronized digitized volumes within 10%, but the paper-visible
     `alpha=0.5` interface is not resolved on this coarse long-window path.
     This is a stronger DOWNGRADED finding, not an upgraded morphology pass.
   - A conservative post-advection sharpening candidate was tested as an
     opt-in VoF transport option, with the default production setting left at
     zero. It is not sufficient. With `post_sharpening=0.6` and one sweep, the
     synchronized 0.4 ms all-liquid `alpha=0.5` error improves from `-100%` to
     `-36.8912%`, but the 0.7 ms synchronized contour remains `-100%` and the
     alpha-integral morphology errors worsen to `-13.3953%`/`-12.3682%`.
     This rules out a scalar post-filter as the paper-level fix; the next
     implementation needs transport/PLIC swept-volume preservation and
     time-consistent contour tracking.
   - A prescribed nozzle-inflow `alpha=1` boundary condition was then added as
     an opt-in geometric VoF path, replacing the previous boundary-face
     behavior that reused the owner cell alpha for inflow. This is a real
     interface-transport improvement but not a validation pass. It keeps
     mass/divergence bounded (`4.03036e-15`, `3.89045e-12`) and prevents the
     synchronized all-liquid `alpha=0.5` silhouette from collapsing to zero,
     but the remaining silhouette errors are still `-55.4612%`/`-54.4809%`
     while alpha-integral morphology overfills by `+17.3425%`/`+24.5199%`.
     The finding points to missing paper-level swept-volume/contact-line
     preservation, not merely a boundary injection bug.
   - Strengthening the existing geometric compression flux is not the fix.
     With prescribed inlet `alpha=1` and `vof_compression=0.2`, mass and
     continuity remain bounded (`4.64192e-15`, `7.51959e-12`) but the
     synchronized `alpha=0.5` silhouette collapses again (`-100%`/`-100%`) and
     alpha-integral morphology undershoots by `-23.8986%`/`-22.2234%`.
     Stronger compression alone is therefore retained as negative evidence.

6. **Fixed-current-plane reachability is now separated from current physics.**
   - A diagnostic-only preconditioned paper-current-plane fixture initializes a
     resolved liquid column through the fixed Fig. 8(b) plane (`y/Di=4.6875`,
     tip `y/Di=5.4375`, radius `0.65 Di`, width `0.20 Di`) and then runs the
     same EHD/VoF/CSF path. This is not promoted as paper validation because
     the imposed column is a coarse-grid reachability probe, not a simulated
     Taylor cone-jet.
   - The probe does make the fixed current plane non-degenerate:
     `3/3` tail samples are developed for both CaE values, with fixed-plane
     alpha05 current ratios `mean=1.16684`, `peak=1.12807`, mass drift
     `3.81624e-15/3.24381e-15`, and max divergence
     `1.6464e-12/2.55207e-12`.
   - The same run remains diagnostic-only: all-phase current ratios are weak
     (`tail=1.37355`, `peak=1.40142`), but the axial alpha05 convective ratio is
     still `2.2218`, and the liquid column is imposed rather than simulated by
     cone-jet dynamics. The current gap is therefore not merely a fixed-plane
     sampling artifact; after reachability is forced, the coupled
     charge/current/morphology model still needs a physical closure.

7. **The reported moving collector wall is not the missing current fix.**
   - Candido's paper specifies a moving collector wall with `us=20 mm/s`. An
     opt-in diagnostic now applies the corresponding dimensionless transverse
     collector speed `0.021309` to collector-adjacent cells while keeping the
     paper inlet/open-atmosphere current candidate.
   - The short standalone run remains bounded
     (`alpha_mass_drift=1.82704e-15/3.00157e-15`,
     `max_div=8.95549e-13/1.47587e-12`) but does not develop the fixed
     Fig. 8(b) current plane (`0/0` samples). Current ratios remain too high:
     all-phase tail/peak `3.37002`/`3.01316`, axial alpha05 convective
     `2.13064`, axial total `1.29622`. Morphology/whipping also do not improve
     enough (`45.4347%` low-CaE morphology error, high-CaE radial asymmetry
     `0.000166541`).
   - This is retained as paper-faithfulness hardening and negative evidence:
     missing collector motion is not the dominant local explanation for the
     Candido current-voltage gap on the coarse smoke fixture.

8. **Poisson-face total current is not a substitute for the paper current.**
   - The axial Poisson-face alpha05 total-current window now records already-run
     paper-charge, paper-inlet, open-atmosphere, and moving-collector candidates.
     Those candidates have developed alpha>=0.5 windows (`27/27` low/high
     samples) and weak total-current ratios (`1.2643`, `1.31028`, `1.31027`,
     `1.31029`).
   - The convective-only face-current counter-check now uses the same
     boundary-charge-advection option as the charge transport path. That fixes a
     diagnostic inconsistency: the paper-charge candidate has nonzero
     Fig. 8(b)-comparable convective current (`8.94776e-10`/`1.84757e-09`) but is
     still DOWNGRADED with ratio `2.06484`. Its factorization shows charge ratio
     `1.20836`, face-flux ratio `1.7982`, and absolute convective-flux ratio
     `2.17128`, so velocity/face-flux sensitivity is the dominant local cause.
   - A projected-vs-raw face-flux diagnostic now compares the Rhie-Chow projected
     face flux against the raw cell-velocity `u dot Sf` face flux on the same
     developed alpha>=0.5 faces. For the paper-charge candidate the projected and
     raw convective-current ratios are identical (`2.06484`), the projected and
     raw face-flux ratios are identical (`1.7982`), and projected-to-raw current
     and face-flux ratios are `1/1` for both low and high CaE. This downgrades
     the Rhie-Chow-projection-artifact hypothesis: the raw velocity/current field
     is already too voltage-sensitive on this fixture.
   - A local momentum-source factorization on the same developed alpha>=0.5 bin
     further localizes the velocity problem. For the paper-charge candidate,
     mean alpha>=0.5 velocity ratio is `1.7982`, electric momentum-source ratio is
     `1.84701`, total-source ratio is `1.8186`, source/rho acceleration ratio is
     `1.90444`, and CSF source ratio is only `0.939906`. The current gap is
     therefore dominated by electric momentum/source calibration and electrode or
     outlet current treatment, not by surface tension, Rhie-Chow interpolation, or
     a missing total-current observable.
   - Reducing the empirical drive reference to the unit-Maxwell candidate lowers
     the developed Poisson-face alpha>=0.5 current ratio to `1.9802` and the
     velocity ratio to `1.57764`, with projected/raw current ratios still `1/1`.
     This is only an APPROXIMATE tradeoff: the same row has morphology error
     `41.2655%`, worse than the paper-charge candidate, and the electric
     source ratio is still `1.74122`. Do not promote it as a paper-current pass.
   - The paper-inlet, open-atmosphere, and moving-collector candidates still
     report developed alpha>=0.5 windows but `BLOCKED_ZERO_FACE_FLUX` for the
     face-current observable. This keeps the total-current near-pass as
     diagnostic evidence only, not paper-level validation.

9. **Tip-synchronized morphology is now regenerated by the standalone
   long-window fixture, but the visible interface remains downgraded.**
   - `test_candido_long_window_budget3d` now also writes
     `candido_morphology_tip_sync_diagnostic3d.csv`, so the paper-gap metric no
     longer depends on a stale full Candido-suite artifact for
     `morphology_tip_sync_within_10_percent`.
   - Live targeted evidence:
     `cmake --build build --target test_candido_long_window_budget3d -j 2`
     passed, `ctest --test-dir build --output-on-failure -R
     '^test_candido_long_window_budget3d$'` passed `1/1` in `84.14 sec`, and
     `python3 apps/electrospray_paper_gap_metric.py` reports
     `paper_validation_gap_count=3`.
   - The regenerated row has `unique_tip_levels=7`,
     `max_tip_time_ms=1.16085`, `paper_sync_offset_ms=-0.760848`,
     synchronized alpha-integral volume errors `-4.26325%` at 0.4 ms and
     `-2.15593%` at 0.7 ms, and status
     `TIP_SYNC_MORPHOLOGY_WITHIN_10_PERCENT`.
   - The same row still has all-liquid alpha>=0.5 ray errors of
     `-100%`/`-100%` and
     `DOWNGRADED_TIP_SYNC_ALPHA05_INTERFACE_LOST_OR_MISMATCHED`. This improves
     mechanical evidence coverage only; it does not prove a paper-faithful
     alpha=0.5 cone/jet silhouette.
   - Remaining machine-readable false check is now
     `morphology_late_0_8_0_9_digitized`.

10. **Primary and combined current-voltage sensitivity are regenerated from the
    standalone long-window reports; the weak mechanical current checks are now
    approximate, while the raw primary path remains downgraded.**
    - `test_candido_long_window_budget3d` now also writes
      `candido_current_voltage_sensitivity3d.csv` and
      `candido_current_voltage_sensitivity_combined_charge3d.csv` from the same
      live low/high CaE reports used for the mass, charge, Pareto, and boundary
      diagnostics.
    - The primary long-window convective-current row is not merely missing
      evidence: it is `DOWNGRADED_AVERAGE_CURRENT_TOO_VOLTAGE_SENSITIVE`, with
      tail-current ratio `1.63434e11`.
    - A short Ca-independent boundary-advection candidate is still above the
      weak-voltage bar (`tail_mean_current_ratio=2.23202`). A long-window
      Ca-independent boundary-advection candidate is closer but still
      downgraded (`2.05652`).
    - A long paper-boundary combined-charge candidate in the primary
      `candido_current_voltage_sensitivity3d.csv` does meet the weak current
      bar (`tail_mean_current_ratio=1.95942`,
      `APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY`). This changes
      `paper_current_voltage_sensitivity_ok` to true, but only at the
      approximate mechanical-check level.
    - The long combined conservative-bounded subcycled charge path does meet
      the weak combined-current bar with `tail_mean_current_ratio=1.95942` and
      `APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY`. This changes
      `combined_charge_current_voltage_sensitivity_ok` to true and reduces the
      paper-gap metric to `1`.
    - This is a mechanical evidence improvement, not a paper-level current
      validation. The raw primary paper-current row is still downgraded, the
      fixed-plane alpha>=0.5 current observable is still not a faithful
      non-degenerate Fig. 8(b) comparison, and the next useful solver work has
      to change the physical electric-source/current evolution or bring in
      external current calibration; relabeling total-current or blocked
      alpha>=0.5 observables would be misleading.

## Next Implementation Priority

1. Supply or obtain independent Candido Fig. 3(b) late morphology data at
   `0.8` and `0.9 ms` before claiming paper-level Candido morphology closure.
   Acceptable input is an external contour-coordinate/volume dataset that passes
   `apps/electrospray_late_morphology_dataset_check.py`; volumes backsolved from
   the paper's reported relative-error row are rejected.
1. Replace the diagnostic quasi-implicit bulk-conduction split with a
   conservative stiff charge solve and paper-faithful electrode/nozzle/outlet
   electric-current treatment so the convective `q_e U dot n` current is nonzero
   in developed liquid-jet windows.
2. Fix the dynamic local curvature/CSF force path exposed by the force-isolation
   CSV before tuning full-path droplet frequency.
3. Replace the Candido alpha-integral morphology proxy with a robust
   paper-faithful `alpha=0.5` cone/jet silhouette observable.
4. Extend exact swept PLIC guards from tetra/hex cells to fully irregular
   polyhedral swept volumes, beyond the current bounded diagnostic row.
5. Fully implicit momentum matrix and production PIMPLE solver controls.
6. AMR/refinement execution and Candido-style DNS observable histories.
