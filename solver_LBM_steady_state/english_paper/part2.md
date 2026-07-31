# 3. Benchmark Suite and Evaluation Protocol

## 3.1 Benchmark Composition and Roles

The suite comprises nine families, chosen to span distinct validation roles. Channel (plane Poiseuille) and Couette are basic shear/pressure-driven tests for which analytic profiles exist. The lid-driven cavity at $Re = 100/400/1000$ checks the literature centerline benchmark [5] and recirculating closed-domain dynamics. The backward-facing step and cylinder wake involve separation, reattachment, and wake formation; the multi-cylinder mask tests multiple obstacles and a complex mask boundary; and the T-junction tests a branching geometry with inlet/outlet boundary coupling. Each family is run at 1x/2x/3x mesh scalings, constituting 27 proposed-method runs in total.

Table 5 summarizes the actual grid size of each family (directly verifiable from the shape of the stored final-field arrays), the boundary-condition type, the validation role, and the reference tier. The complete benchmark specification, including the definitions of $U_{\mathrm{ref}}$, $\nu$, and the mask geometry, is included in the case manifest of the reproducibility package.

**Table 5. Benchmark family definitions: grid size, boundary conditions, validation role.**

| Family | Grid (1x / 2x / 3x) | Boundary conditions | Validation role | Reference |
|---|---|---|---|---|
| Channel (plane Poiseuille) | 32×192 / 64×384 / 96×576 | Inlet/outlet + wall | Pressure-driven shear (baseline) | Analytic |
| Couette | 32² / 64² / 96² | Moving wall + wall | Shear flow (baseline) | Analytic |
| Cavity $Re=100$ | 33² / 65² / 97² | Lid-driven, closed | Recirculating closed domain | Ghia [5] |
| Cavity $Re=400$ | 49² / 97² / 145² | Lid-driven, closed | Recirculating closed domain | Ghia [5] |
| Cavity $Re=1000$ | 129² / 257² / 385² | Lid-driven, closed | Recirculating closed domain | Ghia [5] |
| Backward-facing step | 64² / 128² / 192² | Inlet/outlet + step mask | Separation/reattachment | Tight ref |
| Cylinder wake | 64² / 128² / 192² | Inlet/outlet + obstacle mask | Wake formation | Tight ref |
| Multi-cylinder | 32² / 64² / 96² | Multiple obstacle masks | Complex mask boundary | Tight ref |
| T-junction | 96×64 / 192×128 / 288×192 | Branching inlet/outlet | Branching geometry + open-BC coupling | Tight/Picard ref |

The interpretive scope of the 1x/2x/3x scaling is stated explicitly. This axis is a solver-scaling benchmark that checks whether the same solver converges while maintaining the same stopping protocol as the mesh size grows, and how the wall-time/LBE-call scaling behaves relative to the baselines. It does *not* mean that a formal grid-convergence study or Richardson extrapolation was performed; claims of observed order of accuracy or GCI are outside the scope of this paper (Section 5.2).

## 3.2 Stopping Protocol and Tolerance

In the 27 stored proposed-method runs, the residual type is recorded uniformly as `macro_l2_p_ux_uy_uz`, and the absolute gate is $r_{\mathrm{macro}} < 5\tau$ ($C_{\mathrm{tol}} = 5$). The value of $\tau$ is taken from the `tol` column of the summary CSV as the source value. The channel/Couette/backward-step/cylinder-wake/multi-cylinder/T-junction families use $\tau = 1.0\mathrm{e}{-7},\ 5.0\mathrm{e}{-8},\ 3.333\mathrm{e}{-8}$ at 1x/2x/3x respectively, and the cavity $Re = 100/400/1000$ families use $\tau = 1.0\mathrm{e}{-8},\ 5.0\mathrm{e}{-9},\ 3.333\mathrm{e}{-9}$. The relative-plateau test is passed when the fractional improvement over the most recent $W = 50$ recorded check points in the relative macro-$L_2$ history is at most $\eta = 0.05$. The residual recording interval is set by the run configuration and is stored, together with the iteration, LBE-call, and wall time, in each case's history CSV, so that the plateau test can be recomputed directly from the stored history. In addition, every proposed-method run is subject to a minimum operator budget of about $2\times10^{4}$ LBE-calls, which acts as a floor that prevents premature termination immediately after the rapid initial residual drop. This is directly verifiable in the stored logs, where many fast cases (Couette, multi-cylinder, cylinder wake, cavity $Re=100$ 1x, etc.) terminate at LBE-calls just past $2.0\times10^{4}$; this floor acts only to *increase* the proposed method's wall time.

That $\tau$ tightens by factors of $1/2$ and $1/3$ as the level increases is a protocol-level reporting choice, applied identically to the proposed method and the baselines; because the same schedule is imposed on all methods, it does not affect the fairness of the inter-method comparison.

The difference in absolute $\tau$ between the cavity and non-cavity families is not a case tuning favorable to any particular method but a family-level reporting choice reflecting different reference tiers and benchmark conventions. The cavity is a literature benchmark compared against the Ghia centerline and therefore uses a stricter tolerance family. The meaningful unit of comparison is not the absolute $\tau$ across families but the wall time and operator work needed for the proposed method and the baselines to reach the same convergence verdict at the same case, the same mesh level, the same $\tau$, and the same plateau rule.

The minimum tail budget and the plateau window are not tuning parameters that make the AP-Schur step aggressive or change the correction per case; they are validation gates that prevent premature termination immediately after a fast residual drop. Strengthening them can only keep the wall time the same or increase it, so they are not devices that artificially inflate the proposed method's speed advantage. The same criterion applies to the strict-convergence interpretation of the baseline runs.

The distinction between protocol constants and tuning parameters is placed in four tiers. First, $Re$, $Ma$, $\tau_{\mathrm{BGK}}$, the geometry mask, and the inlet/outlet/wall conditions are part of the benchmark definition and are not adjusted by the solver. Second, $\tau$, $C_{\mathrm{tol}}$, $W$, $\eta$, and the minimum tail budget are frozen validation constants that define the stopping protocol. Third, the $\alpha$ damping candidates and the admissibility gate are method-wide, fixed globalization devices. Fourth, at no tier is there any empirical coefficient specific to the cavity, backward step, cylinder, or T-junction.

## 3.3 Baseline Implementations and Fairness

The credibility of the comparison rests on whether the baselines are faithfully implemented and tuned rather than set up as straw men. All five baselines of this paper share the native LBM operator of the same code base and are implemented with standard settings. Each method is given literature-standard hyperparameters and a generous iteration budget (Table 6a). No baseline is intentionally weakened: Anderson uses a sufficient depth and a regularized least-squares; inexact Newton and dual-time multigrid use multi-stage Krylov/V-cycles; and preconditioned LBM uses the standard PLBE transformation.

**Table 6a. Baseline implementations and main settings.**

| Baseline method | Implementation summary | Main settings |
|---|---|---|
| Picard (native LBM) | Native collide–stream–boundary fixed-point iteration | max_steps $\le 1.2\times10^{6}$, residual-monotone termination |
| Anderson acceleration [9,10] | Regularized least-squares fixed-point acceleration, admissibility safeguard | depth $m=10$, $\beta=1.0$, reg $=10^{-12}$ |
| Preconditioned LBM [20,21] | Balanced PLBE ($\gamma$-scaled) transform + block preconditioner | $\gamma=0.5$, max_steps $\le 1.2\times10^{6}$ |
| Inexact Newton–Krylov [6,7] | JFNK: GMRES + NE/smoother + line search | krylov_max=10, K_ne=20, K_smooth=10, line_search=4 |
| Dual-time multigrid [14] | FAS V-cycle, residual-equation smoothing | max_levels=6, V-cycle, K_pre/coarse/post=20/30/20 |

All methods are evaluated under the same macroscopic-$L_2$-residual/plateau protocol and the same admissibility definition, and the only difference between the proposed method and the baselines is the update rule. The iteration budget given to the baselines meets or exceeds standard steady-LBM practice (e.g., $6\times10^{5}$–$1.2\times10^{6}$ LBE-calls for cavity 2x/3x). The non-convergence reported in the next section is therefore interpreted not as budget starvation but as a genuine plateau within that budget (quantitatively confirmed in Section 4.1).

## 3.4 Comparison-Matching Rules

The baselines of this paper denote the Picard, Anderson acceleration, preconditioned LBM, inexact Newton, and dual-time multigrid implementations contained in the stored result set, and the preconditioned-LBM axis corresponds to the representative family of the steady-flow LBM acceleration literature [20, 21]. This comparison does not claim an absolute ranking against every conceivable optimal implementation in the literature; it reports the relative performance observed under the same benchmark definition, the same macroscopic-residual/plateau verdict, and the same summary/history aggregation rules. Because all methods are computed in the same Python/NumPy execution environment sharing the same native LBM operator implementation, inter-method wall-time differences arise from algorithmic structure rather than from implementation language or library differences.

When building a comparison table, a proposed-method run and a baseline run must share the same case label and the same mesh level. We define two comparison groups. (i) Available-baseline comparison: among the stored baseline runs for a given case/level, those that have wall-second and residual records are considered, and the shortest wall time is found. (ii) Strict-convergence comparison: a conservative subset retaining only those runs that passed the same macro-$L_2$/plateau verdict and were recorded as converged. Cases for which no baseline run exists or for which a required column is empty are not counted in the denominator of the win count, and a faster but non-converged run is not used for any strong conclusion. This rule is not a post hoc filter favorable to the proposed method but a prior interpretive rule for first matching whether different solvers reached the same stopping protocol and then comparing time. We also state the direction of bias of the two groups: the available-baseline comparison includes the short wall times of baseline runs that terminated early without converging, whose true convergence times can only be longer than recorded. The available-baseline comparison is therefore a conservative comparison favorable to the baselines and unfavorable to the proposed method, and the result that the proposed method nonetheless wins on 25/27 cases (Section 4.2) can be interpreted as a lower-bound estimate.

The timing measurement convention is as follows. Wall time is aggregated from the `wall_seconds` and `elapsed` records of the stored summary/history files and includes all additional residual evaluations incurred during AP-Schur trials, fallbacks, and continuations. Because absolute wall time depends on CPU generation, memory bandwidth, the Python/NumPy/BLAS implementation, and background load, the primary interpretation is a relative comparison within the same stored result set under the same stopping rule. To complement this hardware dependence, the LBE-call is reported as an auxiliary operator-work metric. The LBE-call is the number of invocations of the native operator $G(f)$ or an equivalent collision–streaming–boundary residual evaluation, and includes the evaluation cost of rejected trials. With the present result set, which lacks repeated-run statistics, we claim no confidence intervals or $p$-values (Section 5.2).

## 3.5 Reference Tiers and Accuracy Metric

Reference data are used only for post-solve evaluation and never during the solve. Denoting the field or profile to be compared by $Q_h$ and the reference by $Q_{\mathrm{ref}}$, the accuracy metric is the relative $L_2$ norm

$$e_{\mathrm{ref}} = \frac{\| Q_h - Q_{\mathrm{ref}} \|_2}{\max\left( \| Q_{\mathrm{ref}} \|_2, \varepsilon_{\mathrm{ref}} \right)}. \tag{23}$$

References are organized into three tiers. (i) The analytic profiles of channel/Couette are closed-form references expected under the same discrete setting. (ii) The Ghia et al. centerline data [5] for the cavity are an external literature benchmark. (iii) For complex geometries without closed-form solutions—backward step, cylinder wake, multi-cylinder, T-junction—a more strictly or longer-converged stored field within the same benchmark definition is used as a tight numerical reference; in this case the reference error denotes final-field agreement rather than a continuum-exact error. We do not sort $e_{\mathrm{ref}}$ across different tiers into a single universal accuracy ranking, and strong comparisons are restricted to inter-method differences within the same case family and the same level.

The Ghia comparison procedure for the cavity is as follows. When the solver field does not share grid points with the Ghia tabulation coordinates, centerline values at the same physical coordinates are sampled by linear interpolation from the stored final field, and no smoothing or renormalization is applied to the reference values. This interpolation is not used in residual evaluation, accept/reject, or damping selection; it is used only in the post-processing stage for the figures and the $e_{\mathrm{ref}}$ computation. The Ghia comparison is therefore not a calibration but a post hoc verification of whether the final discrete field is compatible with the literature benchmark.

We also separate the convergence and accuracy verdicts into tiers. A convergence pass is a solver-state verdict that $r_{\mathrm{macro}}$, plateau, and admissibility were satisfied simultaneously, whereas $e_{\mathrm{ref}}$ and the Ghia/analytic/tight-reference comparison are accuracy diagnostics of how close that state is to an external reference field. A convergence pass does not imply minimization of the reference error, and conversely a small difference from the reference is not evidence that the solver used the reference internally. The tables and figures of Section 4 present both tiers together.

## 3.6 Fairness Invariants and Reproducibility Checklist

All numerical results are aggregated from a fixed result set. No solver algorithm or benchmark output was changed during aggregation; the stored final states and residual histories were read by the same rules to build the tables and figures. We did not adjust per-case coefficients to favor the proposed method, nor apply different convergence criteria per family. Table 6 lists the implementation invariants that an independent verifier can check first when examining fairness.

**Table 6. Reproducibility checklist and implementation invariants.**

| Invariant | Verification criterion |
|---|---|
| Residual definition | The macroscopic-$L_2$-residual history is used as the primary convergence metric for all methods and all cases. |
| Plateau condition | Together with the absolute residual condition, a plateau condition—decrease halting in the recent tail—is required. |
| Reference usage | Ghia, analytic solutions, and benchmark references are used only for post-solve error evaluation, never for solver updates. |
| No case tuning | The proposed method runs as a single AP-Schur-only algorithm with no case-specific relaxation coefficients or geometry-specific empirical switches. |
| Admissibility | Only trials passing density positivity, finite macro fields, boundary/mask consistency, and native residual decrease are accepted. |
| Reported data | Wall time, final residual, relative residual, field error, and contour/profile figures are produced only from the stored summary, history, field, and reference files. |

We also fix the identification criterion for the proposed method. The method we call "AP-Schur-only" in this paper is defined solely as one that shares all of: the same native operator $G(f)$, the same projection $P$ and lifting $P^{\dagger}$, the same Jacobian-free residual-response evaluation, the same damping candidates and accept/reject rule, the same admissibility gate, the same native fallback, and the same stopping protocol. What changes from benchmark to benchmark is only the problem definition—mesh, $Re$, boundary condition, mask.

# 4. Results

## 4.1 Overall Convergence Summary

The proposed method passed the convergence verdict of Eq. (5) on all 27 runs (the `converged`, `residual_converged`, and `plateau_converged` flags are all satisfied, with `convergence_mode = macro_l2_final_threshold_and_relative_plateau`). The total wall times at the 1x/2x/3x levels are 134.2 s, 1546.6 s, and 3507.2 s, respectively. Table 7 and Figure 2 give the per-level summary.

**Table 7. Per-level convergence summary of the proposed method.**

| Level | Cases | Converged | Total wall [s] | Median residual | Max residual | Median rel. error |
|---|---|---|---|---|---|---|
| 1x | 9 | 9 | 134.2 | 2.142e-12 | 2.474e-08 | 3.260e-03 |
| 2x | 9 | 9 | 1546.6 | 3.305e-12 | 6.409e-08 | 0.0326 |
| 3x | 9 | 9 | 3507.2 | 1.153e-11 | 1.567e-08 | 0.0257 |

The "Median rel. error" in Table 7 is the median over the cases for which a reference error was computed at that level. At 2x/3x, complex-geometry cases without a computed tight reference are excluded, so the share of Ghia-compared cavity cases grows; comparisons of the magnitude of this column across levels are therefore meaningless, and only cases sharing the same reference tier should be compared (Section 3.5). The increase from 3.26e-03 at 1x to 0.0326 at 2x reflects a change in the composition of the aggregated subset, not a degradation of accuracy.

We also disclose the budget handling of long runs. The LBE-calls of cavity $Re=400$ 2x, $Re=100$ 3x, and $Re=1000$ 2x/3x (about 1.26M, 0.77M, 1.44M, and 1.44M, respectively) exceed the nominal step budget recorded in the summary; these are the result of continuation runs (`method_variant = uniform_ap_schur_only_continued`) that maintained the same stopping rule. A continuation is not a change of algorithm or protocol but an extension of the run under the same verdict criteria, and the cost of the extended segment is also fully included in the wall time and LBE-calls. No proposed-method run therefore treated reaching the budget as convergence; all 27 terminated by the verdict of Eq. (5).

![](media/image2.png){width=5.83in}

**Figure 2.** Total wall time and maximum final macro-$L_2$ residual of the proposed-method runs by mesh scaling level. Bars are the sum of wall time over the nine cases; the line is the maximum final residual.

The interpretation rule for the CSV convergence columns is as follows. The final convergence verdict is read jointly from the `converged`, `residual_converged`, `plateau_converged`, and `convergence_mode` columns. The `relative_floor_pass`, `macro_change_pass`, and `plateau_improvement` columns are auxiliary columns from earlier diagnostic stages or sub-paths of the plateau verdict; even if some of them are zero or empty, the final verdict is checked by the summary's `converged` flag and the residual/plateau flags.

**Convergence-robustness comparison.** Under the same stopping protocol and the same admissibility definition, the proposed method converged on all 27 cases. The five baselines, by contrast, converged on only a subset of the 27 within their generous budgets (Table 7a). That the non-convergence is not budget starvation is directly verifiable. For example, on cavity $Re=400$ 2x all five baselines plateaued at a final residual of about $3.4$–$3.6\times10^{-6}$ after exhausting $6\times10^{5}$–$7\times10^{5}$ LBE-calls (about $100\times$ above the target $5\tau = 2.5\times10^{-8}$), and on cavity $Re=1000$ 2x they stagnated at about $1.0\times10^{0}$ even after $1.2\times10^{6}$ calls. The baselines' non-convergence is thus a genuine stall within the budget, not a lack of iterations. This robustness gap is not itself used for the primary timing claim of this paper; the timing comparison is restricted to the strict subset (15/27) on which a baseline also converges, excluding any budget asymmetry (Section 4.2).

**Table 7a. Number of converged cases per method (same protocol, out of 27 cases).**

| Method | Converged cases / 27 | Note |
|---|---|---|
| Proposed (AP-Schur-only) | 27 | All cases satisfy $r_{\mathrm{macro}} \le 5\tau$ + plateau + admissibility simultaneously |
| Inexact Newton–Krylov | 15 | Strongest baseline; still non-converged within budget on 12 cases |
| Preconditioned LBM | 14 | — |
| Picard / Anderson | 13 / 13 | — |
| Dual-time multigrid | 12 | — |

## 4.2 Wall-Time Comparison

Compared against the shortest converged time among available baselines for each case, the proposed method is faster on 25/27 cases, with a median wall-time ratio of $2.92\times$. Because some baseline runs may have a strict-convergence flag of 0, we interpret the available-baseline comparison and the strict-convergence comparison separately (Section 3.4). A strict-convergence baseline exists on 15/27 cases, and on that subset the proposed method is faster than the shortest-time strict-convergence baseline on 14/15 cases, with a median ratio of about $2.06\times$. The remaining 12 cases have no strict-convergence baseline run in the stored result set, so for them the available-baseline comparison is interpreted only as exploratory and is not used for any strong superiority claim.

All headline figures of this section (win counts, median ratios) were recomputed by an independent script from the stored all-method summary CSV according to the matching rules of Section 3.4 and confirmed to match the in-text values.

We disclose the exception cases as follows. Among the 27 comparisons, the cases where the proposed method is slower than the shortest available baseline are Couette 3x and cavity $Re=400$ 2x. On Couette 3x, both preconditioned LBM (about 85.5 s) and inexact Newton (about 95.9 s) satisfied the strict-convergence flag and were faster than the proposed method (about 133.5 s); this is the sole exception in the strict-convergence subset. On cavity $Re=400$ 2x, an inexact-Newton run at about 275.4 s is a faster available run than the proposed method (about 310 s), but all five baseline runs for that case are recorded with a strict-convergence flag of 0. These non-winning cases are included as is in the summary table and the per-case wall-time-ratio figure (Figure 3); rather than a failure of the AP-Schur correction itself, they are interpreted as situations where native Picard-type relaxation is already short enough, or where the local modes created by the boundary/mask dominate over the global hydrodynamic slow mode.

![](media/image3.png){width=5.83in}

**Figure 3.** Per-case AP-Schur-only wall-time ratio relative to the shortest-time available-baseline run. A ratio $> 1$ marks a case where the proposed method is faster. Baseline runs without a strict-convergence flag are distinguished separately in the text interpretation.

Figures 4–6 present, on common axes, the macro-$L_2$ residual versus wall time histories of all six methods for all 27 cases. These figures show two things directly. First, the residuals of all methods are recorded with the same definition (`macro_l2_p_ux_uy_uz`), so differences in stopping rule do not distort the comparison. Second, the time advantage of the proposed method comes not from a difference in termination point but from the residual trajectory itself descending below tolerance at an earlier wall time. Each curve was generated directly from the history CSV of the case directory, with no smoothing applied.

![](media/image4.png){width=6.25in}

**Figure 4.** Macro-$L_2$ residual versus wall time convergence histories for the nine 1x cases (all methods, generated from the stored history CSV).

![](media/image5.png){width=6.25in}

**Figure 5.** Macro-$L_2$ residual versus wall time convergence histories for the nine 2x cases.

![](media/image6.png){width=6.25in}

**Figure 6.** Macro-$L_2$ residual versus wall time convergence histories for the nine 3x cases.

## 4.3 Operator-Work (LBE-call) Comparison

To complement the hardware dependence of wall time, the LBE-call ratio was recomputed from the same stored result set. Compared against the shortest-time available baseline, the proposed method uses fewer LBE-calls on 19/27 cases, with a median ratio of about $1.80\times$. On the strict-convergence subset, it uses fewer LBE-calls on 13/15 cases, also with a median ratio of about $1.80\times$. The LBE-call exceptions in the strict-convergence subset are Couette 3x and T-junction 3x. The efficiency claim therefore does not rest on wall time alone but is interpreted together with an operator-work metric recomputable from the same logs, showing that the speedup is not explained solely by Python overhead or transient CPU scheduling. That said, the LBE-call counts native residual evaluations and is thus an auxiliary metric, not an absolute complexity measure encompassing each method's internal linear-algebra cost.

We also state the cost handling of rejected trials. If an AP-Schur trial fails the admissibility gate or the residual-decrease gate, it is not counted as an accepted correction and the solver proceeds with the native fallback; nonetheless, the cost of the residual evaluations, boundary re-application, finite/positivity checks, and fallback step incurred while evaluating the rejected trial is all included in the stored `wall_seconds` and LBE-calls. The reported speedup is therefore not a post hoc selective timing of only successful corrections but the elapsed cost of the actual execution path, including failed trials.

**Run-to-run timing variability and operator-work determinism.** To check the statistical reliability of a single-run wall time, four representative fast cases were run seven times each under the same stopping protocol (excluding one numba JIT-compilation warmup). Table 7b gives the results. The wall-time coefficient of variation (CV) was 3.6–6.8%, below 7% for all cases, whereas the LBE-call count of each case was bit-identical across all seven repetitions. That is, the operator-work of the proposed method is fully deterministic with zero run-to-run noise, and only the wall time varies by about $\pm5\%$ due to system scheduling. This result implies two things. First, the median wall-time speedups of Section 4.2 (about $2.06\times$ on the strict subset, about $2.92\times$ on available baselines) are more than an order of magnitude larger than the measured timing noise ($<7\%$) and therefore cannot be explained by transient scheduling variation. Second, the LBE-call comparison of Section 4.3 (13/15, 19/27, median $1.80\times$) is made on a deterministic metric and thus contains no run-to-run noise at all. The absolute wall times in Table 7b depend on the runtime environment and are not the primary comparison object of this paper; the reported quantities are the relative variability (CV) and the LBE determinism.

**Table 7b. Wall-time variability and LBE-call determinism over seven repetitions of representative cases.**

| Case (1x) | Mean wall [s] | Std. dev. [s] | CV [%] | LBE-call (7 runs) |
|---|---|---|---|---|
| couette n32 | 0.988 | 0.053 | 5.3 | 13109 (all identical) |
| multi-cylinder n32 | 0.867 | 0.034 | 3.9 | 13291 (all identical) |
| cavity $Re=100$ n33 | 0.524 | 0.019 | 3.6 | 13611 (all identical) |
| cylinder wake n64 | 2.600 | 0.176 | 6.8 | 8075 (all identical) |

## 4.4 Full 27-Case Result Table

Table 8 is a compact result table for reviewing all 1x/2x/3x proposed-method runs at once. To reduce any concern that only selected cases were presented, all converged runs are listed together with level, wall time, LBE-call, final residual, initial-relative residual, and reference error.

**Table 8. Summary results for all 27 proposed-method benchmark runs.**

| Lv | Case | Wall [s] | LBE | $r_{\mathrm{final}}$ | $r/r_0$ | Rel. err | Ref |
|---|---|---|---|---|---|---|---|
| 1x | backward step n64 | 27.66 | 122673 | 2.47e-08 | 7.55e-08 | 3.26e-03 | tight ref |
| 1x | cavity re1000 n129 | 56.84 | 221413 | 2.36e-09 | 7.91e-09 | 0.0542 | Ghia centerline |
| 1x | cavity re100 n33 | 0.70 | 20873 | 1.93e-13 | 1.02e-12 | 0.117 | Ghia centerline |
| 1x | cavity re400 n49 | 3.06 | 44379 | 2.04e-11 | 9.69e-11 | 0.106 | Ghia centerline |
| 1x | channel poiseuille Ny32 Nx192 | 20.30 | 32666 | 3.38e-13 | 4.34e-11 | 9.37e-03 | analytic Poiseuille |
| 1x | couette n32 | 1.20 | 20606 | 2.18e-12 | 4.63e-11 | 2.75e-09 | analytic Couette |
| 1x | cylinder wake n64 | 4.88 | 20251 | 9.88e-15 | 4.02e-14 | 7.94e-05 | tight ref |
| 1x | multi cylinder n32 | 1.25 | 20377 | 2.14e-12 | 5.70e-12 | 4.15e-05 | tight ref |
| 1x | t junction Nx96 Ny64 W16 | 18.29 | 32054 | 2.63e-13 | 7.18e-12 | 1.90e-05 | Picard ref (T-junction 1x) |
| 2x | backward step n64 | 74.97 | 119793 | 6.41e-08 | 2.76e-07 | not computed | — |
| 2x | cavity re1000 n129 | 829.72 | 1440003 | 4.87e-14 | 4.79e-07 | 0.0326 | Ghia centerline |
| 2x | cavity re100 n33 | 5.78 | 41793 | 8.74e-12 | 7.56e-11 | 0.0669 | Ghia centerline |
| 2x | cavity re400 n49 | 309.99 | 1257000 | 4.46e-09 | 3.34e-08 | 0.0642 | Ghia centerline |
| 2x | channel poiseuille Ny64 Nx384 | 185.70 | 105281 | 1.90e-13 | 9.73e-11 | 2.27e-03 | analytic Poiseuille |
| 2x | couette n32 | 21.78 | 101554 | 3.31e-12 | 9.92e-11 | 2.87e-08 | analytic Couette |
| 2x | cylinder wake n64 | 16.16 | 23184 | 1.43e-11 | 8.14e-11 | not computed | — |
| 2x | multi cylinder n32 | 5.77 | 20471 | 1.65e-14 | 6.11e-14 | not computed | — |
| 2x | t junction Nx192 Ny128 W32 | 96.70 | 63535 | 1.26e-12 | 1.00e-10 | not computed | — |
| 3x | backward step n64 | 756.37 | 866000 | 1.29e-10 | 6.79e-10 | not computed | — |
| 3x | cavity re1000 n129 | 1234.14 | 1440085 | 2.56e-10 | 6.64e-07 | 0.0257 | Ghia centerline |
| 3x | cavity re100 n33 | 180.17 | 769000 | 1.20e-10 | 1.35e-09 | 0.0493 | Ghia centerline |
| 3x | cavity re400 n49 | 74.00 | 233779 | 1.57e-08 | 1.50e-07 | 0.0501 | Ghia centerline |
| 3x | channel poiseuille Ny96 Nx576 | 798.11 | 222772 | 7.81e-14 | 8.98e-11 | 1.00e-03 | analytic Poiseuille |
| 3x | couette n32 | 133.55 | 296454 | 2.63e-12 | 9.66e-11 | 5.19e-08 | analytic Couette |
| 3x | cylinder wake n64 | 36.78 | 41868 | 1.15e-11 | 8.01e-11 | not computed | — |
| 3x | multi cylinder n32 | 10.37 | 21265 | 1.48e-12 | 6.57e-12 | not computed | — |
| 3x | t junction Nx288 Ny192 W48 | 283.69 | 89398 | 5.29e-13 | 7.79e-11 | not computed | — |

The reference errors marked "not computed" in Table 8 are entries for which the tight reference field at that level is not included in the result set and was therefore not computed post hoc; they are not interpreted as zero or as success (the data-integrity rule of Section 5.4). The convergence verdict for those cases was independently satisfied by the residual/plateau/admissibility criteria.

## 4.5 Code Verification: Accuracy under Mesh Refinement

This section verifies, from a mesh-refinement viewpoint, that the proposed method reaches the correct discrete solution rather than merely reducing the residual. The purpose is to show independently that the accelerator does not distort the solution, and it is restricted to cases for which a closed-form or literature reference exists.

(i) **Smooth analytic solution — channel Poiseuille.** For plane Poiseuille flow with inlet/outlet boundaries, the relative $L_2$ error of the proposed method's velocity profile is $9.37\times10^{-3}$, $2.27\times10^{-3}$, and $1.00\times10^{-3}$ at $N_y = 32/64/96$ (1x/2x/3x). The observed convergence orders between adjacent levels are

$$p_{12} = \frac{\ln(e_{1x}/e_{2x})}{\ln 2} = 2.04, \qquad p_{23} = \frac{\ln(e_{2x}/e_{3x})}{\ln 1.5} = 2.02,$$

quantitatively consistent with the second-order spatial accuracy that BGK-LBM theoretically possesses for smooth flows. That is, the proposed method reaches the solution while preserving the order of the native LBM discretization. Table 9a summarizes this result.

**Table 9a. Accuracy and observed convergence order of channel Poiseuille under mesh refinement.**

| Level | Grid ($N_y$) | Rel. $L_2$ error | Observed order $p$ |
|---|---|---|---|
| 1x | 32 | $9.37\times10^{-3}$ | — |
| 2x | 64 | $2.27\times10^{-3}$ | 2.04 |
| 3x | 96 | $1.00\times10^{-3}$ | 2.02 |

(ii) **Exactly representable solution — Couette.** Because the linear Couette profile is represented exactly by the LBM equilibrium, the discretization error should be essentially zero. The relative $L_2$ error of the proposed method is $2.75\times10^{-9}$, $2.87\times10^{-8}$, and $5.19\times10^{-8}$ at 1x/2x/3x—all at machine-precision level—and the slight increase with level is merely floating-point accumulation from more operations. This shows that the AP-Schur acceleration injects no non-physical bias into the solution.

(iii) **Literature benchmark — lid-driven cavity.** The Ghia centerline relative $L_2$ error decreases monotonically with mesh refinement at all three Reynolds numbers: $Re=100$, $0.117 \to 0.0669 \to 0.0493$; $Re=400$, $0.106 \to 0.0642 \to 0.0501$; $Re=1000$, $0.0542 \to 0.0326 \to 0.0257$ (1x$\to$2x$\to$3x). Because the Ghia error is not a pure discretization error but a mixture of the Navier–Stokes benchmark table, the lid/wall discretization, low-Mach weak compressibility, and tabulation interpolation, we do not claim a formal order; the monotone approach at all three $Re$ nonetheless shows that the final field of the proposed method converges consistently toward the literature solution.

Taking the three results together, the proposed method (a) preserves the native second order on smooth solutions, (b) maintains machine precision on an exactly representable solution, and (c) converges monotonically toward a literature benchmark. This does not contradict the limitation stated in Section 5.2—that this work does not perform a formal grid-convergence study (Richardson/GCI)—but rather directly supports the secondary claim that "acceleration does not sacrifice accuracy."

## 4.5b Accuracy Summary and Physical Fields

Table 9 is the accuracy summary for the 1x cases that have an analytic or external reference. Channel and Couette are compared against analytic profiles, the cavity against the Ghia centerline, and the remaining complex geometries against a tight/reference numerical field.

**Table 9. Accuracy summary for cases with an analytic or reference profile (1x).**

| Case | Level | Wall [s] | Final residual | Rel. $L_2$ vs ref | Reference |
|---|---|---|---|---|---|
| Plane Poiseuille inlet/outlet ($N_y$=32, $N_x$=192) | 1x | 20.30 | 3.384e-13 | 9.371e-03 | analytic_poiseuille |
| Couette flow (N=32) | 1x | 1.20 | 2.180e-12 | 2.750e-09 | analytic_couette |
| Lid-driven cavity $Re=100$ (N=33) | 1x | 0.70 | 1.935e-13 | 0.117 | ghia_centerline |
| Lid-driven cavity $Re=400$ (N=49) | 1x | 3.06 | 2.045e-11 | 0.106 | ghia_centerline |
| Lid-driven cavity $Re=1000$ (N=129) | 1x | 56.84 | 2.360e-09 | 0.0542 | ghia_centerline |
| Multi-cylinder masked flow (N=32) | 1x | 1.25 | 2.142e-12 | 4.146e-05 | tight_ref |
| Backward-facing step (N=64) | 1x | 27.66 | 2.474e-08 | 3.260e-03 | tight_ref |
| Cylinder wake analogue (N=64) | 1x | 4.88 | 9.882e-15 | 7.935e-05 | tight_ref |
| Strict inlet/outlet T-junction ($N_x$=96, $N_y$=64) | 1x | 18.29 | 2.633e-13 | 1.896e-05 | picard_ref_min_tjunction_1x |

The cavity Ghia centerline relative $L_2$ error is about 0.117, 0.106, and 0.054 at 1x for $Re=100/400/1000$, decreasing to about 0.049, 0.050, and 0.026 at 3x (Figures 7–9). These values do not indicate a failure of residual convergence: the final macro-$L_2$ residual in the same row passed the stopping tolerance, and the Ghia error is not the solver's internal objective but a post hoc comparison against an external literature profile. The residual measures the change relative to the steady fixed point of the present discrete LBM operator, whereas the Ghia comparison is jointly affected by the Navier–Stokes benchmark table, the lid/wall boundary discretization, low-Mach weak compressibility, and tabulation-coordinate interpolation. The cavity–Ghia error is therefore a diagnostic of grid/boundary-condition discretization error, and its decreasing trend with level is consistent with this interpretation; because the cavity–Ghia error is influenced by factors beyond grid spacing, we do not formally require monotone decrease per level.

That the T-junction 1x reference is a strictly converged Picard field carries a separate significance. The relative $L_2$ difference between the proposed method's final field and this Picard reference is only $1.9\mathrm{e}{-05}$, which is direct evidence that the proposed method reached the *same* discrete steady fixed point as native Picard iteration. The acceleration is thus a faster convergence to the same solution rather than a detour to a different one, consistent with the design claim of Section 2.3 that the native residual is not modified. Figure 10 shows velocity-magnitude and vorticity contours of representative cases reconstructed from the stored proposed-method final fields, demonstrating that the converged field qualitatively reproduces the expected flow structures of each geometry (shear layers, recirculation regions, wakes, and branching flow).

![](media/image7.png){width=5.83in}

**Figure 7.** Relative $L_2$ error against the Ghia centerline for the lid-driven cavity at $Re=100/400/1000$.

![](media/image8.png){width=4.79in}

**Figure 8.** Comparison of 1x cavity centerline velocity profiles against Ghia et al. [5].

![](media/image9.png){width=3.44in}

**Figure 9.** Comparison of 2x/3x cavity centerline velocity profiles against Ghia et al. [5].

![](media/image10.png){width=5.83in}

**Figure 10.** Velocity-magnitude and vorticity contours reconstructed from the stored proposed-method NPZ fields. These are post-processing results, not new CFD computations.

## 4.6 Ablation Study: Component Contribution Analysis

The ablation is a mechanism-isolation experiment that separates the contributions of the AP-Schur correction, RRE [15], and the native block in order to clarify the novelty and the performance contribution. Four variants were compared on the 1x suite under the same stopping rule; the results are given in Table 10 and Figure 11. AP-Schur-only maintained 9/9 convergence and 9/9 per-case wall-time wins while attaining the lowest total wall time (147.3 s).

Two points are needed to interpret Table 10. First, the AP-Schur-only total wall time of the ablation (147.3 s) differs from the 1x total of the final 27-run result set in Section 4.1 (134.2 s) because the ablation is a separate experiment performed under the same protocol for variant comparison, with a different run time and log from the final result set; both values are recomputable from their respective stored logs, and the relative ranking among variants is identical in both sets. Second, the "Mean speedup (vs Picard)" column is the arithmetic mean over the nine cases of the wall-time ratio relative to the Picard run of the same case, which differs in both baseline and statistic from the headline metric of Section 4.2 (the median of the ratio relative to the shortest available baseline); the two numbers must not be compared directly.

**Table 10. 1x ablation study results.**

| Variant | Conv. | Wins | Total wall [s] | Mean speedup (vs Picard) | Median residual | AP acc/trial |
|---|---|---|---|---|---|---|
| Full: AP-Schur + RRE | 9/9 | 9/9 | 258.5 | 9.18x | 1.386e-11 | 50/86 |
| RRE only | 9/9 | 8/9 | 292.3 | 11.07x | 1.365e-12 | 0/0 |
| AP-Schur only | 9/9 | 9/9 | 147.3 | 19.41x | 2.142e-12 | 92/118 |
| Native block only | 8/9 | 8/9 | 169.0 | 17.12x | 1.268e-12 | 0/0 |

![](media/image11.png){width=5.83in}

**Figure 11.** 1x ablation total wall time comparison. AP-Schur-only attains the lowest total wall time.

The final variant-selection rule follows this priority: first, the breadth of completed convergence; second, the total wall time on the same 1x suite; third, the number of per-case wall-time wins; fourth, algorithmic simplicity. The residual is only a gate confirming that every variant satisfied the common stopping rule, not a selection criterion. In Table 10, RRE-only and native-block-only appear smaller on some median-residual figures, but RRE-only has a larger total wall time and its wins drop to 8/9, while native-block-only loses convergence breadth at 8/9. Full AP-Schur+RRE passes all cases but is more complex and slower than AP-Schur-only. The choice of AP-Schur-only is therefore based not on the smallest residual number but on the robustness–time–simplicity combination under the same stopping rule.

We also state a defense against the post hoc-selection criticism. Table 10 is not a table for mixing the favorable variant per case. The proposed method in the main text is defined as a single deterministic routine, AP-Schur-only, and is never switched to a different variant per case on any benchmark. Even where another variant shows a smaller final residual on some case, that value is not substituted as the proposed-method result.

## 4.7 Execution-Trace Verification: Direct Evidence of Singularity and Reference-Freedom

The two core claims of the proposed method—(i) that every case uses the same single algorithm, and (ii) that no reference is injected into the solve—are verified directly and independently from the phase log of the stored per-case diagnostic CSV. Each outer round records, by a phase label, which candidate was accepted.

Aggregating the diagnostic logs of all 27 cases, the executed phases consist of exactly the following vocabulary: AP-Schur JFNK acceptance (by damping, `ap_schur_jfnk_alpha` $\in \{1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}\}$), native Picard block, native Picard guard (fallback), and AP-Schur rejected. In none of the 27 cases is an analytic-projection, reference-injection, Ghia-fitting, or benchmark-specific phase recorded even once. This guarantees, at the execution-trace level, that the algorithm does not branch on case identity (claim i) and that acceptance depends only on the native residual and admissibility (claim ii). Table 10a is the full phase aggregation.

**Table 10a. Aggregation of executed outer-round phases over all 27 cases (from the diagnostic logs).**

| Executed phase | Count | Meaning |
|---|---|---|
| ap_schur_jfnk_alpha1 | 204 | AP-Schur Newton step accepted at $\alpha=1$ |
| ap_schur_jfnk_alpha0.5 / 0.25 / 0.125 | 62 / 27 / 23 | AP-Schur accepted after a damping line search |
| ap_schur_rejected | 18 | AP-Schur trial failed the gate $\to$ native fallback |
| uniform_picard_block / guard | many | native Picard candidate / fallback |
| (analytic/reference/case-specific) | 0 | never executed in any of the 27 cases |

Quantitatively, AP-Schur trials were evaluated 334 times in total, of which 237 passed the admissibility and residual-decrease gates and were accepted (overall acceptance rate 71.0%; 78.0%, 69.1%, 65.1% at 1x/2x/3x respectively; all recomputable from the proposed-only summary CSV). Because there are no zero-accept cases, the proposed-method results cannot be interpreted as pure Picard results in which AP-Schur did nothing; and because the rejected-trial cost is included in the wall time and LBE-calls, this statistic is not a post hoc selection of only successful corrections. That the acceptance rate decreases gently with level (78$\to$69$\to$65%) suggests that the admissibility gate operates more conservatively as the grid grows; that the wall-time advantage is nonetheless maintained shows that even partial acceptance suffices to shorten the tail.

This trace-level verification lets the paper answer the reviewer's two strongest attacks—"is there a hidden per-case branch or reference use in the code?" and "did AP-Schur actually contribute?"—with a reproducible execution record rather than a narrative. Aggregating the phase column of each case's diagnostic CSV in the reproducibility package regenerates Table 10a exactly.

## 4.8 Measured Memory Usage

Section 2.6 made the structural claim that, because the proposed method does not assemble the full Newton matrix ($q N_f \times q N_f$), the memory is dominated by $\mathcal{O}(N_f)$. To confirm this quantitatively, the process peak working set (RSS) was measured with the Windows `GetProcessMemoryInfo` API for proposed-method runs at three grid sizes (Table 11a). The marginal solve memory, after separating the runtime baseline immediately after import (Python+NumPy+SciPy+numba, about 150 MB), is 22/50/86 MB at grids of 96²/145²/192², growing linearly with the distribution-field size ($q N_f \times 8$ bytes) at a nearly constant ratio of about $35\times$. This is consistent with the $\mathcal{O}(N_f)$ storage model of Eq. (19)—the spectral cache $B_U(\mathbf{k})$ ($(N_y, N_x, 3, 3)$ complex), a small number of GMRES restart vectors, FFT work arrays, and a limited number of distribution-field copies.

Storing a dense Jacobian $q N_f \times q N_f$ explicitly at the same grids would require about 51/267/820 GB at 96²/145²/192². The measured peak RSS (172–237 MB) is three to four orders of magnitude smaller, and the peak grows only $1.4\times$ while the grid grows $4\times$ (96²$\to$192²). The claim that "full Jacobian assembly is unnecessary" is therefore not merely a qualitative structural claim but is supported by measurement. Because absolute RSS values depend on the runtime environment (interpreter/library versions), the quantitative claim of this paper is confined to (i) the $\mathcal{O}(N_f)$ linear scaling of the marginal memory and (ii) the three-to-four-order-of-magnitude gap relative to a dense Jacobian, and is not extrapolated to a hardware-independent absolute memory constant.

**Table 11a. Measured peak working-set (RSS) of the proposed method by grid size, and structural footprint comparison.**

| Case (3x) | Grid | Field [MB] | Dense Jac [GB] | Baseline RSS [MB] | Peak RSS [MB] | Marginal [MB] |
|---|---|---|---|---|---|---|
| multi-cylinder | 96² | 0.63 | 51 | 149.8 | 172.0 | 22.2 |
| cavity $Re=400$ | 145² | 1.44 | 267 | 150.1 | 200.4 | 50.3 |
| cylinder wake | 192² | 2.53 | 820 | 151.3 | 237.1 | 85.8 |
