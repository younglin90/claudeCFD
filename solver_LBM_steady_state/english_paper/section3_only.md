# **3. Results**

Section 3.1 summarizes the benchmark suite and evaluation protocol; Section 3.2 reports the convergence behavior, covering both the convergence histories and the convergence-rate analysis and robustness; Section 3.3 quantifies the performance of the proposed method relative to the baseline methods through wall-time and operator-work comparisons; Section 3.4 verifies solution accuracy against the available references.

## **3.1 Benchmark suite and evaluation protocol**

The method was evaluated on nine flow configurations, each with an analytic or high-fidelity reference solution. These configurations cover canonical shear flows, separated and reattaching flows, and complex geometries described by a mask, an indicator field marking each lattice node as fluid or solid, together with branching geometries. The suite comprises plane Poiseuille and Couette flow with analytic references, the lid-driven cavity at $Re = 100/400/1000$ against the Ghia centerline benchmark \[5\], and the backward-facing step and cylinder wake, whose physical role is to exhibit separation, reattachment, and wake formation. It also includes the multi-cylinder configuration with its complex mask boundary and the T-junction, a branching geometry with coupled inlet/outlet boundaries. For the backward-facing step and cylinder wake, the accuracy assessment was limited to the relative $L_{2}$ difference of the converged field against a high-fidelity numerical reference and to qualitative reproduction of the expected flow structures; no separation, reattachment, or wake metric was measured. Each test case was solved at three mesh levels, denoted 1x/2x/3x, yielding 27 runs in total. Table 1 lists the grid sizes, boundary conditions, physical role, and reference tier. For cases lacking an analytic or literature solution, the comparison used a reference solution, defined here as a high-fidelity numerical solution obtained by running the native LBE iteration to a residual far below the comparison tolerance on the same discretization. This same construction applies to the backward-facing step, cylinder wake, multi-cylinder, and T-junction cases, and is labeled uniformly in Table 1.

**Table 1. Benchmark case definitions: grid size, boundary conditions, physical role, and reference tier.**

  ----------------------------------------------------------------------------------------------------------------------------------------------
  Family                       Grid (1x / 2x / 3x)         Boundary conditions            Physical role                           Reference
  ---------------------------- --------------------------- ------------------------------ --------------------------------------- --------------
  Channel (plane Poiseuille)   32×192 / 64×384 / 96×576    Inlet/outlet + wall            Pressure-driven shear (baseline)        Analytic

  Couette                      32² / 64² / 96²             Moving wall + wall             Shear flow (baseline)                   Analytic

  Cavity $Re = 100$            33² / 65² / 97²             Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Cavity $Re = 400$            49² / 97² / 145²            Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Cavity $Re = 1000$           129² / 257² / 385²          Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Backward-facing step         64² / 128² / 192²           Inlet/outlet + step mask       Separation/reattachment                 Reference solution

  Cylinder wake                64² / 128² / 192²           Inlet/outlet + obstacle mask   Wake formation                          Reference solution

  Multi-cylinder               32² / 64² / 96²             Multiple obstacle masks        Complex mask boundary                   Reference solution

  T-junction                   96×64 / 192×128 / 288×192   Branching inlet/outlet         Branching geometry + open-BC coupling   Reference solution
  ----------------------------------------------------------------------------------------------------------------------------------------------

Convergence was judged on the macroscopic $L_{2}$ change of the pressure and velocity fields, not the microscopic $f$-RMS, the root-mean-square change of the particle distribution function $f$. The macroscopic pressure and velocity residual is the quantity of physical interest for a steady flow and reflects convergence of the hydrodynamic fields, whereas the $f$-RMS can keep changing through non-hydrodynamic kinetic modes that do not affect the converged macroscopic solution. The macroscopic residual $r_{macro}$ is formed over fluid nodes only from the pressure and velocity increments. In constructing it, the pressure increment is first made gauge-invariant by removing its fluid-domain mean, since the absolute pressure level is arbitrary in a weakly compressible closed or periodic flow, and it is then combined with the velocity increment. Convergence requires three conditions simultaneously: an absolute residual $r_{macro} \leq 5\tau$; a plateau condition, namely a fractional improvement over the last $W = 50$ checks of at most $\eta = 0.05$; and physical admissibility, comprising density positivity, finite fields, and boundary/mask consistency. The base tolerance is $\tau = 10^{-7}$ for the non-cavity cases and $10^{-8}$ for the cavity cases at 1x, multiplied by $1/2$ at 2x and by $1/3$ of the 1x value at 3x. All constants and the admissibility rule were applied identically to every method, with no per-case tuning.

The proposed method, the moment-Schur accelerated LBM (MSA-LBM), was compared against five baselines, that is, established solver methods against which the proposed method was measured, all sharing the native LBM operator of the same code base, as listed in Table 2. The five baselines are the native LBE collide--stream--boundary fixed-point iteration; Anderson acceleration, a regularized least-squares fixed-point accelerator; preconditioned LBM, the standard balanced PLBE (preconditioned lattice Boltzmann equation) transform with a block preconditioner; inexact Newton--Krylov, a JFNK method using GMRES (generalized minimal residual, a Krylov-subspace linear solver) with two smoothing stages and a line search; and dual-time multigrid, an FAS V-cycle scheme. The remaining abbreviations are expanded in Table 2. Each baseline used literature-standard hyperparameters and a generous iteration budget of about $6 \times 10^{5}$ to $1.2 \times 10^{6}$ LBE-calls for cavity 2x/3x, that is, evaluations of the native collide--stream--boundary operator, so the comparison did not disadvantage any baseline. The only difference from the proposed method is the update rule; all methods call the same native operator, so any wall-time difference reflects algorithmic structure.

**Table 2. Baseline implementations and main settings.**

| Baseline method | Implementation summary | Main settings |
|---|---|---|
| native LBE | Native collide--stream--boundary fixed-point iteration | max_steps $\leq 1.2 \times 10^{6}$, residual-monotone termination |
| Anderson acceleration \[9,10\] | Regularized least-squares fixed-point acceleration, admissibility safeguard | depth $m = 10$, $\beta = 1.0$, reg $= 10^{-12}$ |
| Preconditioned LBM \[20,21\] | Balanced PLBE transform ($\gamma$-scaled) + block preconditioner | $\gamma = 0.5$, max_steps $\leq 1.2 \times 10^{6}$ |
| Inexact Newton--Krylov \[6,7\] | JFNK (Jacobian-free Newton--Krylov): GMRES + nonlinear smoother + relaxation smoother + line search | krylov_max=10, K_ne=20, K_smooth=10, line_search=4 |
| Dual-time multigrid \[14\] | FAS (full approximation scheme) V-cycle, residual-equation smoothing | max_levels=6, V-cycle, K_pre/coarse/post=20/30/20 |

Because the methods stop at different accuracies, all were compared by the cost required to first reach a common target residual $\varepsilon$. This is a time-to-threshold measure: for each method, the cost required for its residual to first fall to the common target level $r_{macro} \leq \varepsilon$, with the same residual level imposed on every method so that the comparison is fair. It was read both as wall time and as the LBE-call count, the latter being a separate operator-work count of the same event. The LBE-call count is a hardware-independent, deterministic operator-work metric giving the number of native collide--stream--boundary operator evaluations, denoted $G$ and defined formally in Section 3.3.2. The proposed method solved all 27 problems with a single deterministic routine, only the problem definition of grid, $Re$, boundary, and mask changing per case.


## **3.2 Convergence behavior**

### **3.2.1 Convergence histories**

![](./media/image2.png){width="4.79908573928259in" height="3.6347025371828523in"}

**Figure 2.** Macroscopic $L_{2}$ residual versus wall time for a representative case (cavity $Re = 1000$, 2x), for all six methods. The baseline methods stall at a hydrodynamic plateau, whereas only the proposed method descends below the stopping tolerance.

Figure 2 plots the six methods' residuals against wall time for a representative high-Reynolds cavity case on a logarithmic residual axis. The baselines descend rapidly but stall at a plateau near $10^{-6}$, whereas the proposed method, shown in bold red, penetrates it and decreases monotonically below the stopping tolerance.

![](./media/image3.png){width="5.833333333333333in" height="5.182422353455818in"}

**Figure 3.** Convergence histories of all nine 1x-grid cases (all methods, monotone wall-time axis).

Figure 3 extends the comparison to all nine 1x cases on identical axes and color conventions, showing that the behavior is not specific to a curated subset. In every panel the proposed method, again in bold red, reaches the lowest residual first. The 2x and 3x histories are in the appendix, in Figures A1 and A2.

**Table 3. Per-level convergence summary for the proposed method.**

  ---------------------------------------------------------------------------------------------------------
  Level      Cases      Converged   Total wall \[s\]   Median residual   Max residual   Median rel. error
  ---------- ---------- ----------- ------------------ ----------------- -------------- -------------------
  1x         9          9           134.2              2.142e-12         2.474e-08      3.260e-03

  2x         9          9           1546.6             3.305e-12         6.409e-08      3.260e-02

  3x         9          9           3507.2             1.153e-11         1.567e-08      2.570e-02
  ---------------------------------------------------------------------------------------------------------

Table 3 aggregates the proposed-method results by level; the method satisfied the three simultaneous convergence conditions of Section 3.1 on all 27 runs. The column reporting the median relative error is not comparable across levels, because the set of cases with a comparable reference differs by level, as some complex geometries lack a reference solution at 2x/3x.

### **3.2.2 Convergence rate and robustness**

The behavior of Section 3.2.1 follows from the linearized structure of Section 2. The asymptotic convergence rate of the native LBE iteration is set by the spectral radius of the linearized operator, which separates into kinetic and hydrodynamic modes. The kinetic modes, associated with the block $J_{kk}$ of the Jacobian of the linearized collide-stream operator introduced in Section 2, are damped strongly and locally. The hydrodynamic modes tied to the conserved moments are instead governed by the Schur complement $S_{m} = J_{mm} - J_{mk}J_{kk}^{-1}J_{km}$ formed from the remaining three blocks of that same Jacobian, where subscript $m$ denotes the conserved-moment (macroscopic) degrees of freedom and $k$ the remaining kinetic degrees of freedom. The damping rate of the largest eigenvalue of $S_{m}$ approaches unity. After enough iterations the residual is therefore determined by this single slow mode, forming the plateau in Figures 2 and 3. The proposed method removes this dominant slow mode by applying an analytic approximation of $S_{m}^{-1}$ restricted to the conserved-moment subspace. The residual therefore continues to decrease where the native iteration stalls, consistent with the acceleration targeting the component that constrains late convergence.

This difference was quantified by the robustness gap, the difference in the fraction of cases each method drives to the full convergence criterion. Under the same stopping criterion, admissibility definition, and iteration budget, the proposed method satisfied the convergence criterion of $r_{macro} \leq 5\tau$ with the plateau and admissibility conditions on all 27 cases, whereas the five baselines converged on only a subset: inexact Newton--Krylov on 15, preconditioned LBM on 14, native LBE and Anderson on 13 each, and dual-time multigrid on 12.

The non-convergence arose from numerical stagnation, not budget exhaustion, as the stored histories indicate. This is clearest for the lid-driven cavity, where the baselines stalled at a common residual floor regardless of method. Across all nine cavity configurations at $Re = 100/400/1000$ and three levels they stopped decreasing in the range $10^{-7}\text{–}10^{-5}$. The final residuals clustered narrowly within each configuration, at $1.0\text{–}1.2 \times 10^{-6}$ for $Re = 1000$ 2x and $1.1\text{–}1.2 \times 10^{-6}$ for $Re = 400$ 3x, indicating a structural barrier rather than a method-specific limit. For cavity $Re = 400$ 2x the floor was $3.4\text{–}3.6 \times 10^{-6}$, two orders above the target $5\tau = 2.5 \times 10^{-8}$. This floor was reached at $6\text{–}7 \times 10^{5}$ LBE-calls, well short of the $1.2 \times 10^{6}$-call budget of Table 2. For cavity $Re = 1000$ 2x the floor of $1.0\text{–}1.2 \times 10^{-6}$ was about $1.6\text{–}1.7$ orders of magnitude above the same target and was reached well before the level's $1.2 \times 10^{6}$-call budget was exhausted. The curves entered their asymptotic plateau before the budget was reached, indicating that the plateau, not the budget, was the limiting factor. The resulting robustness gap is reported here in its own right.

## **3.3 Performance comparison with the baseline methods**

### **3.3.1 Wall-time comparison**

The wall time to first reach the common threshold $\varepsilon = 10^{-4}$, that is $r_{macro} \leq \varepsilon$, was compared across all six methods. Because this threshold was attained by every method on all 27 cases, this comparison is the most conservative. Several complementary comparisons were performed on this same 27-case, same-threshold data. They differ in what each case is compared against and are defined below.

The first comparison holds the baseline fixed. The time-to-threshold ratio is defined as the ratio of a baseline's time-to-threshold to that of the proposed method, so a value greater than one means the proposed method was faster. Against each baseline individually, over all 27 cases, the median ratio was $1.64 \times$ relative to preconditioned LBM, $2.42 \times$ to native LBE, $2.84 \times$ to inexact Newton--Krylov, $7.34 \times$ to Anderson, and $18.65 \times$ to dual-time multigrid. The proposed method was faster on 19/27 cases against preconditioned LBM, 21/27 against native LBE, 23/27 against inexact Newton--Krylov, 25/27 against dual-time multigrid, and 27/27 against Anderson.

In the second comparison, the proposed method was compared on each case against whichever baseline was fastest on that particular case. The median ratio was $1.09 \times$, that is, near parity with this per-case best competitor. This is the most demanding comparison, and the near parity shows that the speed advantage comes mostly from cases where the usually fast baselines slow down.

The third comparison pooled every individual baseline run over all 27 cases and five baselines, giving 135 case-baseline ratios; the median of this pooled distribution is reported in Table 4. Table 4 also lists the case-level win rate, the fraction of cases on which the proposed method's time-to-threshold was below the mean of the five baseline times.

A fourth, stricter comparison was restricted to the 15 cases in which at least one baseline also satisfied the full convergence criterion of Section 3.2.2. Results for this strict subset and for the all-baseline comparison are summarized in Table 4.

**Table 4. Strict-subset and all-baseline timing comparison (time-to-threshold,** $\varepsilon = 10^{-4}$**).**

  -------------------------------------------------------------------------------------------------------------------
  Comparison set                                      Metric                      Win rate          Median ratio
  --------------------------------------------------- --------------------------- ----------------- -----------------
  Strict subset (15 cases, baseline also converges)   Wall time                   14/15             2.06×

  Strict subset (15 cases, baseline also converges)   Operator work (LBE-calls)   13/15             1.80×

  All baselines (27 cases)                            Wall time                   25/27             2.92×
  -------------------------------------------------------------------------------------------------------------------

For cavity $Re = 1000$ at 2x/3x and the T-junction at all three levels, the convergence history displayed a characteristic three-stage shape that illustrates the underlying mechanism of the acceleration. First, in the initial transient, native relaxation removed the high-wavenumber kinetic modes and the residual dropped sharply. Second, in the ensuing plateau the residual was dominated by the global slow mode. Under the inexact truncated-GMRES correction and the admissibility-based step-acceptance gate, a line-search-like check that damps the step to $\alpha < 1$ when the full correction would violate admissibility, steps were accepted every iteration but the residual barely decreased. In cavity $Re = 1000$ 2x it therefore lingered near $\sim 10^{-7}$ for about 1100 steps and 740 s, and most of the wall time was spent on correction trials of little benefit. Third, once the iterate entered Newton's quadratic-convergence region, the finite-difference Jacobian--vector approximation became accurate, the full step $\alpha = 1$ passed the gate, and a single correction produced a sharp residual drop, consistent with quadratic convergence rather than a fixed per-step reduction factor. Two cases illustrate this terminal collapse. In the T-junction 2x the single terminal correction step dropped the residual from $2.6 \times 10^{-5}$ to $8.1 \times 10^{-12}$, about 6.5 orders of magnitude, after which a few further steps refined it toward its converged value. In cavity $Re = 1000$ 2x it dropped from $9.6 \times 10^{-8}$ to $4.9 \times 10^{-14}$, about 6.3 orders of magnitude, this being the case's final converged residual. Both single-step drops are far larger than the roughly one order of magnitude per step characteristic of the baselines' linear convergence. Both are also broadly consistent with, though not numerically identical to, the residual squaring expected under exact quadratic convergence. The linearly convergent baselines cannot produce such a collapse, so the curve shape is itself evidence that the MSA-LBM correction acts as a Newton step in the conserved-moment subspace.

The cavity flows showed the largest differences between methods, reflecting a structural gap in attainable residual. Where the baselines stalled at the common floor of $10^{-7}\text{–}10^{-5}$ of Section 3.2.2, the proposed method broke through it on every configuration to the $10^{-8}\text{–}10^{-13}$ level, with final residuals from $1.6 \times 10^{-8}$ to $1.9 \times 10^{-13}$. This is two to eight orders deeper, comparing each configuration's own floor with its own final residual rather than the extremes of the two ranges; full per-case values are in Appendix Table A1. It was also reached in shorter wall time. At $Re = 1000$ 1x the proposed method reached $2.4 \times 10^{-9}$ in 57 s while the baselines spent 531--774 s stalled at the floor. At $Re = 100$ 1x it reached $1.9 \times 10^{-13}$ in 0.7 s versus 72--114 s of baseline stagnation. The penetration recurred across all Reynolds numbers and levels because the cavity is a closed, recirculation-dominated flow governed by the global slow mode that native relaxation cannot damp efficiently. The advantage persisted even at the deepest common threshold all baselines reached, $10^{-5}$. At $Re = 1000$ 2x it reached $10^{-5}$ in 70.8 s, against 178.7 s for native LBE, a ratio of $2.5\times$. This compares with 218.8 s for Anderson, a ratio of $3.1\times$, and 940.3 s for dual-time multigrid, a ratio of $13.3\times$.

Consistent with the mechanism, the net gain scales with how strongly the global slow mode governs convergence; where it is weak the per-iteration correction cost of restriction to the conserved-moment subspace, inner GMRES, and admissibility gate is not amortized, as two cases illustrate. In Couette flow the linear distribution is exact in the LBM equilibrium, so the slow mode is essentially absent. At 1x the proposed method was faster than all baselines by $3.3\text{–}31 \times$ at $10^{-5}$. On refinement, however, native relaxation alone suffices while the correction cost grows, reversing the order. At 2x it took 1.8 s to reach $10^{-4}$, about $3.6 \times$ slower than native LBE at 0.5 s, and at $10^{-5}$ was comparable to native LBE at $0.90 \times$ and preconditioned LBM at $0.96 \times$. In the T-junction, strong inlet/outlet driving makes boundary-local modes dominant. At 2x it took 95.4 s to reach $10^{-5}$, about $5 \times$ slower than native LBE at 18.3 s and preconditioned LBM at 17.8 s, yet still reached a final converged residual of $1.3 \times 10^{-12}$, deeper than any baseline. At 3x it recovered its advantage at $10^{-5}$ over native LBE at $1.30 \times$ and preconditioned LBM at $1.38 \times$. The advantage is thus maximized on slow-mode-dominated flows such as high-Reynolds closed recirculation and can be small or negative in simple shear or boundary-driven flows. Full per-case arrival times are in Appendix Table A1.

### **3.3.2 Operator-work (LBE-call) comparison**

Because wall time depends on processor performance, memory bandwidth, and implementation language, it is complemented here by the operator-work metric introduced in Section 3.1, which is environment-independent and fully deterministic. The most expensive LBM primitive is one lattice update, a single evaluation of $G$ bundling collision, streaming, and boundary handling, shared by all six methods. The total number of $G$ evaluations to reach steady state, the LBE-call count, therefore measures the intrinsic algorithmic work. At the common threshold $\varepsilon = 10^{-4}$ the proposed method's median operator work was smaller by a factor of $1.17$ relative to inexact Newton--Krylov, $1.32$ relative to preconditioned LBM, $1.33$ relative to native LBE, $2.60$ relative to Anderson, and $3.85$ relative to dual-time multigrid. Compared with whichever baseline used the fewest operator evaluations on each case, the proposed method used a median of about 15% more operations at this loose threshold. Each outer iteration spends several $G$ evaluations on finite-difference Jacobian--vector products and inner GMRES, against one per baseline relaxation step, so fewer iterations at higher per-iteration cost balance out to near parity at a loose threshold. Wall time depends on the computing environment, while the operator-work count does not. Since both metrics show the same pattern, near parity at the loose threshold and a growing advantage on the harder problems, the advantage is a property of the algorithm, not of the measurement.

The operator-work advantage was most pronounced on the hard, slow-mode-dominated problems. There the baselines could not remove the slow mode and kept evaluating $G$ while stalled at the floor, whereas the proposed method descended below it in the conserved-moment subspace with fewer operations. On the finest-grid high-Reynolds cavity at $Re = 1000$, for example, it spent only 24,521 $G$ evaluations to reach $10^{-4}$, against 45,592--125,985 for the five baselines, smaller by a factor of $1.9$–$5.1$ for the same accuracy.

## **3.4 Solution accuracy**

![](./media/image4.png){width="5.833333333333333in" height="1.8713287401574803in"}

**Figure 4.** Grid-refinement accuracy. (a) Second-order convergence for Poiseuille flow, (b) near-zero error (at the level of the solver's stopping tolerance) for Couette flow, (c) monotone decrease of the cavity Ghia error.

![](./media/image5.png){width="5.833333333333333in" height="3.491405293088364in"}

**Figure 5.** Cavity centerline $u(y)$ and $v(x)$ profiles on the 3x grid compared against Ghia (1982) for $Re = 100/400/1000$.

Assessing whether the accelerated iteration converges to the correct solution is essential to the validity of the method. The acceleration does not modify the native residual, so its converged solution should coincide with that of the native iteration. This section checks that expectation against analytic solutions, literature benchmarks, and reference solutions.

For plane Poiseuille flow, which admits an analytic solution, the relative $L_{2}$ error of the velocity profile was $9.37 \times 10^{-3}$, $2.27 \times 10^{-3}$, and $1.00 \times 10^{-3}$ at $N_{y} = 32,64,96$. These give observed convergence orders of 2.04 and 2.02, matching the theoretical second-order accuracy of BGK-LBM, the Bhatnagar--Gross--Krook lattice Boltzmann method, for smooth flows, as shown in Figure 4a. The acceleration thus preserves the discretization order. Linear Couette flow is represented exactly by the LBM equilibrium, so its error should be essentially zero. The measured relative $L_{2}$ error was between $2.75 \times 10^{-9}$ and $5.19 \times 10^{-8}$, near the level of the solver's own convergence tolerance rather than reflecting a discretization error, as seen in Figure 4b, indicating that the acceleration introduces no additional error. The error against the Ghia cavity centerline velocities decreases monotonically with refinement at all three Reynolds numbers, converging consistently toward the literature solution, as shown in Figure 4c; no formal order is claimed. The 3x centerline profiles in Figure 5 lie close to the Ghia data at all three Reynolds numbers.

**Table 5. Accuracy summary for cases with an analytic or reference profile (1x).**

  --------------------------------------------------------------------------------------------
  Case                        Wall \[s\]   Final residual   Rel. $L_{2}$ vs ref   Reference
  --------------------------- ------------ ---------------- --------------------- ------------
  Plane Poiseuille (N_y=32)   20.30        3.384e-13        9.371e-03             analytic

  Couette (N=32)              1.20         2.180e-12        2.750e-09             analytic

  Cavity Re=100 (N=33)        0.70         1.935e-13        0.117                 Ghia

  Cavity Re=400 (N=49)        3.06         2.045e-11        0.106                 Ghia

  Cavity Re=1000 (N=129)      56.84        2.360e-09        0.0542                Ghia

  Multi-cylinder (N=32)       1.25         2.142e-12        4.146e-05             Reference solution

  Backward step (N=64)        27.66        2.474e-08        3.260e-03             Reference solution

  Cylinder wake (N=64)        4.88         9.882e-15        7.935e-05             Reference solution

  T-junction (N_x=96)         18.29        2.633e-13        1.896e-05             Reference solution
  --------------------------------------------------------------------------------------------

For complex geometries without an analytic or literature solution, the proposed-method field was compared against the corresponding reference solutions, as summarized in Table 5. The T-junction provides the most direct check. Its reference is a strictly converged native-iteration field, and the relative $L_{2}$ difference from the proposed-method field was only $1.9 \times 10^{-5}$. The acceleration therefore reaches the same discrete steady state faster rather than detouring to a different solution. Of the remaining complex-geometry cases, multi-cylinder and cylinder wake likewise showed relative $L_{2}$ errors at the $10^{-5}$ level (Table 5). The backward-facing step instead showed a difference about two orders of magnitude larger (Table 5). At the step corner the geometry is discontinuous and the velocity gradient is locally singular, so the discrete solution converges slowly there and the reference solution itself likely carries a larger local error; the discrepancy between the two fields is therefore attributed largely to this region. For the backward-facing step and cylinder wake, this relative-$L_{2}$ field agreement and the qualitative field reproduction discussed with Figures 6 and 7 were the only accuracy measures applied; a quantitative validation of separation-point, reattachment-length, or wake metrics was not performed.

![](./media/image6.png){width="5.833333333333333in" height="4.427425634295713in"}

**Figure 6.** Velocity magnitude with streamlines for the nine geometries (3x, proposed-method solution; obstacles shaded gray).

![](./media/image7.png){width="5.833333333333333in" height="4.5858475503062115in"}

**Figure 7.** Vorticity fields for the nine geometries (3x, proposed-method solution). The uniform color of the Couette case reflects the physical fact of linear shear (constant vorticity).

Finally, Figures 6 and 7 show, for all nine geometries on the 3x grid, the velocity field with streamlines and the vorticity from the proposed-method solution. Every geometry qualitatively reproduces the characteristic flow structure---the cavity's primary and secondary recirculation vortices, the backward-facing step's separation and reattachment, the cylinder wake's shear layer, the T-junction flow distribution, and the bypass flow around the multiple cylinders. This complements the quantitative relative-$L_{2}$ verification of Table 5, without implying a quantitative measurement of these structures.
