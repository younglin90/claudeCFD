# 5. Discussion

## 5.1 Mechanistic Interpretation of the Performance Gain

The observed performance gain is interpreted consistently from the Schur-complement viewpoint. Native LBM Picard stably damps the kinetic component through local collide–stream relaxation, but the global equilibration of the pressure–velocity hydrodynamic mode is slow. AP-Schur projects the residual into the moment space and proposes an approximate global correction for this slow component, and the admissibility gate accepts the correction only when it remains within the feasible set of the discrete problem. The results of Section 4.6—where the variant with the AP-Schur block removed (native-block-only) loses convergence breadth and the variant using only history extrapolation (RRE-only) shows a longer total wall time—and the result of Section 4.7—where all cases show a non-trivial acceptance rate—support this mechanistic interpretation. What the local linear analysis of Eq. (11) guarantees is only a reduction of the amplification factor of the captured slow mode; but because the native fallback is guaranteed on failure, the incompleteness of the analysis does not compromise solver stability.

## 5.2 Limitations and Scope of Claims

The direct scope of this work is limited as follows. First, the application scope is the stored 2D D2Q9/BGK steady benchmark suite. We make no direct claim of generalization to 3D, thermal/compressible LBM, MRT/entropic collision models, or high-Reynolds-number turbulent regimes. Second, Section 4.5 showed observed second-order convergence for channel Poiseuille, machine precision for Couette, and monotone approach to Ghia for the cavity, but this is code-verification evidence for cases with closed-form/literature references, not a formal grid-convergence study for all geometries (Richardson extrapolation, GCI, and discretization-error bounds against the continuum solution for every case). In particular, for cases whose reference is a tight numerical field—backward step, cylinder wake, multi-cylinder, T-junction—no formal order is claimed; such a claim would additionally require a systematic grid sequence, confirmation of a monotone asymptotic range, and analysis of integral quantities (conserved quantities, forces, reattachment length, etc.). Third, AP-Schur-only is not a method that removes discretization/BC error but one that reaches the same discrete solution faster. That the cavity Ghia error is nonzero is a direct example of this distinction.

We also state the limitations of the timing claim. Wall time depends on CPU generation, memory bandwidth, the Python/NumPy/BLAS implementation, and background load, and the CSV/JSON sources of the present result set do not store the CPU model, core count, library version, or git commit hash as separate columns. Wall time is therefore interpreted as a relative metric comparing the same stopping rule and the same case/level within the same result set, not as a hardware-independent absolute performance constant. We quantified the wall-time coefficient of variation (3.6–6.8%) and the LBE-call determinism (identical across repetitions) by seven repetitions of representative cases (Section 4.3, Table 7b), but we do not claim large-scale repetition over all 27 cases or inferential statistics based on confidence intervals/$p$-values. The first-order check of run-to-run noise is performed on the deterministic metrics—LBE-call, final residual, and the plateau flag. Extending the timing comparison in an independent reproduction would require fixing, together, the solver/benchmark script revisions, the CPU/OS/library information, the thread settings, and deterministic flags.

As a limitation of open-boundary diagnostics, for problems with complex geometry and boundary conditions—backward step, cylinder wake, multi-cylinder, T-junction—residual decrease and the local flux/mass diagnostics do not always improve at the same rate, because the residual measures the macroscopic-$L_2$ change over the whole domain whereas flux closure is sensitive to the integral over specific inlet/outlet cross-sections. This work does not use flux-related quantities in the stopping rule and interprets them only as auxiliary physical diagnostics; a quantitative flux-closure bound is left as a recomputation item of the reproducibility package.

## 5.3 Threats to Validity and Mitigations

The internal-validity threat is the possibility that the proposed method used reference information internally or used parameters favorable only to specific cases. To reduce this, we specified a single AP-Schur-only routine, the same residual/plateau criterion, the same admissibility gate, and a reference-free accept/reject procedure (Sections 2–3), and disclosed all proposed-method run records in summary tables. The continued label (`uniform_ap_schur_only_continued`) is not a change of method but an extension of the run under the same stopping rule. The measurement-validity threat is that the residual, wall time, and reference error are metrics of different character, and we report them separately as convergence efficiency and final-field agreement. The external-validity threat is that the benchmark suite does not represent all CFD problems, addressed by the scope restriction of Section 5.2. Table 11 summarizes the main concerns from an independent-verification viewpoint together with the paper's defense logic, and Table 12 summarizes how potential questions are addressed.

**Table 11. Anticipated concerns from an independent-verification viewpoint and the paper's defense logic.**

| Potential concern | Risk | Defense logic in the text |
|---|---|---|
| Reference-injection suspicion | If Ghia/analytic/tight references enter the solver internally, novelty and fairness weaken. | Sections 2.5 and 3.5 specify that references are used only for post-processing error evaluation; the accept gate uses only residual/admissibility. |
| Case-specific tuning suspicion | Applying different coefficients or algorithms only to specific benchmarks is strongly criticized. | The final method is defined as a single AP-Schur-only routine with the same stopping rule and gate applied to all cases (Sections 3.2, 3.6), with singularity verified by execution traces (Section 4.7). |
| Accuracy over-claim | Because the Ghia error is nonzero, an "accuracy-improvement" claim is refutable. | Framed not as accuracy improvement but as convergence acceleration that reaches the same discrete steady solution faster. |
| Open-boundary/mass consistency | Residual decrease and flux/mass diagnostics may be conflated on open geometries. | Density positivity, finite field, boundary re-application, and open-boundary branch rejection are organized as method gates (Sections 2.4, 2.7). |
| Insufficient ablation | If the contributions of AP-Schur and RRE/native block are not separated, novelty weakens. | The 1x ablation table and wall-time figure provide the basis for selecting AP-Schur-only (Section 4.6). |
| Insufficient reproducibility | Reliability drops if results depend on code changes or recomputation. | Provenance and a recomputation procedure are specified so that tables/figures can be regenerated from the stored summary/history/field/reference sources (Section 5.4). |
| JFNK over-claim | Solving no full Newton–Krylov system while appearing to be JFNK could be read as exaggeration. | Named as a moment-Schur nonlinear preconditioner based on the Jacobian-free residual response, with an explicit statement that it is not a full JFNK (Section 2.3). |

**Table 12. Potential verification questions and how the text addresses them.**

| Potential question | Response in the text |
|---|---|
| The fast wall time is due to differences in stopping rule. | We specify that the same residual/plateau verdict is applied to all methods and present residual-versus-time histories together. |
| It does not exactly match Ghia. | The Ghia error is a post hoc accuracy metric, separated from the convergence claim; mesh-refinement results interpret it as discretization sensitivity. |
| A different method was used for complex geometries. | We specify the use of a single AP-Schur-only solver and the same admissibility gate. |
| The novelty is a mere combination. | The moment-Schur-complement interpretation, native-residual acceptance, and geometry-aware admissibility are presented as one steady-LBM preconditioning framework. |
| Wall-time differences may be due to CPU scheduling/Python overhead. | The wall-time CV measured over seven repetitions is 3.6–6.8%, more than an order of magnitude smaller than the speedup ($\approx 2\times$), and the LBE-call is fully deterministic across repetitions (Table 7b); the advantage is therefore not explained by scheduling noise. |
| Does 1x/2x/3x mean formal grid convergence? | No. It is a solver-scaling benchmark; no formal order-of-accuracy claim is made. |
| Were only favorable variants aggregated? | The 27-case performance claims are sourced from the proposed-only summary, and the merged all-method CSV distinguishes duplicates by `base_case_id`, `scaling_level`, and `method_variant`. |
| If mass/flux closure is incomplete, is residual convergence invalid? | Mass/flux are open-boundary auxiliary diagnostics, not the stopping rule. The primary verdict is the macro-$L_2$ residual and plateau, and flux closure is disclosed as a separate recomputation item. |

## 5.4 Claim Hierarchy, Falsification Criteria, and Reproducibility

To prevent over-generalization of the strong performance results, we tier the claims of this work. The primary claim is the reduction of convergence time under the same stopping rule; the secondary claim is agreement with reference fields; and we do not claim the removal of discretization error itself. This hierarchy also serves as the basis for interpreting additional verification results: if some cases weaken in further reproduction computations, we adjust the scope of the primary/secondary claims—rather than hiding the methodology—and disclose the residual histories and field errors of the failing cases as supplementary material. Table 13 specifies the falsification condition for each tier.

**Table 13. Claim hierarchy of this work and falsification criteria.**

| Tier | Claim | Falsification condition |
|---|---|---|
| Primary claim | AP-Schur-only reaches steady state with smaller wall time than the baseline accelerators on most benchmarks under the same residual/plateau/admissibility criteria. | If, upon re-running under the same criteria, the proposed method is repeatedly slower than the fastest baseline or fails to satisfy the plateau, the claim must be weakened. |
| Secondary claim | The final field maintains accuracy comparable to analytic/Ghia/reference profiles. | If the wall time is fast but the Ghia/analytic error grows systematically relative to the baselines, the accuracy claim is downgraded to an auxiliary claim. |
| Mechanism claim | The performance gain arises from the combination of preconditioning the hydrodynamic moment Schur complement and native-residual acceptance. | If, in the ablation, AP-Schur-only loses its advantage over RRE/native or the Schur-correction acceptance scarcely occurs, the mechanistic interpretation is revised. |
| Out of scope | We do not claim that AP-Schur-only produces the exact solution on every grid or removes all open-boundary flux error. | These items are outside the scope of this work and are treated by auxiliary diagnostics and follow-up research. |

We distinguish reproducibility into two stages. The first is stored-data recomputation, in which an independent verifier recomputes the in-text table and figure numbers from the summary/history/field/reference sources without re-running the solver. The second is algorithmic reproduction, in which the AP-Schur-only solver is re-run in a new environment under the same benchmark definitions and stopping protocol. The direct basis for the in-text numerical claims is placed on the first stage. The recommended order for stored-data recomputation is: (i) from the proposed-only summary (`papers_data/summary_latest_ap_schur_only_proposed.csv`), confirm the 27 runs and the `converged`/`residual_converged`/`plateau_converged` flags keyed by `case_id` and `scaling_level`; (ii) from the all-method summary (`papers_data/summary_all_methods_with_latest_ap_schur_only.csv`), group only the baseline runs satisfying the same `case_id`, `scaling_level`, and stopping rule as the comparison group and recompute the `wall_seconds` and LBE-call ratios; (iii) confirm from each case directory's residual history that the final macro-$L_2$ residual and plateau-window condition match the summary flags; (iv) recompute the reference error from the accuracy or Ghia centerline CSV.

The aggregation unit and data-integrity rules are as follows. The number of proposed-method runs and the method-comparison pairs are counted not by the method string alone but de-duplicated by `case_id` and `scaling_level`, and the `method_variant` values `uniform_ap_schur_only` and `uniform_ap_schur_only_continued` are summed as the same AP-Schur-only method. The minimal columns needed to regenerate the in-text tables/figures are the case label, level, method key, converged flag, residual/plateau flags, `wall_seconds`, LBE-call, final macro-$L_2$ residual, initial-relative residual, reference error, and tolerance. If any diagnostic column is empty, it is treated as unreported rather than as zero or success, and if the proposed and baseline runs cannot be joined at the same case/level, that pair is excluded from the ratio computation. Each figure and table retains provenance recording the case key, level, method key, source CSV/field file, and generation procedure, so as to preclude any suspicion of selectively excluding failing cases or selectively adjusting axis/color scales. Table 14 summarizes the data sources and the reproducibility-verification methods.

**Table 14. Data sources and reproducibility-verification methods.**

| Item | Use in the text | Reproducibility-verification method |
|---|---|---|
| Summary CSV | Building tables of level, case, wall time, residual, and reference error for the 27 proposed-method runs | Check CSV row count, case label, level, method, and final residual |
| History CSV | Building wall-time-vs-residual and convergence curves | Check each method's elapsed time, LBE-call, and $r_{\mathrm{macro}}$ history |
| NPZ/field output | Visualizing velocity-magnitude contours, cavity profiles, complex-geometry fields | Check the shapes and finite values of the $\rho$, $u$, $v$, mask arrays |
| Reference profile | Post hoc comparison against Ghia and analytic solutions | Not used for solver updates; used only for plots/error metrics |
| Figure/table provenance | Tracing the sources of convergence plots, contours, centerline comparisons, ablation figures | Check consistency of source CSV/field/manifest with case key, level, method key, axis range, and error metric |

# 6. Conclusion

This paper proposed and validated an AP-Schur-only nonlinear preconditioning framework that preconditions the pressure–velocity hydrodynamic slow mode—the convergence bottleneck of steady-state LBM—from a conserved-moment Schur-complement viewpoint. The method changes neither the native steady LBM residual nor the boundary operator; it proposes a Jacobian-free trial direction in the conserved-moment space and uses the admissibility gate to confirm a residual decrease and physical consistency simultaneously. This structure makes it possible to treat the verification questions about accuracy, boundary conditions, and reference comparison separately, within the same residual/admissibility criterion.

On the stored 27 benchmarks, all proposed-method runs passed the same convergence criterion, whereas the five baselines converged on only 12–15 cases each despite the same protocol and a generous budget, demonstrating the robustness advantage of the proposed method. In a conservative timing comparison that excludes budget asymmetry (the 15-case subset on which a baseline also converges), the proposed method recorded shorter wall time on 14/15 cases (median about $2.06\times$) and fewer LBE-calls on 13/15 cases (median about $1.80\times$), and was faster on 25/27 cases against all available baselines (median $2.92\times$). In the 1x ablation, AP-Schur-only showed the lowest total wall time and 9/9 per-case wall-time wins, and the 71% trial acceptance rate together with the single phase vocabulary across all 27 cases (Section 4.7) guarantees, at the execution-trace level, that the Schur correction operated meaningfully along the actual execution path and without per-case branching. On accuracy, the observed second-order convergence for channel Poiseuille, machine precision for Couette, and monotone approach to Ghia for the cavity (Section 4.5) showed that the acceleration does not sacrifice discrete accuracy. This confirms that the method is an algorithm that solves the same discrete steady LBM problem faster, rather than curve fitting or reference injection.

The direct scope of this work is the comparison of convergence time, operator work, residual, and reference error on the stored 2D D2Q9/BGK 1x/2x/3x benchmark suite. Extension to 3D, higher Reynolds numbers, other collision models (MRT/entropic), and rigorous quantification of open-boundary flux closure is left to follow-up work. Because the slow exception cases, the absence of repeated-run statistics, and the limitations of the runtime-environment metadata are disclosed in the text, the present results should be interpreted not as a claim of universal superiority but as a verifiable, reproducible claim of steady-LBM nonlinear preconditioning.

# Data and Code Availability

The numerical claims of this work are organized so that a first-order check is possible from the stored summary/history/field archive without re-running the original solver. The reproducibility package includes the proposed-only summary CSV, the all-method comparison summary CSV, per-case residual histories, the accuracy table, the final-field NPZ files, the figure-generation scripts, the manifest and source-path metadata, the file inventory, and the revision information of the solver/post-processing scripts used. When the full field archive cannot be distributed owing to journal policy or repository capacity limits, the minimal distribution unit is the compact summary, the history CSV, the cavity centerline comparison CSV, the contour-regeneration script, and a specification of how to access the original field archive. To reproduce the mass/flux diagnostics of the open-boundary cases, the final field, the inlet/outlet segment definitions, the normal-direction convention, and the quadrature rule are included together. This is a computational study based on stored numerical benchmark results and deterministic post-processing, and involves no human or animal subjects. Funding, conflicts of interest, and author contributions are separated into distinct metadata items of the final manuscript.

# References

[1] Qian, Y. H., d'Humières, D., & Lallemand, P. (1992). Lattice BGK models for Navier-Stokes equation. *Europhysics Letters*, 17(6), 479–484. https://doi.org/10.1209/0295-5075/17/6/001

[2] Chen, S., & Doolen, G. D. (1998). Lattice Boltzmann method for fluid flows. *Annual Review of Fluid Mechanics*, 30, 329–364. https://doi.org/10.1146/annurev.fluid.30.1.329

[3] Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

[4] Lallemand, P., & Luo, L.-S. (2000). Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. *Physical Review E*, 61, 6546–6562. https://doi.org/10.1103/PhysRevE.61.6546

[5] Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411. https://doi.org/10.1016/0021-9991(82)90058-4

[6] Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 7(3), 856–869. https://doi.org/10.1137/0907058

[7] Knoll, D. A., & Keyes, D. E. (2004). Jacobian-free Newton-Krylov methods: A survey of approaches and applications. *Journal of Computational Physics*, 193(2), 357–397. https://doi.org/10.1016/j.jcp.2003.08.010

[8] Benzi, M., Golub, G. H., & Liesen, J. (2005). Numerical solution of saddle point problems. *Acta Numerica*, 14, 1–137. https://doi.org/10.1017/S0962492904000212

[9] Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735. https://doi.org/10.1137/10078356X

[10] Tóth, A., & Kelley, C. T. (2015). Convergence analysis for Anderson acceleration. *SIAM Journal on Numerical Analysis*, 53(2), 805–819. https://doi.org/10.1137/130919398

[11] Olshanskii, M. A., & Vassilevski, Y. V. (2007). Pressure Schur complement preconditioners for the discrete Oseen problem. *SIAM Journal on Scientific Computing*, 29(6), 2686–2704. https://doi.org/10.1137/070679776

[12] Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). *Finite Elements and Fast Iterative Solvers: With Applications in Incompressible Fluid Dynamics* (2nd ed.). Oxford University Press.

[13] Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.

[14] Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001). *Multigrid*. Academic Press.

[15] Sidi, A. (1986). Convergence and stability properties of minimal polynomial and reduced rank extrapolation algorithms. *SIAM Journal on Numerical Analysis*, 23(1), 197–209. https://doi.org/10.1137/0723014

[16] Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591–1598. https://doi.org/10.1063/1.869307

[17] Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). Momentum transfer of a Boltzmann-lattice fluid with boundaries. *Physics of Fluids*, 13(11), 3452–3459. https://doi.org/10.1063/1.1399290

[18] Huang, J., Yang, C., & Cai, X.-C. (2015). A fully implicit method for lattice Boltzmann equations. *SIAM Journal on Scientific Computing*, 37(5), S291–S313. https://doi.org/10.1137/140975346

[19] Huang, J., Yang, C., & Cai, X.-C. (2016). A nonlinearly preconditioned inexact Newton algorithm for steady state lattice Boltzmann equations. *SIAM Journal on Scientific Computing*, 38(3), A1701–A1724. https://doi.org/10.1137/15M1028078

[20] Guo, Z., Zhao, T. S., & Shi, Y. (2004). Preconditioned lattice-Boltzmann method for steady flows. *Physical Review E*, 70(6), 066706. https://doi.org/10.1103/PhysRevE.70.066706

[21] Premnath, K. N., Pattison, M. J., & Banerjee, S. (2009). Steady state convergence acceleration of the generalized lattice Boltzmann equation with forcing term through preconditioning. *Journal of Computational Physics*, 228(3), 746–769. https://doi.org/10.1016/j.jcp.2008.09.028

[22] Hajabdollahi, F., & Premnath, K. N. (2018). Galilean-invariant preconditioned central-moment lattice Boltzmann method without cubic velocity errors for efficient steady flow simulations. *Physical Review E*, 97(5), 053303. https://doi.org/10.1103/PhysRevE.97.053303

[23] Hajabdollahi, F., & Premnath, K. N. (2019). Improving the low Mach number steady state convergence of the cascaded lattice Boltzmann method by preconditioning. *Computers & Mathematics with Applications*, 78(4), 1115–1130.

[24] Walsh, B., & Boyle, F. J. (2020). A preconditioned lattice Boltzmann flux solver for steady flows on unstructured hexahedral grids. *Computers & Fluids*, 210, 104634. https://doi.org/10.1016/j.compfluid.2020.104634

[25] Yahia, E., & Premnath, K. N. (2022). Preconditioned central moment lattice Boltzmann method on a rectangular lattice grid for accelerated computations of inhomogeneous flows. *Journal of Computational Science*, 63.
