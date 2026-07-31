# Statistical analysis of aggregate performance and solution agreement

*Publication-ready English text for insertion into Section 3 (Results). Figure
numbers are provisional (shown as Fig. A/B/C); renumber to follow the existing
sequence. "MSA-LBM" and "native LBE" follow the paper's established terms.*

---

## 3.3.3 Aggregate performance across the benchmark suite

The per-case metrics of Sections 3.3.1–3.3.2 are summarized over the whole test
suite by the performance profile of Dolan and Moré [Dolan2002]. For a solver *s*
and a problem *p* in the set *P* of the twenty-seven benchmark cases, let
*c*(*p*,*s*) denote the cost required by *s* to satisfy the convergence criterion
of Section 3.1 on problem *p*, with *c*(*p*,*s*) = ∞ when *s* fails to converge.
The performance ratio is

  *r*(*p*,*s*) = *c*(*p*,*s*) / min₍s′₎ *c*(*p*,*s′*),

and the performance profile of *s* is the cumulative distribution

  ρ_*s*(τ) = |{ *p* ∈ *P* : *r*(*p*,*s*) ≤ τ }| / |*P*|,  τ ≥ 1.

Thus ρ_*s*(1) is the fraction of cases on which *s* is the fastest solver, the
right-hand asymptote lim₍τ→∞₎ ρ_*s*(τ) is the fraction it solves at all, and a
profile lying above all others indicates the preferred method. Figure A reports
ρ_*s* for two cost measures: wall-clock time, which reflects a specific
implementation and machine, and native-operator (LBE) calls, which is
hardware-independent and counts only evaluations of the collide–stream–boundary
operator.

Under both measures the MSA-LBM profile lies above every baseline for all τ.
Two features are quantitative. First, at τ = 1 the MSA-LBM profile attains 0.963
in wall-clock time and 0.926 in operator calls: the proposed method is the
fastest, or tied for fastest, on 26 and 25 of the 27 cases, respectively, whereas
no baseline is fastest on more than one case. Second, the MSA-LBM profile reaches
unity at τ = 1.56 (wall-clock time) and τ = 1.28 (operator calls), so on every
case its cost is within a factor of 1.6 of the best solver for that case. The
baseline profiles instead saturate well below unity — at 0.44 to 0.56 — because
each baseline converges on only 12 to 15 of the 27 cases (Section 3.2); their
right-hand asymptotes are exactly these solved fractions. The vertical gap
between the MSA-LBM profile and the baselines at large τ is therefore a direct
measure of robustness, and the horizontal gap at fixed ρ is a measure of
efficiency. The proposed method dominates in both.

## 3.3.4 Distribution of the per-case speed-up

The median speed-ups reported in Table 4 are point summaries of the distributions
shown in Figure B, which plots, for each baseline, the ratio of that baseline's
cost to the MSA-LBM cost to reach the common threshold ε = 10⁻⁴ on each of the
27 cases. In wall-clock time the median speed-up ranges from 1.6× against the
preconditioned LBM to 18.6× against dual-time multigrid, with the interquartile
range spanning roughly a decade for the slowest baselines; in operator calls the
medians are lower, from 1.2× to 3.8×. The difference between the two measures is
informative: the operator-call speed-up isolates the purely algorithmic gain,
while the additional wall-clock advantage reflects the per-iteration overhead
that Anderson mixing, the Krylov solve of the preconditioned method, and the
dual-time sub-iterations incur beyond the native operator itself. The
operator-call figure is thus the more conservative statement of the improvement.

A minority of cases fall below parity (speed-up < 1×), and these are identified
rather than omitted. They are confined to the analytically simple Couette and
coarse channel flows — where native iteration already reaches the loose threshold
ε = 10⁻⁴ within a few tens of sweeps, before the dense per-mode factorization of
the moment-Schur operator is amortized — the deepest instance being Couette flow
at 0.28×. This behaviour is a property of the loose threshold and the trivial
cases: at the full convergence criterion (Figure A) the proposed method is the
fastest or tied on 26 of 27 cases, and the sub-parity instances disappear. The
advantage widens monotonically as the tolerance is tightened and as the Reynolds
number increases, which is the regime of practical interest.

## 3.4.x Agreement of the accelerated solution with reference data

Acceleration is useful only if it does not change the solution. Two distinct
notions of agreement are examined in Figure C. Panel (a) tests agreement with an
independent benchmark: the MSA-LBM velocity along the vertical centreline of the
lid-driven cavity is plotted against the tabulated data of Ghia et al. [Ghia1982]
for Re = 100, 400, and 1000. The points collapse onto the line of perfect
agreement with a root-mean-square deviation of 8.8 × 10⁻³ in units of the lid
speed, consistent with the second-order spatial accuracy established in
Section 3.4 on these grids; the accelerated method reproduces the benchmark to
discretization accuracy.

Panel (b) tests a stronger and more specific property: that the accelerated
iteration converges to the same numerical fixed point as the unaccelerated native
LBE, rather than to a nearby spurious state. The MSA-LBM velocity magnitude is
plotted against the reference solution node by node, pooled over 149 385 fluid
nodes from six cases spanning channel, Couette, cylinder-wake, backward-step,
multi-cylinder, and T-junction geometries. The maximum nodewise deviation is
4.3 × 10⁻⁵, four orders of magnitude below the panel-(a) discretization error and
of the order of the convergence tolerance itself. The admissibility-preserving
gate therefore alters only the path to the fixed point, not its location: the
proposed method solves the same discrete system as native LBE, and does so on the
full suite, including the twelve cases on which every baseline fails to converge.

---

### Suggested figure captions

**Figure A.** Performance profiles [Dolan2002] over the 27-case suite at the full
convergence criterion, in (a) wall-clock time and (b) native-operator calls.
ρ_*s*(τ) is the fraction of cases solved by method *s* within a factor τ of the
fastest method. MSA-LBM lies above all baselines for every τ, is fastest on
26/27 (wall) and 25/27 (calls) cases at τ = 1, and reaches all cases by τ ≤ 1.56;
each baseline saturates at its solved fraction, 0.44–0.56.

**Figure B.** Distribution of the per-case speed-up of MSA-LBM over each baseline
to reach ε = 10⁻⁴, across the 27 cases (violin: kernel density; box: quartiles;
points: individual cases; dashed line: parity). Bold labels give the medians of
Table 4. (a) wall-clock time; (b) native-operator calls. Sub-parity cases are the
analytically trivial Couette and coarse-channel flows.

**Figure C.** Agreement of the MSA-LBM solution with reference data. (a) Cavity
vertical-centreline velocity against Ghia et al. [Ghia1982] for three Reynolds
numbers (RMS deviation 8.8 × 10⁻³). (b) Nodewise velocity magnitude against the
reference fixed point, pooled over 149 385 fluid nodes from six geometries
(maximum deviation 4.3 × 10⁻⁵). Dashed line: perfect agreement.

### References to add

- Dolan, E.D., Moré, J.J. (2002). *Benchmarking optimization software with
  performance profiles.* Mathematical Programming 91(2), 201–213.
- Ghia, U., Ghia, K.N., Shin, C.T. (1982). *High-Re solutions for incompressible
  flow using the Navier–Stokes equations and a multigrid method.* J. Comput.
  Phys. 48(3), 387–411. *(already cited)*
