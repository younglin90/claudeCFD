## **3.5 Aggregate performance across the benchmark suite**

Sections 3.3 and 3.4 compared the methods case by case. This section summarizes the twenty-seven cases collectively, first through performance profiles that rank the methods over the whole suite, then through the distribution of the per-case speed-up, and finally through the agreement of the accelerated solution with reference data.

### **3.5.1 Performance profiles**

The per-case metrics of Section 3.3 are aggregated over the whole test suite by the performance profile of Dolan and Moré [43]. For a solver $s$ and a problem $p$ in the set $P$ of the twenty-seven benchmark cases, let $c_{p,s}$ denote the cost required by $s$ to satisfy the convergence criterion of Section 3.1 on problem $p$, with $c_{p,s}=\infty$ when $s$ fails to converge. The performance ratio is

$$r_{p,s} = \frac{c_{p,s}}{\min_{s'} c_{p,s'}},$$

and the performance profile of $s$ is the cumulative distribution

$$\rho_s(\tau) = \frac{\bigl|\{\, p \in P : r_{p,s} \le \tau \,\}\bigr|}{|P|}, \qquad \tau \ge 1.$$

Here $\rho_s(1)$ is the fraction of cases on which $s$ is the fastest solver, the right-hand asymptote $\lim_{\tau\to\infty}\rho_s(\tau)$ is the fraction it solves at all, and a profile lying above all others indicates the preferred method. Figure 7 reports $\rho_s$ for two cost measures: wall-clock time, which reflects a specific implementation and machine, and native-operator (LBE) calls, which is hardware-independent and counts only evaluations of the collide–stream–boundary operator.

Under both measures the MSA-LBM profile lies above every baseline for all $\tau$. Two features are quantitative. First, at $\tau=1$ the MSA-LBM profile attains $0.963$ in wall-clock time and $0.926$ in operator calls: the proposed method is the fastest, or tied for fastest, on 26 and 25 of the 27 cases, respectively, whereas no baseline is fastest on more than one case. Second, the MSA-LBM profile reaches unity at $\tau=1.56$ (wall-clock time) and $\tau=1.28$ (operator calls), so on every case its cost is within a factor of $1.6$ of the best solver for that case. The baseline profiles instead saturate well below unity, at $0.44$ to $0.56$, because each baseline converges on only 12 to 15 of the 27 cases (Section 3.2); their right-hand asymptotes are exactly these solved fractions. The vertical gap between the MSA-LBM profile and the baselines at large $\tau$ therefore measures robustness, and the horizontal gap at fixed $\rho$ measures efficiency. The proposed method dominates in both.

![](figS1_performance_profile.png){width=5.83in}

**Figure 7.** Performance profiles over the 27-case suite at the full convergence criterion, in (a) wall-clock time and (b) native-operator calls. $\rho_s(\tau)$ is the fraction of cases solved by method $s$ within a factor $\tau$ of the fastest method. MSA-LBM lies above all baselines for every $\tau$, is fastest on 26/27 (wall) and 25/27 (calls) cases at $\tau=1$, and reaches all cases by $\tau\le 1.56$; each baseline saturates at its solved fraction, 0.44–0.56.

### **3.5.2 Distribution of the per-case speed-up**

The median speed-ups of Table 4 are point summaries of the distributions in Figure 8, which plots, for each baseline, the ratio of that baseline's cost to the MSA-LBM cost to reach the common threshold $\varepsilon=10^{-4}$ on each of the 27 cases. In wall-clock time the median speed-up ranges from $1.6\times$ against the preconditioned LBM to $18.6\times$ against dual-time multigrid, with the interquartile range spanning roughly a decade for the slowest baselines; in operator calls the medians are lower, from $1.2\times$ to $3.8\times$. The difference between the two measures is informative: the operator-call speed-up isolates the purely algorithmic gain, whereas the additional wall-clock advantage reflects the per-iteration overhead that Anderson mixing, the Krylov solve of the preconditioned method, and the dual-time sub-iterations incur beyond the native operator itself. The operator-call figure is thus the more conservative statement of the improvement.

A minority of cases fall below parity, and these are identified rather than omitted. They are confined to the analytically simple Couette and coarse channel flows, where native iteration already reaches the loose threshold $\varepsilon=10^{-4}$ within a few tens of sweeps, before the dense per-mode factorization of the moment-Schur operator is amortized; the deepest instance is Couette flow at $0.28\times$. This behaviour is a property of the loose threshold and the trivial cases: at the full convergence criterion (Figure 7) the proposed method is the fastest or tied on 26 of 27 cases, and the sub-parity instances disappear. The advantage widens as the tolerance is tightened and as the Reynolds number increases, which is the regime of practical interest.

![](figS2_speedup_distribution.png){width=5.83in}

**Figure 8.** Distribution of the per-case speed-up of MSA-LBM over each baseline to reach $\varepsilon=10^{-4}$, across the 27 cases (violin: kernel density; box: quartiles; points: individual cases; dashed line: parity). Bold labels give the medians of Table 4. (a) wall-clock time; (b) native-operator calls.

### **3.5.3 Agreement with reference data**

Acceleration is useful only if it does not change the solution. Two distinct notions of agreement are examined in Figure 9. Panel (a) tests agreement with an independent benchmark: the MSA-LBM velocity along the vertical centreline of the lid-driven cavity is plotted against the tabulated data of Ghia et al. [38] for $\mathrm{Re}=100$, $400$, and $1000$. The points collapse onto the line of perfect agreement with a root-mean-square deviation of $8.8\times10^{-3}$ in units of the lid speed, consistent with the second-order spatial accuracy of Section 3.4 on these grids; the accelerated method reproduces the benchmark to discretization accuracy.

Panel (b) tests a stronger property: that the accelerated iteration converges to the same numerical fixed point as the unaccelerated native LBE, rather than to a nearby spurious state. The MSA-LBM velocity magnitude is plotted against the reference solution node by node, pooled over $149\,385$ fluid nodes from six cases spanning channel, Couette, cylinder-wake, backward-step, multi-cylinder, and T-junction geometries. The maximum nodewise deviation is $4.3\times10^{-5}$, four orders of magnitude below the panel-(a) discretization error and of the order of the convergence tolerance itself. The admissibility gate therefore alters only the path to the fixed point, not its location: the proposed method solves the same discrete system as native LBE, and does so on the full suite, including the twelve cases on which every baseline fails to converge.

![](figS3_parity.png){width=5.83in}

**Figure 9.** Agreement of the MSA-LBM solution with reference data. (a) Cavity vertical-centreline velocity against Ghia et al. [38] for three Reynolds numbers (RMS deviation $8.8\times10^{-3}$). (b) Nodewise velocity magnitude against the reference fixed point, pooled over $149\,385$ fluid nodes from six geometries (maximum deviation $4.3\times10^{-5}$). Dashed line: perfect agreement.
