Artificial diffusion for convective and acoustic low Mach number flows I: Analysis of the modified equations, and application to Roe-type schemes

> Joshua Hope-Collins†‡§ Luca di Mare†

> > March, 2023

#### Abstract

Three asymptotic limits exist for the Euler equations at low Mach number - purely convective, purely acoustic, and mixed convective-acoustic. Standard collocated density-based numerical schemes for compressible flow are known to fail at low Mach number due to the incorrect asymptotic scaling of the artificial diffusion. Previous studies of this class of schemes have shown a variety of behaviours across the different limits and proposed guidelines for the design of low-Mach schemes. However, these studies have primarily focused on specific discretisations and/or only the convective limit.

In this paper, we review the low-Mach behaviour using the modified equations - the continuous Euler equations augmented with artificial diffusion terms - which are representative of a wide range of schemes in this class. By considering both convective and acoustic effects, we show that three diffusion scalings naturally arise. Singleand multiple-scale asymptotic analysis of these scalings shows that many of the important low-Mach features of this class of schemes can be reproduced in a straightforward manner in the continuous setting.

As an example, we show that many existing low-Mach Roe-type finite-volume schemes match one of these three scalings. Our analysis corroborates previous analysis of these schemes, and we are able to refine previous guidelines on the design of low-Mach schemes by including both convective and acoustic effects. Discrete analysis and numerical examples demonstrate the behaviour of minimal Roe-type schemes with each of the three scalings for convective, acoustic, and mixed flows.

Keywords: Low Mach number flows; Compressible Euler Equations; Computational fluid dynamics; Asymptotic analysis; Numerical dissipation; Roe scheme.

#### Highlights:

- Asymptotic analysis of numerical schemes at low Mach number
- Considers the convective, acoustic, and mixed convective-acoustic low Mach limits
- Derivation of required asymptotic scaling of artificial diffusion at each limit
- Use of modified equations applies to finite-volume or finite-difference schemes
- Application to Roe schemes shows excellent agreement with previous literature

<sup>†</sup>Oxford Thermofluids Institute, University of Oxford, Osney Mead, Oxford, OX2 0ES, England, United Kingdom

<sup>‡</sup>Department of Mathematics, Imperial College London, London, SW7 2AZ, England, United Kingdom

<sup>§</sup>Corresponding author. Email address: joshua.hope-collins@eng.ox.ac.uk

# Contents

| 1 |     | Asymptotic expansion of the Euler equations at low Mach number          | 4  |
|---|-----|-------------------------------------------------------------------------|----|
|   | 1.1 | Single timescale convective limit<br>                                   | 4  |
|   | 1.2 | Two timescale convective-acoustic limit<br>                             | 5  |
| 2 |     | Previous work                                                           | 6  |
| 3 |     | Design of artificial diffusion at low Mach number                       | 7  |
|   | 3.1 | Artificial diffusion for purely convective flows<br>                    | 8  |
|   | 3.2 | Artificial diffusion for purely acoustic flows<br>                      | 9  |
|   | 3.3 | Artificial diffusion for mixed convective-acoustic flows<br>            | 10 |
|   | 3.4 | Adaptive schemes                                                        | 11 |
| 4 |     | Analysis of the continuous Euler equations with artificial diffusion    | 11 |
|   | 4.1 | Expansion of the Euler equations with convective diffusion              | 12 |
|   | 4.2 | Expansion of the Euler equations with acoustic diffusion                | 13 |
|   | 4.3 | Expansion of the Euler equations with mixed diffusion                   | 13 |
|   | 4.4 | Spectral radius estimates<br>                                           | 14 |
| 5 |     | Analysis of the discrete Euler equations with artificial diffusion      | 15 |
|   | 5.1 | A general form for low Mach number finite-volume schemes<br>            | 15 |
|   | 5.2 | Comparison to previous work<br>                                         | 16 |
|   |     | 5.2.1<br>Previous guidelines                                            | 16 |
|   |     | 5.2.2<br>Classification of existing Roe-type schemes<br>                | 17 |
|   |     | 5.2.3<br>Behaviour of the off-diagonal diffusion<br>                    | 18 |
|   | 5.3 | Expansion of the discrete Euler equations with artificial diffusion<br> | 18 |
|   |     | 5.3.1<br>Expansion with convective diffusion<br>                        | 18 |
|   |     | 5.3.2<br>Expansion with acoustic diffusion                              | 21 |
|   |     | 5.3.3<br>Expansion with mixed diffusion                                 | 21 |
|   | 5.4 | Stability of the discrete scheme<br>                                    | 23 |
|   |     | 5.4.1<br>Eigenvalues of the artificial diffusion Jacobian               | 23 |
|   |     | 5.4.2<br>von Neumann symbols of the first order scheme                  | 23 |
| 6 |     | Numerical examples                                                      | 26 |
|   | 6.1 | One dimensional examples                                                | 26 |
|   |     | 6.1.1<br>Isolated soundwave                                             | 26 |
|   |     | 6.1.2<br>Low Mach number shocktube                                      | 27 |
|   | 6.2 | Two dimensional examples                                                | 28 |
|   |     | 6.2.1<br>Circular cylinder<br>                                          | 28 |
|   |     | 6.2.2<br>Soundwave-vortex interaction                                   | 28 |
| 7 |     | Conclusions                                                             | 32 |
|   |     | Appendices                                                              | 37 |
| A |     | Transonic schemes at low Mach number                                    | 37 |
| B |     | Non-dimensionalised artificial diffusion for finite-volume schemes      | 37 |
|   |     |                                                                         |    |

# Introduction

Low Mach number flows with non-zero divergence - distinct from properly incompressible divergence-free flows - are of interest in a wide range of applications. Flows with significant heat transfer or external body forces experience compressibility and buoyancy effects, for example environmental flows [\[7\]](#page-32-0) and cooling systems for electronic devices and turbomachinery [\[63,](#page-35-0) [11\]](#page-32-1). Low Mach number acoustic phenomena are also of interest [\[28\]](#page-33-0), for example noise prediction for wind turbines or aeroplanes during take-off and landing. Some flows experience both of these effects, such as low Mach number combustion, where heatrelease induced compressibility and acoustic interactions both play important roles in governing the flow evolution [\[62,](#page-35-1) [13\]](#page-32-2). Some flows contain regions of both low and high Mach number in the same domain, for example aerofoils at high angle of attack where the background flow has low Mach number but regions of higher Mach numbers occur in the vicinity of the aerofoil [\[50\]](#page-34-0). To properly simulate any of these cases requires a numerical scheme which is accurate at low Mach number.

There are three common approaches for designing numerical methods for low Mach number flow. The first uses asymptotic analysis and other assumptions to simplify the full governing equations (either Euler or Navier-Stokes) into a reduced set of equations which can then be solved using specialised solution strategies [\[48\]](#page-34-1), such as the Boussinesq buoyancy equations [\[51,](#page-34-2) [43\]](#page-34-3) or multiple pressure variable (MPV) methods [\[31,](#page-33-1) [57\]](#page-35-2). This approach leads to schemes which are highly specialised and often very efficient, however they are restricted to the regimes where the simplifying assumptions hold. The second approach extends pressure based methods for incompressible flows - such as SIMPLE or fractional step - to the low Mach number regime. These methods have proved particularly useful for low Mach number aeroacoustics [\[77,](#page-36-2) [79\]](#page-36-3). These methods can leverage the large existing research and code bases for incompressible methods, however they may only handle mixed Mach number flow with moderate, not high Mach number. The third approach is the one considered in this paper: extending density based methods for compressible flows into the low Mach number regime. These methods face efficiency and accuracy problems in this regime, but Turkel 1993 [\[70\]](#page-35-3) states three reasons justifying the effort placed into overcoming these issues which are still valid today. Firstly, these methods can handle mixed Mach number flows, a major advantage over the other approaches. Secondly, these methods handle very naturally flows with large thermal effects which cause large density variations and strong coupling of the energy equation. Lastly, there is a large body of research and software built upon these methods which can be taken advantage of, including effective acceleration techniques and handling of complex geometries [\[3\]](#page-32-3).

Collocated density based compressible numerical methods face two substantial problems in the low Mach number regime: efficiency and accuracy. The efficiency problem is easily understood by considering the wavespeeds of the Euler equations: as M = u/a → 0, the condition number (u + a)/u → 1/M, and the convective wavespeed u evolves very slowly relative to the acoustic wavespeed a. If the step size of an iterative scheme (for time-accurate evolution or for steady-state convergence) is chosen according to the acoustic wavespeed, then a large number of steps are required to resolve the convective features. The common solutions to this problem are semi-implicit methods which remove the acoustic CFL limit [\[77\]](#page-36-2), or low Mach preconditioners which improve the condition number and restore efficiency [\[71,](#page-35-4) [69\]](#page-35-5). The design and analysis of low Mach number preconditioning is a substantial research topic in itself, but is not the focus of the current work - although the impact of the artificial diffusion on efficiency will be discussed. See Turkel 1999 [\[69\]](#page-35-5) for a review of low Mach number preconditioning.

The accuracy problem is less easily demonstrated, and is the main focus of this paper. Using asymptotic analysis, three limits for the Euler equations can be found at low Mach number. The limit most often of interest is the convective single-scale limit, where only convective features exist, and solutions approach the incompressible limit as M → 0 [\[30\]](#page-33-2). At the acoustic single-scale limit, only acoustic features exist, and the solutions approach those of the wave equations for linear acoustics. Finally, a single-space-scale multipletime-scale (or multiple-space-scale single-time-scale) analysis produces a mixed convective-acoustic limit with convective features on a slow timescale (short space-scale) co-existing with acoustic features on a fast timescale (long space-scale) [\[48\]](#page-34-1). The accuracy problem describes the inability of some schemes to produce solutions which match those of the desired limit (usually the convective limit).

Volpe 1993 [\[76\]](#page-36-4) and Godfrey et al. 1993 [\[21\]](#page-33-3) showed that a na¨ıve application of transonic numerical schemes to flows at the convective low Mach number limit can produce inaccurate, physically inconsistent results. Godfrey et al. showed that calculating the artificial diffusion based on the preconditioned system instead of the original Euler system produces vastly improved results. Since then, many papers have been published presenting different methods for modifying the artificial diffusion to achieve accurate low Mach number results. These use a variety of strategies, building on central schemes [\[70,](#page-35-3) [74,](#page-35-6) [73,](#page-35-7) [75\]](#page-36-5), flux-difference [\[21,](#page-33-3) [78,](#page-36-6) [26,](#page-33-4) [25,](#page-33-5) [23,](#page-33-6) [44,](#page-34-4) [65,](#page-35-8) [15,](#page-32-4) [53,](#page-34-5) [49,](#page-34-6) [50,](#page-34-0) [5,](#page-32-5) [32,](#page-33-7) [35,](#page-34-7) [34,](#page-33-8) [33,](#page-33-9) [20\]](#page-33-10), and flux-vector splitting [\[18,](#page-33-11) [39,](#page-34-8) [37,](#page-34-9) [61,](#page-35-9) [58,](#page-35-10) [36\]](#page-34-10) methods. The majority of studies consider only the convective limit, although a number also address flows with acoustic features [\[74,](#page-35-6) [73,](#page-35-7) [75,](#page-36-5) [50,](#page-34-0) [5,](#page-32-5) [61,](#page-35-9) [58,](#page-35-10) [46\]](#page-34-11). There have been a number of excellent review and analysis articles published on artificial diffusion at low Mach number, covering the discrete equations of Roe-type schemes at the different limits [\[26,](#page-33-4) [25,](#page-33-5) [23,](#page-33-6) [33,](#page-33-9) [5\]](#page-32-5), the modified equations [\[15\]](#page-32-4), and the relationship to preconditioning [\[46,](#page-34-11) [74,](#page-35-6) [73,](#page-35-7) [75\]](#page-36-5). We cover the literature on low Mach number artificial diffusion schemes and their analysis in more detail in section [2.](#page-5-0)

In this paper we review the role of the artificial diffusion in the class of collocated, density based schemes at low Mach number. We have endeavoured to find a formulation of the problem which is as simple as possible whilst still demonstrating the most important findings across this research area. We hope that this approach will help highlight the root causes of these findings, and enable it to act as an introduction for those with little experience in this field.

To achieve this, we consider all three low Mach number limits - the purely convective, purely acoustic, and mixed convective-acoustic - which are introduced in section [1.](#page-3-0) After this, we can review the previous literature in section [2.](#page-5-0) Previous studies usually begin by showing that a classical transonic scheme is inaccurate for the convective limit, then proceed to 'fix' this scheme. Instead, in section [3](#page-6-0) we use the continuous modified equations in the entropy variables to design artificial diffusion schemes which naturally match each of the three limits independently of the specific discretisation. In section [4](#page-10-1) we apply singleand multiple-scale asymptotic expansions to the modified equations of each scheme. In section [5](#page-14-0) we transform the artificial diffusion to the conservative variables to find the equivalent finite-volume flux functions. We compare to previous works, and proceed to repeat the asymptotic expansions on the discrete equations. Finally, in section [6](#page-25-0) we show a series of numerical examples which verify the results of the previous sections. In a sequel paper[†](#page-0-0) we extend this to three families of convection-pressure flux splittings: AUSM [\[40\]](#page-34-12), Zha-Bilgen [\[80\]](#page-36-7) and Toro-Vasquez [\[68\]](#page-35-11).

# 1 Asymptotic expansion of the Euler equations at low Mach number

We review the theoretical results for the asymptotic solutions of the Euler equations with an ideal gas at low Mach number using singleand multiple-timescale analysis. We present only the most relevant results, and work in the entropy variables w = (p, u, v, s) T . For a more extensive analysis in the conserved variables, see M¨uller 1999 [\[48\]](#page-34-1), and for a multiple space scale analysis see [\[31\]](#page-33-1). First, all variables are non-dimensionalised:

$$\rho = \frac{\tilde{\rho}}{\rho_{\infty}}, \quad \underline{u} = \frac{\underline{\tilde{u}}}{u_{\infty}}, \quad \underline{x} = \frac{\underline{\tilde{x}}}{L_{\infty}}, \quad E = \frac{\tilde{E}}{p_{\infty}/\rho_{\infty}}, \quad H = \frac{\tilde{H}}{p_{\infty}/\rho_{\infty}}, \quad p = \frac{\tilde{p}}{p_{\infty}}, \quad t = \frac{\tilde{t}}{L_{\infty}/u_{\infty}}$$
(1)

where tildes indicate local dimensional quantities, ∞ indicates reference dimensional quantities, and underlining indicates vector-valued quantities. E and H are the specific total energy and enthalpy respectively. The Euler equations in entropy variables become:

$$\partial_t p + \underline{u} \cdot \nabla p + \gamma p \nabla \cdot \underline{u} = 0 \tag{2a}$$

$$\rho \partial_t \underline{u} + M^{-2} \nabla p + \rho \underline{u} \cdot \nabla \underline{u} = 0$$
 (2b)

$$\partial_t s + \underline{u} \cdot \nabla s = 0 \tag{2c}$$

Where M = <sup>√</sup>γu<sup>∞</sup> a<sup>∞</sup> is the reference Mach number. The relation γp = ρa<sup>2</sup> has been used in the 3rd term in [\(2a\)](#page-3-2), which is only valid for a perfect gas or barotropic gases which are polytropic (including isothermal or isentropic gases). It is not valid for a general barotropic gas, although the same results can still be obtained in this case. The entropy equation [\(2c\)](#page-3-3) is not required for barotropic gases.

## 1.1 Single timescale convective limit

Using M as a small parameter, we treat the system [\(2\)](#page-3-4) as a perturbation problem and expand all variables as power series of M:

$$\psi(\underline{x}, t, M) = \psi^{(0)}(\underline{x}, t) + M\psi^{(1)}(\underline{x}, t) + M^2\psi^{(2)}(\underline{x}, t) + \mathcal{O}(M^3)$$
(3)

The expansions [\(3\)](#page-3-5) are inserted into [\(2\)](#page-3-4) and terms are grouped by powers of the parameter M. The lowest order terms from the velocity equations are O(M−<sup>2</sup> ), O(M−<sup>1</sup> ) and O(M<sup>0</sup> ) due to the M−<sup>2</sup> coefficient on the pressure gradient:

$$\nabla p^{(0)} = 0 \tag{4a}$$

$$\nabla p^{(1)} = 0 \tag{4b}$$

$$\rho^{(0)}\partial_t \underline{u}^{(0)} + \nabla p^{(2)} + \rho u^{(0)} \cdot \nabla \underline{u}^{(0)} = 0$$

$$\tag{4c}$$

<sup>†</sup>Artificial diffusion for convective and acoustic low Mach number flows II: Application to Liou-Steffen, Zha-Bilgen and Toro-Vasquez flux splitting schemes, J. Hope-Collins & L. di Mare, in progress.

Relations (4a) and (4b) mean that the zeroth and first order pressure terms vary only in time. If  $p^{(1)}$  is zero at all points on the boundary at all times then the pressure expansion (3) can be replaced with:

$$p(\underline{x}, t, M) = p^{(0)}(t) + M^2 p^{(2)}(\underline{x}, t) + \mathcal{O}(M^3)$$

The spatial variations of the non-dimensional convective pressure are  $\mathcal{O}(M^2)$ , so the dimensional pressure variations are  $\mathcal{O}(\rho u^2)$ , i.e. independent of the Mach number. For a barotropic gas, this implies that the density also has  $\mathcal{O}(M^2)$  spatial variations. The last relation (4c) is the evolution equation for the convective velocity. Expanding the pressure equation (2a) and making use of relations (4a) and (4b), we find the following  $\mathcal{O}(M^0)$  and  $\mathcal{O}(M^1)$  relations:

$$d_t p^{(0)} + \gamma p^{(0)} \nabla \cdot u^{(0)} = 0 \tag{5a}$$

$$d_t p^{(1)} + \gamma (p \nabla \cdot u)^{(1)} = 0 \tag{5b}$$

The relations (5a) and (5b) imply that the divergence of the zeroth and first order velocities are spatially uniform and react everywhere instantaneously to temporal variations in the background pressure. Lastly, we expand the entropy equation. Every order n of the expansion is identical to the original equation, showing that the entropy is simply convected.

$$\partial_t s^{(n)} + (\underline{u} \cdot \nabla s)^{(n)} = 0 \tag{6}$$

Note that there are no acoustic effects in this single-scale expansion, which should not be surprising given our choice of the convective timescale  $L_{\infty}/u_{\infty}$  to non-dimensionalise the time derivatives in equation (2).

## 1.2 Two timescale convective-acoustic limit

A two-timescale, one space scale asymptotic analysis can be used to include acoustic effects. Defining an additional non-dimensional time  $\tau$  using the acoustic speed:

$$\tau = \frac{\bar{t}}{L_{\infty}/a_{\infty}} = \frac{t}{M} \tag{7}$$

leads to the power series expansion:

$$\psi(\underline{x}, t, M) = \psi^{(0)}(\underline{x}, t, \tau) + M\psi^{(1)}(\underline{x}, t, \tau) + M^2\psi^{(2)}(\underline{x}, t, \tau) + \mathcal{O}(M^3)$$
(8)

The time derivatives at constant x and M are now:

$$\partial_t \psi \Big|_{x,M} = \left(\partial_t + \frac{1}{M} \partial_\tau\right) \psi \tag{9}$$

As for the convective limit, we start with the three lowest order relations from the velocity equation:

$$\nabla p^{(0)} = 0 \tag{10a}$$

$$\rho^{(0)}\partial_{\tau}u^{(0)} + \nabla p^{(1)} = 0 \tag{10b}$$

$$\partial_{\tau} \rho u^{(1)} + \rho^{(0)} \partial_{t} \underline{u}^{(0)} + \nabla p^{(2)} + \rho u^{(0)} \cdot \nabla \underline{u}^{(0)} = 0$$
(10c)

The zeroth order pressure is again constant in space, however the same is no longer true for the first order pressure term, which now varies with zeroth order velocity fluctuations on the acoustic timescale. The second order pressure is still the relevant pressure for the zeroth order velocity fluctuations on the convective timescale (10c). Next, the pressure equation is expanded using (8) and making use of the relation (10a):

$$\partial_{\tau} p^{(0)} = 0 \tag{11a}$$

$$\partial_{\tau} p^{(1)} + d_t p^{(0)} + \gamma p^{(0)} \nabla \cdot \underline{u}^{(0)} = 0$$
(11b)

Relations (10a) and (11a) imply that the background pressure varies only on the convective timescale. Relation (11b) shows that the leading order velocity divergence now has spatial variations on the acoustic timescale related to first order pressure variations. In fact, relations (10b) and (11b) are the equations for linear acoustics at low Mach number, with a source term  $d_t p^{(0)}$  (see [48] for more details). The relation (10c) is again the equation for the convective velocity variations. The correct expansion for the pressure at the two-timescale limit is (8), except the zeroth order term which becomes  $p^{(0)}(\underline{x}, t, \tau) = p^{(0)}(t)$ . Note that a purely acoustic limit of (2) can be found with a single-timescale expansion with the acoustic timescale  $\tau$ , which results in the relations (10a,10b,11a), and (11b) without the  $d_t p^{(0)}$  term. We now expand the entropy equation using the two timescale expansion. This time we find:

$$\partial_{\tau} s^{(0)} = 0 \tag{12a}$$

$$\partial_{\tau} s^{(n+1)} + \partial_{t} s^{(n)} + (u \cdot \nabla s)^{(n)} = 0, \quad n > 0$$
 (12b)

From which we see that the zeroth order entropy is constant on the acoustic timescale, which is consistent with a leading order approximation of isentropic sound waves.

To summarise the important points of this section, we have briefly covered two asymptotic limits of the inviscid homogeneous Euler equations at low Mach number. Firstly, a convective limit with spatially uniform velocity divergence and O(M<sup>2</sup> ) spatial pressure variations associated with convective features on the slow timescale t. Secondly, a mixed convective-acoustic limit which has spatially varying divergence and O(M) spatial pressure variations associated with acoustic features on the fast timescale τ in addition to the convective features on the slow timescale. A single-scale acoustic limit can be constructed using only the fast timescale τ , which contains only the acoustic variations on the fast timescale.

# 2 Previous work

As highlighted in the introduction, Godfrey et al. showed in 1993 that calculating the artificial diffusion for a flux-difference-splitting scheme based on the preconditioned system produced accurate results for steady convective flows, whereas calculating the diffusion from the unmodified system produced physically inconsistent results [\[21\]](#page-33-3). This approach was adopted by a number of other researchers for both fluxdifference-splitting methods [\[78,](#page-36-6) [26\]](#page-33-4) and central difference methods [\[69,](#page-35-5) [9,](#page-32-6) [74\]](#page-35-6). Turkel et al 1994 and Turkel 1999 [\[71,](#page-35-4) [69\]](#page-35-5) used the modified equations and the known low Mach number scalings for the convective limit to show that the artificial diffusion of a standard transonic scheme is poorly balanced for this limit, having velocity diffusion which is too high and pressure diffusion which is too low. They also showed that the preconditioned artificial diffusion rectifies both of these issues, having pressure and velocity diffusion terms which are larger and smaller by a factor of M respectively compared to the unpreconditioned scheme. We note that artificial diffusion is not the only approach to achieving stability at low Mach number. For example the kinetic energy consistent scheme of Subbareddy & Candler 2009 [\[64\]](#page-35-12) is accurate and stable at low Mach number without artificial diffusion. However, artificial diffusion is the predominant method for stabilising collocated schemes in many applications.

Guillard & Viozat 1999 [\[26\]](#page-33-4) used asymptotic expansions of the discrete equations to show that the unmodified Roe scheme [\[56\]](#page-35-13) produces discrete solutions which do not match the convective asymptotic solutions of the exact Euler Equations, in that they contain spurious spatial oscillations of the first order pressure p (1). In a later paper [\[25\]](#page-33-5) they showed that this is because the diffusion of the unmodified scheme is derived from the acoustic component of the Euler equations, so produces a residual in the acoustic pressure p (1) even for initial conditions containing purely convective variations. Asymptotic expansion of the discrete equations of the preconditioned Roe scheme shows that this scheme produces discrete solutions which do match the exact asymptotic convective limit - the smaller velocity diffusion preventing the creation of spurious p (1) modes, and the larger pressure diffusion preventing chequerboard modes on p (1) .

The preconditioned diffusion proved to be popular and successful, especially because it can be used in combination with preconditioned timestepping, resulting in a scheme which is stable under a convective CFL condition, eliminating the efficiency problem. However, Birken & Meister 2005 [\[2\]](#page-32-7) showed that for time accurate timestepping the preconditioned diffusion has a timestep limit of ∆t ∼ O(M<sup>2</sup> ) for stability of an explicit scheme, compared to ∆t ∼ O(M) for the unmodified scheme, so requires implicit timestepping to be practical for time-accurate simulations.

Following the finding of Birken & Meister, a number of papers proposed a different approach to creating a diffusion scheme which is accurate at low Mach number. Thornber et al. 2008 [\[66\]](#page-35-14) showed that the inaccuracies of standard Godunov schemes at low Mach number are linked to spurious entropy generation by the velocity diffusion, and subsequently proposed a modification to Roe's flux which reduced the velocity diffusion by a factor of M, resulting in accurate solutions at low Mach number [\[67\]](#page-35-15). Other methods of achieving this reduction in the velocity diffusion have been proposed, including those by Thornber et al. [\[65\]](#page-35-8), Dellacherie [\[15\]](#page-32-4), Rieper (LMRoe) [\[53\]](#page-34-5), and Miczek et al. [\[47\]](#page-34-13). All of these methods retain the ∆t ∼ O(M) stability limit of the original scheme, but with improved low-Mach accuracy.

Dellacherie 2010 and Dellacherie et al 2016 [\[15,](#page-32-4) [14\]](#page-32-8) used the modified equations for the linear acoustic waves with artificial diffusion to rigorously show that reducing the velocity diffusion of a standard Godunov scheme by at least a factor of M will prevent the production of spurious acoustic p (1) modes and ensure accuracy for the convective low Mach number limit, provided the initial conditions are wellprepared[†](#page-0-0) . This shows that the increased pressure diffusion in the preconditioned artificial diffusion is in fact not necessary for low-Mach accuracy. However, it has been shown [\[15,](#page-32-4) [23\]](#page-33-6) that the increased pressure diffusion introduces a Brezzi-Pitk¨aranta type stabilisation [\[4\]](#page-32-9), which prevents pressure chequerboard instabilities on collocated schemes, and has previously been applied to finite-element [\[27\]](#page-33-12) and finite-volume

<sup>†</sup>Well-prepared initial conditions are solutions which are close to the solution space of the convective limit i.e. contain vanishingly small acoustic components. See Schochet [\[59\]](#page-35-16) and Dellacherie et al [\[15,](#page-32-4) [14\]](#page-32-8) for a formal definition.

A third approach, the all-speed Roe scheme, introduced by Li & Gu 2008 [32, 35] reduces not only the velocity diffusion by a factor of M, but also the pressure diffusion compared to the standard Roe scheme, so that the diffusion is essentially a diagonal upwinding on the convective velocity scale. The reduction in velocity diffusion means that this scheme is accurate for convective low Mach number flow, but is susceptible to severe chequerboard instabilities. Li & Gu resolve this by reintroducing a pressure diffusion term into the physical mass flux using a momentum interpolation method similar to that proposed by Rhie & Chow [52], and used by Mary and Sagaut [44] and in AUSM schemes [40, 38, 37]. The scheme was later extended to unsteady flows by adding a timestep dependence to the pressure diffusion [34].

The discrete forms of these three approaches are reviewed and compared by Li & Gu 2013 [33] and Guillard & Nkonga 2017 [23]. Li & Gu compare the scaling of the coefficients of the pressure and velocity diffusion terms, and propose a set of guidelines on these scalings for accuracy and stability. Guillard & Nkonga review the origin of the accuracy problem with respect to the multiple possible low Mach number limits, and also highlight the dependence on grid type of the various low-Mach schemes. For a first order Roe scheme on a simplex mesh (triangles in 2D or tetrahedrons in 3D), the velocity degrees of freedom are reduced such that the jumps in the normal velocity over cell interfaces vanish. This eliminates the velocity diffusion, meaning that the unmodified Roe scheme actually provides accurate results for this cell geometry, despite its inaccuracy on other meshes. See [54, 55, 24, 17] and section 3.3 of [23] for details of this particular behaviour.

The literature covered so far has focused on the convective limit. A number of papers have also covered schemes for flows with acoustic effects. In a series of papers, Venkateswaran and Merkle showed that preconditioned dual-time schemes are ill conditioned at the acoustic limit and display poor convergence [72], and that the preconditioned diffusion is inaccurate for flow with acoustics, having excessive pressure dissipation, whereas the unmodified scheme is both efficient and accurate for purely acoustic flow [74, 46]. They developed an 'enhanced' diffusion scheme having the same pressure diffusion as the unmodified scheme but reduced velocity diffusion, as in the preconditioned scheme [73, 75]. This diffusion scheme was shown to produce accurate results with reasonable convergence for both convective and acoustic flows. The later low-Mach fix of Thornber/Dellacherie/Rieper [67, 15, 53] is in a sense equivalent to the earlier 'enhanced' scheme, although appears to have been developed independently. Thornber and Rieper present numerical evidence that their schemes are accurate for 1D acoustics, although their analyses do not cover acoustic effects.

The enhanced/Low-Mach fix scheme is generally suitable for both convective and acoustic flows. However, Potsdam et al. 2007 [50] and Sachdev et al. 2012 [58] showed numerically that under certain conditions, it is susceptible to slight instabilities on the convective pressure field and acoustic velocity field (the instability of the acoustic velocity grid-mode was also demonstrated and analysed in [16]). By introducing timestep dependence into the diffusion formulation, they developed adaptive schemes which vary between different diffusion schemes depending on whether acoustic waves are resolved. This adaptive methodology was shown to maintain accuracy over a range of Strouhal numbers, and is applicable to Roe-type [50, 6] and AUSM type [58] schemes.

Whereas the adaptive methodology aims to return to the preconditioned or unmodified scheme in situations where the enhanced/LMRoe scheme may produce oscillatory solutions, Bruel et al. 2019 [5] tried to modify the LMRoe scheme to eliminate the acoustic velocity instability and the related degradation in the CFL condition ( $\sigma < \frac{1}{2}$ ) of this scheme. Using discrete multiple-scale asymptotic analysis, they confirmed the suitability of the Roe, preconditioned Roe and LMRoe schemes for purely acoustic, purely convective, and either acoustic or convective flows respectively. By reintroducing off-diagonal diffusion terms into LMRoe the acoustic instabilities and CFL degradation were eliminated, although the final scheme produces symmetry-breaking solutions for some convective flows which the authors attribute to the scheme not maintaining Galilean invariance.

## 3 Design of artificial diffusion at low Mach number

In this section, we find the appropriate asymptotic scaling for the artificial diffusion terms of a numerical scheme, such that the scheme provides accurate results at low Mach number. We consider each of the three low Mach number regimes in turn. The diffusion scaling for purely convective flow is well-known, and its form was derived in Turkel et al 1994 and Turkel 1999 [71, 69]. We repeat this derivation here for completeness, before reapplying Turkel's method to derive the diffusion scaling for purely acoustic flow. By examining the two schemes, we can then design a diffusion scaling suitable for mixed convective-acoustic flow.

In equation (13) we augment the continuous x-split 2-dimensional Euler equations with artificial

diffusion terms with a form that mirrors that of the flux Jacobian, which maximises the degrees of freedom without increasing the coupling between equations. The accuracy problem for the convective limit occurs only for multi-dimensional flow [15]. However, the vast majority of schemes in the class under consideration use some form of dimension-splitting (e.g. finite-difference derivatives along grid lines, or finite-volume interface fluxes over cell faces), and the dimension-split form is simpler than the full multi-dimensional form whilst still displaying the most important low-Mach behaviours, which justifies its use over a truly multi-dimensional analysis here. The dimension-split results can be readily extended to the y-split part of the 2D equations, to the 3D equations, and to the non-dimension-split equations. For analysis of the multi-dimensional modified equations see [15], and for a recent example of a truly multi-dimensional low Mach number scheme see [1].

$$\partial_t p + u \partial_x p + \gamma p \partial_x u = A_{11} \partial_{xx} p + A_{12} \partial_{xx} u \tag{13a}$$

$$\rho \partial_t u + M^{-2} \partial_x p + \rho u \partial_x u = A_{21} \partial_{xx} p + A_{22} \partial_{xx} u \tag{13b}$$

$$\partial_t v + u \partial_x v = A_{33} \partial_{xx} v \tag{13c}$$

$$\partial_t s + u \partial_x s = A_{44} \partial_{xx} s \tag{13d}$$

We have written the artificial diffusion as linear second order terms for simplicity, but any consistent diffusive term is acceptable<sup>†</sup>. The results of the analysis still hold if, for example, we assume that  $A\partial_{xx}\psi \sim \mathcal{O}(M^n)$  implies both  $\partial_x(A\partial_x\psi) \sim \mathcal{O}(M^n)$  and  $A\partial_x^4\psi \sim \mathcal{O}(M^n)$  for non-linear or fourth order diffusion terms respectively. This underlines the fact that we are interested in the limit of vanishing Mach number at a fixed, finite mesh spacing. If we also took the limit of vanishing mesh spacing, then the diffusion terms on the right hand side of the system (13) would disappear, as is necessary for a consistent numerical scheme.

The aim of this section is now to derive the appropriate asymptotic scaling for the coefficients  $A_{ij}$ , such that the solutions to the modified system (13) have the same scaling behaviour as solutions of the original Euler equations at each low Mach number regime. Turkel's method for deriving these scalings for purely convective flow (which we will also use for purely acoustic flow) is as follows:

- 1. Find the limiting form of the left-hand side  $(\mathcal{L})$  of equations (13) by retaining only the largest term(s) as  $M \to 0$ .
- 2. Force the artificial diffusion terms to be retained in the limit by choosing the order of the coefficients  $A_{ij}$  such that all terms on the right-hand side  $(\mathcal{R})$  are the same order as the largest term(s) in  $\mathcal{L}$ , i.e.  $\mathcal{R} \sim \mathcal{O}(\mathcal{L})^{\ddagger}$ .

Because the mesh spacing is fixed, we can assume that  $\partial_x \psi \sim \mathcal{O}(M^n) \implies \partial_{xx} \psi \sim \mathcal{O}(M^n)$  and the scalings found in section 1 can be used for both  $\mathcal{L}$  and  $\mathcal{R}$  in step 1. These scalings are summarised in table 1. We can immediately find the artificial diffusion scaling for the linear vorticity and entropy fields (13c,13d). For these equations,  $\mathcal{L} \sim \mathcal{O}(M^0)$  in both the singleand multiple-timescale limits, so for  $\mathcal{R} \sim \mathcal{O}(\mathcal{L})$  as  $M \to 0$ , we require:

$$A_{33} \sim \mathcal{O}(M^0), \quad A_{44} \sim \mathcal{O}(M^0)$$
 (14)

This is consistent with a standard upwind scheme with a diffusion coefficient on the order of the advecting velocity u, which is appropriate given that (13c,13d) have the form of the linear scalar advection equation.

The pressure and velocity equations (13a,13b) form a coupled subsystem which is the distinguishing feature between the different low Mach number regimes and, by extension, between the different schemes. The rest of this section is concerned with finding the correct scaling for the artificial diffusion coefficients for this subsystem only, assuming the scaling (14) for the linear fields.

## 3.1 Artificial diffusion for purely convective flows

First, we find the required scaling of the coefficients  $A^c_{ij}$ , i, j = 1, 2 for purely convective flow. Using the scalings in table 1, the largest terms in  $\mathcal{L}$  for (13a, 13b) are  $\mathcal{O}(M^0)$ . The diagonal coefficients  $A_{11}$  and  $A_{22}$  should be  $\mathcal{O}(M^{-2})$  and  $\mathcal{O}(M^0)$  respectively for the diagonal artificial diffusion terms to remain in the limit equations. By the same argument, the off-diagonal coefficients  $A_{12}$  and  $A_{21}$  should be  $\mathcal{O}(M^0)$  and  $\mathcal{O}(M^{-2})$  respectively<sup>§</sup>. Collecting these results, the convective scaling of the diffusion matrix is:

$$\underline{\underline{A}}^c \sim \mathcal{O} \begin{pmatrix} M^{-2} & M^0 \\ M^{-2} & M^0 \end{pmatrix} \tag{15}$$

<sup>&</sup>lt;sup>†</sup>A consistent diffusion term is proportional to  $\Delta x^n$  for some positive n, where  $\Delta x$  is the grid spacing, so vanishes as  $\Delta x \to 0$ . <sup>‡</sup>Strictly, this should be  $\mathcal{R} \sim \Theta(\mathcal{L})$ , because if  $\mathcal{R} \sim o(\mathcal{L})$  then  $\mathcal{R} \sim \mathcal{O}(\mathcal{L})$  still holds, but the diffusion  $\mathcal{R}$  would vanish asymptotically. We will continue to use big-O instead of big-theta notation to remain consistent with the rest of the literature, with the understanding that a stricter interpretation is required for (at least the diagonal) diffusion coefficients.

<sup>§</sup>It is acceptable for the off-diagonal terms to be  $o(\mathcal{L})$  and vanish in the limit, however the diagonal terms should be  $\Theta(\mathcal{L})$  for stability.

|                | Convective                                                         | Acoustic                                           |  |
|----------------|--------------------------------------------------------------------|----------------------------------------------------|--|
| p              | $p^{(0)} \sim$                                                     | $\mathcal{O}(M^0)$                                 |  |
| u              | $u^{(0)} \sim \mathcal{O}(M^0)$ $\rho^{(0)} \sim \mathcal{O}(M^0)$ |                                                    |  |
| ρ              |                                                                    |                                                    |  |
| $\partial_x p$ | $\partial_x p^{(2)} \sim \mathcal{O}(M^2)$                         | $\partial_x p^{(1)} \sim \mathcal{O}(M^1)$         |  |
| $\partial_x u$ | $\partial_x u^{(0)}$                                               | $\sim \mathcal{O}(M^0)$                            |  |
| $\partial_x v$ |                                                                    | $\sim \mathcal{O}(M^0)$                            |  |
| $\partial_x s$ | $\partial_x s^{(0)} \sim$                                          | $\sim \mathcal{O}(M^0)$                            |  |
| $\partial_t p$ | $\partial_t p^{(0)} \sim \mathcal{O}(M^0)$                         | $\partial_{\tau} p^{(1)} \sim \mathcal{O}(M^0)$    |  |
| $\partial_t u$ | - ( )                                                              | $\partial_{\tau} u^{(0)} \sim \mathcal{O}(M^{-1})$ |  |
| $\partial_t v$ | $\partial_t v^{(0)} \sim$                                          | $\sim \mathcal{O}(M^0)$                            |  |
| $\partial_t s$ | $\partial_t s^{(0)} \sim \mathcal{O}(M^0)$                         |                                                    |  |

Table 1: Low Mach number scaling of the various terms in equations (13a, 13b) for purely convective or acoustic variation. Note that the time derivatives scaling for the acoustic variations of p and u make use of equation (9).

Retaining only the leading order  $\mathcal{O}(M^0)$  terms from the modified equations (13a,13b), the limit equations for convective flow variations and convective diffusion scaling  $\operatorname{are}^{\dagger}$ :

$$\partial_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-2)} \partial_{xx} p^{(2)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(16a)

$$\rho^{(0)}\partial_t u^{(0)} + \partial_x p^{(2)} + \rho u^{(0)}\partial_x u^{(0)} = A_{21}^{(-2)}\partial_{xx} p^{(2)} + A_{22}^{(0)}\partial_{xx} u^{(0)}$$
(16b)

where  $A_{ij}^{(n)} \Longrightarrow A_{ij} \sim \Theta(M^n)$ . The limit equations (16) are identical to the  $\mathcal{O}(M^0)$  pressure and velocity relations (5a,4c) from the single-scale asymptotic expansion, with the addition of artificial diffusion terms. They also closely resemble the governing equations of Chorin's Artificial Compressibility Method [10], which was influential in much of the early work on low Mach number preconditioning [70, 71, 78]. As is appropriate for close-to-incompressible flow, the magnitude of both limit equations becomes independent of the Mach number, and at steady-state the velocity divergence will approach zero (up to the value of the diffusion terms) according to the pressure equation (16a).

#### 3.2Artificial diffusion for purely acoustic flows

Next we consider flow with purely acoustic variation, repeating the process above to obtain the correct scaling of the artificial diffusion  $\underline{A}^a$  in this regime. Using the acoustic flow variations from table 1, the largest terms in  $\mathcal{L}$  for (13a,13b) are  $\mathcal{O}(M^0)$  and  $\mathcal{O}(M^{-1})$  respectively. For all artificial diffusion terms to be retained in the limit, we require:

$$\underline{\underline{\underline{A}}}^a \sim \mathcal{O} \begin{pmatrix} M^{-1} & M^0 \\ M^{-2} & M^{-1} \end{pmatrix} \tag{17}$$

The limit equations using this diffusion scaling and acoustic flow variations are:

$$\partial_{\tau} p^{(1)} + \gamma p^{(0)} \partial_{x} u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(18a)

$$\partial_{\tau} p^{(1)} + \gamma p^{(0)} \partial_{x} u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$

$$\rho^{(0)} \partial_{\tau} u^{(0)} + \partial_{x} p^{(1)} = A_{21}^{(-2)} \partial_{xx} p^{(1)} + A_{22}^{(-1)} \partial_{xx} u^{(0)}$$

$$(18a)$$

which are the equations governing linear acoustics at low Mach number, as we found in section 1 with the relations (11b) and (10b), plus the artificial diffusion terms. Most schemes designed for transonic flow approach a diagonal approximation to this scheme, which resembles upwinding of the two acoustic waves (see appendix A). Note that if the off-diagonal terms vanish  $(A_{12} \sim o(M^0))$  and  $A_{21} \sim o(M^{-2})$ , then the limit equations (18) are equivalent to Dellacherie's first order modified equations of a standard Godunov scheme for the linear acoustic equations (equations (61) and (62) in [15] with  $\nu_r = \nu_{u_k} = a_* \Delta x / 2M$ ), which are shown to destroy convective low Mach number accuracy. In practice, this scaling is not used for low Mach number simulations because it will create spurious acoustic waves from any non-trivial convective variations (as we will see later), and if the flow has zero (or enforced) convective variations then specialist acoustics solvers are much more suitable.

<sup>&</sup>lt;sup>†</sup>Turkel [69] uses  $p \sim \mathcal{O}(M^2)$  for both the background pressure and the pressure gradient, so the term  $\partial_t p$  in (16a) disappears from his limit equation (11a in [69]). For steady flows these are equivalent, but the time-derivative must be retained for flows where the background pressure varies in time, for example due to heat addition or non-zero net mass flow over the boundaries [48].

## 3.3 Artificial diffusion for mixed convective-acoustic flows

Acoustic variations are often the result of, or coexist with, convective phenomena, making it desirable to have a scheme with acceptable limit equations for both convective and acoustic variations. By examining the limit equations (16) and (18), we can identify two requirements for an acceptable form of the limit equations. The first is that all terms in  $\mathcal{L}$  from the relevant physical relations are retained, i.e. the momentum and divergence relations (5a,4c) for convective flow, and the linear acoustics relations (11b,10b) for acoustic flow, which is necessary for the solutions to the modified equations to have the same scaling as solutions to the original equations. The second property is that some diffusion terms are retained in  $\mathcal{R}$ , which will be necessary once we attempt to construct stable discrete schemes.

Next we will see if the limit equations for convective diffusion with acoustic variations, and for acoustic diffusion with convective variations satisfy these requirements. The limit equations for convective diffusion scaling  $\underline{A}^c$ , equation (15), and acoustic flow variations are:

$$0 = A_{11}^{(-2)} \partial_{xx} p^{(1)} \tag{19a}$$

$$\rho^{(0)}\partial_{\tau}u^{(0)} + \partial_{x}p^{(1)} = A_{21}^{(-2)}\partial_{xx}p^{(1)}$$
(19b)

The diagonal diffusion term in the pressure equation (19a) is too large  $A_{11}^{(-2)}\partial_{xx}p^{(1)} \sim \mathcal{O}(M^{-1})$ , so  $\mathcal{R} \sim \omega(\mathcal{L})$  resulting in a parabolic equation for the acoustic pressure  $p^{(1)}$ . This means that the convective scheme  $\underline{A}^c$  effectively filters acoustic variations from the solution, just as the single-timescale expansion of the original Euler equations does, making this scheme unsuitable for purely acoustic or mixed convective-acoustic flow. Conversely, the velocity equation (19b) retains all the required terms in  $\mathcal{L}$  from the acoustic velocity relation (10b), although the diagonal diffusion term vanishes asymptotically. The limit equations for the acoustic diffusion scaling  $\underline{A}^a$ , equation (17), and convective flow variations are:

$$\partial_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(20a)

$$0 = A_{22}^{(-1)} \partial_{xx} u^{(0)} \tag{20b}$$

The diagonal diffusion term in the velocity equation (20b) is too large  $A_{22}^{(-1)}\partial_{xx}u^{(0)} \sim \mathcal{O}(M^{-1})$ , so  $\mathcal{R} \sim \omega(\mathcal{L})$  resulting in a parabolic equation for the convective velocity  $u^{(0)}$ . This will smooth out any convective variations of the velocity, making this scheme unsuitable for purely convective or mixed convective-acoustic flow. On the other hand, the pressure equation (20a) retains all the required terms in  $\mathcal{L}$  from the convective pressure relation (5a), but the diagonal diffusion term is asymptotically vanishing. This over-damping of the velocity field and under-damping of the pressure field is a well-known problem with conventional (transonic) compressible schemes at the single-scale low Mach number limit [26, 25, 23].

We have shown that, unsurprisingly, neither the convective scheme  $\underline{\underline{A}}^c$  nor the acoustic scheme  $\underline{\underline{A}}^a$  completely fulfil the requirements for a mixed convective-acoustic scheme. However, the pressure limit equation (20a) with  $\underline{\underline{A}}^a$  and the velocity limit equation (19b) with  $\underline{\underline{A}}^c$  both retain the relevant terms in  $\mathcal{L}$ , even if some diffusion terms vanish. This suggests we can form a scheme for mixed flow by combining the acoustic pressure diffusion  $A_{11}^a$ ,  $A_{12}^a$  and the convective velocity diffusion  $A_{21}^c$ ,  $A_{22}^c$ :

$$\underline{\underline{A}}^m \sim \mathcal{O} \begin{pmatrix} M^{-1} & M^0 \\ M^{-2} & M^0 \end{pmatrix} \tag{21}$$

which we call the mixed scheme. We shall see during the discrete analysis in section 5 that most modern schemes use this scaling, although to the authors' knowledge the only studies which explicitly identify the use of acoustic scaling on the pressure equation and convective scaling on the velocity equation are those of Venkateswaran and co-workers [73, 75, 50, 58]. The limit equations for the mixed diffusion scaling with convective flow variations are:

$$\partial_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{12}^{(0)} \partial_{xx} u^{(0)}$$
 (22a)

$$\rho^{(0)}\partial_t u^{(0)} + \partial_x p^{(2)} + \rho u^{(0)}\partial_x u^{(0)} = A_{21}^{(-2)}\partial_{xx} p^{(2)} + A_{22}^{(0)}\partial_{xx} u^{(0)}$$
(22b)

and with acoustic flow variations are:

$$\partial_{\tau} p^{(1)} + \gamma p^{(0)} \partial_{x} u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(23a)

$$\rho^{(0)}\partial_{\tau}u^{(0)} + \partial_{x}p^{(1)} = A_{21}^{(-2)}\partial_{xx}p^{(1)}$$
(23b)

from which we see that for both convective and acoustic flow variations, all terms in  $\mathcal{L}$  are retained compared to the desired form of the limit equations (16,18). The limit equations for the convective velocity and the acoustic pressure (22b,23a) retain both artificial diffusion terms, however the convective pressure and acoustic velocity limit equations (22a, 23b) retain only the off-diagonal diffusion terms. This

may degrade the stability of the corresponding discrete scheme, especially if the off-diagonal diffusion terms vanish. The issues of degraded stability on the convective pressure and acoustic velocity were numerically demonstrated in [\[58\]](#page-35-10), and the acoustic velocity instability was analysed in [\[16,](#page-33-16) [5\]](#page-32-5); here we show that the cause of both can be identified in the continuous setting. Note that for vanishing offdiagonal terms, the limit equations [\(23\)](#page-9-9) are equivalent to Dellacherie's first order modified equations for a low Mach Godunov scheme applied to the linear acoustic equations (equations (61) and (62), or (75), in [\[15\]](#page-32-4) with ν<sup>r</sup> = a∗∆x/2M and νu<sup>k</sup> = 0), which are shown to be accurate for convective low Mach number flow.

## 3.4 Adaptive schemes

We have identified three scalings of the diffusion matrix - convective, acoustic, and mixed - each suited to different flow regimes. However, it would be useful to have a single numerical method that could select the most appropriate scaling, either so the same method could be used for multiple problems, or because different flow regimes exist in a single problem [\[50\]](#page-34-0). The major difference between the schemes is whether they allow acoustic features, so a sensible metric for which scheme to use would be whether acoustic waves can be resolved. This can be estimated by the ratio of the acoustic timescale τ and the simulation timestep ∆t:

$$\frac{\tau}{\Delta t} = \frac{L_{\infty}/a}{\Delta t} = \frac{L_{\infty}/\Delta t}{a} = M_u \tag{24}$$

When ∆t > τ acoustic waves cannot be temporally resolved and M<sup>u</sup> is small. When ∆t < τ acoustic waves can be resolved and M<sup>u</sup> is large. The parameter τ /∆t is normally presented as the unsteady Mach number Mu. This parameter was first introduced by Venkateswaran & Merkle 1995 [\[72\]](#page-35-17) to improve the conditioning of a low Mach preconditioned dual-time scheme, and has since been used to control the diffusion scaling [\[75,](#page-36-5) [50,](#page-34-0) [58,](#page-35-10) [6\]](#page-32-10). In practice, M<sup>u</sup> is bounded between M and 1 so M<sup>u</sup> → 1 as ∆t → 0 and M<sup>u</sup> → M as ∆t → ∞. The coefficients A<sup>11</sup> and A<sup>22</sup> can be made to vary between the convective scaling when M<sup>u</sup> ≈ M and the acoustic scaling when M<sup>u</sup> ≈ 1 using:

$$A_{11} \sim \mathcal{O}(M^{-1}M_u^{-1}), \quad A_{22} \sim \mathcal{O}(M^{-1}M_u)$$
 (25)

The effect of adaptive diffusion on accuracy and efficiency is discussed more in [\[50\]](#page-34-0) and [\[58\]](#page-35-10) but will not be covered in any more detail here.

In this section we have shown how artificial diffusion schemes can be derived for purely convective or purely acoustic low Mach number flow using Turkel's method [\[69\]](#page-35-5) of balancing the artificial diffusion terms R with the physical terms L in the limit as M → 0. These schemes were then combined to create a mixed scheme suitable for both convective and acoustic flow, although this scheme has asymptotically vanishing artificial diffusion on the convective pressure and acoustic velocity equations.

We emphasise that these three diffusion scalings are not novel to the present study - all are in use today for low Mach number simulations, with the exception of the purely acoustic scheme, for the reasons discussed above. However, we have shown that many of the low-Mach behaviours of this class of schemes can be demonstrated independently of any specific discretisation.

In deriving and analysing these schemes so far, we have enforced the variations to obey either the convective or acoustic scalings from table [1.](#page-8-1) While this is useful for designing the diffusion schemes, we do not yet know how the schemes will perform if the variations are not enforced.

# 4 Analysis of the continuous Euler equations with artificial diffusion

In this section we will further analyse the three artificial diffusion schemes described in the previous section to better understand their behaviour. The bulk of this section will be six asymptotic expansions: a single-timescale and a multiple-timescale expansion of the modified equations [\(13\)](#page-7-1) for each of the three diffusion schemes A c , A a and A <sup>m</sup>. Because all variables will be expanded according to either equation [\(3\)](#page-3-5) or [\(8\)](#page-4-4), the scaling of the variations will not be enforced, overcoming the main limitation of the previous section. At the end of this section we estimate the asymptotic scaling for the spectral radius of each scheme, which will affect their stability.

Before carrying out the asymptotic expansions, we consider how to judge whether a relation obtained from the expansion of the modified equations [\(13\)](#page-7-1) "matches" - in some sense - the corresponding relation from the expansion of the physical equations [\(2\)](#page-3-4). All terms in L will match exactly by construction. However, we identify three types of relation with differing requirements on the terms in R:

1. Pressure variation relations. Relations [\(4a,](#page-3-6)[4b\)](#page-3-7) must be enforced exactly, else lower order pressure variations will swamp the higher order terms of interest, leading to catastrophic loss of accuracy.

- 2. Transport equation relations. The convective momentum relation (4c), and the acoustic relations (10b,11b) should retain some artificial diffusion terms for the scheme to remain stable.
- 3. Divergence constraint relations. It can be argued that relations (5a,5b) should be matched exactly, so that the  $\mathcal{O}(M^0)$  velocity field becomes divergence-free [33]. However, the inf-sup condition means that some discrete schemes are susceptible to pressure-velocity decoupling (chequer-board modes) in the incompressible limit. Certain collocated schemes [27, 19] can avoid this issue by using Brezzi-Pitkäranta stabilisation [4], which introduces a pressure diffusion term into the continuity equation. As noted in [23, 15] this technique can be applied to density-based schemes, so it can also be argued that the velocity divergence should be zero only up to the value of some pressure diffusion term. Both choices are valid, and the impact of this choice will be discussed later.

## 4.1 Expansion of the Euler equations with convective diffusion

First, we carry out a single timescale expansion of (13a,13b) with convective diffusion scaling (15). The  $\mathcal{O}(M^{-2})$ ,  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^{0})$  velocity relations are:

$$\partial_x p^{(0)} = A_{21}^{(-2)} \partial_{xx} p^{(0)} \tag{26a}$$

$$\partial_x p^{(1)} = A_{21}^{(-2)} \partial_{xx} p^{(1)} \tag{26b}$$

$$\rho^{(0)}\partial_t u^{(0)} + \partial_x p^{(2)} + \rho u^{(0)}\partial_x u^{(0)} = A_{21}^{(-2)}\partial_{xx} p^{(2)} + A_{22}^{(0)}\partial_{xx} u^{(0)}$$
(26c)

The relation (26c) is exactly the physical momentum relation (4c) plus all the artificial diffusion terms, as we required, and as found in the limit equation (16b). Upon first inspection, the pressure variation relations (26a,26b) do not appear to satisfy the physical relations (4a,4b) because of the presence of the artificial diffusion terms. Expanding the pressure equation gives the  $\mathcal{O}(M^{-2})$ ,  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^{0})$  relations:

$$0 = A_{11}^{(-2)} \partial_{xx} p^{(0)} \tag{27a}$$

$$0 = A_{11}^{(-2)} \partial_{xx} p^{(1)} \tag{27b}$$

$$d_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-2)} \partial_{xx} p^{(2)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(27c)

Now we see that the spatial variations of  $p^{(0,1)}$  are constrained by the two systems (26a,27a) and (26b,27b) respectively. Both systems admit the desired constant solutions, but we cannot yet tell whether they also admit non-constant solutions<sup>†</sup>. We will assume  $p^{(0,1)}$  have constant initial conditions, and boundary conditions which are either periodic, or constant in both time and space. As for the physical relations (4),  $p^{(1)}$  is assumed to have initial and boundary conditions equal to zero. These assumptions lead to constant solutions, satisfying the pressure variation relations (4a,4b). See Guillard & Viozat [26] for a more in-depth discussion of the conditions under which the discrete equivalents of these relations (which we will see later) lead to constant  $p^{(0,1)}$  solutions. Note that it is useful to have restraints on both the first and second derivatives, despite the more involved reasoning about the constancy of  $p^{(0,1)}$ . Assuming that central gradient approximations  $(\partial_x p|_i \approx \frac{p_{i+1}-p_{i-1}}{2\Delta x})$  will be used in the final discrete schemes, the discrete  $\partial_x p = 0$  constraint alone could lead to odd-even decoupling and allow non-constant grid modes. These modes are suppressed by the additional constraint  $\partial_{xx} p = 0$ .

Assuming constant  $p^{(0,1)}$  then  $\partial_t p^{(0)} \to d_t p^{(0)}$  and relation (27c) is exactly the physical divergence relation (5a), plus all the artificial diffusion terms. The retention of the artificial pressure diffusion term allows for Brezzi-Pitkäranta stabilisation as discussed above. Therefore, all relevant relations from the single-timescale expansion are reproduced by the convective diffusion scheme.

Next, we carry out a multiple-timescale expansion of (13a,13b) with convective diffusion scaling (15). The  $\mathcal{O}(M^{-2})$ ,  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^{0})$  velocity relations are:

$$\partial_x p^{(0)} = A_{21}^{(-2)} \partial_{xx} p^{(0)} \tag{28a}$$

$$\rho^{(0)}\partial_{\tau}u^{(0)} + \partial_{x}p^{(1)} = A_{21}^{(-2)}\partial_{xx}p^{(1)}$$
(28b)

$$\partial_{\tau}\rho u^{(1)} + \rho^{(0)}\partial_{t}u^{(0)} + \partial_{x}p^{(2)} + \rho u^{(0)}\partial_{x}u^{(0)} = A_{21}^{(-2)}\partial_{xx}p^{(2)} + A_{22}^{(0)}\partial_{xx}u^{(0)}$$
(28c)

The  $\mathcal{O}(M^{-2})$ ,  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^{0})$  pressure relations are:

$$0 = A_{11}^{(-2)} \partial_{xx} p^{(0)} \tag{29a}$$

$$\partial_{\tau} p^{(0)} = A_{11}^{(-2)} \partial_{xx} p^{(1)} \tag{29b}$$

$$\partial_{\tau} p^{(1)} + d_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-2)} \partial_{xx} p^{(2)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(29c)

We do not make use of  $A\partial_{xx}p = 0 \implies \partial_{xx}p = 0$  as this is invalid for nonlinear artificial diffusion terms of the form  $\partial_x(A\partial_x p)$ .

The  $p^{(0)}$  relations (28a,29a) are identical to the single-timescale relations (26a,27a), so  $p^{(0)}$  is constant under the same assumptions as before. If we assume no forcing of  $p^{(0)}$  on the acoustic timescale - which seems reasonable - then  $\partial_{\tau}p^{(0)} = 0$  and  $\partial_{t}p^{(0)} \to d_{t}p^{(0)}$ , and relation (29b) becomes a parabolic equation for  $p^{(1)}$  equivalent to relation (19a). This will quickly damp any acoustic variations in the pressure, confirming that the convective diffusion scaling  $\underline{A}^{c}$  is unsuitable for simulations with acoustic variations.

## 4.2 Expansion of the Euler equations with acoustic diffusion

The single-timescale expansion of the modified equations (13a,13b) with  $\underline{\underline{A}}^a$  leads to the following  $\mathcal{O}(M^{-2})$  and  $\mathcal{O}(M^{-1})$  velocity relations:

$$\partial_x p^{(0)} = A_{21}^{(-2)} \partial_{xx} p^{(0)} \tag{30a}$$

$$\partial_x p^{(1)} = A_{21}^{(-2)} \partial_{xx} p^{(1)} + A_{22}^{(-1)} \partial_{xx} u^{(0)}$$
(30b)

and the  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^0)$  pressure relations:

$$0 = A_{11}^{(-1)} \partial_{xx} p^{(0)} \tag{31a}$$

$$d_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(31b)

The  $p^{(0)}$  relations (30a,31a) lead to constant  $p^{(0)}$  under the assumptions described above. On the other hand, the relation (30b) means that  $\partial_x p^{(1)} \neq 0$  unless the velocity field is trivial and the term  $A_{22}^{(-1)} \partial_{xx} u^{(0)} = 0$ , making the acoustic diffusion scheme unsuitable for convective low Mach number flows. For a diagonal scheme at steady-state, relations (30b,31b) resemble the Stokes flow equations, indicating that the solution will be dominated by the diffusive effects, balanced by variations in  $p^{(1)}$ . The production of unphysical acoustic modes by  $A_{22}^{(-1)}$ , even from a well-prepared initial field, has been shown numerous times for specific discrete schemes [26, 67, 33, 53], and in the continuous setting in [25, 15].

A two timescale expansion of equations (13a,13b) with acoustic diffusion gives the  $\mathcal{O}(M^{-2})$  and  $\mathcal{O}(M^{-1})$  velocity relations:

$$\partial_x p^{(0)} = A_{21}^{(-2)} \partial_{xx} p^{(0)} \tag{32a}$$

$$\rho^{(0)}\partial_{\tau}u^{(0)} + \partial_{\tau}p^{(1)} = A_{21}^{(-2)}\partial_{\tau\tau}p^{(1)} + A_{22}^{(-1)}\partial_{\tau\tau}u^{(0)}$$
(32b)

and the  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^0)$  pressure relations:

$$\partial_{\tau} p^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(0)} \tag{33a}$$

$$\partial_{\tau} p^{(1)} + d_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(33b)

The  $p^{(0)}$  relations (32a,33a) lead to constant  $p^{(0)}$  under the assumptions described above. The relations (32b,33b) are exactly the physical acoustic relations (10b,11b) plus all artificial diffusion terms, so we expect this scheme to be able to simulate at least purely acoustic flow. However, this scheme is still unsuitable for mixed convective-acoustic flow, as it will be unable to properly resolve the convective component as shown in the single timescale expansion.

#### 4.3 Expansion of the Euler equations with mixed diffusion

Finally, we carry out the asymptotic expansions with the mixed diffusion scheme  $\underline{\underline{A}}^m$  (21). The single timescale expansion results in the  $\mathcal{O}(M^{-2})$ ,  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^0)$  velocity relations:

$$\partial_x p^{(0)} = A_{21}^{(-2)} \partial_{xx} p^{(0)} \tag{34a}$$

$$\partial_x p^{(1)} = A_{21}^{(-2)} \partial_{xx} p^{(1)} \tag{34b}$$

$$\rho^{(0)}\partial_t u^{(0)} + \partial_x p^{(2)} + \rho u^{(0)}\partial_x u^{(0)} = A_{21}^{(-2)}\partial_{xx} p^{(2)} + A_{22}^{(0)}\partial_{xx} u^{(0)}$$
(34c)

and the  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^{0})$  pressure relations:

$$0 = A_{11}^{(-1)} \partial_{xx} p^{(0)} \tag{35a}$$

$$d_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(35b)

The  $p^{(0)}$  relations (34a,35a) lead to constant  $p^{(0)}$  under the assumptions described above. On the other hand  $p^{(1)}$  is constrained by relation (34b) and the continuity relation (35b). While the constant vector

is clearly a solution to (34b) under the previous assumptions, it is now less clear whether it is the only solution. As we shall see later, almost all schemes which use the mixed diffusion scaling have asymptotically diagonal diffusion. In this case, (34b) becomes  $\partial_x p^{(1)} = 0$ , so the physical relation (4b) is matched exactly, although as discussed earlier, the discrete scheme will be susceptible to odd-even decoupling on  $p^{(1)}$  without the constraint on  $\partial_{xx}p^{(1)}$ . At steady-state, the continuity relation (35b) for a diagonal scheme becomes  $p^{(0)}\partial_x u^{(0)} = A_{11}^{(-1)}\partial_{xx}p^{(1)}$ . This means that if  $p^{(1)}$  is constant then  $u^{(0)}$  is divergence-free, just as the incompressibility condition requires, but cannot be stabilised against chequerboard modes on  $p^{(\geq 2)}$  in the manner of the Brezzi-Pitkäranta.

The last relation, velocity relation (34c), exactly matches (26c), which matches the physical relation (4c) with both diffusion terms retained. As every relevant physical relation is matched by the mixed diffusion scheme, we can expect that discrete schemes with matching modified equations will be suitable for simulations of purely convective low Mach number flow, although potentially at the risk of chequerboard modes on  $p^{(\geq 2)}$ .

Two timescale expansion of (13a,13b) with mixed diffusion scaling (21) gives the  $\mathcal{O}(M^{-2})$ ,  $\mathcal{O}(M^{-1})$ and  $\mathcal{O}(M^0)$  velocity relations:

$$\partial_x p^{(0)} = A_{21}^{(-2)} \partial_{xx} p^{(0)} \tag{36a}$$

$$\rho^{(0)}\partial_{\tau}u^{(0)} + \partial_{x}p^{(1)} = A_{21}^{(-2)}\partial_{xx}p^{(1)}$$
(36b)

$$\partial_{\tau}\rho u^{(1)} + \rho^{(0)}\partial_{t}u^{(0)} + \partial_{x}p^{(2)} + \rho u^{(0)}\partial_{x}u^{(0)} = A_{21}^{(-2)}\partial_{xx}p^{(2)} + A_{22}^{(0)}\partial_{xx}u^{(0)}$$
(36c)

and the  $\mathcal{O}(M^{-1})$  and  $\mathcal{O}(M^0)$  pressure relations:

$$\partial_{\tau} p^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(0)} \tag{37a}$$

$$\partial_{\tau} p^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(0)}$$

$$\partial_{\tau} p^{(1)} + d_t p^{(0)} + \gamma p^{(0)} \partial_x u^{(0)} = A_{11}^{(-1)} \partial_{xx} p^{(1)} + A_{12}^{(0)} \partial_{xx} u^{(0)}$$
(37a)
$$(37b)$$

The  $p^{(0)}$  relations (36a,37a) again lead to constant  $p^{(0)}$  under the assumptions described above. Relations (36b,37b) match the physical acoustic relations (10b,11b), plus some artificial diffusion terms. The presence of  $\partial_{xx}p^{(1)}$  in relation (37b) now provides the appropriate diffusion on the acoustic pressure variations. On the other hand, the relation (36b) lacks a diagonal diffusion term, just as in the limit equation (23b), so a diagonal scheme will have no diffusion on the acoustic velocity variations.

Relations (36b,37b) are almost equivalent to the modified equations of the scheme proposed by Bruel et al. (section 3.1.5.1 in [5]). This scheme differs from the current scheme in the precise form of the diffusion in the off-diagonal terms, but importantly the Mach number scaling of these terms matches ours. These differences will be discussed further in section 5. The mixed diffusion scheme  $A^m$  is suitable for both convective and mixed convective-acoustic low Mach number flows. however, the scheme is not without disadvantages, having potentially degraded stability on the convective pressure and acoustic velocity variations.

## Spectral radius estimates

We now estimate the scaling of the spectral radii  $\Lambda(\underline{\underline{A}})$  of  $\underline{\underline{A}}^c$ ,  $\underline{\underline{A}}^a$  and  $\underline{\underline{A}}^m$  as  $M \to 0$ . The spectral radius of the combined flux components determines the CFL bound for stability of explicit timestepping schemes and affects the convergence of implicit schemes. The flux Jacobian of the Euler equations has spectral radius  $u + a \sim \mathcal{O}(M^{-1})$ , so the artificial diffusion will induce a more stringent stability bound if  $\Lambda(\underline{A}) \sim \omega(M^{-1})$ . Because diffusion matrices are positive (semi-)definite, we can estimate the scaling of the spectral radius with the scaling of the trace<sup>†</sup>. By inspection, the traces of the coefficient matrices (15), (17) and (21) are:

$$\operatorname{tr}(\underline{\underline{A}}^c) \sim \mathcal{O}(M^{-2}), \quad \operatorname{tr}(\underline{\underline{A}}^a) \sim \operatorname{tr}(\underline{\underline{A}}^m) \sim \mathcal{O}(M^{-1})$$
 (38)

 $tr(\underline{A}^a)$  and  $tr(\underline{A}^m)$  are the same order as the physical spectral radius, so should not change the stability requirements - up to a constant factor. On the other hand,  $tr(\underline{A}^c)$  is larger than the physical spectral radius by an order of  $M^{-1}$ , so will require a CFL bound which decreases one order faster than the physical CFL bound in order to remain stable as  $M \to 0$ . This scaling was first proven by Birken & Meister 2005 [2] for the specific case of preconditioned matrix-Rusanov artificial diffusion. The estimate (38) also agrees with [15] (equation (80) with  $\kappa_r = M^{-1}$ ) that this restriction will hold for any scheme whose pressure diffusion converges to the convective limit as  $M \to 0$ . For example, in the companion paper we will see

<sup>&</sup>lt;sup>†</sup>The trace cannot grow faster than the spectral radius for any fixed rank matrix. The trace of a matrix can grow more slowly than the spectral radius due to cancellation. For example, the trace of the flux Jacobian of the Euler equations in Ndimensions is (N+2)u, which is  $\mathcal{O}(M^0)$ , even though the spectral radius u+a is  $\mathcal{O}(M^{-1})$ . However, cancellation is impossible for a positive (semi-)definite matrix by definition. As the trace of a positive (semi-)definite matrix cannot grow asymptotically faster or slower than the spectral radius, it is an appropriate estimate for the scaling of the spectral radius.

| Symbol               | Definition                                                                                       |  |  |  |  |
|----------------------|--------------------------------------------------------------------------------------------------|--|--|--|--|
| $\psi_i$             | Quantity $\psi$ in cell $i$                                                                      |  |  |  |  |
| $\psi$               | Edge/face average of quantity $\psi$ (only in edge/face summations)                              |  |  |  |  |
| $\Omega_i$           | Area/volume of cell $i$                                                                          |  |  |  |  |
| $S_{il}$             | Length/area of edge/face between cells $i$ and $l$                                               |  |  |  |  |
| $\underline{n}_{il}$ | Outgoing normal vector of edge/face between cells $i$ and $l$                                    |  |  |  |  |
| $\mathcal{V}(i)$     | Set of all cells neighbouring cell $i$                                                           |  |  |  |  |
| U                    | Edge/face normal velocity $\underline{u} \cdot \underline{n}_{il}$                               |  |  |  |  |
| $\Delta_{il}\psi$    | Interface jump $\psi_l - \psi_i$                                                                 |  |  |  |  |
| $\Delta_{il}U$       | Interface jump in normal velocity $(\underline{u}_l - \underline{u}_i) \cdot \underline{n}_{il}$ |  |  |  |  |

Table 2: Nomenclature used for the discrete asymptotic expansions.

that this is true for existing AUSM schemes that converge to this limit, which explains the problems encountered in [45, 29, 8].

In this section, we have used single and multiple-scale asymptotic analysis to investigate each artificial diffusion scheme at the convective and mixed convective-acoustic limits. We have demonstrated several known properties of discrete low Mach number schemes. Including: the spurious forcing of the acoustic pressure  $p^{(1)}$  by the acoustic scheme  $\underline{\underline{A}}^a$ ; the over-damping of the acoustic pressure by the convective scheme  $\underline{\underline{A}}^c$ ; the susceptibility to chequer-board modes on the convective pressure  $p^{(2)}$  and instabilities on the acoustic velocity variations of the mixed scheme  $\underline{\underline{A}}^m$ ; and the CFL limit of the convective scheme. So far we have worked in the entropy variables and the continuous setting to simplify the presentation, but in the next section we transfer the artificial diffusion to the conservative variables.

# 5 Analysis of the discrete Euler equations with artificial diffusion

The continuous analysis of the previous section is general to most schemes with the modified equations (13). In this section we demonstrate how the findings of the continuous analysis transfer to the discrete setting in the particular case of a first order cell-centred finite volume Roe-type scheme. First we identify the form of the interface flux in the conserved variables that matches the diffusion in the entropy variables (13), which we can then compare against previous Roe-type schemes from the literature. We then carry out the same six asymptotic expansions as in the previous section, but this time on the discrete equations. We finish the section by comparing the von Neumann symbols for each of the three schemes.

#### 5.1 A general form for low Mach number finite-volume schemes

We consider the semi-discrete equations for a first order cell-centred finite-volume scheme in conservative variables  $\underline{q} = (\rho, \underline{\rho u}, \rho E)^T$ , where the time derivatives are left continuous using a method of lines approach. Using the nomenclature of Guillard & Viozat [26] (table 2), the evolution in time of the solution at each cell i is described by the ODE:

$$\Omega_i \frac{d\underline{q}_i}{dt} + \sum_{\mathcal{V}(i)}^l S_{il} \{\underline{f}\}_{il} = \sum_{\mathcal{V}(i)}^l S_{il} \underline{f}_{il}^d$$
(39)

Where  $\{\underline{f}\}_{il} = \frac{1}{2}(\underline{f}(\underline{q}_i) + \underline{f}(\underline{q}_l))$  is the central approximation of the exact physical flux  $\underline{f}(\underline{q})$  and  $\underline{f}_{il}^d$  is the artificial diffusion flux between cells i and l. This nomenclature is general to any cell-centred scheme, but we will consider only quadrilaterals in 2D and hexahedra in 3D because, as mentioned in section 2, the accuracy problem for the convective limit disappears on simplex meshes [54, 55, 24, 17, 23]. In the non-dimensional entropy variables and a 2D face-aligned coordinate frame, we choose the elements  $A_{ij}$  of the Jacobian of the diffusive flux to have the form:

$$|\underline{\underline{A}}| = \mu_u \frac{|U|}{2} \underline{\underline{\mathcal{I}}} + \frac{1}{2} \begin{pmatrix} M^{-2} \frac{\gamma p}{\rho |v|} \mu_{11} & \pm \gamma p \mu_{12} & 0 & 0\\ \pm M^{-2} \mu_{21} & \rho |v| \mu_{22} & 0 & 0\\ 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0 \end{pmatrix}$$

$$(40)$$

The first term is the convective upwinding. The second term is the diffusion on the pressure (first row) and normal velocity (second row) which was discussed in the previous sections. |v| is some  $\mathcal{O}(M^0)$ 

velocity scale which is included to ensure correct dimensionality.  $\mu_{\alpha}$  ( $\alpha \in \{u, 11, 12, 21, 22\}$ ) are positive expressions whose form and Mach number scaling are specific to each discrete scheme. The precise forms of the elements of this Jacobian are somewhat arbitrary, so long as their scaling can be chosen to match one of the three diffusion scalings  $\underline{\underline{A}}^{c,a,m}$  - achieved here by varying the scaling of the coefficients  $\mu_{\alpha}$ . This particular form is chosen because it simplifies the diffusion coefficients in the conserved variables. The off-diagonal terms may be negative so long as positive (semi-)definiteness of the Jacobian is maintained - this will be discussed in more detail below. Transforming (40) to the dimensional conserved variables, the diffusive flux between cells i and l is:

$$\underline{\tilde{f}}_{il}^{d} = \frac{1}{2} \left[ \mu_{u} |\tilde{U}| \begin{pmatrix} \Delta_{il} \tilde{\rho} \\ \Delta_{il} \tilde{\rho} \underline{\tilde{u}} \\ \Delta_{il} \tilde{\rho} \tilde{E} \end{pmatrix} + \delta U \begin{pmatrix} \tilde{\rho} \\ \tilde{\rho} \underline{\tilde{u}} \\ \tilde{\rho} \tilde{H} \end{pmatrix} + \delta p \begin{pmatrix} 0 \\ \underline{n} \\ \tilde{U} \end{pmatrix} \right]$$
(41)

where the interface velocity and pressure perturbations  $\delta U$  and  $\delta p$  are defined as:

$$\delta U = \frac{\mu_{11}}{\tilde{\rho}|\tilde{v}|} \Delta_{il} \tilde{p} + \frac{\tilde{U}_{il}}{|\tilde{U}_{il}|} \mu_{12} \Delta_{il} \tilde{U}$$

$$\delta p = \frac{\tilde{U}_{il}}{|\tilde{U}_{il}|} \mu_{21} \Delta_{il} \tilde{p} + \tilde{\rho}|\tilde{v}| \mu_{22} \Delta_{il} \tilde{U}$$
(42)

This is precisely the Liu & Vinokur form [42], which was also used by Weiss & Smith [78] and Li & Gu [33] for low Mach number Roe-type fluxes. The first term is the natural upwinding for the convective system. From the definitions of  $\delta U$  and  $\delta p$  we can see that the second and third terms are, respectively, the diffusion on the pressure and velocity equations in the entropy variables. We expand the coefficients  $\mu_{\alpha}$  as:

$$\mu_{\alpha} = \epsilon_{\alpha} \nu_{\alpha} M^n \tag{43}$$

 $\epsilon_{\alpha}$  is a constant positive real valued diffusion coefficient, for example  $\epsilon=1$  gives the standard first order upwind diffusion, or  $1/64 < \epsilon < 1/32$  can be used for fourth derivative diffusion [3]. Usually  $\epsilon_{\alpha}$  are all equal for a particular scheme, although they do not have to be. The most common example of this is  $\epsilon_{11}$  for the convective scheme, which should instead be chosen in the recommended range for the Brezzi-Pitkäranta stabilisation coefficient. For the rest of this section we will use  $\epsilon_{\alpha}=1$  for simplicity.  $\nu_{\alpha} \sim \mathcal{O}(M^0)$  is some (non-dimensionalised) scheme specific expression and is the distinguishing feature between different schemes with the same diffusion scaling. The exponent n determines whether the diffusion has the convective, acoustic or mixed scaling, with the required exponents listed in table 3. The  $M^{-2}$  coefficients on  $\mu_{11}$  and  $\mu_{21}$  in (40) mean that the convective scaling exponents are all zero (i.e. M independent). Clearly, the convective upwinding coefficient should be  $\mu_{u} = \epsilon_{u}$  for all regimes. Although expanding out  $\mu_{\alpha}$  in this manner adds some complexity, it allows the separation of: the magnitude of the diffusion  $(\epsilon_{\alpha})$ ; the type of diffusion scaling  $(M^{n})$ ; and the specific discrete form  $(\nu_{\alpha})$  of each scheme. The full expressions for the diffusive fluxes can be found in appendix B.

## 5.2 Comparison to previous work

## 5.2.1 Previous guidelines

We can compare the scalings in table 3 with guidelines put forward by previous authors. Dellacherie proved in [15] that for Godunov type upwind schemes, the velocity diffusion must be  $\mu_{22} \sim \mathcal{O}(M^0)$  for accuracy at the convective limit, which was also shown in [26, 69]. It is clear that our findings for the convective and mixed scalings agree with this scaling. Dellacherie primarily studies the mixed scaling, but notes that the convective diffusion scaling (specifically the preconditioned Roe-Turkel scheme) also satisfies  $\mu_{22} \sim \mathcal{O}(M^0)$ , although with a larger pressure diffusion equivalent to the Brezzi-Pitkäranta stabilisation. These two scalings are both accurate for convective flow, but by considering both the convective and acoustic limits we have shown why this choice exists, and that they are distinct schemes with significantly different properties for acoustic and mixed flows. It is also shown in [15] that  $\mu_{22} \sim o(M^0)$  is accurate for convective flow. We can see from (40) that if  $\mu_{22} \sim o(M^0)$  then there will still be  $\mathcal{O}(M^0)$  velocity diffusion in this position from the convective upwinding term, albeit with a slightly different form, so the overall diffusion scaling is comparable to the  $\mu_{22} \sim \mathcal{O}(M^0)$  scheme.

In their survey of low Mach number Roe-type schemes for the convective limit, Gu & Li [33] offered three guidelines for the design of low Mach number schemes. The first guideline states that  $\mu_{22} \sim \mathcal{O}(M^0)$  or smaller is necessary for accuracy at the convective limit, as previously discussed for [15]. The second guideline states that  $\mu_{11}$  should be between  $\mathcal{O}(M)$  and  $\mathcal{O}(M^0)$  inclusive, where  $\mu_{11} \sim \mathcal{O}(M)$  may allow some small pressure chequerboards, but  $\mu_{11} \sim \mathcal{O}(M^0)$  will suppress all pressure chequerboards. This

 $<sup>^{\</sup>dagger}\mu_u = \epsilon_u \nu_u$  could be used where  $\nu_u$  is some non-linear function used to control the dissipation level, such as in [29].

| $\alpha$ | Convective         | Acoustic              | Mixed              |
|----------|--------------------|-----------------------|--------------------|
| u        | $\mathcal{O}(M^0)$ | $\mathcal{O}(M^0)$    | $\mathcal{O}(M^0)$ |
| 11       | $\mathcal{O}(M^0)$ | $\mathcal{O}(M)$      | $\mathcal{O}(M)$   |
| 12       | $\mathcal{O}(M^0)$ | $\mathcal{O}(M^0)$    | $\mathcal{O}(M^0)$ |
| 21       | $\mathcal{O}(M^0)$ | $\mathcal{O}(M^0)$    | $\mathcal{O}(M^0)$ |
| 22       | $\mathcal{O}(M^0)$ | $\mathcal{O}(M^{-1})$ | $\mathcal{O}(M^0)$ |

Table 3: Required scaling of the coefficients in  $\delta U$  and  $\delta p$ .

guideline matches the analysis for the convective and mixed scalings in the previous section, but again by including acoustic effects in the analysis we can see why this choice exists, and which flows each scaling is suitable for. Gu & Li also state that if  $\mu_{11} \sim \mathcal{O}(M^0)$  only in the continuity equation, very little improvement in the control of pressure chequerboards is seen compared to  $\mu_{11} \sim \mathcal{O}(M)$ . Equations (40) and (41,42) show that if  $\mu_{11}$  in  $\delta U$  is different in each equation then the equivalent diffusion in the entropy variables will not exactly match (40), which could explain the degraded performance. The third guideline states that a cut-off Mach number should only be used in denominators, i.e. when it decreases the diffusion. The issue of cut-off Mach numbers has not been covered here, but broadly speaking this guideline means, firstly, that  $\mu_{22}$  will not be increased beyond  $\mathcal{O}(M^0)$  by a cut-off Mach number, which would compromise the accuracy for convective flow features, and secondly,  $\mu_{11}$  can still be decreased in the convective scaling, which alleviates the stability issue related to the  $\mathcal{O}(M^{-2})$  spectral radius. See [33] for details of this implementation.

## 5.2.2 Classification of existing Roe-type schemes

Now that we have an expression for the artificial diffusion in the conserved variables (41,42), we can classify existing schemes as having either convective, mixed or acoustic diffusion scaling at low Mach number. Table 4 details many low-Mach Roe-type schemes from the literature. Some are adaptive schemes, and have different scalings for large timesteps using the convective CFL,  $\sigma_u = u\Delta t/\Delta x \approx 1$ , and for small timesteps using the acoustic CFL,  $\sigma_a = a\Delta t/\Delta x \approx 1$ . We have also noted whether each diffusion scheme is asymptotically diagonal  $(\mu_{12}, \mu_{21} \sim o(M^0))$  or upper/lower triangular in the entropy variables. The discrete form of  $\mu_{ij}$  for a number of schemes in table 4 can be found in [33]. Several trends are apparent in this table.

Almost all of the earlier schemes use the convective scaling [21, 70, 78, 26, 25, 44]. These early schemes use a diffusive Jacobian derived from the preconditioned physical Jacobian (except [44]), which leads naturally to the convective diffusion scaling (see [69] and [26] for more detail). On the other hand, many of the more recent schemes use the mixed diffusion [73, 75, 50, 67, 65, 15, 53, 58, 49, 5] even when the target regime is the convective limit. These schemes are derived primarily by identifying that for the original Roe scheme, which has the acoustic scaling, the accuracy problem is caused by  $A_{22}^{(-1)}$ . These schemes selectively reduce  $A_{22}$  by a factor of M, thus reaching the mixed scheme. Most of these studies do not discuss the acoustic low Mach capability of the scheme - exceptions being Venkateswaran and coauthors [73, 75, 50, 58] and Bruel et al. [5]<sup>†</sup>. All non-adaptive mixed schemes are diagonal except for the scheme of Bruel et al. [5].

The main development of adaptive schemes is due to Venkateswaran and co-authors [72, 73, 75, 50, 58], to allow accurate simulation of acoustic phenomena with the mixed scheme, but to return to the more favourable convergence and stability properties of the convective scaling when  $\Delta t$  is too large to resolve the acoustic waves. To the authors' knowledge, [73] is the earliest use of the mixed diffusion scaling in the literature. The scalar diffusion schemes in [73, 75, 50] are diagonal for both large and small timesteps. The matrix diffusion schemes in [58] return to the preconditioned Roe scheme for large timesteps, where all four diffusion coefficients are well balanced, but for small timesteps they approach mixed schemes which are upper/lower triangular in the entropy variables (i.e. either  $\mu_{12}$  or  $\mu_{21}$  is  $o(M^0)$ ).<sup>‡</sup> The first scheme (equation (14) in [58]) has adaptive scaling on the  $\delta U$  diffusion terms  $\mu_{11}$  and  $\mu_{12}$ , meaning they both return to the values in the original Roe scheme for small timestep -  $\mu_{11}$  becomes  $\mathcal{O}(M)$  and  $\mu_{12}$  becomes  $o(M^0)$ , so  $\underline{A}$  is lower triangular. Conversely, the second scheme (equation (15) in [58]) has adaptive scaling on the pressure diffusion terms  $\mu_{11}$  and  $\mu_{21}$ , so becomes upper triangular. The authors state that the matrix diffusion scheme of Potsdam et al. [50] is almost equivalent to the second scheme in [58]. Despite these differences, Sachdev et al. report no visible differences between results with these three schemes.

<sup>&</sup>lt;sup>†</sup>Dellacherie [15] analyses the acoustic equations in depth, but makes only a minor distinction between the convective and mixed diffusion scalings, instead focusing on the necessity of reducing  $\mu_{22}$  compared to the acoustic scaling.

<sup>&</sup>lt;sup>‡</sup>This can be seen by identifying that  $C\phi^T$  and  $C_p'$  in equations (12,13) in [58] correspond to the  $\delta U$  and  $\delta p$  terms respectively in (41,42).

Interestingly, the unsteady momentum interpolation method of Li & Gu [\[34\]](#page-33-8) also appears to be an adaptive scheme. The earlier All-speed scheme [\[32,](#page-33-7) [35\]](#page-34-7) approaches the convective scheme of Mary & Sagaut [\[44\]](#page-34-4) at low Mach number. However, in [\[34\]](#page-33-8) the pressure diffusion µ<sup>11</sup> is scaled by the timestep so will be a factor of M smaller with σ<sup>a</sup> ≈ 1 than with σ<sup>u</sup> ≈ 1, hence approaching the mixed scaling for small timesteps. This can be seen from equations (19,20) and figures 3-4 of [\[34\]](#page-33-8) where slight chequerboards are seen for small timesteps but are controlled for larger timesteps.

The final point to note from table [4](#page-18-0) is that a few schemes reduce at least some components of the convective upwinding term by a factor of M [\[65,](#page-35-8) [15,](#page-32-4) [49\]](#page-34-6). We cannot assess the effects of this change using our analysis so far, but we will return to consider it at the end of the section when we examine the von Neumann symbols of the first order scheme [\(39\)](#page-14-4).

## 5.2.3 Behaviour of the off-diagonal diffusion

The off-diagonal diffusion terms in [\(42\)](#page-15-3) contain the switch Uil/|Uil| = ±1. This ensures that these terms have a diffusive character but also gives the possibility of negative diffusion coefficients.[†](#page-0-0) Previous schemes often use just Uil, but we prefer a non-dimensional O(1) expression. In 2D the off-diagonal terms result in anisotropic diffusion of the form (±∂xxu ± ∂yyv) on all equations through δU, and ±∂xxp on the xmomentum equation and ±∂yyp on the y-momentum equation through δp. The sign on ∂xxu and ∂xxp will be the same, as will the sign on ∂yyv and ∂yyp, but the sign may differ between the x and y derivatives. As stated earlier, off-diagonal anti-diffusion is tolerable so long as positive (semi-)definiteness is maintained. This requires µ11µ<sup>22</sup> − µ12µ<sup>21</sup> ≥ 0, where equality implies semi-definiteness. For asymptotically diagonal or triangular schemes this holds trivially. This covers all acoustic schemes, almost all mixed schemes and some convective schemes in table [4.](#page-18-0) The full convective schemes are due to the preconditioned diffusion, and the inequality holds for these schemes.[‡](#page-0-0) For the mixed scheme µ11µ<sup>22</sup> vanishes asymptotically for both convective and acoustic variations, therefore µ12µ<sup>21</sup> must also vanish asymptotically else the diffusion will be negative definite and the scheme unstable.[§](#page-0-0) The one exception to this is the full scheme of Bruel et al [\[5\]](#page-32-5), where the off-diagonal terms are designed to remove the acoustic instability and to obtain a CFL limit σ<sup>a</sup> < 1 compared to σ<sup>a</sup> < 0.5 for the diagonal mixed scheme. This scheme modifies the off-diagonal diffusion to be isotropic with the form ∇2u + ∇2v in δU and ∇2p on all momentum equations through δp. Positive semi-definiteness is ensured by enforcing that the off-diagonal terms have opposite sign, but which term is positive and which negative is indeterminate. However, the scheme produces asymmetric solutions for flow around a circular cylinder, which Bruel et al attribute to this free choice of sign and the resulting loss of Galilean invariance.

## 5.3 Expansion of the discrete Euler equations with artificial diffusion

In this section we carry out singleand multiple-scale asymptotic expansions of the discrete equations [\(39\)](#page-14-4) with diffusive flux [\(41,](#page-15-2)[42\)](#page-15-3) for each of the three scalings in table [\(3\)](#page-16-1). For simplicity we have taken <sup>α</sup> = 1 throughout. We will keep the discussion of these expansions brief, because most points have already been discussed in the previous section, and many of the expansions have been presented in the literature (although usually for one specific flux scheme). The discrete single-timescale expansion with the convective diffusion is investigated in detail in the seminal papers from Guillard & coauthors [\[26,](#page-33-4) [25\]](#page-33-5), and the most important features are also reported in the reviews by Guillard & Nkonga [\[23\]](#page-33-6) and Li & Gu [\[33\]](#page-33-9). The discrete single-timescale expansion with the acoustic diffusion can be found in a several papers, as it shows why classical transonic schemes fail at low Mach number, see [\[26,](#page-33-4) [25,](#page-33-5) [23,](#page-33-6) [53\]](#page-34-5). Several of these papers also present the discrete single-timescale expansion with the mixed diffusion [\[15,](#page-32-4) [53,](#page-34-5) [32,](#page-33-7) [36,](#page-34-10) [33,](#page-33-9) [23\]](#page-33-6). To the authors' knowledge, the only previous study in the literature to carry out multiple-timescale expansions of the discrete equations is Bruel et al. [\[5\]](#page-32-5), who present multiple-timescale expansions of the baratropic Euler equations with all three schemes.

## 5.3.1 Expansion with convective diffusion

First, we carry out the single timescale expansion of the discrete equations [\(39\)](#page-14-4) with convective diffusion scaling. The relations are exactly equivalent to those found by Guillard & Viozat [\[26\]](#page-33-4), except for the more general notation for the diffusion coefficients, so will only be discussed briefly - consult [\[26\]](#page-33-4) for more in-depth discussion on these discrete equations, they are included here mainly for completeness. The

<sup>†</sup>The authors would like to thank the anonymous reviewer who demonstrated that these terms did not have a diffusive form in an earlier draft. Correcting this led to the discussion in section [5.2.3.](#page-17-0)

<sup>‡</sup>As M → 0 the coefficients of this scheme become µ<sup>11</sup> ≈ 2|v˜|/(U √ 5), µ<sup>22</sup> ≈ U(3 − 5)/(|v˜| √ 5), and µ<sup>12</sup> = µ<sup>21</sup> ≈ 1/ 5.

<sup>§</sup>When iterative methods that require diagonal dominance are used for implicit timestepping, the semi-definiteness of the mixed scheme also necessitates the use of inconsistent diffusion on the left-hand-side [\[50,](#page-34-0) [58,](#page-35-10) [60\]](#page-35-18).

| Scheme                                                                 | $\begin{array}{ c c c }\hline Scaling \\\hline \sigma_u \sim \mathcal{O}(1) & \sigma_a \sim \mathcal{O}(1) \\\hline \end{array}$ |          | - Diagonal                                                                                                                              | Comments                                                                                                 |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
|                                                                        |                                                                                                                                  |          |                                                                                                                                         |                                                                                                          |
| Roe 1981 [56]                                                          | Acou                                                                                                                             |          | Yes                                                                                                                                     | Original transonic scheme                                                                                |
| Preconditioned Roe 1993 [21]                                           | Convective                                                                                                                       |          | No                                                                                                                                      | Preconditioned Roe scheme                                                                                |
| Venkateswaran & Merkle<br>1995 [72]                                    | Convective                                                                                                                       | Acoustic | Yes                                                                                                                                     | Central difference dissipation                                                                           |
| Weiss & Smith 1995 [78]                                                | Convective                                                                                                                       |          | No                                                                                                                                      | Preconditioned Roe scheme                                                                                |
| Guillard & Viozat 1999<br>[26]                                         | Convective                                                                                                                       |          | No                                                                                                                                      | First asymptotic expansion of discrete preconditioned Roe scheme                                         |
| Venkateswaran & Merkle 20(00/03) [73, 75]                              | Convective                                                                                                                       | Mixed    | Yes                                                                                                                                     | Adaptive scalar diffusion. $\mu_{11}$ only applied to the continuity equation                            |
| Mary & Sagaut 2002 [44]                                                | Convective                                                                                                                       |          | Yes                                                                                                                                     | No $\delta p$ , and introduces only $\mu_{11}$ into the mass flux in $\mathcal{L}$ , similar to [52, 40] |
| Guillard & Murrone 2004 [25]                                           | Convective                                                                                                                       |          | No                                                                                                                                      | Godunov scheme based on the preconditioned system                                                        |
| Postdam et al. 2007 [50]                                               | Convective                                                                                                                       | Mixed    | (Yes/No)/(Yes/UT)                                                                                                                       | Two adaptive diffusion schemes (scalar/matrix)                                                           |
| Thornber et al. 2008 [67, 65]                                          | et al. 2008 [67, Mixed                                                                                                           |          | Yes                                                                                                                                     | [65] also modifies inviscid flux and multiplies momentum upwinding by $M$                                |
| Li & Gu 2008/9 All-speed<br>Roe [32, 35]                               | Convective                                                                                                                       |          | Yes                                                                                                                                     | No $\delta p$ , and introduces only $\mu_{11}$ as in [44]. [34] ex-                                      |
| Li & Gu 2010 time-<br>marching MIM [34]                                | Convective                                                                                                                       | Mixed    | Yes                                                                                                                                     | tends [32, 35] with $\Delta t$ dependence in $\mu_{11}$                                                  |
| Dellacherie 2010 [15] and Mixed/Convective Dellacherie et al 2016 [14] |                                                                                                                                  | Yes      | Recommends mixed scaling with $\delta p$ removed [15] or reduced by $M$ [14]. Analysis of $\delta p$ applies also to convective scaling |                                                                                                          |
| Rieper 2011 (LMRoe) [53]                                               | Mixed                                                                                                                            |          | Yes                                                                                                                                     | Reduces $\mu_{22}$ in Roe scheme by factor of $M$                                                        |
| Sachdev et al. 2012 [58]                                               | Convective                                                                                                                       | Mixed    | No/(UT/LT)                                                                                                                              | Two adaptive schemes with different forms for small timesteps                                            |
| Оßwald et al. 2016<br>(L2Roe) [49]                                     | Mixed                                                                                                                            |          | Yes                                                                                                                                     | Modifies LMRoe to reduce vorticity upwinding by factor of $M$                                            |
| Caraeni & Weiss 2017 [6]                                               | Convective                                                                                                                       | Mixed    | No                                                                                                                                      | Modified $\mu_{11}$ of [78] to approach the mixed scheme as $\Delta t \to 0$                             |
| Bruel et al. 2019 [5] Mixed                                            |                                                                                                                                  | No       | Restores off-diagonal terms to mixed scheme for acoustic flows.                                                                         |                                                                                                          |

Table 4: Diffusion scaling of a number of existing low Mach number Roe-type or central difference schemes. Some schemes have different scalings when the timestep is calculated using the convective CFL  $(\sigma_u)$  and  $M_u \sim \mathcal{O}(M)$ , or using the acoustic CFL  $(\sigma_a)$  and  $M_u \sim \mathcal{O}(1)$ . A diagonal scheme has vanishing  $\mu_{12}, \mu_{21} \sim o(M^0)$ . UT stands for upper triangular, with vanishing  $\mu_{21} \sim o(M^0)$  and  $\mu_{12} \sim \mathcal{O}(M^0)$ . LT stands for lower triangular with vanishing  $\mu_{12} \sim o(M^0)$  and  $\mu_{21} \sim \mathcal{O}(M^0)$ .

 $\mathcal{O}(M^{-2})$  and  $\mathcal{O}(M^{-1})$  relations are identical, and are:

$$0 = \sum_{\mathcal{V}(i)}^{l} S_{il} \rho^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0,1)}$$
(44a)

$$\sum_{\mathcal{V}(i)}^{l} S_{il} p_l^{(0,1)} \underline{n}_{il} = \sum_{\mathcal{V}(i)}^{l} S_{il} (\underline{\rho u}^{(0)} \frac{\nu_{11}}{\rho |v|} + \frac{U_{il}}{|U_{il}|} \nu_{21} \underline{n}_{il}) \Delta_{il} p^{(0,1)}$$
(44b)

$$0 = \sum_{\mathcal{V}(i)}^{l} S_{il} \rho H^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0,1)}$$
(44c)

The right-hand sides of (44a,44c) are in the form of the discrete Laplacian. In N dimensions, equations (44) are two elliptic systems each of N+2 equations for  $p^{(0,1)}$ , which are the discrete equivalents of the continuous relations (26a,27a) and (26b,27b). Under a small number of reasonable assumptions, they enforce constant  $p^{(0,1)}$  over the entire domain [26]. Using the non-dimensional  $\mathcal{O}(M^0)$  equation of state  $p^{(0)}=(\gamma-1)\rho E^{(0)}=\rho^{(0)}T^{(0)}$  [48], the  $\mathcal{O}(M^0)$  relations are:

$$\Omega_{i} \frac{d\rho_{i}^{(0)}}{dt} + \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \rho_{l}^{(0)} U_{l}^{(0)}$$

$$= \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} (|U^{(0)}| \Delta_{il} \rho^{(0)} + \rho^{(0)} (\frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(2)} + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U^{(0)}))$$

$$\Omega_{i} \frac{d\rho u_{-i}^{(0)}}{dt} + \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} (\underline{\rho} u_{l}^{(0)} U_{l}^{(0)} + p_{l}^{(2)} \underline{n}_{il})$$

$$= \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} (|U^{(0)}| \Delta_{il} \underline{\rho} u^{(0)} + \underline{\rho} u^{(0)} (\frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(2)} + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U^{(0)}) + \underline{n}_{il} (\frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p^{(2)} + \rho |v| \nu_{22} \Delta_{il} U^{(0)}))$$

$$\Omega_{i} \frac{d\rho E_{i}^{(0)}}{dt} + \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \rho H_{l}^{(0)} U_{l}^{(0)} = \frac{\Omega_{i}}{\gamma - 1} \frac{d\rho^{(0)}}{dt} + \frac{\rho H^{(0)}}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} U_{l}^{(0)}$$

$$= \frac{\rho H^{(0)}}{2} \sum_{i=1}^{l} S_{il} (\frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(2)} + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U^{(0)})$$
(45a)

(45a)

Where we have used constant  $p^{(0)}$  to imply constant  $\rho E^{(0)}$  and  $\rho H^{(0)}$  to simplify relation (45c). If the zeroth order entropy  $s^{(0)}$  is constant, then the zeroth order temperature and density  $T^{(0)}$  and  $\rho^{(0)}$  also become constant. The second term in (45a) is the discrete divergence, and relations (45a,45c) are the discrete equivalents of the continuity relation (27c). At steady state on a regular grid, the pressure diffusion takes exactly the form of the Brezzi-Pitkäranta stabilisation [4] used with a finite-volume scheme by Eymard et al. [19], so this scheme is expected to be free of chequerboard instabilities. The momentum relation (45b) is the discrete equivalent of relation (26c)<sup>†</sup>, having all the necessary terms in  $\mathcal{L}$ , and properly balanced diffusion in  $\mathcal{R}$ . The discrete single scale expansion with the convective diffusion scaling therefore reproduces all the necessary relations for the convective low Mach number limit, so is a consistent scheme for this limit.

Next, we carry out the two timescale expansion with the convective diffusion scaling. The  $\mathcal{O}(M^{-2})$  relations are identical to those from the single-timescale expansion, so we assume  $p^{(0)}$  and related zeroth order thermodynamic quantities are constant again. The  $\mathcal{O}(M^{-1})$  relations are:

$$\Omega_i \frac{d\rho_i^{(0)}}{d\tau} = \frac{1}{2} \sum_{V(i)}^l S_{il} \rho^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(1)}$$
(46a)

$$\Omega_{i} \frac{d\underline{\rho}\underline{u}_{i}^{(0)}}{d\tau} + \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} p_{l}^{(1)} \underline{n}_{il} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} (\underline{\rho}\underline{u}^{(0)} \frac{\nu_{11}}{\rho |v|} + \frac{U_{il}}{|U_{il}|} \nu_{21} \underline{n}_{il}) \Delta_{il} p^{(1)}$$
(46b)

$$\Omega_i \frac{d\rho E_i^{(0)}}{d\tau} = \frac{\rho H^{(0)}}{2} \sum_{\nu(i)}^l S_{il} \frac{\nu_{11}}{\rho |\nu|} \Delta_{il} p^{(1)}$$
(46c)

Using the  $\mathcal{O}(M^0)$  equation of state, we can see that relations (46a,46c) are the discrete equivalents to (29b), which is a parabolic equation for  $p^{(1)}$ , assuming  $\partial_{\tau}p^{(0)} \to 0$ . This means  $p^{(1)}$  will become

Properly, it is a discrete combination of both (26c) and (27c), as can be seen from the diffusion terms.

constant on the acoustic timescale  $\tau$  unless  $p^{(0)}$  is forced on the acoustic timescale (which would violate the physical relation (11a)), so the convective diffusion scaling is indeed unsuitable for acoustic or mixed convective-acoustic simulations.

## 5.3.2 Expansion with acoustic diffusion

The single timescale expansion of the discrete equations (39) with acoustic diffusion scaling leads to the  $\mathcal{O}(M^{-2})$  momentum relation:

$$\sum_{\mathcal{V}(i)}^{l} S_{il} p_l^{(0)} \underline{n}_{il} = \sum_{\mathcal{V}(i)}^{l} S_{il} \frac{U_{il}}{|U_{il}|} \nu_{21} \underline{n}_{il} \Delta_{il} p^{(0)}$$
(47)

and the  $\mathcal{O}(M^{-1})$  relations:

$$0 = \sum_{\nu(i)}^{l} S_{il} \rho^{(0)} \frac{\nu_{11}}{\rho |\nu|} \Delta_{il} p^{(0)}$$
(48a)

$$\sum_{\mathcal{V}(i)}^{l} S_{il} p_l^{(1)} \underline{n}_{il} = \sum_{\mathcal{V}(i)}^{l} S_{il} \underline{n}_{il} \left( \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p^{(1)} + \rho |v| \nu_{22} \Delta_{il} U^{(0)} \right)$$
(48b)

$$0 = \sum_{\mathcal{V}(i)}^{l} S_{il} \rho H^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0)}$$
(48c)

The relations (47,48a,48c) are the same elliptic system for  $p^{(0)}$  as we have seen previously, so we assume  $p^{(0)}$  is constant. We have used a constant  $p^{(0)}$  to simplify relation (48b), which is a discretisation of the continuous relation (30b). The first order pressure  $p^{(1)}$  cannot be constant unless the jumps in normal velocity  $\Delta_{il}U^{(0)}$  are zero at every cell face - this is even clearer if the scheme is diagonal, which the acoustic scheme usually is. As previously stated, this condition is actually enforced for simplex meshes, leading to a divergence free solution with the correct pressure scaling [54, 55, 24, 17]. This is why we restricted our analysis to quadrilaterals and hexahedra at the beginning of this section. For these cell types, (48b) forces either trivial velocity fields or non-constant  $p^{(1)}$ , so this scheme is unsuitable for convective low Mach number flows.

The two timescale expansion with acoustic diffusion scaling leads to the same  $\mathcal{O}(M^{-2})$  momentum relation (47), and the following  $\mathcal{O}(M^{-1})$  relations:

$$\Omega_i \frac{d\rho_i^{(0)}}{d\tau} = \frac{1}{2} \sum_{\mathcal{V}(i)}^l S_{il} \rho^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0)}$$
(49a)

$$\Omega_{i} \frac{d\rho u_{i}^{(0)}}{d\tau} + \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} p_{l}^{(1)} \underline{n}_{il} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \underline{n}_{il} \left( \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p^{(1)} + \rho |v| \nu_{22} \Delta_{il} U^{(0)} \right)$$
(49b)

$$\Omega_i \frac{d\rho E_i^{(0)}}{d\tau} = \frac{1}{2} \sum_{\mathcal{V}(i)}^l S_{il} \rho H^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0)}$$
(49c)

The  $\mathcal{O}(M^0)$  density relation is:

$$\Omega_{i}\left(\frac{d\rho_{i}^{(1)}}{d\tau} + \frac{d\rho_{i}^{(0)}}{dt}\right) + \frac{1}{2}\sum_{\nu(i)}^{l} S_{il}\rho^{(0)}U_{l}^{(0)} = \frac{1}{2}\sum_{\nu(i)}^{l} S_{il}\left(|U^{(0)}|\Delta_{il}\rho^{(0)} + \rho^{(0)}\left(\frac{\nu_{11}}{\rho|\nu|}\Delta_{il}p^{(1)} + \frac{U_{il}}{|U_{il}|}\nu_{12}\Delta_{il}U^{(0)}\right)\right)$$
(50)

Relations (47,49a,49c) are now an unsteady elliptic system on the acoustic timescale for  $p^{(0)}$ , leading variations in  $p^{(0)}$  to be dissipated on the acoustic timescale. Constant  $p^{(0)}$  has been used to simplify relations (49b,50), which for constant entropy (i.e. constant  $p^{(0)}$ ) are the discrete equivalents of the linear acoustics relations (32b,33b). This shows that the scheme is suitable for simulating purely acoustic low Mach number flows (for example 1D shocktube flows), but not mixed convective-acoustic flows due to the issues highlighted in the single-scale expansion.

#### 5.3.3 Expansion with mixed diffusion

Lastly, we carry out the expansions with the mixed diffusion scaling. For much of the discussion we assume that the scheme is diagonal for simplicity, but we have retained both off-diagonal diffusion terms

in the expansions on the understanding that at least one of  $\mu_{12}$  and  $\mu_{21}$  must vanish according to the discussion in section 5.2.3. The single timescale expansion leads to the  $\mathcal{O}(M^{-2})$  momentum relation:

$$\sum_{\mathcal{V}(i)}^{l} S_{il} p_l^{(0)} \underline{n}_{il} = \sum_{\mathcal{V}(i)}^{l} S_{il} \frac{U_{il}}{|U_{il}|} \nu_{21} \underline{n}_{il} \Delta_{il} p^{(0)}$$
(51)

and the  $\mathcal{O}(M^{-1})$  relations:

$$0 = \sum_{\mathcal{V}(i)}^{l} S_{il} \rho^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0)}$$
(52a)

$$\sum_{\mathcal{V}(i)}^{l} S_{il} p_l^{(1)} \underline{n}_{il} = \sum_{\mathcal{V}(i)}^{l} S_{il} \left( \underline{\rho} \underline{u}^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0)} + \frac{U_{il}}{|U_{il}|} \nu_{21} \underline{n}_{il} \Delta_{il} p^{(1)} \right)$$
(52b)

$$0 = \sum_{\mathcal{V}(i)}^{l} S_{il} \rho H^{(0)} \frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(0)}$$
(52c)

Relations (51,52a,52c) are the elliptic system for  $p^{(0)}$ . Assuming constant  $p^{(0)}$ , (52b) becomes an elliptic equation for  $p^{(1)}$ , for which constant  $p^{(1)}$  may not be the only solution. For diagonal or upper triangular schemes the right hand side of relation (52b) vanishes. On a regular 2D Cartesian grid, the momentum relations then become  $p^{(1)}_{i+1,j} - p^{(1)}_{i-1,j} = 0$  and  $p^{(1)}_{i,j+1} - p^{(1)}_{i,j-1} = 0$ , which will, on their own, admit chequer-board modes. The  $\mathcal{O}(M^0)$  relations are:

$$\Omega_{i} \frac{d\rho_{i}^{(0)}}{dt} + \frac{1}{2} \sum_{\nu(i)}^{l} S_{il} \rho^{(0)} U_{l}^{(0)} \qquad (53a)$$

$$= \frac{1}{2} \sum_{\nu(i)}^{l} S_{il} (|U^{(0)}| \Delta_{il} \rho^{(0)} + \rho^{(0)} (\frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(1)} + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U^{(0)}))$$

$$\Omega_{i} \frac{d\rho u_{i}^{(0)}}{dt} + \frac{1}{2} \sum_{\nu(i)}^{l} S_{il} (\underline{\rho} u_{l}^{(0)} U_{l}^{(0)} + p_{l}^{(2)} \underline{n}_{il})$$

$$= \frac{1}{2} \sum_{\nu(i)}^{l} S_{il} (|U^{(0)}| \Delta_{il} \underline{\rho} u^{(0)} + \underline{\rho} u^{(0)} (\frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(1)} + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U^{(0)}) + \underline{n}_{il} (\frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p^{(2)} + \rho |v| \nu_{22} \Delta_{il} U^{(0)}))$$

$$\Omega_{i} \frac{d\rho E_{i}^{(0)}}{dt} + \frac{\rho H^{(0)}}{2} \sum_{\nu(i)}^{l} S_{il} U_{l}^{(0)}$$

$$= \frac{\rho H^{(0)}}{2} \sum_{\nu(i)}^{l} S_{il} (\frac{\nu_{11}}{\rho |v|} \Delta_{il} p^{(1)} + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U^{(0)})$$
(53a)

Relations (53a,53c) are discrete equivalents of the continuity relation (35b). Assuming  $\mu_{12} \sim o(M^0)$ , at steady state these relations mean that  $p^{(1)}$  is free of chequer-boards *iff* the discrete divergence of the zeroth order velocity is zero. In practice, we know of no case - either in the literature or in our own studies - where this scheme has produced steady solutions with non-constant  $p^{(1)}$ , and Rieper [53] provides an argument that this scheme will damp the divergence on the convective timescale on Cartesian grids. Although approaching a divergence free solution as  $M \to 0$  is appealing, it means that this scheme will be susceptible to chequerboard-modes on  $p^{(2)}$ , as we shall see in numerical examples later. Lastly, the momentum relation (53b) is consistent with the continuous relation (34c). The scheme therefore reproduces all the relevant asymptotic relations, and is suitable for simulating convective low Mach number flow, but with the caveat of potential chequer-board modes on  $p^{(2)}$ .

The two timescale expansion with mixed diffusion scaling gives the same  $\mathcal{O}(M^{-2})$  momentum relation and  $\mathcal{O}(M^{-1})$  density and energy relations as for the two timescale expansion with acoustic diffusion (47,49a,49c), which lead to constant  $p^{(0)}$ . Subsequently, the  $\mathcal{O}(M^{-1})$  momentum relation is:

$$\Omega_i \frac{d\underline{\rho}\underline{u}_i^{(0)}}{d\tau} + \sum_{\mathcal{V}(i)}^l S_{il} p_l^{(1)} \underline{n}_{il} = \sum_{\mathcal{V}(i)}^l S_{il} \frac{U_{il}}{|U_{il}|} \nu_{21} \underline{n}_{il} \Delta_{il} p^{(1)}$$

$$(54)$$

and the  $\mathcal{O}(M^0)$  relations are:

$$\Omega_{i}\left(\frac{d\rho_{i}^{(1)}}{d\tau} + \frac{d\rho_{i}^{(0)}}{dt}\right) + \frac{1}{2}\sum_{\nu(i)}^{l} S_{il}\rho^{(0)}U_{l}^{(0)}$$

$$= \frac{1}{2}\sum_{\nu(i)}^{l} S_{il}\left(|U^{(0)}|\Delta_{il}\rho^{(0)} + \rho^{(0)}\left(\frac{\nu_{11}}{\rho|\nu|}\Delta_{il}p^{(1)} + \frac{U_{il}}{|U_{il}|}\nu_{12}\Delta_{il}U^{(0)}\right)\right)$$

$$\Omega_{i}\left(\frac{d\rho u_{i}^{(1)}}{d\tau} + \frac{d\rho u_{i}^{(0)}}{dt}\right) + \frac{1}{2}\sum_{\nu(i)}^{l} S_{il}\left(\underline{\rho u}_{l}^{(0)}U_{l}^{(0)} + p_{l}^{(2)}\underline{n}_{il}\right)$$

$$= \frac{1}{2}\sum_{\nu(i)}^{l} S_{il}\left(|U^{(0)}|\Delta_{il}\underline{\rho u}^{(0)} + \underline{\rho u}^{(0)}\left(\frac{\nu_{11}}{\rho|\nu|}\Delta_{il}p^{(1)} + \frac{U_{il}}{|U_{il}|}\nu_{12}\Delta_{il}U^{(0)}\right) + \underline{n}_{il}\left(\frac{U_{il}}{|U_{il}|}\nu_{21}\Delta_{il}p^{(2)} + \rho|\nu|\nu_{22}\Delta_{il}U^{(0)}\right)\right)$$

$$\Omega_{i}\left(\frac{d\rho E_{i}^{(1)}}{d\tau} + \frac{d\rho E_{i}^{(0)}}{dt}\right) + \frac{\rho H^{(0)}}{2}\sum_{\nu(i)}^{l} S_{il}U_{l}^{(0)}$$

$$= \frac{\rho H^{(0)}}{2}\sum_{\nu(i)}^{l} S_{il}\left(\frac{\nu_{11}}{\rho|\nu|}\Delta_{il}p^{(1)} + \frac{U_{il}}{|U_{il}|}\nu_{12}\Delta_{il}U^{(0)}\right)$$
(55a)

The relations (54) and (55a,55c) are the discrete equivalents of the linear acoustics relations (36b,37b). It can be seen that, as in the continuous analysis, the velocity diffusion vanishes from the acoustic momentum relation (54), while the acoustic relations (55a,55c) retain the pressure diffusion term. Therefore this scheme is consistent for acoustic flows, however potentially has instabilities in the acoustic velocity variations. The momentum relation (55b) is the discrete equivalent of (36c), confirming that this scheme is suitable for mixed convective-acoustic flows.

The discrete expansions of the first order finite-volume scheme have confirmed that all findings of the continuous analysis in sections 3 and 4 carry over to the discrete setting for this particular discrete form, with each diffusion scaling behaving as expected for both the singleand multiple-scale limits.

#### 5.4 Stability of the discrete scheme

We now consider the stability of the discrete schemes by finding expressions for the eigenvalues of the artificial diffusion Jacobian and the von Neumann symbols of the fully discrete first order scheme.

## 5.4.1 Eigenvalues of the artificial diffusion Jacobian

The eigenvalues of the diffusive flux Jacobian (40) are:

$$\lambda_{1,2} = \frac{M^{-2}|u|\mu_{11} + |u|\mu_{22} + 2|U| \pm \sqrt{|u|^2(M^{-2}\mu_{11} - \mu_{22})^2 - M^{-2}p\mu_{12}\mu_{21}}}{2}$$
(56a)

$$\lambda_{3.4} = |U| \tag{56b}$$

If  $\mu_{12}\mu_{21} = 0$  then:

$$\lambda_{1,2,3,4} = \{ M^{-2} | u | \mu_{11} + | U |, | u | \mu_{22} + | U |, | U |, | U | \}$$
(57)

and the spectral radius is:

$$\Lambda(\underline{A}) = \max(\lambda_i) = |u| \max(M^{-2}\mu_{11}, \mu_{22}) + |U|$$
(58)

Using the scalings in table 3, we see that the scaling of  $\Lambda(\underline{\underline{A}})$ , either from (58) or from a leading order approximation of (56), agrees with the trace estimates (38) for all three diffusion scalings.

#### 5.4.2 von Neumann symbols of the first order scheme

We now find the stability limits for the linearised first order scheme with each diffusion scaling using von Neumann analysis. Discretising equation (39) using central differences in space and forward Euler in time, we have the linearised update for the solution vector  $\underline{q}_i^n$  in cell i at time level n:

$$\frac{\underline{q}_i^{n+1} - \underline{q}_i^n}{\Delta t} + \underline{\underline{J}} \left( \frac{\underline{q}_{i+1}^n - \underline{q}_{i-1}^n}{2\Delta x} \right) = \underline{\underline{A}} \left( \frac{\underline{q}_{i+1}^n - 2\underline{q}_i^n + \underline{q}_{i-1}^n}{2\Delta x} \right)$$
(59)

![](_page_23_Figure_0.jpeg)

Figure 1: Real and imaginary components for the von Neumann symbols for the convective scheme (63) with  $(\sigma_a M^{-1}) = 0.6$ .  $\lambda_1^c$  (left) and  $\lambda_2^c$  (right).

where  $\underline{\underline{J}}$  is the physical flux Jacobian and  $\underline{\underline{A}}$  is the diffusive flux Jacobian. In this section we assume all diffusion matrices are diagonal in the entropy variables, as it simplifies the presentation. For stability analysis with the off-diagonal terms included, see [2] for the convective scheme and [5] for the mixed scheme. Substituting  $\underline{q}_j^n = \underline{\hat{q}}^n e^{ikj\Delta x}$ , where  $k = n\pi\Delta x/l$  is the discrete wavenumber, we find the amplification matrix in Fourier space is:

$$\underline{\underline{\hat{G}}}(k) = \underline{\underline{\mathcal{I}}} - \frac{\Delta t}{\Delta x} \left( 2S_{k/2}^2 \underline{\underline{A}} + iS_k \underline{\underline{I}} \right)$$
 (60)

where  $\underline{\underline{I}}$  is the identity matrix,  $S_{k/2} = \sin(k/2)$  and  $S_k = \sin(k)$ . For stability, the eigenvalues  $\lambda_i$  of  $(\underline{\underline{\hat{G}}}(k))$  must be  $|\lambda_i| \leq 1 \,\,\forall \, k$ . We consider the 1D equations, as stability of the 1D scheme is a necessary condition for stability of the multidimensional schemes. In the entropy variables the entropy equation decouples and the associated eigenvalue is identical for all three diffusion schemes, and is exactly that of the first order upwind scheme for scalar advection:

$$\lambda_3 = 1 - (\sigma_a M)(2S_{k/2}^2 + iS_k) \tag{61}$$

where  $\sigma_a = a\Delta t/\Delta x$  is the acoustic CFL number. This eigenvalue depends on the convective CFL number  $\sigma_a M$ , as we would expect, and is stable for  $0 \le (\sigma_a M) \le 1$ . The eigenvalues for the pressure-velocity subsystem using the acoustic diffusion scheme are:

$$\lambda_{1,2}^{a} = 1 - (\sigma_a M)(2S_{k/2}^2 + iS_k) - \sigma_a(2S_{k/2}^2 \pm iS_k)$$
(62)

Where we can identify the upwind discretisations of the convective system in the second term, and the forward/backward travelling acoustic waves in the third term. The stability is dominated by the acoustic terms so, to leading order, the scheme is stable for  $0 \le \sigma_a \le 1$ . The full expression for the eigenvalues for the convective scheme is:

$$\lambda_{1,2}^c = 1 - (\sigma_a M)(2S_{k/2}^2 + iS_k) - \sigma_a \left( M^{-1} S_{k/2}^2 \pm i\sqrt{S_k^2 - M^{-2} S_{k/2}^4} \right)$$
(63)

The second term is the convective subsystem, the third term is due to the large  $(\sigma_a M^{-1})$  pressure diffusion, and the radicand has contributions from both the pressure diffusion and from the acoustic waves. The leading order term is the  $(\sigma_a M^{-1})$  pressure diffusion, so the stability limit will scale a factor of M worse than the usual stability limit (which depends on  $\sigma_a$ ), as originally shown for preconditioned diffusion by Birken & Meister [2] and as predicted for all convective schemes by the spectral radius estimates (38) in

section 4. Contours of the real and imaginary parts of  $\lambda_1^c$  and  $\lambda_2^c$  for varying k and M are plotted in figure 1. The eigenvalues have two distinct regions, either side of the locus  $M = \tan(k/2)/2$  where the radicand goes to zero (labelled R = 0 in figure 1). In the low wavenumber region to the left of this locus the radicand is positive so this term is imaginary, and to the right in the moderate/high wavenumber region the radicand is negative and this term is real. We can simplify the square root in each region using binomial expansions for k << M and k >> M. In the low wavenumber region this leads to the following expression:

$$\lambda_{1,2}^c = 1 - (\sigma_a M)(2S_{k/2}^2 + iS_k) - (\sigma_a M^{-1})S_{k/2}^2 \pm i\sigma_a S_k$$
(64)

and in the high wavenumber region:

$$\lambda_1^c = 1 - (\sigma_a M)(2 + iS_k) \tag{65a}$$

$$\lambda_2^c = 1 - (\sigma_a M)((4S_{k/2}^2 - 2) + iS_k) - (\sigma_a M^{-1})(2S_{k/2}^2)$$
(65b)

These expressions can be interpreted from the perspective of a multiple-space scale approach, where acoustic and convective phenomena occur on large and small spatial scales respectively, instead of the multiple-time scale approach used in the rest of the paper. In the low wavenumber 'acoustic' region, the leading order imaginary term for both eigenvalues is due to the  $\sigma_a$  forward/backward travelling acoustic waves. However, these terms are dominated by the  $(\sigma_a M^{-1})$  diffusion, so to leading order both eigenvalues resemble those of a diffusion equation, corresponding to the rapid damping of acoustic waves by this scheme. Long wavelength variations in both pressure and (normal) velocity are due to acoustic effects and must be damped.

On the other hand, in the high wavenumber 'convective' region the leading order imaginary terms in both eigenvalues are on the  $(\sigma_a M)$  convective scale, and only  $\lambda_2^c$  is dominated by the  $(\sigma_a M^{-1})$  diffusion. At short wavelengths, (normal) velocity is no longer an acoustic quantity, but joins entropy (and vorticity in 2/3D) as a convected quantity. The pressure is still dominated by the large diffusion, which means that the Brezzi-Pitkäranta type stabilisation is retained across all wavenumbers.

The boundary between the two regions reduces approximately linearly with M because of the scale separation between acoustic and convective phenomena as  $M \to 0$ . In real applications implicit timestepping is used for this scheme to overcome the stringent  $(\sigma_a M^{-1})$  stability limit, but the von Neumann analysis of the explicit first order scheme provides an interesting insight into the behaviour of the scheme.

The first two eigenvalues of the mixed scheme are:

$$\lambda_{1,2}^{m} = 1 - (\sigma_a M)(2S_{k/2}^2 + iS_k) - \sigma_a \left(S_{k/2}^2 \pm i\sqrt{S_k^2 - S_{k/2}^4}\right)$$
(66)

We can identify the convective second term proportional to  $(\sigma_a M)$ . The acoustic third term is proportional to  $(\sigma_a)$  but the diffusive term  $S_{k/2}^2$  is half that of the standard upwind scheme, and the advective term has an additional contribution from the second term in the radicand. The magnitudes and imaginary component of the eigenvalues  $\lambda_{1,2}^m$ , ignoring the convective term, are plotted in figure 2 alongside the equivalent values for the standard upwind scheme. For the low wavenumber range  $0 \le k < 2 \arctan(2) \approx 2.2$ , the radicand is real and the scheme behaves similarly to the upwind scheme except with lower diffusion, evident in figure 2a, and an additional dispersion error, evident in figure 2b. For the high wavenumber range  $2 \arctan(2) \le k \le \pi$ , the radicand is imaginary and the scheme is entirely diffusive. Figure 2a shows that  $|\lambda_i|$  bifurcates in the high wavenumber range. As  $\pi - k = k' \to 0$ , the eigenvalues  $\lambda_{1,2}^m$  can be expressed as:

$$\lambda_1^m \approx 1 - (\sigma_a)(2S_{k/2}^2) + \mathcal{O}(k'^2\sigma_a) \tag{67a}$$

$$\lambda_2^m \approx 1 - (\sigma_a)(2C_{k/2}^2) + \mathcal{O}(k'^4\sigma_a) \approx 1 - \mathcal{O}(k'^2\sigma_a) \tag{67b}$$

where  $C_{k/2} = \cos(k/2)$ .  $\lambda_1^m$  approaches the pure diffusion scheme as  $k \to \pi$ , as does the upwind scheme. To leading order,  $\lambda_2^m$  approaches 1 independently of  $\sigma_a$ , indicating that there is a grid mode which is undamped by the mixed scheme. This was shown by Dellacherie [16] by considering the time evolution of the energy of the grid mode using the mixed scheme. The low wavenumber range is stable under the reduced CFL condition  $0 < \sigma_a < 0.5$ , which is the same as found by Dellacherie [15], and Bruel et al. [5]. The high wavenumber range is (linearly) stable under the CFL condition  $0 < \sigma_a < 1$ , although nonlinear interactions in the full scheme will excite the undamped grid mode in many circumstances.

Lastly, we return to the question of reducing the convective upwinding by a factor of M, as done by Thornber [65] and Oßwald et al. [49] in the velocity equations. The eigenvalue of the upwind scheme for the convective terms (61) using a diffusion coefficient of  $\alpha |U|/2$  instead of |U|/2 is:

$$\lambda^{\alpha} = 1 - (\sigma_a M)(2\alpha S_{k/2}^2 - iS_k) \tag{68}$$

which is linearly stable under the CFL condition  $0 \le (\sigma_a M) \le \alpha$ . If  $\alpha = M$  and  $\sigma_a \approx 1$ , then the upwind scheme with reduced diffusion is linearly stable. This argument also justifies reducing the convective upwinding for every component, not just the velocity. However, because we rely on  $\sigma_a \approx 1$  this argument does not hold for preconditioned schemes where  $\sigma_u \approx 1$  and  $\sigma_a \approx 1/M$ .

![](_page_25_Figure_0.jpeg)

Figure 2: von Neumann symbols for the 1D mixed diffusion scheme with forward Euler time integration. (a) Magnitude of the von Neumann symbols (amplification). (b) Imaginary component of the von Neumann symbols (dispersion).

# 6 Numerical examples

We now show numerical examples which demonstrate the behaviours of the three low Mach number schemes. The discrete equations for a cell-centred finite volume scheme [\(39\)](#page-14-4) are solved using a first order scheme in both time and space. The first order explicit Euler scheme is used for time integration. Diagonal diffusion is used for all examples. Following the discussion of section [5.2.3,](#page-17-0) the mixed scheme with full diffusion is unstable with the diffusion form [\(41,](#page-15-2)[42\)](#page-15-3), and upper/lower triangular mixed diffusion was found by Potsdam et al [\[50\]](#page-34-0) and Sachdev et al [\[58\]](#page-35-10) to have the same behaviour as the diagonal scheme. For the acoustic scheme diagonal diffusion is equivalent to standard upwinding (see appendix [A\)](#page-36-0). For the convective scheme almost no difference was seen between full or diagonal diffusion, so the diagonal diffusion results are shown for consistency with the other schemes. If higher order reconstructions and time integration are used, higher resolution results are obtained, but the qualitative behaviour remains the same, although smaller differences between the fluxes are observed due to the reduction in the diffusion. All simulations are carried out with an ideal gas with ratio of specific heats γ = 1.4 and specific gas constant R = 287.058.

## 6.1 One dimensional examples

In one dimension, the only solutions which are compatible with the convective limit [\(4,](#page-3-9)[5\)](#page-4-10) are trivial, with both constant pressure and velocity [\[15\]](#page-32-4). As such, all flow variations are either acoustic or entropy waves, and one dimensional examples can be used to isolate the behaviour of the numerical fluxes for purely acoustic flow.

## 6.1.1 Isolated soundwave

The first test case is an isolated soundwave in one dimension. This test case will show the differences between the fluxes for a smooth purely acoustic flow. The pressure p(x, t) is initialised with a sinusoidal profile, the density ρ(x, t) is initialised assuming isentropic flow, and the velocity u(x, t) is initialised using the Riemann invariant for a forward travelling sound wave:

$$p(x,0) = p_{\infty}(1 + M_{\infty}g(x))$$

$$\rho(x,0) = \rho_{\infty} \left(\frac{p}{p_{\infty}}\right)^{\frac{1}{\gamma}}$$

$$u(x,0) = u_{\infty} + \frac{2(a - a_{\infty})}{\gamma - 1}$$
(69)

![](_page_26_Figure_0.jpeg)

(a) Isolated soundwave after travelling for a single period. (b) Low Mach number shocktube. Inset figure is a close-up of 16 ≤ x ≤ 112 and 0.96 ≤ y ≤ 1.04.

Figure 3: 1D acoustic examples

using the following conditions:

$$p_{\infty} = 1$$
,  $\rho_{\infty} = 1$ ,  $M_{\infty} = 0.01$ ,  $g(x) = c_0 \sin(2\pi x/l)$ ,  $c_0 = 0.1$ 

Figure [3a](#page-26-1) shows the non-dimensional gauge pressure distributions (p(x, t) − p∞)/p<sup>∞</sup> after one period using 16 points per wavelength and a CFL number σ = 0.125, requiring 128 timesteps for the acoustic and mixed fluxes, and 128/M = 12, 800 timesteps for the convective flux. The results for the acoustic and mixed schemes show the expected behaviour: the wave has travelled almost one wavelength and has been slightly diffused. The slight dispersion dispersion error of 2 −3% for the acoustic and mixed schemes is consistent with what would be expected from the von Neumann analysis in equations [\(62\)](#page-23-3) and [\(66\)](#page-24-0). The acoustic scheme has diffused the pressure peaks roughly twice as much as the mixed scheme. The convective scheme however has almost entirely smoothed out the wave, which has travelled only a small portion of a wavelength. These results are entirely consistent with our earlier analysis which found that the acoustic and mixed schemes are both suitable for smooth acoustic flow, whereas the convective scheme is overly diffusive of acoustic variations.

## 6.1.2 Low Mach number shocktube

The second one dimensional case is a low Mach number shock tube. This case demonstrates the performance of the different fluxes for discontinuous purely acoustic flow, and was first used by Sachdev et al. [\[58\]](#page-35-10) for testing adaptive schemes. The left and right initial states are:

$$p_L = 100028.04 \text{Pa}, \quad p_R = 100000 \text{Pa}, \quad u_{L/R} = 0, \quad T_{L/R} = 300 \text{K}$$

The very small pressure difference produces a contact wave moving to the right at M<sup>∞</sup> = 0.0001 between two receding weak shockwaves. Figure [3b](#page-26-1) shows a close-up of the Mach number distributions between the two shock waves obtained with the acoustic and the mixed flux schemes after 96 timesteps with a CFL of σ = 0.4. The acoustic flux gives a monotone solution, as expected for a first order, upwind scheme. However, the solution found with the mixed flux has significant oscillations originating at the shockwaves - an undamped grid mode as predicted by the von Neumann analysis.

From the one-dimensional examples we can verify that: the convective scheme is completely unsuitable for flow with acoustic variations; the acoustic scheme is suitable for flows with both smooth and discontinuous acoustic variations; and the mixed scheme is suitable for flows with smooth acoustic variations, but has too little diffusion to properly handle acoustic variations close to the grid scale.

## 6.2 Two dimensional examples

Non-trivial solutions to the convective limit exist in two and higher dimensions. We present two numerical examples in 2D, one which tests the schemes' capabilities for steady purely convective flow, and one which tests the capabilities for unsteady mixed convective-acoustic flow.

## 6.2.1 Circular cylinder

This classic test case will demonstrate the performance of each flux scheme for steady purely convective flow. The correct solutions should tend to the incompressible solution as  $M \to 0$ . The farfield state is:

$$\rho_{\infty} = 1, \quad u_{\infty} = 1, \quad M_{\infty} = 0.01$$

The background pressure is calculated from  $p_{\infty} = \rho_{\infty}/(\gamma M_{\infty}^2)$ . The cylinder is centred at the origin with radius r=1, and is meshed using an O-mesh which extends out to 50 radii from the origin, with 64 and 48 cells in the circumferential and radial directions. The first row of cells next to the cylinder has a radial height of 0.036r. The cell height grows at a rate of 1.11, with the final row having a radial height of 4.9r. Curvature-corrected boundary conditions are used at the inviscid cylinder wall [12], and ghost cells at freestream boundaries are set using the upstream velocity and entropy and the downstream static pressure. Convergence is reached by pseudo-timestepping at a CFL of  $\sigma=0.4$ , which is accelerated by local timestepping and preconditioning using Weiss & Smith's preconditioning matrix [78] (preconditioning is only used for the convective and mixed schemes, as preconditioning does not reduce the spectral radius of the acoustic scheme [21]).

The results for the convective flux is shown in figure 4b, and closely resemble the exact pressure distribution from the classical incompressible potential solution in figure 4a. The solution found with the acoustic flux is shown in figure 4c, and is visibly very different from the incompressible solution. The artificial diffusion dominates over the other terms, and the flow more closely resembles a Stokes flow than the inviscid solution, with significantly larger pressure variations. Lastly, the solution found with the mixed flux is shown in figure 4d and is very similar to the convective scheme and the incompressible solution. However, on close inspection there is a small chequer-board mode in the radial direction. This is more easily seen in figure 5 which shows the difference between the exact pressure and the pressure found with the mixed and convective flux schemes. The error for the convective flux is smooth with no chequer-board modes, however the error for the mixed flux clearly shows chequer-board modes in the radial direction.

#### 6.2.2 Soundwave-vortex interaction

The last numerical example is the interaction of a one-dimensional soundwave passing through a low Mach number Gresho vortex [22, 41] with stationary background conditions. This provides a very simple test of a numerical scheme's ability to not only simulate both convective and acoustic features, but also their interaction, and has been used previously by [5, 47].

The domain is  $(x,y) \in [-2h,2h) \times [-h,h)$  with periodic boundary conditions in both dimensions, and is discretised with  $256 \times 128$  square cells. The initial conditions are shown in figure 6a, composed of the superposition of a Gresho vortex centred at (0,0) with a diameter 0.4h, and a right-travelling soundwave with a Gaussian profile centred along the line (-h,y). The soundwave profile is calculated according to equations (69), with:

$$\rho_{\infty} = 1$$
,  $M_{\infty} = 0.01$ ,  $u_{\infty} = 0$ ,  $g(x) = c_1 G(x+h)$ 

Where G(x) is the Gaussian function with standard deviation of 0.1h,  $c_1$  is chosen such that the maximum acoustic velocity is 1, and  $p_{\infty} = \rho_{\infty}/(\gamma M_{\infty}^2)$  again. The Gresho vortex has circumferential velocity  $u_{\theta}(r)$  and pressure p(r) according to [47]:

$$u_{\theta}(r) = \begin{cases} 5r, & 0 \le r < 0.2\\ 2 - 5r, & 0.2 \le r < 0.4\\ 0, & 0.4 \le r \end{cases}$$
 (70)

$$p(r) = \begin{cases} p_0 + \frac{25}{2}r^2, & 0 \le r < 0.2\\ p_0 + \frac{25}{2}r^2 + 4(1 - 5r - \ln 0.2 + \ln r), & 0.2 \le r < 0.4\\ p_0 - 2 + 4\ln 2, & 0.4 \le r \end{cases}$$

$$4\ln 2.$$

$$(71)$$

with  $p_0 = p_{\infty} + 2 - 4 \ln 2$ .

The results for each flux after a single acoustic period  $(4h/a_{\infty})$  and after ten acoustic periods (equivalent to 1/10th of a convective period  $4h/(a_{\infty}M_{\infty})$ ) are shown in figures 6 and 7 respectively, using a CFL of  $\sigma = 0.4$ .

![](_page_28_Figure_0.jpeg)

Figure 4: Gauge pressure around a 2D cylinder at M = 0.01 using the various flux scalings.

![](_page_28_Figure_2.jpeg)

Figure 5: Gauge pressure error around a 2D cylinder for the convective and mixed flux scalings.

![](_page_29_Figure_0.jpeg)

Figure 6: Velocity profiles for an acoustic wave and Gresho vortex after 1 acoustic period τ = l/a with various flux schemes.

Figure [6b](#page-29-0) shows the solution found with the convective scheme after one acoustic time. The vortex shape is very well preserved, with barely any reduction in peak velocity. The soundwave on the other hand has been completely dissipated and is no longer visible. The only indication of the soundwave is the slight asymmetry in the vortex velocity over the y axis, which happens as the soundwave is smeared out over the domain.

The solution from the acoustic scheme is shown in figure [6c.](#page-29-0) The vortex has a drastically reduced peak velocity, and is misshapen, having become aligned with the grid due to the anisotropic artificial diffusion on the face-normal velocity. The soundwave is still visible, having travelled once around the domain. It is also damped, although not to an unsurprising degree for a first order upwind scheme, given there were fewer than 20 cells across the initial width of the soundwave.

As predicted by the asymptotic analysis, the results for the mixed scheme (figure [6d\)](#page-29-0) combine the favourable behaviour of both the convective and acoustic schemes. Both the vortex and the soundwave are well preserved. The vortex shows comparable diffusion to the convective scheme (although without the asymmetry), and the soundwave is less damped than that found with the acoustic scheme, in agreement with the von Neumann analysis in the previous section.

The results for the convective scheme after 10 acoustic periods (0.1 convective periods) are shown in [7b.](#page-30-0) The vortex shape is mostly preserved with only minimal distortion around the corners, where the flow is most misaligned with the grid. The peak vortex velocity is around 75% of the initial peak value. The acoustic scheme results are shown in figure [7c.](#page-30-0) The vortex is even more distorted than in figure [6c,](#page-29-0) and has polluted a large part of the domain. After integration over 10 periods, the soundwave is barely visible on the left side of the figure behind this distortion. The result for the mixed scheme again retains the favourable properties of the other two schemes (figure [7d\)](#page-30-0). The velocity magnitude of the vortex is similar to that found with the convective scheme, but with marginally less distortion to the shape. The soundwave is almost completely diffused, although less so than with the acoustic scheme.

We now investigate the effect of reduced convective upwinding, as proposed in [\[65,](#page-35-8) [53,](#page-34-5) [49\]](#page-34-6). Figure [8b](#page-30-1) shows the solution after 10 acoustic times found with the mixed scheme where the convective upwinding - the first term in [\(40\)](#page-14-3) - has been reduced by a factor of M. The peak vortex velocity is 87% of the initial value compared to 75% with the original mixed scheme (figure [8a\)](#page-30-1), a moderate improvement. Inspection of the diffusion matrix [\(40\)](#page-14-3) shows that for the mixed scheme the µ<sup>22</sup> term is of the same magnitude as the convective upwind term. Figure [8c](#page-30-1) shows the solution found by reducing only the µ<sup>22</sup> term by a factor of M. The peak velocity is 82% of the initial value, a comparable improvement to reducing the convective

![](_page_30_Figure_0.jpeg)

Figure 7: Velocity profiles for an acoustic wave and Gresho vortex after 10 acoustic periods (0.1 convective time periods l/u) with various flux schemes.

![](_page_30_Figure_2.jpeg)

Figure 8: Velocity profiles for an acoustic wave and Gresho vortex after 10 acoustic times τ with the mixed flux scheme, with various diffusion terms reduced by a factor of M.

upwind term. Figure [8d](#page-30-1) shows the results found by reducing both the convective upwinding and µ<sup>22</sup> terms by a factor of M compared to the original mixed scheme. The improvement is striking, with a peak vortex velocity of 99% of the initial value. Some small instabilities can be seen around the corners of the vortex, where the flow is most misaligned with the grid and the 1D von Neumann analysis of section [5](#page-14-0) is least valid. Although figure [8d](#page-30-1) shows a significant reduction in the diffusion without significant loss of stability, recall from the von Neumann analysis that the stability of this scheme relies on σ<sup>a</sup> ∼ O(1), making it unsuitable for preconditioned schemes where σ<sup>u</sup> ∼ O(1).

The convective upwinding reductions in [\[53,](#page-34-5) [65,](#page-35-8) [49\]](#page-34-6) do not exactly correspond to those discussed here. Rieper [\[53\]](#page-34-5) reduces the jump in the face-normal velocity component by a factor of M, corresponding to reducing the µ<sup>22</sup> term from the acoustic scheme scaling to the mixed scheme scaling, and reducing by a factor of M the convective upwinding on the face-normal component of velocity only. Thornber et al. [\[65\]](#page-35-8) and Oßwald et al. [\[49\]](#page-34-6) reduce the jumps in all velocity components by a factor of M, so for vortical flows are equivalent to the reduced convective upwinding scheme in figure [8b.](#page-30-1) However, reducing the entire convective upwinding term also reduces the diffusion on the entropy convection, so can be expected to give better results for cases with significant entropy variations such as heat transfer simulations.

In this section we have used numerical examples to verify the findings of sections [3,](#page-6-0) [4](#page-10-1) and [5.](#page-14-0) The one dimensional examples demonstrated that the acoustic scheme is well suited to purely acoustic flow, while the convective scheme is unable to resolve any acoustic phenomena. The mixed scheme is suitable for smooth acoustic flow - resolving these phenomena with less diffusion than the acoustic scheme - although is unstable for discontinuous acoustic flow, possessing an undamped grid mode. Using steady flow around a two dimensional cylinder, we verified that the acoustic scheme is unsuitable for purely convective flow, whilst the convective scheme approaches the incompressible limit as M → 0. The mixed scheme also approaches the incompressible limit, avoiding the catastrophic failure of the acoustic scheme, however we saw it is susceptible to chequerboard modes on the second order convective pressure. Finally we saw that neither the acoustic nor convective scheme is suitable for mixed convective-acoustic flow. The mixed scheme on the other hand can resolve both convective and acoustic features in the same flow, and can also achieve this with dramatically reduced convective upwind diffusion.

# 7 Conclusions

Low Mach number flows include a range of applications of scientific and engineering interest. Collocated, density based solvers for compressible flow are one method of simulating low Mach number flow, but care must be taken to ensure that the artificial diffusion does not compromise their accuracy in this regime. In this paper we have reviewed the behaviour of the artificial diffusion in this class of schemes at low Mach number using the modified equations. By considering both the convective and acoustic low Mach number limits, we have shown how three artificial diffusion scalings naturally arise in the entropy variables - one suitable for purely convective flow, one for purely acoustic flow, and one for flow with both convective and acoustic features. Singleand multiple-scale asymptotic expansions of the modified equations established the behaviours of each scheme for different flows. The convective scaling is compatible with convective flows, but damps out acoustic variations on a very fast timescale. The acoustic scaling is compatible with acoustic flows, but will produce spurious pressure waves for convective flow. The mixed scaling is compatible with both convective, acoustic, and mixed flow, but has vanishing diffusion on the convective pressure and the acoustic velocity, which may lead to pressure chequerboard modes and acoustic grid-mode instabilities respectively.

Transforming the artificial diffusion to a Roe-type finite volume scheme in the conserved variables enabled us to compare to a number of existing low Mach number methods. Each of these methods matched one of the three scalings, and our analysis agrees with previous theoretical analyses and wellknown behaviours of these existing methods. The convective and mixed scalings conform with previous guidelines for accurate schemes for convective low Mach number flow [\[15,](#page-32-4) [33\]](#page-33-9), but by considering accuracy requirements for both convective and acoustic effects, we were able to explain why there are two possible scalings suitable for the convective limit, and what their relative advantages and disadvantages are. The price of the mixed scheme's flexibility is its compromised stability and the additional constraints on the off-diagonal diffusion. It is the authors' belief that remedying the stability of this scheme, and in particular whether the off-diagonal terms can be leveraged to achieve this, is the most pertinent open question in this research area. Asymptotic expansion of the discrete equations showed that all findings of the continuous analysis apply for the first order Roe-type finite volume scheme, and von Neumann analysis confirmed the stability estimates obtained from the continuous analysis. Finally, four numerical examples demonstrated the performance of each diffusion scaling for acoustic, convective and mixed convective-acoustic flow.

There is a significant body of literature investigating this class of schemes in the limit of vanishing Mach number. We have shown that the most important behaviours of this class of schemes at low Mach number can be found and explained in a simple manner using the continuous modified equations in the entropy variables. This form can be used to compare schemes and predict their capabilities independently of the specific discretisation, as well as to provide guidelines for the development of novel low-Mach schemes.

# Competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Acknowledgements

The authors gratefully acknowledge support from the EPSRC Center for Doctoral Training in Gas Turbine Aerodynamics and Rolls-Royce plc. We also thank the anonymous reviewers whose feedback significantly improved the manuscript, particularly in the treatment of the off-diagonal diffusion terms.

# References

- [1] W. Barsukow. "Truly multi-dimensional all-speed schemes for the Euler equations on Cartesian grids". Journal of Computational Physics 435 (2021).
- [2] P. Birken and A. Meister. "Stability of Preconditioned Finite Volume Schemes at Low Mach Numbers". BIT Numerical Mathematics 45.3 (2005).
- [3] J. Blazek. Computational Fluid Dynamics: Principles and Applications. 3rd. Butterworth-Heinemann Ltd, 2015.
- [4] F. Brezzi and J. Pitk¨aranta. "On the Stabilization of Finite Element Approximations of the Stokes Equations". Efficient Solutions of Elliptic Systems: Proceedings of a GAMM-Seminar Kiel, January 27 to 29, 1984. Ed. by W. Hackbusch. Notes on Numerical Fluid Mechanics. Wiesbaden: Vieweg+Teubner Verlag, 1984.
- [5] P. Bruel et al. "A low Mach correction able to deal with low Mach acoustics". Journal of Computational Physics 378 (2019).
- [6] D. A. Caraeni and J. M. Weiss. "Unsteady Low-Mach Preconditioning for Roe Flux-Differencing Scheme". 23rd AIAA Computational Fluid Dynamics Conference. American Institute of Aeronautics and Astronautics, 2017.
- [7] E. P. Chassignet, C. Cenedese, and J. Verron, eds. Buoyancy-Driven Flows. Cambridge: Cambridge University Press, 2012.
- [8] S.-s. Chen et al. "An improved entropy-consistent Euler flux in low Mach number". Journal of Computational Science 27 (2018).
- [9] Y. .-. Choi and C. L. Merkle. "The Application of Preconditioning in Viscous Flows". Journal of Computational Physics 105.2 (1993).
- [10] A. J. Chorin. "A numerical method for solving incompressible viscous flow problems". Journal of Computational Physics 2.1 (1967).
- [11] F. Coletti et al. "Turbulent flow in rib-roughened channel under the effect of Coriolis and rotational buoyancy forces". Physics of Fluids 26.4 (2014).
- [12] A. Dadone and B. Grossman. "Surface boundary conditions for the numerical solution of the Euler equations". AIAA Journal 32.2 (1994).
- [13] G. Daviller, G. Oztarlik, and T. Poinsot. "A generalized non-reflecting inlet boundary condition for steady and forced compressible flows with injection of vortical and acoustic waves". Computers & Fluids 190 (2019).
- [14] S. Dellacherie et al. "Construction of modified Godunov-type schemes accurate at any Mach number for the compressible Euler system". Mathematical Models and Methods in Applied Sciences 26.13 (2016).
- [15] S. Dellacherie. "Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number". Journal of Computational Physics 229.4 (2010).

- [16] S. Dellacherie. "Checkerboard Modes and Wave Equation". Proceedings of ALGORITMY 2009. 2009.
- [17] S. Dellacherie, P. Omnes, and F. Rieper. "The influence of cell geometry on the Godunov scheme applied to the linear wave equation". Journal of Computational Physics 229.14 (2010).
- [18] J. R. Edwards and M.-S. Liou. "Low-Diffusion Flux-Splitting Methods for Flows at All Speeds". AIAA Journal 36.9 (1998).
- [19] R. Eymard, R. Herbin, and J. C. Latch´e. "On a stabilized colocated Finite Volume scheme for the Stokes problem". ESAIM: Mathematical Modelling and Numerical Analysis 40.3 (2006).
- [20] P. Fillion et al. "FLICA-OVAP: A new platform for core thermal–hydraulic studies". Nuclear Engineering and Design. 13th International Topical Meeting on Nuclear Reactor Thermal Hydraulics (NURETH-13) 241.11 (2011).
- [21] A. Godfrey, R. Walters, and B. Van Leer. "Preconditioning for the Navier-Stokes equations with finite-rate chemistry". 31st Aerospace Sciences Meeting. American Institute of Aeronautics and Astronautics, 1993.
- [22] P. M. Gresho and S. T. Chan. "On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite element method that also introduces a nearly consistent mass matrix. Part 2: Implementation". International Journal for Numerical Methods in Fluids 11.5 (1990).
- [23] H. Guillard and B. Nkonga. "Chapter 8 - On the Behaviour of Upwind Schemes in the Low Mach Number Limit: A Review". Handbook of Numerical Analysis. Ed. by R. Abgrall and C.-W. Shu. Vol. 18. Handbook of Numerical Methods for Hyperbolic Problems. Elsevier, 2017.
- [24] H. Guillard. "On the behavior of upwind schemes in the low Mach number limit. IV: P0 approximation on triangular and tetrahedral cells". Computers & Fluids 38.10 (2009).
- [25] H. Guillard and A. Murrone. "On the behavior of upwind schemes in the low Mach number limit: II. Godunov type schemes". Computers & Fluids 33.4 (2004).
- [26] H. Guillard and C. Viozat. "On the behaviour of upwind schemes in the low Mach number limit". Computers & Fluids 28.1 (1999).
- [27] T. J. R. Hughes, L. P. Franca, and M. Balestra. "A new finite element formulation for computational fluid dynamics: V. Circumventing the babuˇska-brezzi condition: a stable Petrov-Galerkin formulation of the stokes problem accommodating equal-order interpolations". Computer Methods in Applied Mechanics and Engineering 59.1 (1986).
- [28] O. Inoue and N. Hatakeyama. "Sound generation by a two-dimensional circular cylinder in a uniform flow". Journal of Fluid Mechanics 471 (2002).
- [29] K. Kitamura and A. Hashimoto. "Reduced dissipation AUSM-family fluxes: HR-SLAU2 and HR-AUSM+-up for high resolution unsteady flow simulations". Computers & Fluids 126 (2016).
- [30] S. Klainerman and A. Majda. "Compressible and incompressible fluids". Communications on Pure and Applied Mathematics 35.5 (1982).
- [31] R. Klein. "Semi-implicit extension of a godunov-type scheme based on low mach number asymptotics I: One-dimensional flow". Journal of Computational Physics 121.2 (1995).
- [32] X.-s. Li and C.-w. Gu. "An All-Speed Roe-type scheme and its asymptotic analysis of low Mach number behaviour". Journal of Computational Physics 227.10 (2008).
- [33] X.-s. Li and C.-w. Gu. "Mechanism of Roe-type schemes for all-speed flows and its application". Computers & Fluids 86 (2013).
- [34] X.-s. Li and C.-w. Gu. "The momentum interpolation method based on the time-marching algorithm for All-Speed flows". Journal of Computational Physics 229.20 (2010).

- [35] X.-s. Li, C.-w. Gu, and J.-z. Xu. "Development of Roe-type scheme for all-speed flows based on preconditioning method". Computers & Fluids 38.4 (2009).
- [36] B.-X. Lin, C. Yan, and S.-S. Chen. "Density enhancement mechanism of upwind schemes for low Mach number flows". Acta Mechanica Sinica 34.3 (2018).
- [37] M.-S. Liou. "A sequel to AUSM, Part II: AUSM+-up for all speeds". Journal of Computational Physics 214.1 (2006).
- [38] M.-S. Liou. "A Sequel to AUSM: AUSM+". Journal of Computational Physics 129.2 (1996).
- [39] M.-S. Liou and J. Edwards. "Numerical speed of sound and its application to schemes for all speeds". 14th Computational Fluid Dynamics Conference. American Institute of Aeronautics and Astronautics, 1999.
- [40] M.-S. Liou and C. J. Steffen. "A New Flux Splitting Scheme". Journal of Computational Physics 107.1 (1993).
- [41] R. Liska and B. Wendroff. "Comparison of Several Difference Schemes on 1D and 2D Test Problems for the Euler Equations". SIAM Journal on Scientific Computing 25.3 (2003).
- [42] Y. Liu and M. Vinokur. "Upwind algorithms for general thermo-chemical nonequilibrium flows". 27th Aerospace Sciences Meeting. Aerospace Sciences Meetings. American Institute of Aeronautics and Astronautics, 1989.
- [43] A. MAJDA and J. SETHIAN. "The Derivation and Numerical Solution of the Equations for Zero Mach Number Combustion". Combustion Science and Technology 42.3-4 (1985).
- [44] I. Mary and P. Sagaut. "Large Eddy Simulation of Flow Around an Airfoil Near Stall". AIAA Journal 40.6 (2002).
- [45] S. Matsuyama. "Performance of all-speed AUSM-family schemes for DNS of low Mach number turbulent channel flow". Computers & Fluids 91 (2014).
- [46] C. Merkle and S. Venkateswaran. "The use of asymptotic expansions to enhance computational methods". 2nd AIAA, Theoretical Fluid Mechanics Meeting. American Institute of Aeronautics and Astronautics, 1998.
- [47] F. Miczek, F. K. R¨opke, and P. V. F. Edelmann. "New numerical solver for flows at various Mach numbers". Astronomy & Astrophysics 576 (2015).
- [48] B. M¨uller. "Low Mach Number Asymptotics of the Navier-Stokes Equations and Numerical Implications". 30th Computational Fluid Dynamics Lecture Series. von Karman Institute for Fluid Dynamics, 1999.
- [49] K. Oßwald et al. "L2Roe: a low dissipation version of Roe's approximate Riemann solver for low Mach numbers". International Journal for Numerical Methods in Fluids 81.2 (2016).
- [50] M. Potsdam, V. Sankaran, and S. Pandya. "Unsteady Low Mach Preconditioning with Application to Rotorcraft Flows". 18th AIAA Computational Fluid Dynamics Conference. American Institute of Aeronautics and Astronautics, 2007.
- [51] R. G. Rehm and H. R. Baum. "The equations of motion for thermally driven, buoyant flows". Journal of Research of the National Bureau of Standards 83.3 (1978).
- [52] C. M. Rhie and W. L. Chow. "Numerical study of the turbulent flow past an airfoil with trailing edge separation". AIAA Journal 21.11 (1983).
- [53] F. Rieper. "A low-Mach number fix for Roe's approximate Riemann solver". Journal of Computational Physics 230.13 (2011).
- [54] F. Rieper. "Influence of cell geometry on the behaviour of the first-order Roe scheme in the low Mach number regime". Proceedings of the Fifth Conference on Finite Volumes for Complex Applications. Wiley, 2008.
- [55] F. Rieper and G. Bader. "The influence of cell geometry on the accuracy of upwind schemes in the low mach number regime". Journal of Computational Physics 228.8 (2009).

- [56] P. L. Roe. "Approximate Riemann solvers, parameter vectors, and difference schemes". Journal of Computational Physics 43.2 (1981).
- [57] S. Roller et al. "Calculation of low Mach number acoustics: a comparison of MPV, EIF and linearized Euler equations". ESAIM: Mathematical Modelling and Numerical Analysis 39.3 (2005).
- [58] J. Sachdev, A. Hosangadi, and V. Sankaran. "Improved Flux Formulations for Unsteady Low Mach Number Flows". 42nd AIAA Fluid Dynamics Conference and Exhibit. Fluid Dynamics and Co-located Conferences. American Institute of Aeronautics and Astronautics, 2012.
- [59] S. Schochet. "Fast Singular Limits of Hyperbolic PDEs". Journal of Differential Equations 114.2 (1994).
- [60] E. Shima and K. Kitamura. "New approaches for computation of low Mach number flows". Computers & Fluids. International Workshop on Future of CFD and Aerospace Sciences 85 (2013).
- [61] E. Shima and K. Kitamura. "Parameter-Free Simple Low-Dissipation AUSM-Family Scheme for All Speeds". AIAA Journal 49.8 (2011).
- [62] C. F. Silva et al. "Assessment of combustion noise in a premixed swirled combustor via Large-Eddy Simulation". Computers & Fluids. LES of turbulence aeroacoustics and combustion 78 (2013).
- [63] S. M. Sohel Murshed and C. A. Nieto de Castro. "A critical review of traditional and emerging techniques and fluids for electronics cooling". Renewable and Sustainable Energy Reviews 78 (2017).
- [64] P. K. Subbareddy and G. V. Candler. "A fully discrete, kinetic energy consistent finitevolume scheme for compressible flows". Journal of Computational Physics 228.5 (2009).
- [65] B. Thornber et al. "An improved reconstruction method for compressible flows with low Mach number features". Journal of Computational Physics 227.10 (2008).
- [66] B. Thornber et al. "On entropy generation and dissipation of kinetic energy in highresolution shock-capturing schemes". Journal of Computational Physics 227.10 (2008).
- [67] B. J. R. Thornber and D. Drikakis. "Numerical dissipation of upwind schemes in low Mach flow". International Journal for Numerical Methods in Fluids 56.8 (2008).
- [68] E. F. Toro and M. E. V´azquez-Cend´on. "Flux splitting schemes for the Euler equations". Computers & Fluids 70 (2012).
- [69] E. Turkel. "Preconditioning Techniques in Computational Fluid Dynamics". Annual Review of Fluid Mechanics 31.1 (1999).
- [70] E. Turkel. "Review of preconditioning methods for fluid dynamics". Applied Numerical Mathematics. SPECIAL ISSUE 12.1 (1993).
- [71] E. Turkel, A. Fiterman, and B. Van Leer. "Preconditioning and the limit of the compressible to the incompressible flow equations for finite difference schemes". Frontiers of Computational Fluid Dynamics 1994. Editors D. A. Caughey & M. M. Hafez. John Wiley & Sons Ltd, 1994.
- [72] S. Venkateswaran and C. Merkle. "Dual time-stepping and preconditioning for unsteady computations". 33rd Aerospace Sciences Meeting and Exhibit. Aerospace Sciences Meetings. American Institute of Aeronautics and Astronautics, 1995.
- [73] S. Venkateswaran and C. Merkle. "Efficiency and accuracy issues in contemporary CFD algorithms". Fluids 2000 Conference and Exhibit. American Institute of Aeronautics and Astronautics, 2000.
- [74] S. Venkateswaran and C. L. Merkle. "Evaluation of artificial dissipation models and their relationship to the accuracy of Euler and Navier-Stokes computations". Sixteenth International Conference on Numerical Methods in Fluid Dynamics. Ed. by C.-H. Bruneau. Lecture Notes in Physics. Berlin, Heidelberg: Springer, 1998.

- [75] S. Venkateswaran and C. Merkle. "Artficial Dissipation Control for Viscous and Unsteady Computations". 16th AIAA Computational Fluid Dynamics Conference. American Institute of Aeronautics and Astronautics, 2003.
- [76] G. Volpe. "Performance of compressible flow codes at low Mach numbers". AIAA Journal 31.1 (1993).
- [77] C. Wall, C. D. Pierce, and P. Moin. "A Semi-implicit Method for Resolution of Acoustic Waves in Low Mach Number Flows". Journal of Computational Physics 181.2 (2002).
- [78] J. M. Weiss and W. A. Smith. "Preconditioning applied to variable and constant density flows". AIAA Journal 33.11 (1995).
- [79] L. Yu, S. Diasinos, and B. Thornber. "A fast transient solver for low-Mach number aerodynamics and aeroacoustics". Computers & Fluids 214 (2021).
- [80] G.-C. Zha and E. Bilgen. "Numerical solutions of Euler equations by using a new flux vector splitting scheme". International Journal for Numerical Methods in Fluids 17.2 (1993).

# Appendices

# A Transonic schemes at low Mach number

In this appendix we show, in a very approximate manner, why many numerical schemes designed for transonic flow approach the acoustic diffusion scaling at low Mach number. Diffusion schemes for the Euler equations can be classified by how many waves they resolve. For example, scalar Lax-Friedrichs/Rusonov is a 1-wave scheme, HLL is 2-wave scheme, and Roe, HLLC and AUSM are all 3-wave schemes. In the characteristic variables, the diffusion Jacobian for the the acoustic waves of a 12- or 3-waves scheme will be close to:

$$\begin{pmatrix} |u+a| & 0\\ 0 & |u\pm a| \end{pmatrix} \tag{72}$$

Where the bottom right entry has an addition for 1-wave schemes and a subtraction for 2and 3wave schemes. At low Mach number this Jacobian can be approximated by:

$$|u| \begin{pmatrix} M^{-1} + 1 & 0 \\ 0 & M^{-1} \pm 1 \end{pmatrix} \approx |u| \begin{pmatrix} M^{-1} & 0 \\ 0 & M^{-1} \end{pmatrix}$$
 (73)

The diffusion on the acoustic waves du ± dp/ρa is then:

$$(|u|M^{-1})\underline{\mathcal{I}} \tag{74}$$

(75c)

which matches the acoustic scaling. Because the matrix [\(74\)](#page-36-8) is a scalar multiple of the identity matrix, any variable basis for the acoustic system (for example dp and du) will also have this diffusion Jacobian at low Mach number.

# B Non-dimensionalised artificial diffusion for finite-volume schemes

The non-dimensional diffusion with convective scaling and <sup>α</sup> = 1 is:

$$f_{\rho}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \rho + \rho \left( \frac{M^{-2} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) \right)$$

$$f_{\underline{\rho u}}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \underline{\rho u} + \underline{\rho u} \left( \frac{M^{-2} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) + \underline{n}_{il} \left( M^{-2} \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p + \rho |v| \nu_{22} \Delta_{il} U \right) \right)$$

$$(75a)$$

$$f_{\underline{\rho E}}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \rho E + \rho H \left( \frac{M^{-2} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) + U M^{2} \left( M^{-2} \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p + \rho |v| \nu_{22} \Delta_{il} U \right) \right)$$

The non-dimensional diffusion with acoustic scaling and <sup>α</sup> = 1 is:

$$f_{\rho}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \rho + \rho \left( \frac{M^{-1} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) \right)$$

$$f_{\underline{\rho u}}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \underline{\rho u} + \underline{\rho u} \left( \frac{M^{-1} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) + \underline{n}_{il} \left( M^{-2} \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p + M^{-1} \rho |v| \nu_{22} \Delta_{il} U \right) \right)$$

$$(76b)$$

$$f_{\rho E}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \rho E + \rho H \left( \frac{M^{-1} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) + U M^{2} \left( M^{-2} \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p + M^{-1} \rho |v| \nu_{22} \Delta_{il} U \right) \right)$$

$$(76c)$$

The non-dimensional diffusion with blended scaling and <sup>α</sup> = 1 is:

$$f_{\rho}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \rho + \rho \left( \frac{M^{-1} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) \right)$$

$$f_{\underline{\rho u}}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \underline{\rho u} + \underline{\rho u} \left( \frac{M^{-1} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) + \underline{n}_{il} \left( M^{-2} \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p + \rho |v| \nu_{22} \Delta_{il} U \right) \right)$$

$$(77b)$$

$$f_{\rho E}^{d} = \frac{1}{2} \sum_{\mathcal{V}(i)}^{l} S_{il} \left( |U| \Delta_{il} \rho E + \rho H \left( \frac{M^{-1} \nu_{11}}{\rho |v|} \Delta_{il} p + \frac{U_{il}}{|U_{il}|} \nu_{12} \Delta_{il} U \right) + U M^{2} \left( M^{-2} \frac{U_{il}}{|U_{il}|} \nu_{21} \Delta_{il} p + \rho |v| \nu_{22} \Delta_{il} U \right) \right)$$

$$(77c)$$