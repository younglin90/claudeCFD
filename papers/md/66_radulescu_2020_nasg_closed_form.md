# Compressible flow in a Noble-Abel Stiffened-Gas fluid

M. I. Radulescu Department of Mechanical Engineering University of Ottawa, Ottawa (ON) K1N 6N5 Canada

April 21, 2020

#### Abstract

While compressible flow theory has relied on the perfect gas model as its workhorse for the past century, compressible dynamics in dense gases, solids and liquids have relied on many complex equations of state, yielding limited insight on the hydrodynamic aspect of the problems solved. Recently, Le Mtayer and Saurel studied a simple yet promising equation of state owing to its ability to model both the thermal and compressibility aspects of the medium. It is a hybrid of the Noble-Abel equation of state and the stiffened gas model, labeled the Noble-Able Stiffened Gas (NASG) equation of state. In the present work, we derive the closed form analytical framework for modelling compressible flow in a medium approximated by the NASG equations of state. We derive the expressions for the isentrope, sound speed, the isentropic exponent, Riemann variables in the characteristic description, and jump conditions for shocks, deflagrations and detonations. We also illustrate the usefulness by addressing the Riemann problem. The closed form solutions generalize in a transparent way the well-established models for a perfect gas, highlighting the role of the medium's compressibility.

## 1 Introduction

Recenty, Le Mtayer and Saurel (henceforth LS) have analyzed in detail an equation of state consisting of a blend of the Noble-Abel equation of state and the stiffened gas equation of state, labeled the Noble-Abel Stiffened Gas (NASG) [\[1\]](#page-8-0). This equation of state has proven quite successful in numerical work treating compressible flows of multi-phase and multi-component flows, both inert and reactive[\[2,](#page-8-1) [3\]](#page-8-2). It has also been shown to empirically capture very well the compressibility of metals, liquids and dense gases [\[4,](#page-8-3) [5\]](#page-10-0).

The model generalizes the model usually referred to in the modern literature as the modified Tait , Tammann [\[6\]](#page-10-1) (although Tammann postuated the present form of the NASG model [\[7\]](#page-10-2)) or stiffened gas [\[8\]](#page-10-3), which has an ill-defined thermodynamic temperature when γ in [\(1\)](#page-1-0) is artificially increased to account for the correct compressibility of the mediumsee Radulescu [\[9\]](#page-10-4) for discussion. It includes the co-volume factor present in the Noble-Abel equation of state, which was present in the early work of Tammann. This permits to recover the correct compressibility while also capturing the thermal behaviour.

In the present study, we would like to show that the NASG equation of state also permits to analytically tackle problems of compressible inert or reactive flows quite simply. We extend the results of LS and derive the relevant quantities to treat compressible flows, i.e., the isentrope, sound speed, Riemann variables in the characteristic form of the Euler equations and their weak solutions for shocks, deflagrations and detonation waves. We generalize the well accepted results of a perfect gas; the NASG model offers the same simplicity and power to tackle problems in dense gases, solids and liquids.

# 2 The isentrope, sound speed and the isentropic exponent

The NASG equation of state for a single component relates the internal energy e of the medium to the medium's pressure p and specific volume v:

$$e(p,v) = \frac{p + \gamma p_{\infty}}{\gamma - 1} (v - b) + q \tag{1}$$

where p∞, b and q are fitting parameters and γ is the ratio of specific heats. The physical meaning of b is the usual co-volume, or minimum effective volume that the fluid can occupy given its finite physical size. The parameter p<sup>∞</sup> accounts for the attraction forces between the molecules, such that gas like behaviour is only expected for pressures significantly larger than p∞.

LS and Radulescu [\[9\]](#page-10-4) derived the corresponding temperature equation of state starting from [\(1\)](#page-1-0):

$$T = \frac{(p+p_{\infty})(v-b)}{C_v(\gamma-1)} \tag{2}$$

The right hand side is the functional form for an isotherm postulated by Tammann [\[7\]](#page-10-2) and subsequently used to model dense gases, liquids and solids [\[4,](#page-8-3) [5\]](#page-10-0).

To obtain the insentrope, sound speed and the isentropic exponent, we begin by the first Gibbs, or T dS equation, given by

$$de = Tds - pdv (3)$$

which we re-write in terms of density ρ = 1/v as

$$de = Tds + \frac{p}{\rho^2}d\rho \tag{4}$$

or as a perfect differential

$$de(p,\rho) = \left(\frac{\partial e}{\partial p}\right)_{\rho} dp + \left(\frac{\partial e}{\partial \rho}\right)_{p} d\rho \tag{5}$$

Equating both expressions to eliminate de we obtain

$$dp = \frac{\frac{p}{\rho^2} - \left(\frac{\partial e}{\partial \rho}\right)_p}{\left(\frac{\partial e}{\partial p}\right)_\rho} d\rho + \frac{T}{\left(\frac{\partial e}{\partial p}\right)_\rho} ds \tag{6}$$

Now write  $p(\rho, s)$  as a perfect differential of the form

$$dp = \left(\frac{\partial p}{\partial \rho}\right)_s d\rho + \left(\frac{\partial p}{\partial s}\right)_\rho ds \tag{7}$$

and comparing (6) with (7), we get immediately the expression for the sound speed

$$c^{2} \equiv \left(\frac{\partial p}{\partial \rho}\right)_{s} = \frac{\frac{p}{\rho^{2}} - \left(\frac{\partial e}{\partial \rho}\right)_{p}}{\left(\frac{\partial e}{\partial p}\right)_{\rho}} \tag{8}$$

Evaluating the partial derivatives  $\left(\frac{\partial e}{\partial \rho}\right)_p$  and  $\left(\frac{\partial e}{\partial p}\right)_\rho$  from (1), we obtain immediately the expression for the sound speed in a NASG fluid:

$$c^2 = \gamma \frac{p + p_{\infty}}{\rho(1 - \rho b)} \tag{9}$$

and similarly using (2), we can re-write (6) as

$$dp = c^2 d\rho + (p + p_{\infty}) \frac{ds}{C_v}$$
(10)

On the isentrope, ds = 0, and (10) integrates to:

$$(p+p_{\infty})(v-b)^{\gamma} = const. \tag{11}$$

We thus recover the result for a perfect gas with  $p + p_{\infty}$  replacing p and v - b replacing v. Note that for b=0, the isentrope is what is referred to in the literature as the Tait equation of state, popularized by Kirkwood and Bethe [10].

The isentropic exponent becomes

$$\gamma_s \equiv \left(\frac{\partial \ln p}{\partial \ln \rho}\right)_s = \frac{\rho}{p}c^2 = \gamma \frac{1 + \frac{p_\infty}{p}}{1 - \frac{b}{p}} \tag{12}$$

It highlights how the compressibility of a medium departs from that of a perfect gas when the pressure is comparable or less than the stiffening pressure and/or when the volume approaches the minimum co-volume.

# 3 Characteristic description of quasi-1D flows

To appreciate the usefulness of the NASG equation of state to describe the general motion of a compressible fluid, we first derive the general formulation for an arbitrary equation of state. The mass and momentum conservation for a quasi-1D flow in a stream tube of varying cross-section, in the presence of a body force f are

$$\frac{\mathrm{D}\rho}{\mathrm{D}t} + \rho \frac{\partial u}{\partial x} = -\rho u \frac{1}{A} \frac{\partial A}{\partial x} \tag{13}$$

$$\rho \frac{\mathrm{D}u}{\mathrm{D}t} = -\frac{\partial p}{\partial x} + f \tag{14}$$

These are of course supplemented by a statement for conservation of energy along a particle path, for example [\(6\)](#page-2-0) written in general form:

$$\frac{\mathrm{D}p}{\mathrm{D}t} = c^2 \frac{\mathrm{D}\rho}{\mathrm{D}t} + T \left(\frac{\partial p}{\partial e}\right)_{\rho} \frac{\mathrm{D}s}{\mathrm{D}t} \tag{15}$$

Eliminating the density derivative and taking linear combinations of the remaining two equations, we can obtain the two characteristic equations:

$$\frac{1}{\rho c} \frac{D_{\pm}}{Dt} p \pm \frac{D_{\pm}}{Dt} u = -uc \frac{\partial \ln A}{\partial x} + \frac{T}{\rho c} \left( \frac{\partial p}{\partial e} \right)_{\rho} \frac{Ds}{Dt} \pm \frac{f}{\rho}$$
 (16)

where

$$\frac{D_{\pm}}{Dt} = \frac{\partial}{\partial t} + (u \pm c) \frac{\partial}{\partial x}$$
 (17)

are the convective derivatives along the C<sup>±</sup> characteristic directions dx/dt = u ± c.

So far, relation [\(16\)](#page-3-0) applies to any medium. If the flow is isentropic, all variables appearing in the first term can be expressed in terms of a single thermodynamic variable. After some manipulations using the isentrope and the form of the sound speed above, the first term can be re-written as:

$$\frac{1}{\rho c}dp = d\left(\frac{2}{\gamma - 1}\sqrt{\gamma(p + p_{\infty})(v - b)}\right)$$
(18)

such that the characteristic equation for isentropic flow becomes

$$\frac{D_{\pm}}{Dt}J_{\pm} = -uc\frac{\partial \ln A}{\partial x} + \pm \frac{f}{\rho}$$
 (19)

where the Riemann variables J<sup>±</sup> for a NASG fluid are quite simply:

$$J_{\pm} = \left(\frac{2}{\gamma - 1}\sqrt{\gamma(p + p_{\infty})(v - b)}\right) \pm u \tag{20}$$

Clearly, this reduces to the classic form 2c/(γ − 1) ± u in a perfect gas, for which c <sup>2</sup> = γpv. For isentropic flows without lateral divergence and body forces, simple wave solutions can be obtained in the usual way by exploiting the constancy of one of the Riemann variables everywhere in the flow [\[11\]](#page-10-6). Simple wave extensions for weak shocks can be exploited in the usual manner [\[11\]](#page-10-6).

Non-isentropic flow, for example reactive flow, flows with strong shocks or with generic heat addition or losses, can be treated as well by involving thermodynamic derivatives of the thermodynamic part of the Riemann variable θ ≡ γ−1 p γ(p + p∞)(v − b).

$$dp(\theta, S) = \left(\frac{\partial p}{\partial \theta}\right)_s d\theta + \left(\frac{\partial p}{\partial s}\right)_\theta ds \tag{21}$$

This permits to re-write [\(16\)](#page-3-0) in terms of the Riemann variables and entropy changes along particle paths and C<sup>±</sup> characteristics.

# 4 Inert shock waves

The jump conditions across shock waves also yield very simple expressions for a NASG fluid. The weak form of the conservation laws of mass, momentum and energy across a shock wave moving at speed D normal to its surface, into gas moving at speed u<sup>1</sup> normal to the shock take the usual form [\[11\]](#page-10-6).

$$\frac{D - u_1}{v_1} = \frac{D - u_2}{v_2} \tag{22}$$

$$p_1 + \frac{(D - u_1)^2}{v_1} = p_2 + \frac{(D - u_2)^2}{v_2}$$
 (23)

$$e_1 + p_1 v_1 + \frac{1}{2} (D - u_1)^2 = e_2 + p_2 v_2 + \frac{1}{2} (D - u_2)^2$$
 (24)

The post shock state is indicated with subscript 2. It is simpler to work with the mass conservation [\(22\)](#page-4-0) and linear combinations with the two other equations, leading to the so-called Hugoniot curve and Rayleigh line. The former is obtained by eliminating the speeds from the three equations, yielding

$$e_2 - e_1 = \frac{1}{2}(p_1 + p_2)(v_1 - v_2)$$
 (25)

The Rayleigh line is the combination of the mass and momentum used to eliminate the speed u2, yielding

$$p_2 - p_1 = \frac{(D - u_1)^2}{v_1} \left( 1 - \frac{v_2}{v_1} \right) \tag{26}$$

The mass conservation [\(22\)](#page-4-0), the Rayleigh line [\(23\)](#page-4-1) and the Hugoniot curve [\(25\)](#page-4-2) now provide the system of equations that determine the jump conditions. These are general for any medium.

Upon substitution of the equation of state expressions for a NASG fluid (1) for  $e_1$  and  $e_2$ , the Hugoniot relation (25) can be written after some manipulations:

$$\left(\frac{\overline{p}_2}{\overline{p}_1} + \frac{\gamma - 1}{\gamma + 1}\right) \left(\frac{\overline{v}_2}{\overline{v}_1} - \frac{\gamma - 1}{\gamma + 1}\right) = 1 - \left(\frac{\gamma - 1}{\gamma + 1}\right)^2 \tag{27}$$

where  $\overline{p} = p + p_{\infty}$  and  $\overline{v} = v - b$ . This is the same expression as for the Hugoniot curve for a perfect gas,  $\overline{p}$  replacing p and  $\overline{v}$  replacing v.

Noting that the square of the shock Mach number can be written as

$$M_s^2 \equiv \frac{(D - u_1)^2}{c_1^2} = \frac{(D - u_1)^2 \overline{v}_1}{\gamma \overline{p}_1 v_1^2}$$
 (28)

The Rayleigh line takes again the same form as that for a perfect gas, with  $\overline{p}$  replacing p and  $\overline{v}$  replacing v:

$$\frac{\overline{p}_2}{\overline{p}_1} = 1 + \gamma M_s^2 \left( 1 - \frac{\overline{v}_2}{\overline{v}_1} \right) \tag{29}$$

Solving for the pressure and specific volume jumps by satisfying the Rayleigh and Hugoniot conditions yield again the same expressions as for a perfect gas with  $\bar{p}$  replacing p and  $\bar{v}$  replacing v:

$$\frac{\overline{p}_2}{\overline{p}_1} = 1 + \frac{2\gamma \left(M_s^2 - 1\right)}{\gamma + 1} \tag{30}$$

$$\frac{\overline{v}_2}{\overline{v}_1} = \frac{(\gamma - 1)M_s^2 + 2}{(\gamma + 1)M_s^2} \tag{31}$$

The change in particle speed across the shock wave follows from the specific volume jump and the conservation of mass (22), which, after some manipulations, yields

$$\frac{u_2 - u_1}{c_1} = \left(1 - \frac{b}{v_1}\right) \frac{2(M_s^2 - 1)}{(\gamma + 1)M_s} \tag{32}$$

This expression is also the same as for a perfect gas with the addition of the term  $\left(1 - \frac{b}{v_1}\right)$  accounting for the reduction in the medium's compressibility as  $v_1 \to b$ , as already noted when discussing the isentropic exponent.

It is now clear that other jump conditions of interest, such as temperature, sound speed, flow Mach number, Riemann variables, entropy, etc... are easily obtained.

# 5 The Riemann problem

To illustrate the usefulness of the analytical treatment in a NASG medium, we can consider for example the solution to the initial value problem of two separate media initially at different but constant mechanical and thermodynamic states.

![](_page_6_Figure_0.jpeg)

Figure 1: The solution to the Riemann problem for  $p_2 > p_1$  and  $u_1 = u_2 = 0$  for the mechanical equilibration at the interface between two different materials.

This is known as the Riemann problem. Its importance is primarily in numerical work at grid cell interfaces (see, for example, Gottlieb and Groth's summary [12]). Consider a medium A at uniform state  $(p_1, v_1, u_1)$ , separated from a medium B at state  $(p_2, v_2, u_2)$ . The properties may be different in the two media, labeled respectively with subscripts A and B. The general case is allowing  $u_1 \neq u_2$  and  $p_1 \neq p_2$ . Five different wave patterns are possible, involving either shock or expansion waves driven into each respective medium, one of which is shown schematically in the space-time diagram of Fig. 1. While the general case is clearly relevant for numerical work in multi-material situations, and will be communicated in a sequel, we will focus here on the particular case  $u_1 = u_2 = 0$ of Fig. 1 for illustrative purposes. This is known as the shock tube problem, since its solution provides an idealized solution of a high pressure gas initially at rest discharging into a low pressure one, driving a shock wave. When  $p_2 > p_1$ , a shock wave will be driven into medium A, changing its state from 1 to 4 in Fig. 1. The compression of medium A is associated with the expansion of medium B, whose state changes from 2 to 3. The solution to this problem is determining the strength and structure of the expansion wave and the strength of the shock wave that give rise to mechanical equilibrium at the interface (or contact surface) between the two media, i.e.,  $p_3 = p_4$  and  $u_3 = u_4$ .

A  $C^+$  characteristic connects state 2 to state 3. Since the expansion of the medium along a particle path is also isentropic, state 3 is linked to state 2 by the invariance of the Riemann variable  $J_+$  given by (20) and the isentropic expansion condition (11). These two expressions can be combined to eliminate the specific volume at state 3, yielding:

$$\frac{2}{\gamma_B - 1} \sqrt{\gamma(p_2 + p_{\infty,B})(v_2 - b_B)} \left( 1 - \left( \frac{p_3 + p_{\infty,B}}{p_2 + p_{\infty,B}} \right)^{\frac{\gamma_B - 1}{2\gamma_B}} \right) - u_3 = 0 \quad (33)$$

The unknown pressure  $p_3$  and particle speed  $u_3$  satisfy the mechanical equilibrium condition  $p_3 = p_4$  and  $u_3 = u_4$ . Making these substitutions in (33) results

in a condition linking  $p_4$  and  $u_4$ . Since these satisfy the shock jump equations derived in the previous section in terms of the shock Mach number  $M_s$ , namely

$$\frac{p_4 + p_{\infty,A}}{p_1 + p_{\infty,A}} = 1 + \frac{2\gamma_A \left(M_s^2 - 1\right)}{\gamma_A + 1} \tag{34}$$

and

$$\frac{u_4}{c_1} = \left(1 - \frac{b_A}{v_1}\right) \frac{2(M_s^2 - 1)}{(\gamma_A + 1)M_s} \tag{35}$$

we have obtained a single algebraic condition for  $M_s$ . Given the shock and expansion wave strengths, the structure of the expansion wave is also found with minor effort in closed form using the simple wave argument [11]. The shock tube problem treated for two different materials applies evidently to the case when either or both of the media are ideal gases, by setting the coefficients of b,  $\gamma$  and  $p_{\infty}$  accordingly.

#### 6 Deflagrations and detonations

Gasdynamic reactive discontinuities in a NASG fluid can also be quite simply modelled by extensions to the well-established perfect gas model. Here we seek the possible discontinuities with energy addition or withdraw by simply changing the constants  $q_1$  and  $q_2$  accounting for changes in reference internal energy of the medium ahead and behind the wave. If we let the effective heat release across the wave be  $Q = q_1 - q_2$ , proceeding as for the inert shocks treated above, the Hugoniot curve given by (25) becomes:

$$\left(\frac{\overline{p}_2}{\overline{p}_1} + \frac{\gamma - 1}{\gamma + 1}\right) \left(\frac{\overline{v}_2}{\overline{v}_1} - \frac{\gamma - 1}{\gamma + 1}\right) = 1 - \left(\frac{\gamma - 1}{\gamma + 1}\right)^2 + 2\frac{\gamma - 1}{\gamma + 1}\frac{Q}{\overline{p}_1\overline{v}_1} \tag{36}$$

This is again the same Hugoniot expression as for a perfect gas, with  $\overline{p}$  replacing p and  $\overline{v}$  replacing v. The Hugoniot curve is shown in Fig. 2 in the compression region for detonations  $(v_2 < v_1)$  and in the expansion region  $(v_2 > v_1)$  for deflagrations.

The Rayleigh line expression is the same as for an inert shock. Its combination with the Hugoniot curve readily permits to obtain the jump equations for the pressure and specific volume:

$$\frac{\overline{p}_2}{\overline{p}_1} = \frac{1 + \gamma M_s^2}{\gamma + 1} \mp \gamma M_s^2 \sqrt{\zeta} \tag{37}$$

$$\frac{\overline{v}_2}{\overline{v}_1} = \frac{1 + \gamma M_s^2}{M_s^2 (\gamma + 1)} \pm \sqrt{\zeta} \tag{38}$$

where

$$\zeta = \frac{\left(M_s - \frac{1}{M_s}\right)^2 - \frac{1(\gamma^2 - 1)}{\gamma} \frac{Q}{\overline{p}_1 \overline{v}_1}}{(\gamma - 1)^2 M_s^2} \tag{39}$$

The solutions are again identical with those for a perfect gas. These solutions are shown in Fig. [2](#page-9-0) for detonations and deflagrations. These are shown separately, for clarity. These are discussed at length in the literature, see for example Lee's monograph on detonations [\[13\]](#page-10-8) or by Landau and Lifshitz [\[14\]](#page-10-9). Briefly, it shows that the pressure and volume changes from the initial state 1 admit two solutions for the same wave speed, denoted as strong or weak for respectively the upper and lower sign in the signed expressions involving the term containing ζ. Detonations are supersonic and compressive, whether deflagrations are subsonic and expansive. Chapman-Jouguet detonations or deflagrations are obtained when the weak and strong solutions merge and the terms involving ζ vanish. This requires ζ to vanish, yielding the two CJ reaction waves possible, detonations and deflagrations:

$$M_{s,CJ}^2 = 1 + \frac{\gamma^2 - 1}{\gamma} \frac{Q}{\overline{p}_1 \overline{v}_1} \pm \sqrt{\left(\frac{\gamma^2 - 1}{\gamma} \frac{Q}{\overline{p}_1 \overline{v}_1} + 1\right)^2 - 1}$$
 (40)

These are again the same expressions as for a perfect gas, with p replacing p and v replacing v and the appropriate expression for the sound speed entering the definition of the Mach number. Other jump relations are easily formulated as for the inert shocks. It can be shown that the Mach number of the flow in the frame of reference of the wave, i.e., (D − u2)/c<sup>2</sup> is unity for the Chapman-Jouguet detonations and deflagrations.

## 7 Conclusions

We have derived the main results required for analytical work of inert and reactive gasdynamics in a medium approximated by the NASG equation of state. The expressions obtained are transparent generalizations of the perfect gas relations and can be useful in the modelling of compressible flows in inert and reactive dense gases, liquids and solids.

## References

- [1] O. Le Mtayer, R. Saurel, The Noble-Abel Stiffened-Gas equation of state, Physics of Fluids 28 (4).
- [2] D. Furfaro, R. Saurel, L. David, F. Beauchamp, Towards sodium combustion modelling with liquid water, Journal of Computational Physics 403 (2020) 109060.
- [3] P. Boivin, M. A. Cannac, O. Le Mtayer, A thermodynamic closure for the simulation of multiphase reactive flows, International Journal of Thermal Sciences 137 (2019) 640–649.
- [4] T. W. Richards, Compressibility, internal pressure and atomic magnitudes, Journal of the American Chemical Society 1923, 45, 2, 422-437 45 (2) (1923) 422–437.

![](_page_9_Figure_0.jpeg)

Figure 2: Weak (W), strong (S) and Chapman-Jouguet (CJ) solutions for detonation and deflagrations at the intersection of the Rayleigh lines (purple) with the Hugoniot curve (blue) for  $\frac{Q}{\overline{p}_1\overline{v}_1}=50$  and  $\gamma=1.2$ ; the CJ Rayleigh line is green and the initial state is blue.

- [5] P. Bridgman, The compressibility of five gases to high pressures, Proceedings of the American Academy of Arts and Sciences 59 (8) (1924) 173–211.
- [6] M. J. Ivings, D. M. Causon, E. F. Toro, On Riemann solvers for compressible liquids, International Journal for Numerical Methods in Fluids 28 (3) (1998) 395–418.
- [7] G. Tammann, Uber Zustandsgleichungen im Gebiete kleiner Volumen, An- ¨ nalen der Physik 342 (5) (1912) 975–1013.
- [8] R. Menikoff, Empirical equations of state for solids, Vol. 2, Springer, 2007, book section 4.
- [9] M. I. Radulescu, On the Noble-Abel stiffened-gas equation of state, Physics of Fluids 31 (11) (2019) 111702.
- [10] J. Kirkwood, H. Bethe, The pressure wave produced by an under-water explosion, basic propagation theory, Part 1, Tech. Rep. 588, Office of Scientific Research and Development, Washington, DC (1942).
- [11] G. B. Whitham, Linear and nonlinear waves, Pure and applied mathematics, Wiley, New York,, 1974.
- [12] J. Gottlieb, C. Groth, Assessment of Riemann solvers for unsteady onedimensional inviscid flows of perfect gases, Journal of Computational Physics 78 (2) (1988) 437 – 458.
- [13] J. H. S. Lee, The detonation phenomenon, Cambridge University Press, Cambridge ; New York, 2008.
- [14] L. D. Landau, E. M. Lifshitz, Fluid mechanics, 2nd Edition, Course of theoretical physics, Pergamon Press, Oxford, 1987.