# An LES model with finite-rate phase change and subgrid spray based on a thermodynamically consistent four-equation multiphase model 

Henry Collis[a,] _[∗]_ , Shahab Mirjalili[b,a] , Makrand Khanwale[a] , Ali Mani[a] , Gianluca Iaccarino[a] 

> _aDepartment of Mechanical Engineering, Stanford, CA 94305, USA_ 

> _bFLOW, Department of Engineering Mechanics, KTH Royal Institute of Technology, SE-10044 Stockholm, Sweden_ 

## **Abstract** 

In this work, an LES model with finite-rate phase change and subgrid spray based on a high-resolution numerical scheme for multiphase multi-component simulations which satisfies interface equilibrium and phase immiscibility conditions [1] is proposed. The multiphase model is based on a robust implementation of the four-equation multiphase model which assumes a strict subgrid equilibrium of pressure, temperature, and velocity. Critically, the equilibrium assumptions of the four-equation model provide large computational savings compared to modeling the full non-equilibrium multiphase system. To obtain predictive capabilities with these restrictive equilibrium assumptions, a new phase-confined form of the Eulerian Σ spray model is proposed to predict subgrid interfacial surface area while avoiding unphysical leakage across interfaces. Additionally, an improved finite rate phase change model which is thermodynamically bounded by the equilibration of the Gibbs-free energy is coupled with the Σ equation to model complex phase change regimes. The full modeling framework is validated using the Engine Combustion Network (ECN) Spray A case in non-evaporating and evaporating conditions and shows excellent agreement with experimental measurements. 

_Keywords:_ Subgrid spray, phase change, compressible, four-equation model, multicomponent, multiphase 

## **1. Introduction** 

Modeling engineering-scale multiphase multi-component systems is critical for multiple applications, including combustion engines with fuel injection atomizers, rocket propellant injection, and fire-suppression systems. To incorporate the critical physics required by these applications, multiphysics modeling is needed that captures intraphase mixing, high-density ratio interfaces, and mass-transfer models. In this work, a high-resolution numerical strategy for simulating multiphase multi-component flows based on the fourequation model is extended to include finite-rate phase change models and subgrid spray. Simulations involving subgrid spray physics can physically expect to have non-equilibrium between phase velocity, as the disperse and discrete phases do not need to have the same inertial properties. One of the overarching goals of this work is to showcase the extent in which the four-equation multiphase equilibrium conditions can be pushed while retaining robust and accurate multiphase simulations. Compared to the fully non-equilibrium multiphase models, the four-equation approach assumes mechanical (pressure) and thermal (temperature) equilibrium between phases. The governing system is simplified to a shared momentum and energy equation, greatly reducing the computational cost compared to the non-equilibrium models. In this work, we show that coupling the four-equation numerical framework proposed in [1] with appropriate models for subgrid spray and phase change in the four-equation setting achieves excellent agreement with experiments without requiring the full non-equilibrium multiphase system. 

> _∗_ Corresponding author 

> _Email addresses:_ `hcollis@stanford.edu` (Henry Collis), `msey@kth.se` (Shahab Mirjalili), `khanwale@stanford.edu` 

> (Makrand Khanwale), `alimani@stanford.edu` (Ali Mani), `jops@stanford.edu` (Gianluca Iaccarino) 

## _1.1. Phase Change_ 

Modeling the phase change process, including boiling, evaporation, flashing, and condensation, is critical in multiphase systems. In particular, engineering applications including diesel spray engines and liquid rocket propulsion systems require transfer of mass from the liquid propellant to gas to achieve successful ignition. In general, interphase mass transfer is governed by a non-equilibrium chemical potential between components in separate phases (details in Section 4). Modeling the rate at which the system reaches chemical equilibrium is a critical aspect for accurate phase change modeling. Two general assumptions exist for phase change modeling which can separate the modeling mechanism for the phase change processes listed above (e.g. evaporation vs boiling): instantaneous and finite rate equilibration of the chemical potential. 

## _1.1.1. Instantaneous Equilibrium_ 

Assuming an instantaneous equilibration of the chemical potential between phases is known as the Homogeneous Equilibrium Model (HEM) [2]. The HEM model is appropriate for two regimes: fully resolved DNS where the phase change rate is governed by heat conduction (evaporation), and under-resolved spray which are governed by rapid phase change processes, including flashing and cavitation events [3, 4]. For the cases between the extremes of interface resolved multiphase interfaces and substantially under-resolved simulations of flashing or cavitation, the HEM will over-predict the time-scale of mass transfer. The infinite rate relaxation of the chemical potential to equilibrium of the HEM is not applicable for interfacial structures which have a finite surface area but are not resolvable by a computational grid. These regimes could include using the HEM to model the evaporation of subgrid ligaments and droplets. 

## _1.1.2. Finite-Rate Models_ 

Finite-rate phase change models are designed to alleviate the issues of overpredicting mass transfer which occurs with the HEM. The finite-rate mass transfer time-scale is dependent on both the thermodynamic state, the composition, and the interfacial surface area. The simplest approach for modeling finite rate phase change is assuming a relaxation time-scale which is constant for all space and time throughout a simulation. This approach has been overviewed by multiple past studies [5, 6] which have qualitatively shown how important engineering quantities, including the vaporization mass present in a high-pressure fuel injector [5], are strongly impacted by the assigned phase change rate, showing that the ad-hoc approach of assigning a constant spatial-temporal phase change rate to a system is not practical for producing a predictive and generalizable computational model. The results of these works motivate the need for a predictive model of the finite rate phase change process. 

The Homogeneous Relaxation Model (HRM) defines a functional dependence in space and time to determine the finite-rate evaporation rate. The original HRM model was proposed to approximate flashing experiments [7]. The HRM model has been used extensively in computational studies of flashing and spray flows. These include works on flashing cryogenic nitrogen [8], water [9], and include studying CO2 expansion during refrigeration cycles [2]. Although the HRM model has seen success when applied to flashing flows using fits to experimental data, general HRM models have not been proposed to capture the range of evaporation rates present in the dilute zones created by flashing sprays. Instead, droplet evaporation should follow the D2 law, which states that evaporation is proportional to the surface area (which for a droplet is proportional to the diameter squared). Similar to HRM models, correlations with experiments have been created to model droplet evaporation. A popular correlation was proposed by [10] which uses film theory to model heat transfer through a droplet. 

In addition to correlation models for droplet flows, there are kinetic models [11, 12, 13] that are built on fundamental molecular assumptions and can capture a wider range of physical processes, including evaporation, condensation, and flashing. A popular model is based on the Hertz-Knudsen approach [11, 12] and has been explored in multiple works [14, 15]. Although the Hertz-Knudsen phase change model is derived in a manner to achieve the generality required to handle flows ranging from both flashing and droplet evaporation systems, a critical aspect of modeling mass transfer with the Hertz-Knudsen model is determining the interfacial surface area. 

2 

## _1.2. Spray Modeling_ 

Several approaches exist for modeling interfacial area at different levels of fidelity. A widely used approach is known as the Discrete Droplet Model (DDM) [16] where the liquid phase is transported using Lagrangian particles that represent a grouping of droplets with identical velocity and diameter in order to reduce computational cost. In the DDM, the dynamics of the liquid are informed by the gas phase which is transported using Eulerian methods. Several studies have shown that the DDM approach is effective in capturing sprays [17, 18, 19]. In general, determining the size of the droplets as well as the modeling of the dynamics, involve model tuning with experiments [20, 21, 22, 23]. 

However, modeling the liquid with an Eulerian model has been shown to converge well when enough spatial resolution is provided to represent interfaces using interface capturing methods. If fully resolved, these simulations are considered a detailed numerical simulation which is nearly representative of reality (same as direct numerical simulation, absent capturing the smallest satellite droplets) [24]. Multiple works have attempted this with advanced numerical techniques, including AMR [25, 26], to simulate primary atomization. In this work, larger LES grids are of interest to reduce computational costs while attempting to retain a high-fidelity prediction. In a consistent framework, the diffuse interface methodology used to capture interfaces as described in [1] can be used to represent a subgrid transport of unresolved liquid features. The surface density approach (Σ) involves adding a transport equation to the governing equation which represents the surface area of the liquid [27, 28, 29, 30, 31]. In an LES setting, this transport equation represents the subgrid liquid content which is not resolvable by the mesh. Multiple studies have shown that Eulerian Σ models can achieve reasonable agreement with experiments in dense spray regions [21, 32, 33, 34]. Furthermore, the models for dense spray can either be combined with Lagrangian spray models (ELSA) to capture the dilute zone in sprays, or additional Eulerian models can be designed to predict the dilute spray. 

The Σ models have been successfully applied in both RANS and LES contexts [30, 33, 9, 31]. In the RANS context [30], all of the breakup regimes are modeled with the spray model. In the LES context, hybrid approaches have been explored that use interface capturing schemes in resolvable regions of the spray, and the Σ spray model for the unresolved regions [31, 35]. Furthermore, LES studies have been completed in which the primary spray features are fine enough to justify using a Σ spray model for all scales [34, 36]. 

## _1.3. Outline_ 

This work extends the system described in [1] to include additional multi-physics effects. This starts by generalizing the system of equations from an interface resolved multiphase system to include sub-grid scale models for mass, momentum, and liquid spray. Additionally, phase change capabilities are added to model the transfer of mass required to reach thermo-chemical equilibrium. The additional modeling capabilities are derived in a manner consistent with the phase constrained species diffusion model described in [1]. Section 2 overviews the filtered governing equations used for LES simulation. Section 3 summarizes the baseline numerical method where more details can be found in [1]. Section 4 describes the basis for phase change modeling and the proposed implementation of a thermodynamically bounded finite-rate phase change model. Section 5 covers the sub-grid model for interfacial surface area used to inform the finite-rate phase change approach. Section 6 presents results for a non-evaporating and an evaporating Spray A case, and show that both match well with the experimentally reported results. Finally, section 7 provides conclusions and future outlook. 

## **2. Large Eddy Simulation** 

In order to extend the capabilities of the solver to handle engineering scale problems while retaining high-fidelity accuracy, large eddy simulations (LES) will be used in this work. To obtain the system of equations used in LES the compressible multiphase multi-component governing equations can be filtered and Favre averaged. In this work, the filter is represented by the grid, and the Favre averaging operator applied to a general field _f_ is given by, 

**==> picture [252 x 24] intentionally omitted <==**

3 

After applying a filter and Favre averaging to the governing equations (and ignoring the subgrid content from the viscous and interface regularization terms) the system can be written as, 

**==> picture [403 x 56] intentionally omitted <==**

**==> picture [395 x 28] intentionally omitted <==**

where, to clearly distinguish between different phases, the phase of a material will be indicated by _p_ , where the subscript _p_ = 1 _,_ 2, and the components within a given phase are indicated by the superscript _c_ , where 1 _≤ c ≤ N_ . Index notation will be used for coordinates and tensor components, although we do not imply index notation for _p_ and _c_ . The definition for the primitive variables includes _Yp[c]_[as][the][mass][of][component] _c_ of phase _p_ per total mass, _ui_ = [ _u, v, w_ ] _[T]_ as the velocity vector, _P_ as the pressure, _T_ as the temperature, _ρ_ as the mixture density, and _ρE_ as the total energy per unit volume defined as _ρE_ = _ρe_ + 2[1] _[ρu][i][u][i]_[where] _[e]_ is the internal energy per unit volume. 

To model the sub-grid content present in an LES setting, sub-grid stresses have been added to the righthand-side of the system and are indicated with superscript _t_ . For instance, the combined viscous stress tensor and the subgrid momentum flux, _τij_ + _τij_ ~~_[t]_~~ , are defined as, 

**==> picture [341 x 25] intentionally omitted <==**

~~_t_~~ and _Qi_ + _Qi_ is defined as, 

**==> picture [349 x 31] intentionally omitted <==**

where _λ_ is the heat conductivity of the mixture and _h[c] p_[is][the][specific][enthalpy][of][component] _[c]_[of][phase] _[p]_[.] Following the model used for the species mass diffusion in [1], the filtered species diffusion along with the subgrid mass diffusion term is given by, 

**==> picture [449 x 32] intentionally omitted <==**

where _Xp[c]_[is][the][mixture][molar][fraction][of][component] _[c]_[of][phase] _[p]_[,] _[X][p]_[is][the][mixture][molar][fraction][of] phase _p_ , _Yp_ is the mixture mass fraction of phase _p_ , _Wp[c]_[is][the][molecular][weight][of][component] _[c]_[,] _[W][p]_[is][the] molecular weight of phase _p_ , and _Dp[c]_[is][the][mass][diffusivity][of][component] _[c]_[within][phase] _[p]_[.] The subgrid terms are closed using a constant _σ_ -model [37] for _µ[t]_ , _λ[t]_ is modeled assuming a constant turbulent Prandlt number, _Cpµ[t] /λ[t]_ = 0 _._ 7, and _Dp[c]_[is][modeled][assuming][a][constant][turbulent][Schmidt] number, _µ[t] /_ ( _ρDp[c]_[) = 0] _[.]_[7][.][The mixing rules and calculations for all thermodynamic quantities and equations] of state follow what was described in [1]. To model the subgrid mixing content of sprays, the turbulent liquid flux is closed using, 

**==> picture [284 x 30] intentionally omitted <==**

where for resolved interfaces, the liquid flux can be closed using an interface regularization model, as explored in past work [1]. In addition to modeling the subgrid content for species transport, momentum, total energy, and spray, a term in Eq. 2 to model the mass transfer from phase change, _m_ ˙ _[c] p_ , is added to the system. The phase change modeling is discussed in Section 4. 

4 

## **3. Numerical Method** 

The spatial discretization is a hybridization of a high-order kinetic energy and entropy preserving skewsymmetric scheme [38, 39, 40] with a high-resolution Godunov scheme for sharp gradients described in [1]. The Godunov scheme used for gradient regions is based on essentially non-oscillatory interpolations designed to satisfy the interface equilibrium conditions across material interfaces and reconstructs numerical fluxes using an approximate HLLC Riemann solver [1]. To avoid unphysical numerical states, a positivitypreserving check is used to locally switch to a first-order reconstruction to guarantee a robust scheme [41]. The baseline high-order hybrid numerical scheme provides low-dissipation in smooth regions and high-resolution capturing of sharp gradients without introducing spurious numerical oscillations [42]. The implementation details of the hybrid scheme are provided in Appendix A. The system of equations is implemented in generalized curvilinear coordinates for general application on both Cartesian, rectilinear, and generalized curvilinear grids [43]. The time-integration is an explicit third-order strong stability preserving Runge Kutta scheme [44]. For all simulations in this work, a temporal CFL based on advection and diffusion timescales of 0.5 is used. Finally, Navier-Stokes characteristic boundary conditions (NSCBC) are used for enforcing non-reflective inflow and outflow boundary conditions [45, 46, 47, 48, 42]. **4. Phase Change Modeling** The following section overviews the modeling assumptions to add phase change to an existing hydrodynamic solver. When a multiphase multi-component mixture is at thermo-chemical equilibrium no mass transfer can occur. In terms of thermodynamic quantities, thermo-chemical equilibrium for an isolated system is reached when _dS_ = 0 where _S_ = _S_ ( _U, V, Nij_ ) is the mole (N) weighted entropy for the mixture given by, 

**==> picture [388 x 31] intentionally omitted <==**

where _U_ is the molal internal energy and _V_ is the molal internal volume. As _U_ , _V_ , and _Nij_ are independent, each derivative term must individually be zero under equilibrium. For clarity of explanation, consider a pure two-phase substance ( _c_ = 1, _p_ = 2) in a closed system. For the first term, the change in entropy during phase exchange can be described as, 

**==> picture [327 x 27] intentionally omitted <==**

At equilibrium, _dS_ = 0, and after we note that energy stays constant in the closed system – implying that _U_ 1 + _U_ 2 = constant – we can note that _dU_ 1 = _−dU_ 2 and 

**==> picture [315 x 27] intentionally omitted <==**

_∂S_ Finally, with � _∂U_ � _V,Nij_[= 1] _[/T]_[,][we][can][observe][that][reaching][equilibrium][(] _[dS]_[= 0)][requires] _[T]_[1][=] _[ T]_[2][.][Or,][in] other words, the system reaches thermal equilibrium. Similarly for the second term, since _V_ 1 + _V_ 2 is constant, and � _∂V∂S_ � _U,Nij_[= (] _[P/T]_[)][, using the identical analysis as above we can obtain the additional requirement that] at equilibrium _P_ 1 = _P_ 2, i.e. mechanical equilibrium. The first two constraints are assumed in the formulation of the four-equation multiphase model which is used in this work. In other models, like the seven equation multiphase model [49], the thermal and mechanical equilibrium between phases would need to be reached as part of the phase change process. Instead, in this work, only the relaxation of the final term in Eq. 9 must be captured to model phase change. 

The third term in Eq. 9 can be analyzed by recognizing, 

**==> picture [276 x 40] intentionally omitted <==**

where the chemical potential _µi_ is given by, 

**==> picture [385 x 28] intentionally omitted <==**

with _G_ as the Gibbs free-energy, _H_ as the enthalpy, and _A_ as the Helmoltz free-energy. Therefore, at thermo-chemical equilibrium for a one component two phase mixture we can write, 

**==> picture [279 x 11] intentionally omitted <==**

From conservation of mass we can say _d_ ( _N_ 1 + _N_ 2) = 0 so _dN_ 1 = _−dN_ 2. So, in equilibrium, _µ_ 1 = _µ_ 2. When the chemical potential is not in equilibrium, 

**==> picture [344 x 25] intentionally omitted <==**

where if we combine with the conservation of mass equation we can show, 

**==> picture [319 x 21] intentionally omitted <==**

So, if _µ_ 1 _> µ_ 2 _⇒ dN_ 2 _>_ 0, and material moves from phase 1 to phase 2. Conversely, if _µ_ 2 _> µ_ 1 _⇒ dN_ 1 _>_ 0, and material moves from phase 2 to phase 1. From this analysis, we can see that reaching equilibrium with the chemical potential is the driving force of mass exchange between phases. 

With the concept of chemical potential driving phase change, we can see from the definition of chemical potential in Eq. 13 for a specified temperature and pressure for all phases (which is naturally enforced with the four-equation model) the chemical potential is defined in terms of the Gibbs free energy, _G_ , where _G_ =[�] _i[g][i][N][i]_[and] _[g][i]_[=] _[ h][i][ −][s][i][T]_[.][Therefore,][a][general][form][for] _m_ ˙ _[c] p_ in the four-equation context is, 

**==> picture [276 x 12] intentionally omitted <==**

where _ν_ = _ν_ ( _P, T, Yp[c][,]_[ Σ)][contains][a][chemical][relaxation][inverse][timescale][and][is][dependent][on][the][thermo-] dynamic state as well as the interfacial area between phases, Σ. For more general applicability and ease of implementation in a general CFD solver, an equivalent model in the LES setting can be defined as, 

**==> picture [283 x 21] intentionally omitted <==**

where _Y eq[c] p_[is][the][mass][fraction][of][component] _[c]_[in][phase] _[p]_[when][the][system][has][reached][thermo-chemical] equilibrium. The following sections overview two approaches to modeling the timescale _τ_ . Section 4.1 provides a computational approach if we assume that _τ ⇒_ 0, and Section 4.2 provides a methodology for determining a finite value for _τ_ . 

## _4.1. Homogeneous Equilibrium Model_ 

As described in Section 1.1.1, a popular approach for modeling phase change is assuming an infinite rate relaxation term for the Gibbs free-energy time-scale ( _ν ⇒∞_ or _τ ⇒_ 0). Using this assumption is known as the Homogeneous Equilibrium Model (HEM). The infinite relaxation rate assumption of the HEM allows for multiple paths for numerically reaching a thermo-chemical equilibrium state. For example, numerical methodologies, including exact iterative procedures [3], or approximate solvers [4], have been studied. In this work, a fast approximate algorithm is used to determine the equilibrium state. The strategy avoids an expensive Newton-Raphson iterative solver and instead uses an algebraic update which approximates the thermo-chemical state and converges over multiple simulation time-steps. 

6 

## _4.1.1. Approximate HEM Solver_ 

The approximate HEM approach relaxes the system using a UV-flash approach. In a UV-flash algorithm, both the mixture internal energy and the mixture specific volume stay constant over the phase change process. During the phase change process, no energy or mass leaves the computational cell and the equilibrium mixture pressure and temperature change to account for the mass transfer. The approximate algorithm proposed by [4] to find an estimate for the mass fraction of component _c_ in the gas phase _g_ after phase change is described below, 

1. Estimate updated mass fraction to satisfy mass conservation for all ( _P, T_ ) using, 

**==> picture [290 x 26] intentionally omitted <==**

where, 

**==> picture [341 x 37] intentionally omitted <==**

and the system above is evaluated at the conditions, 

**==> picture [291 x 13] intentionally omitted <==**

2. Estimate updated mass fraction to satisfy energy conservation for all ( _P, T_ ) using, 

**==> picture [279 x 26] intentionally omitted <==**

where, 

**==> picture [321 x 37] intentionally omitted <==**

and the system above is evaluated at the conditions, 

**==> picture [291 x 12] intentionally omitted <==**

3. Estimate the updated mass fraction to satisfy chemical equilibrium using, 

**==> picture [292 x 28] intentionally omitted <==**

where the system above is evaluated at the original pressure and temperature condition and the definition for _Psat_ is defined in Eq. 27. 

4. Check bound states 

**==> picture [438 x 23] intentionally omitted <==**

5. Take the estimate for the new _Yp[c]_[which][has][the][smallest][variation][for][the][original][using,] 

**==> picture [378 x 14] intentionally omitted <==**

7 

Once the system is at the equilibrium state, all mass fraction estimates from the algorithm above are identical. The saturation state is determined based on a fitted Antoine equation given by, 

**==> picture [277 x 14] intentionally omitted <==**

where the parameters, _A_ , _B_ , and _C_ are found in the NIST database [50]. 

To evaluate the chemical equilibrium state in the current work, a time-splitting approach is used. During time-integration, at the end of each SSP-RK3 sub-step (after the conserved variables have been updated in time including advection and diffusion processes), the mass transfer from phase change is added to the conserved variables. Since the phase change algorithm is a UV-flash algorithm, the total momentum and energy in the system are unchanged, and only the species mass equations involved in the phase change algorithm must be updated. 

## _4.1.2. HEM Verification_ 

The implementation of the HEM solver can be tested on multiple 1-D shock-tube simulations that are designed to be sensitive to the thermodynamic composition. These tests do not contain resolved phase interfaces, and instead represent a homogeneous mixture going through a phase change process, which is indicative of an under-resolved multiphase flow scenario common in spray applications. Additionally, in these tests the liquid-gas interface is not regularized using the CDI model as it has been in [1]. Recall, that for the four-equation multiphase model the volume fraction is implicitly determined as a function of the local thermodynamic state. When phase change is active, the mass transfer will determine the interface location by relaxing towards thermo-chemical equilibrium. Regularizing the phase interface using a phase field method can move mass away from the thermo-chemical equilibrium location and we hypothesis that these differing dynamics between CDI and the HEM can create an incompatibility. Attempting to solve for the interface shape with both the phase chance and the CDI model can create an unphysical competition between interface regularization towards a tanh profile, and relaxing interface towards achieving thermo-chemical equilibrium. The competition could potentially be avoided by adding a condition to the regularization terms described in [1] to only activate in multiphase zones near thermo-chemical equilibrium, though this theory has not been tested. Future work is required to develop a phase field method that is intrinsically compatible with both the four-equation model and relaxation toward thermo-chemical equilibrium during a phase change process. 

The following section will verify the implementation of the HEM phase change algorithm using a code-tocode verification test with a reference from [51]. As this is a code-to-code test, the same material parameters used in [51] are used here. For the phase change shock-tube simulations in this work, a spatial resolution of (400 x 1) was used with an advection CFL = 0.5. 

The first test starts with an air-water mixture which is far from a phase change boundary. The initial mass fractions are liquid water with _Yl[water]_ = 0 _._ 1, _Yg[water]_ = 0 _._ 2, and _Yg[air]_ = 0 _._ 7 uniformly distributed throughout the shock-tube. Across the shock located at _x_ = 0 _._ 5 in a domain that ranges from 0 _≤ x ≤_ 1, a pressure jump of _PL_ = 0 _._ 2 MPa to _PR_ = 0 _._ 1 MPa is present. The temperature and density are set in order to have the mixture in thermo-chemical equilibrium, and the velocity is initially zero in the domain. For this mixture, thermo-chemical equilibrium results in _ρL ≃_ 1 _._ 874 kg/m[3] , _ρR ≃_ 0 _._ 984 kg/m[3] , _TL ≃_ 360 _._ 48K, and _TR ≃_ 343 _._ 22K. The results compared with the published work from [51] are shown in Figure 1 at _t_ = 1ms. As shown, the results agree well between the two solvers. As this mixture is far from the phase change boundary, the results without phase change (given by the gray line: ) show a large difference compared to the phase change location in multiple fields, including the temperature and liquid mass fraction. In this case, the generality of the HEM phase change approach is showcased as both evaporation and condensation are correctly predicted. 

The second case used to verify the implementation of the HEM phase change solver is a shock in an air dominated mixture. The initial temperature is set to _T_ = 293K throughout the domain and the same initial pressure ratio of 2 from the previous case is used. In this case, the mass fraction _Yg[air]_ = 0 _._ 98 everywhere in the domain, and the mass fractions of liquid water and water vapor are deduced from satisfying thermochemical equilibrium with the initial condition. This results in _Y[water] l L_[=][0] _[.]_[013][,][and] _[Y][water] l R_[=][0] _[.]_[006][.] Figure 2 compares the results from the current work with [51] at _t_ = 1 ms. Similar to the first case in Figure 

8 

