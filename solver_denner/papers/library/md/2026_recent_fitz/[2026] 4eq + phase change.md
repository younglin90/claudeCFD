
## An LES model with finite-rate phase change and subgrid spray based on a thermodynamically consistent four-equation multiphase model

Henry Collisa,∗, Shahab Mirjalilib,a, Makrand Khanwalea, Ali Mania, Gianluca Iaccarinoa

aDepartment of Mechanical Engineering, Stanford, CA 94305, USA bFLOW, Department of Engineering Mechanics, KTH Royal Institute of Technology, SE-10044 Stockholm, Sweden

Abstract

In this work, an LES model with finite-rate phase change and subgrid spray based on a high-resolution numerical scheme for multiphase multi-component simulations which satisfies interface equilibrium and phase immiscibility conditions [1] is proposed. The multiphase model is based on a robust implementation of the four-equation multiphase model which assumes a strict subgrid equilibrium of pressure, temperature, and velocity. Critically, the equilibrium assumptions of the four-equation model provide large computational savings compared to modeling the full non-equilibrium multiphase system. To obtain predictive capabilities with these restrictive equilibrium assumptions, a new phase-confined form of the Eulerian Σ spray model is proposed to predict subgrid interfacial surface area while avoiding unphysical leakage across interfaces. Additionally, an improved finite rate phase change model which is thermodynamically bounded by the equilibration of the Gibbs-free energy is coupled with the Σ equation to model complex phase change regimes. The full modeling framework is validated using the Engine Combustion Network (ECN) Spray A case in non-evaporating and evaporating conditions and shows excellent agreement with experimental measurements.

Keywords: Subgrid spray, phase change, compressible, four-equation model, multicomponent, multiphase

1. Introduction

Modeling engineering-scale multiphase multi-component systems is critical for multiple applications, including combustion engines with fuel injection atomizers, rocket propellant injection, and fire-suppression systems. To incorporate the critical physics required by these applications, multiphysics modeling is needed that captures intraphase mixing, high-density ratio interfaces, and mass-transfer models. In this work, a high-resolution numerical strategy for simulating multiphase multi-component flows based on the fourequation model is extended to include finite-rate phase change models and subgrid spray. Simulations involving subgrid spray physics can physically expect to have non-equilibrium between phase velocity, as the disperse and discrete phases do not need to have the same inertial properties. One of the overarching goals of this work is to showcase the extent in which the four-equation multiphase equilibrium conditions can be pushed while retaining robust and accurate multiphase simulations. Compared to the fully non-equilibrium multiphase models, the four-equation approach assumes mechanical (pressure) and thermal (temperature) equilibrium between phases. The governing system is simplified to a shared momentum and energy equation, greatly reducing the computational cost compared to the non-equilibrium models. In this work, we show that coupling the four-equation numerical framework proposed in [1] with appropriate models for subgrid spray and phase change in the four-equation setting achieves excellent agreement with experiments without requiring the full non-equilibrium multiphase system.

∗Corresponding author Email addresses: hcollis@stanford.edu (Henry Collis), msey@kth.se (Shahab Mirjalili), khanwale@stanford.edu (Makrand Khanwale), alimani@stanford.edu (Ali Mani), jops@stanford.edu (Gianluca Iaccarino)


# arXiv:2604.23846v1  [physics.flu-dyn]  26 Apr 2026

1.1. Phase Change

Modeling the phase change process, including boiling, evaporation, flashing, and condensation, is critical in multiphase systems. In particular, engineering applications including diesel spray engines and liquid rocket propulsion systems require transfer of mass from the liquid propellant to gas to achieve successful ignition. In general, interphase mass transfer is governed by a non-equilibrium chemical potential between components in separate phases (details in Section 4). Modeling the rate at which the system reaches chemical equilibrium is a critical aspect for accurate phase change modeling. Two general assumptions exist for phase change modeling which can separate the modeling mechanism for the phase change processes listed above (e.g. evaporation vs boiling): instantaneous and finite rate equilibration of the chemical potential.

1.1.1. Instantaneous Equilibrium Assuming an instantaneous equilibration of the chemical potential between phases is known as the Homogeneous Equilibrium Model (HEM) [2]. The HEM model is appropriate for two regimes: fully resolved DNS where the phase change rate is governed by heat conduction (evaporation), and under-resolved spray which are governed by rapid phase change processes, including flashing and cavitation events [3, 4]. For the cases between the extremes of interface resolved multiphase interfaces and substantially under-resolved simulations of flashing or cavitation, the HEM will over-predict the time-scale of mass transfer. The infinite rate relaxation of the chemical potential to equilibrium of the HEM is not applicable for interfacial structures which have a finite surface area but are not resolvable by a computational grid. These regimes could include using the HEM to model the evaporation of subgrid ligaments and droplets.

1.1.2. Finite-Rate Models Finite-rate phase change models are designed to alleviate the issues of overpredicting mass transfer which occurs with the HEM. The finite-rate mass transfer time-scale is dependent on both the thermodynamic state, the composition, and the interfacial surface area. The simplest approach for modeling finite rate phase change is assuming a relaxation time-scale which is constant for all space and time throughout a simulation. This approach has been overviewed by multiple past studies [5, 6] which have qualitatively shown how important engineering quantities, including the vaporization mass present in a high-pressure fuel injector [5], are strongly impacted by the assigned phase change rate, showing that the ad-hoc approach of assigning a constant spatial-temporal phase change rate to a system is not practical for producing a predictive and generalizable computational model. The results of these works motivate the need for a predictive model of the finite rate phase change process. The Homogeneous Relaxation Model (HRM) defines a functional dependence in space and time to determine the finite-rate evaporation rate. The original HRM model was proposed to approximate flashing experiments [7]. The HRM model has been used extensively in computational studies of flashing and spray flows. These include works on flashing cryogenic nitrogen [8], water [9], and include studying CO2 expansion during refrigeration cycles [2]. Although the HRM model has seen success when applied to flashing flows using fits to experimental data, general HRM models have not been proposed to capture the range of evaporation rates present in the dilute zones created by flashing sprays. Instead, droplet evaporation should follow the D2 law, which states that evaporation is proportional to the surface area (which for a droplet is proportional to the diameter squared). Similar to HRM models, correlations with experiments have been created to model droplet evaporation. A popular correlation was proposed by [10] which uses film theory to model heat transfer through a droplet. In addition to correlation models for droplet flows, there are kinetic models [11, 12, 13] that are built on fundamental molecular assumptions and can capture a wider range of physical processes, including evaporation, condensation, and flashing. A popular model is based on the Hertz-Knudsen approach [11, 12] and has been explored in multiple works [14, 15]. Although the Hertz-Knudsen phase change model is derived in a manner to achieve the generality required to handle flows ranging from both flashing and droplet evaporation systems, a critical aspect of modeling mass transfer with the Hertz-Knudsen model is determining the interfacial surface area.

2

1.2. Spray Modeling

Several approaches exist for modeling interfacial area at different levels of fidelity. A widely used approach is known as the Discrete Droplet Model (DDM) [16] where the liquid phase is transported using Lagrangian particles that represent a grouping of droplets with identical velocity and diameter in order to reduce computational cost. In the DDM, the dynamics of the liquid are informed by the gas phase which is transported using Eulerian methods. Several studies have shown that the DDM approach is effective in capturing sprays [17, 18, 19]. In general, determining the size of the droplets as well as the modeling of the dynamics, involve model tuning with experiments [20, 21, 22, 23]. However, modeling the liquid with an Eulerian model has been shown to converge well when enough spatial resolution is provided to represent interfaces using interface capturing methods. If fully resolved, these simulations are considered a detailed numerical simulation which is nearly representative of reality (same as direct numerical simulation, absent capturing the smallest satellite droplets) [24]. Multiple works have attempted this with advanced numerical techniques, including AMR [25, 26], to simulate primary atomization. In this work, larger LES grids are of interest to reduce computational costs while attempting to retain a high-fidelity prediction. In a consistent framework, the diffuse interface methodology used to capture interfaces as described in [1] can be used to represent a subgrid transport of unresolved liquid features. The surface density approach (Σ) involves adding a transport equation to the governing equation which represents the surface area of the liquid [27, 28, 29, 30, 31]. In an LES setting, this transport equation represents the subgrid liquid content which is not resolvable by the mesh. Multiple studies have shown that Eulerian Σ models can achieve reasonable agreement with experiments in dense spray regions [21, 32, 33, 34]. Furthermore, the models for dense spray can either be combined with Lagrangian spray models (ELSA) to capture the dilute zone in sprays, or additional Eulerian models can be designed to predict the dilute spray. The Σ models have been successfully applied in both RANS and LES contexts [30, 33, 9, 31]. In the RANS context [30], all of the breakup regimes are modeled with the spray model. In the LES context, hybrid approaches have been explored that use interface capturing schemes in resolvable regions of the spray, and the Σ spray model for the unresolved regions [31, 35]. Furthermore, LES studies have been completed in which the primary spray features are fine enough to justify using a Σ spray model for all scales [34, 36].

1.3. Outline

This work extends the system described in [1] to include additional multi-physics effects. This starts by generalizing the system of equations from an interface resolved multiphase system to include sub-grid scale models for mass, momentum, and liquid spray. Additionally, phase change capabilities are added to model the transfer of mass required to reach thermo-chemical equilibrium. The additional modeling capabilities are derived in a manner consistent with the phase constrained species diffusion model described in [1]. Section 2 overviews the filtered governing equations used for LES simulation. Section 3 summarizes the baseline numerical method where more details can be found in [1]. Section 4 describes the basis for phase change modeling and the proposed implementation of a thermodynamically bounded finite-rate phase change model. Section 5 covers the sub-grid model for interfacial surface area used to inform the finite-rate phase change approach. Section 6 presents results for a non-evaporating and an evaporating Spray A case, and show that both match well with the experimentally reported results. Finally, section 7 provides conclusions and future outlook.

2. Large Eddy Simulation

In order to extend the capabilities of the solver to handle engineering scale problems while retaining high-fidelity accuracy, large eddy simulations (LES) will be used in this work. To obtain the system of equations used in LES the compressible multiphase multi-component governing equations can be filtered and Favre averaged. In this work, the filter is represented by the grid, and the Favre averaging operator applied to a general field f is given by,


![Equation](images/[2026] 4eq + phase change_eq001.png)


![Equation](images/[2026] 4eq + phase change_eq002.png)

3

After applying a filter and Favre averaging to the governing equations (and ignoring the subgrid content from the viscous and interface regularization terms) the system can be written as,


![Equation](images/[2026] 4eq + phase change_eq003.png)


![Equation](images/[2026] 4eq + phase change_eq004.png)


![Equation](images/[2026] 4eq + phase change_eq005.png)


![Equation](images/[2026] 4eq + phase change_eq006.png)


![Equation](images/[2026] 4eq + phase change_eq007.png)


![Equation](images/[2026] 4eq + phase change_eq008.png)


![Equation](images/[2026] 4eq + phase change_eq009.png)


![Equation](images/[2026] 4eq + phase change_eq010.png)


![Equation](images/[2026] 4eq + phase change_eq011.png)


![Equation](images/[2026] 4eq + phase change_eq012.png)


![Equation](images/[2026] 4eq + phase change_eq013.png)


![Equation](images/[2026] 4eq + phase change_eq014.png)


![Equation](images/[2026] 4eq + phase change_eq015.png)

where, to clearly distinguish between different phases, the phase of a material will be indicated by p, where the subscript p = 1, 2, and the components within a given phase are indicated by the superscript c, where 1 ≤c ≤N. Index notation will be used for coordinates and tensor components, although we do not imply index notation for p and c. The definition for the primitive variables includes Y c p as the mass of component c of phase p per total mass, ui = [u, v, w]T as the velocity vector, P as the pressure, T as the temperature, ρ as the mixture density, and ρE as the total energy per unit volume defined as ρE = ρe + 1

2ρuiui where e is the internal energy per unit volume. To model the sub-grid content present in an LES setting, sub-grid stresses have been added to the righthand-side of the system and are indicated with superscript t. For instance, the combined viscous stress tensor and the subgrid momentum flux, τij + τijt, are defined as,


![Equation](images/[2026] 4eq + phase change_eq016.png)


![Equation](images/[2026] 4eq + phase change_eq017.png)


![Equation](images/[2026] 4eq + phase change_eq018.png)


![Equation](images/[2026] 4eq + phase change_eq019.png)

and Qi + Qi t is defined as,


![Equation](images/[2026] 4eq + phase change_eq020.png)


![Equation](images/[2026] 4eq + phase change_eq021.png)


![Equation](images/[2026] 4eq + phase change_eq022.png)


![Equation](images/[2026] 4eq + phase change_eq023.png)


![Equation](images/[2026] 4eq + phase change_eq024.png)

where λ is the heat conductivity of the mixture and hc p is the specific enthalpy of component c of phase p. Following the model used for the species mass diffusion in [1], the filtered species diffusion along with the subgrid mass diffusion term is given by,

Jc p,i + Jc p,i t = −ρ


![Equation](images/[2026] 4eq + phase change_eq025.png)


![Equation](images/[2026] 4eq + phase change_eq026.png)


![Equation](images/[2026] 4eq + phase change_eq027.png)


![Equation](images/[2026] 4eq + phase change_eq028.png)


![Equation](images/[2026] 4eq + phase change_eq029.png)


![Equation](images/[2026] 4eq + phase change_eq030.png)


![Equation](images/[2026] 4eq + phase change_eq031.png)


![Equation](images/[2026] 4eq + phase change_eq032.png)


![Equation](images/[2026] 4eq + phase change_eq033.png)


![Equation](images/[2026] 4eq + phase change_eq034.png)


![Equation](images/[2026] 4eq + phase change_eq035.png)


![Equation](images/[2026] 4eq + phase change_eq036.png)

where Xc p is the mixture molar fraction of component c of phase p, Xp is the mixture molar fraction of phase p, Yp is the mixture mass fraction of phase p, W c p is the molecular weight of component c, Wp is the molecular weight of phase p, and Dc p is the mass diffusivity of component c within phase p. The subgrid terms are closed using a constant σ-model [37] for µt, λt is modeled assuming a constant turbulent Prandlt number, Cpµt/λt = 0.7, and Dc p is modeled assuming a constant turbulent Schmidt number, µt/(ρDc p) = 0.7. The mixing rules and calculations for all thermodynamic quantities and equations of state follow what was described in [1]. To model the subgrid mixing content of sprays, the turbulent liquid flux is closed using,


![Equation](images/[2026] 4eq + phase change_eq037.png)


![Equation](images/[2026] 4eq + phase change_eq038.png)

where for resolved interfaces, the liquid flux can be closed using an interface regularization model, as explored in past work [1]. In addition to modeling the subgrid content for species transport, momentum, total energy, and spray, a term in Eq. 2 to model the mass transfer from phase change, ˙ mcp, is added to the system. The phase change modeling is discussed in Section 4.

4

3. Numerical Method

The spatial discretization is a hybridization of a high-order kinetic energy and entropy preserving skewsymmetric scheme [38, 39, 40] with a high-resolution Godunov scheme for sharp gradients described in [1]. The Godunov scheme used for gradient regions is based on essentially non-oscillatory interpolations designed to satisfy the interface equilibrium conditions across material interfaces and reconstructs numerical fluxes using an approximate HLLC Riemann solver [1]. To avoid unphysical numerical states, a positivitypreserving check is used to locally switch to a first-order reconstruction to guarantee a robust scheme [41]. The baseline high-order hybrid numerical scheme provides low-dissipation in smooth regions and high-resolution capturing of sharp gradients without introducing spurious numerical oscillations [42]. The implementation details of the hybrid scheme are provided in Appendix A. The system of equations is implemented in generalized curvilinear coordinates for general application on both Cartesian, rectilinear, and generalized curvilinear grids [43]. The time-integration is an explicit third-order strong stability preserving Runge Kutta scheme [44]. For all simulations in this work, a temporal CFL based on advection and diffusion timescales of 0.5 is used. Finally, Navier-Stokes characteristic boundary conditions (NSCBC) are used for enforcing non-reflective inflow and outflow boundary conditions [45, 46, 47, 48, 42].

4. Phase Change Modeling

The following section overviews the modeling assumptions to add phase change to an existing hydrodynamic solver. When a multiphase multi-component mixture is at thermo-chemical equilibrium no mass transfer can occur. In terms of thermodynamic quantities, thermo-chemical equilibrium for an isolated system is reached when dS = 0 where S = S(U, V, Nij) is the mole (N) weighted entropy for the mixture given by,


![Equation](images/[2026] 4eq + phase change_eq039.png)


![Equation](images/[2026] 4eq + phase change_eq040.png)


![Equation](images/[2026] 4eq + phase change_eq041.png)


![Equation](images/[2026] 4eq + phase change_eq042.png)


![Equation](images/[2026] 4eq + phase change_eq043.png)


![Equation](images/[2026] 4eq + phase change_eq044.png)


![Equation](images/[2026] 4eq + phase change_eq045.png)


![Equation](images/[2026] 4eq + phase change_eq046.png)


![Equation](images/[2026] 4eq + phase change_eq047.png)


![Equation](images/[2026] 4eq + phase change_eq048.png)


![Equation](images/[2026] 4eq + phase change_eq049.png)


![Equation](images/[2026] 4eq + phase change_eq050.png)

where U is the molal internal energy and V is the molal internal volume. As U, V , and Nij are independent, each derivative term must individually be zero under equilibrium. For clarity of explanation, consider a pure two-phase substance (c = 1, p = 2) in a closed system. For the first term, the change in entropy during phase exchange can be described as,


![Equation](images/[2026] 4eq + phase change_eq051.png)


![Equation](images/[2026] 4eq + phase change_eq052.png)


![Equation](images/[2026] 4eq + phase change_eq053.png)


![Equation](images/[2026] 4eq + phase change_eq054.png)


![Equation](images/[2026] 4eq + phase change_eq055.png)

At equilibrium, dS = 0, and after we note that energy stays constant in the closed system – implying that U1 + U2 = constant – we can note that dU1 = −dU2 and �∂S1 ∂U1


![Equation](images/[2026] 4eq + phase change_eq056.png)


![Equation](images/[2026] 4eq + phase change_eq057.png)


![Equation](images/[2026] 4eq + phase change_eq058.png)


![Equation](images/[2026] 4eq + phase change_eq059.png)

Finally, with �∂S ∂U �

V,Nij = 1/T, we can observe that reaching equilibrium (dS = 0) requires T1 = T2. Or, in other words, the system reaches thermal equilibrium. Similarly for the second term, since V1+V2 is constant, and �∂S ∂V �

U,Nij = (P/T), using the identical analysis as above we can obtain the additional requirement that at equilibrium P1 = P2, i.e. mechanical equilibrium. The first two constraints are assumed in the formulation of the four-equation multiphase model which is used in this work. In other models, like the seven equation multiphase model [49], the thermal and mechanical equilibrium between phases would need to be reached as part of the phase change process. Instead, in this work, only the relaxation of the final term in Eq. 9 must be captured to model phase change. The third term in Eq. 9 can be analyzed by recognizing, �∂S ∂Ni


![Equation](images/[2026] 4eq + phase change_eq060.png)


![Equation](images/[2026] 4eq + phase change_eq061.png)


![Equation](images/[2026] 4eq + phase change_eq062.png)

5

where the chemical potential µi is given by,


![Equation](images/[2026] 4eq + phase change_eq063.png)


![Equation](images/[2026] 4eq + phase change_eq064.png)


![Equation](images/[2026] 4eq + phase change_eq065.png)


![Equation](images/[2026] 4eq + phase change_eq066.png)


![Equation](images/[2026] 4eq + phase change_eq067.png)


![Equation](images/[2026] 4eq + phase change_eq068.png)


![Equation](images/[2026] 4eq + phase change_eq069.png)


![Equation](images/[2026] 4eq + phase change_eq070.png)


![Equation](images/[2026] 4eq + phase change_eq071.png)

with G as the Gibbs free-energy, H as the enthalpy, and A as the Helmoltz free-energy. Therefore, at thermo-chemical equilibrium for a one component two phase mixture we can write,


![Equation](images/[2026] 4eq + phase change_eq072.png)

From conservation of mass we can say d(N1 + N2) = 0 so dN1 = −dN2. So, in equilibrium, µ1 = µ2. When the chemical potential is not in equilibrium,


![Equation](images/[2026] 4eq + phase change_eq073.png)


![Equation](images/[2026] 4eq + phase change_eq074.png)


![Equation](images/[2026] 4eq + phase change_eq075.png)

where if we combine with the conservation of mass equation we can show,


![Equation](images/[2026] 4eq + phase change_eq076.png)


![Equation](images/[2026] 4eq + phase change_eq077.png)

So, if µ1 > µ2 ⇒dN2 > 0, and material moves from phase 1 to phase 2. Conversely, if µ2 > µ1 ⇒dN1 > 0, and material moves from phase 2 to phase 1. From this analysis, we can see that reaching equilibrium with the chemical potential is the driving force of mass exchange between phases. With the concept of chemical potential driving phase change, we can see from the definition of chemical potential in Eq. 13 for a specified temperature and pressure for all phases (which is naturally enforced with the four-equation model) the chemical potential is defined in terms of the Gibbs free energy, G, where G = �

i giNi and gi = hi −siT. Therefore, a general form for ˙ mcp in the four-equation context is,


![Equation](images/[2026] 4eq + phase change_eq078.png)

where ν = ν(P, T, Y c p , Σ) contains a chemical relaxation inverse timescale and is dependent on the thermodynamic state as well as the interfacial area between phases, Σ. For more general applicability and ease of implementation in a general CFD solver, an equivalent model in the LES setting can be defined as,


![Equation](images/[2026] 4eq + phase change_eq079.png)


![Equation](images/[2026] 4eq + phase change_eq080.png)

where Y eqc p is the mass fraction of component c in phase p when the system has reached thermo-chemical equilibrium. The following sections overview two approaches to modeling the timescale τ. Section 4.1 provides a computational approach if we assume that τ ⇒0, and Section 4.2 provides a methodology for determining a finite value for τ.

4.1. Homogeneous Equilibrium Model

As described in Section 1.1.1, a popular approach for modeling phase change is assuming an infinite rate relaxation term for the Gibbs free-energy time-scale (ν ⇒∞or τ ⇒0). Using this assumption is known as the Homogeneous Equilibrium Model (HEM). The infinite relaxation rate assumption of the HEM allows for multiple paths for numerically reaching a thermo-chemical equilibrium state. For example, numerical methodologies, including exact iterative procedures [3], or approximate solvers [4], have been studied. In this work, a fast approximate algorithm is used to determine the equilibrium state. The strategy avoids an expensive Newton-Raphson iterative solver and instead uses an algebraic update which approximates the thermo-chemical state and converges over multiple simulation time-steps.

6

4.1.1. Approximate HEM Solver The approximate HEM approach relaxes the system using a UV-flash approach. In a UV-flash algorithm, both the mixture internal energy and the mixture specific volume stay constant over the phase change process. During the phase change process, no energy or mass leaves the computational cell and the equilibrium mixture pressure and temperature change to account for the mass transfer. The approximate algorithm proposed by [4] to find an estimate for the mass fraction of component c in the gas phase g after phase change is described below,

1. Estimate updated mass fraction to satisfy mass conservation for all (P, T) using,


![Equation](images/[2026] 4eq + phase change_eq081.png)

where,


![Equation](images/[2026] 4eq + phase change_eq082.png)


![Equation](images/[2026] 4eq + phase change_eq083.png)


![Equation](images/[2026] 4eq + phase change_eq084.png)


![Equation](images/[2026] 4eq + phase change_eq085.png)


![Equation](images/[2026] 4eq + phase change_eq086.png)


![Equation](images/[2026] 4eq + phase change_eq087.png)


![Equation](images/[2026] 4eq + phase change_eq088.png)


![Equation](images/[2026] 4eq + phase change_eq089.png)


![Equation](images/[2026] 4eq + phase change_eq090.png)

and the system above is evaluated at the conditions,


![Equation](images/[2026] 4eq + phase change_eq091.png)

2. Estimate updated mass fraction to satisfy energy conservation for all (P, T) using,


![Equation](images/[2026] 4eq + phase change_eq092.png)

where,


![Equation](images/[2026] 4eq + phase change_eq093.png)


![Equation](images/[2026] 4eq + phase change_eq094.png)


![Equation](images/[2026] 4eq + phase change_eq095.png)


![Equation](images/[2026] 4eq + phase change_eq096.png)


![Equation](images/[2026] 4eq + phase change_eq097.png)


![Equation](images/[2026] 4eq + phase change_eq098.png)


![Equation](images/[2026] 4eq + phase change_eq099.png)


![Equation](images/[2026] 4eq + phase change_eq100.png)


![Equation](images/[2026] 4eq + phase change_eq101.png)

and the system above is evaluated at the conditions,


![Equation](images/[2026] 4eq + phase change_eq102.png)

3. Estimate the updated mass fraction to satisfy chemical equilibrium using,


![Equation](images/[2026] 4eq + phase change_eq103.png)


![Equation](images/[2026] 4eq + phase change_eq104.png)


![Equation](images/[2026] 4eq + phase change_eq105.png)

where the system above is evaluated at the original pressure and temperature condition and the definition for Psat is defined in Eq. 27.

4. Check bound states

(a) if (Y c g m −Y c g )(Y c g e −Y c g ) < 0 or if (Y c g m −Y c g )(Y c g sat−Y c g ) < 0 no phase change occurs. Otherwise, move on to the next step.

5. Take the estimate for the new Y c p which has the smallest variation for the original using,


![Equation](images/[2026] 4eq + phase change_eq106.png)

7

Once the system is at the equilibrium state, all mass fraction estimates from the algorithm above are identical. The saturation state is determined based on a fitted Antoine equation given by,


![Equation](images/[2026] 4eq + phase change_eq107.png)

where the parameters, A, B, and C are found in the NIST database [50]. To evaluate the chemical equilibrium state in the current work, a time-splitting approach is used. During time-integration, at the end of each SSP-RK3 sub-step (after the conserved variables have been updated in time including advection and diffusion processes), the mass transfer from phase change is added to the conserved variables. Since the phase change algorithm is a UV-flash algorithm, the total momentum and energy in the system are unchanged, and only the species mass equations involved in the phase change algorithm must be updated.

4.1.2. HEM Verification The implementation of the HEM solver can be tested on multiple 1-D shock-tube simulations that are designed to be sensitive to the thermodynamic composition. These tests do not contain resolved phase interfaces, and instead represent a homogeneous mixture going through a phase change process, which is indicative of an under-resolved multiphase flow scenario common in spray applications. Additionally, in these tests the liquid-gas interface is not regularized using the CDI model as it has been in [1]. Recall, that for the four-equation multiphase model the volume fraction is implicitly determined as a function of the local thermodynamic state. When phase change is active, the mass transfer will determine the interface location by relaxing towards thermo-chemical equilibrium. Regularizing the phase interface using a phase field method can move mass away from the thermo-chemical equilibrium location and we hypothesis that these differing dynamics between CDI and the HEM can create an incompatibility. Attempting to solve for the interface shape with both the phase chance and the CDI model can create an unphysical competition between interface regularization towards a tanh profile, and relaxing interface towards achieving thermo-chemical equilibrium. The competition could potentially be avoided by adding a condition to the regularization terms described in [1] to only activate in multiphase zones near thermo-chemical equilibrium, though this theory has not been tested. Future work is required to develop a phase field method that is intrinsically compatible with both the four-equation model and relaxation toward thermo-chemical equilibrium during a phase change process. The following section will verify the implementation of the HEM phase change algorithm using a code-tocode verification test with a reference from [51]. As this is a code-to-code test, the same material parameters used in [51] are used here. For the phase change shock-tube simulations in this work, a spatial resolution of (400 x 1) was used with an advection CFL = 0.5. The first test starts with an air-water mixture which is far from a phase change boundary. The initial mass fractions are liquid water with Y water l = 0.1, Y water g = 0.2, and Y air g = 0.7 uniformly distributed throughout the shock-tube. Across the shock located at x = 0.5 in a domain that ranges from 0 ≤x ≤1, a pressure jump of PL = 0.2 MPa to PR = 0.1 MPa is present. The temperature and density are set in order to have the mixture in thermo-chemical equilibrium, and the velocity is initially zero in the domain. For this mixture, thermo-chemical equilibrium results in ρL ≃1.874 kg/m3, ρR ≃0.984 kg/m3, TL ≃360.48K, and TR ≃343.22K. The results compared with the published work from [51] are shown in Figure 1 at t = 1ms. As shown, the results agree well between the two solvers. As this mixture is far from the phase change boundary, the results without phase change (given by the gray line: ) show a large difference compared to the phase change location in multiple fields, including the temperature and liquid mass fraction. In this case, the generality of the HEM phase change approach is showcased as both evaporation and condensation are correctly predicted. The second case used to verify the implementation of the HEM phase change solver is a shock in an air dominated mixture. The initial temperature is set to T = 293K throughout the domain and the same initial pressure ratio of 2 from the previous case is used. In this case, the mass fraction Y air g = 0.98 everywhere in the domain, and the mass fractions of liquid water and water vapor are deduced from satisfying thermochemical equilibrium with the initial condition. This results in Y water l L = 0.013, and Y water l R = 0.006. Figure 2 compares the results from the current work with [51] at t = 1 ms. Similar to the first case in Figure

8


> **Figure 1: Water-air phase change shock tube. With HEM phase change ( ), without phase change ( ) and reference with phase change [51] ( )**

1, there is excellent agreement between the codes, even though different numerical schemes are used for the spatial discretization. The effect of including phase change is still apparent, as the temperature field has a strong dependence on the application of the HEM solver. Furthermore, the liquid water experiences both evaporation and condensation, showcasing the capabilities of the consistent application of thermo-chemical relaxation to predict complex phase change. The final case used to verify the ability of the HEM phase change solver to find the thermo-chemical equilibrium state is a 1D cavitation problem. The initial state involves a constant pressure, temperature, and density field at thermo-chemical equilibrium. The pressure is at 1MPa, temperature is at 293K, and density is defined using the equation of state. The velocity field is set with a positive velocity in the right of the domain of 1 m/s, and a negative velocity in the left (expansion fan) of -1 m/s. During the simulation, the velocity field induces a low pressure zone in the center of the domain and a cavitation event produces vapor from the pure liquid. The thermodynamic parameters for determining the phase change saturation state are obtained from [51], but in this case the mixture is defined using the NASG stiffened gas parameters from [4] to obtain results that are consistent with past works [4, 51]. As shown in Figure 3, the predicted vapor mass fraction results match very well with the reference solution [51] at t=3.5 ms, providing confidence in the current implementation of the HEM algorithm.

4.2. Finite-Rate Phase Change As described in Section 4.1, the assumptions of an infinite rate relaxation of the chemical potential to equilibrium are not physical for interfacial structures which have a finite surface area but are not resolvable

9


> **Figure 2: Air-water phase change shock tube. With HEM phase change ( ), without phase change ( ), and reference with phase change [51] ( ).**

by a LES grid. The current section will briefly overview approaches to modeling the finite rate timescale. As mentioned in Section 4, the timescale is dependent on both the thermodynamic state (P, T), the composition (Y c p ), and the interfacial surface area Σ. As discussed in Section 1.1.2, most established finite-rate phase change approaches do not incorporate all of the required dependencies, and instead rely on assumptions or correlations to match experiments. In this work, the Hertz-Knudsen model is used because it is generally derived from kinetic theory and is applicable to flows ranging from flashing conditions to droplet evaporation. The form used in this work is given by,


![Equation](images/[2026] 4eq + phase change_eq108.png)

λ is an O(1) user tuning parameter which can be tuned to experiments and either defined as a function using correlations [14], or can defined as a constant [15]. The Hertz-Knudsen phase change model is derived in a manner that achieves the generality required to handle flows ranging from both flashing and droplet evaporation systems. What becomes critical for accurately modeling mass transfer with the Hertz-Knudsen model is determining the interfacial surface area. A modeling framework designed to predict the sub-grid interfacial area will be described in Section 5.

10


> **Figure 3: Water expansion (cavitation) phase change shock tube. With HEM phase change ( ), without phase change ( ), and reference with phase change [51] ( ).**

4.2.1. Thermodynamically Bounding Finite Rate Phase Change Unlike the HEM model, an issue with the Hertz-Knudsen, D2-law, and HRM (described in Appendix B) finite rate phase change models is that they are not bound to obey the chemical potential equilibrium condition described in Section 4. Critically, this means that these finite rate models could unreasonably predict phase change when none should occur, or overshoot the prediction of the HEM (which is unphysical). In order to avoid this issue, we propose a new implementation for finite rate phase change models. Instead of a direct mass transfer term, the finite rate phase change models can be formulated as a time scale, which informs the following expression,


![Equation](images/[2026] 4eq + phase change_eq109.png)


![Equation](images/[2026] 4eq + phase change_eq110.png)

where Y eqc p is defined using the HEM, and τ is defined using a finite rate phase change model. As the Hertz-Knudsen model is based on more fundamental concepts, it will be used as the finite rate phase change model in this work. To have a thermodynamically consistent bound on the phase change rate, we define τ as,


![Equation](images/[2026] 4eq + phase change_eq111.png)


![Equation](images/[2026] 4eq + phase change_eq112.png)


![Equation](images/[2026] 4eq + phase change_eq113.png)


![Equation](images/[2026] 4eq + phase change_eq114.png)


![Equation](images/[2026] 4eq + phase change_eq115.png)


![Equation](images/[2026] 4eq + phase change_eq116.png)

where in this expression ˙mc p,HK is defined using Eq. 28. With this implementation, the Hertz-Knudsen finite rate phase change model will always predict a physically achievable thermodynamic state. In order

11

to inform the Hertz-Knudsen phase change model, a representation of the subgrid surface area is required. The following section overviews the subgrid modeling approach used in this work.

5. Sub-grid Spray Modeling

As described in Section 4.2, a critical aspect of modeling the rate of phase change is estimating the interfacial surface area. In this section, the focus is on extending the physical application of the numerical schemes described in Section 3 to consistently include the Σ model for LES flows with features far smaller than the grid scale.

5.1. Phase Constrained Σ′ modeling

The original application of the Σ model to the LES setting involves transporting the equation below [31],


![Equation](images/[2026] 4eq + phase change_eq117.png)


![Equation](images/[2026] 4eq + phase change_eq118.png)


![Equation](images/[2026] 4eq + phase change_eq119.png)


![Equation](images/[2026] 4eq + phase change_eq120.png)


![Equation](images/[2026] 4eq + phase change_eq121.png)


![Equation](images/[2026] 4eq + phase change_eq122.png)


![Equation](images/[2026] 4eq + phase change_eq123.png)

where Σ′ represents the phase interface surface area density that is not resolvable by the mesh size. The total interfacial surface area is obtained by, Σ = Σmin +Σ′ where Σmin represents the theoretical amount of surface area that is possible in an LES setting. Although the resolved surface area is mathematically defined as the magnitude of the gradient of the volume fraction, |∇ϕl|, this definition is not always appropriate when the Σ model is used in an LES. To illustrate this point, consider a region of the mesh that is too coarse to resolve the relevant flow structures. Instead of using an interface regularization model, this region relies on a subgrid spray model to distribute the liquid volume over several grid cells. Downstream of the breakup event, it is possible that the mesh resolution becomes sufficient to resolve theoretical spray features. However, since the upstream liquid has already been represented as a diffuse distribution and transported as an unresolved spray, the fine-scale interface structures cannot be easily reintroduced into the simulation where the exact expression for Σmin = |∇ϕl| can be used. To address such situations, where interfaces may be partially resolved or artificially diffused even in high-resolution regions, a minimum surface density model, Σmin, is commonly employed as,


![Equation](images/[2026] 4eq + phase change_eq124.png)


![Equation](images/[2026] 4eq + phase change_eq125.png)


![Equation](images/[2026] 4eq + phase change_eq126.png)

where Σmin represents the features of the spray that theoretically could be fully resolved by the grid if no spray model was present [52, 34, 36]. In Eq. 31, the diffusion term on the right-hand-side represents the diffusion of surface area density due to the sub-grid transport. In the literature this is known as the turbulent diffusion flux and is generally based on a gradient law closure as shown in Eq. 31 [30]. The final term on the right-hand-side, Σ′ int, consists of source and sink terms to represent physical spray events that include flashing, dense turbulent breakup, dilute breakup, and coalescence. The model forms for the source terms will be described in Section 5.2 and Section 5.3. The transport equation in Eq. 31 has been used in past work to successfully model the transport and production of surface area throughout Eulerian simulations [29, 31]. In general, the Σ′ equation is a scalar transport equation that should consistently follow the Eulerian flow field produced by the Navier-Stokes equations. In the absence of source terms, the quantity Σ′ should only physically exist within the volume of the gas phase, since it represents the liquid surface area. A pure liquid core should not contain Σ′, as there is no surface area in a pure phase. Furthermore, within a computational cell that contains liquid surface area, all surface area should exist as a confined scalar within the volume occupied by the gaseous phase. In other words, even in the subgrid setting, the liquid droplets, ligaments, and other interfacial features that make up Σ, will exist within the gaseous volume and either be partially or fully surrounded by the gas phase. However, this desired modeling quality will not be satisfied for the modeled transport equation shown in Eq. 31. In Eq. 31, the transport equation Σ′ occupies the entire volume of each cell, which has been shown

12

in past work to lead to leakage of confined scalars across phase interfaces [53]. Following the requirements presented in [53], Eq. 31 can be rewritten as


![Equation](images/[2026] 4eq + phase change_eq127.png)


![Equation](images/[2026] 4eq + phase change_eq128.png)


![Equation](images/[2026] 4eq + phase change_eq129.png)


![Equation](images/[2026] 4eq + phase change_eq130.png)


![Equation](images/[2026] 4eq + phase change_eq131.png)


![Equation](images/[2026] 4eq + phase change_eq132.png)


![Equation](images/[2026] 4eq + phase change_eq133.png)


![Equation](images/[2026] 4eq + phase change_eq134.png)


![Equation](images/[2026] 4eq + phase change_eq135.png)


![Equation](images/[2026] 4eq + phase change_eq136.png)


![Equation](images/[2026] 4eq + phase change_eq137.png)


![Equation](images/[2026] 4eq + phase change_eq138.png)


![Equation](images/[2026] 4eq + phase change_eq139.png)


![Equation](images/[2026] 4eq + phase change_eq140.png)


![Equation](images/[2026] 4eq + phase change_eq141.png)

where the first term represents the physical diffusion of the transport of Σ′ by the subgrid flow. In this work, a form of the turbulent diffusion term has been proposed to confine the diffusion of the scalar, Σ′ to the gas phase. Compared to the diffusion term in Eq. 31, the diffusion term now contains the addition of ˜ϕg to limit the diffusion of the scalar quantity, Σ′, to the volume of the gas. Additionally, a second term is added to the system as a required consistency term to ensure that Σ′ is properly transported when the turbulent liquid flux spray model is applied to the mass, momentum, and energy equations. Together, these terms allow the transport of Σ′ to remain fully consistent and leakage-free across phase interfaces. In this work, the gradient diffusion closure of, Rc g,i = νt Sct ∇�ϕ, is used to close the unclosed turbulent liquid flux [31].

Note, unlike the species diffusion subgrid flux (Jc p,j t) which acts to model subgrid diffusion between species within phases (intraphase mixing for multi-component mixtures), the turbulent liquid flux model defines the closure of the interaction between phases (interphase mixing between multiphase mixtures). In the proposed phase-confined transport for the subgrid surface area equation, the final term on the right-hand-size contains sources and sinks for modeling the breakup and coalescence of liquid droplets. Most past works have used a non-linear form for a general source term as [27, 29, 9, 31],


![Equation](images/[2026] 4eq + phase change_eq142.png)


![Equation](images/[2026] 4eq + phase change_eq143.png)


![Equation](images/[2026] 4eq + phase change_eq144.png)


![Equation](images/[2026] 4eq + phase change_eq145.png)


![Equation](images/[2026] 4eq + phase change_eq146.png)

Although this has been the standard application of the source term in Σ modeling and has shown success in multiple applications, for small time-scales (at minimum equal to the simulation time-step) the non-linearity in the definition of S can result in the integration of ˙Σ′ over/under-shooting past the predicted equilibrium value Σeq. Instead, a linear relaxation term is used in this work given as,


![Equation](images/[2026] 4eq + phase change_eq147.png)


![Equation](images/[2026] 4eq + phase change_eq148.png)

The modeling of all spray-physics can be framed using Eq. 35 towards a predicted equilibrium value without over/under-shooting Σ′ eq during time-integration. For different breakup processes, both a time-scale and an equilibrium value are modeled. The following sections will briefly review the source terms explored in this work and the required changes to allow for consistent integration into the current computational framework.

5.2. Dense Spray Source Terms

Within a dense region, multiple sources of surface area production can exist. In this work, two sources of production events will be modeled. The first is a surface area production caused by droplet breakup as a result of turbulence and aerodynamic interactions. The second is a production of surface area resulting from a flashing event within a dense liquid core.

5.2.1. Turbulent breakup For a dense spray with turbulent breakup, the modified approach of [54] is used to estimate the equilibrium surface density. The form of this equation is given by,


![Equation](images/[2026] 4eq + phase change_eq149.png)


![Equation](images/[2026] 4eq + phase change_eq150.png)

where σ is the surface tension coefficient and Wec is the critical Weber number which is a user-defined parameter. For the turbulent breakup model, Wec = 1.5 is used for all cases in this work. In the LES

13

setting, the time-scale used for the relaxation towards equilibrium is the magnitude of the strain-rate tensor as,


![Equation](images/[2026] 4eq + phase change_eq151.png)

and ksgs is the subgrid turbulent kinetic energy which we define as,


![Equation](images/[2026] 4eq + phase change_eq152.png)


![Equation](images/[2026] 4eq + phase change_eq153.png)

where ∆is the local grid size and Cs = 0.1 [55, 56]. This model for dense turbulent breakup has been successfully applied in multiple approaches including RANS and LES [57, 36].

5.2.2. Flashing Although not as commonly explored, explicit models for surface area production in dense zones from flashing have been recently formulated in the RANS context [57]. In the LES context this model can be written as,


![Equation](images/[2026] 4eq + phase change_eq154.png)


![Equation](images/[2026] 4eq + phase change_eq155.png)

where Wec is an O(1) user parameter for the flashing model defined using the correction factor described in [57]. Additionally, ˙rf is the growth rate of the nucleation sites at the time of breakup. Although the model contains terms describing bubble growth in a flashing setting, the source term is representative of the surface area production of the liquid/droplet phase. Consistent to what was done in [57], the timescale can be defined as, τflash = rf/ ˙rf. (40)

where rf is the final radius reached before the bubbles merge or breakup. The remaining challenge is predicting the growth rate and the final bubble radius. We follow the same modeling choices as [57], though we add a thermodynamic consistency check to limit the growth rate to the maximum amount of phase change which is possible by equilibrating the chemical potential. The original model [57] defines the bubble growth rate assuming the heat diffusion limit [58],


![Equation](images/[2026] 4eq + phase change_eq156.png)


![Equation](images/[2026] 4eq + phase change_eq157.png)

where the Jakob number is defined as,


![Equation](images/[2026] 4eq + phase change_eq158.png)


![Equation](images/[2026] 4eq + phase change_eq159.png)

with Lv as the latent heat of vaporization and Dc g is the binary diffusion coefficient for the gas. Similar to [57], the bubble final radius can be defined using the following expressions,


![Equation](images/[2026] 4eq + phase change_eq160.png)


![Equation](images/[2026] 4eq + phase change_eq161.png)


![Equation](images/[2026] 4eq + phase change_eq162.png)


![Equation](images/[2026] 4eq + phase change_eq163.png)

where ϕd is the volume ratio of the dissolved gas in the fluid needed to form nucleation sites, η is the maximum bubble growth volume decided as η = 0.74 which is the largest volume fraction achievable by orderly spherical packing. Additionally, the critical radius rcrit is defined using the Young-Laplace equation as the maximum radius that sustains a bubble’s internal pressure for a given amount of surface tension. In that term, Pve is the vapor pressure inside the bubble given by the Gibbs-Duhem equation as,


![Equation](images/[2026] 4eq + phase change_eq164.png)


![Equation](images/[2026] 4eq + phase change_eq165.png)

14

Although the model described above has been shown to successfully model the flashing of cryogenic nitrogen [57], similar to the HRM, it is not bound to follow the thermodynamically defined chemical potential equilibrium conditions. To address this issue, instead of using Eq. 41 directly, we can estimate the amount of mass transfer from reaching thermo-chemical equilibrium with bubble growth as,


![Equation](images/[2026] 4eq + phase change_eq166.png)


![Equation](images/[2026] 4eq + phase change_eq167.png)


![Equation](images/[2026] 4eq + phase change_eq168.png)


![Equation](images/[2026] 4eq + phase change_eq169.png)


![Equation](images/[2026] 4eq + phase change_eq170.png)


![Equation](images/[2026] 4eq + phase change_eq171.png)

According to these modeling assumptions, the maximum growth rate of the bubbles during a flashing process can be bound by the amount of phase change present when reaching chemical equilibrium. Therefore, Eq. 41 can be modified using the results from Eq. 47 to consistently predict flashing in zones that have the potential to experience an evaporation or vaporization event as,


![Equation](images/[2026] 4eq + phase change_eq172.png)


![Equation](images/[2026] 4eq + phase change_eq173.png)


![Equation](images/[2026] 4eq + phase change_eq174.png)


![Equation](images/[2026] 4eq + phase change_eq175.png)


![Equation](images/[2026] 4eq + phase change_eq176.png)


![Equation](images/[2026] 4eq + phase change_eq177.png)


![Equation](images/[2026] 4eq + phase change_eq178.png)


![Equation](images/[2026] 4eq + phase change_eq179.png)

5.3. Dilute spray source terms In the dilute spray regime the spray features are assumed to be small droplets which can undergo further secondary breakup, or can coalesce into larger droplets, reducing the overall interfacial surface area.

5.3.1. Secondary breakup The model for the secondary breakup of droplets is given by [30],


![Equation](images/[2026] 4eq + phase change_eq180.png)

where urel is an estimate of the relative velocity, and the equilibrium Weber number is given as Wec = 12.0 [30, 57] in this work for the secondary breakup model. Furthermore, the breakup timescale is,


![Equation](images/[2026] 4eq + phase change_eq181.png)

where r is the current estimate of the radius of the droplets calculated with the Sauter mean diameter (SMD), and T is an experimentally determined correlation for secondary droplet breakup [59]. The correlation is given by, T = 1.9(We2ndBreak −12.0)−0.25(1.0 + 2.2Oh)1.6. (51)

where We2ndBreak = ρgu2 rel2r/σ, and the Ohnesorge number, Oh = µl √ρlσ2r.

5.3.2. Coalescence Lastly, in the dilute region the coalescence of droplets is estimated in the LES setting using particle collision theory and the subgrid turbulent kinetic energy as [30, 57],


![Equation](images/[2026] 4eq + phase change_eq182.png)


![Equation](images/[2026] 4eq + phase change_eq183.png)

where, We = 4ϕlρ∗ksgs


![Equation](images/[2026] 4eq + phase change_eq184.png)


![Equation](images/[2026] 4eq + phase change_eq185.png)


![Equation](images/[2026] 4eq + phase change_eq186.png)

Wec = 12.0 [30, 57] is used for the coalescence model for all cases in this work.

15

5.4. Evaporation source term The resulting surface area predictions achieved by incorporating the production and destruction mechanisms for surface area described above directly impacts the phase change rate. Critically, the transfer of mass also impacts the surface area of the liquid phase. In order to keep the surface area predictions consistent, the surface area must be updated when mass is transferred during the phase change process. In general, it is difficult to determine whether mass transfer from phase change increases or decreases surface area. Assuming all phase change due to evaporation and condensation takes place with non-interacting spherical droplets, the following equation is appropriate [30, 57],


![Equation](images/[2026] 4eq + phase change_eq187.png)


![Equation](images/[2026] 4eq + phase change_eq188.png)

where evaporation decreases surface area, and condensation increases. Some authors do not include this term as they argue the change of surface area from phase change is implicitly handled by the other source terms [36]. In the current study, a modified form of Eq. 54 is used as,


![Equation](images/[2026] 4eq + phase change_eq189.png)

which is proportional to the surface area update in Eq. 54 and captures the general effect of phase change without any robustness issues when Y c l gets small. Although we have added Eq. 55 to be philosophically consistent with phase change impacting surface area, this term is not expected to dominate in relation to the other production/destruction sources of Σ′ and could be neglected without a strong impact on the results, as discussed in [36].

5.5. Spray regime indicators The integration of the source terms mentioned above are classified into two broad sections,


![Equation](images/[2026] 4eq + phase change_eq190.png)

where the SSpray terms contain the source terms associated with dense turbulent breakup, dilute secondary breakup, and dilute coalescence. However, the SP haseChange terms contain a production term associated with the flashing process (as this requires mass transfer) and a term associated with evaporation/condensation. For both the spray and phase change sections, it is necessary to determine the regime of phase change in order to use the appropriate models.

5.5.1. Dense vs. Dilute The underlying spray physics for a dense spray are expected to consist of either a connected phase (discrete phase), or a large collection of droplets which would generally correspond to a large liquid volume fraction of the cell. In contrast, the spray physics for the dilute regime can be modeled as a disperse phase that does not occupy a large volume fraction of the cell. In this work, two indicators are used to determine if the spray region consists of a dense or dilute spray. First, the dense vs. dilute indicator proposed by [57], which is based on the liquid volume fraction, is used to define the dense region. Instead of solely using the indicator from [57], an additional indicator based on an a-priori estimate of the Hinze scale can be used to distinguish between dense and dilute regions. The a-priori Hinze scale estimate [26] is given by,


![Equation](images/[2026] 4eq + phase change_eq191.png)


![Equation](images/[2026] 4eq + phase change_eq192.png)


![Equation](images/[2026] 4eq + phase change_eq193.png)


![Equation](images/[2026] 4eq + phase change_eq194.png)

where D is the diameter of the injector for the spray, and Reg = ρgUiD

µg with Ui as the maximum inlet velocity. So, overall, in this work the dense vs. dilute spray region indicator is given by,


![Equation](images/[2026] 4eq + phase change_eq195.png)


![Equation](images/[2026] 4eq + phase change_eq196.png)

16

where the final term in Eq. 58 is given by,


![Equation](images/[2026] 4eq + phase change_eq197.png)


![Equation](images/[2026] 4eq + phase change_eq198.png)

which compares the current local SMD size (SMD = 6ϕ/Σ) to the a-priori estimate of the Hinze scale. In comparison to the original indicator based on the volume fraction, the additional Hinze scale indicator adds a new classification mechanism for the low volume fraction zone of ϕl < 0.26. Even though the volume fraction is low, it is still possible that either large drops or connected structures exist in the subgrid which should be treated with the dense spray models. As such, to use the dilute models we ensure that both the volume fraction and the estimated droplet sizes are small as expected for a droplet which could either be dominated by coalescence or secondary breakup mechanics. All together the spray source terms are written as, SSpray = ΨSturbdense + (1 −Ψ)(S2ndBreak + Scol). (60)

5.5.2. Flashing vs. Evaporation In addition to indicating the dense and dilute regions of the flow, it is important to indicate which regions of the flow undergo phase change processes dominated by flashing, and which regions which are dominated by evaporation. We propose using an indicator based on correlations of flashing vs. mechanical breakup processes proposed by fitting correlations experimentally [60]. The indicator used in this work can be written as,


![Equation](images/[2026] 4eq + phase change_eq199.png)


![Equation](images/[2026] 4eq + phase change_eq200.png)

max


![Equation](images/[2026] 4eq + phase change_eq201.png)


![Equation](images/[2026] 4eq + phase change_eq202.png)


![Equation](images/[2026] 4eq + phase change_eq203.png)


![Equation](images/[2026] 4eq + phase change_eq204.png)


![Equation](images/[2026] 4eq + phase change_eq205.png)

and the critical Jakob number is defined as [60],


![Equation](images/[2026] 4eq + phase change_eq206.png)

Given the indicator for flashing, the addition of the phase change related source terms can be added to the equations as, SP haseChange = ΨΓSflashing + Sevap(1 −ΨΓ). (63)

With the indicators determined, the computational modeling framework is fully defined. The overall system consists of the multiphase multi-component compressible Navier-Stokes equations which are coupled through a finite-rate phase change model to a consistently transported phase-constrained Σ′ equation for the subgrid interfacial surface area density. Using this governing system, simulations of the classic engine combustion network (ECN) Spray A at non-evaporating and evaporating conditions are used to validate the approach with the ECN experiments.

6. LES ECN Spray A Simulations

To validate the modeling methodologies for both phase change and spray models presented in Sections 4 and 5, the ECN Spray A configuration is studied in this section. The ECN non-evaporating conditions are used to validate the spray modeling terms from Section 5, and the ECN evaporating conditions are used to test both the finite-rate phase change model and the fully coupled system involving both spray characterization and mass transfer. The ECN Spray A case consists of the high-pressure injection of ndodecane into a quiescent environment of nitrogen [61]. The physical parameters for both the evaporating and non-evaporating cases are shown in Table 1. The properties used to determine the state of the mixture in the simulations are in Table 2. The NASG EOS parameters were defined using the approach described in [62]. As opposed to capturing the full injector geometry, in this work the injection of liquid n-dodecane into the domain is modeled using a uniform inlet matching the injector nozzle diameter of 0.0894 mm [63]. Similar

17

Thermodynamic Property Non-evaporating Spray Evaporating Spray Injection pressure [MPa] 150 150 Injection temperature [K] 343 363 Ambient temperature [K] 303 900 Ambient pressure [MPa] 2 6 Ambient composition N2 N2


> **Table 1: Conditions for both non-evaporating and evaporating ECN Spray A cases[61].**

Parameter Liquid n-dodecane Vapor n-dodecane Nitrogen


![Equation](images/[2026] 4eq + phase change_eq207.png)


![Equation](images/[2026] 4eq + phase change_eq208.png)


![Equation](images/[2026] 4eq + phase change_eq209.png)

A [-] 4.10549 - B [-] 1625.928 - C [-] -92.839 -


> **Table 2: Simulation parameters for Spray A evaporation and non-evaporation cases.**

to past works, synthetic turbulent forcing is used to trigger the transition to a turbulent jet as opposed to simulating the upstream injector geometry [36, 34]. Additionally, the base inflow conditions use the ECN injector calculator to define time-dependent mass-flow rate obtained from the CMT injection simulator [64]. A grid-convergence study is completed for all quantities of interest for the non-evaporating Spray A case. Stretched Cartesian grids are used for all cases and a sample grid used in this work can be seen in Figure 4. The grids are defined using the parameters in Table 3.


![Equation](images/[2026] 4eq + phase change_eq210.png)


> **Table 3: Grid parameters for setting the grid. The axial and transverse expansion ratios (ER) are constant for all cases. The base spacing used differs with varying resolutions. For Grid 4, a minimum grid spacing of 5 × 10−6 is used for both axial and transverse directions until the spacing using the compounding expansion ratios (starting at the base spacing of 2 × 10−6) compounds more than 5 × 10−6. This allows the smallest grid spacing of 5 × 10−6 to be used for a larger area of cells before increasing to larger sizes.**

Away from the inlet, all boundaries are specified using NSCBC non-reflective outflow conditions to limit impacts from the boundary on the flow [46]. As mentioned in Section Appendix A, the simulations are based on a high-order skew-symmetric KEEP central scheme hybridized with a WENO5Z Godunov scheme [65] used across shocks and material interfaces. The details of the Godunov scheme are in [1]. A temporal CFL=0.5 is used for all cases.

18


> **Figure 4: Grid 1 used for Spray A non-evaporating case. Panel a) shows the y vs. z plane, and panel b) shows the x vs. y plane. The axial span is from 0 to 20 mm, and the transverse span is from -1.25 to 1.25 mm.**

6.1. Spray A: non-evaporating case

To validate the application of finite-rate phase change informed by the phase constrained Σ model, we compare the predictions of the Σ model with those from the ECN Spray A experiment. First, we show the projected mass density (PMD [66]) near-injector results averaged over the flow times ranging from 0.4-0.8 ms. Figure 5 shows that the simulations qualitatively capture the spread of the liquid jet in the near-injector region. Visually, at any given axial location (x), both the experiment [66, 67] and the simulation shows a similar magnitude of PMD. Although the experimental results show a less symmetric solution, likely from a bias present in the complex upstream injection of n-dodecane, the radial agreement visually matches between the simulations and experiments.


> **Figure 5: Temporal average of the projected mass density [µg/mm2] (PMD) in the x vs. y plane. Top row is the LES results achieved with Grid 4 and the bottom row is the experimental result [67].**

A more quantitative analysis for the PMD is shown in Figure 6, where the experimental profiles are overlaid on the PMD results for varying mesh resolutions. Qualitatively, Figure 6 shows a result similar to Figure 5, where both the magnitude and the overall spread of the PMD are reasonably matched between the experiments and simulations. Additionally, Figure 6 shows how the refinement of the mesh decreases the differences between the spray width in the simulations and experiments. Furthermore, refining the mesh

19


> **Figure 6: Temporally averaged projected mass density [µg/mm2] (PMD) along the transverse direction for multiple axial locations. The left panel is for x = 0.1mm, middle panel is for x = 2 mm, and right panel is for x = 6 mm. Grid 1 ( ), Grid 2 ( ), Grid 3 ( ), Grid 4 ( ), and experiments [67]( ).**

nearly achieves mesh convergence between the two highest resolution runs. The overall agreement between the experimental and simulation PMD provides additional confidence that the underlying LES simulation is representative of the experiment. Although the phase change model is active during the simulation, since the flow field does not predict any evaporation (as this is a non-evaporating test-case) the surface area density equation is essentially decoupled from the Navier-Stokes equations. As such, matching the PMD can be viewed as properly simulating the dynamics of the spray with the turbulent liquid flux closure, but it does not validate the surface area predictions of the finite-rate phase change models in the simulations. So, although the results from the PMD analysis are promising, the critical step of validating the spray modeling with the ECN Spray A non-evaporating case requires additional comparison with experimental data. To investigate the accuracy of the spray model, Figure 7 shows the projected surface area density predicted by the phase constrained Σ model compared to the results from the experiments [68] for varying mesh resolutions. Similar to the results for the PMD, refining the grid in Figure 7 increases the accuracy of the surface area prediction relative to the ECN experiments [68]. Additionally, the two highest grid resolutions are nearly mesh converged. In all cases, the initial rise of the surface area density near the injection of liquid into the domain starts from a lower value than what is observed in the ECN experimental results. The initial rise is dominated by the effects of both spatial resolution and inflow conditions. The qualitative mismatch between simulations and experiments implies that modeling the injection of n-dodecane without the injector geometry does not fully capture the surface area injected into the domain. Even so, further into the domain we see that the spray model properly predicts the overall maximum magnitude of the projected surface area, as well as the reduction in surface area in the far-field region. The agreement between the simulations and experiments provides initial validation on the applicability of the four-equation model coupled with the

20


> **Figure 7: Temporal average of the projected surface area [m2/m2] (PSA) along the axial centerline. Grid 1 ( ), Grid 2 ( ), Grid 3 ( ), Grid 4 ( ), and experiments [68]( ).**

phase-constrained Σ′ transport equation to capture the relevant spray physics in the ECN Spray A case. Further comparisons between the simulation and the experiment can be made using the averaged SMD sizes at varying locations in the domain. Figure 8 shows the comparison of the simulation predictions with all grids, as well as the averaged SMD from experiments [68]. In all locations (except the near-field 4mm axial location where the simulations do not predict a wide enough spray), the predicted average SMD sizes from the simulations are in the same range as the experiments. In particular, some near-field locations, including the 4 mm axial location at -0.1 mm radial location, and some far field locations including the 8 mm axial at 0.5 mm radially, obtained close agreement between simulations and experiments.

6.2. Spray A: evaporating case

Following the validation of the non-evaporating Spray A case, the fully coupled system is tested with the evaporating case. As near grid-converged solutions are obtained with the finest grid studied in the non-evaporating case, the fine grid resolution is used for the evaporating ECN Spray A case. The additional physics present in this case includes the finite-rate phase change model coupled with the predictions from the Σ model, as well as the potential for flashing in the diesel jet changing the surface area predictions. A qualitative comparison with experiments [69, 71, 70, 72] can be seen in Figure 9, where the liquid ndodecane penetration length is shown with a blue iso-contour and the full extent of the gaseous n-dodecane is shown in the gray. The qualitative agreement between both the liquid penetration length and the gas extent into the domain is visually close to the experimental results. In the simulation, the liquid contour is defined using the 0.15% liquid volume fraction to be consistent with the experimental measurements following Mie-Scattering theory [73], and the gas n-dodecane contour is defined using the 0.1% gas mass fraction following experimental guidelines [74]. The overall visual behavior of both the liquid core throughout time, as well as the penetration of the gas cloud, match reasonably well between the experiments and fully coupled simulations. A more quantitative comparison between experiments and simulation with the liquid penetration length can be seen in Figure 10. Here, two experimental techniques are shown to classify the extent of the liquid

21


> **Figure 8: Temporal average of the Sauter mean diameter [m] (SMD) at varying transverse and axial locations compared to experiments [68].**


> **Figure 9: Qualitative comparison between evaporating Spray A experiments [69](left) and LES simulations (right) over time. From top to bottom, the times shown in both experiment and simulation are, 30µs, 50µs, 70µs, 89.9µs, 109.9µs. Blue contour represents liquid penetration using volume fraction of 0.15% and gray represents gaseous n-dodecane penetration with mass fraction of 0.1%.**

spray into the domain. In addition, the two experimental techniques included in Figure 10 provide a sense of the uncertainty range associated with the experimental readings. After an initial over-penetration of the liquid in the simulation, the majority of the simulation results fall within the experimental measurements. Overall, the agreement in liquid penetration length over time between the experiments and simulations validates the approach of using the four-equation model described in [1] coupled with the proposed thermo-

22


> **Figure 10: Quantitative comparison of liquid penetration into the domain compared to two experimental measurement techniques. LES ( ), experimental DBI [69, 70] ( ), and experimental Mie-scattering [71, 72] ( ).**

dynamically bounded finite-rate phase change model based on the surface area predictions to capture both complex finite-rate phase change and spray dynamics. The final quantity of interest to compare with experiments is the gas penetration length. The n-dodecane gas phase solely exists due to the finite-rate phase change of the injected liquid. The extent of the gas phase n-dodecane into the domain over time can be seen in Figure 11. In the simulation, there is a small over-estimation of the gas-penetration throughout time compared to the experimental measurements. As indicated by [36], the over-prediction of gas transport could be attributed to using the ideal gas law to represent a n-dodecane at high temperature and pressure, when in reality it behaves as a supercritical fluid. Future investigation and improvement of the simulation framework would involve increasing the fidelity of the equation of state to allow for more accurate representations of the supercritical n-dodecane. Overall, using a thermodynamically bounded and consistent finite-rate phase change model coupled with the predictions of subgrid surface area from the phase-constrained Σ′ equation provided reasonable agreement with experiments for both the non-evaporating and evaporating ECN spray A cases. Future investigations can include increasing the fidelity of the equation of state, as well as using the hybridization of the interface regularization terms for resolved regions with the surface area representation of the subgrid zones.

7. Conclusion

This work presented an extension of the thermodynamically consistent four-equation model presented in [1] to include finite-rate phase change and subgrid spray models. Critically, the thermal and mechanical equilibrium assumptions of the four-equation model enable a simple and cost-effective modeling framework for modeling spray dynamics compared to the fully non-equilibrium multiphase model. To achieve agreement with experimental results, a new finite-rate phase change model was formulated to remain thermodynamically bounded by the equilibration of the chemical potential to remove the ability for the phase change model to under/over-predict unphysical mass transfer. Additionally, the finite-rate phase change model was informed using a newly proposed phase-constrained form of the Eulerian Σ spray model. The Σ transport equation included models to capture the breakup and coalescence processes expected in sprays containing dense

23


> **Figure 11: Comparison of vapor n-dodecane penetration over time between experiments with confidence intervals, and LES simulations. LES ( ), experimental results [71, 72] with confidence bounds ( ).**

and dilute zones and a newly proposed thermodynamically bounded model for flashing in the LES setting. The full simulation framework was validated against the evaporating and non-evaporating ECN Spray A experiments by showing excellent agreement with multiple experimental diagnostic measurements. Future work can include adding mass transfer from chemical reactions to study spray combustion, extending the equation of state to higher fidelity models (e.g. tabular EOS), and investigating hybridizing the spray in the LES setting with regularization of the multiphase interface in resolved zones.

Appendix A. Hybrid Numerical Approach

The complete algorithm for the hybrid scheme used in this work can be seen in Figure A.12 and is described in additional detail in [42]. First, the computational cells are labeled using the indicator from Section Appendix A.1. After finding a convective flux with either the Godunov approach described in [1] or with the skew-symmetric approach described in [38, 39, 40], the solution is checked to be admissible [41]. If the skew-symmetric scheme returns a non-admissible solution, the Godunov approach is attempted before going directly to a 1st-order HLLC reconstruction [75] to ensure that the highest fidelity solution is obtained throughout the domain.

Appendix A.1. Density Shock Sensor

Before applying either the Godunov approach or the skew-symmetric scheme to a face, a density-based indicator is evaluated over the domain to label which cells require the more dissipative Godunov approach. The indicator is based on the evaluation of the sub-stencil smoothness parameter of the TENO6-A scheme [76]. As described in [76], the cutoff value CT is changed locally in space and time for the adaptive TENO6 scheme. In this work, the density indicator is adapted locally using,

24

Shock Sensor: Partial Densities

Yes

Godunov Approach Skew Symmetric

No

Admissible

Solution?

High-resolution

Reconstruction

Admissible

Solution?

Discontinuity?

High-order Reconstruction

No

Yes Yes

1st Order Reconstruction

No

Figure A.12: Flowchart of hybrid discretization scheme for the convective terms with positivity preserving limiter.


![Equation](images/[2026] 4eq + phase change_eq211.png)

where d = (1 −Φ)10(1 + 10Φ) and Φ is the maximum value of the modified Ducros sensor over the stencil with points i = 0, ...is, given by,


![Equation](images/[2026] 4eq + phase change_eq212.png)


![Equation](images/[2026] 4eq + phase change_eq213.png)


![Equation](images/[2026] 4eq + phase change_eq214.png)

For a given direction and species equation, the TENO6-A scheme is evaluated as smooth using,


![Equation](images/[2026] 4eq + phase change_eq215.png)

where λi is the same smoothness indicator described in [76]. The final cell label is determined based on the union of Eq A.3 across all species equations. If any sub-stencil for any species equation is not smooth, the Godunov approach is used for both the current cell and its immediate neighbors in that direction (not including diagonal).

Appendix B. Finite-rate Phase Change Model Forms

Appendix B.0.1. Homogeneous Relaxation Model As described in Section 1.1.2, unlike the instantaneous thermo-chemical equilibrium model (HEM), the Homogeneous Relaxation Model (HRM) defines finite evaporation rate using a model form with parameters

25

informed from experiments. The original HRM was proposed to approximate flashing experiments [7]. It is given by the form,


![Equation](images/[2026] 4eq + phase change_eq216.png)


![Equation](images/[2026] 4eq + phase change_eq217.png)

with hc l as the liquid enthalpy, hc g as the gas enthalpy, and a time-scale,


![Equation](images/[2026] 4eq + phase change_eq218.png)

where ψ was originally proposed for low-pressure experiments as,


![Equation](images/[2026] 4eq + phase change_eq219.png)


![Equation](images/[2026] 4eq + phase change_eq220.png)

and high-pressure experiments as,


![Equation](images/[2026] 4eq + phase change_eq221.png)


![Equation](images/[2026] 4eq + phase change_eq222.png)

with Pc as the pressure at the critical point. Furthermore, in the HRM ϵ is the void fraction given by 1 −ϕ where ϕ is the volume fraction of the liquid. All other parameters in Eq. B.1 are constants that have been determined by fitting the evaporation rates to experiments. For example, for low-pressure flashing flows [7] proposed, Θ0 = 6.51 × 10−4s, γ = −0.257, and θ = −2.24, and for high-pressure flashing flows [7] proposed, Θ0 = 3.84 × 10−7, γ = −0.54, and θ = −1.76.

Appendix B.0.2. Droplet Evaporation (D2 Law) For droplet dominated flows, correlations with experimental measurements can be defined to capture the D2 evaporation law. As discussed in Section 1.1.2, a popular correlation from [10] uses thin film theory to approximately capture the effects of heat transfer through a droplet throughout the evaporation process. The overall expression for the droplet mass transfer rate is,


![Equation](images/[2026] 4eq + phase change_eq223.png)

where ρg is the density of the gas mixture in the film, Dg is the binary diffusion coefficient of the gas, rs is the droplet radius, and Sh∗is a modified Sherwood number given by,


![Equation](images/[2026] 4eq + phase change_eq224.png)

with Sh0 = 2 + 0.552Re1/2Sc1/3, where Re is the droplet Reynolds number and Sc is the Schmidt number. Additionally, FM is defined as,


![Equation](images/[2026] 4eq + phase change_eq225.png)


![Equation](images/[2026] 4eq + phase change_eq226.png)

where BM = Y c l s−Y c l ∞ 1−Y c l s and Y c l s is the mass fraction of the liquid on the surface of the droplet, and Y c l ∞ is the mass fraction of the liquid far from the surface. A critical addition of this model compared to the HRM formulation is the dependence on the radius of the droplet. Although this increases the accuracy of this model for droplet evaporation, it requires a representation of the droplet sizes throughout a simulation which is not present in many solvers that use Eulerian representations for the liquid. Furthermore, this model is only applicable for droplets that are undergoing evaporation and does not generalize to all flows, including boiling, flashing, or condensation dominated flows.


## References

[1] H. Collis, D. A. Bezgin, S. Mirjalili, A. Mani, A robust four-equation model for compressible multi-phase multi-component flows satisfying interface equilibrium and phase-immiscibility conditions, Journal of Computational Physics (2026) 114827.

26

[2] M. Palacz, M. Haida, J. Smolka, A. J. Nowak, K. Banasiak, A. Hafner, Hem and hrm accuracy comparison for the simulation of co2 expansion in two-phase ejectors for supermarket refrigeration systems, Applied Thermal Engineering 115 (2017) 160–169.

[3] R. Saurel, P. Boivin, O. Le Métayer, A general formulation for cavitating, boiling and evaporating flows, Computers & Fluids 128 (2016) 53–64.

[4] A. Chiapolino, P. Boivin, R. Saurel, A simple and fast phase transition relaxation solver for compressible multicomponent two-phase flows, Computers & Fluids 150 (2017) 31–45.

[5] M. Pelanti, Arbitrary-rate relaxation techniques for the numerical modeling of compressible two-phase flows with heat and mass transfer, International Journal of Multiphase Flow 153 (2022) 104097.

[6] A. D. Demou, N. Scapin, M. Pelanti, L. Brandt, A pressure-based diffuse interface method for low-mach multiphase flows with mass transfer, Journal of Computational Physics 448 (2022) 110730.

[7] P. Downar-Zapolski, Z. Bilicki, L. Bolle, J. Franco, The non-equilibrium relaxation model for onedimensional flashing liquid flow, International journal of multiphase flow 22 (3) (1996) 473–483.

[8] J. W. Gaertner, A. Kronenburg, A. Rees, J. Sender, M. Oschwald, G. Lamanna, Numerical and experimental analysis of flashing cryogenic nitrogen, International Journal of Multiphase Flow 130 (2020) 103360.

[9] K. G. Lyras, S. Dembele, J. X. Wen, Numerical simulation of flashing jets atomisation using a unified approach, International Journal of Multiphase Flow 113 (2019) 45–58.

[10] B. Abramzon, W. A. Sirignano, Droplet vaporization model for spray combustion calculations, International journal of heat and mass transfer 32 (9) (1989) 1605–1618.

[11] H. Hertz, Ueber die verdunstung der flüssigkeiten, insbesondere des quecksilbers, im luftleeren raume, Annalen der Physik 253 (10) (1882) 177–193.

[12] M. Knudsen, Die maximale verdampfungsgeschwindigkeit des quecksilbers, Annalen der Physik 352 (13) (1915) 697–708.

[13] S. S. Sazhin, Advanced models of fuel droplet heating and evaporation, Progress in energy and combustion science 32 (2) (2006) 162–214.

[14] D. Fuster, G. Hauke, C. Dopazo, Influence of the accommodation coefficient on nonlinear bubble oscillations, The Journal of the Acoustical Society of America 128 (1) (2010) 5–10.

[15] T. Lyras, I. K. Karathanassis, N. Kyriazis, P. Koukouvinis, M. Gavaises, Modelling of liquid oxygen and nitrogen injection under flashing conditions, Applied Thermal Engineering 237 (2024) 121773.

[16] J. K. Dukowicz, A particle-fluid numerical model for liquid sprays, Journal of computational Physics 35 (2) (1980) 229–253.

[17] R. Reitz, et al., Modeling atomization processes in high-pressure vaporizing sprays, Atomisation and Spray technology 3 (4) (1987) 309–337.

[18] B. M. Devassy, C. Habchi, E. Daniel, Atomization modelling of liquid jets using a two-surface-density approach, Atomization and Sprays 25 (1).

[19] Q. Xue, S. Som, P. K. Senecal, E. Pomraning, Large eddy simulation of fuel-spray under non-reacting ic engine conditions, Atomization and Sprays 23 (10).

[20] Q. Xue, M. Battistoni, C. Powell, D. Longman, S. Quan, E. Pomraning, P. Senecal, D. Schmidt, S. Som, An eulerian cfd model and x-ray radiography for coupled nozzle flow and spray in internal combustion engines, International Journal of Multiphase Flow 70 (2015) 77–88.

27

[21] J. M. Desantes, J. M. Garcia-Oliver, J. M. Pastor, A. Pandal, A comparison of diesel sprays cfd modeling approaches: Ddm versus σ-y eulerian atomization model, Atomization and Sprays 26 (7).

[22] G. M. Magnotti, C. L. Genzale, Detailed assessment of diesel spray atomization models using visible and x-ray extinction measurements, International Journal of Multiphase Flow 97 (2017) 33–45.

[23] M. Jia, H. Pan, Y. Bian, Z. Zhang, Y. Chang, H. Liu, Calibration of the constants in the kelvin-helmholtz rayleigh-taylor (kh-rt) breakup model for diesel spray under wide conditions based on advanced data analysis techniques, Atomization and Sprays 32 (6).

[24] G. Tryggvason, R. Scardovelli, S. Zaleski, Direct numerical simulations of gas–liquid multiphase flows, Cambridge university press, 2011.

[25] M. Herrmann, On simulating primary atomization using the refined level set grid method, Atomization and Sprays 21 (4).

[26] M. A. Khanwale, K. Saurabh, M. Ishii, H. Sundar, B. Ganapathysubramanian, Breakup dynamics in primary jet atomization using mesh-and interface-refined cahn-hilliard navier-stokes, arXiv preprint arXiv:2209.13142.

[27] A. Vallet, R. Borghi, Modélisation eulerienne de l’atomisation d’un jet liquide, Comptes Rendus de l’Académie des Sciences-Series IIB-Mechanics-Physics-Astronomy 327 (10) (1999) 1015–1020.

[28] A. Burluka, R. Borghi, et al., Development of a eulerian model for the “atomization” of a liquid jet, Atomization and sprays 11 (6).

[29] R. Lebas, G. Blokkeel, P.-A. Beau, F.-X. Demoulin, Coupling vaporization model with the eulerianlagrangian spray atomization (elsa) model in diesel engine conditions, Tech. rep., SAE Technical Paper (2005).

[30] R. Lebas, T. Menard, P.-A. Beau, A. Berlemont, F.-X. Demoulin, Numerical simulation of primary break-up and atomization: Dns and modelling study, International Journal of Multiphase Flow 35 (3) (2009) 247–260.

[31] J. Anez, A. Ahmed, N. Hecht, B. Duret, J. Reveillon, F. Demoulin, Eulerian–lagrangian spray atomization model coupled with interface capturing method for diesel injectors, International Journal of Multiphase Flow 113 (2019) 325–342.

[32] J. Desantes, J. M. García-Oliver, J. Pastor, A. Pandal, E. Baldwin, D. Schmidt, Coupled/decoupled spray simulation comparison of the ecn spray a condition with the σ-y eulerian atomization model, International Journal of Multiphase Flow 80 (2016) 89–99.

[33] A. Pandal, J. M. Pastor, R. Payri, A. Kastengren, D. Duke, K. Matusik, J. S. Giraldo, C. Powell, D. Schmidt, Computational and experimental investigation of interfacial area in near-field diesel spray simulation, SAE International Journal of Fuels and Lubricants 10 (2) (2017) 423–431.

[34] J. Desantes, J. M. García-Oliver, J. Pastor, I. Olmeda, A. Pandal, B. Naud, Les eulerian diffuse-interface modeling of fuel dense sprays near-and far-field, International Journal of Multiphase Flow 127 (2020) 103272.

[35] G. Nykteri, M. Gavaises, Droplet aerobreakup under the shear-induced entrainment regime using a multiscale two-fluid approach, Physical Review Fluids 6 (8) (2021) 084304.

[36] H. Gaballa, C. Habchi, J.-C. de Hemptinne, Modeling and les of high-pressure liquid injection under evaporating and non-evaporating conditions by a real fluid model and surface density approach, International Journal of Multiphase Flow 160 (2023) 104372.

28

[37] F. Nicoud, H. B. Toda, O. Cabrit, S. Bose, J. Lee, Using singular values to build a subgrid-scale model for large eddy simulations, Physics of fluids 23 (8).

[38] C. A. Kennedy, A. Gruber, Reduced aliasing formulations of the convective terms within the navier– stokes equations for a compressible fluid, Journal of Computational Physics 227 (3) (2008) 1676–1700.

[39] Y. Kuya, S. Kawai, High-order accurate kinetic-energy and entropy preserving (keep) schemes on curvilinear grids, Journal of Computational Physics 442 (2021) 110482. doi:10.1016/J.JCP.2021.110482.

[40] S. S. Jain, P. Moin, A kinetic energy–and entropy-preserving scheme for compressible two-phase flows, Journal of Computational Physics 464 (2022) 111307. doi:10.1016/J.JCP.2022.111307.

[41] M. L. Wong, J. B. Angel, C. C. Kiris, A positivity-preserving eulerian two-phase approach with thermal relaxation for compressible flows with a liquid and gases, arXiv preprint arXiv:2208.04488.

[42] H. Collis, A robust high order framework for compressible multi-phase multi-component flows with interface regularization, phase change, and spray modeling, Stanford University, 2025.

[43] H. Collis, S. Mirjalili, A. Mani, Diffuse interface treatment in generalized curvilinear coordinates with grid-adapting interface thickness, Journal of Computational Physics 544 (2026) 114440. doi:10.1016/ j.jcp.2025.114440. URL https://www.sciencedirect.com/science/article/pii/S0021999125007223

[44] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM review 43 (1) (2001) 89–112.

[45] K. W. Thompson, Time-dependent boundary conditions for hyperbolic systems, ii, Journal of computational physics 89 (2) (1990) 439–461.

[46] T. J. Poinsot, S. Lelef, Boundary conditions for direct simulations of compressible viscous flows, Journal of computational physics 101 (1) (1992) 104–129.

[47] N. Okong’o, J. Bellan, Consistent boundary conditions for multicomponent real gas mixtures based on characteristic waves, Journal of Computational Physics 176 (2) (2002) 330–344.

[48] B. Péden, J. Carmona, P. Boivin, T. Schmitt, B. Cuenot, N. Odier, Numerical assessment of diffuseinterface method for air-assisted liquid sheet simulation, Computers & Fluids 266 (2023) 106022.

[49] M. R. Baer, J. W. Nunziato, A two-phase mixture theory for the deflagration-to-detonation transition (ddt) in reactive granular materials, International journal of multiphase flow 12 (6) (1986) 861–889.

[50] P. Linstrom, Nist chemistry webbook, nist standard reference database 69, (No Title) (1997) 20899.

[51] X. Deng, P. Boivin, Diffuse interface modelling of reactive multi-phase flows applied to a sub-critical cryogenic jet, Applied Mathematical Modelling 84 (2020) 405–424.

[52] J. Chesnel, J. Reveillon, T. Menard, F.-X. Demoulin, Large eddy simulation of liquid jet atomization, Atomization and Sprays 21 (9).

[53] S. Mirjalili, M. Khanwale, A. Mani, Consistent modeling of scalar transport in multiphase flows using conservative phase field methods, Proceedings of the Summer Program 2022, Stanford University Center for Turbulence Research.

[54] B. Duret, J. Reveillon, T. Menard, F. Demoulin, Improving primary atomization modeling through dns of two-phase flows, International Journal of Multiphase Flow 55 (2013) 130–137.

[55] A. Yoshizawa, K. Horiuti, A statistically-derived subgrid-scale kinetic energy model for the large-eddy simulation of turbulent flows, Journal of the Physical Society of Japan 54 (8) (1985) 2834–2839.

29

[56] OpenCFD Ltd., OpenFOAM v2206 API Guide: Foam::LESModels::WALE Class, [Online; accessed 14-August-2025] (2022). URL https://www.openfoam.com/documentation/guides/v2206/api/classFoam_1_1LESModels_ 1_1WALE.html

[57] J. W. Gärtner, A. Kronenburg, A novel elsa model for flash evaporation, International Journal of Multiphase Flow 174 (2024) 104784.

[58] A. Prosperetti, Vapor bubbles, Annual review of fluid mechanics 49 (1) (2017) 221–248.

[59] M. Pilch, C. Erdman, Use of breakup time data and velocity history data to predict the maximum size of stable fragments for acceleration-induced breakup of a liquid drop, International journal of multiphase flow 13 (6) (1987) 741–757.

[60] V. Cleary, P. Bowen, H. Witlox, Flashing liquid jets and two-phase droplet dispersion: I. experiments for derivation of droplet atomisation correlations, Journal of hazardous materials 142 (3) (2007) 786–796.

[61] Engine Combustion Network (ECN), ECN, https://ecn.sandia.gov, an international collaboration among experimental and computational researchers in engine combustion, maintained by the Applied Combustion Department, Sandia National Laboratories (2025).

[62] P. Boivin, M. Cannac, O. Le Métayer, A thermodynamic closure for the simulation of multiphase reactive flows, International Journal of Thermal Sciences 137 (2019) 640–649.

[63] Engine Combustion Network (ECN), Spray a geometry, https://ecn.sandia.gov/ diesel-spray-combustion/target-condition/spray-a-nozzle-geometry (2022).

[64] CMT, Virtual injection rate generator, https://www.cmt.upv.es/#/ecn/download/ InjectionRateGenerator/InjectionRateGenerator.

[65] R. Borges, M. Carmona, B. Costa, W. S. Don, An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws, Journal of computational physics 227 (6) (2008) 3191–3211.

[66] A. Kastengren, F. Tilocco, D. Duke, C. Powell, S. Moon, X. Zhang, et al., Time-resolved x-ray radiography of diesel injectors from the engine combustion network, ICLASS Paper 1369.

[67] Engine Combustion Network (ECN), Near-nozzle mixture derived from x-ray radiography, https:// ecn.sandia.gov/rad675/ (2022).

[68] I. Karathanassis, P. Koukouvinis, M. Gavaises, Comparative evaluation of phase-change mechanisms for the prediction of flashing flows, International Journal of Multiphase Flow 95 (2017) 257–270.

[69] J. Manin, M. Bardi, L. M. Pickett, Sp2-4 evaluation of the liquid length via diffused back-illumination imaging in vaporizing diesel sprays (sp: Spray and spray combustion, general session papers), in: The Proceedings of the International symposium on diagnostics and modeling of combustion in internal combustion engines 2012.8, The Japan Society of Mechanical Engineers, 2012, pp. 665–673.

[70] Engine Combustion Network (ECN), Measurements of the liquid length for nozzle 210675 using diffused back-illumination (dbi) at spray a and other conditions, https://ecn.sandia.gov/data/dbi675/ (2022).

[71] M. Bardi, R. Payri, L. M. C. Malbec, G. Bruneaux, L. M. Pickett, J. Manin, T. Bazyn, C. L. Genzale, Engine combustion network: comparison of spray development, vaporization, and combustion in different combustion vessels, Atomization and Sprays 22 (10).

[72] Engine Combustion Network (ECN), Diesel data search, https://ecn.sandia.gov/ecn-data-search

(2022).

30

[73] L. M. Pickett, C. L. Genzale, J. Manin, Uncertainty quantification for liquid penetration of evaporating sprays at diesel-like conditions, Atomization and Sprays 25 (5).

[74] Engine Combustion Network (ECN), Modeling standards and recommendations, https://ecn.sandia. gov/diesel-spray-combustion/computational-method/modeling-standards/ (2022).

[75] E. F. Toro, Riemann solvers and numerical methods for fluid dynamics: a practical introduction, Springer Science & Business Media, 2013.

[76] L. Fu, X. Y. Hu, N. A. Adams, A new class of adaptive high-order targeted eno schemes for hyperbolic conservation laws, Journal of Computational Physics 374 (2018) 724–751.

31

