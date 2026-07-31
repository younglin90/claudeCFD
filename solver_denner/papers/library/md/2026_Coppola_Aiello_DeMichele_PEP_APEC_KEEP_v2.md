
# PRESSURE-EQUILIBRIUM-PRESERVING AND FULLY CONSERVATIVE DISCRETIZATION OF COMPRESSIBLE FLOW EQUATIONS FOR REAL AND THERMALLY PERFECT GASES

A PREPRINT

Gennaro Coppola Dipartimento di Ingegneria Industriale Università di Napoli “Federico II” Napoli, Italy gcoppola@unina.it

Alessandro Aiello Dipartimento di Ingegneria Industriale Università di Napoli “Federico II” Napoli, Italy alessandro.aiello@unina.it

Carlo De Michele Gran Sasso Science Institute (GSSI) L’Aquila, Italy carlo.demichele@gssi.it

May 4, 2026


### ABSTRACT

Numerical simulations of compressible real-fluid flows are notoriously plagued by spurious pressure oscillations arising in regions of abrupt flow variations. As a possible remedy, several numerical formulations enforce the pressure equilibrium condition for the compressible Euler equations, typically at the cost of spoiling the correct conservation of total energy or by overspecifying the thermodynamical variables. This study proposes for the first time a numerical discretization procedure which is able to discretely preserve the full conservation of the linear invariants (mass, momentum and total energy) and to exactly enforce the pressure equilibrium condition. The method also preserves the conservation of kinetic energy by convection, and is based on the specification of nonlinear numerical fluxes for mass and internal energy which depend on the details of the equation of state. Both thermally perfect and real gases with an arbitrary equation of state are considered, and a simplified approximate pressure equilibrium preserving formulation with excellent performances is also proposed. The effectiveness of the novel formulations is assessed through a series of numerical simulations in supercritical and transcritical conditions with some of the most popular cubic equations of state.

Keywords Compressible flow · Pressure-equilibrium-preserving methods · Kinetic-energy-preserving methods · Real gases · Thermally perfect gases


### 1 Introduction

The accurate numerical simulation of compressible flows for real and thermally perfect gases is of great importance in a wide array of modern engineering and scientific disciplines. From the design of high-speed aerospace vehicles and advanced propulsion systems to the analysis of supercritical fluids in power generation and high-pressure turbomachinery, the assumption of a calorically perfect gas frequently breaks down [1, 2, 3]. In these extreme thermodynamic regimes, the fluid behavior deviates significantly from the ideal gas law, requiring the use of complex, non-linear Equations of State (EoS) to model phenomena such as variable specific heats, dense gas effects, and phase transitions.

However, the robust discretization of the compressible Euler equations coupled with a generic real gas EoS presents profound numerical challenges. Foremost among these is the generation of spurious, non-physical oscillations across


# arXiv:2605.03617v2  [physics.flu-dyn]  26 May 2026

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

contact discontinuities [4]. This phenomenon is fundamentally analogous to the well-documented difficulties encountered in the simulation of multicomponent flows [5]. Just as the abrupt variation of the specific heat ratio across a multi-fluid interface triggers severe numerical artifacts, the highly non-linear dependence of pressure on density and internal energy in a real gas EoS induces similar unphysical fluctuations. If left unmitigated, these spurious oscillations in pressure and velocity can propagate rapidly, leading to non-physical thermodynamic states and ultimately resulting in the catastrophic failure of the simulation.

The root of this issue lies in the extreme difficulty of simultaneously satisfying two fundamental numerical requirements: preserving local pressure equilibrium and maintaining strict, discrete conservation of total energy. When fully conservative numerical schemes are applied to real gases, the primary variables (density, momentum and total energy) are updated and exactly conserved. However, the subsequent non-linear inversion of the EoS, required to recover the pressure from the updated density and internal energy, frequently fails to maintain pressure equilibrium across moving contact discontinuities. Conversely, many existing strategies attempt to enforce a Pressure-Equilibrium-Preserving (PEP) condition by modifying the energy equation or employing primitive-variable formulations (i.e., discretizing the pressure equation directly in place of the total energy one). While these successfully suppress spurious oscillations, they inherently sacrifice exact discrete conservation of total energy. This loss of full conservation can be highly detrimental, as, in the presence of strong shocks, the scheme may fail to converge to the correct weak solution [5, 6].

To address the spurious oscillations, a variety of numerical strategies have been proposed over the past decades. Early efforts, predominantly rooted in the study of multicomponent flows, focused on quasi-conservative formulations. As demonstrated by Karni [7], discretizing the Euler equations in primitive variables avoids the energy-to-pressure EoS inversion, successfully maintaining local pressure equilibrium. To recover partial conservation, Abgrall [5] and later Shyue [8] introduced quasi-conservative schemes wherein the standard conservative fluid equations are solved alongside non-conservative advection equations for the thermodynamic parameters (e.g., the specific heat ratio). By carefully discretizing these auxiliary equations, these methods achieve the PEP property. However, the explicit inclusion of non-conservative terms inherently sacrifices exact discrete conservation of total energy, leading to incorrect shock propagation speeds and thermodynamic anomalies across strong discontinuities. Similar problems were encountered by evolving the pressure equation for transcritical and supercritical fluids [4, 9], and even the use of enthalpy and internal energy has been attempted [10].

Recognizing the deficiencies of globally non-conservative models, double-flux methods have been introduced to bridge the gap between interface stability and shock fidelity. Originally formalized by Abgrall and Karni [11], this approach was later adapted for real-gas and transcritical flows by Ma et al. [12]. The double-flux strategy employs standard, fully conservative fluxes in the bulk of the flow, but locally switches to a thermodynamically “frozen", non-conservative auxiliary flux at sharp density gradients to enforce pressure equilibrium. While highly effective at mitigating unphysical waves in practice, double-flux methods still rely on local non-conservation. Furthermore, they require complex and empirical sensor functions to blend the two flux formulations, making them computationally rigid and highly sensitive to tuning parameters when applied to arbitrary, highly non-linear real gases.

To overcome these limitations, efforts were made to modify numerical fluxes directly within a fully conservative framework. For multicomponent flows, Fujiwara et al. [13] sought to address this by deriving equilibrium conditions to construct specialized numerical fluxes without relying on auxiliary transport equations. Terashima et al. [14] extended this approach to the complex thermodynamic regimes of real gases, developing consistent numerical fluxes maintaining exact conservation of the primary variables. However, the final formulation yields only an approximately PEP scheme, with a fixed spatial order of accuracy. As a result, residual pressure oscillations can still manifest when simulating extremely severe thermodynamic gradients or phase transitions.

An alternative philosophy has sought to guarantee exact pressure equilibrium and numerical robustness by coupling PEP conditions with Kinetic-Energy-Preserving (KEP) schemes. The KEP framework was originally developed to prevent non-linear aliasing errors in standard compressible turbulence simulations without relying on numerical dissipation [15, 16, 17]. For calorically perfect ideal gases, researchers have successfully derived discretizations that combine KEP properties with exact pressure equilibrium [18, 19, 20, 21, 22]. For real-fluid simulations, a recent work by Bernades et al. [23] proposed a specifically designed KEP and PEP scheme. Yet, to strictly maintain both pressure equilibrium and kinetic energy preservation across complex thermodynamic states, they were forced to abandon exact total energy conservation, opting instead to solve a non-conservative pressure evolution equation. Consequently, a numerical methodology that simultaneously guarantees full conservation—particularly of total energy—and exact pressure equilibrium for real and thermally perfect gases remains a missing link in the literature.

To address this critical gap, the present study introduces a novel numerical framework that, for the first time, achieves a fully conservative, kinetic-energy-preserving, and exactly pressure-equilibrium-preserving discretization of the compressible Euler equations for thermally perfect and real gases, with an arbitrary EoS. Our approach demonstrates that it is mathematically and practically possible to retain the exact discrete evolution of the total energy equation without

2

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

triggering spurious interfacial oscillations, offering a robust and thermodynamically consistent foundation for simulating extreme real-gas phenomena. To overcome the intrinsic difficulties associated with simulations in transcritical conditions, an approximate formulation is also proposed, based on a slight modification of the set of exact fluxes. By retaining excellent PEP properties, with unrivaled performances among existing formulations, the set of approximate fluxes shows robust behavior also in extreme conditions.

The remainder of this paper is organized as follows. Section 2 establishes the problem formulation, detailing the continuous compressible Euler equations and the underlying discretization framework. In Section 3, we present the core methodology, rigorously deriving the fully conservative numerical fluxes and proving their KEP and PEP properties at the discrete level. This generic formulation is subsequently particularized for specific thermodynamic models: Section 4 addresses thermally perfect gases, while Section 5 extends the approach to real gases, with a specific focus on the van der Waals and Peng–Robinson equations of state. Section 6 presents a suite of numerical experiments—including a standard density wave test and a multidimensional mixing case—designed to validate both the PEP property and exact discrete conservation. Finally, Section 7 summarizes the key findings and offers concluding remarks.


### 2 Problem formulation

2.1 Governing equations

The compressible Euler equations, which express the conservation of mass, momentum and total energy for compressible inviscid flows, can be written as:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq001.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq002.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq003.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq004.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq005.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq006.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq007.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq008.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq009.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq010.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq011.png)

where ρ is the density, uα is the Cartesian velocity component along xα, p is the pressure and E the total energy per unit mass, which is the sum of the internal (e) and kinetic (κ = uαuα/2) energies per unit mass: E = e + κ. We will assume the convention that Greek subscripts, such as α or β, refer to the components of Cartesian vectors, whereas Latin subscripts, such as i or j appearing in the subsequent sections of this paper, are used to denote the values of the discretized variable on a nodal point. Unless otherwise stated, the summation convention over repeated Greek indices is assumed. The system (1)–(3) is closed by an equation of state, relating p, ρ and the temperature T, and by specifying the dependence of internal energy on temperature and density, which is typically provided through suitable departure functions.

In what follows, we will also need to consider the induced balance equations for additional quantities related to the primary variables ρ, ρuβ, and ρE. These equations are easily derived by combining Eqs. (1)–(3) and by applying the chain and product rules of differentiation, with respect to both temporal and spatial variables. In particular, we will use the evolution equations for the kinetic and internal energies:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq012.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq013.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq014.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq015.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq016.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq017.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq018.png)

whose sum returns the total energy equation (3). Moreover, to discuss the PEP property, which is the main subject of the present paper, we will use the evolution equations for the velocity components uβ, and the pressure p, which can be written as ∂uβ


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq019.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq020.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq021.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq022.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq023.png)

where c = � (∂p/∂ρ)s is the speed of sound, s being the specific entropy. Eqs. (6) and (7) show that if at a given time instant a spatially constant state is assumed for both uβ and p, all the spatial derivatives at the right-hand sides vanish. This also causes the temporal derivatives at the left-hand side to vanish, meaning that the constant distribution

3

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

of pressure and velocity remains constant also in time. This is the Pressure Equilibrium property for the compressible Euler equations, which allows for the existence of density wave solutions traveling at constant velocity in a uniform pressure field.

2.2 Discretization setting

Our focus is on the discretization of the spatial terms in Eqs. (1)–(3), for which we always use central (non-dissipative) schemes. To isolate spatial errors, we assume in the theoretical analysis that the manipulation of time derivatives can be performed as in the continuous case. This type of semidiscretized analysis leads to a system of Ordinary Differential Equations (ODE) whose temporal integration is performed by using standard solvers. In this paper, the theory will be developed for the one-dimensional version of Eq. (1)–(3), which is obtained by omitting the Greek subscripts. The extension to the general three-dimensional case is straightforward and involves simply including the analogous spatial terms along the additional Cartesian components. The analysis is developed by referring to a Finite Difference (FD) treatment of the convective terms in Eqs. (1)–(3) on a uniform mesh {xi} of width h = xi+1 −xi, although most of our results could also be applied in other frameworks, such as Finite Volume or Finite Element methods, and can be generalized to nonuniform or curvilinear meshes, following standard approaches [24, 25]. Finally, we focus on the discretization of the various spatial operators at internal points, neglecting the effects of boundary conditions, which could be studied with ad hoc methods.

Although we work in a FD framework, we will design the discretization of the convective spatial terms in Eqs. (1)–(3) by specifying the numerical fluxes, i.e. by assuming a locally conservative discretization in the form


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq024.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq025.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq026.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq027.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq028.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq029.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq030.png)

where ϕ is a generic transported quantity per unit mass (1, u or E in Eq. (1), (2) or (3), respectively). The numerical flux F i+ 1

2 ρϕ is a consistent approximation of ρuϕ at xi+1/2 = xi + h/2, and is a function of the sets of nodal values ρi, ui and ϕi. Note that, even if the right-hand sides of Eq. (1)–(3) can be expressed as the divergence of a single term including convective and pressure mechanisms, Eq. (8) refers to the specification of the numerical fluxes for the convective contribution only (the first terms at the r.h.s. of Eqs. (1)–(3)). The terms involving the pressure in Eq. (2) and (3) are discretized by using standard central derivative formulas, as specified in Section 2.3. Eq. (8) reproduces the divergence structure of the convective terms in Eqs. (1)–(3) and implies also global conservation of the invariant ρϕ by virtue of the telescoping property.

To simplify the notation, we will make use of the difference and average operators


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq031.png)

which allows to write the central difference as φi+1 −φi−1 = 2δ −φ i and to express Eq. (8) as


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq032.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq033.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq034.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq035.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq036.png)

In most cases, we will omit the apex when referring to the flux evaluated at xi+1/2: Fρϕ = F i+ 1

2 ρϕ , and, when no ambiguity can arise, we also omit the suffix in the expressions of the difference or of the average of a generic quantity: ϕ = ϕ i, δ −ϕ = δ −ϕi.

Within the described setting, the spatial discretization of the r.h.s. of Eqs. (1)–(3) is completely determined by the specification of the convective fluxes Fρ, Fρu, FρE and by the discretization details of the pressure terms ∂p/∂x and ∂up/∂x in Eqs. (2) and (3), respectively. The derivation exposed in the next sections considers second-order approximations, which implies that the numerical flux Fρϕ is in general a two-point interpolation of (ρuϕ)i+1/2, and is a function of ρ, u and ϕ at nodes xi and xi+1 only. The extension to higher orders is obtained by using standard arguments and is detailed in several references [26, 27, 28, 21, 29].

2.3 Kinetic Energy Preserving formulation

A first constraint on the possible choice for the numerical fluxes is given by the requirement that the formulation satisfies the Kinetic Energy Preserving (KEP) property [15, 16, 30, 31], which amounts to the requirement that the discrete induced evolution equation for the kinetic energy has a convective term that is locally and globally conservative, as for the continuous equation (cf. Eq. (4)). It is well known ([16, 32, 33]) that, for second order fluxes, the KEP property is satisfied when the convective fluxes for mass and momentum are linked by the simple relation


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq037.png)

4

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

In this case, kinetic energy evolves according to a discrete equation analogous to Eq. (4), in which the convective term can be cast as a difference of numerical fluxes, with flux [33, 34]


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq038.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq039.png)

The requirement that the discrete total energy evolves coherently with the induced evolution of kinetic energy suggests designing the convective numerical flux for total energy FρE as the sum of the kinetic-energy flux given by Eq. (9) and of a convective flux for internal energy Fρe. The conservative pressure term is correspondingly built as the sum of the induced discretized pressure term in Eq. (4) (which belongs to the discrete pressure term in the momentum equation (2)) and of a consistent discretization of the pressure term in Eq. (5). Assuming standard second-order central schemes for these terms, one has


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq040.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq041.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq042.png)

which gives the correct discretization of the pressure term in the total energy equation as consistent with the advective form of the derivative of the product: ∂up


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq043.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq044.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq045.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq046.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq047.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq048.png)

where up = (uipi+1 + piui+1) /2 is the product mean [31].

In conclusion, a locally conservative and KEP discretization is characterized by the set of total fluxes (i.e. including both convective and pressure contributions) given by


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq049.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq050.png)

where the mass and internal energy fluxes Fρ and Fρe are still unspecified, except for the obvious requirement that they are consistent approximations of mass and internal energy fluxes. The set of fluxes reported in Eq. (11) defines a formulation which is locally conservative of linear invariants ρ, ρu and ρE and preserves the (local and global) conservation of kinetic energy by convection. It embodies many KEP formulations available in the literature that have gained popularity in recent years. As examples, the simple choice Fρ = ρ u and Fρe = ρ u e gives the KEEP scheme [35] which, even if it does not have exact additional structural properties, has been widely used because of its good performances for ideal gases. The choice Fρ = ρ u and Fρe = ρ u eH (where eH = 2eiei+1/(ei + ei+1) is the harmonic mean) gives the zeroth-order AEC scheme, developed in [20], whereas Fρ = ρ u and Fρe = ρe u give the KEEPPE scheme [18], these last two formulations being PEP for ideal gases. Recently, the present authors also determined mass and internal energy fluxes such that the formulation in Eq. (11) is Entropy Conservative (EC), i.e. it is able to discretely preserve the correct balance of entropy, for the case of real gases with an arbitrary equation of state [29] and for thermally perfect gases [36]. In the next section, we derive an expression for Fρ and Fρe which gives a formulation able to satisfy the PEP condition for thermally perfect and real gases with an arbitrary EoS.


### 3 PEP formulation

In this section, we analyze the general problem of enforcing the PEP condition for the system of compressible Euler equations, without making any assumptions about the equation of state. In Sections 4 and 5 we explicitly work out the particular cases of a thermally perfect gas and of various thermodynamic models for real gases.

3.1 Theoretical derivation of exact PEP fluxes

In order to discretely enforce the pressure equilibrium property, it is necessary to consider the discrete evolution equations for velocity components and pressure, which are the induced discrete counterparts of Eqs. (6) and (7). Among these, the pressure evolution equation is the more challenging one, due to its explicit dependence on the EoS. We therefore begin with the discrete velocity equation, which is independent of the EoS. General conditions for the discrete enforcement of the velocity equilibrium have previously been derived for the case of calorically perfect gases [21, 19], where it was shown that the KEP condition is sufficient to enforce it. This result still applies but, for completeness, we briefly revisit the derivation here.

First of all, we explicitly write the semidiscrete evolution equations for mass and momentum as obtained by discretizing Eqs. (1) and (2) according to the prescriptions illustrated in the previous section: dρi


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq051.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq052.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq053.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq054.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq055.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq056.png)

5

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

The discrete evolution equation for the velocity can be now obtained by using the expression of the time derivative of the velocity u in terms of the time derivatives of ρ and ρu


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq057.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq058.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq059.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq060.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq061.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq062.png)

From this relation, substituting Eqs. (12) and (13), one obtains


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq063.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq064.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq065.png)

In the case of spatially constant pressure pi = P and velocity ui = U, the pressure contribution at the right-hand side vanishes, and one is left with


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq066.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq067.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq068.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq069.png)

where we denoted with ˆFρ the form assumed by Fρ for uniform velocity and pressure fields. Eq. (16) shows that for the KEP discretization adopted, a uniform spatial distribution of pressure and velocity always induces a zero time derivative for ui, which implies that the velocity field remains uniform as time evolves.

Moving on to the discrete evolution of pressure, we start by considering the internal energy per unit volume ρe and use an equation of state in the form ρe = ρe(ρ, p). By taking the time derivative of this relation, we get


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq070.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq071.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq072.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq073.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq074.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq075.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq076.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq077.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq078.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq079.png)

Substituting the temporal derivatives with the right-hand sides of the mass and internal energy equations (1) and (5), we obtain ∂ρue


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq080.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq081.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq082.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq083.png)

where we defined


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq084.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq085.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq086.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq087.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq088.png)

Now we assume to discretize the convective terms in the mass and internal energy equations conservatively, i.e. as the difference of numerical fluxes δ −F/h, and to use the simple central discretization for the pressure term in the internal energy equation given in Eq. (10). This gives


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq089.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq090.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq091.png)

Assuming again uniform spatial distributions of pressure and velocity, one is left with


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq092.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq093.png)

where, as usual, ˆFρe and ˆFρ denote the form assumed by the numerical fluxes for uniform velocity and pressure fields. Eq. (17) reveals that to discretely maintain the uniform pressure distribution (i.e. to enforce the condition dpi/dt = 0), the mass and internal energy fluxes have to satisfy the constraint


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq094.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq095.png)

A condition analogous to that in Eq. (18) has already been analyzed in [13] for the case of a multicomponent mixture of real gases and has been the starting point to obtain approximate PEP formulations in [14]. An exact enforcement of Eq. (18) seems problematic at first sight, since it requires the specification of the numerical fluxes Fρ and Fρe such that the product between the exact difference δ−ˆFρ and an arbitrary function ˆα(ρ) can be expressed as the difference of numerical fluxes δ−ˆFρe. In Eq. (18), the function ˆα(ρ) is fixed by the thermodynamic model, whereas ˆFρ and ˆFρe can be chosen arbitrarily, provided that they are consistent approximations of the mass and internal energy fluxes.

To solve the problem expressed by Eq. (18) we proceed inspired by the steps of the theory of Tadmor [37, 38] and observe that ˆαi δ−ˆFρ is an exact difference if (and only if) ˆFρ δ+ˆαi is, since the sum aiδ−bi + biδ+ai (which is a consistent discretization of the advective form of the derivative of the product ab) admits the decomposition as a difference of fluxes aiδ−bi + biδ+ai = δ−(ai+1bi). This allows us to rewrite Eq. (18) in the equivalent form


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq096.png)

6

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

and we are now faced with the problem of finding a numerical flux Fρ such that ˆFρδ +ˆαi is an exact difference. We proceed now by observing that the flux Fρ should be a consistent approximation of the analytical flux fρ(ρ, u) = ρu at xi+1/2. We assume that the function ˆα(ρ) is (at least locally) invertible, in such a way that the inverse function ρ(ˆα) can be considered, and the relation ρ – ˆα is a local one-to-one mapping. This confers to the variable ˆα the role of an entropy variable, in the terminology of the theory of Tadmor. In this case we can express the function ˆfρ(ρ) = fρ(ρ, U) through the new variable ˆα by defining the function ˆgρ(ˆα) = ˆfρ(ρ(ˆα)), which is approximated by the numerical flux ˆGρ(ˆαi, ˆαi+1) = ˆFρ(ρ(ˆαi), ρ(ˆαi+1)).

To express now ˆGρδ +ˆα as an exact difference, we make use of the primitive function ψ(ˆα) defined by


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq097.png)

Using the Integral Mean Value Theorem, we can write


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq098.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq099.png)

where ˆα∗is a unknown value between ˆαi and ˆαi+1 and the value ˆgρ(ˆα∗) is a second-order approximation of ˆgρ(ˆαi+ 1

2 ).

Eq. (19) suggests the adoption of the value ˆGρ = ˆgρ(ˆα∗), i.e.


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq100.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq101.png)

which, expressed in the variables ρi gives:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq102.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq103.png)

where ψ(ρ) = ψ(ˆα(ρ)). This gives:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq104.png)

where we used δ +ψi = δ −ψi+1. Eq. (21) reduces to Eq. (18) with the choice ˆFρe = ˆαi+1 ˆFρ −ψi+1. The form of ˆFρe can be made more symmetric by observing that ˆαi+1 = ˆα i + δ+ˆαi/2 which implies


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq105.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq106.png)

2 eventually giving

ˆFρe = ˆα i ˆFρ −ψ i. (22) Equations (20) and (22) satisfy Eq. (18), and constitute the basis to build the general fluxes Fρ and Fρe enforcing the PEP property.

To show how the function ψ and the fluxes in Eqs. (20) and (22) can be practically calculated, let us define the function


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq107.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq108.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq109.png)

which is related to α by the equation α = e + ρλ. By using fρ(ρ, u) = ρu the function ψ(ρ) can be calculated as


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq110.png)

where primes denote differentiation and, as usual, ˆλ(ρ) = λ(ρ, U). Eq. (24) allows us to write the fluxes satisfying Eq. (18) in the explicit form


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq111.png)

The final form of Fρ and Fρe can now be obtained in several ways, all reducing to Eq (25) for uniform velocity. In what follows we use the simple extension of Eq. (25) obtained by adopting the arithmetic mean for u, leading to:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq112.png)

where we defined the average ρ λ = δ+(ρ2 i λi)/δ+αi.

7

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

3.2 Treatment of the singularity and approximate PEP formulation

Equations (26) and (11) furnish the final formulation satisfying the KEP and PEP properties for an arbitrary EoS, whose details are embodied in the functions α(ρ, p) and λ(ρ, p). As it is typical in these cases, the nonlinear average ρλ in the mass flux is potentially singular for uniform distributions of α. A similar situation occurs, for example, in the EC and PEP (for ideal gases) flux by Ranocha [39], which employs the logarithmic mean (ϕ log = δ +ϕ/δ + log ϕ), and in the EC flux for real gases recently developed in [29], which also uses a potentially singular flux for uniform distributions of temperature. This phenomenon, which is fundamentally linked to the definition of the average through the use of the integral mean value theorem [40], can cause severe limitations in the applications, especially when large regions of uniform distributions of temperature or density are expected to occur. To avoid the singularity, suitable fixes can be devised, typically implemented by locally reverting to non-singular numerical fluxes when the denominators of the singular averages (δ +αi in our case) fall under a specific tolerance. These non-singular schemes can be either a suitable Taylor expansion of the original mean, when possible, as in the case of the logarithmic mean used in entropy conservative methods for ideal gases ([41, 39, 20, 22]) or standard non-singular schemes chosen among those with good performances, to minimize errors, as in [29]. Recently, a more advanced technique based on the theory of discrete gradient operators has been also used [42].

In the present case, since a full series expansion of the singular mean ρλ appearing in the mass flux Fρ seems cumbersome for an arbitrary EoS, we choose to use the simple modification of the flux Fρ in Eq. (26) obtained by adopting the arithmetic mean ρ in place of ρλ, i.e. by using the fluxes


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq113.png)

The set of fluxes given by Eqs. (11) and (27) actually shows excellent performances on highly challenging tests, as reported in Section 6. These results tempted us to use it not only as a fix for the exact PEP formulation given by Eq. (26), but also as an approximate PEP formulation which can be used during the whole simulation, irrespective of the local value of δ +αi. Similar very good performances have also been observed by using other classical algebraic means for the density in the mass flux (e.g. the harmonic or geometric means), suggesting that the form of the interpolation for the density in the mass flux has only marginal effects on the enforcement of the PEP property. For the sake of simplicity, we adopt here the arithmetic mean, leaving a thorough analysis of this subject as a possible future work. In conclusion, although only approximately PEP, the excellent performances of the formulation (11)–(27) induce us to offer it as a sufficiently simple formulation which could be used in place of Eqs. (11)–(26). We will refer to the Exact PEP formulation (EPEP-RG) for the scheme given by Eqs. (11)–(26), and to the Approximate PEP formulation (APEP-RG) for that given by Eqs. (11)–(27).

As a final comment, we observe that in the case of a calorically perfect gas, for which ρe = p/(γ −1), the function α is identically zero, and the whole derivation exposed in Section 3.1, that led to Eq. (26), breaks down. However, even if the mass flux in Eq. (26) remains undefined, the internal energy flux can be safely determined, and turns out to reduce to Fρe = ρe u , which is the internal energy flux of the formulation KEEPPE by Shima et al. [18]. Hence, the APEP-RG formulation defined by the fluxes in Eq. (27) nicely reduces to the KEEPPE formulation in the case of a calorically perfect gas.


### 4 Thermally perfect gases

In this section, we work out the particular case of a thermally perfect gas model, for which the usual perfect-gas EoS is assumed: p = ρRT, where R is the gas constant and T is the absolute temperature. The internal energy depends on temperature through temperature-variable isochoric specific heat capacity cv(T), which implies


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq114.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq115.png)

and the ref" subscript indicates some reference condition. We use a polynomial-based approach for the modeling of cv(T), [43], by which the isochoric specific heat is expressed using temperature-based polynomial fittings:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq116.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq117.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq118.png)

This functional dependence is widely used when thermal equilibrium is assumed [43], and is also easily tractable from an analytical point of view. We will detail the derivation for the general formulation with arbitrary N, although only

8

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

5, 7 or 9 coefficients are typically used to experimentally fit the gas behaviour [44, 45]. By substituting Eq. (29) into Eq. (28) and using the perfect-gas EoS one has


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq119.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq120.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq121.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq122.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq123.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq124.png)

with Ak = kck/(k + 1) and T(ρ, p) = p/ρR. The final form of the PEP fluxes corresponding with the fluxes in Eq. (26) for a thermally perfect gas is


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq125.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq126.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq127.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq128.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq129.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq130.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq131.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq132.png)

As for the general form of the fluxes in Eq. (26), the mass flux for thermally perfect gases in Eq. (31) is potentially singular. However, the particular polynomial form assumed for the cv(T) allows one to derive a formulation that is singularity-free in all conditions, without the need for a fix. To obtain this formulation, we need to derive the general form of the mass flux by starting from the PEP condition in Eq. (25), which, using Eq. (30), can be written, for thermally perfect gases, as


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq133.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq134.png)

where we used the ideal gas equation of state particularized to the case of uniform pressure: ρ = P/RT. Using now the general identity


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq135.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq136.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq137.png)

one finally gets


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq138.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq139.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq140.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq141.png)

for which the denominator does not vanish. The final form of the flux can now be obtained by assuming an arithmetic interpolation for pressure and velocity:


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq142.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq143.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq144.png)

where Sk i = �k−1 j=0 T k−1−j i T j i+1. The flux can be calculated more efficiently by observing the identity Sk+1 i = TiSk i + T k i+1. The mass flux in Eq. (32), together with the internal energy flux in Eq. (31), is a more efficient PEP formulation for thermally perfect gases, and is the one adopted in the numerical tests in Section 6. Note that in practical applications, the polynomial fitting for the cv in Eq. (29) can also include terms with negative exponents. The treatment can be easily generalized to this case as in [36], and the fluxes in Eqs. (31) and (32) are consequently adapted.


### 5 Real gases

For real gases, we adopt the usual representation of the internal energy by means of departure functions, whose expressions are given, for example, in [46] assuming the form


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq145.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq146.png)

where eTP is the thermally perfect contribution given in Eq. (28) and De is the suitable departure function for internal energy [46]. In general, we are denoting with Dϕ the departure function for the thermodynamic quantity ϕ, which is defined by Dϕ = ϕ −ϕTP and ϕTP is the thermally perfect contribution.

Standard manipulations give


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq147.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq148.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq149.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq150.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq151.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq152.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq153.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq154.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq155.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq156.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq157.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq158.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq159.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq160.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq161.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq162.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq163.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq164.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq165.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq166.png)

9

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

where T = T(ρ, p) is the explicit form of the equation of state. Gathering common terms and noting that, by definition, (∂De/∂T)ρ = Dcv(ρ, T), yields


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq167.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq168.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq169.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq170.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq171.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq172.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq173.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq174.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq175.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq176.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq177.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq178.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq179.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq180.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq181.png)

Finally, for the evaluation of the speed of sound c, required by the use of the pressure evolution equation in the main Euler system, we will always use the standard expression (particularized for each equation of state)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq182.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq183.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq184.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq185.png)

βs, βT being the isentropic and isothermal compressibility, respectively.

In the following sections, specializations for the van der Waals and Peng–Robinson models of Eq. (35) will be derived in non-dimensional form.

5.1 Van der Waals model

The van der Waals model is taken into account due to its simplicity as a first simple real-gas correction to the ideal gas equation of state. The equation of state for pressure reads


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq186.png)

with a∗= (27/64)(R∗T ∗ c )2/p∗ c and b∗= (1/8)R∗T ∗ c /p∗ c, with the superscript ∗indicating dimensional values and the suffix ‘c’ referring to critical conditions. Internal energy is given by


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq187.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq188.png)

with De = −aρ. With such definitions, Eq. (35) becomes

λ(ρ, p) = � cv(T) � a −p


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq189.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq190.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq191.png)

5.2 Peng–Robinson model

The Peng–Robinson model is considered because of its widespread application in high-pressure/low-temperature flows, as it overcomes the intrinsic instability of the van der Waals model near the critical region. The equation of state for pressure is


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq192.png)

with


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq193.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq194.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq195.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq196.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq197.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq198.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq199.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq200.png)

and a∗= 0.45724(R∗T ∗ c )2/p∗ c, b∗= 0.0778(R∗T ∗ c /p∗ c). A(T) is the Soave function for accounting temperature dependence on potential energy, while ω = 0.2249 is the acentric factor for CO2. Internal energy has the expression


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq201.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq202.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq203.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq204.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq205.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq206.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq207.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq208.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq209.png)

For this specific model, Dcv ̸= 0, and is


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq210.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq211.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq212.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq213.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq214.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq215.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq216.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq217.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq218.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq219.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq220.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq221.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq222.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq223.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq224.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq225.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq226.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq227.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq228.png)

10

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

Scheme Ref. Fρ Fρe PEP IG PEP RG Fully Cons.

EPEP-RG new ρλu α Fρ −u ρ2λ n.a.  

APEP-RG new ρ u α Fρ −u ρ2λ  H 

APEC Terashima et al. [14] ρ u � ρe −δ +α δ +ρ


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq229.png)

KEEP Kuya et al. [35] ρ u Fρ e   

KEEPPE Shima et al. [18] ρ u ρe u   

KGPPt Bernades et al. [23] ρ u –   


> **Table 1: Summary of the compared numerical discretizations. : property verified, H: property verified approximately, : property not verified, n.a. : not applicable. λ = (∂e/∂ρ)p, α = (∂ρe/∂ρ)p, ρλ = δ +(ρ2λ)/δ +α. Momentum and total energy fluxes are calculated according to Eq. (11).**


### 6 Numerical results

In this section, we present numerical tests designed to assess the proposed discrete formulations. The focus in on verifying the correct fulfillment of the PEP property and the exact total-energy conservation in inviscid flows. To place the performance of the proposed schemes in context, we compare them against classical formulations from the literature; a summary of all considered schemes in real gases simulations is provided in Table 1. For the thermally perfect gas simulations, our formulation employs the mass flux in Eq. (32) and the internal energy flux contained in Eq. (31). To compare the performances of our newly derived schemes, we consider the KEEP scheme by Kuya et al. [35], commonly used for simulations of calorically perfect gases, and its variant KEEPPE by Shima et al. [18] which is PEP for calorically perfect gases; regarding formulations specifically designed for real-gas simulations we consider the Approximately Pressure Equilibrium Conserving (APEC) scheme developed by Terashima et al. [14] and the pressure-based KGPPt scheme studied in Bernades et al. [23]. In all tests, time integration is carried out using the standard fourth-order Runge–Kutta method.

We consider the compressible Euler equations in dimensionless form: reference quantities are set as the standard ambient conditions for temperature and pressure (SATP), i.e. T ∗ SATP = 298.15 K and p∗ SATP = 1 atm. This normalization enables a consistent treatment of different thermodynamic regimes and allows the equation of state to be adapted to conditions representative of each model. In particular, we examine thermally perfect gases at high enthalpy, van der Waals fluids at supercritical conditions, and Peng–Robinson fluids at both supercritical and transcritical regimes.

The assessment is carried out using two benchmark problems. A one-dimensional density wave is used to evaluate the PEP property and to compare the proposed schemes against classical counterparts. Additionally, a two-dimensional double-jet configuration is employed to investigate the schemes behavior in a more demanding flow setting, studying both the energy conservation and the insurgence of non-physical pressure oscillations.

6.1 One-dimensional density wave

The one-dimensional density wave is solved in the domain Ω: x ∈[0, L] with L = 1, discretized in N = 41 evenly spaced points (h = 0.025). A fourth-order accurate spatial discretization is employed. To minimize contamination from time-integration errors, the CFL number has been set to 5 × 10−3 for all the considered cases. The initial conditions correspond to a smooth density perturbation convected at constant velocity, defined as


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq230.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq231.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq232.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq233.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq234.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq235.png)

with {ρ0, p0} = {ρc, 100} for the van der Waals and Peng–Robinson models in the supercritical regime, while {ρ0, p0} = {ρSATP, 0.45} for the thermally perfect one, where ρ∗ SATP = 1.795 Kg/m3. Modulation constants are A = 0.07 and B = 0.12. The value u0 = 1 sets the reference time tref = L/u0 = 1, hence t = t∗.

11

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 5 10 16 -1

0

1 10-4

0 8 16 -1

0

1 10-14

(a) Thermally perfect gas

0 2 4 6 8 10 12 13 -1

0

1 10-4

0 7 13 0

1

2

10-14

(b) Van der Waals EoS (supercritical conditions)

0 2 4 6 8 10 12 14 -1

0

1 10-4

0 7 14 -2

0

2 10-14

(c) Peng–Robinson EoS (supercritical conditions).


> **Figure 1: Global kinetic-energy evolution for the one-dimensional density wave test with various gas models.**

To monitor the onset of numerical instabilities, we track the normalized variation of a generic quantity ϕ defined as


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq236.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq237.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq238.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq239.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq240.png)

In particular, we consider the kinetic energy ρκ. Since both pressure and velocity are expected to remain constant for this problem, the kinetic energy should be preserved up to machine precision. Therefore, deviations in ⟨ρκ⟩provide a sensitive indicator of scheme robustness.


> **Fig. 1 reports the evolution of ⟨ρκ⟩for the schemes under consideration. All shown non-PEP schemes exhibit numerical blow-up. Among them, the first to become unstable is KEEP, which is not PEP even for ideal gases. This is followed by KEEPPE, which satisfies the PEP property only for calorically perfect gases, and by APEC, which is only approximately PEP for real-gas models. In contrast, EPEP-RG maintains the error on kinetic energy at machine-zero throughout the simulation. The APEP-RG scheme (not displayed) has performances almost indistinguishable from that of the EPEP-RG in this time interval. These trends are consistent across all the tested thermodynamic models: thermally perfect, van der Waals, and Peng–Robinson.**

Further insight is provided in Fig. 2, which shows solution snapshots at a time when KEEP and KEEPPE have already become unstable, while APEC is still running. For EPEP-RG, the pressure remains equal to its initial value and the density closely matches the exact solution. On the other hand, APEC exhibits spurious oscillations in both pressure and density, indicating the onset of numerical degradation despite not having blown up yet.

For the Peng–Robinson model, the EPEP-RG, APEP-RG and KGPPt simulations have been performed up to a final time t = 100, without recording any instability. In Fig. 3, the density profile solution is reported at t = 100, together with the time history of the total energy. Interestingly enough, the density profiles of the APEP-RG and KGPPt formulations show similar dispersion features, a circumstance that should probably be attributed to the fact that the two formulations share the same mass and momentum fluxes. Fig. 3b shows that, as predicted, the global total energy remains constant up to machine precision in this long simulation for the EPEP-RG and APEP-RG schemes, whereas for the KGPPt scheme a slow accumulation is present during the whole simulation, reaching a final value of the order 10−3.

To assess the performances of the newly proposed schemes in transcritical conditions, a simulation has been also carried out with the Peng–Robinson model for initial conditions {A, B, ρ0, p0} = {0, 2/3, ρc, 135}. In this transcritical case, in addition to the KEEP, APEC and KEEPPE formulations, also the EPEP-RG scheme shows strong instabilities, probably due to the near-singular behavior of the thermodynamic derivatives in the vicinity of the pseudo-critical line, where the main thermodynamic quantities—such as internal energy or the speed of sound—undergo rapid and large variations that can violate the assumptions on which the exact discrete formulation relies. Nevertheless, the APEP-RG formulation shows a good robustness, similar to that of the KGPPt scheme, which is exactly PEP for this case. Note that for this transcritical case, which is particularly susceptible to instabilities, even the KGPPt scheme eventually diverges at long times. In fact, a long simulation shows blow up for both the APEP-RG and KGPPt schemes at t ≈96 and t ≈107, respectively. Consequently, it seems that no numerical method currently available is capable of handling this test for sufficiently long times. This result could possibly be consistent with the previous observations about the instability near the pseudo-critical line as a consequence of the direct computation of the thermodynamic derivatives to solve the main system of equations. In fact, the KGPPt involves the expression of the speed of sound, whose evaluation inherently requires several such derivatives, making the scheme susceptible to some ill-conditioning in the transcritical region. In Fig. 4, APEP-RG and KGPPt density profiles and total energy evolutions are reported at t = 50. APEP-RG

12

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 0.2 0.4 0.6 0.8 1 0

0.1

0.2

0.3

0.4

0.5

(a)

0 0.2 0.4 0.6 0.8 1 -0.3

-0.2

-0.1

0

0 0.5 1 -1

0

1 10-13

(b)

0 0.2 0.4 0.6 0.8 1 20

40

60

80

100

120

(c)

0 0.2 0.4 0.6 0.8 1 -0.3

-0.2

-0.1

0

0 0.5 1 -2

0

2 10-13

(d)

0 0.2 0.4 0.6 0.8 1 0

20

40

60

80

100

120

(e)

0 0.2 0.4 0.6 0.8 1 -0.3

-0.2

-0.1

0

0 0.5 1 -2

0

2 10-13

(f)


> **Figure 2: Density and pressure profiles for the one-dimensional density wave test. (a)-(b): thermally perfect gas (t = 16), (c)-(d) van der Waals equation of state in supercritical conditions (t = 13), (e)-(f) Peng–Robinson equation of state in supercritical conditions (t = 14).**

13

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 0.2 0.4 0.6 0.8 1 20

40

60

80

100

120

(a)

0 20 40 60 80 100

0

2

4

6

8

10

12 10-4

0 50 100 -2

0

2 10-14

(b)


> **Figure 3: Density profile (a) and total energy evolution (b) for the one-dimensional density wave test for the Peng– Robinson equation of state in supercritical conditions at t = 100.**

0 0.2 0.4 0.6 0.8 1 0

100

200

300

400

500

(a)

0 10 20 30 40 50

0

2

4

6

10-4

0 50 -1

0

1 10-14

(b)


> **Figure 4: Density profile (a) and total energy evolution (b) for the one-dimensional density wave test for the Peng– Robinson equation of state in transcritical conditions at t = 50.**

shows a slightly better agreement with the exact solution, while KGPPt exhibits a lack of total energy conservation, with an error accumulating during the evolution and reaching a final value of approximately 6 × 10−4.

6.2 Two-dimensional inviscid double-jet flow at high-enthalpy and supercritical conditions

The two-dimensional, inviscid double-jet flow is simulated in the rectangular domain (x, y) ∈[0, L] × [−L/4, L/4], with L = 1, discretized in Nx ×Ny = 65×33 evenly spaced points with second-order accurate spatial discretizations, with CFL set as 0.01 to constrain time-integration errors. The flow is initialized as


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq241.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq242.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq243.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq244.png)


![Equation](images/2026_Coppola_Aiello_DeMichele_PEP_APEC_KEEP_v2_eq245.png)

14

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 0.2 0.4 0.6 0.8 1 9.94

9.96

9.98

10

10.02

10.04

10.06

10.08 10-1

(a)

-0.2 -0.1 0 0.1 0.2 9.92

9.94

9.96

9.98

10

10.02

10.04

10.06

10.08 10-1

(b)

0 0.5 1 1.5 2 2.5 -2.5

-2

-1.5

-1

-0.5

0

0.5 10-4

0 1 2 -1

0

1 10-14

(c)

0 0.5 1 1.5 2 2.5 -2

0

2

4

6

8 10-5

(d)


> **Figure 5: Two-dimensional inviscid double-jet flow for a thermally perfect gas. (a) pressure profiles evaluated at y = 0.33 and t/tref = 2.5, (b) pressure profiles evaluated at x = 0 and t/tref = 2.5, (c) total energy evolution, (d) entropy evolution.**

with the velocity field being the same for each gas model, given {Au, ε, m} = {1/2, 0.05, 3}, to have a transversal jet that generates three roll-up vortices. Temperature and pressure fields are, on the other hand, established with respect to the specific equation of state. Thus, we set {a, At, p0} = {2.6, 2/3, 0.1} for the thermally perfect case and {a, At, p0} = {2.5, 1/2, 150} for the supercritical test, carried out by means of the van der Waals model. Finally, the parameter θ = 30 represents the thickness of the shear layer, and a reference time is defined as tref = m−1/ maxx,y u(x, y, 0) ≈0.445.

As the numerical mass flux defined in Eq. (25) is singular in the wide, initially uniform regions, we decided to use only the APEP-RG formulation for this test case, compared against the KEEPPE, APEC, and KGPPt schemes, which have shown better performances in the previous 1D test case with respect to the KEEP scheme. In Fig. 5, the results of the high-enthalpy case are reported, in terms of pressure profiles along horizontal and vertical lines at y = 0.33 and x = 0, respectively (Fig. 5a and 5b), together with the time evolution of the total energy and entropy. Although the results are generally satisfactory for all the schemes considered, Fig. 5b shows that KEEPPE and APEC are starting to exhibit point-to-point oscillations in the vertical profile of pressure. The total energy evolution depicted in Fig. 5cconfirms that the KGPPt scheme steadily deviates from exact conservation, whereas the global entropy evolution reported in Fig. 5d shows that all the considered formulations violate the exact entropy preservation, as none of them is exactly entropy conservative. The pressure oscillations highlighted for the KEEPPE and APEC schemes in Fig. 5b are much more visible in the supercritical case computed with the van der Waals model, reported in Fig. 6a and 6b. In this simulation the APEP and KGPPt schemes remain essentially free of oscillation.

15

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 0.2 0.4 0.6 0.8 1 9.85

9.9

9.95

10

10.05

10.1 10-1

(a)

-0.2 -0.1 0 0.1 0.2 9.85

9.9

9.95

10

10.05

10.1

10.15 10-1

(b)

0 0.5 1 1.5 2 2.5 -8

-7

-6

-5

-4

-3

-2

-1

0

1 10-4

0 1 2 -1

0

1 10-14

(c)

0 0.5 1 1.5 2 2.5 -2

0

2

4

6

8

10

12 10-5

(d)


> **Figure 6: Two-dimensional inviscid double-jet flow for a van der Waals gas at supercritical conditions. (a) pressure profiles evaluated at y = 0.33 and t/tref = 2.5, (b) pressure profiles evaluated at x = 0 and t/tref = 2.5, (c) total energy evolution, (d) entropy evolution.**

6.3 Two-dimensional inviscid double-jet flow at transcritical conditions

Transcritical conditions necessitate a more detailed analysis due to the different behavior of thermodynamic derivatives near the critical point, especially when discretizing the energy equation which requires the computation of internal energy and its gradients in the thermodynamic space. In this section, the inviscid double-jet flow is simulated onto the same grid and within the same numerical setup presented in Section 6.2. Initial conditions also share the same symbolic form, with the {a, At, p0} = {2, 1/2, 180}. This ensures a dimensional temperature T ∗∈[∼298.8, ∼587.9] K, therefore crossing the critical temperature for CO2, T ∗ c = 304.12 K. Corresponding dimensional pressure is p∗≈ 2.43 × p∗ c = 2.43 × 73.8 atm. This time, the Peng–Robinson model has been used to carry out the simulations. Figures 7a and 7b report the usual pressure profiles at t = 1.6. In this case, the pressure oscillations are much more evident for the KEEPPE and APEC schemes, even if the simulation is stopped at an earlier time. The total energy evolution is consistent with the other simulations, whereas the entropy evolution reported in Fig. 7d shows an oscillating behavior, which is exacerbated for the KGPPt formulation. Fig. 8 reports a snapshot of the two-dimensional pressure field as calculated by the various formulations, confirming the contamination of the solution due to the growing oscillations in the KEEPPE and APEC schemes.

16

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 0.2 0.4 0.6 0.8 1 9.8

9.85

9.9

9.95

10

10.05

10.1

10.15

10.2

10.25 10-1

(a)

-0.2 -0.1 0 0.1 0.2 9.8

9.85

9.9

9.95

10

10.05

10.1

10.15 10-1

(b)

0 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 -20

-15

-10

-5

0

5 10-5

0 0.5 1 1.5 -1

0

1 10-14

(c)

0 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 -2

-1.5

-1

-0.5

0

0.5

1

1.5 10-4

(d)


> **Figure 7: Two-dimensional inviscid double-jet flow for a Peng–Robinson gas at transcritical conditions. (a) pressure profiles evaluated at y = 0.33 and t/tref = 2.5, (b) pressure profiles evaluated at x = 0.5 and t/tref = 2.5, (c) total energy evolution, (d) entropy evolution.**


### 7 Conclusions

As highlighted throughout this work, the numerical simulation of compressible flows for non-ideal fluids is frequently challenged by the generation of spurious pressure oscillations. While addressing this issue requires numerical methods that satisfy the pressure-equilibrium-preserving (PEP) property, ensuring strict PEP compliance for general equations of state has historically proven difficult. Previous attempts often compromise the exact discrete conservation of total energy, a property that remains essential for maintaining physical consistency and correctly capturing shocks.

In this work, we demonstrate that the exact discrete conservation of mass, momentum, and total energy is not mutually exclusive with the PEP condition. Although the present mathematical framework is derived in the context of finitedifference discretizations, the resulting two-point numerical fluxes are highly versatile. They can be seamlessly applied to other spatial discretization frameworks, such as structured and unstructured finite volume or discontinuous Galerkin formulations.

Our main contribution is the development of a fully conservative and exactly pressure-equilibrium-preserving scheme, denoted as Exact PEP (EPEP-RG). We derive a generic formula applicable to any arbitrary equation of state and provide specific, computationally viable flux formulations for thermally perfect, van der Waals, and Peng–Robinson gases. Building upon this theoretical foundation, and recognizing the complexities inherent to certain thermodynamic regimes, we also propose a robust practical alternative: the Approximate PEP (APEP-RG) scheme. The APEP-RG

17

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

0 0.5 1 -0.25

0

0.25

0.978

0.989

0.999

1.010

1.020

(a) APEP-RG

0 0.5 1 -0.25

0

0.25

0.978

0.989

0.999

1.009

1.020

(b) KGPPt

0 0.5 1 -0.25

0

0.25

0.977

0.988

0.999

1.010

1.021

(c) APEC

0 0.5 1 -0.25

0

0.25

0.975

0.987

1.000

1.013

1.025

(d) KEEPPE


> **Figure 8: Instantaneous pressure fields for the two-dimensional inviscid double-jet flow for a Peng–Robinson gas at transcritical conditions and t/tref = 1.6.**

formulation strictly maintains full primary conservation, including kinetic-energy preservation by convection, while enforcing the PEP condition in an approximate sense.

We have validated the proposed schemes using rigorous numerical benchmarks. In the density wave advection test, the EPEP-RG scheme is able to successfully preserve pressure equilibrium across a variety of EoS. This behavior contrasts favorably with standard formulations from the literature, which either become unstable or fail to conserve total energy. Furthermore, simulations of a compressible mixing layer confirm the exact conservation of total energy alongside the elimination of spurious oscillations in the pressure field, effectively addressing the energy conservation limitations of previous non-oscillatory schemes. While the exact EPEP-RG formulation can encounter numerical singularities due to problematic thermodynamic derivatives in transcritical regimes, the APEP-RG scheme reliably circumvents these issues. Overall, the proposed schemes demonstrate highly favorable robustness, stability, and accuracy. The APEPRG framework, in particular, emerges as a simple and resilient tool capable of managing severe thermodynamic nonlinearities—including transcritical phenomena—without sacrificing primary conservation or numerical stability.

Despite these successes, certain limitations remain that pave the way for future research. The EPEP-RG scheme currently requires a specialized fix to properly handle the singularities of thermodynamic derivatives across phase boundaries in transcritical regimes or in areas where thermodynamic quantities are nearly uniform. The APEP-RG scheme provides a robust workaround for these issues, while also introducing a degree of flexibility regarding the choice of the density averaging operator. Future work could systematically explore the characteristics of various averaging operators to identify the optimal formulation, potentially exploiting this degree of freedom to impart additional structure-preserving properties to the scheme, such as discrete entropy conservation or stability.

Further efforts will focus on extending this framework to multi-component and multi-phase flow formulations, as well as evaluating the schemes’ performances on more complex, physically realistic configurations, such as wall-bounded turbulent flows. Ultimately, the conservative and equilibrium-preserving methodologies introduced herein represent a significant step forward toward a unified, robust framework for the high-fidelity simulation of complex real-gas flows.

18

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT


## References

[1] R. Penik, E. Rinaldi, P. Colonna, Computational fluid dynamics of a radial compressor operating with supercritical CO2, J. Eng. Gas Turbines Power 134 (2012). [2] L. Jofre, J. Urzay, Transcritical diffuse-interface hydrodynamics of propellants in high-pressure combustors of chemical propulsion systems, Prog. Energy Combust. Sci. 82 (2021) 100877. [3] A. Guardone, P. Colonna, M. Pini, A. Spinelli, Nonideal compressible fluid dynamics of dense vapors and supercritical fluids, Annu. Rev. Fluid Mech. 56 (2024) 241–269. [4] H. Terashima, M. Koshi, Approach for simulating gasliquid-like flows under supercritical pressures using a high-order central differencing scheme, J. Comput. Phys. 231 (2012) 6907–6923. [5] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: A quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160. [6] T. Y. Hou, P. G. LeFloch, Why nonconservative schemes converge to wrong solutions: error analysis, Math. Comput. 62 (1994) 497–530. [7] S. Karni, Multicomponent flow calculations by a consistent primitive algorithm, J. Comput. Phys. 112 (1994) 31–43. [8] K.-M. Shyue, A fluid-mixture type algorithm for compressible multicomponent flow with MieGrüneisen equation of state, J. Comput. Phys. 171 (2001) 678–707. [9] S. Kawai, H. Terashima, H. Negishi, A robust and accurate numerical method for transcritical turbulent flows at supercritical pressure with an arbitrary equation of state, J. Comput. Phys. 300 (2015) 116–135. [10] G. Lacaze, T. Schmitt, A. Ruiz, J. C. Oefelein, Comparison of energy-, pressureand enthalpy-based approaches for modeling supercritical flows, Comput. Fluids 181 (2019) 35–56. [11] R. Abgrall, S. Karni, Computations of compressible multifluids, J. Comput. Phys. 169 (2001) 594–623. [12] P. C. Ma, Y. Lv, M. Ihme, An entropy-stable hybrid scheme for simulations of transcritical real-fluid flows, J. Comput. Phys. 340 (2017) 330–357. [13] Y. Fujiwara, Y. Tamaki, S. Kawai, Fully conservative and pressure-equilibrium preserving scheme for compressible multi-component flows, J. Comput. Phys. 478 (2023) 111973. [14] H. Terashima, N. Ly, M. Ihme, Approximately pressure-equilibrium-preserving scheme for fully conservative simulations of compressible multi-species and real-fluid interfacial flows, J. Comput Phys. 524 (2025) 113701. [15] W. J. Feiereisen, W. C. Reynolds, J. H. Ferziger, Numerical Simulation of Compressible, Homogeneous Turbulent Shear Flow, Technical Report TF-13, Stanford University, 1981. [16] A. Jameson, The construction of discretely conservative finite volume schemes that also globally conserve energy or entropy, J. Sci. Comput. 34 (2008) 152–187. [17] S. Pirozzoli, Generalized conservative approximations of split convective derivative operators, J. Comput. Phys. 229 (2010) 7180–7190. [18] N. Shima, Y. Kuya, Y. Tamaki, S. Kawai, Preventing spurious pressure oscillations in split convective form discretization for compressible flows, J. Comput. Phys. 427 (2021) 110060. [19] H. Ranocha, G. Gassner, Preventing pressure oscillations does not fix local linear stability issues of entropy-based split-form high-order schemes, Commun. Appl. Math. Comput. 4 (2022) 880–903. [20] C. De Michele, G. Coppola, Asymptotically entropy-conservative and kinetic-energy preserving numerical fluxes for compressible Euler equations, J. Comput. Phys. 492 (2023) 112439. [21] C. De Michele, G. Coppola, Novel pressure-equilibrium and kinetic-energy preserving fluxes for compressible flows based on the harmonic mean, J. Comput. Phys. 518 (2024) 113338. [22] S. Kawai, S. Kawai, Logarithmic mean approximation in improving entropy conservation in KEEP scheme with pressure equilibrium preservation property for compressible flows, J. Comput. Phys. 530 (2025) 113897. [23] M. Bernades, L. Jofre, F. Capuano, Kinetic-energyand pressure-equilibrium-preserving schemes for real-gas turbulence in the transcritical regime, J. Comput. Phys. 493 (2023) 112477. [24] S. Pirozzoli, Stabilized non-dissipative approximations of Euler equations in generalized curvilinear coordinates, J. Comput. Phys. 230 (2011) 2997–3014. [25] Y. Kuya, S. Kawai, High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids, J. Comput. Phys. 442 (2021) 110482.

19

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

[26] P. G. LeFloch, J. M. Mercier, C. Rohde, Fully discrete, entropy conservative schemes of arbitrary order, SIAM J. Numer. Anal. 40 (2002) 1968–1992. [27] T. C. Fisher, M. H. Carpenter, High-order entropy stable finite difference schemes for nonlinear conservation laws: Finite domains, J. Comput. Phys. 252 (2013) 518–557. [28] H. Ranocha, Comparison of some entropy conservative numerical fluxes for the Euler equations, J. Sci. Comput. 76 (2018) 216–242. [29] A. Aiello, C. De Michele, G. Coppola, Entropy conservative discretization of compressible Euler equations with an arbitrary equation of state, J. Comput. Phys. 528 (2025) 113836. [30] G. Coppola, F. Capuano, L. de Luca, Discrete energy-conservation properties in the numerical solution of the Navier–Stokes equations, Appl. Mech. Rev. 71 (2019) 010803–1 – 010803–19. [31] G. Coppola, F. Capuano, S. Pirozzoli, L. de Luca, Numerically stable formulations of convective terms for turbulent compressible flows, J. Comput. Phys. 382 (2019) 86–104. [32] A. E. P. Veldman, A general condition for kinetic-energy preserving discretization of flow transport equations, J. Comput. Phys. 398 (2019) 108894. [33] G. Coppola, A. E. P. Veldman, Global and local conservation of mass, momentum and kinetic energy in the simulation of compressible flow, J. Comput. Phys. 475 (2023) 111879. [34] C. De Michele, G. Coppola, Numerical treatment of the energy equation in compressible flows simulations, Comput. Fluids 250 (2023) 105709. [35] Y. Kuya, K. Totani, S. Kawai, Kinetic energy and entropy preserving schemes for compressible flows by split convective forms, J. Comput. Phys. 375 (2018) 823–853. [36] A. Aiello, C. De Michele, G. Coppola, Formulation of entropy-conservative discretizations for compressible flows of thermally perfect gases, arXiv:2507.08115 [physics.flu-dyn] (2025). [37] E. Tadmor, The numerical viscosity of entropy stable schemes for systems of conservation laws. I, Math. Comput. 179 (1987) 91–103. [38] E. Tadmor, Entropy stability theory for difference approximations of nonlinear conservation laws and related time-dependent problems, Acta Numerica 12 (2003) 451512. [39] H. Ranocha, Entropy conserving and kinetic energy preserving numerical methods for the Euler equations using summation-by-parts operators, in: S. J. Sherwin, D. Moxey, J. Peiró, P. E. Vincent, C. Schwab (Eds.), Spectral and High Order Methods for Partial Differential Equations ICOSAHOM 2018. Lecture Notes in Computational Science and Engineering, volume 134, Springer, Cham, 2020, p. 525 535. doi:10.1007/978-3-030-39647-3_ 42. [40] C. De Michele, A. K. Edoh, G. Coppola, Finite-difference compatible entropy-conserving schemes for the compressible Euler equations, J. Comput. Phys. (2025) 114262. [41] F. Ismail, P. L. Roe, Affordable, entropy-consistent Euler flux functions II: Entropy production at shocks, J. Comput. Phys. 228 (2009) 5410–5436. [42] R. Klein, B. Sanderse, P. Costa, R. Pecnik, R. Henkes, Generalized Tadmor conditions and structurepreserving numerical fluxes for the compressible flow of real gases, 2026. doi:10.48550/arXiv.2603.15112. arXiv:2603.15112. [43] M. A. Hansen, T. C. Fisher, Entropy Stable Discretization of Compressible Flows in Thermochemical Nonequilibrium, Technical Report, Sandia National Lab. (SNL-NM), Albuquerque, NM (United States), 2019. URL: https://www.osti.gov/biblio/1763209. doi:10.2172/1763209. [44] M. Chase, NIST-JANAF Thermochemical Tables, 4th Edition, American Institute of Physics, 1998. URL:

https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf. [45] B. J. McBride, M. J. Zehe, S. Gordon, NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species, Technical Report, Glenn Research Center, Cleveland, Ohio, 2002. URL: https://ntrs. nasa.gov/api/citations/20020085330/downloads/20020085330.pdf. [46] I. Tosun, The Thermodynamics of Phase and Reaction Equilibria, second edition, Elsevier B.V., 2021.

20

