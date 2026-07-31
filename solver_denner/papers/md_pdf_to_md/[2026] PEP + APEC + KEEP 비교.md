
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


# arXiv:2605.03617v1  [physics.flu-dyn]  5 May 2026

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

contact discontinuities [4]. This phenomenon is fundamentally analogous to the well-documented difficulties encountered in the simulation of multicomponent flows [5]. Just as the abrupt variation of the specific heat ratio across a multi-fluid interface triggers severe numerical artifacts, the highly non-linear dependence of pressure on density and internal energy in a real gas EoS induces similar unphysical fluctuations. If left unmitigated, these spurious oscillations in pressure and velocity can propagate rapidly, leading to non-physical thermodynamic states and ultimately resulting in the catastrophic failure of the simulation.

The root of this issue lies in the extreme difficulty of simultaneously satisfying two fundamental numerical requirements: preserving local pressure equilibrium and maintaining strict, discrete conservation of total energy. When fully conservative numerical schemes are applied to real gases, the primary variables (density, momentum and total energy) are updated and exactly conserved. However, the subsequent non-linear inversion of the EoS, required to recover the pressure from the updated density and internal energy, frequently fails to maintain pressure equilibrium across moving contact discontinuities. Conversely, many existing strategies attempt to enforce a Pressure-Equilibrium-Preserving (PEP) condition by modifying the energy equation or employing primitive-variable formulations (i.e., discretizing the pressure equation directly in place of the total energy one). While these successfully suppress spurious oscillations, they inherently sacrifice exact discrete conservation of total energy. This loss of full conservation can be highly detrimental, as, in the presence of strong shocks, the scheme may fail to converge to the correct weak solution [5, 6].

To address the spurious oscillations, a variety of numerical strategies have been proposed over the past decades. Early efforts, predominantly rooted in the study of multicomponent flows, focused on quasi-conservative formulations. As demonstrated by Karni [7], discretizing the Euler equations in primitive variables avoids the energy-to-pressure EoS inversion, successfully maintaining local pressure equilibrium. To recover partial conservation, Abgrall [5] and later Shyue [8] introduced quasi-conservative schemes wherein the standard conservative fluid equations are solved alongside non-conservative advection equations for the thermodynamic parameters (e.g., the specific heat ratio). By carefully discretizing these auxiliary equations, these methods achieve the PEP property. However, the explicit inclusion of non-conservative terms inherently sacrifices exact discrete conservation of total energy, leading to incorrect shock propagation speeds and thermodynamic anomalies across strong discontinuities. Similar problems were encountered by evolving the pressure equation for transcritical and supercritical fluids [4, 9], and even the use of enthalpy and internal energy has been attempted [10].

Recognizing the deficiencies of globally non-conservativemodels, double-flux methods have been introduced to bridge the gap between interface stability and shock fidelity. Originally formalized by Abgrall and Karni [11], this approach was later adapted for real-gas and transcritical flows by Ma et al. [12]. The double-flux strategy employs standard, fully conservative fluxes in the bulk of the flow, but locally switches to a thermodynamically “frozen", non-conservativeauxiliary flux at sharp density gradients to enforce pressure equilibrium. While highly effective at mitigating unphysical waves in practice, double-flux methods still rely on local non-conservation. Furthermore, they require complex and empirical sensor functions to blend the two flux formulations, making them computationally rigid and highly sensitive to tuning parameters when applied to arbitrary, highly non-linear real gases.

To overcome these limitations, efforts were made to modify numerical fluxes directly within a fully conservative framework. For multicomponent flows, Fujiwara et al. [13] sought to address this by deriving equilibrium conditions to construct specialized numerical fluxes without relying on auxiliary transport equations. Terashima et al. [14] extended this approach to the complex thermodynamicregimes of real gases, developing consistent numerical fluxes maintaining exact conservation of the primary variables. However, the final formulation yields only an approximately PEP scheme, with a fixed spatial order of accuracy. As a result, residual pressure oscillations can still manifest when simulating extremely severe thermodynamic gradients or phase transitions.

An alternative philosophy has sought to guarantee exact pressure equilibrium and numerical robustness by coupling PEP conditions with Kinetic-Energy-Preserving (KEP) schemes. The KEP framework was originally developed to prevent non-linear aliasing errors in standard compressible turbulence simulations without relying on numerical dissipation [15, 16, 17]. For calorically perfect ideal gases, researchers have successfully derived discretizations that combine KEP properties with exact pressure equilibrium [18, 19, 20, 21, 22]. For real-fluid simulations, a recent work by Bernades et al. [23] proposed a specifically designed KEP and PEP scheme. Yet, to strictly maintain both pressure equilibrium and kinetic energy preservation across complex thermodynamic states, they were forced to abandon exact total energy conservation, opting instead to solve a non-conservative pressure evolution equation. Consequently, a numerical methodology that simultaneously guarantees full conservation—particularly of total energy—and exact pressure equilibrium for real and thermally perfect gases remains a missing link in the literature.

To address this critical gap, the present study introduces a novel numerical framework that, for the first time, achieves a fully conservative, kinetic-energy-preserving,and exactly pressure-equilibrium-preserving discretization of the compressible Euler equations for thermally perfect and real gases, with an arbitrary EoS. Our approach demonstrates that it is mathematically and practically possible to retain the exact discrete evolution of the total energy equation without

2

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

triggering spurious interfacial oscillations, offering a robust and thermodynamically consistent foundation for simulating extreme real-gas phenomena. To overcome the intrinsic difficulties associated with simulations in transcritical conditions, an approximate formulation is also proposed, based on a slight modification of the set of exact fluxes. By retaining excellent PEP properties, with unrivaled performances among existing formulations, the set of approximate fluxes shows robust behavior also in extreme conditions.

The remainder of this paper is organized as follows. Section 2 establishes the problem formulation, detailing the continuous compressible Euler equations and the underlying discretization framework. In Section 3, we present the core methodology, rigorously deriving the fully conservative numerical fluxes and proving their KEP and PEP properties at the discrete level. This generic formulation is subsequently particularized for specific thermodynamic models: Section 4 addresses thermally perfect gases, while Section 5 extends the approach to real gases, with a specific focus on the van der Waals and Peng–Robinson equations of state. Section 6 presents a suite of numerical experiments—including a standard density wave test and a multidimensional mixing case—designed to validate both the PEP property and exact discrete conservation. Finally, Section 7 summarizes the key findings and offers concluding remarks.


### 2 Problem formulation

2.1 Governing equations

The compressible Euler equations, which express the conservation of mass, momentum and total energy for compressible inviscid flows, can be written as:


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq001.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq002.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq003.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq004.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq005.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq006.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq007.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq008.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq009.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq010.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq011.png)

where ρ is the density, uα is the Cartesian velocity component along xα, p is the pressure and E the total energy per unit mass, which is the sum of the internal (e) and kinetic (κ = uαuα/2) energies per unit mass: E = e + κ. We will assume the convention that Greek subscripts, such as α or β, refer to the components of Cartesian vectors, whereas Latin subscripts, such as i or j appearing in the subsequent sections of this paper, are used to denote the values of the discretized variable on a nodal point. Unless otherwise stated, the summation convention over repeated Greek indices is assumed. The system (1)–(3) is closed by an equation of state, relating p, ρ and the temperature T , and by specifying the dependence of internal energy on temperature and density, which is typically provided through suitable departure functions.

In what follows, we will also need to consider the induced balance equations for additional quantities related to the primary variables ρ, ρuβ, and ρE. These equations are easily derived by combining Eqs. (1)–(3) and by applying the chain and product rules of differentiation, with respect to both temporal and spatial variables. In particular, we will use the evolution equations for the kinetic and internal energies:


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq012.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq013.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq014.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq015.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq016.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq017.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq018.png)

whose sum returns the total energy equation (3). Moreover, to discuss the PEP property, which is the main subject of the present paper, we will use the evolution equations for the velocity components uβ, and the pressure p, which can be written as ∂uβ


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq019.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq020.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq021.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq022.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq023.png)

where c = � (∂p/∂ρ)s is the speed of sound, s being the specific entropy. Eqs. (6) and (7) show that if at a given time instant a spatially constant state is assumed for both uβ and p, all the spatial derivatives at the right-hand sides vanish. This also causes the temporal derivatives at the left-hand side to vanish, meaning that the constant distribution

3

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

of pressure and velocity remains constant also in time. This is the Pressure Equilibrium property for the compressible Euler equations, which allows for the existence of density wave solutions traveling at constant velocity in a uniform pressure field.

2.2 Discretization setting

Our focus is on the discretization of the spatial terms in Eqs. (1)–(3), for which we always use central (non-dissipative) schemes. To isolate spatial errors, we assume in the theoretical analysis that the manipulation of time derivatives can be performed as in the continuous case. This type of semidiscretized analysis leads to a system of Ordinary Differential Equations (ODE) whose temporal integration is performed by using standard solvers. In this paper, the theory will be developed for the one-dimensional version of Eq. (1)–(3), which is obtained by omitting the Greek subscripts. The extension to the general three-dimensional case is straightforward and involves simply including the analogous spatial terms along the additional Cartesian components. The analysis is developed by referring to a Finite Difference (FD) treatment of the convective terms in Eqs. (1)–(3) on a uniform mesh {xi} of width h = xi+1 −xi, although most of our results could also be applied in other frameworks, such as Finite Volume or Finite Element methods, and can be generalized to nonuniform or curvilinear meshes, following standard approaches [24, 25]. Finally, we focus on the discretization of the various spatial operators at internal points, neglecting the effects of boundary conditions, which could be studied with ad hoc methods.

Although we work in a FD framework, we will design the discretization of the convective spatial terms in Eqs. (1)–(3) by specifying the numerical fluxes, i.e. by assuming a locally conservative discretization in the form


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq024.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq025.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq026.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq027.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq028.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq029.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq030.png)

where φ is a generic transported quantity per unit mass (1, u or E in Eq. (1), (2) or (3), respectively). The numerical flux F i+ 1

2 ρφ is a consistent approximation of ρuφ at xi+1/2 = xi + h/2, and is a function of the sets of nodal values ρi, ui and φi. Note that, even if the right-hand sides of Eq. (1)–(3) can be expressed as the divergence of a single term including convective and pressure mechanisms, Eq. (8) refers to the specification of the numerical fluxes for the convective contribution only (the first terms at the r.h.s. of Eqs. (1)–(3)). The terms involving the pressure in Eq. (2) and (3) are discretized by using standard central derivative formulas, as specified in Section 2.3. Eq. (8) reproduces the divergence structure of the convective terms in Eqs. (1)–(3) and implies also global conservation of the invariant ρφ by virtue of the telescoping property.

To simplify the notation, we will make use of the difference and average operators


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq031.png)

which allows to write the central difference as ϕi+1 −ϕi−1 = 2δ −ϕ i and to express Eq. (8) as


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq032.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq033.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq034.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq035.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq036.png)

In most cases, we will omit the apex when referring to the flux evaluated at xi+1/2: Fρφ = F i+ 1

2 ρφ , and, when no ambiguity can arise, we also omit the suffix in the expressions of the difference or of the average of a generic quantity: φ = φ i, δ −φ = δ −φi.

Within the described setting, the spatial discretization of the r.h.s. of Eqs. (1)–(3) is completely determined by the specification of the convective fluxes Fρ, Fρu, FρE and by the discretization details of the pressure terms ∂p/∂x and ∂up/∂x in Eqs. (2) and (3), respectively. The derivation exposed in the next sections considers second-order approximations, which implies that the numerical flux Fρφ is in general a two-point interpolation of (ρuφ)i+1/2, and is a function of ρ, u and φ at nodes xi and xi+1 only. The extension to higher orders is obtained by using standard arguments and is detailed in several references [26, 27, 28, 21, 29].

2.3 Kinetic Energy Preserving formulation

A first constraint on the possible choice for the numerical fluxes is given by the requirement that the formulation satisfies the Kinetic Energy Preserving (KEP) property [15, 16, 30, 31], which amounts to the requirement that the discrete induced evolution equation for the kinetic energy has a convective term that is locally and globally conservative, as for the continuous equation (cf. Eq. (4)). It is well known ([16, 32, 33]) that, for second order fluxes, the KEP property is satisfied when the convective fluxes for mass and momentum are linked by the simple relation


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq037.png)

4

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

In this case, kinetic energy evolves according to a discrete equation analogous to Eq. (4), in which the convective term can be cast as a difference of numerical fluxes, with flux [33, 34]


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq038.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq039.png)

The requirement that the discrete total energy evolves coherently with the induced evolution of kinetic energy suggests designing the convective numerical flux for total energy FρE as the sum of the kinetic-energy flux given by Eq. (9) and of a convective flux for internal energy Fρe. The conservative pressure term is correspondingly built as the sum of the induced discretized pressure term in Eq. (4) (which belongs to the discrete pressure term in the momentum equation (2)) and of a consistent discretization of the pressure term in Eq. (5). Assuming standard second-order central schemes for these terms, one has


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq040.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq041.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq042.png)

which gives the correct discretization of the pressure term in the total energy equation as consistent with the advective form of the derivative of the product: ∂up


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq043.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq044.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq045.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq046.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq047.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq048.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq049.png)

where up = (uipi+1 + piui+1) /2 is the product mean [31].

In conclusion, a locally conservative and KEP discretization is characterized by the set of total fluxes (i.e. including both convective and pressure contributions) given by


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq050.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq051.png)

where the mass and internal energy fluxes Fρ and Fρe are still unspecified, except for the obvious requirement that they are consistent approximations of mass and internal energy fluxes. The set of fluxes reported in Eq. (11) defines a formulation which is locally conservative of linear invariants ρ, ρu and ρE and preserves the (local and global) conservation of kinetic energy by convection. It embodies many KEP formulations available in the literature that have gained popularity in recent years. As examples, the simple choice Fρ = ρ u and Fρe = ρ u e gives the KEEP scheme [35] which, even if it does not have exact additional structural properties, has been widely used because of its good performances for ideal gases. The choice Fρ = ρ u and Fρe = ρ u eH (where eH = 2eiei+1/(ei + ei+1) is the harmonic mean) gives the zeroth-order AEC scheme, developed in [20], whereas Fρ = ρ u and Fρe = ρe u give the KEEPPE scheme [18], these last two formulations being PEP for ideal gases. Recently, the present authors also determined mass and internal energy fluxes such that the formulation in Eq. (11) is Entropy Conservative (EC), i.e. it is able to discretely preserve the correct balance of entropy, for the case of real gases with an arbitrary equation of state [29] and for thermally perfect gases [36]. In the next section, we derive an expression for Fρ and Fρe which gives a formulation able to satisfy the PEP condition for thermally perfect and real gases with an arbitrary EoS.


### 3 PEP formulation

In this section, we analyze the general problem of enforcing the PEP condition for the system of compressible Euler equations, without making any assumptions about the equation of state. In Sections 4 and 5 we explicitly work out the particular cases of a thermally perfect gas and of various thermodynamic models for real gases.

3.1 Theoretical derivation of exact PEP fluxes

In order to discretely enforce the pressure equilibrium property, it is necessary to consider the discrete evolution equations for velocity components and pressure, which are the induced discrete counterparts of Eqs. (6) and (7). Among these, the pressure evolution equation is the more challenging one, due to its explicit dependence on the EoS. We therefore begin with the discrete velocity equation, which is independent of the EoS. General conditions for the discrete enforcement of the velocity equilibrium have previously been derived for the case of calorically perfect gases [21, 19], where it was shown that the KEP condition is sufficient to enforce it. This result still applies but, for completeness, we briefly revisit the derivation here.

First of all, we explicitly write the semidiscrete evolution equations for mass and momentum as obtained by discretizing Eqs. (1) and (2) according to the prescriptions illustrated in the previous section: dρi


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq052.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq053.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq054.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq055.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq056.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq057.png)

5

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

The discrete evolution equation for the velocity can be now obtained by using the expression of the time derivative of the velocity u in terms of the time derivatives of ρ and ρu


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq058.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq059.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq060.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq061.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq062.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq063.png)

From this relation, substituting Eqs. (12) and (13), one obtains


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq064.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq065.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq066.png)

In the case of spatially constant pressure pi = P and velocity ui = U, the pressure contribution at the right-hand side vanishes, and one is left with dui


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq067.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq068.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq069.png)

where we denoted with ˆFρ the form assumed by Fρ for uniform velocity and pressure fields. Eq. (16) shows that for the KEP discretization adopted, a uniform spatial distribution of pressure and velocity always induces a zero time derivative for ui, which implies that the velocity field remains uniform as time evolves.

Moving on to the discrete evolution of pressure, we start by considering the internal energy per unit volume ρe and use an equation of state in the form ρe = ρe(ρ, p). By taking the time derivative of this relation, we get


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq070.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq071.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq072.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq073.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq074.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq075.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq076.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq077.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq078.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq079.png)

Substituting the temporal derivatives with the right-hand sides of the mass and internal energy equations (1) and (5), we obtain ∂ρue


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq080.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq081.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq082.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq083.png)

where we defined


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq084.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq085.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq086.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq087.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq088.png)

Now we assume to discretize the convective terms in the mass and internal energy equations conservatively, i.e. as the difference of numerical fluxes δ −F/h, and to use the simple central discretization for the pressure term in the internal energy equation given in Eq. (10). This gives


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq089.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq090.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq091.png)

Assuming again uniform spatial distributions of pressure and velocity, one is left with


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq092.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq093.png)

where, as usual, ˆFρe and ˆFρ denote the form assumed by the numerical fluxes for uniform velocity and pressure fields. Eq. (17) reveals that to discretely maintain the uniform pressure distribution (i.e. to enforce the condition dpi/dt = 0), the mass and internal energy fluxes have to satisfy the constraint


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq094.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq095.png)

A condition analogous to that in Eq. (18) has already been analyzed in [13] for the case of a multicomponent mixture of real gases and has been the starting point to obtain approximate PEP formulations in [14]. An exact enforcement of Eq. (18) seems problematic at first sight, since it requires the specification of the numerical fluxes Fρ and Fρe such that the product between the exact difference δ−ˆFρ and an arbitrary function ˆα(ρ) can be expressed as the difference of numerical fluxes δ−ˆFρe. In Eq. (18), the function ˆα(ρ) is fixed by the thermodynamic model, whereas ˆFρ and ˆFρe can be chosen arbitrarily, provided that they are consistent approximations of the mass and internal energy fluxes.

To solve the problem expressed by Eq. (18) we proceed inspired by the steps of the theory of Tadmor [37, 38] and observe that ˆαi δ−ˆFρ is an exact difference if (and only if) ˆFρ δ+ ˆαi is, since the sum aiδ−bi + biδ+ai (which is a consistent discretization of the advective form of the derivative of the product ab) admits the decomposition as a difference of fluxes aiδ−bi + biδ+ai = δ−(ai+1bi). This allows us to rewrite Eq. (18) in the equivalent form


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq096.png)

6

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

and we are now faced with the problem of finding a numerical flux Fρ such that ˆFρδ + ˆαi is an exact difference. We proceed now by observing that the flux Fρ should be a consistent approximation of the analytical flux fρ(ρ, u) = ρu at xi+1/2. We assume that the function ˆα(ρ) is (at least locally) invertible, in such a way that the inverse function ρ(ˆα) can be considered, and the relation ρ – ˆα is a local one-to-one mapping. This confers to the variable ˆα the role of an entropy variable, in the terminology of the theory of Tadmor. In this case we can express the function ˆfρ(ρ) = fρ(ρ, U) through the new variable ˆα by defining the function ˆgρ(ˆα) = ˆfρ(ρ(ˆα)), which is approximated by the numerical flux ˆGρ(ˆαi, ˆαi+1) = ˆFρ(ρ(ˆαi), ρ(ˆαi+1)).

To express now ˆGρδ +ˆα as an exact difference, we make use of the primitive function ψ(ˆα) defined by


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq097.png)

Using the Integral Mean Value Theorem, we can write


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq098.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq099.png)

where ˆα∗is a unknown value between ˆαi and ˆαi+1 and the value ˆgρ(ˆα∗) is a second-order approximation of ˆgρ(ˆαi+ 1

2 ).

Eq. (19) suggests the adoption of the value ˆGρ = ˆgρ(ˆα∗), i.e.


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq100.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq101.png)

which, expressed in the variables ρi gives:


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq102.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq103.png)

where ψ(ρ) = ψ(ˆα(ρ)). This gives:


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq104.png)

where we used δ +ψi = δ −ψi+1. Eq. (21) reduces to Eq. (18) with the choice ˆFρe = ˆαi+1 ˆFρ −ψi+1. The form of ˆFρe can be made more symmetric by observing that ˆαi+1 = ˆα i + δ+ˆαi/2 which implies


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq105.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq106.png)

2 eventually giving ˆFρe = ˆα i ˆFρ −ψ i. (22) Equations (20) and (22) satisfy Eq. (18), and constitute the basis to build the general fluxes Fρ and Fρe enforcing the PEP property.

To show how the function ψ and the fluxes in Eqs. (20) and (22) can be practically calculated, let us define the function


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq107.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq108.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq109.png)

which is related to α by the equation α = e + ρλ. By using fρ(ρ, u) = ρu the function ψ(ρ) can be calculated as


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq110.png)

where primes denote differentiation and, as usual, ˆλ(ρ) = λ(ρ, U). Eq. (24) allows us to write the fluxes satisfying Eq. (18) in the explicit form


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq111.png)

The final form of Fρ and Fρe can now be obtained in several ways, all reducing to Eq (25) for uniform velocity. In what follows we use the simple extension of Eq. (25) obtained by adopting the arithmetic mean for u, leading to:


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq112.png)

where we defined the average ρ λ = δ+(ρ2 i λi)/δ+αi.

7

PEP and fully conservative discretization for compressible flows of real and TP gases A PREPRINT

3.2 Treatment of the singularity and approximate PEP formulation

Equations (26) and (11) furnish the final formulation satisfying the KEP and PEP properties for an arbitrary EoS, whose details are embodied in the functions α(ρ, p) and λ(ρ, p). As it is typical in these cases, the nonlinear average ρλ in the mass flux is potentially singular for uniform distributions of α. A similar situation occurs, for example, in the EC and PEP (for ideal gases) flux by Ranocha [39], which employs the logarithmic mean (φ log = δ +φ/δ + log φ), and in the EC flux for real gases recently developed in [29], which also uses a potentially singular flux for uniform distributions of temperature. This phenomenon, which is fundamentally linked to the definition of the average through the use of the integral mean value theorem [40], can cause severe limitations in the applications, especially when large regions of uniform distributions of temperature or density are expected to occur. To avoid the singularity, suitable fixes can be devised, typically implemented by locally reverting to non-singular numerical fluxes when the denominators of the singular averages (δ +αi in our case) fall under a specific tolerance. These non-singular schemes can be either a suitable Taylor expansion of the original mean, when possible, as in the case of the logarithmic mean used in entropy conservative methods for ideal gases ([41, 39, 20, 22]) or standard non-singular schemes chosen among those with good performances, to minimize errors, as in [29]. Recently, a more advanced technique based on the theory of discrete gradient operators has been also used [42].

In the present case, since a full series expansion of the singular mean ρλ appearing in the mass flux Fρ seems cumbersome for an arbitrary EoS, we choose to use the simple modification of the flux Fρ in Eq. (26) obtained by adopting the arithmetic mean ρ in place of ρλ, i.e. by using the fluxes


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq113.png)

The set of fluxes given by Eqs. (11) and (27) actually shows excellent performances on highly challenging tests, as reported in Section 6. These results tempted us to use it not only as a fix for the exact PEP formulation given by Eq. (26), but also as an approximate PEP formulation which can be used during the whole simulation, irrespective of the local value of δ +αi. Similar very good performances have also been observed by using other classical algebraic means for the density in the mass flux (e.g. the harmonic or geometric means), suggesting that the form of the interpolation for the density in the mass flux has only marginal effects on the enforcement of the PEP property. For the sake of simplicity, we adopt here the arithmetic mean, leaving a thorough analysis of this subject as a possible future work. In conclusion, although only approximately PEP, the excellent performances of the formulation (11)–(27) induce us to offer it as a sufficiently simple formulation which could be used in place of Eqs. (11)–(26). We will refer to the Exact PEP formulation (EPEP-RG) for the scheme given by Eqs. (11)–(26), and to the Approximate PEP formulation (APEP-RG) for that given by Eqs. (11)–(27).

As a final comment, we observe that in the case of a calorically perfect gas, for which ρe = p/(γ −1), the function α is identically zero, and the whole derivation exposed in Section 3.1, that led to Eq. (26), breaks down. However, even if the mass flux in Eq. (26) remains undefined, the internal energy flux can be safely determined, and turns out to reduce to Fρe = ρe u , which is the internal energy flux of the formulation KEEPPE by Shima et al. [18]. Hence, the APEP-RG formulation defined by the fluxes in Eq. (27) nicely reduces to the KEEPPE formulation in the case of a calorically perfect gas.


### 4 Thermally perfect gases

In this section, we work out the particular case of a thermally perfect gas model, for which the usual perfect-gas EoS is assumed: p = ρRT , where R is the gas constant and T is the absolute temperature. The internal energy depends on temperature through temperature-variable isochoric specific heat capacity cv(T ), which implies


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq114.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq115.png)

and the “ref" subscript indicates some reference condition. We use a polynomial-based approach for the modeling of cv(T ), [43], by which the isochoric specific heat is expressed using temperature-based polynomial fittings:


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq116.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq117.png)


![Equation](images/[2026] PEP + APEC + KEEP 비교_eq118.png)

This functional dependence is widely used when thermal equilibrium is assumed [43], and is also easily tractable from an analytical point of view. We will detail the derivation for the general formulation with arbitrary N, although only

8

