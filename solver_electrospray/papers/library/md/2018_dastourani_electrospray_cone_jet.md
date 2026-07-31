Contents lists available at ScienceDirect


# International Journal of Heat and Fluid Flow

journal homepage: www.elsevier.com/locate/ijhff


# A physical insight into electrospray process in cone-jet mode: Role of operating parameters


## H. Dastourania, M.R. Jahannamab,⁎, A. Eslami-Majdc

a Aerospace Research Institute, Mahestan Street, Sana't Square, Tehran, Iran b Sprays Research Laboratory, Iranian Space Research Center, Sheikh Fadhlullah Highway, Tehran, Iran c Electrical and Electronics Engineering Department, Malek Ashtar University of Technology, Tehran, Iran

A R T I C L E I N F O

Keywords: Cone-jet mode Electrospray Electric potential Flow rate Numerical simulation

A B S T R A C T

This article investigates the formation of cone-jet structure in an electrospray process based on a two-phase numerical simulation. The numerical approach takes account of the coupled governing equations of fluid flow and electrostatics in conjunction with the charge conservation equation and a VOF interface tracking method on the basis of a CSF model. The temporal and spatial evolutions of the cone-jet mode are examined in connection with the operating parameters, i.e. liquid flow rate and electric potential. Under the influence of these parameters, this study elucidates the physical aspects of the geometrical growth and extension along with the electric charge dispersion within the cone-jet structure. Furthermore, the flow patterns developed in the two-phase flow are studied revealing how orderly the operating parameters can alter the flow configuration. The results are compared with experimental data indicating good agreements, which, in turn, confirm the effectiveness of the simulation methodology concerning the electrospray phenomenon.

1. Introduction

Electrohydrodynamics (EHD) would be deemed as a branch of fluid mechanics that deals with the effects of electrical forces on liquids (Castellanos, 1998). In this context, electrospray can be regarded as that part of the EHD, which is especially involved with the electrical charging of liquids for the generation of liquid droplets. A typical electrospray arrangement consists of two major elements, i.e. emitter and electrode, held at different electric potentials. The main aspect of the electrospray concerns the liquid flow deformation at the emitter exit acquiring a conical structure referred to as a Taylor cone (Taylor, 1964). When the apex of the Taylor cone emits a jet of liquid leading to the breakup and generation of droplets, this is termed a cone-jet mode of the electrospray operation. The electrospray process is used in a variety of applications, a few of which include mass spectrometry as an ionization technique (Fenn et al., 1989; Chetwani et al., 2010; Banerjee and Mazumdar, 2012), electrospinning for nanofiber production (Yu et al., 2008; Agarwal et al., 2013; Ghelich et al., 2016), surface coating based on accurate deposition (Salata, 2005; Jaworek and Sobczyk, 2008; Yoon et al., 2011; Sweet et al., 2014) and electrohydrodynamic printing (Park et al., 2007). The functioning of an electrospray process is dependent on various factors. These factors would be divided into four groups

comprising operating parameters, physical properties, geometrical features and surrounding conditions. Depending on the factors, especially the operating parameters inherent in the liquid flow rate and electric potential, the electrospray process would take different modes. The diversity of the modes may encompass the states of dripping, micro dripping, spindle, multiple spindle, oscillating-jet, precession, cone-jet and multi-jet (Jaworek and Krupa, 1999a, b; Cloupeau and PrunetFoch, 1990). Among the electrospray modes, the cone-jet is the most important and widely used mode since it is approved as a very useful technique to generate the monodisperse sprays with droplet diameters in the range of tens of nanometers to hundreds of microns depending on the liquids used (Cloupeau and Prunet-Foch, 1989; Chen et al., 1995; GameroCastano, 2008). However, the formation of the cone-jet mode requires minimum magnitudes of the electric potential and the liquid flow rate. The minimum electric potential, namely the onset voltage, can be estimated as given by (Morris et al., 2013),

⎜ ⎟ = ⎛ ⎝

⎞ ⎠


![Equation](images/2018_dastourani_electrospray_cone_jet_eq001.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq002.png)

where de and L, respectively represent the emitter diameter and the emitter to ground electrode distance, γ is the surface tension coefficient and ɛ0 denotes vacuum permittivity equal to 8.854 × 10−12 CV−1m−1.

https://doi.org/10.1016/j.ijheatfluidflow.2018.02.012 Received 1 November 2017; Received in revised form 2 February 2018; Accepted 22 February 2018

⁎ Corresponding author. E-mail address: m.jahannama@isrc.ac.ir (M.R. Jahannama).

International Journal of Heat and Fluid Flow 70 (2018) 315–335

0142-727X/ © 2018 Elsevier Inc. All rights reserved.

T

In addition, the minimum liquid flow rate would be estimated using the following relation (Rosell-Llompart and De La Mora, 1994);

= Q γ ρK ɛ ɛ min r 0


![Equation](images/2018_dastourani_electrospray_cone_jet_eq003.png)

where ρ, ɛr and K are the density, relative permittivity and electrical conductivity of liquid, respectively. The research work on electrospray can be divided into experimental and theoretical studies. The experimental work of Zeleny (1914, 1917) would be acknowledged as the first systematic study on the electrospray whereon the following research studies to-date are based and evolved (Taylor, 1969; De La Mora and Loscertales, 1994; Gañán-Calvo et al., 1997; López-Herrera et al., 2004; Yu et al., 2016). In contrast, the theoretical study of Taylor (1964) can be distinguished as a leading methodical attempt that founded a robust basis for the subsequent analytical and numerical investigations until present. Although the succeeding theoretical efforts in the electrospray initially focused on analytical methods, particularly oriented towards the instability of liquid jets (Chaudhary and Redekopp, 1980; Setiawan and Heister, 1997; Cherney, 1999; Hartman et al., 2000), this was the numerical simulation which has drawn the attention of research studies during the recent decades mainly owing to the tremendous advancements achieved in computing facilities. Nevertheless, it seems that the first numerical simulations on electrically charged liquids can be tracked back to three decades ago with a main focus on the formation of a stable liquid meniscus from which no jet was emerged. In fact, this viewpoint would be thought as a zero flow rate limit for the cone-jet mode that does not encounter the singularity at the cone apex due to the liquid jet emanation. In this connection, Joffre et al. (1982) initiated an axisymmetric equilibrium approach based on a balance among the forces arising from the surface tension, hydrostatic pressure and electric field, which could determine the liquid shape profile. They further extended the model to include the corona discharge from the meniscus surface inserting the space charge effects in the meniscus formation process (Joffre and Cloupeau, 1986). Following on from these works, Pantano et al. (1994) employed the same electrohydrostatic equilibrium strategy by taking account of a small perturbation analysis to overcome the mathematical singularity in the liquid meniscus profile

with a pointed apex. This also led to a profounder study on the electrospray physics by Gañán-Calvo (1997) who described the transition between Taylor cone and jet regions proposing asymptotic universal scaling laws for both the jet size and the issued electric currents. The first simulations of electrically charged jets were inspired by numerical models on uncharged liquid jets (Jeong and Moffatt, 1992; Eggers and Dupont, 1994; Brenner et al., 1997). Hartman et al. (1999b) developed a physical model to simulate the electrospray cone-jet mode based on a steady state one-dimensional axial momentum equation. This equation was established over a balance among the hydrodynamic potential and kinetic sources of energy, tangential electric stress and the dissipation viscous stress, which could ultimately determine the conejet shape. Although they also proposed another model using a Lagrangian approach to predict drop dynamics (Hartman et al., 1999a), the model was not an extension of their cone-jet model and solely relied on the experimental data as the input information.

Yan et al. (2003) simulated the formation of a liquid meniscus and the arising liquid jet due to an electric field in the cone-jet mode. This model would be considered as an extension of the work of Hartman et al. (1999b) to an axisymmetric two-dimensional model based on employing the Navier–Stokes equations and the Gauss law (except for the liquid-gas interface). The interface was dealt with by a current balance to accommodate the jump in the normal electric field.

Lastow and Balachandran (2006) employed the commercial code CFX to numerically model the cone-jet mode using the Navier–Stokes equations in conjunction with the Laplace equation. The simulation did not include the parts of current and conductivity in the governing equations implying the insert of a dielectric body in an electric field with no charge flow. The aforementioned models did not comprise the liquid jet breakup into drops and, thus, appeared to require subsequent extensions with a particular focus on intricacies of the liquid free surface. These goals were pursued by taking account of various numerical schemes developed for the multiphase flows to scrutinize the moving interfaces between the different fluid phases (Puckett et al., 1997; Tryggvason et al., 2001). In this respect, Lim et al. (2011) simulated the cone-jet mode involved with the jet breakup and drop formation based on a two

Nomenclature

C Volume fraction; Ca Capillary number; D32 Sauter mean diameter (m); ddisk Disk diameter (m); di, e Emitter inner diameter (m); di, o Emitter outer diameter (m); ⎯→ ⎯E Electric field vector (Vm−1); ⎯→ ⎯FES Electric force vector (Nm−3); ⎯→ ⎯FST Surface tension force vector (Nm−3); →g Gravity acceleration (ms−2); →J Electric charge flux (Cm−2s−1); K Electrical conductivity (Sm−1); l Characteristic length (m);

− Lc j Cone jet length (m); → n Normal vector; ̂n Unit normal vector; P Pressure (Pa); Q Flow rate (m3s−1); r Radial coordinate (m); Re Reynolds number; te Electric relaxation time (s); tm Magnetic characteristic time (s); → u Velocity vector (ms−1);

We Weber number; z Axial coordinate (m).

Greek symbols

γ Surface tension coefficient (Nm−1); ɛ0 Vacuum permittivity (CV−1m−1); ɛr Relative permitivity; κ Curvature of interface (m); μ Viscosity (Pa s); μm Magnetic permeability (Hm−1); ρ Density (kgm−3); ρe Volume electric charge density (Cm−3); ρS Surface electric charge density (Cm−2); Φ Electric potential (V); χ Taylor number.

Subscripts

− c j Cone-jet surface; g Gas; j Jet; l Liquid; on Onset; VC Vortex center.

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

316

dimensional axisymmetric approach. They utilized a front tracking method to treat the liquid-air interface further to assuming a constant surface charge density on the whole liquid-air interface. The interfacial charge density was chosen on the basis of a trial-and-error method by fitting the numerical data to experimental results to attain the best suitability. This assumption highly simplified the charge transport phenomenon on the interface degrading the accuracy of the electric model established.

Herrada et al. (2012) proposed a numerical scheme to simulate a steady axisymmetric electrospray cone-jet mode excluding the jet breakup. They implemented a tracking method by setting an equilibrium condition for tangential and normal stresses at the liquid-air interface. This led to an interface shape function, and in turn, to explicit relations of the electric field components in terms of that function. The electric charge was assumed to be confined to the liquid free surface implying a zero space charge density, which notably simplified the computations. The results were verified in comparison with an open source code, i.e. GERRIS, which was developed and extended by LópezHerrera et al. (2011) to simulate electrohydrodynamic processes using Volume of Fluid (VOF) tracking method. Furthermore, Ferrera et al. (2013) employed the similar approach to simulate the dynamical behavior of electrified pendant drops, i.e. a zero flow rate electrospray process, using the GERRIS whose results found good agreement compared with experimental data.

Wei et al. (2013) simulated the electrospray cone-jet mode including the jet breakup using the OpenFOAM as an open source computational fluid dynamics (CFD) software toolbox. The simulation was carried out using an unsteady approach towards the fluid flow governing equations while the liquid-gas interface was treated using the VOF model. However, the role of electric field was treated using a leaky-dielectric model, which simplified the charge conservation to a steady state equation.

Xu et al. (2013) developed a numerical model to simulate the electrospray cone-jet mode for a core-shell configuration using the commercial code Fluent. The interfaces of fluids were tracked using the VOF method while a modified leakydielectric model was proposed to determine the interface charge density. However, the modified model was based on a tuning factor, which was tailored according to a trialand-error procedure in order to provide the best fit with their experimentally observed cone-jet profile. Although this work (within a group study) was further employed by Davoodi et al. (2015) and Yan et al. (2016) to examine impacts of the nozzle tip configurations on the flow formation, no essential development would be realized in the overall structure of the simulation approach. The present study is concerned with the numerical simulation of electrospray process in the cone-jet mode and the subsequent jet breakup into drops. The study is intended to elucidate the underlying physics involved with the process via the role of operating parameters. This is carried out in a transient axisymmetric two dimensional framework taking simultaneous account of the fluid and electrical coupled governing equations. In particular, the charge transfer process is surveyed using the full charge conservation equation with no simplification. The study is introduced and discussed in two main parts. In the first part, the fluid flow, interface tracking and electrical charging are mathematically formulated and the solution method, computational domain and grid dependency analyses are described. In the second part, the results are given and the effects of liquid flow rate and electric potential on the electrospray structure are investigated further to being validated.

2. Mathematical formulation

2.1. Fluid flow

Computational approach for simulating the electrospray process

needs solution of the governing equations on fluid flow (conservation of mass and momentum) along with the liquid-gas interface tracking. The mass conservation equation for an incompressible fluid flow is given by (Lim et al., 2011);


![Equation](images/2018_dastourani_electrospray_cone_jet_eq004.png)

where → u is the fluid field velocity vector. The Navier–Stokes equations including the forces of surface tension on the liquid-gas interface (⎯→ ⎯FST), electrical charging of liquid (⎯→ ⎯FES) and gravity can be expressed by (Ouedraogo et al., 2017)

⎡


![Equation](images/2018_dastourani_electrospray_cone_jet_eq005.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq006.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq007.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq008.png)

where P is the pressure, →g the gravity vector, while ρ and μ denote the

fluid density and viscosity, respectively. Moreover, ⎯→ ⎯FES and ⎯→ ⎯FST represent the electric and surface tension forces, respectively. The electric force in Eq. (4) can be given as follows (Tomar et al., 2007):


![Equation](images/2018_dastourani_electrospray_cone_jet_eq009.png)

2


![Equation](images/2018_dastourani_electrospray_cone_jet_eq010.png)

where ⎯→ ⎯E is the electric field, ρe the volume charge density and ε the permittivity of fluid ( = ɛ ɛ ɛ r 0 with ɛ0 as vacuum permittivity and ɛr as


> **Fig. 1. Flowchart diagram of solution.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

317

relative permittivity). The first term on the right hand side of Eq. (5) is the Coulomb force, which implies the interaction between the electric charges acting in the direction of the electric field. The second term represents dielectric force that due to ∇ɛ acts perpendicular to the interface. The surface tension force can be determined using the continuum surface force (CSF) model introduced by Brackbill et al. (1992). In the CSF model, the surface tension is presented as a volumetric force that based on a constant surface tension coefficient would be stated as follows:

⎯→ ⎯ = → F γκn ST (6)

where κ is the curvature of interface and → n a normal vector to the interface. According to Eq. (6), the representative force of surface tension operates perpendicular to the interface of liquid-gas. In the CSF model, the curvature of the interface may be defined by = −∇ κ n · where ̂n is a unit normal vector to the interface and is defined by ̂ = →→ n n n / . Thus, Eq. (6) can be rewritten as;


![Equation](images/2018_dastourani_electrospray_cone_jet_eq011.png)

In the CSF model, the interface between the phases is assumed as a very thin layer with the order of computational grid through which physical properties change smoothly from one phase to another. If the characteristic function representative of the interface is denoted by ∼C, the normal vector to the interface can be defined as →= ∇∼ n C. Accordingly, the curvature of interface and, in turn, the surface tension force can be calculated. The characteristic function related to the interface tracking method used in the present study is based on the volume of fluid (VOF) method.

2.2. Interface tracking

In a two-phase flow, the interface between the phases is moving. The interface can be tracked using different methods among which the VOF (Hirt and Nichols, 1981; Youngs, 1982; Gueyffier et al., 1999) and Level Set (LS) (Osher and Sethian, 1988; Sussman et al., 1994; Sussman et al., 1998; Sussman et al., 1999) would be considered as the most common methods used in research studies. As cited above, the VOF

method is used for the interface tracking in this study. The VOF method is based on a volume fraction C where C = 0 for the cells filled with gas, 0 < C < 1 for the cells filled with both gas and liquid and C = 1 for the cells filled with liquid. The volume fraction C is a scalar function whose transport equation in a standard form is as follows (Hirt and Nichols, 1981):


![Equation](images/2018_dastourani_electrospray_cone_jet_eq012.png)

Since the mass and volume properties are interchangeable in an incompressible twophase flow, this makes the use of volume conservation of any fluid as an important advantage in the VOF method (Hirt and Nichols, 1981). If the function of volume fraction C is defined as a characteristic function ( = ∼C C) representing the interface of liquidgas, the normal vector would be →= ∇ n C whose substitution in Eq. (7) leads to;


> **Fig. 2. A schematic of (a) physical domain, (b) computational domain.**


> **Table 1 Values of geometrical parameters.**

di,e (mm) do,e (mm) Led (mm) ddisk (mm)

0.12 0.45 30 24


> **Table 2 Physical properties of fluids used in simulations.**


![Equation](images/2018_dastourani_electrospray_cone_jet_eq013.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq014.png)


> **Fig. 3. Maximum magnitude of electric field versus minimum cell size on different z lines.**


> **Fig. 4. Liquid profiles at the emitter exit for various minimum cell sizes.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

318

⎜ ⎟ ⎯→ ⎯ = −⎡ ⎣⎢∇⎛ ⎝

∇ ∇ ⎞ ⎠ ⎤ ⎦⎥∇ F γ C C C · ST


![Equation](images/2018_dastourani_electrospray_cone_jet_eq015.png)

In implementation of the interface tracking method for a two-phase flow, two immiscible fluids are considered as a single effective fluid in the whole computational domain. Hence, the physical properties of the effective fluid can be calculated by a weighted average of the volume fraction as given below (Tomar et al., 2007);


![Equation](images/2018_dastourani_electrospray_cone_jet_eq016.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq017.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq018.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq019.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq020.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq021.png)

K K K C

C


![Equation](images/2018_dastourani_electrospray_cone_jet_eq022.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq023.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq024.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq025.png)

l g

l g

l g


![Equation](images/2018_dastourani_electrospray_cone_jet_eq026.png)

2.3. Electrical equations

To calculate the electric force (⎯→ ⎯FES) in Eq. (4), the electric field due


> **Table 3 Boundary conditions.**

Boundary Type Variable


![Equation](images/2018_dastourani_electrospray_cone_jet_eq027.png)

1–6 Inlet flow ∇ = P 0 = = u u Q πd 0 4 /( ) r z i e, 2 C = 1 ∇ = ρ 0 e ∇ = Φ 0

6–7, 7–8, 8–9 Emitter wall ∇ = P 0 ur = uz = 0 ∇ = C 0 ∇ = ρ 0 e Φ = Φ0 3–4, 4–5, 5–9 Free-stream Total pressure* ∇→= u 0 Inlet-Outlet** ∇ = ρ 0 e ∇ = Φ 0

2–3 Ground electrode ∇ = P 0 ur = uz = 0 ∇ = C 0 ∇ = ρ 0 e Φ = 0 1–2 Axis of symmetry ∇ = P 0 n ur = 0 ∇ = C 0 n ∇ = ρ 0 n e ∇ = Φ 0 n

⁎ This is a fixed value condition calculated from specified total pressure p0 and local velocity → u . ** This is a zero gradient condition when flow is outwards and is a fixed value when flow is inwards.


> **Fig. 5. Typical patterns of equipotential (color lines) and electric field lines (arrowed black lines) in the computational domain for Q = 3.4 mLh−1 and Φ0 = + 4000 V at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

319

to electric potential difference between the emitter-disk electrodes should be determined. In an electrohydrodynamic process, the magnetic effects can be ignored because the characteristic time for the magnetic phenomena = t μ Kl m m (μm denotes the magnetic permeability and l the characteristic length) is several orders of magnitude smaller than the characteristic time for the electric phenomena, i.e. the electric relaxation time = t K ɛ/ e . Therefore, the electrical aspects of the phenomena can be described by (Tomar et al., 2007; Saville, 1997):

∇ ⎯→ ⎯ = E ρ ·(ɛ ) e (11)

Moreover, the negligible magnetic effects refer to an irrotational electric field represented mathematically by ∇× ⎯→ ⎯ = E 0. Hence, the electric field, ⎯→ ⎯E , can be expressed as a gradient of the electric potential, i.e. ⎯→ ⎯ = −∇ E Φ where Φ denotes the electric potential. Thus, Eq. (11) can be converted to Poisson equation as follows:


![Equation](images/2018_dastourani_electrospray_cone_jet_eq028.png)

On the other hand, the conservation law for the electric charge necessitates consideration of the relevant governing equation as stated by (Tomar et al., 2007);


![Equation](images/2018_dastourani_electrospray_cone_jet_eq029.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq030.png)

t J· 0 e


![Equation](images/2018_dastourani_electrospray_cone_jet_eq031.png)

where →J is the electric charge flux as defined by (López-Herrera et al.,

2011);


![Equation](images/2018_dastourani_electrospray_cone_jet_eq032.png)

Applying the vector differential operator to Eq. (14) and using ⎯→ ⎯ = −∇ E Φ, one can convert Eq. (13) to the electric charge conservation equation as follows:


![Equation](images/2018_dastourani_electrospray_cone_jet_eq033.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq034.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq035.png)

In this equation, the second and third terms represent the convection and conduction of electric charge, respectively.

3. Solution method

The simulation of electrospray process needs the simultaneous solution of the coupled fluid flow and electrical equations discussed in the previous section. For this purpose, the OpenFOAM as an open source CFD software package is utilized in the present study. The liquid-gas interface tracking is performed based on the VOF method by using the InterFoam solver (Deshpande et al., 2012; Klostermann et al., 2013; Raees et al., 2011; Emad, 2014) as a powerful tool for the incompressible two-phase flow computations in the OpenFOAM. This should be noted that the volume fraction equation in this solver is different from its standard form introduced in Eq. (8) and is given by


> **Fig. 6. Temporal Cone-jet formation for Φ = + 4000 V and (a) Q = 0.25 mLh−1, (b) Q = 0.50 mLh−1, (c) Q = 0.80 mLh−1, (d) Q = 0.93 mLh−1, (e) Q = 1.20 mLh−1.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

320

(Rusche, 2003);

∂ ∂ + ∇→ + ∇→ − = C t u C u C C ·( ) ·[ (1 )] 0 r (16)

where →= →−→ u u u r l g is the vector of relative velocity between the fluid

phases. Since the interFoam is a hydrodynamic two phase flow solver, it does not include the electrical equations. Thus, after arranging the background conditions, the electrical part (Eqs. (12) and (15)) has been added on to the solver and, thus, the Navier–Stokes equations in


> **Fig. 7. Temporal Cone-jet formation for Φ = + 4000 V and (a) Q = 2.60 mLh-1, (b) Q = 3.40 mLh-1, (c) Q = 4.90 mLh-1, (d) Q = 6.20 mLh−1, (e) Q = 10.0 mLh−1.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

321


> **Fig. 7. (continued)**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

322

accordance with Eq. (4) have been modified. The solution procedure used in the current simulation follows the flowchart as shown in Fig. 1 and has the main features as given below:

1. Initial and boundary conditions, e.g. liquid flow rate, electric potential, volume fraction distribution and physical properties, are applied to the computational domain. 2. To determine the volume fraction as well as the correction of effective physical properties, the volume fraction equation (Eq. (16)) is solved. 3. To compute the electric force, Eqs. (12) and (15) are solved.

4. To determine the velocity field in the computational domain, the Navier–Stokes equations are solved. It should be noted that the coupled fields of velocity and pressure are solved using the PIMPLE algorithm which is a combination of SIMPLE and PISO algorithms. 5. Since the numerical solution is performed in a transient state, the time step is recalculated based on the maximum Courant number in each step. The solution continues to achieve a sustainable process.

4. Solution domain and conditions

The solution domain and conditions used in the present simulation


> **Fig. 8. Temporal distribution of electric charge density for Φ = + 4000 V and (a) Q = 0.25 mLh−1, (b) Q = 0.93 mLh−1, (c) Q = 2.60 mLh−1, (d) Q = 4.90 mLh−1.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

323


> **Fig. 9. Distribution of electric charge density for various liquid flow rates and Φ = + 4000 V at t = 1.2 ms.**


> **Fig. 10. Radial distributions of (a) volume fraction and (b) electric charge density in z = 0.03 mm for various liquid flow rates and Φ = + 4000 V at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

324

are introduced in this section. The geometrical configurations of electrodes as well as the physical properties employed in this study are corresponding to the experimental work of Tang and Gomez (1996). Fig. 2 shows schematics of the physical and computational domains. In


> **Fig. 2 parameters di,e, do,e, Led and ddisk denote emitter inner diameter, emitter outer diameter, emitter to disk distance and disk diameter, respectively, and their values are given in Table 1. Working fluids are considered Heptane as the liquid and air as the gas whose physical**


> **Fig. 11. Cone-jet shapes for various liquid flow rates and Φ = + 4000 V at t = 1.2 ms.**


> **Fig. 12. Variations of acting forces on liquid-gas interface versus liquid flow rate for Φ = + 4000 V at t = 1.2 ms on (a) z = 0.03 mm and (b) z = 0.05 mm.**


> **Fig. 13. Variations of cone-jet dimensions based on various liquid flow rates for Φ = + 4000 V at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

325

properties are given in Table 2. In the present study, the simulations are performed in an axisymmetric state. The computational domain is discretized using a structured and non-uniform grid so that the minimum grid size is positioned at the emitter exit. To examine the grid independency on the solution data, seven different mesh sizes with the minimum cell size 1, 1.5, 2, 2.5, 3, 5 and 10 µm are tested, respectively. The corresponding simulations are performed for a flow rate of 12 mLh−1 and an electric potential of + 4000 V. The maximum electric field versus the minimum cell size is plotted in Fig. 3 for different z lines. Moreover, liquid profiles in the emitter exit for the various minimum cell sizes are shown in Fig. 4. According to these figures, the mesh with the minimum cell size of 2 µm is chosen for the following simulations. For the selected mesh spacing, the total number of grids is 735,050. The boundary conditions with a reference to Fig. 2, (b) are treated as presented in Table 3. Since the simulations are carried out in a transient situation, all dependent variables, i.e. velocity, pressure, electric potential and charge density, are set to zero at the starting time of liquid injection into the domain. At this time, the emitter is assumed to be fully filled with the liquid, i.e. C = 1, whereas rest of the domain is filled with the air, i.e. C = 0.

5. Results and discussion

The simulation results of electrospray process are presented in this section. It is worth mentioning that the present study is built upon a numerous number of simulation cases of which a proportionate number of pertinent cases is reported.

The equipotential and electric field lines calculated in this study exhibit a typical pattern as shown in Fig. 5. The magnified part in the figure provides a closer view into the proximity of the emitter wherein the creation of electric field in the domain can be seen in conjunction with the liquid flow development. Thus, all the results presented in the following sections are involved with the similar patterns of the electric field and potential.

5.1. Flow rate effects

In this part, effects of the liquid flow rate on the electrospray conejet formation are studied. Referring to Eqs. (1) and (2), the onset voltage and minimum flow rate required for the formation of a cone-jet mode can be estimated. Using the geometrical and physical properties tabulated in Tables 1 and 2, the corresponding onset voltage and minimum flow rate are found to be +2452 V and 1.20 mLh−1, respectively. In this regard, the simulations associated with the liquid flow rate are performed based on various liquid flow rates at a constant electric potential +4000 V.


> **Fig. 6 depicts a temporal view of the electrospray process for the liquid flow rates smaller than and equal to the estimated minimum flow rate. According to the figure, the liquid flow emerged from the emitter steadily being affected by the electric field, initially shaping a meniscus progressing towards a convex conical shape from whose apex fine droplets are ejected. More increase in the flow rate converts the meniscus to a concave conical shape in addition to stretching it out. However, the electrospray process is short of producing a full cone-jet mode at the flow rates lower than 0.93 mLh−1 implying insufficiency of these flow rates for the jet emergence. Since the evolution of electrospray process attains a cone-jet structure at 0.93 mLh−1, this reveals a lower limit than the estimated minimum flow rate. This would lie in the derivative conditions of Eq. (1), which arguably covers the liquids with K > 10−4 Sm−1 (Chen and Pui, 1997). This should be noted that the main goal of using Eq. (1) in this study is to gain an estimate of the minimum flow rate in order to quantitatively tune the parameter around the cone-jet point formation. In this regard, the results confirm appropriateness of the estimation predicting a right magnitude of order for the formation flow rate.**


> **Fig. 7 shows the temporal process of cone-jet formation for the flow rates larger than the minimum flow rate required for the cone-jet formation. Based on this figure, the evolution of liquid flow, in addition to comprising the main features of the process in Fig. 6, directs towards the jet emergence from the cone apex by stretching it down to the point**


> **Fig. 14. Temporal variation of cone-jet length for various liquid flow rates at Φ = + 4000 V.**


> **Fig. 15. Variation of cone-jet length versus liquid flow rate for Φ = + 4000 V at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

326

wherein the ultimate instability of the jet occurs resulting in a chain of droplet issuance from its tip. This confirms the sufficiency of the liquid flow rate causing a longer integrity of the liquid flow under the disintegrating effects of the electric field, which leads to the full

establishment of the cone-jet mode.

Figs. 8 and 9, in accordance with Figs. 6 and 7, show typical temporal and steady distributions of electric charge density, respectively. In these figures, it can be seen that the electrical charging of liquid flow leads to accumulation of the electric charge on a thin layer on the surface of the liquid (on interface between liquid and gas). Since the liquid used in this simulation is considered as a conducting liquid, i.e. K = 1.4 × 10−6 Sm−1 > 10−8 Sm−1 (Barrero et al., 1999), this agrees well with the charging of conducting liquids in which the electric charge would migrate to the surface layer of the liquid. This is also shown in Fig. 10 that the electric charge accumulation is limited to the buffer zone between the liquid and gas phases leaving the bulk of the liquid electrically neutral.


> **Fig. 11 depicts the shapes acquired by the liquid flow for different flow rates at a certain time instant, i.e. t = 1.2 ms. It is clear from the figure that an increase in the flow rate by emerging a greater amount of liquid from the emitter forms a conical profile with a longer jet tail. Although an increase in the flow rate initially transforms the convex conical meniscus to a concave one as viewed in Fig. 11, this trend converts towards a straight cone generator at the larger flow rates. In this regard, the variations of three main forces acting on the liquid-gas surface are shown in Fig. 12 for two different axial positions. According to this figure, the increase in the flow rate causes an increase in the force of hydrodynamic pressure whereas it results in a decrease in the**


> **Fig. 17. Velocity of liquid-gas interface versus cone-jet length for various liquid flow rates and Φ = + 4000 V at t = 1.2 ms.**


> **Fig. 18. Total surface charge density on cone-jet interface versus flow rate at t = 1.2 ms.**


> **Fig. 19. A typical pattern of Streamlines formed in whole physical domain for Q = 6.20 mLh−1 and Φ = + 4000 V at t = 1.2 ms.**


> **Fig. 16. Electric charge density on liquid-gas interface versus cone-jet length for various liquid flow rates and Φ = + 4000 V at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

327

surface tension and electrostatic forces. In fact, the increase in the liquid flow rate originated from the rising trend of the hydrodynamic pressure weakens the surface tension and electrostatic effects expanding the cone structure and, in turn, leading to the straight cone generator at the larger flow rates. It should be noted that the hydrodynamic pressure mainly acts as a creating force for the fluid motion and, thus, the cone expansion is a lateral effect of this force with an order of magnitude higher than the equal orders of surface tension and electrostatic forces. In addition, the increase in the flow rates lower than the minimum flow rate largely suppresses the surface tension effects causing this force to approach to the electrostatic force. This implies a twofold role for the liquid flow rate including that its rise at small magnitudes results in the formation of the cone-jet structure and at large magnitudes in the expansion of that structure.


> **Fig. 13, corresponding to Fig. 11, shows the variation of the cone-jet radius versus the cone-jet length providing a quantitative view into the cone-jet dimensions. It is obvious that the increase in the liquid flow rate in addition to thickening the cone-jet radius elongates the jet by transferring its instability, i.e. surface undulations, further downstream**

leading to a higher cone-jet length as drawn in Figs. 14 and 15. This implies that an increase in the flow rate by inflicting a larger amount of liquid to a constant electric potential necessitates more liquid flow development to reach the electrohydodynamic instability required for


> **Fig. 20. Patterns of streamlines formed in the emitter exit for various liquid flow rates and Φ = + 4000 V at t = 1.2 ms.**


> **Fig. 21. Displacement of vortex center versus liquid flow rate for Φ = + 4000 V at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

328

the jet disintegration.


> **Fig. 16 shows the variation of electric charge density along the liquid-gas interface using various liquid flow rates for a constant electric potential at t = 1.2 ms. As seen in the figure, the electric charge density exhibits an ascending-descending trend along the cone-jet interface on which the maximum charge density coincides with the cone-jet neck having the highest concentration of electric charge. Fig. 17 corresponding to Fig. 16 depicts the velocity of the liquid-gas interface along the cone-jet structure. The figure shows that an increase in the flow rate leads to a reduction in the interface velocity. This is associated with the lower levels of surface charge density at the larger flow rates, as shown in Fig. 18, which in turn decelerates the flow owing to the electrostatic effect.**


> **Fig. 19 illustrates a typical pattern of streamlines formed in the whole physical domain while Fig. 20 provides a closer insight into the streamlines at the emitter exit. In general, the formation of streamlines in the surrounding air stems from the liquid flow in which the accumulation of electric charge on its surface accelerates the liquid-gas interface by the electric field. This leads to the creation of two vortices including a clockwise vortex within the liquid meniscus (Fig. 20) and a counter-clockwise one in the surrounding air (Fig. 19). Fig. 20 also shows that an increase in the liquid flow rate after the establishment of the cone-jet structure, i.e. Q > 0.93 mLh−1, in addition to altering the cone profile from a convex shape to the concave and straight cones as mentioned before, shrinks the vortex within the meniscus by**

contracting and shifting it further downstream as also quantitatively is shown in Fig. 21. Moreover, Fig. 22 shows the radial variation of axial velocity on a plane passing through the center of a center. According to this figure, the shrinkage of a vortex at the larger flow rates is accompanied with the higher velocities at the liquid-gas interface since the vortex takes place at a far-off axial section compared with a lower flow rate.


> **Fig. 23 shows the distribution of Sauter mean diameter (D32) of droplets against the liquid flow rate for the present simulation in comparison with the work of Tang and Gomez (1996). As seen in the**


> **Fig. 22. Radial variation of axial velocity on a sectional plane passing through the vortex center for various flow rates and Φ = + 4000 V at t = 1.2 ms.**


> **Fig. 23. Variation of Sauter mean diameter of droplets versus liquid flow rate for Φ = + 4000 V and t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

329

figure, the simulation results agree well both qualitatively and quantitatively with the experimental data. This confirms the validity of the results and effectiveness of the simulating model used in this study. Moreover, Fig. 23 shows that the increase in the liquid flow rate results in the larger droplets since the electric potential as the main source of liquid instability and disintegration is held constant. In the present study, the attention is also paid to the simulation

results in connection with the axisymmetric model. This necessitates to ensure that the formation of the stable cone-jet mode acquired on the flow rates leading to the jetting-dripping transition is valid under the axisymmetric approach used. In this respect, an analysis would be taken into account based on the dimensionless numbers as given by (GañánCalvo and Montanero, 2009),


> **Fig. 24. Reynolds number versus Capillary number for jets in various liquid flow rates.**


> **Fig. 25. Temporal Cone-jet formation for Q = 3.4 mLh−1 and electric potentials (a) Φ = + 2000 V, (b) Φ = + 2250 V, (c) Φ = + 2452 V.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

330


> **Fig. 26. Temporal Cone-jet formation for Q = 3.4 mLh−1 and electric potentials (a) Φ = + 3000 V, (b) Φ = + 4000 V, (c) Φ = + 5000 V.**


> **Fig. 27. Cone-jet shapes for various electric potentials and Q = 3.4 mLh−1 at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

331

=

= =

= =

=


![Equation](images/2018_dastourani_electrospray_cone_jet_eq036.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq037.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq038.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq039.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq040.png)


![Equation](images/2018_dastourani_electrospray_cone_jet_eq041.png)

We Re Ca

ɛ

·

j j

j j j

j

j j

j

j j j

0 0 2

2


![Equation](images/2018_dastourani_electrospray_cone_jet_eq042.png)

where χj, Rej, Caj and Wej denote Taylor, Reynolds, Capillary and Weber numbers, respectively. Gañán-Calvo and Montanero (2009) and López-Herrera et al. (2010) collected and analyzed a considerable number of experimental data from the literature for various liquids electrosprayed or electrospun in a stable and steady cone-jet mode. They also developed and validated an analytical model showing that there are theoretical critical curves in the Rej-Caj plane above which a stable and steady cone-jet would be shaped. In this regard, Fig. 24 compares the present simulation data with the critical curve limits showing that the results are located above the curves. This confirms that the stable and steady cone-jet structures attained in this study for 0.93 ≤Q ≤12.0 mLh−1 at a constant electric potential, with Weber and Taylor numbers ranged in 2.1 ≤Wej ≤46.7 and χj < 0.003, are in agreement with the instability limits proposed in the literature.

5.2. Electric potential effects

In this section, effects of the electric potential on the cone-jet formation and droplet size have been studied. As stated before, using the


> **Fig. 28. Cone-jet dimensions for various electric potentials and Q = 3.4 mLh−1 at t = 1.2 ms.**


> **Fig. 29. Magnitude of electric field against electric potential for Q = 3.4 mLh−1 at t = 1.20 ms.**


> **Fig. 30. Total length of cone-jet versus electric potential for Q = 3.4 mLh−1 at time instants 0.8 ms and 1.2 ms.**


> **Fig. 31. Patterns of streamlines in emitter exit for various electric potentials for Q = 3.4 mLh−1 at t = 1.2 ms..**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

332

correlations 1 and 2 based on the geometry and liquid properties listed in Tables 1 and 2 results in Ф = 2452 V and Q = 1.20 mLh−1 as the onset voltage and minimum flow rate, respectively. The simulations presented in this section are for the liquid flow rate Q = 3.4 mLh−1 and various positive electric potentials.

Figs. 25 and 26 provide a temporal view of the electrospray process

for various electric potentials at a constant flow rate. According to Fig. 25, it can be seen that the cone-jet formation corresponds to the estimated onset voltage, i.e. 2452 V, which corroborates the suitability of the correlation used. It is clear from the figures that the transient evolution of the electrospray leads to the cone-jet formation in the electric potentials exceeding the onset voltage.


> **Fig. 27 shows the shapes of electrospray for different electric potentials in the constant liquid flow rate at t = 1.2 ms. It is clear from Fig. 27 that the increase in the electric potential converts the liquid meniscus from a convex shape to a right cone, which is then followed by concave cones at the larger electric potentials. This trend is also shown quantitatively in Fig. 28, which illustrates a higher degree of meniscus concavity at the larger electric potentials. This is in connection with a stronger electric field achieved at a larger electric potential, as plotted in Fig. 29, which declines the influence of surface tension by providing a higher curvature to the liquid meniscus.**


> **Fig. 30 gives a quantitative view to the liquid profile length versus the charging electric potential. According to the figure, the liquid profile length exhibits an ascending-descending trend with an increase in the electric potential. This shows that there is an electric potential above which the electrohdyrodynamic instability accelerates the jet breakup leading to shorter cone-jet lengths.**


> **Fig. 31 shows the patterns of streamlines formed in the emitter exit for various electric potentials. As seen in the figure, an increase in the electric potential by elevating the electrostatic effect alters the convex liquid meniscus to a concave profile in addition to confining the vortex core to a smaller area. However, this trend seems to continue to a certain electric potential, e.g. + 3000 V, after which the vortex also being pushed back upstream. Moreover, a comparison between Figs. 20 and 31 reveals that, although an increase in both the liquid flow rate and the electric potential contracts the vortex size, the higher contraction as well as the larger displacement of a vortex only occurs due to a change of the liquid flow rate implying the stronger role of hydrodynamic pressure than the electric potential.**


> **Fig. 32 shows the variation of electric charge density along the liquid-gas interface using various electric potentials for a constant flow rate at t = 1.2 ms. As seen in the figure, the electric charge density exhibits an ascending-descending trend along the cone-jet interface on which the maximum charge density coincides with the cone-jet neck having the highest concentration of electric charge. The figure also show that an increase in the electric potential results in a higher level of the electric charge density owing to the establishment of a stronger electric field as was shown in Fig. 29.**


> **Fig. 33 corresponding to Fig. 32 depicts the velocity of liquid-gas interface along the cone-jet structure. The figure shows that the increase in the electric potential leads to a higher interface velocity, which is associated with the higher levels of the electric charge density at the larger electric potentials, as shown in Fig. 32.**


> **Fig. 34 shows the distribution of Sauter mean diameter of droplets against the electric potential. According to the figure, the droplet size reduces as the electric potential increases, which originates from the creation of a larger electric field by imposing a stronger electrohydrodynamic force on the liquid flow. The simulation results are also evaluated with the experimental data of Tang and Gomez (1996). The comparison reveals good agreement between the simulation and the experiment confirming the utility and validity of the numerical modeling approach used in the study. Fig. 35 in association with Fig. 24 shows the simulation results for various electric potentials at a constant flow rate in comparison with the model of López-Herrera et al. (2010). As seen in the figure, the simulation results (symbols) are located above the theoretical curves indicating that the formation of stable and steady cone-jet structures achieved for 9.5 ≤Wej ≤18.4 and χ < 0.0014 fall into the Rej-Caj region proposed in the literature.**


> **Fig. 33. Velocity distribution on liquid-gas interface versus cone-jet length for Q = 3.40 mLh−1 at t = 1.2 ms.**


> **Fig. 34. Variation of Sauter mean diameter of droplets versus electric potential for Q = 3.4 mLh−1 at t = 1.2 ms.**


> **Fig. 32. Electric charge density on liquid-gas interface versus cone-jet length for Q = 3.40 mLh−1 at t = 1.2 ms.**

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

333

6. Conclusions

A two-phase numerical simulating model was developed to study the electrospray formation in the cone-jet mode taking account of the liquid flow rate and electric potential as the operating parameters. To have a magnitude basis for the parameters, the adoption of two available correlations proposed for the minimum liquid flow rate and onset voltage revealed that the correlations could appropriately predict the establishment of the cone-jet mode. To ensure the validity of the simulation methodology, the numerical results were compared with the existing experimental data, which showed good correspondence asserting the suitable capability of the method for the study of electrospray process. This capability was also evidenced by the stability of the computational procedure, which was attained during the subsequent transient solutions for a substantial number of cases. The evolution of the electrospray structure illustrates that the liquid flow rate and electric potential have minimum values beyond which a cone-jet profile can be shaped. The profiles also indicate that the accumulation of the electric charge occurs in the liquid-gas interfacial layer making the interior of the liquid flow electrically uncharged. In this respect, the electric charge density reveals an ascending-descending trend along the cone-jet interface assigning the maximum concentration to the cone-jet intersection. However, the effect of the electric potential on the electric charge density is more obvious than for the liquid flow rate, which stems from the ascending trend of electric field strength against the electric potential. The results show that the increase in the liquid flow rate leads to the longer jet tails with a transformation of the cone generator from a concave curve to a straight line. However, the larger liquid flow rates are associated with the lower levels of surface charge density, which, in turn, reduce the liquid-gas interfacial velocity. In contrast, the cone-jet length indicates an ascending-descending trend versus the electric potential. The fluid flow patterns reveal the creation of vortices within the liquid flow at the emitter exit. The liquid flow rate and electric potential affect the characteristics of the vortices. The patterns indicate that the sizing and positioning of the vortices are drastically dependent on the liquid flow rate as compared with their moderate dependency on the electric potential. As anticipated, the liquid flow rate and electric potential play opposite roles on the mean droplet size. The validity of axisymmetric approach used in this study is assessed

in association with the simulation results. The analysis of the results with regard to the cone-jet formation in a jetting-dripping transition reveal that the magnitudes of liquid flow rates and electric potentials explored in this study fall in the convective instability region wherein a stable and steady cone-jet structure can be formed.


## References

Agarwal, S., Greiner, A., Wendorff, J.H., 2013. Functional materials by electrospinning of polymers. Prog. Polym. Sci. 38 (6), 963–991. Banerjee, S., Mazumdar, S., 2012. Electrospray ionization mass spectrometry: a technique to access the information beyond the molecular weight of the analyte. Int. J. Anal. Chem. 2012 (1), 1–40. Barrero, A., Gañán-Calvo, A.M., Davila, J., Palacios, A., Gomez-Gonzalez, E., 1999. The role of the electrical conductivity and viscosity on the motions inside Taylor cones. J. Electrost. 47 (1-2), 13–26. Brackbill, J.U., Kothe, D.B., Zemach, C., 1992. A continuum method for modeling surface tension. J. Comput. Phys. 100 (2), 335–354. Brenner, M.P., Eggers, J., Joseph, K., Nagel, S.R., Shi, X.D., 1997. Breakdown of scaling in droplet fission at high Reynolds number. Phys. Fluids 9 (6), 1573–1590. Castellanos, A, 1998. Electrohydrodynamics, second ed. Springer, New York. Chaudhary, K.C., Redekopp, L.G., 1980. The nonlinear capillary instability of a liquid jet. Part 1. Theory. J. Fluid Mech. 96 (2), 257–274. Chen, D.R., Pui, D.Y., 1997. Experimental investigation of scaling laws for electrospraying: dielectric constant effect. Aerosol Sci. Technol. 27 (3), 367–380. Chen, D.R., Pui, D.Y., Kaufman, S.L., 1995. Electrospraying of conducting liquids for monodisperse aerosol generation in the 4 nm to 1.8 µm diameter range. J. Aerosol Sci. 26 (6), 963–977. Cherney, L.T., 1999. Structure of Taylor cone-jets: limit of low flow rates. J. Fluid Mech. 378, 167–196. Chetwani, N., Cassou, C.A., Go, D.B., Chang, H.C., 2010. High-frequency AC electrospray ionization source for mass spectrometry of biomolecules. J. Am. Soc. Mass Spectrom. 21 (11), 1852–1856. Cloupeau, M., Prunet-Foch, B., 1989. Electrostatic spraying of liquids in cone-jet modes. J. Electrostat. 22 (1), 135–159. Cloupeau, M., Prunet-Foch, B., 1990. Electrostatic spraying of liquids: main functioning modes. J. Electrostat. 25 (2), 165–184. Davoodi, P., Feng, F., Xu, Q., Yan, W.C., Tong, Y.W., Srinivasan, M.P., Sharma, V.K., Wang, C.H., 2015. Coaxial electrohydrodynamic atomization: microparticles for drug delivery applications. J. Control. Release 205, 70–82. De La Mora, J.F., Loscertales, I.G., 1994. The current emitted by highly conducting Taylor cones. J. Fluid Mech. 260, 155–184. Deshpande, S.S., Anumolu, L., Trujillo, M.F., 2012. Evaluating the performance of the two-phase flow solver interFoam. Comput. Sci. Discov. 5 (1), 014016. Eggers, J., Dupont, T.F., 1994. Drop formation in a one-dimensional approximation of the Navier–Stokes equation. J. Fluid Mech. 262, 205–221. Emad, V., 2014. Evaluating the Performance of Various Convection Schemes on Free Surface Flows by Using Interfoam Solver. Doctoral dissertation. Eastern Mediterranean University. Fenn, J.B., Mann, M., Meng, C.K., Wong, S.F., Whitehouse, C.M., 1989. Electrospray ionization for mass spectrometry of large biomolecules. Science 246 (4926), 64–71.

Fig. 35. Reynolds number versus capillary number for jets in various electric potentials.

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

334

Ferrera, C., López-Herrera, J.M., Herrada, M.A., Montanero, J.M., Acero, A.J., 2013. Dynamical behavior of electrified pendant drops. Phys. Fluids 25 (1), 012104. Gamero-Castano, M., 2008. The structure of electrospray beams in vacuum. J. Fluid Mech. 604, 339–368. Gañán-Calvo, A.M., 1997. Cone-jet analytical extension of Taylor's electrostatic solution and the asymptotic universal scaling laws in electrospraying. Phys. Rev. Lett. 79 (2), 217–220. Gañán-Calvo, A.M., Davila, J., Barrero, A., 1997. Current and droplet size in the electrospraying of liquids. Scaling laws. J. Aerosol Sci. 28 (2), 249–275. Gañán-Calvo, A.M., Montanero, J.M., 2009. Revision of capillary cone-jet physics: electrospray and flow focusing. Phys. Rev. E 79 (6), 066305. Ghelich, R., Mehdinavaz Aghdam, R., Torknik, F.S., Jahannama, M.R., 2016. Low temperature carbothermal reduction synthesis of ZrC nanofibers via cyclized electrospun PVP/Zr (OPr)4 hybrid. Int. J. Appl. Ceram. Technol. 13 (2), 352–358. Gueyffier, D., Li, J., Nadim, A., Scardovelli, R., Zaleski, S., 1999. Volume-of-fluid interface tracking with smoothed surface stress methods for three-dimensional flows. J. Comput. Phys. 152 (2), 423–456. Hartman, R.P.A., Borra, J.P., Brunner, D.J., Marijnissen, J.C.M., Scarlett, B., 1999a. The evolution of electrohydrodynamic sprays produced in the cone-jet mode; a physical model. J. Electrostat. 47 (3), 143–170. Hartman, R.P.A., Brunner, D.J., Camelot, D.M.A., Marijnissen, J.C.M., Scarlett, B., 1999b. Electrohydrodynamic atomization in the cone–jet mode physical modeling of the liquid cone and jet. J. Aerosol Sci. 30 (7), 823–849. Hartman, R.P.A., Brunner, D.J., Camelot, D.M.A., Marijnissen, J.C.M., Scarlett, B., 2000. Jet break-up in electrohydrodynamic atomization in the cone-jet mode. J. Aerosol Sci. 31 (1), 65–95. Herrada, M.A., López-Herrera, J.M., Gañán-Calvo, A.M., Vega, E.J., Montanero, J.M., Popinet, S., 2012. Numerical simulation of electrospray in the cone-jet mode. Phys. Rev. E 86 (2), 026305. Hirt, C.W., Nichols, B.D., 1981. Volume of fluid (VOF) method for the dynamics of free boundaries. J. Comput. Phys. 39 (1), 201–225. Jaworek, A., Krupa, A., 1999a. Classification of the modes of EHD spraying. J. Aerosol Sci. 30 (7), 873–893. Jaworek, A., Krupa, A., 1999b. Jet and drops formation in electrohydrodynamic spraying of liquids: a systematic approach. Exp. Fluids 27 (1), 43–52. Jaworek, A., Sobczyk, A.T., 2008. Electrospraying route to nanotechnology: an overview. J. Electrostat. 66 (3), 197–219. Jeong, J.T., Moffatt, H.K., 1992. Free-surface cusps associated with flow at low Reynolds number. J. Fluid Mech 241 (1), 1–22. Joffre, G., Prunet-Foch, B., Berthomme, S., Cloupeau, M., 1982. Deformation of liquid menisci under the action of an electric field. J. Electrostat. 13 (2), 151–165. Joffre, G.H., Cloupeau, M., 1986. Characteristic forms of electrified menisci emitting charges. J. Electrostat. 18 (2), 147–161. Klostermann, J., Schaake, K., Schwarze, R., 2013. Numerical simulation of a single rising bubble by VOF with surface compression. Int. J. Numer. Methods Fluids 71 (8), 960–982. Lastow, O., Balachandran, W., 2006. Numerical simulation of electrohydrodynamic (EHD) atomization. J. Electrostat. 64 (12), 850–859. Lim, L.K., Hua, J., Wang, C.H., Smith, K.A., 2011. Numerical simulation of cone‐jet formation in electrohydrodynamic atomization. AIChE J. 57 (1), 57–78. López-Herrera, J.M., Barrero, A., Boucard, A., Loscertales, I.G., Marquez, M., 2004. An experimental study of the electrospraying of water in air at atmospheric pressure. J. Am. Soc. Mass Spectrom. 15 (2), 253–259. López-Herrera, J.M., Gañán-Calvo, A.M., Herrada, M.A., 2010. Absolute to convective instability transition in charged liquid jets. Phys. Fluids 22 (6), 062002. López-Herrera, J.M., Popinet, S., Herrada, M.A., 2011. A charge-conservative approach for simulating electrohydrodynamic two-phase flows using volume-of-fluid. J. Comput. Phys. 230 (5), 1939–1955. Morris, T., Malardier-Jugroot, C., Jugroot, M., 2013. Characterization of electrospray beams for micro-spacecraft electric propulsion applications. J. Electrostat. 71 (5), 931–938. Osher, S., Sethian, J.A., 1988. Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations. J. Comput. Phys. 79 (1), 12–49. Ouedraogo, Y., Gjonaj, E., Weiland, T., De Gersem H., SteinhausenC., Lamanna, G., Weigand, B., Preusche, A., Dreizler, A., Schremb, M., 2017. Electrohydrodynamic simulation of electrically controlled droplet generation. Int. J. Heat Fluid Flow 64,

120–128. Pantano, C., Gañán-Calvo, A.M., Barrero, A., 1994. Zeroth-order electrohydrostatic solution for electrospraying in cone-jet mode. J. Aerosol Sci. 25 (6), 1065–1077. Park, J.U., Hardy, M., Kang, S.J., Barton, K., Adair, K., kishore Mukhopadhyay, D., Lee, C.Y., Strano, M.S., Alleyne, A.G., Georgiadis, J.G., Ferreira, P.M., 2007. High-resolution electrohydrodynamic jet printing. Nat. Mater. 6 (10), 782–789. Puckett, E.G., Almgren, A.S., Bell, J.B., Marcus, D.L., Rider, W.J., 1997. A high-order projection method for tracking fluid interfaces in variable density incompressible flows. J. Comput. Phys. 130 (2), 269–282. Raees, F., Van der Heul, D.R., Vuik, C., 2011. Evaluation of the Interface-Capturing Algorithm of OpenFoam for the Simulation of Incompressible Immiscible Two-Phase Flow. Delft University of Technology. Rosell-Llompart, J., De La Mora, J.F., 1994. Generation of monodisperse droplets 0.3 to 4 µm in diameter from electrified cone-jets of highly conducting and viscous liquids. J. Aerosol Sci. 25 (6), 1093–1119. Rusche, H., 2003. Computational Fluid Dynamics of Dispersed Two-Phase Flows at High Phase Fractions, Ph.D. dissertation. Imperial College. Salata, O.V., 2005. Tools of nanotechnology: electrospray. Curr. Nanosci. 1 (1), 25–33. Saville, D.A., 1997. Electrohydrodynamics: the Taylor–Melcher leaky dielectric model. Ann. Rev. Fluid Mech. 29 (1), 27–64. Setiawan, E.R., Heister, S.D., 1997. Nonlinear modeling of an infinite electrified jet. J. Electrostat. 42 (3), 243–257. Sussman, M., Almgren, A.S., Bell, J.B., Colella, P., Howell, L.H., Welcome, M.L., 1999. An adaptive level set approach for incompressible two-phase flows. J. Comput. Phys. 148 (1), 81–124. Sussman, M., Fatemi, E., Smereka, P., Osher, S., 1998. An improved level set method for incompressible two-phase flows. Comput. Fluids 27 (5), 663–680. Sussman, M., Smereka, P., Osher, S., 1994. A level set approach for computing solutions to incompressible two-phase flow. J. Comput. Phys. 114 (1), 146–159. Sweet, M.L., Pestov, D., Tepper, G.C., McLeskey, J.T., 2014. Electrospray aerosol deposition of water soluble polymer thin films. Appl. Surf. Sci. 289 (1), 150–154. Tang, K., Gomez, A., 1996. Monodisperse electrosprays of low electric conductivity liquids in the cone-jet mode. J. Colloid Interface Sci. 184 (2), 500–511. Taylor, G., 1969. Electrically driven jets. Math. Phys. Eng. Sci. 313 (1515), 453–475. Taylor, G.I., 1964. Disintegration of water drops in an electric field. Math. Phys. Sci. 280 (1382), 383–397 1964. Tomar, G., Gerlach, D., Biswas, G., Alleborn, N., Sharma, A., Durst, F., Welch, S.W.J., Delgado, A., 2007. Two-phase electrohydrodynamic simulations using a volume-offluid approach. J. Comput. Phys. 227 (2), 1267–1285. Tryggvason, G., Bunner, B., Esmaeeli, A., Juric, D., Al-Rawahi, N., Tauber, W., Han, J., Nas, S., Jan, Y.J., 2001. A front-tracking method for the computations of multiphase flow. J. Comput. Phys. 169 (2), 708–759. Wei, W., Gu, Z., Wang, S., Zhang, Y., Lei, K., Kase, K., 2013. Numerical simulation of the cone–jet formation and current generation in electrostatic spray—modeling as regards space charged droplet effect. J. Micromech. Microeng. 23 (1), 1–11. Xu, Q., Qin, H., Yin, Z., Hua, J., Pack, D.W., Wang, C.H., 2013. Coaxial electrohydrodynamic atomization process for production of polymeric composite microspheres. Chem. Eng. Sci. 104, 330–346. Yan, F., Farouk, B., Ko, F., 2003. Numerical modeling of an electrostatically driven liquid meniscus in the cone–jet mode. J. Aerosol Sci. 34 (1), 99–116. Yan, W.C., Davoodi, P., Tong, Y.W., Wang, C.H., 2016. Computational study of core‐shell droplet formation in coaxial electrohydrodynamic atomization process. AIChE J. 62 (12), 4259–4276. Yoon, H., Woo, J.H., Ra, Y.M., Yoon, S.S., Kim, H.Y., Ahn, S., Yun, J.H., Gwak, J., Yoon, K., James, S.C., 2011. Electrostatic spray deposition of copper–indium thin films. Aerosol Sci. Technol. 45 (12), 1448–1455. Youngs, D.L., 1982. Time-dependent multi-material flow with large fluid distortion. Numer. Methods Fluid Dyn. 24 (1), 273–285. Yu, J., Qiu, Y., Zha, X., Yu, M., Rafique, J., Yin, J., 2008. Production of aligned helical polymer nanofibers by electrospinning. Eur. Polym. J. 44 (9), 2838–2844. Yu, M., Ahn, K.H., Lee, S.J., 2016. Design optimization of ink in electrohydrodynamic jet printing: effect of viscoelasticity on the formation of Taylor cone jet. Mater. Des. 89, 109–115. Zeleny, J., 1914. The electrical discharge from liquid points, and a hydrostatic method of measuring the electric intensity at their surfaces. Phys. Rev. 3 (2), 69–91. Zeleny, J., 1917. Instability of electrified liquid surfaces. Phys. Rev. 10 (1), 1–16.

H. Dastourani et al. International Journal of Heat and Fluid Flow 70 (2018) 315–335

335

