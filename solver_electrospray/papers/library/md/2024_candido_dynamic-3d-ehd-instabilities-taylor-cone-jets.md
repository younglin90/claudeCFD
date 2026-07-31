
# 

View Online 

Export Citation

RESEARCH ARTICLE |  MAY 15 2023


# Dynamics of three-dimensional electrohydrodynamic instabilities on Taylor cone jets using a numerical approach

Sílvio Cândido  ; José C. Páscoa


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq001.png)

https://doi.org/10.1063/5.0151109


### Articles You May Be Interested In

A three-dimensional numerical study of the onset of electrohydrodynamic cone-jet whipping instability

Physics of Fluids (September 2025)

Electrohydrodynamic instability and disintegration of low viscous liquid jet

Physics of Fluids (December 2022)

Building water bridges in air: Electrohydrodynamics of the floating water bridge

Physics of Fluids (December 2010)

# Dynamics of three-dimensional electrohydrodynamic instabilities on Taylor cone jets using a numerical approach

Cite as: Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 Submitted: 18 March 2023 . Accepted: 2 May 2023 .

Published Online: 15 May 2023

S�ılvio C^andidoa) and Jos�e C. P�ascoa

AFFILIATIONS

Department of Electromechanical Engineering, C-MAST, University of Beira Interior, 6200-001 Covilh~a, Portugal

a)Author to whom correspondence should be addressed: silvio.candido@ubi.pt

ABSTRACT

Electrohydrodynamic (EHD) jets are a highly promising technology for the generation of three-dimensional microand nanoscale structures, but the advancement of this technology is hindered by the insufficient understanding of many aspects of its flow mechanisms, such as the whipping behavior under larger electric potentials. A fully coupled numerical simulation of the three-dimensional electrohydrodynamic jet flow is used here since non-symmetric effects govern most of their EHD regimes. By applying considerable electric capillary numbers (CaE > 0:25), we capture radial instabilities that until now no other numerical simulation was able to present. A comparison against previous two-dimensional axis-symmetric and validation with experimental studies of the Taylor cone jet is initially done. An exciting gain in accuracy was obtained, having an error of around 1.101% on the morphology against experimental results. Moreover, our numerical model takes into consideration the contact angle between the surface of the nozzle and the liquid, which is shown to be a very important variable for improved accuracy in the morphologic shape of the Taylor cone. Moreover, the three-dimensional structures and flow dynamics, under different electric capillary numbers, and their connection to the instabilities of the jet are studied. We present a novel visualization of the formation of droplet generation with the receded Taylor cone and the whipping dynamics.

V C 2023 Author(s). All article content, except where otherwise noted, is licensed under a Creative Commons Attribution (CC BY) license (http:// creativecommons.org/licenses/by/4.0/). https://doi.org/10.1063/5.0151109

I. INTRODUCTION

Electrohydrodynamic (EHD) jet atomization has been a scientific topic with interest in many areas for many years. Since the miniaturization and fabrication of three-dimensional micro/nanoscopic multifunctional structures becomes increasingly important to many existing technologies,1 EHD jet technology and its flow studies understanding can be very useful. Electrohydrodynamic jet printing (EHDP) exceeds the limitations of conventional jet systems and offers an efficient technique to produce high-resolution liquid droplet/fragment of a wide variety of liquids by simply imposing a strong electric field between the nozzle and the ground pale.2–4 This is a technological area that can benefit from EHD advances. Therefore, due to their potential applications in a variety of fields, including high-resolution printing,5,6 biotechnology (for the encapsulation of bioactive compounds), colloidal propulsion through electrospray beams,7 surface coating,8 and electrohydrodynamic flow dynamics have been extensively studied in the last decade. When a liquid is exposed to a high electrostatic potential, in comparison to its surroundings, a cone of liquid (often referred to as a Taylor cone)

forms and, if the circumstances are right, emits a microjet of liquid that eventually disintegrates into tiny charged droplets. The behavior was already experimentally tested for many liquids, and the effect of the electric field on the inlet flow rate have been studied.9,10 Still these types of jets are characterized by high pulse frequency and the dynamics of this radial instabilities is not yet fully understood. The liquid is fed via a capillary nozzle to which a high voltage is applied, in this zone between the nozzle and a collector, to create these electrified jets. Figure 1 shows the physical setup. The electrostatic force impact due to the electric field on the liquid can then result in the formation of a narrow jet. The primary factors that affect the jet’s stability are the inlet flow characteristics and the applied voltage. In general, there are different parts of the EHD spray that can be considered. A droplet plume can be produced after the formation of a Taylor cone, a transition zone, a continuous fine jet, and the splitting of the jet into droplets are different zones that can be considered.11

Most electrospray research focuses on the initial part of the jet, with Taylor cone formation and primary jet breakup (the cone-jet region) being the target, or on the dynamics of the plume spray

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-1

V C Author(s) 2023


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

particles (the primary spray plume region). The first investigations that mentioned this problem used mainly algorithms based on Boundary Element Methods (BEM),2 Volume-of-Fluid (VoF),12–14

and phase-field.15,16 In this second type, the Lagrangian particle approach and Newtonian laws of motion are commonly used to govern particle trajectories.17–19 More comprehensive studies show that three-dimensional simulations are needed to fully understand the flow fields generated by the EHD effects.20

As far as we know, a transient and fully coupled numerical simulation of the three-dimensional EHD jet flow has not been performed. This is due to the cost of this computations, where a fully threedimensional (3D) model need a very refined grid in order to capture well the interface of the liquid, that is where the electric charge is accumulated and has a profound impact. So far, studies of EHD jets have been restricted to the axis-symmetric condition of operation.12 It is, thus, considered that the realistic macroscopic complex pattern of a single phase EHD flow can only be obtained with three dimensional computations.20 This has been observed in studies on the deformation of liquid droplets due to the external electric field21–23 and the interaction between two droplets.24

The formation of electrospray and electrospinning jets25 is well known for its multiscale nature, in which different scales are at work. The order of geometry is millimeters, the order of scale of the jet is typically ten times smaller, and the diameter of the droplets can be close to 100 times smaller than the diameter of the capillary nozzle. As a result, the axis-symmetric condition is assumed in the majority of EHD flow simulation applications, and the authors tend to ignore the unsteady regions of this type of flow. The current study will look into the dynamics of the EHD jet’s unsteady operation modes. Numerical techniques using the Volume of Fluid (VoF) for interface capturing have shown to be effective at simulating multiphase flows. Due to its robust capacity to describe two immiscible fluids, which enables the comprehension of the liquid–gas interaction, this approach has evolved into a standard model for multiphase simulations.26 In the VoF approach, the interface is often captured using a

reconstruction method and a phase transport equation. It takes more computation time to approximate with the approaches that use reconstruction in the phase transport equation since they typically involve two steps: geometric interface reconstruction and interface propagation. The conventional algebraic method was used in earlier VoF simulations of electrified jets (see Ref. 27). According to Refs. 13 and 28, the liquid phase in our simulations of electrified jets invariably separates into a jet and subsequent droplets and this requires a high degree of interface fidelity. By including a transient element,12 we are able to close the transport equation for electric charges. In order to create jets with a wider diameter, a low-conductivity fluid is frequently used in capillary nozzle tubes. A stable cone jet mode is typically simulated using axisymmetric models. However, a fully three-dimensional plume cannot be accurately modeled using those axisymmetric models, even if they are considered accurate in the initial sections of the jet. As a result, in some circumstances, the early phase of the jet is studied using the VoF approach, and the radial expansion of the droplets, which make up the so-called plume, is studied using a particle model.29 The geometric VoF technique approximates the phase fraction fluxes over the cell surfaces and reconstructs the fluid interface within a cell using geometric operations.30 By comparing the solution to the algebraic and geometric VoF as benchmarks for droplet deformation31 and Taylor cone jets,32 the enhancement of the interface resolution was demonstrated for our case. The outline of the present paper is as follows: first at Sec. II, we detail the methodology of the numerical simulation, with the description of the governing equations and the numerical framework used. In Sec. III, we present the validation of our method by considering experimental and numerical results available in the literature. After the validation, in Sec. IV, we analyze the flow field, in particular, with increased electric potential and we obverse the grow of radial instabilities and their implication on the practical uses. In Sec. V, this paper concludes with an assessment of the method, remarks on the simulation of the formation of electrosprays, and outlines future work.


> **FIG. 1. Schematic of the physical setup, where the capillary nozzle wall is at a high voltage U relative to a ground collector. A closer look to the capillary nozzle is seen, showing an inner diameter (Di) and an outer diameter (Do). The blue surface represents the liquid interface.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-2

V C Author(s) 2023

II. MODELS AND METHODS A. Problem description

In these investigations, we focus on the electrohydrodynamic jet formed by a Taylor cone, as mentioned in Sec. I, this is possible with the setup of Fig. 1. The geometry is composed of a capillary nozzle at a certain distance from a ground plate. The nozzle is the inlet of our problem, where a fully developed liquid flow is created. Our problem boundary conditions are based on previously studied cases to identify the model’s accuracy (done in the validation section).33,34

The inlet nozzle is defined by an inner diameter Di of 160 lm and an outer diameter D0 of 260 lm. The length of the nozzle (L) is set to 300 lm. This length of the nozzle ensures a fully developed flow at the tip and prevents recirculation at the inlet. The tip of the nozzle is set to a fixed distance from the ground collector of H ¼ 1.5 mm. To ensure computational accuracy, the ground collector has a diameter Dc ¼ 5 mm, allowing the flow to adjust to the dynamics and not perturbing the jet. In Fig. 2, we have represented the different forces that take action on the liquid interface. These forces have a hydrodynamic component, which includes the gravitational body force g, the surface tension fc and the viscous force fl. Furthermore, an electrical part, which includes the electric polarization force Qþ / and the normal and tangential electric stresses on the surface En, Et, acts on the surface. All of these forces are calculated using the FVM method, for which the governing equations are described below. We consider that the liquid phase is represented by the domain X1 and the medium is the domain X2.

B. Electrohydrodynamic mathematical formulation

The governing equations needed to solve the two isothermal, incompressible, and immiscible fluids are continuity, momentum, and

the interface advection equations. The numerical simulation methodology for the EHD jet is implemented in OpenFOAM that includes a fine-volume method (FVM) based solver, where the governing equations are applied. An overview of the governing equations is given below. An important detail is the method for interface tracking, which in our case is the geometric Volume-of-Fluid (VoF) method. With this method, only one scalar is defined to represent the liquid and gas phases, and just one equation solves both phases and, in the end, the sum of the volume fraction is equal to one.35 This scalar is a and is defined as


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq002.png)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq003.png)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq004.png)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq005.png)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq006.png)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq007.png)

where u is the velocity vector, composed for a Cartesian coordinate system by the three components, ux, uy, and uz. We are considering a flow with a mass density q, a dynamic viscosity l, a local fluid pressure p and subjected to a gravitational acceleration g. In this case into the source term of the momentum, we added the force due to the surface tension f c and one due to the external imposed electric field, as f e. The contributions of surface tension, electric force, and gravitational force are treated as source terms of the momentum equation. The surface tension, f c, is obtained using the Continuum Surface Force (CSF) model, initially developed by Brackbill et al.36 In the CSF model, the surface tension is represented as a volumetric force that, based on a constant surface tension coefficient, is written as a function of the surface tension coefficient, c. The surface tension, f c, is determined using the Continuum Surface Force (CSF) model, initially developed by Brackbill et al.36 The surface tension in the CSF model is presented as a volumetric force that, based on a constant surface tension coefficient, is stated as a function of the fluid property surface tension coefficient, c. This method computes the local curvature of the interface, j, as a function of the scalar of the phase fraction, and as the divergence of ra=jraj. With the normalized curvature, the surface tension force is expressed as,


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq008.png)

The surface tension at the interface generates an additional pressure gradient resulting in a force evaluated per unit volume. In this kind of model the interface spreads over a few cells.37 The surface compression term is only imposed at the interface, so the advection of both individual flow fields is not affected. The implementation of the electric body force that acts on the fluid is made by adding a source term described as f e, that is determined using the reduced form of Maxwell’s equations for the


> **FIG. 2. Schematic diagram of the electrohydrodynamic Taylor cone formation, with the actuating forces represented. The red face is the capillary nozzle, set to a high electric potential U, and blue is the iso-surface of a ¼ 0:5 that represents the free surface of the liquid.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-3

V C Author(s) 2023

electrostatics. This force is computed by using the Maxwell Stress Tensor (MST). To determine the MST we need first to obtain the electric field on the domain. The external electric potential /e is set to a fixed value in time with a magnitude of U0. As the electric potential applied is static, since the dynamic currents are considerably small, and for that specific reason, there is no significant magnetic induction,14 the governing equations are thus the following:

@qe @t þ r �qeu ð Þ þ r �ðrEÞ ¼ 0; (6)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq009.png)

Equation (6) represents the conservation of the bulk-free charge, which is given by the sum of charge transport by conduction (rE) and by convection (qeu). The bulk volumetric force, f e, is calculated with the electrostatic Maxwell stress tensor. The MST (T) is a force per unit area acting on the surface so the force, per volume, inserted on the momentum Eq. (3), is equal to the divergence of the MST,

f e ¼ er �T ¼ er � EE �1 2 jjEjj2I � � : (9)

The divergence of the MST will produce a body force in the domain, with a significant magnitude in the interface region due to the involved gradient of the permittivity and electric potential. Since we are considering an incompressible flow there is no variation of the electric permittivity with the density on the phases;38 therefore, the electric body force, f e, can be expressed as


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq010.png)

The first term of the right side of the equation is related to the Coulomb forces and the second term is the Lorentz force. The implementation of the electric force directly in the momentum equation solves, simultaneously, the dynamics of the flow in the defined computational mesh.

C. Numerical considerations

Since the flow is composed of two phases that have different properties, the hydrodynamic and electric properties of the flow need to be appropriately updated. This update is made using state equations that will take into consideration the boundedness, between zero and one, at the interface. The hydrodynamic properties of the flow, such as mass density and dynamic viscosity, are computed with weighted arithmetic mean interpolation (WAM). Regarding the electric phase properties, instead of using a WAM interpolation, the electric permittivity and the conductivity are obtained with a weighted harmonic mean interpolation (WHM), both weighted with the phase fraction. This difference is because this approach provides better accuracy for this region of transition.14 These two calculations are described through the following equations:

fq; lg ¼ fq1; l1ga þ fq2; l2gð1 �aÞ; (11)

fr�1; e�1g ¼ fr�1 1 ; e�1 2 ga þ fr�1 2 ; e�1 2 gð1 �aÞ: (12)

In our model, the WHM was just implemented in the electric properties, but in further investigations, the implementation on the hydrodynamic properties could also be tested.21

Moreover, the simulation is considered laminar, and the interface reconstruction scheme is the plicRDF,30 an improved geometric reconstruction method named isoAdvector algorithm is implemented using the OpenFOAM. Furthermore, the coupling of the pressure and velocity field is made with the PIMPLE algorithm, providing a very robust pressure–velocity coupling.32,39,40

Finally, to ensure the temporal accuracy, the time step of the calculations is defined by limiting the electric Courant number to 0.1. This electric Courant number is the minimum time step that ensures that both the hydrodynamic relaxation time, given by the standard Courant number, and the electric relaxation time se ¼ r=e28 is fulfilled.

D. Boundary conditions and initial condition

The imposed boundary conditions are as follows: at the capillary nozzle inlet, a Neumann condition is used for pressure, while a Dirichlet condition (a ¼ 1) is imposed for the phase fraction scalar. A Dirichlet condition for a fully developed laminar profile is generated by mapping the velocity from inside the capillary nozzle domain, such that the mean velocity is forced to the bulk value of uin at each time step. At the nozzle walls, a no-slip (u ¼ 0) Dirichlet condition is used for the velocity, while Neumann conditions are applied for pressure, phase fraction, and electric charge. A mixed type boundary condition is used at the outlet boundary, which allows the ambient gas to flow in or out through this boundary of the numerical domain. When the flow is out of the domain, Neumann conditions (zero gradient) are used for the phase fraction, the electric charge, and the velocity. The electric potential is fixed for the wall /e ¼ U and zero-gradient at the atmospheric boundary r/e ¼ 0. When the flow is directed into the domain, Dirichlet conditions are used for both fields (a ¼ 0 and u extrapolated from inside the domain). In both cases, the total pressure is fixed to 0 Pa at the outlet boundary (atmospheric condition). To ensure that the conditions are the same as the ones of the experimental test-case to ensure comparison, the collector wall has a fixed velocity of u ¼ ð�us; 0; 0Þ, where us is imposed as the experimental collector velocity of 20 mms�1. This was done for every computation, thus observing the effect of just one condition, for a better comparison on the dynamics. The fluid is dripped from the nozzle at a constant flow rate, Qi, corresponding to an inlet velocity uin ¼ Qi=ðpD2 i =4Þ. A fixed electric potential is applied as /e ¼ U0. These two parameters are what most influences the stability of the Taylor cone and the jet formation. The initial condition for the phase fraction a ¼ 1 considers two regions, the region inside the inlet nozzle given by fx2 þ z2 �ðDi=2Þ2g8y 2 ½�L; 0�and the initial region of the interface P 0 : fx2 þ y2 þ z2 �ð0:95D0=2Þ2g. This last one is constrained by ð0:95D0=2Þ to ensure that the liquid is not at the corner of the nozzle, since this region is of high electrical stress and can lead to unreasonable solutions due to the possibility of a Coulomb explosion. As we will discuss later, our model has different grid refinement levels. To ensure a high accuracy at the interface, our model uses nonuniform meshing to keep the interface with a minimum grid size.41

The mesh was generated using the snappyhexmesh built-in meshing


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-4

V C Author(s) 2023

tool of OpenFoam. This generates three-dimensional grids containing hexahedra and split-hexahedra cells and allows local refinement. This is particularly important when we have such different scales in play, e.g., the domain size and the jet diameter. Our present work focuses only on one liquid solution for the computations. This liquid solution is a mixture of glycerol, water, and sodium chloride (NaCl), whose physical and electrical properties are given at Table I.33

The wettability of the wall is a necessary boundary condition for these types of flows (capillary flows). A hydrophilic wall (low contact angle) promotes spreading of the liquid, while a hydrophobic wall (high contact angle) promotes a more compact liquid droplet.42 In our investigations, the wettability of the non-slip walls (electrode nozzle and ground plate) is controlled by a static contact angle hs, with relates to the Young’s equations cos hs ¼ cwc�cwd c , where cwc is the surface tension of the surface-continuous phase (air) and cwl the surface tension of surface-liquid. All simulations were performed with a contact angle of 51�for the non-slip walls.

III. MODEL VALIDATION

The first step of our work is to validate the numerical results against experimental results in order to support the reliability of the numerical model. This validation is made with a grid sensitivity test and a comparison of the morphology of the obtained Taylor cone. This validation section is very important because, for the model used, the resolution of the interface is very demanding on the accuracy of the electric charge density. The results are compared with previous experimental work by Guo et al.33 and numerical results from Guan et al.34

The working conditions for the validation tests are kept constant. The electric potential at the capillary nozzle is U0 ¼ 2:18 kV and the inlet flow rate is Qi ¼ 16:1 nl s�1. An important non-dimensional number, the electric capillary CaE, is defined, to characterize freesurface flows that are ruled, mainly, by the conditions of the external applied electric field and the surface tension, this later being predominantly large due to the capillary dimensions of the nozzle (O1 lm).43

The CaE is, thus, defined as


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq011.png)

where electric field magnitude E0 is defined as U0=½Do ln 4H=Do�. This number is closely linked to the working regime of the jet. Usually, for small electric capillary numbers (<0:1), there is no jet formation, in a range from �½0:1; 0:5�jet formation happens, transiting from a stable emission of droplets, into a continuous jet deposition and, finally, jet whipping.34,44,45 Numbers >0.5 create a big electric stretch on the surface, while the viscous component is much smaller, and thus, the Taylor cone creates multiple jets that will follow the applied

electric field. The two last regimes will be observed in Sec. IV. At this point, we just focus on a CaE equal to 0.25 for validation. In addition to the capillary electric number, the inlet flow rate has a big impact on the stability of the jet.46 We, thus, define the electric Weber number WeE,


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq012.png)


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq013.png)

Q0 is the minimum inlet flow rate for a steady cone-jet regime, showed by experimental studies for different liquids.46,47 Since we only have a single set of liquid properties our overall electric Weber number is 20.4. An important scale to be considered is the hydrodynamic surface tension timescale given as sh ¼ ffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi q1D3 i =c p .48 This timescale is the relevant scale for capillary liquid flows and so it is directly linked with our current flow type. Another important number is the hydrodynamic Weber number, given as We ¼ qu2 dDdc�1, that is defined for the generated liquid droplets with a diameter of Dd and traveling at a velocity ud.49

A. Validation of Taylor cone morphology

To validate the model, one must first perform a grid sensitivity test and compare the morphology of the Taylor cone with experimental and other numerical results. We should notice that the previous numerical results available in the literature have never considered the full three-dimensional effects. The domain of computation is discretized in the grid presented in Fig. 3(a). This grid is successfully refined in order to get a very fine grid in the nozzle region, which is where the liquid jet will form. As we can see in the figure, there are three refinement regions: the most refined zone, the far zone, and a buffer layer that separates the last two. The most refined region is the region of the nozzle, containing the nozzle and the space linking to the collector, that is a cylindrical region, along all the y-axis, with a diameter of 1:2D0. The less refined grid is distanced from the nozzle with 2D0. Between these two regions there is a buffer region, which is a transition from the two refinement levels, the most refined zone is �5 more refined than the outer region. It is important to highlight that between each transition layer of refinement there are two grid cells. Four different grids were tested, by changing the minimum grid size with had values of dx : f4:00; 3:08; 2:50; 2:02g lm. The Taylor cone iso-surface comparative with the three different refinement levels can be observed in Appendix. These values were suggested by the previous studies using the same approach to solve EHD flow, although they used an axisymmetric methodology as mentioned previously. The grid with dx ¼ 4:00 was excluded from any comparison because it did not even generate a jet condition. The optimum grid size was the one with dx ¼ 2:00, since there was a convergence of the Taylor cone morphology that will be discussed ahead. All of these considerations can be observed in Fig. 3(a), which shows the final grid. The final grid used is composed of �11 million cells. This requires a parallel computation, which was done using 60 cores (Xeon Gold 6226R/2.9GHz). In average, for the most refined grid and the validation case, 10 ls of flow time takes 1 h of calculation, and a total of 20 ms are required, meaning a total of 200 h for the simulation of one condition. This time tends only to increase with the increase in the instability, due to the need of refinement of the grid in the radial direction.

TABLE I. Hydrodynamic and electrical properties of the liquid and the air.33

Phase l (mPa s) er ð�Þ r (S m�1) q (kg m�3) c (m N m�1)

Liquid (X1) 60.00 55.6 60 �10�6 1208.4 64.5 Air (X2) 0.0120 1.0 1 �10�15 1.225 ���


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-5

V C Author(s) 2023

A comparison of the free-surface results obtained by other authors and our numerical model is now analyzed. These results are depicted as snapshots of the Taylor cone formation, jet formation, and cone recession. For a better comparison, the results were compared only for a value of a ¼ 0:5, that is, the iso-surface of the liquid. The results are shown in Fig. 3(b), which show an excellent prediction of the Taylor cone formation. The lines of the images were transposed in order to properly compare the shape of the cone. The results show an improvement in the comparison of the numerical with the experimental dotted curves. It is important to notice that the liquid, as the experimental results show, tends to wet the nozzle. If was not happening, the shape of the cone would not correspond to the real behavior, as we show. The time of the snapshots is normalize in order to synchronize with the results, since the initial condition produces a variation of the initial dynamics. This normalization consist in setting the t¼ 0.4ms to the instant of the maximum cone length but without the emission of the jet. For validation, the other times should be synchronized, as we see from the snapshots. Calculating the liquid volume as V ¼ P px2 a¼0:5 for the values of y 2 ½0; ytip�, we get the error of morphology, given as �r ¼ ðVnum �V exp ÞV�1 exp . For the times of t : f0:0; 0:4; 0:7; 0:8; 0:9g ms, the relative error was �r : f1:97; 0:747; �1:44; �0:395; �0:948g%, respectively. Notice that the negative values mean that the volume is less than the experimental. This gives an average error of around 1.101%, which is significantly lower than the 22% calculated for the curves of the previous numerical axis-symmetric computation.34

Moreover, Guan et al.34 have used the same order of magnitude for the grid size, and they showed that a dx ¼ 1:6 to 2.0 lm produces no significant difference in results. Although they used a pure algebraic VoF, as showed in other multiple studies,40,50 the use of the geometric advection scheme for the interface narrows the spread of the interface within the cells. In comparison, if using the algebraic VoF scheme, we would need the double of cells to achieve the same level of sharpness of the interface.39 With this, we can state that the independence of grid

size is coherent with the previous studies, which increases our certain on the validation of the model. Moreover, in the further results, we will observe temporal dynamics and droplet emission that also are coherent to the experimental results.

IV. RESULTS AND DISCUSSION

With our numerical validated, we will now explore the working conditions for a larger electric capillary number and observe the predictions of the liquid jet instability, such as whipping and multi jet formation. The goal here will be to see if the flow dynamics is reproduced in the developed model.

A. Temporal dynamics

The shape of the Taylor cone and the liquid jet depends on the forces that are acting on the flow field and their temporal dynamics. As mentioned, computations are performed using an adaptive time step with the minimum time step selected between the courant condition for 0.1 and the electric relaxation se. This means that when we performed the grid sensitivity, we were refining the space resolution and also the time resolution, since the courant condition depends on the grid cell size. Now, we will confirm the temporal dynamics of the solution by analyzing the evolution of the liquid phase over time.


> **Figure 4 shows the temporal evolution of the jet 3D solution for a validation case with CaE ¼ 0:25. The figure shows the free surface of the liquid colored by the magnitude of the velocity. Here, we observe that, even under this low CaE condition, the droplets have some nonsymmetric behavior (see the droplets in the red region). Here, we also define some important time marks. The time tc is the starting point of the emission, and it stops at tf, giving a jetting time of tj. The time t0 is not the initial condition, but the time when the liquid is in equilibrium of the forces, meaning that the velocity of all the interface is zero. The time before t0 is a receding zone, where the tip of the liquid actually goes up (in the opposite of the E field) due to the stabilization of the**


> **FIG. 3. (a) Computational grid on a cut plane YoZ. (b) Here, we represent in blue is the initial condition for the interface P 0 and in red the inlet nozzle. The comparison of the free-surface of the liquid for our computations and previous studies, for validation of the EHD model. In our results, the surface points are colored by the coordinate of the point in the z axis.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-6

V C Author(s) 2023

contact angle of the liquid with the nozzle, and so this first few time steps are discarded, since they will vary from the initial condition. Although, to make sure this is a true t0 the calculation ends when this same shape is repeated, confirming a good solution, not depending on the initial condition.


> **Figure 5 shows the evolution of the tip, DLt, and the centroid of the liquid phase, DLc, on the domain. This D is defined as the difference between the axial position of the interface compared with the initial condition of P 0. The evolution of the tip of the liquid allows us to determine the point of the cone formation, the starting of the ejection and the mean/average axial coordinates of the liquid phase give us the stabilization of the jet. Notice that we introduce a region of tr, this region is the receding time and the gray region on the plot represents the regions where the velocity is receding (in the negative direction of the axis). The initial zone of this receding behavior is due to the surface tension relaxation, meaning that the wall force is due to the stabilization of the contact of the liquid with the wall and the equilibrium between all the forces. During this time, the electric charges accumulate at the interface and the jetting process starts. Notice that jetting time tj is shown in Fig. 5 with an arrowed interval, and below there is the same length interval.**

The first one is the jetting time from the point of view of the cone emission, and the second is the jetting time from the point of view of the ground plate. The delay of the second is just the time of flight of the liquid phase. The time tr is removed from the further working conditions computations, being our t0. t0 was defined as a temporal hydrodynamic relaxations before the Taylor cone maximum tip tc.

B. Single droplet ejection

As we observe, a droplet is ejected from the jet, forming a deposition on the collector. The shape of this droplet is also compared with the experimental results. In the experimental results, the authors show that the droplets are deposited in the collector and have a contact angle with the surface of hs ¼ 50:9�. From our numerical results, we can observe that the ejected droplet deposition in Fig. 6 has exactly the same contact angle. This was ensured by the constant contact angle boundary condition of the ground plate wall and set to an angle of hs ¼ 51�. Although it is worth notice that the droplet at the collector is subject to the ionic wind that we can observe from the instantaneous pattern of streamlines of the velocity vector field and also to the


> **FIG. 4. A temporal sequence of the 3D iso surface (a ¼ 0:5) on a xy view plane. The surface is colored by the velocity u�=uin, where the velocity magnitude u if multiplied as u�¼ 1000u.**


> **FIG. 5. Evolution of the axial position variation of the tip (DLt) and the centroid of the liquid phase (DLc) on the domain of the liquid, from initial condition to ground deposition (black). Evolution of the velocity at the liquid tip (green).**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-7

V C Author(s) 2023

electrostatic force, that we can observe from the instantaneous pattern of streamlines shown at the same figure. Focusing now on the instantaneous pattern of streamlines of the vector fields, namely, velocity and electric field of Fig. 6, the width of the lines is proportional to the magnitude of the field at that point. It is to be noticed that, for both fields, the magnitude of the vector field inside the droplet is proportionally very small as compared to that in the outside. The electric field, inside the droplet is very small due to the conductivity of the liquid, which is much larger than that of the air. Moreover, the figure also shows the density of the electric charge, which, as we see, is concentrated at the interface, as it should be due to the leaky dielectric model used. However, it is interesting to notice that this is a concentration of negative charges. Intriguingly, the Taylor cone jet has a dominant positive charge at the interface, see Fig. 5, but when the liquid phase touches the ground plate the charges become negative, ensuring the conservation of charge. It is possible to see that the droplet is not centered. This is due to the moving plate condition. With these results, we see that the contact angle of the droplet with the collector is well imposed, thus validating our model. The electric field accelerates the electric charges on the liquid surface in the axial (y) and radial (x, z) directions. This electric field can be shown by the instantaneous pattern of streamlines of the field. Although the liquid solution is a leaky-dielectric fluid, the electric charges will suffer convection from the free surface and travel in all the flow domain. If the liquid does not travel as fast as the electric charge, the electric charge will saturate the region and travel in the radial direction. This creates non-symmetric behavior and can lead to creation of waves on the liquid jet, and we will see this further in higher conditions of a CaE.

C. Effect of electrical potential

The previous case, CaE ¼ 0:25, was a very stable operating condition with a single droplet emission. As reported on an experimental

study for this case, even with a slightly increase on the electric capillary number to CaE ¼ 0:26, there will be non-symmetric droplets generated around the main droplet emitted. The results will consider higher electric potentials. Increasing the electric potential will increase the electric current and the liquid jet, as well as maximum electric charge density, in order to maintain the jet stable. This will induce radial instabilities that until now no other numerical simulation capture. Figure 7 shows the temporal evolution of the liquid jet for a capillary electric number of CaE 2 f0:26; 0:32; 0:38; 0:42g. This is an increase from the previous working condition of 4%; 28%; 52%, and 65%, respectively. As observed from the velocity of the tip, the increase in the electric potential does not allow the liquid jet to completely recessed. This means that there is a main liquid emission, during tj, that have a morphology as depicted in snapshot (i) and, after this time, the Taylor cone recedes and a tiny jet is formed emitting small droplets, see shape (ii). This is the reason why for larger electric potential there are small droplets surrounding the main emission, that were identified in the experimental results of Fig. 5(c) in the paper of Guo et al.33 Moreover, the electric potential increases the maximum velocity, as is normal but diminishes the tc when we increased the electric body force. With an increase on the electric potential the time for the formation of the Taylor cone into the ejection is reduced. Figure 8(a) shows the velocity profile of the liquid free-surface. As we notice for the same flow time t ¼ 0:4sh, the strength of the jet is bigger with bigger electric potential. Although we see somehow a convergence of the velocity profile, meaning that the Taylor cone velocity profile will remain steady and what will change is the jet emission. The electric current ie is now calculated as ie ¼ Ð

SqeU �n dS, where S is the surface of the cross section of the liquid jet, n is the normal vector to the surface, qe is the charge density, and u is the velocity of the liquid, see Fig. 8(b). The cross section S, was set fixed to the middle of the domain, proving the correct information of the current


> **FIG. 6. Visualizations of the instantaneous pattern of streamlines for (left) the velocity for the cone-jet emission (t ¼ 0.7 ms) and (right-top) electric field and (right-bottom) the velocity field on the region of the droplet (t ¼ 2.7 ms ¼ 10sh). For the two of the right, the width of the lines is proportional to the magnitude of the field. The broken red line is the shape of the droplet from the image of the experimental study [Fig. 1(c) of Guo et al.33]**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-8

V C Author(s) 2023

going through the liquid jet. As we observe in the dynamics of the electric current, this increases at the same time as the jetting time (compare with Fig. 7). These electric current values are compared with the red line, which shows the scaling law for the spray current, defined as I �ðcrQÞ1=2.45 We can observe a slight increase in the maximum electric current, reaching the ejection peak, but the average is not influenced. By increasing the electric potential, the electric field strength around the emitter will increase, which can increase the force on the charged droplets and cause them to move more quickly, leading to an increase in electric current. Although, as previous experimental studies demonstrated, the electric potential increase has little effect on the electric current, thus a correlation between the increase in instability and the increase in the electric current cannot be made.51,52

As a matter of fact, observing the relationship between the regime (stable or unstable jetting) and the scaling laws of the electric current, it is the inlet flow rate which affects the electric current the most.45


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq014.png)

D. Satellite droplet formation


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq015.png)


> **FIG. 7. Variation of the temporal evolution of the (green) velocity of the tip of the liquid phase with an increase in the electric potential CaE 2 f0:26; 0:32; 0:38; 0:42g (in the direction of the arrows) and (black) the mean height of the liquid, for the corresponding electric potentials. Two snapshots of the isosurface (i) and (ii) are shown to demonstrate the type of behavior of the two regions, the jetting region tj and receding region tr.**


> **FIG. 8. (a) Variation of the velocity profile of the surface with an increase in the electric potential CaE 2 f0:26; 0:32; 0:38; 0:42g (in the direction of curved arrow). In the top, the two figures are representing the instantaneous pattern of streamlines of the velocity field and in red the iso-curve of the interface. Comparison is made for the same flow time of t ¼ 0:4sh. (b) Effect of the applied external electric potential on the electric current of the jet, for the same CaE as previously.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-9

V C Author(s) 2023

time. Under this working condition, we observe a completely nonsymmetric behavior of the liquid jet, and there are small windows where the liquid is deposited in a stable way, as time 1:9sh shows, these last around 0:5sh. Outside this window the jet is unstable. From 2:2sh to 2:8sh, the Taylor cone recedes with a radial effect that is the whipping of the liquid. After the recede of the Taylor cone, a small jet remains emitting very small droplets. These small droplets have, for this working condition, a mean diameter of 3:96 lm, which was determined using a normal distribution of the measured diameters of the droplets. Furthermore, these droplets have a mean flight velocity of 4.97 m s�1, thus resulting on a mean hydrodynamic Weber number of 1.35. This indicates that the droplets’ transport is on a vibrational breakup, which reinforces that our regime of computation can be applied because there is no turbulent bag breakup.


> **Figure 9(a) also shows the distribution of the density of the electric charge. Positive electric charges are accumulated at the Taylor cone jet until the moment of breakup. The liquid segments that detached from the main jet have a region of accumulation of a negative electric charge and also a positive one. As discussed, the electric charge accumulates at the interface although, due to the saturation and conservation of the electric charge in the bulk, the charges need to travel outside the interface of the liquid. Even the density is just a fraction of the charge accumulated at the interface. This can be observed by the tree-dimensional structures formed at the domain, as we see at Fig. 9(b). The figure depicts the isosurface enclosing the main electric charge transport for a value of j�qej 6 50%, where j�qej is the mean value on all domain. It is also possible to see the structure following a wavelike behavior outside the liquid interface. This finding implies that the whipping of the jet and its radial instabilities are linked to electric charge transport on the bulk,**

rather than just to the charge polarization at the interface. This is an important result, since the computations of the space charges are important for the accuracy of the solution on electrosprays.53

E. Liquid jet deposition

By fixing the electric potential to CaE ¼ 0:26, we will now analyze the dynamics of the liquid deposition. Figure 10 shows the snapshots of the liquid deposition for the characteristic time steps. The first of these traces show the wave-like effect that occurs on the liquid jet. As we see in Fig. 10(a) due to the fast ejection of the liquid, the jet is squeezed and a wave-like deposition happens. As we saw previously, this wave was originated at an earlier time in the middle of the jet flight, see 1:6sh in Fig. 9(a), which traveled uniformly over the domain until de deposition. This represents the first nonstop contact of the Taylor cone with the collector, which happens at the moment when the peak region of velocity is reached, as we see at Fig. 7. An axis-symmetric deposition happens at time 2:1sh, showed in Fig. 10(b). At this time, the Taylor cone is fully connected with the collector and steadily ejecting. Although from the velocity vector field represented by the arrows in the figure, even if the velocity on the liquid surface is symmetrical on the air medium, it is not. We observe a large velocity at the left of the jet, which indicates the formation of the whipping instability. This is confirmed at the next snapshot, in Fig. 10(c) for the time 2:5sh.

F. Dynamics of whipping jet


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq016.png)


> **FIG. 9. (a) Temporal evolution of the jet, for the ejection time, colored by the density of the electric charge. (b) Three-dimensional ISO structures of the electric charge density qe=j�qej : f0:5; 1:0; 1:5g, at time t=s ¼ 4:5 and CaE ¼ 0:26. Black circumference represents a radial distance of 10Di. CaE ¼ 0:26.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-10

V C Author(s) 2023

chosen because, since the deposition of the jet is constant, we can determine the dynamics of the wave behavior, and we can characterize the liquid behavior with the spatiotemporal evolution of the liquid surface. A better understanding of the behavior of this type of jet can allow us to achieve an increase in the control of the jet, for all the reasons previously mentioned. First, Fig. 11 shows the spatiotemporal dynamics of the jet colored by the velocity magnitude. As we see, although the jet has a breakup along the domain, the jet deposition is in a continuous mode, and the Taylor cone has a very short receding time (as observed at Fig. 7). An interesting result is that the whip of the jet has a very steady initial position, which is marked at Fig. 11 by the horizontal red line. Furthermore, the wave transport has also the same velocity, given at the figure by uw ¼ Dy=Dt, with a flight time of 5% of sh. It is worth noticing that this flight time is considerable large, showing that this whipping behavior has a high frequency, and is difficult to capture experimentally. To understand the dynamics of the liquid jet whip instability, we will now focus on one specific time step, more precisely at 0:64sh.

This time represents the first full contact of the Taylor cone with the collector. Figure 12(a) shows this time snapshot, where the liquid surface is colored by the electric flux density D ¼ qeu. This is the vector field that shows the intensity of the transport of the electric charges in a density manner and is linked with the electric current density. As we see, the wave has a beginning at the point of largest electric flux density. Interestingly, the tip of the wave stays aligned with the radial axis. If the liquid surface cannot handle more electric charge accumulation the charge will be pulled toward the electric field lines, as we see in the Fig. 12(b), in radial way, from the y axis to the ground plate into outside the liquid. Furthermore, the electric field will be warped due to the presence of the electric charges on the liquid surface. This effect of the jet on the electric field can be observed in Fig. 12(b). This figure shows the isolines for the electric potential, which, as we see, are warped by the jet. The final consideration to make for the time step is its effect on the flow field. Figure 12(c) shows the instantaneous pattern of streamlines of the velocity field and the iso-lines of the vorticity magnitude. The vorticity is computed from x ¼ @ux @x �@uy @y . The vorticity of the


> **FIG. 10. Snapshots of three different times. (a) The initial jet impact and (b) the symmetrical jet deposition just before the impact of the second wave in (c). The floor is colored by the electric charge density, and the velocity vector field at the interface of the liquid is shown as (U�¼ 1000U). A white line shows the displacement of the floor corresponding moment (the displacement is considered zero at the moment of the first contact of the liquid with the collector plate, corresponding to t0s ¼ 1:0). Conditions of CaE ¼ 0:26 and us ¼ 20 mm s�1. (a) t=sh ¼ 1:9 (b) t=sh ¼ 2.1 (c) t=sh ¼ 2.5.**


> **FIG. 11. Spatiotemporal evolution of the liquid jet free-surface for case II. Two perpendicular views are presented, a side view (XY) and a top view (ZX) into the region of the droplet. The iso-surface is colored by the velocity magnitude (u�¼ 1000u). The horizontal red line marks the region of initiation of the wave whip (y=Di ¼ 3:44), and the diagonal red lines shows the translation of the node of the wave whip. In this case, the working condition is CaE ¼ 0:42 and WeE ¼ 20:45.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-11

V C Author(s) 2023

flow field show to us the pulling and pushing of the jet in different regions. Notice that positive values mean that the flow has been rotated clockwise and negative values means rotated in the opposite direction. It is interesting to note the parasite vorticity at the tip of the Taylor cone, which indicates that there is a grow in the instability and a wave is produced. As we observed in the previous results, for large CaE numbers, the dynamics is very complex and presents a chaotic behavior. It is important to analyze the average behavior of the dynamics. Since a normal average in time would buffer important characteristics of the liquid dynamics, we devised an interesting alternative way to show the average of the instabilities. This is done by overlapping the isosurfaces of the free surface of the liquid and changing its color with time. This type of plot can be observed at Fig. 13. The figure shows the temporal dynamics of the liquid from a full temporal perspective and for different electric potential. We notice that, with the increase on the electric potential, the instabilities of the liquid jet become wider, although, after a certain point, there is no pollution by small droplets. The readers should remember that these small droplets generated in the receding time region tr (see Fig. 7) are very chaotic, divided mainly by the velocity field created, and this can be a problem, due to the creation of scattered droplets. For example, in jet printing applications, it can be

helpful to contain structures to ensure high precision. We should observe that by increasing the electric potential to CaE ¼ 0:38, we remove the generation of this type of droplet by smashing the receding time (see Fig. 7), and creating a continuous liquid ejection. With a continuous liquid ejection, the scattering of small droplets (and nonprecision deposition) can be avoided. If we do not want too much deposition in one place, we can increase the collector velocity. Although as was mentioned previously, this is a very narrow operating window, if we increase the electric potential, we will create droplet bursts in the middle of the liquid jet or even a multi-jet operation. As we recall, the last mentioned modes of operation are not covered in this work, on purpose, because we determined that a turbulence model must be applied in that case, because the radial scattering of the liquid jet on the domain will be enlarged.

V. CONCLUSIONS

In this paper, we presented a fully three-dimensional simulation for an electrohydrodynamic jet using the Volume of Fluid method coupled with the Maxwell Stress Tensor and electric charge conservation. This 3D two phase conservative EHD model has been validated against experiments, showing a good accuracy on the computation of the temporal Taylor cone formation, also on the jet emission and on


> **FIG. 12. (a) Iso-surface of the liquid jet, pointing toward the node of the whip, colored by the normalized electric flux density D ¼ qeu (b) Two-dimensional planar cut, showing the instantaneous pattern of streamlines of the electric field E and the iso-lines of the electric potential /e. (c) Instantaneous pattern of streamlines of the velocity field and isolines of the vorticity x ¼ @ux @x �@uy @y.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-12

V C Author(s) 2023

the jet/droplet impact on the surface, showing an improvement when compared to axis-symmetric present in the literature, demonstrating power, capability, and high accuracy of this type of model. The fully three-dimensional model allowed the visualization of the spatiotemporal evolution of the electrohydrodynamic liquid jet, when injected from a capillary nozzle, and tested its stability for different conditions of electric potential. The instability of the flow is due to the leakage of electric charges from the liquid jet, this tiny fraction of electric charges on the bulk creates a non-symmetric flow that affects the electric field and, thus, the electric body force on the liquid interface. In the region of jet breakup, competition for the stability of the liquid jet occurs because of the extreme increase in the electric flux density, leading to complex flow patterns such as charged droplet scattering due to the velocity field generated, or whipping of the liquid jet due to the radial component of the electric field. The operating conditions of droplet emission without the formation of the main Taylor cone where visualized, including secondary and satellite droplets that are generated in a chaotic way. The authors suggest some possibilities for the proposed numerical model. First, the droplets generated by the receding Taylor cone can be useful for printing jet applications to increase the surface area and connect the main droplets, although the control of these droplets is extremely difficult. To use this effect, the transient mode between single droplet ejection and whipping mode needs to be carefully designed. Our model is automated with a script, in which the user input the nozzle dimensions, inner/outer diameters, and length, allowing to be used for parametric studies easily. Further investigation concerning multi jet simulations by including turbulence effects in the small droplets that we observe, one the main reason why our simulation did not increase further the CaE number and this is a future topic to explore.

ACKNOWLEDGMENTS

This work was supported with Portuguese National Funds by FCT, Foundation for Science and Technology, I.P., through the individual research under Grant No. 2020.04517. B.D. The work was also supported by C-MAST Center for Mechanical and Aerospace


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq017.png)

AUTHOR DECLARATIONS Conflict of Interest

The authors have no conflicts to disclose.

Author Contributions


![Equation](images/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets_eq018.png)

DATA AVAILABILITY

The data that support the findings of this study are available from the corresponding author upon reasonable request.

APPENDIX: GRID SENSITIVITY


> **Figure 14 shows the sensitivity of the results to the grid. Although the minimum grid size varies by just 0.5 lm, this represents 3% of the inlet of the nozzle and a meaningful 30% of the jet diameter. Moreover, since this a full tri-dimensional geometry the increase on the number of cells from the grid with dx ¼ 3:0 lm to the dx ¼ 2:0 lm is from 4 M cells to 11 M cells, which shows how an increase in just 1 lm as a big computational cost. A grid of dx ¼ 4:0 lm was also tested, although it did not produce meaningful results since there was no jet formation. The morphology of the Taylor cone is the most important characteristic of this type of flow. As we can observe, the curvature of the Taylor cone for these three grid spacings is identical, the most significant change is on the emitted jet. The grid with dx ¼ 3:0 cannot produce a steady jet as it should, but the other two give a very close solution and, are coherent with the experimental results (Fig. 3).**


> **FIG. 13. Visualization of the full dynamics of the liquid jet under increasing CaE numbers. Iso-surfaces colored by time tc. Full time of overlapping is 1.5 ms. The collector plate is colored by the intensity of the electric field E. (a) CaE ¼ 0.26, (b) CaE ¼ 0.32, and (c) CaE ¼ 0.42.**


## Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-13

V C Author(s) 2023

## References

1C. Cong, X. Li, W. Xiao, J. Li, M. Jin, S. H. Kim, and P. Zhang, “Electrohydrodynamic printing for demanding devices: A review of processing and applications,” Nanotechnol. Rev. 11, 3305–3334 (2022). 2M. Gamero-Casta~no and M. Magnani, “Numerical simulation of electrospraying in the cone-jet mode,” J. Fluid Mech. 859, 247–267 (2019). 3A. M. Ga~n�an-Calvo, J. M. L�opez-Herrera, M. A. Herrada, A. Ramos, and J. M. Montanero, “Review on the physics of electrospray: From electrokinetics to the operating conditions of single and coaxial Taylor cone-jets, and AC electrospray,” J. Aerosol Sci. 125, 32–56 (2018). 4J. U. Park, M. Hardy, S. J. Kang, K. Barton, K. Adair, D. K. Mukhopadhyay, C. Y. Lee, M. S. Strano, A. G. Alleyne, J. G. Georgiadis, P. M. Ferreira, and J. A. Rogers, “High-resolution electrohydrodynamic jet printing,” Nat. Mater. 6, 782–789 (2007). 5M. S. Islam, B. C. Ang, A. Andriyana, and A. M. Afifi, “A review on fabrication of nanofibers via electrospinning and their applications,” SN Appl. Sci. 1, 1248 (2019). 6Y. Pan and L. Zeng, “Simulation and validation of droplet generation process for revealing three design constraints in electrohydrodynamic jet printing,” Micromachines 10, 94 (2019). 7X. Suo, K. Zhang, X. Huang, D. Wang, H. Jia, F. Yang, W. Zhang, J. Li, L. Tu, and P. Song, “Electrospray beam currents in the cone-jet mode based on numerical simulation,” Phys. Fluids 35, 013603 (2023). 8M. R. Pendar and J. C. P�ascoa, “Numerical modeling of electrostatic spray painting transfer processes in rotary bell cup for automotive painting,” Int. J. Heat Fluid Flow 80, 108499 (2019). 9L. L. F. Agostinho, C. U. Yurteri, E. C. Fuchs, and J. C. M. Marijnissen, “Monodisperse water microdroplets generated by electrohydrodynamic atomization in the simple-jet mode,” Appl. Phys. Lett. 100, 244105 (2012). 10A. M. Ga~n�an-Calvo, J. M. L�opez-Herrera, N. Rebollo-Mu~noz, and J. M. Montanero, “The onset of electrospray: The universal scaling laws of the first ejection,” Sci. Rep. 6, 32357 (2016). 11J. Rosell-Llompart, J. Grifoll, and I. G. Loscertales, “Electrosprays in the cone-jet mode: From Taylor cone formation to spray development,” J. Aerosol Sci. 125, 2–31 (2018). 12H. Dastourani, M. R. Jahannama, and A. Eslami-Majd, “A physical insight into electrospray process in cone-jet mode: Role of operating parameters,” Int. J. Heat Fluid Flow 70, 315–335 (2018). 13W. Wei, Z. Gu, S. Wang, Y. Zhang, K. Lei, and K. Kase, “Numerical simulation of the cone-jet formation and current generation in electrostatic spray–modeling as regards space charged droplet effect,” J. Micromech. Microeng. 23, 015004 (2013). 14G. Tomar, D. Gerlach, G. Biswas, N. Alleborn, A. Sharma, F. Durst, S. W. J. Welch, and A. Delgado, “Two-phase electrohydrodynamic simulations using a volume-of-fluid approach,” J. Comput. Phys. 227, 1267–1285 (2007).

15Q. Yang, B. Q. Li, Z. Zhao, J. Shao, and F. Xu, “Numerical analysis of the Rayleigh-Taylor instability in an electric field,” J. Fluid Mech. 792, 397–434 (2016). 16Q. Yang, B. Q. Li, and F. Xu, “Electrohydrodynamic Rayleigh-Taylor instability in leaky dielectric fluids,” Int. J. Heat Mass Transfer 109, 690–704 (2017). 17M. R. Pendar and J. C. P�ascoa, “Numerical analysis of charged droplets size distribution in the electrostatic coating process: Effect of different operational conditions,” Phys. Fluids 33, 033317 (2021). 18D. A. Kessler and M. Merrill, “A Lagrangian-Eulerian method for simulating electrospray deposition,” AIAA Paper No. AIAA 2019-3722, 2019. 19A. K. Arumugham-Achari, J. Grifoll, and J. Rosell-Llompart, “Two-way coupled numerical simulation of electrospray with induced gas flow,” J. Aerosol Sci. 65, 121–133 (2013). 20K. Luo, T. F. Li, J. Wu, H. L. Yi, and H. P. Tan, “Mesoscopic simulation of electrohydrodynamic effects on laminar natural convection of a dielectric liquid in a cubic cavity,” Phys. Fluids 30, 103601 (2018). 21Q. Yang, B. Q. Li, and Y. Ding, “3D phase field modeling of electrohydrodynamic multiphase flows,” Int. J. Multiphase Flow 57, 1–9 (2013). 22Q. Yang, B. Q. Li, J. Shao, and Y. Ding, “A phase field numerical study of 3D bubble rising in viscous fluids under an electric field,” Int. J. Heat Mass Transfer 78, 820–829 (2014). 23M. Shen, B. Q. Li, and Q. Yang, “A 3-D phase field study of dielectric droplet impact under a horizontal electric field,” Int. J. Multiphase Flow 162, 104385 (2023). 24S. K. Das, A. Dalal, and G. Tomar, “Electrohydrodynamic-induced interactions between droplets,” J. Fluid Mech. 915, A88 (2021). 25Z. Wang, Y. Tian, C. Zhang, Y. Wang, and W. Deng, “Massively multiplexed electrohydrodynamic tip streaming from a thin disc,” Phys. Rev. Lett. 126, 064502 (2021). 26J. M. L�opez-Herrera, S. Popinet, and M. A. Herrada, “A charge-conservative approach for simulating electrohydrodynamic two-phase flows using volumeof-fluid,” J. Comput. Phys. 230, 1939–1955 (2011). 27M. Rahmanpour and R. Ebrahimi, “Numerical simulation of electrohydrodynamic spray with stable Taylor cone–jet,” Heat Mass Transfer 52, 1595–1603 (2016). 28S. Candido and J. C. Pascoa, “Numerical analysis on the stability conditions of an electrohydrodynamic jet,” in ASME International Mechanical Engineering Congress and Exposition, Proceedings (IMECE), 2020. 29R. E. Wirz, A. L. Collins, A. Thuppul, P. L. Wright, N. M. Uchizono, H. Huh, M. J. Davis, J. K. Ziemer, and N. R. Demmons, “Electrospray thruster performance and lifetime investigation for the LISA mission,” AIAA Paper No. AIAA 2019-3816, 2019. 30J. R. Pedersen, B. E. Larsen, H. Bredmose, and H. Jasak, “A new volume-of-fluid method in openfoam,” in MARINE 2017 Computational Methods in Marine Engineering VII, edited by M. Visonneau, P. Queutey, and D. Le Touz�e (International Center for Numerical Methods in Engineering., 2017), pp. 266–278. 31B. E. Larsen, D. R. Fuhrman, and J. Roenby, “Performance of interfoam on the simulation of progressive waves,” Coastal Eng. J. 61, 380–400 (2019). 32S. C^andido and J. C. P�ascoa, “Numerical simulation of electrified liquid jets using a geometrical VoF method,” in ASME International Mechanical Engineering Congress and Exposition, 2021. 33L. Guo, Y. Duan, Y. A. Huang, and Z. Yin, “Experimental study of the influence of ink properties and process parameters on ejection volume in electrohydrodynamic jet printing,” Micromachines 9, 522 (2018). 34Y. Guan, S. Wu, M. Wang, Y. Tian, W. Lai, and Y. Huang, “Numerical analysis of electrohydrodynamic jet printing under constant and step change of electric voltages,” Phys. Fluids 34, 062005 (2022). 35V. R. Gopala and B. G. van Wachem, “Volume of fluid methods for immiscible-fluid and free-surface flows,” Chem. Eng. J. 141, 204–221 (2008). 36J. U. Brackbill, D. B. Kothe, and C. Zemach, “A continuum method for modeling surface tension,” J. Comput. Phys. 100, 335–354 (1992). 37P. Cifani, W. R. Michalek, G. J. M. Priems, J. G. M. Kuerten, C. W. M. van der Geld, and B. J. Geurts, “A comparison between the surface compression method and an interface reconstruction method for the VoF approach,” Comput. Fluids 136, 421–435 (2016). 38Y. Ouedraogo, E. Gjonaj, T. Weiland, H. D. Gersem, C. Steinhausen, G. Lamanna, B. Weigand, A. Preusche, A. Dreizler, and M. Schremb,

FIG. 14. Results from grid sensitivity studies. Comparison of the iso-surface for different resolutions of the grid dx (lm), results depicted for the time t ¼ 0.7 ms.

Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-14

V C Author(s) 2023

“Electrohydrodynamic simulation of electrically controlled droplet generation,” Int. J. Heat Fluid Flow 64, 120–128 (2017). 39J. Roenby, H. Bredmose, and H. Jasak, “A computational method for sharp interface advection,” R. Soc. Open Sci. 3, 160405 (2016). 40L. Gamet, M. Scala, J. Roenby, H. Scheufler, and J. L. Pierson, “Validation of volume-of-fluid OpenFOAMV R isoAdvector solvers using single bubble benchmarks,” Comput. Fluids 213, 104722 (2020). 41L.-M. Li, D.-Q. Hu, Y.-C. Liu, B.-T. Wang, C. Shi, J.-J. Shi, and C. Xu, “Large eddy simulation of cavitating flows with dynamic adaptive mesh refinement using OpenFOAM,” J. Hydrodyn. 32, 398–409 (2020). 42M. P. Boruah, A. Sarker, P. R. Randive, S. Pati, and K. C. Sahu, “Tuning of regimes during two-phase flow through a cross-junction,” Phys. Fluids 33, 122101 (2021). 43E. Lac and G. M. Homsy, “Axisymmetric deformation and stability of a viscous drop in a steady electric field,” J. Fluid Mech. 590, 239–264 (2007). 44W. Yang, H. Duan, C. Li, and W. Deng, “Crossover of varicose and whipping instabilities in electrified microjets,” Phys. Rev. Lett. 112, 054501 (2014). 45H. H. Xia, A. Ismail, J. Yao, and J. P. Stark, “Scaling laws for transition from varicose to whipping instabilities in electrohydrodynamic jetting,” Phys. Rev. Appl. 12(1), 014031 (2019).

46A. M. Ga~n�an-Calvo, J. D�avila, and A. Barrero, “Current and droplet size in the electrospraying of liquids. Scaling laws,” J. Aerosol Sci. 28, 249–275 (1997). 47S. Yang, Z. Wang, Q. Kong, B. Li, and J. Wang, “Visualization on electrified micro-jet instability from Taylor cone in electrohydrodynamic atomization,” Chin. J. Chem. Eng. 44, 456–465 (2022). 48W. van Hoeve, S. Gekle, J. H. Snoeijer, M. Versluis, M. P. Brenner, and D. Lohse, “Breakup of diminutive Rayleigh jets,” Phys. Fluids 22, 122003 (2010). 49Y. Pan and K. Suga, “A numerical study on the breakup process of laminar liquid jets into a gas,” Phys. Fluids 18, 052101 (2006). 50M. Magnini, F. Municchi, I. E. Mellas, and M. Icardi, “Liquid film distribution around long gas bubbles propagating in rectangular capillaries,” Int. J. Multiphase Flow 148, 103939 (2022). 51J. Fern�andez De La Mora and I. G. Loscertales, “The current emitted by highly conducting Taylor cones,” J. Fluid Mech 260, 155–184 (1994). 52E. Grustan-Gutierrez and M. Gamero-Casta~no, “Microfabricated electrospray thruster array with high hydraulic resistance channels,” J. Propul. Power 33, 984–991 (2017). 53Z. Jiang, Y. Gan, and Y. Shi, “An improved model for prediction of the cone-jet formation in electrospray with the effect of space charge,” J. Aerosol Sci. 139, 105463 (2020).

Physics of Fluids ARTICLE pubs.aip.org/aip/pof

Phys. Fluids 35, 052110 (2023); doi: 10.1063/5.0151109 35, 052110-15

V C Author(s) 2023

