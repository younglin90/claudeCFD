
# A charge-conservative approach for simulating electrohydrodynamic two-phase flows using volume-of-fluid


## J.M. López-Herrera a,⇑, S. Popinet b, M.A. Herrada a

a Dept. Ingeniería Aerospacial y Mecánica de Fluidos, E.S.I., Universidad de Sevilla, Camino de los Descubrimientos s/n, 41092 Seville, Spain b National Institute of Water and Atmospheric Research, P.O. Box 14-901, Kilbirnie, Wellington, New Zealand

a r t i c l e i n f o

Article history: Received 17 June 2010 Received in revised form 22 November 2010 Accepted 26 November 2010 Available online 4 December 2010

Keywords: VOF Electrohydrodynamics EHD Interfacial flows Two-phase flows Gerris

a b s t r a c t

In the present study we propose a charge-conservative scheme to solve two-phase electrohydrodynamic (EHD) problems using the volume-of-fluid (VOF) method. EHD problems are usually simplified by assuming that the fluids involved are purely dielectric (insulators) or purely conducting. Gases can be considered as perfect insulators but pure dielectric liquids do not exist in nature and insulating liquids have to be approximated using the ‘‘Taylor– Melcher leaky dielectric model’’ [1,2] in which a leakage of charge through the liquid due to ohmic conduction is allowed. It is also a customary assumption to neglect the convection of charge against the ohmic conduction. The scheme proposed in this article can deal with any EHD problem since it does not rely on any of the above simplifications. An unrestricted EHD solver requires not only to incorporate electric forces in the Navier– Stokes equations, but also to consider the charge migration due to both conduction and convection in the electric charge conservation equation [3]. The conducting or insulating nature of the fluids arise on their own as a result of their electric and fluid mechanical properties. The EHD solver has been built as an extension to Gerris, a free software solver for the solution of incompressible fluid motion using an adaptive VOF method on octree meshes developed by Popinet [4,5]. �2010 Elsevier Inc. All rights reserved.

1. Introduction

Electrohydrodynamics (EHD) describes the motion of liquids subjected to electric fields. Typically the liquid will be set in motion by electrical stresses, thereby modifying the geometry and charge distribution, which in turn modifies the electric field. Under the influence of an electric field two effects occur in a fluid: the fluid molecules may get polarized, giving rise to dipoles, and an ohmic migration of charged ions/free electrons through the fluid is induced. This leads to two distinct limits in the electrical behavior of a fluid: perfect dielectric and perfect conductor. A perfect dielectric fluid is a fluid without any ion or free-electron, only polarization effects being present. If polarization effects are homogeneous, electrical forces only appear at the fluid interface, where dipoles are unbalanced, and act in the normal direction. Apolar liquids such a benzene are considered dielectric fluids, however most liquids are known for their ability to dissolve impurities by creating ionic pairs, and could hardly be considered perfect dielectric fluids. Therefore these fluids have to be considered to some extent as conducting [1,2]. Perfect conductors are those where the conductivity is high enough to consider the ohmic conduction as the only agent causing charge transport. In this limit it is assumed that the free charges migrate instantaneously from the bulk to the fluid interface, which becomes an electric equipotential surface. Leaky dielectrics are different because a

0021-9991/$ - see front matter �2010 Elsevier Inc. All rights reserved. doi:10.1016/j.jcp.2010.11.042

⇑Corresponding author. E-mail addresses: jmlopez@us.es (J.M. López-Herrera), s.popinet@niwa.co.nz (S. Popinet), herrada@us.es (M.A. Herrada).

Journal of Computational Physics 230 (2011) 1939–1955

Contents lists available at ScienceDirect


journal homepage: www.elsevier.com/locate/jcp

tangential electrical stress appears at the interface, setting the fluid in motion until viscous stresses provide balance. Saville [3] used a scaling analysis to rigorously derive the Taylor–Melcher leaky dielectric model while identifying the approximations made. Note that Saville retains the temporal term and the convection of charges term in the charge conservation equation, as well as the electrical body force terms that Melcher and Taylor ignored. Electric forces may be used to control and handle fluids in several ways. For example, many technical and industrial processes which require supplying liquids in the form of small droplets, such as ink jet printing, fuel atomization, or many biotechnological applications, involve the breakup of charged jets [6–8], which is referred to as EHD liquid spraying. Another substantial application area is based on the inducement of a fluid bulk motion by charge injection from metallic tip or blades [9,10]. A subject of increasing importance is the design and characterization of microfluidic devices [11]. Many basic operations that occur in these devices such as generation, translocation, merging or fission of droplets are carried out by a careful manipulation of electric fields [12–14]. The solution of the Navier–Stokes equations with a free surface or interface is not an easy task; this is complicated further when electrostatic effects are coupled to the fluid dynamics. Thus most EHD problems have been addressed experimentally [15,16] or with simplified theoretical models [17,18]. Numerical approaches are sometimes the only available option for simulating complex interdisciplinary phenomena occurring in complex geometries. The preferred numerical scheme in these simulations is the Boundary Element Method (BEM), which is used to solve either the electric field or the flow pattern [19–22]. However, the BEM method is only applicable to the solution of problems in the limit of inviscid or Stokes flows. Finite element methods (FEM) have been used in the study of the breakup of charged jets [23] or pendant droplets formation in electric fields [24]. Several methods can be used to describe the moving interface: in tracking methods a set of marker points is used to locate the interface; level-set methods describe the interfacial geometry through an implicit function of the distance to the interface; and volume-of-fluid methods (VOF) use a volume fraction field. In Fern’andez et al. [25] the front-tracking method is extended to account for electric fields and applied to evaluate droplet distribution in a channel. The level-set method has been adapted to EHD problems in Teigen and Munkejord [26]. In this work very accurate results are obtained by treating the discontinuities with a ghost-fluid method, but the model is restricted to perfect dielectric fluids. Tomar et al. [27] proposed a different, and very accurate, methodology for computing electrical forces; since in most situations the only electrical forces are located at the free interface, they make use of the Continuum-Surface-Force (CSF) approach devised by Brackbill [28] to model interfacial electric stresses. Unfortunately this approach is only applicable if both fluids behave as perfect dielectrics or perfect conductors. Commercial codes such as Fluent, Flow3D or CFX 4.4 are experiencing a growing use as tools for scientific studies. These codes provide models to simulate the Navier–Stokes equations but have to be extended and adapted, with more or less flexibility, for multidisciplinary subjects, among them EHD problems. Zeng and Korsmeyer [11] extended Flow3D to simulate droplet-based labs-on-a-chip, while Sen et al. [29] used Flow3D to analyze electrospray ionization and Lastow and Balachandran [30] adapted CFX 4.4 to study EHD atomization. In general details like the implementation or convergence of the numerical treatment of the EHD extension with these commercial codes are not available leaving to their users the responsibility of a reliable validation of the computed results. The present work proposes a conservative approach to deal with two-phase EHD problems using the VOF method. The proposed method does not require any simplifications of the electrical behavior of the fluids involved. An unrestricted EHD problem requires not only to incorporate electric forces into the Navier–Stokes equation, but also to consider the charge migration due to both conduction and convection. The EHD code has been built as an extension of the Gerris solver [4,31]. Gerris combines an adaptive quad/octree spatial discretisation with a VOF approach to solve incompressible two-phase fluid motions. Gerris can accurately simulate surface-tension-driven flows using a combination of balanced-force CSF and a height-function estimation of the curvature of the interface [5]. The present paper is organized as follows. In Section 2 the complete EHD equations are developed. The numerical methodology used is described in Section 3, paying special attention to the numerical treatment of the electric forces and the charge continuity equation. The proposed model is tested in Section 4 using analytical problems and finally the main conclusions are presented in Section 5.

2. Governing equations

The set of equations governing the incompressible fluid motion are the continuity and momentum equations,

r �u ¼ 0; ð1Þ


### q @u @t þ u �ru � � ¼ �rp þ r �Tv þ Fe þ rjdsn; ð2Þ

where q is the fluid density, u is the velocity vector, r the surface tension coefficient, j the interface curvature and n the normal to the interface. The surface tension term only acts on the interface. This is represented using the Dirac distribution function ds. Tv is the viscous stress tensor given by,


## Tv ¼ 2lD; ð3Þ

1940 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

where l is viscosity and D is the deformation tensor, D ¼ 1 2 ðru þ ruTÞ. Fe stands for the volume electric forces. To compute Fe, Maxwell’s electromagnetic equations need to be considered. In electrohydrodynamic flows, the magnetic effects can be ignored and the electrostatic equations are an accurate approximation since, as pointed out by Saville [3], the characteristic time for the magnetic phenomena tm �lMK‘2 (lM is the magnetic permeability, K is the conductivity and ‘ the characteristic length) is several orders of magnitude smaller than the characteristic time for electric phenomena i.e. the electric relaxation time te �e/K, where e is the electric permittivity.1 Accordingly the electrical phenomena are described by:


## r �ðeEÞ ¼ qe and r �E ¼ 0; ð4Þ


### where qe is the volumetric charge density and E the electric field. In terms of the electric potential, /, the electrostatic limit follows the Poisson equation,


## r �ðer/Þ ¼ �qe: ð5Þ

Finally, the conservation equation of the bulk free charge should be imposed,


### @qe @t þ r �J ¼ 0; ð6Þ

where J is the vector current density (flux of electric charge) given by


### J ¼ KE þ qeu: ð7Þ

The first term is the ohmic charge conduction while the second is due to the convection of charges. Taking into account the electrostatic relationship (4) the conduction term can be further developed and Eq. (6) can be written as


### @qe @t þ r �ðqeuÞ ¼ �K e qe þ E � K e re �rK � � : ð8Þ


### If the electrical properties of the fluid K and e are homogeneous, this reduces to


### @qe @t þ r �ðqeuÞ ¼ �K e qe: ð9Þ

The volumetric electric forces in the bulk Fe can be derived from the electrostatic Maxwell stress tensor


## Te ¼ e EE �E2

2 I

!


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq001.png)

by applying the divergence operator


### Fe ¼ r �Te ¼ qeE �1 2 E2re: ð11Þ

The first term represents the electric forces exerted on the free charges seeded in the fluid, while the second term represents the electric forces exerted on the electric dipoles induced in dielectric mediums.


> **Fig. 1. Sketch of the fluid–fluid interface.**

1 For deionized water lM �10�6 H/m, e �10�11 F/m and K �10�6 S/m, gives for a millimetric scale ‘ �10�3 m: tm/te �10�13.

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1941

In two-phase flows an interface separates the non-miscible fluids (the media 1 and 2 as sketched in Fig. 1). The interface is free to move, its position being given by the equation F(x,t) = 0. We denote the normal and the tangent vectors to the free surface by n and t (for the sake of simplicity we adopt a bi-dimensional description where a single tangential vector is considered). The evolution of the interface is governed by the kinematic condition

@F @t þ u �rF ¼ 0: ð12Þ

Some quantities are continuous through the interface such as the velocity, the electric potential k/k = 0, and the tangential component of the electric field Et = E �t, kEtk = 0 where kk denotes the jump across the interface. Note that the continuity of the tangential component of the electric field is a consequence of the continuity of the electric potential. Therefore imposing both conditions is redundant. The stress balance at the interface should be satisfied in the tangential direction

t �kTvk �n þ t �kTek �n ¼ 0 ð13Þ

and in the normal direction


## kpk þ n �kTvk �n þ n �kTek �n ¼ rj: ð14Þ

The pressure is not continuous through the interface due to surface tension and the normal electrical stresses acting on the interface. The normal electric field En = E �n is also discontinuous through the interface


## keEnk ¼ q ð15Þ

with q the free charge per unit area accumulated at the interface. The expressions of the electrical tangential and normal stresses acting on the interface are respectively


## t �kTek �n ¼ ðe1En1 �e2En2ÞEt ¼ q Et ð16Þ

and

n �kTek �n ¼ 1 2 e1E2 n1 �e2E2 n2 �ðe1 �e2ÞE2 t h i ; ð17Þ

where the continuity of Et has been used. Finally a conservation equation for the surface electrical charge density q should be satisfied

@q @t þ u �rsq �qn �ðn �rÞ �u þ kKEnk ¼ 0; ð18Þ

where rs denotes the surface divergence. This equation reflects how the surface charge density evolves in time due to surface charge convection (second term), the dilation of the interface (third term) and the net charge added/withdrawn from the bulk by ohmic conduction (fourth term). Depending on the conductivities and permittivities of the fluids several limits can be distinguished. If both fluids are dielectric the ohmic conduction is absent (K1 = K2 = 0) and the fluid is free of charges (qe = 0). Under these conditions the equation for the potential (5) reduces to the Laplace equation and the electric forces are acting only at the interface in the normal direction, see Eq. (16). Another limit is observed when both fluids are perfect conductors. This limit is reached when the electric relaxation time te of both fluids is much shorter than the characteristic hydrodynamic time th which depends on the problem considered; for example in slightly viscous capillary droplets and jets th is the capillary time given by (qD3/r)1/2 where D is the diameter; in other problems th is the residence time ‘/Uo or the viscous time (q‘2/l). The much shorter time scale for electric phenomena leads to an essentially instantaneous charge migration through the fluid so that the fluid bulk becomes free of charges (qe = 0). In this limit, Eq. (5) reduces to the Laplace equation. Eq. (18) is approximated by kK Enk = 0 and the electric forces are acting only at the interface. If one fluid is a perfect conductor and the other is dielectric the relaxed electric charge accumulates at the free surface. The electric field in the conducting domain is negligible compared to the dielectric field, the free surface can then be assumed equipotential and there is no electrical tangential stress on the interface. In a general case, electric relaxation and hydrodynamic times would be of the same order, te �th, and the terms for charge migration, convection and conduction, would be comparable in Eqs. (6) and (18). For example, using the fluid properties for deionized water (e = 80eo; eo = 8.85 �10�12 F/m; K = 10�5 S/m; q = 103 kg m�3 and l = 10�3 Pa s), we get te/th �7 �10�5 at the millimetric scale (‘ = 10�3 m) but if the length scale of the problem, ‘, is 10�5 m (microfluidics), one would obtain te/ th �1.

3. Numerical scheme

Gerris is an open-source solver for the solution of incompressible fluid motion using the finite-volume approach. It was developed by Popinet [4,5]. Gerris uses the volume-of-fluid (VOF) method to deal with two-phase flows. In this method the Navier–Stokes equations are written as

1942 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

r �u ¼ 0;


### q @u @t þ u �ru � � ¼ �rp þ r �ð2lDÞ þ rjdsn þ Fe;

@c @t þ r �ðc uÞ ¼ 0;


### q ¼ cq1 þ ð1 �cÞq2; l ¼ cl1 þ ð1 �cÞl2;


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq002.png)

where a variable c has been added which enables tracking of the interface position; this is the volume fraction, c(x,t). The surface tension stress is modeled as a fluid bulk volumetric force using the Continuum-Surface-Force (CSF) approach of Brackbill [28]. This method can suffer from parasitic currents which are avoided using a balanced-force description of the surface tension and pressure gradient together with an accurate curvature estimate [5]. Gerris makes use of a staggered-in-time discretisation, which is second-order accurate, combined with a time-splitting projection method. Combined with the discretisation of the electric field equation and the charge evolution equation this gives the following timestepping scheme

cnþ1 2 �cn�1 2 Dt þ r �ðcnunÞ ¼ 0; ð20Þ


### ðqeÞnþ1 2 �ðqeÞn�1 2 Dt þ r �ðqeÞnun þ Kn�1 2En�1 2


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq003.png)


## r �rðenþ1=2Unþ1=2Þ � � ¼ �ðqeÞnþ1=2: ð22Þ


### qnþ1 2

u��un Dt þ unþ1 2 �runþ1 2

� � ¼ r �lnþ1 2ðDn þ D�Þ � � þ rjdsn ð Þnþ1 2 þ ðFeÞnþ1 2; ð23Þ

unþ1 ¼ u��Dt qnþ1 2 rpnþ1 2; ð24Þ

r �unþ1 ¼ 0; ð25Þ

where the * subscript indicates that the value of the corresponding variable is provisional. Combining Eqs. (24) and (25) of the above set results in the following Poisson equation:

r � Dt qnþ1 2 rpnþ1 2

!


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq004.png)

The momentum Eq. (23) can be reorganized as


### qnþ1 2 Dt u��r �ðlnþ1 2D�Þ ¼ r �ðlnþ1 2DnÞ þ ðrjdsnÞnþ1 2 þ ðFeÞnþ1 2 þ qnþ1 2


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq005.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq006.png)

where the velocity advection term unþ1 2 �runþ1 2 is estimated by means of the Bell–Colella–Glaz second-order unsplit upwind scheme [4,32]. Space is discretised using an octree where the unknown variables are located at the center of each cubic discretisation volume and are interpreted as the average value of the variable in the cell. The octree discretisation used in Gerris allows an efficient mesh refinement or coarsening. The mesh can be adapted at every time-step on demand with a minimal impact on overall performance. In the above equations the spatial values of the electrical properties follows from the volume fraction c. Similarly to the fluid properties q and l, the electric properties can be interpolated using the weighted arithmetic mean interpolation (WAM) that writes e ¼ ce1 þ ð1 �cÞe2 and K ¼ cK1 þ ð1 �cÞK2: ð28Þ

Tomar et al. [27] obtained much more accurate results using the weighted harmonic mean interpolation (WHM) given by

1 e ¼ c e1 þ ð1 �cÞ e2 and 1 K ¼ c K1 þ ð1 �cÞ K2 ð29Þ

Tomar et al. [27] analyzed the influence of the interpolation scheme applied to the permittivity since it is the only relevant property in their work. In the proposed scheme we investigate the influence of the interpolation scheme when applied to both electrical properties, the permittivity and the conductivity, using the following combinations: (a) both e and K uses WAM, (b) e uses WHM and K uses WAM and (c) both e and K uses WHM. The numerical procedure for a timestep is as follows:

1. Cell centered volume fraction at the intermediate timestep cc nþ1=2 are calculated from Eq. (20) using a VOF scheme. 2. Charge density at the intermediate timestep, qe ð Þc nþ1=2, is then calculated from Eq. (21) where the advection term is estimated with the Bell–Colella–Glaz second-order unsplit upwind scheme.

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1943

3. The electric potential at instant n + 1/2, /n+1/2, is calculated by solving the electric potential Poisson Eq. (22). The efficient octree multilevel Poisson solver described in Popinet [4] is reused. The electric field is then straightforwardly computed as En+1/2 = �r/n+1/2. 4. The electric body force (Fe)n+1/2 is computed from En+1/2. 5. The auxiliary cell-centered velocity uc �is calculated from the Helmholtz-type Eq. (27) using a variant of the multilevel Poisson solver. 6. The pressure at time n + 1/2 is computed by solving the Poisson Eq. (26) with the multilevel solver. 7. The cell-centered velocity field un+1 is computed using a cell-centered approximation of Eq. (24).

We refer the reader to Popinet [4,5] (and references cited therein) for a more detailed presentation of the quad/octree data structure and the numerical integration procedure of the incompressible Navier–Stokes scheme.

3.1. Finite-volume approximation of the electrical forces Fe

The volume-averaged electrical forces at step n + 1/2 can be written in each discretised cell C as Z

C ðFeÞnþ1=2 ¼ Z


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq007.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq008.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq009.png)

or simply


### ðFeÞnþ1=2 ¼ ðqeÞnþ1=2Enþ1=2 �1 2 ðE2Þnþ1=2renþ1=2: ð31Þ

The above equation applies either in cells fully immersed in the bulk (volume fraction c = 1 or c = 0) or in cells crossed by the interface (0 < c < 1). In the most usual situations, such as the limit cases described in the previous section, (Fe)n+1/2 would only be different from zero in cells crossed by the interface. In interfacial cells Eq. (31) should be able to describe the dynamical effects of the interfacial electrical stresses given by Eqs. (16) and (17). Note that in an interfacial cell either the electric field or the permittivity suffers very abrupt changes. An accurate evaluation of (Fe)n+1/2 by means of Eq. (31) will then be difficult since it would require very accurate center values of the terms involved, i.e. r e, (E2),. . . This numerical inconvenience is more evident if the mesh is refined. Note also that Eq. (11) (the continuum version of Eq. (31)) is broadly used in previous works, such as those extending commercial codes to deal with EHD problems [33,30]. Tomar et al. [27] show that a suitable CSF treatment of the electric field force gives good results for electrohydrodynamic problems in the limit cases of dielectric–dielectric and conducting–conducting fluids. So far a CSF treatment for the general case is not available since this approach relies on rewriting Eq. (17) in these two limits. Here we propose a general conservative approach to calculate (Fe)n+1/2. Using Gauss’ theorem we get

Z

C ðFeÞnþ1=2 ¼ Z


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq010.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq011.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq012.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq013.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq014.png)

where h is the cell size, Ef nþ1=2 is the component of the electric field normal to the cell face (evaluated at the cell face) and nf is the normal unit vector at the face. Computing the volumetric electric forces in a cell C as the resultant of electrical stresses acting at the cell face @C provides a formally exact and numerically conservative calculation of the electrical stresses acting at an interface; any remaining inaccuracy can be ascribed to the discrete character inherent to any numerical scheme.

3.2. Finite-volume approximation of the ohmic conduction term

In a similar spirit, we compute the ohmic conduction term of the discrete Eq. (21) through the values of variables located at the cell faces @C as

h r �ðKn�1=2En�1=2Þ � � ¼ Z


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq015.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq016.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq017.png)

This formulation ensures numerical conservation of the charge density. The charge conservation equation in the form given by expression (9) has sometimes been used to model two-phase EHD problems [34] despite not being valid uniformly in the domain. Indeed, close to the interface the electric properties undergo a steep jump and the additional terms of expression (8) should be included. For the sake of comparison, we have also made simulations using the form (9), for which the staggered time discretisation reads


### ðqeÞnþ1=2 �ðqeÞn�1=2 Dt þ r �½ðqeÞnun�¼ �Kn�1=2 en�1=2 ðqeÞn�1=2: ð34Þ

1944 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

4. Results and discussion

In this section we investigate the performance of the different schemes presented in the present study. The planar test cases allow us to show that: (a) the proposed approach gives accurate results irrespective of the electrical behavior of the fluids involved, (b) the electric forces are much more accurately calculated using scheme (32) than using (31). Section 4.2 is devoted to assess the superior accuracy of scheme (21) for two-phase flows, contrasting it to scheme (34). Finally, in Section 4.3 we show the applicability of the scheme to three-dimensional problems and we test the proposed scheme against a realistic EHD problem: the deformation of a droplet by an electric field.

4.1. Planar layers

A potential V is imposed between two parallel electrodes at a distance L. The gap between the electrodes is completely filled with two layers of different fluids having homogeneous electric properties as shown in Fig. 2(a). In this setup the problem is uni-dimensional and the electric potential decreases linearly along y, although generally at a different rate in each layer. As test cases we use the following limits: dielectric–dielectric (K1 = K2 = 0), conducting–conducting and dielectric–conducting (K2 = 0). In the cases where a conducting medium is present, we allow the electrostatics to evolve to a steady state from a starting initial condition (set as qe(x,t = 0) = 0), the electric forces being neglected in the transient stage. Once the steady state is reached, the pressure jump is calculated and compared to the exact value given by

Dpex ¼ p1 �p2 ¼ �1 2 ðEex 2 Þ2 �bðEex 1 Þ2 h i : ð35Þ

In Table 1 we summarize the analytical expressions for the electric potential and electric field in each medium, as well as the pressure jump. The electric potential has been made dimensionless with V, the length with L and the pressure jump with e2V2/L2. b and g are the ratio of permittivities, e1/e2, and conductivities, K1/K2, respectively. For the numerical test the free parameters have been set to the following values: dielectric–dielectric, b = 3, conducting–conducting, b = 2 and g = 3.


> **Table 2 shows the convergence of electric fields in medium 1 and 2, 1 �E1=Eex 1 and 1 �E2=Eex 2 , respectively, and pressure jumps using the schemes given by Eqs. (32) and (31) for the different test cases. The results shown in Table 2 have been computed using the WAM interpolation for the electric properties. As can be seen, using Eq. (31) gives rise to very large errors in the pressure jump, increasingly so as the grid is refined. These large errors have their origin in the steep electric permittivity gradient across the interface, which leads to an inaccurate estimate of the term �1 2 ðE2Þnþ1=2renþ1=2 at the center of interfacial cells. In contrast, the proposed scheme of Eq. (32) provides accurate results for every test case, converging to exact values as the grid is refined. Note that the error halves as the grid mesh doubles in accordance to a first-order in space convergence. The error in the pressure jump (second column in Table 2) is practically the sum of the error in the computed electric field in each medium (sum of third and fourth column). This reflects the fact that Eq. (32) allows a conservative calculation of the electrical stresses acting on every cell in the computational domain regardless of whether the cells are interfacial or not; the only prerequisite is an accurate evaluation of the electric field Welch and Biswas [35] also conclude that the electric forces are much more accurate when computed using the conservative divergence form. We have also investigated the influence of the interpolation scheme applied to the electrical properties in these planar test cases. Our results are in accordance with those reported by Tomar et al. [27]. For the dielectric–dielectric limit, using the WHM interpolation for the permittivity we obtain the exact values for both the electric field and the pressure jump irrespective of the grid adopted (32, 64 or 128 points). Exact values are also computed irrespective of the grid in the conducting– conducting limit when the WHM interpolation is applied to both the electrical properties. Note however that if WAM is used**


> **Fig. 2. Sketch of the geometry and electrical conditions used in: (a) the planar test cases, (b) the isolated conducting cylinder test case.**

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1945

for the conductivity (keeping WHM for the permittivity), the accuracy of the calculations reduces to the values shown in Table 2. Finally, in the dielectric–conducting configuration, the use of WHM for both properties does not lead to exact values but to roughly half the errors reported in Table 2, i.e. the errors with a grid of 32 points and WHM interpolation are similar to those using a grid of 64 points and WAM. However, it cannot be concluded that in general the WHM interpolation is better than the WAM interpolation. In fact, for a more complicated and realistic problem (see Section 4.3; in particular Fig. 7) we have obtained an accuracy similar for both interpolation schemes (with a slight increase in charge diffusion near the interface when using WHM compared to WAM). Thus, in the following we will use WAM except where indicated. An interesting issue we have found is that the electric force, regardless of the accuracy reached in its computation, can induce spurious numerical fluid currents. This is similar to what happens for naive implementations of CSF schemes for surface tension. We have been able to eliminate these spurious currents in the particular case of the planar dielectric–dielectric configuration by imposing a balanced-force description of the electric stresses similar to the one applied by Popinet [5] for surface tension. Generalising this balanced scheme to general interface configurations and electrical properties is non-trivial however and will be the subject of future work.

4.2. Time relaxation of a charge density distribution

In this subsection we compare the accuracy of the numerical scheme given by Eq. (21) with the scheme based on Eq. (34); both are used to simulate the time evolution of the charge density.

4.2.1. Bulk relaxation In this test case, a concentrated bump of charge density, initially set at the center of a square domain of width L, is allowed to relax freely in time. The entire domain is occupied by a single fluid, at rest, whose electrical properties are K and e, the electrical boundary condition being set at the border of the domain, / = 0. The initial shape of the bump is a Gaussian bell given by equation


### qeðx; t ¼ 0Þ ¼ e�r2=ð2a2Þ


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq018.png)

where r2 = x2 + y2 and a is a free parameter setting the width and height of the bell. If the domain borders are far enough from the concentrated charge bump, a �L, the problem has a simple analytical solution given by an exponential time decay of the bump


### qeðx; tÞ ¼ qeðx; t ¼ 0Þ e�Kt=e: ð37Þ


> **Table 1 Analytical dimensionless solutions for the planar test cases. Potentials in medium 1 and 2, /1 and /2, have been scaled with V, the length with L and the pressure jump with e2V2/L2. b and g are the ratios of permittivities, e1/e2, and conductivities, K1/K2, respectively.**

Test case /ex 1 Eex 1 /ex 2 Eex 2 Dpex

Dielectric–dielectric �2yþb 1þb


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq019.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq020.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq021.png)

Conducting–conducting �2yþg 1þg


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq022.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq023.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq024.png)

Dielectric–conducting 1 0 �2y + 1 2 �2


> **Table 2 Deviations from theoretical values of the computed pressure jump and electric field using the approach given by Eqs. (32) and (31), for different test cases and spatial resolutions. The electric properties have been interpolated using the WAM scheme.**

Grid Error (%) (1 �Dp/Dpex) Error (%) ð1 �E2=Eex 2 Þ Error (%) ð1 �E1=Eex 1 Þ


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq025.png)

Dielectric–dielectric 32 1.589 3489 0.796 0.787 64 0.816 7022 0.392 0.392 128 0.376 14033 0.196 0.196

Conducting–conducting 32 1.588 3295 0.787 0.787 64 0.783 6638 0.392 0.392 128 0.377 13323 0.196 0.196

Dielectric–conducting 32 6.571 3735.8 3.229 – 64 3.204 7328.9 1.591 – 128 1.591 14520.8 0.791 –

1946 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955


> **Fig. 4 illustrates the accuracy of the proposed EHD charge density model. In addition this test also shows that the mesh refinement capabilities of the octree spatial discretisation of Gerris are maintained when solving the EHD equations (see Fig. 3). A static mesh is used where cells contained within a circle of radius 0.19 centered on the origin have a level L ¼ 7 (cell size h ¼ 2�L), the cells then coarsen gradually away from the origin until a level of L ¼ 4 is reached for cells at the boundaries of the domain. In this particular test, the free parameters of the problem have been chosen as follows: L = 1, a = 0.05, e = 2 and K = 1. Not surprisingly, since there is a single medium, the same numerical performance is obtained regardless of whether the surface charge density equation is modeled with Eq. (21) or with Eq. (34), as shown in Fig. 4. In the upper plot we show that the numerical simulation reproduces the exponential temporal decay of the charge density at a given point. The selected point is the center of the square domain (x = 0,y = 0) where the maximum is located. In the lower plot we illustrate the decay of the Gaussian bump, whose geometry is preserved, with time (t = 0,2,4 and 6).**

4.2.2. Charge relaxation of an isolated conducting cylinder A conducting cylinder of radius R and electrical properties K1 and e1 is located at the center of a square domain of width L. The region between the cylinder and the square borders is filled with a second isolating medium (K2 = 0) of electrical permittivity e2. Initially, a uniform charge distribution qeo is set in the cylinder, the total charge per unit length of the cylinder being Q = pR2qeo. As time proceeds, the seeded charges repel each other, leading to accumulation of the free charge at the surface of the cylinder. Notice, however, that the global amount of charge in the cylinder, Q, should remain unchanged. The electric potential distribution in the dielectric medium remains unaltered with time since it depends only on Q. An analytic solution of the steady state is provided if the dielectric medium is assumed unbounded

EðrÞ ¼

Q 2pe2


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq026.png)

0 for r < R:

(


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq027.png)

In Fig. 5 we explore the accuracy of modeling the charge density equation with Eq. (21) and with Eq. (34) in a two-phase situation, using as benchmark the charge relaxation of an isolated conducting cylinder. The free parameters of the case are set to the following quantities: R = 0.05, L = 1, K1 = 3, e1 = 3, e = 2 and qeo = 0.5. In the upper plot we depict the time evolution of the total amount of charge in the domain, scaled with the initial amount of charge Q. It can be observed that the scheme based on Eq. (34) fails to conserve the total charge in the domain, in contrast with the accurate conservation behavior of the scheme based on Eq. (21). In the lower plot of Fig. 5 we show the spatial distribution of the electric field once the steady state is reached for different levels of mesh refinement. User-defined criteria can easily be used to adapt the mesh within Gerris. We have adopted a gradient criterion given by jrcjh < �, where c can be any of the variables. Gradient adaptivity allows to use high resolution in regions of large gradients. We chose to apply gradient adaptivity to the volume fraction to ensure a good description of the variables in the vicinity of the interface. The maximum resolution (on the interface) is set to Lmax (i.e. a cell size h ¼ L 2�Lmax). Cells coarsen further away from the interface reaching a minimum level, Lmin. In the adapted mesh depicted in Fig. 5 we have set Lmax ¼ 10 and Lmin ¼ 6. The electric field distribution in the dielectric medium is accurately recovered independently of the refinement of the mesh used in the simulations. As expected, a better description of the electric field jump across the interface is obtained as the mesh is locally refined.


> **Fig. 3. Isocontours of potential (red lines) at instant t = 0 and corresponding spatial discretisation. The isoncontour range is / = 0 �0.02 with D/ = 0.002 intervals. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)**

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1947

4.3. Electrohydrodynamic deformation of droplets

Finally in this section we study a more complete and realistic problem. We simulate the deformations experienced by a liquid droplet of radius Rd suspended in a bath of a second liquid when subjected to an imposed electrical electric field E1 as shown in Fig. 6. The liquids are immiscible and separated by an interface with surface tension coefficient c. Buoyant forces are absent since the densities of the fluids are identical. Due to the applied electric field the droplet deforms, eventually adopting a stable spheroidal form. The spheroid can be prolate (if the greatest deformation is produced in the direction of the applied electric field) or oblate (the largest deformations occurs perpendicularly to the electric field, see Fig. 6) depending on the electrical and fluid mechanical properties of the fluids involved. Within the literature dedicated to this problem (see [36] and the references therein), the work of Taylor [37] is especially relevant. Taylor characterises the total deformation of the droplet by means of the parameter D given by the expression

0 1 2 3 4 5 6 0

2

4

6

8

t


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq028.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq029.png)

0

2

4

6

8

x


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq030.png)

Num. simul. [Eq. (21)] Num. simul. [Eq. (34)] Analytical sol.


> **Fig. 4. Comparison between the analytical solution given by Eq. (37) and the numerical simulations modeling the charge density conservation equation with Eq. (21) or Eq. (34). The parameters used in the analysis are L = 1, a = 0.05, e = 2 and K = 1. Upper plot: time decay of the peak charge density, qmax e , located at the center of the square domain. The continuous line is the analytical solution of Eq. (37), �symbols show simulation results with Eq. (21) and h symbols show simulation results with (34). Lower plot: spatial distribution of charge density along the x-axis. Times equal to 0, 2, 4 and 6 are shown. The continuous line is the analytical solution and �symbols indicate the simulation results.**

0 2 4 6 8 10 12 14 16 18 0

0.5

1

t


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq031.png)

0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 −5

0

5


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq032.png)

x

E


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq033.png)

Eq. (35) Level 5 Level 6 Level 7 Adapted


> **Fig. 5. Results for the simulations describing charge relaxation of an isolated conducting cylinder. The free parameters are chosen as: R = 0.05, L = 1, K1 = 3, e1 = 3, e = 2 and qeo = 0.5. In the upper plot the time evolution of the total amount of charge in the domain is represented, scaled with the initial amount of charge (pR2 qeo). �Symbols indicate simulation results where the charge density equation is modeled with Eq. (21) and ⁄ symbols show simulation results with Eq. (34). In the lower plot the spatial distribution of the electric field is shown, once the steady state is reached for different levels of mesh refinement. The continuous line shows the analytical approximation given by Eq. (38), �symbols are obtained with a uniform mesh of level L ¼ 5 (mesh grid given by 2L �2L), e symbols are obtained with a uniform mesh of level 6, O with a uniform mesh of level L ¼ 7, and �with an adapted non-uniform mesh ðLmax ¼ 10; Lmin ¼ 6Þ.**

1948 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

D ¼ b �a a þ b ; ð39Þ

where b and a are the sizes of the spheroid in the direction parallel and perpendicular to the electric field respectively (see Fig. 6). Prolate spheroids correspond to D > 0 and oblate ones to D < 0. Using a linearised asymptotic analysis and assuming that both fluids are extremely viscous and conducting, Taylor provided an expression for D as a function of the fluid properties and the electric field intensity,

D ¼ 9 16

CaE ð2 þ RÞ2 1 þ R2 �2Q þ 3 5 ðR �QÞ 2 þ 3k 1 þ k


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq034.png)

where R = K1/K2, Q = e1/e2 and k = l1/l2 stand for the ratio of the inner to the outer conductivities, permittivities and viscosities, respectively. CaE is the electric capillary number given by CaE ¼ E2 1Rde2=c. Note that as a consequence of adopting the S.I system of units, the factor in Eq. (40) is 9/16 [38,39] rather than 9/8p appearing in Eq. (25) of Taylor [37]. The expression (33) of Hua et al. [36] set the factor to 9/8p when apparently they also use the S.I system of units. Most of the numerical simulations we present in this section have been performed using an axisymmetric version of the numerical scheme. For validation purposes, some simulations have been repeated with a fully three-dimensional scheme. The testing has been performed in two steps. First, we have confirmed that both our schemes, three-dimensional and axisymmetric, reproduce the electrostatic analytical solution derived by Taylor [37] for a spherical drop. Once the electrostatic part of the code has been checked, we have simulated the complete, coupled, EHD problem. The electric field solution in polar coordinates shown in Fig. 6 reads for the outer fluid,

E2r ¼ �1 þ 2ðR �1Þ 2 þ R

1 r3

� � cos h and E2h ¼ 1 �R �1 2 þ R

1 r3


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq035.png)

and for the inner fluid,

E1r ¼ � 3 2 þ R cos h and E1h ¼ 3 2 þ R sin h: ð42Þ

In the above expressions the electric field has been made dimensionless with E1 and the radius r with Rd. In Fig. 7 we plot the radial electric field distribution along the negative branch of the x-axis (h = p) for the axisymmetric and 3D schemes with R = 2.5 and Rd = 0.1. In the numerical computations we have applied the gradient adaptation criterion to the volume fraction. The maximum and minimum level used for adaptation are Lmax ¼ 9 and Lmin ¼ 5. Fig. 7 confirms that the electric field computation using both schemes agrees very well with the analytical solution. To explore extensively the influence of the different parameters on the droplet deformation is beyond the scope of this section. Thus we focus on the cases reported by Tomar et al. [27]. Following Tomar et al. [27], we set the ratio of permittivities and viscosities to Q = 10 and k = 1, respectively. The remaining parameters have been set to Rd = 0.1, e2 = 1, E1 = 1.34 and c = 1(CaE ’ 0.18); the viscosity of the outer medium is l2 = 0.1 in order to be close to a Stokes flow as assumed by Taylor [37]. With this value of the viscosity the Reynolds number, Re = qvcRd/l2 ’ 10�1 with vc ¼ Rde2E2 1=l2, is small. We have first simulated the case in which the droplet should not deform (R = 5.1) while recirculation is induced by the electrical tangential stresses. In Fig. 8 we display the computed radial and azimuthal components of the velocity (see Fig. 6) along a h = p/4 transect together with the analytical solution [27,40] which can be written, for the inner fluid


> **Fig. 6. Sketch of the geometry, electrical conditions and computational domain for the study of the electrohydrodynamic deformation of droplets of Section 4.3.**

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1949


## v1r ¼ Arð1 �r2Þð3 cos2 h �1Þ and v1h ¼ 3A 2 r 1 �5 3 r2 � � sin 2h ð43Þ

and for the outer fluid,


## v2r ¼ A r�4 �r�2 � � ð3 cos2 h �1Þ and v1h ¼ �Ar�4 sin 2h; ð44Þ

where r has been made dimensionless with Rd and A stands for

A ¼ �9 10


## Rde2E2 1 l2


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq036.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq037.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq038.png)

The velocities in Fig. 8 have been normalized with the characteristic velocity vc. Excellent agreement between the computed and analytical values is obtained for both components. Note that to obtain such an agreement, it is necessary to minimise the influence of domain boundaries (confinement) which can be very significant for Stokes flows due to the elliptic nature of the equations in that limit. To do so, the domain extent has been set to L = 2 (compared to Rd = 0.1) and free-slip boundary conditions (@v/@n = 0) were imposed. The computational cost of using such a large domain is greatly minimised by using an adaptive spatial resolution with a minimum level of refinement Lmin ¼ 4 ð Þ and a maximum level of Lmax ¼ 10 (so that hmin ¼ L 2�Lmax and Rd/hmin = 51.2).

0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 0.6

0.8

1

1.2

1.4

1.6

r

Er

Analytical Sol. Sim. Full 3D Sim. Axisym. (WAM) Sim. Axisym. (WHM)


> **Fig. 7. Comparison between the analytical dimensionless electrostatic solution given by Eqs. (41) and (42) and our numerical simulations for a conducting droplet suspended in a conducting liquid (along a h = p transect). The conductivity ratio is R = 2.5. �Symbols indicate simulation results with the full threedimensional scheme. h and } symbols correspond to axisymmetric numerical simulations using the WAM and WHM interpolations schemes, respectively. All simulations use an adaptive mesh with Lmax ¼ 9 and Lmin ¼ 5. The continuous line corresponds to the analytical solution.**

0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 −0.05


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq039.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq040.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq041.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq042.png)

0

0.01

0.02

0.03

r/Rd

v/vc

vr Simul.


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq043.png)

vr Theory


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq044.png)


> **Fig. 8. Axisymmetric velocity profiles along a h = p/4 transect obtained with the present scheme compared to the theoretical solution of Taylor [37] (Eqs. (43) and (44)).**

1950 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

In Fig. 9 we plot the evolution of the deformation D with the ratio of conductivities R obtained using our scheme, the scheme of Tomar et al. [27] and Taylor’s theoretical solution (Eq. (40)). To further check that confinement effects are minimal, we display our results for two domain sizes: L = 1 and L = 2 and two viscosities: l = 0.1 and l = 1.0. The independence of the deformations from the domain size for identical viscosities shows that the droplet dynamic is unaffected by the boundaries provided they are sufficiently far away. For small values of the deformation (�0.05 < D < 0.05) both sets of sim-

1 3 5 7 9 11 13 15 −0.2


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq045.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq046.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq047.png)

0

0.05

0.1

R

Deformation D


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq048.png)


> **Fig. 9. Deformation D as a function of the conductivity ratio R (Q = 10). �and �symbols correspond to our simulations with viscosity l = 0.1 and a domain of size L = 1 and L = 2 respectively. + symbols correspond to simulations with viscosity l = 1.0 (L = 2) and h symbols to the simulations of Tomar et al. [27]. The continuous line is Taylor’s analytical solution, Eq. (40).**


## (a) (b)


> **Fig. 10. Velocity field for a conducting drop in a bath of a conducting liquid subjected to an electric field. No deformation case corresponding to R = 5.1 and Q = 10 using (a) the axisymmetric scheme and (b) the full three-dimensional scheme. The simulations was performed with a final adapted mesh Lmax ¼ 10ðRd=hmin ¼ 51:2Þ.**


> **Fig. 11. Isocontours of pressure (red lines) and recirculating velocity field in a conducting drop immersed in a bath of a conducting liquid (R = 1.81 and Q = 10). The isoncontour range is p = 16–22 with Dp = 1.0 intervals. The simulation was performed with an adapted mesh ðLmax ¼ 10; Rd=hmin ¼ 51:2Þ. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)**

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1951

ulations agree well with the analytical solution. For larger deformations, the solutions diverge, most likely because the analytical solution relies on a linearised asymptotic analysis valid only for small deformations. The departure from Taylor’s theory is smaller for a larger value of the viscosity because such a flow is closer to the pure Stokes flow assumed by Taylor. For a viscosity l = 1.0 our results agree well with those of Tomar et al. [27].


> **Fig. 10 illustrates the velocity field distribution using a final adapted mesh Lmax ¼ 10ðRd=hmin ¼ 51:2Þ and the axisymmetric (a) or the full three-dimensional scheme (b). As expected the deformation D is negligible. The deformation calculated using the axisymmetric scheme with Lmax ¼ 9 (Rd/hmin = 25.6) is D = �5.5 �10�4. The deformation decreases to D = 10�4 if the mesh is finer ðLmax ¼ 10Rd=hminÞ ¼ 51:2Þ; with the full 3D scheme and a similar mesh, the deformation is slightly higher, D = 4.1 �10�4. In accordance with Taylor [37] recirculations are induced inside and outside of the droplet in the direction determined by the tangential electric stress.**


> **Fig. 12. Isocontours of charge density (red lines: positive values of charge density; blue lines: negative values) around the spheroid tip for two different viscosities (R = 1.81,Q = 10). The isocontour range is 450 P qe P �450 at intervals Dqe = 100. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)**


> **Table 3 Deformation D in the cases of R = 5.1 and R = 1.81 for the different viscosities and grid sizes used.**

Rd/hmin R = 5.1 R = 1.81


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq049.png)

0.8 1 1.2

0

0.2

0.4

0.6

0.8

1

1.2

r


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq050.png)

0 1 2 3


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq051.png)


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq052.png)

0

0.5

1

1.5

r

Elec. field components


![Equation](images/2011_lopez_herrera_charge_conservative_ehd_vof_eq053.png)

WAM

WHM

c


> **Fig. 13. Left plot: charge density distribution across the interface along a h = p/4 transect computed using both the WAM and WHM schemes The volume fraction c is also shown. The charge density has been normalized with the maximum value obtained using the WAM scheme. The radius has been made dimensionless with the droplet radius. Right plot: radial and azimuthal components of the non-dimensional electric field, Er and Eh, as functions of the dimensionless radius r along the h = p/4 transect. The analytical solution given by Eqs. (41) and (42) is also plotted.**

1952 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

Finally Fig. 11 illustrates the pressure distribution and the recirculating velocity field induced inside and outside the droplet when the ratio of conductivities is lowered to R = 1.81 (the viscosity is set to 0.1 and the other parameters remain unchanged). In this case the droplet deformation we calculate is D = �0.113 (for l = 0.1) while the computational results of Tomar et al. [27] as well as our result for l = 1.0 give a deformation slightly more oblate (D ’ �0.147). Taylor’s prediction, Eq. (40), is D = �0.195. In Table 3 we summarize for a better comparison the deformation D in the cases of R = 5.1 and R = 1.81 for the different viscosities and grid sizes. The slight difference in the deformation D we found for viscosities of l = 0.1 and l = 1 could be due to the presence of some convection of charge on the tip of the spheroid. In Fig. 12 we show the distribution of charge in the vicinity of the tip for both viscosities. In the left figure (l = 0.1) it can be observed that, close to the tip, the charge spreads slightly into the droplet as it is pulled by the flow in that region. If the viscosity is increased to one, the velocity decreases by one order of magnitude and, consequently, the convection of charge cannot compete against the relaxation by conduction anymore; i.e. all the charge is accumulated at the interface (see Fig. 12 right). Note that both Taylor [37] and Tomar et al. [27] ignore charge convection in their models. We have also carried out some simulations assuming that the suspended droplet behaves as an isolating medium (a bubble). Accordingly, we have set the conductivity of the inner fluid to K1 = 0 (R = 0). Note that the permittivity of a gas is close to the permittivity of the vacuum eo, therefore, the ratio of permittivity Q has to be smaller than one. We have assumed that the outer fluid is an apolar one (for example heptane) with a permittivity e2 = 2e1 (Q = 0.5) (heptane has a permittivity e = 1.92eo). Since the inner fluid is assumed to be a gas, its viscosity and density have been set to values a thousand time smaller than the outer fluid, q1 = 10�3q2 and l1 = 10�3l2. The remaining parameters have been kept similar to the values used above: l2 = 0.1, E1 = 1.0, Rd = 0.1 and q2 = 1.0. In the simulation we have neglected both compressibility and buoyancy forces and it has been performed using both WAM and WHM interpolation and mesh adaptation ðLmax ¼ 10; Rd=hminÞ ¼ 51:2Þ. As can be observed in the left plot of Fig. 13 very similar charge distributions across the interface are obtained using the WAM and WHM interpolations, although with the WHM interpolation the charge distribution is slightly more diffuse than using WAM. In the same plot it can be observed that the charge tends to accumulate on the ‘‘inside’’ of the diffuse interface, i.e. the peak of charge is localized for a radius slightly smaller than the radius of the droplet. This trend causes the computed electric field inside the bubble to be slightly different from the analytical result given by Eqs. (41) and (42) (see also right plot in Fig. 13). Naturally, this error could be reduced further using a finer grid since the distance between the peak of charge and the interface position would decrease. In Fig. 14 we show the effect of the electric forces on the fluids involved. The electric tangential stress acting on the interface set a strong recirculation both inside and outside the bubble (see Fig. 14(a)) although the electrical stresses at the interface do not cause any appreciable deformation of the bubble. In Fig. 14(b) we plot the computed velocity pattern. It can be observed that the velocity profiles are very similar to the one created in the conducting droplet immersed in a conducting medium (see Fig. 8) although one order of magnitude weaker.

5. Conclusion

A volume-of-fluid (VOF) method has been presented, adapted to the solution of the governing equations for two-phase EHD problems. Special attention has been paid to the calculation of the electric forces and to the solution of the charge den-


## (a) (b)


> **Fig. 14. (a) Velocity pattern inside and around a gas bubble. (b) Computed radial and azimuthal components of the non-dimensional velocity, vr and vh, as functions of the radius r along a h = p/4 transect (red line). (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)**

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1953

sity equation. The proposed method does not require any restriction concerning the electrical behaviour of the fluids involved and is especially well-suited to deal with interfacial flows due to its conservative nature. This makes the method applicable to the study of problems in which the bulk charge conduction and convection can play a relevant role, such as the characterization of the cone-to-jet transition region (also referred to as the neck region) appearing in EHD cone–jet electrosprays [41]. Note also that the proposed method allows the analysis of the transient stages occurring during the relaxation of charge from the bulk to the interface. The capabilities of the proposed model to provide accurate solutions for the interfacial pressure jump caused by electrical stresses has been tested with planar geometries for different limits of the electric fluid behavior. The numerical scheme proposed for the charge conservation equation accurately predicts the time evolution of the charge distribution. In addition the overall charge conservation has been checked. Finally, the scheme has been tested against a fully coupled EHD problem; the deformations of conducting droplets immersed in a conducting bath, with excellent results. Although Gerris is designed to solve hydrodynamic problems in complex geometries [4], the Gerris-EHD extension currently assumes that all cells are fully occupied by fluid. In a near future we intend to generalise the solver to be able to deal with mixed cells (i.e. cells partially occupied by a solid). This will allow the numerical study of electrohydrodynamic problems in complex geometries such as the simulation of the electro-flow-focusing method of spraying [42] or the characterization of microfluidic devices.

Acknowledgments

This work is partially supported by the Spanish Ministry of Science and Technology, Grant No. DPI2007-63559, and by the Junta de Andalucía, Excellence Project No. TEP-1190. JLH would like to express his gratitude to Drs. Pérez-Lombard and Riesco-Chueca for their useful comments and help in the preparation of this manuscript.


## References

[1] G.I. Taylor, Disintegration of water drops in a electric field, Proc. Roy. Soc. Lond. A (280) (1964) 383–397. [2] J.R. Melcher, G.I. Taylor, Electrohydrodynamics: a review of the role of interfacial shear stresses, Annu. Rev. Fluid Mech. 1 (1969) 111–146. [3] D.A. Saville, Electrohydrodynamics: the Taylor–Melcher leaky dielectric model, Annu. Rev. Fluid Mech. 29 (1997) 27–64. [4] S. Popinet, Gerris: a tree-based adaptive solver for the incompressible Euler equations in complex geometries, J. Comput. Phys. 190 (2) (2003) 572–600. [5] S. Popinet, An accurate adaptive solver for surface-tension-driven interfacial flows, J. Comput. Phys. 228 (2009) 5838–5866. [6] J.B. Fenn, M. Mann, C.K. Meng, S.F. Wong, C.M. Whitehouse, Electrospray ionization for mass spectrometry of large biomolecules, Science 246 (4926) (1989) 64–71. [7] H.T. Yudistira, V.D. Nguyen, P. Dutta, D. Byun, Flight behavior of charged droplets in electrohydrodynamic inkjet printing, Appl. Phys. Lett. 96 (2) (2010). [8] J.S. Shrimpton, A.J. Yule, Characterization of charged hydrocarbon sprays for application in combustion systems, Exp. Fluids 26 (5) (1999) 460–469. [9] P. Kazemi, P. Selvaganapathy, C. Ching, Electrohydrodynamic micropumps with asymmetric electrode geometries for microscale electronics cooling, IEEE Trans. Dielect. Electr. Insul. 16 (2) (2009) 483–488. [10] D.J. Laser, J.G. Santiago, A review of micropumps, J. Micromech. Microeng. 14 (6) (2004) R35–R64. [11] J. Zeng, T. Korsmeyer, Principles of droplet electrohydrodynamics for lab-on-a-chip, Lab on a Chip – Miniaturisation Chem. Biol. 4 (4) (2004) 265–277. [12] M. Felten, W. Staroske, M.S. Jaeger, P. Schwille, C. Duschl, Accumulation and filtering of nanoparticles in microchannels using electrohydrodynamically induced vortical flows, J. Electrophoresis 29 (14) (2008) 2987–2996. [13] S.K. Cho, H. Moon, C.-J. Kim, Creating, transporting, cutting, and merging liquid droplets by electrowetting-based actuation for digital microfluidic circuits, J. Microelectromech. Syst. 12 (1) (2003) 70–80. [14] O.D. Velev, B.G. Prevo, K.H. Bhatt, On-chip manipulation of free droplets, Nature 426 (6966) (2003) 515–516. [15] J.M. López-Herrera, A. Barrero, A. Lopez, I. Loscertales, M. Marquez, Coaxial jets generated from electrified Taylor cones. Scaling laws, J. Aerosol Sci. 34 (5) (2003) 535–552. [16] A.M. Gañán-Calvo, J. Dávila, A. Barrero, Current and droplets size in the electrospraying of liquid. Scaling laws, J. Aerosol Sci. 28 (2) (1997) 249–275. [17] J.M. López-Herrera, A.M. Gañán-Calvo, M. Perez-Saborid, One-dimensional simulation of the breakup of capillary jets of conducting liquids. Application to E.H.D. spraying, J. Aerosol Sci. 30 (7) (1999) 895–912. [18] A.M. Gañán-Calvo, Cone–jet analytical extension of Taylor’s electrostatic solution and the asymptotic universal scaling laws in electrospraying, Phys. Rev. Lett. 79 (2) (1997) 217–220. [19] J.D. Sherwood, Breakup of fluid droplets in electric and magnetic fields, J. Fluid Mech. 188 (1988) 133–146. [20] J.C. Baygents, N.J. Rivette, H.A. Stone, Electrohydrodynamic deformation and interaction of drop pairs, J. Fluid Mech. 368 (1998) 359–375. [21] E.R. Setiawan, S.D. Heister, Nonlinear modeling of an infinite electrified jet, J. Electrostat. 42 (3) (1997) 243–257. [22] F.J. Higuera, Emission of drops from the tip of an electrified jet of an inviscid liquid of infinite electrical conductivity, Phys. Fluids 19 (7) (2007) 072113. [23] R.T. Collins, M.T. Harris, O.A. Basaran, Breakup of electrified jets, J. Fluid Mech. 588 (2007) 75–129. [24] P.K. Notz, O.A. Basaran, Dynamics of drop formation in a electric field, J. Colloid Interf. Sci. 213 (1999) 218–237. [25] A. Fernández, G. Tryggvason, J. Che, S.L. Ceccio, The effects of electrostatic forces on the distribution of drops in a channel flow: two-dimensional oblate drops, Phys. Fluids 17 (9) (2005) 1–15. [26] K. Teigen, S. Munkejord, Sharp-interface simulations of drop deformation in electric fields, IEEE Trans. Dielect. Electr. Insul. 16 (2) (2009) 475–482. [27] G. Tomar, D. Gerlach, G. Biswas, N. Alleborn, A. Sharma, F. Durst, S.W.J. Welch, A. Delgado, Two-phase electrohydrodynamic simulations using a volume-of-fluid approach, J. Comput. Phys. 227 (2) (2007) 1267–1285. [28] J.U. Brackbill, A continuum method for modeling surface tension, J. Comput. Phys. 100 (2) (1992) 335–354. [29] A.K. Sen, J. Darabi, D.R. Knapp, J. Liu, Modeling and characterization of a carbon fiber emitter for electrospray ionization, J. Micromech. Microeng. 16 (3) (2006) 620–630. [30] O. Lastow, W. Balachandran, Numerical simulation of electrohydrodynamic (EHD) atomization, J. Electrostat. 64 (12) (2006) 850–859. [31] S. Popinet, The Gerris Flow Solver, URL <http://gfs.sourceforge.net>. [32] J.B. Bell, P. Colella, H.M. Glaz, A second-order projection method for the incompressible Navier–Stokes equations, J. Comput. Phys. 85 (1989) 257–283. [33] Y.Q. Zu, Y.Y. Yan, A numerical investigation of electrohydrodynamic (EHD) effects on bubble deformation under pseudo-nucleate boiling conditions, Int. J. Heat Fluid Flow 30 (4) (2009) 761–767. [34] C.W. Hirt, Electrohydrodynamic of semi-conducting fluids: with application to electrospraying, Technical report FSI-04-TN70, Flow-3D, 2004. [35] S.W.J. Welch, G. Biswas, Direct simulation of film boiling including electrohydynamic forces, Phys. Fluids 19 (1) (2007). 012106-1–012106-11.

1954 J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955

[36] J. Hua, L.K. Lim, C. Wang, Numerical simulation of deformation/motion of a drop suspended in viscous liquids under influence of steady electric fields, Phys. Fluids 20 (11) (2008) 120–234. [37] G.I. Taylor, Studies in electrohydrodynamics. I. The circulation produced in a drop by an electric field, Proc. Roy. Soc. Lond. Ser. A 291 (1966) 159–166. [38] S. Torza, R. Cox, S.G. Mason, Electrohydrodynamic deformation and burst of liquid drops, Phil. Trans. Roy. Soc. Lond. Ser. A. Math. Phys. Sci. 269 (1198) (1971) 295–310. [39] J.Q. Feng, T.C. Scott, A computational analysis of electrohydrodynamics of a leaky dielectric drop in an electric field, J. Fluid Mech. 311 (1996) 289–326. [40] J.M. López-Herrera, S. Popinet, Equilibrium of a droplet suspended in an electric field, URL <http://gfs.sourceforge.net/tests/tests/electro.html>. [41] J.F. de la Mora, The fluid dynamics of Taylor cones, Annu. Rev. Fluid Mech. 39 (2007) 217–243. [42] A.M. Gañán-Calvo, J.M. Lopez-Herrera, P. Riesco-Chueca, The combination of electrospray and capillary flow focusing, J. Fluid Mech. 566 (2006) 421– 455.

J.M. López-Herrera et al. / Journal of Computational Physics 230 (2011) 1939–1955 1955

