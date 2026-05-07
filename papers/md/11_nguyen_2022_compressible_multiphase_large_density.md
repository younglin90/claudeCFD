Contents lists available at ScienceDirect


# Ultrasonics Sonochemistry

journal homepage: www.elsevier.com/locate/ultson


# Assessing the accuracy of the coupled-spherical-bubble approach for bubble pairs in an acoustic field


## Dániel Nagy ∗, Ferenc Hegedűs

Department of Hydrodynamic Systems, Faculty of Mechanical Engineering, Budapest University of Technology and Economics, Műegyetem rkp. 3., H-1111 Budapest, Hungary

A R T I C L E  I N F O

Keywords: Bubble pair Bubble dynamics Ultrasound Acoustic emission Bubble jet Multiphase flow

A B S T R A C T

This study evaluates the accuracy of coupled-spherical-bubble models in acoustic fields by comparing them to direct numerical simulations (DNS). The coupled-spherical-bubble approach refers to the method of modeling multi-bubble systems, where the spherical bubble dynamics are governed by a simplified equation and these equations are coupled through the pressure emissions of the bubbles. Tested spherical models are the Keller– Miksis and Gilmore equation, and pressure emission models include the incompressible, quasi-acoustic and Kirkwood–Bethe hypothesis. Emphasis is placed on peak bubble pressure during collapse and the accuracy of pressure emission models. First, a single bubble in a spherical standing wave is analyzed. Among the simplified approaches, the Gilmore model provides closer agreement with DNS at Mach numbers approaching unity in water. In high-viscosity glycerol spherical models break down independently of the Mach number. Pressure wave emissions are accurately tracked by all tested models that assume a finite propagation velocity; however, shock wave emissions at high compression ratios can only be tracked by the Kirkwood–Bethe model. In the second part, a bubble pair is subjected to an ultrasonic pulse, and spherical volume oscillations and pressure emissions of bubbles are compared using various coupled-spherical-bubble approaches. DNS results show that jetting during collapse reduces gas compression, leading spherical models to overpredict internal pressure. While spherical models are effective for isolated bubbles in ideal conditions, DNS is essential for accurately capturing inter-bubble interactions. Nevertheless, spherical models provide good accuracy in the case of a bubble collapse without jetting, even when perfect sphericity is not preserved.

1. Introduction

Modeling multi-bubble systems excited by ultrasound is crucial for many applications, ranging from sonochemical reactors [1–9] to surface cleaning and water treatment [10–17] and even medical procedures [18–27]. Currently, multi-bubble systems can be modeled in several ways. A numerically efficient approach involves assuming spherical bubbles and calculating the pressure emission of the bubbles to account for the inter-bubble interactions [28–36]. We refer to this type of modeling as the coupled-spherical-bubble approach. One of the simplest ways is to assume an incompressible liquid and solve the radial pulsation of the bubble using a Rayleigh–Plesset type equation [37–42]. The Keller–Miksis equation [43–45] extends the Rayleigh– Plesset equation by assuming first-order compressibility; that is, it assumes a finite speed of sound at the bubble wall. Alternatively, the Gilmore equation [46–48] can be employed, which accounts for liquid compressibility via an appropriate equation of state. Inter-bubble interactions can be modeled by considering the acoustic pressure emission

∗Corresponding author.

E-mail addresses: dnagy@hds.bme.hu (D. Nagy), fhegedus@hds.bme.hu (F. Hegedűs).

of bubbles using the incompressible liquid assumption [49], the quasiacoustic assumption [50,51], the Kirkwood–Bethe hypothesis [52–54], or through full hydrodynamic simulation of the liquid between the bubbles [55]. In bubble cloud simulations using the coupled-sphericalbubble approach, accurate modeling of the spherical bubble dynamics as well as the acoustic pressure emissions of bubbles is essential, though this approach inherently involves several simplifications.

The first simplification of spherical bubble models relates to the modeling of compressibility. The accuracy of the Rayleigh–Plesset equation is limited due to the incompressible liquid assumption and the physically unrealistic assumption of infinite sound speed. Vokurka [56] conducted a comparison of spherical models and established that the error of the Rayleigh–Plesset equation exceeds 5% for 𝑅max∕𝑅𝐸 > 2, where 𝑅max and 𝑅𝐸 are the maximum and equilibrium bubble radii, respectively. The Keller–Miksis equation assumes a finite speed of sound with constant liquid density. There is also experimental validation of this equation for acoustically driven bubbles [57,58].

https://doi.org/10.1016/j.ultsonch.2025.107651 Received 14 August 2025; Received in revised form 15 October 2025; Accepted 24 October 2025

Ultrasonics Sonochemistry 123 (2025) 107651

Available online 28 October 2025 1350-4177/© 2025 The Author(s). Published by Elsevier B.V. This is an open access article under the CC BY-NC-ND license ( http://creativecommons.org/licenses/bync-nd/4.0/ ).

D. Nagy and F. Hegedűs

However, the assumption of constant density limits the applicability of the equation as the results may become inaccurate for large bubble compressions, where the pressure in the liquid surrounding the bubble (and the speed of sound) is significantly increased. The Gilmore model handles compressibility more accurately, as it uses an equation of state to calculate the speed of sound and enthalpy at the bubble wall. Historically, the Tait equation of state was used [46,59,60]; however, it breaks down at high temperatures. To overcome this issue, Denner [47] implemented the Noble–Abel stiffened gas equation of state into the Gilmore model. The Gilmore equation has been experimentally validated for both acoustically driven bubbles [61] and laser-induced bubbles [62–64]. We can categorize the models based on their accuracy with respect to the Mach number of the bubble wall [46]: the Rayleigh– Plesset equation is accurate to zeroth order, the Keller–Miksis equation to first order, and the Gilmore model to second order.

The second issue is viscosity. In the derivation of the Keller–Miksis equation [43], viscosity is omitted when deriving the velocity potential of the fluid motion. However, shear viscosity is considered at the bubble wall. Viscosity is treated in the same way in the Rayleigh– Plesset and Gilmore models. Viscosity is also neglected in all the above-mentioned models for acoustic pressure emission, even though it dissipates energy at the shock front [65,66]. In compressible flow with higher viscosity, the effect of the second viscous coefficient (bulk viscosity) can also play a role; and can be included at the bubble surface [67], but is neglected in most applications.

The third simplification is the assumption of a homogeneous bubble with uniform pressure and density inside. Although in most applications this is a valid assumption; for laser-induced bubbles it is known to break down [63,68]. However, extremely strong bubble collapses in an acoustic field could also lead to inhomogeneities in pressure and density.

It is well established that the Keller–Miksis and Gilmore equations are accurate for spherical bubbles under conditions of low Mach numbers and low viscosity [57,61,69]. When a bubble is close to a wall, another bubble, or is subjected to non-spherical waves, it loses its spherical shape and violates the fundamental assumption of the aforementioned models: sphericity [70–77]. Nonetheless, simulating multibubble systems consisting of spherical bubbles while modeling the interaction between them through acoustic pressure emission remains compelling due to its simplicity (so-called coupled-spherical-bubble approach). This may be a severe simplification, as at higher compression ratios, bubbles may lose their spherical shape or even jet [78]. Despite this, such coupled-spherical-bubble simulations can still predict the void fraction as a wave passes through a bubble cloud [79], cavitation performance and streaming in sonoreactors [80,81], and resonance patterns of bubbly screens [51,82].

A computationally more intensive approach of studying multibubble systems and pressure emissions involves multiphase hydrodynamic simulations [83–87]. However, such studies are challenging due to the extremely high computational cost and the need to reliably model the gas-liquid interface. In cases where bubbles are organized along a row, the dimensionality of the problem can be reduced to two by exploiting axial symmetry [78]. In this study, we employ the ALPACA multiphase flow solver [88,89] to carry out direct numerical simulations (DNS) of bubbles. The main advantage of using the ALPACA multiphase flow solver lies in its combined treatment of many physical processes, including acoustics, shock-wave dynamics, and gas–liquid interface dynamics. The solver is inherently compressible, allowing for the simulation of acoustic phenomena. High-order numerical schemes, such as the WENO5 method used in the solver, enable accurate reconstruction of shock waves [90]. The level-set formulation provides highly accurate bubble shapes after collapse [91]. Although DNS simulations are computationally expensive, ALPACA reduces the cost through an efficient multiresolution algorithm and effective parallelization. The solver has been validated for a wide range of bubble-dynamics-related phenomena. For example, Hoppe et al. [88] accurately reproduced the

interaction of a Mach 1.22 shock wave in air with a helium bubble, as well as the interaction of three bubbles with a shock wave in water. Kaiser et al. [92] studied the fragmentation of a cylindrical liquid drop following shock passage and found both qualitative and quantitative agreement with experimental data. Finally, in our earlier work, we successfully reproduced measurements of non-spherical bubble oscillations in ultrasonic fields [93]. The only limitation of using ALPACA with the level-set method is its dissipative nature; this can, however, be mitigated by monitoring the total bubble mass and ensuring sufficiently high resolution.

The first objective of our study is to investigate bubble dynamics and acoustic pressure emissions in spherical standing waves in liquids of various viscosities to establish the applicability limits of simplified spherical models in terms of the Mach number of the bubble wall and viscosity. The investigated spherical models are the Keller–Miksis and Gilmore equations, which are compared against the solution of the governing equations of fluid flow under spherical symmetry. In this case, the comparison between the spherical models and the fluid flow simulations reveals the effect of simplifications of the spherical models regarding the simplified treatment of compressibility, viscosity and speed of sound.

The second objective is to assess the accuracy of various acoustic pressure emission models for a single spherical bubble. Acoustic pressure emissions are considered using the incompressible, quasi-acoustic, and Kirkwood–Bethe models. The results are compared against the solution of the governing equations of compressible fluid flow under spherical symmetry. An accurate description of the emitted pressure is essential when inter-bubble interactions are considered.

The final objective is to assess the limitations of the coupledspherical-bubble approach for a bubble pair. The dynamics and acoustic emissions of bubble pairs excited by a single-cycle ultrasonic pulse are investigated using both the coupled-spherical-bubble models and the ALPACA multiphase flow solver with axial symmetry. In the case of bubble pairs excited with high-amplitude ultrasound, sphericity is no longer maintained [94,95], and bubble jets occur; the direction depends on the distance between the bubbles. The ultimate aim is to evaluate the applicability of various bubble and acoustic emission models for such scenarios. This is done by comparing the pressure field around bubble pairs throughout their expansion and subsequent collapse using the different models. The advantage of using a single-cycle ultrasonic pulse lies in the control of triggering non-spherical bubble collapses in a single expansion-collapse cycle of a bubble.

The article is structured as follows: Section 2 provides a detailed introduction to the simplified bubble and pressure emission models listed in Table 1. Section 3 validates the models with experiments of moderate bubble collapse, then compares the models for the expansion and subsequent collapse of a single bubble under spherical symmetry. Section 4 investigates bubble pairs oriented along an axis, exploiting axisymmetry, and examines the validity of the simplified models. Finally, Section 5 discusses the main findings and Section 6 draws the conclusions.

2. Methods

We employ the ALPACA multiphase flow solver [88,89] to simulate bubbles under spherical symmetry and bubble pairs under axial symmetry, then use these results as a reference. First, we provide some notes on the investigated coupled-spherical-bubble approaches for a bubble pair, followed by a detailed description of the models.

The bubble pair is oriented on a vertical symmetry axis in this study. The coupled-spherical-bubble models under investigation are summarized in Table 1. In the simplest models (1a–1c), we solve the Keller–Miksis equation for radial dynamics and couple the bubbles through incompressible pressure emission. In Model 1a, the bubbles are simultaneously struck by the ultrasonic pulse, with no time delay between them. Then, in Model 1b, it is assumed that the top bubble

Ultrasonics Sonochemistry 123 (2025) 107651

2

D. Nagy and F. Hegedűs


> **Table 1 Simplified coupled-spherical-bubble models under investigation. Model # Name Radial dynamics Pressure emission Note Solver 1a ICa Keller–Miksis Incompressible – Wolfram Mathematica 1b ICb Keller–Miksis Incompressible 𝜏geo considered Wolfram Mathematica 1c ICc Keller–Miksis Incompressible 𝜏geo, 𝜏𝑒 considered Wolfram Mathematica 2 QA Keller–Miksis Quasi acoustic 𝜏geo, 𝜏𝑒 considered APECSS 3 KB Gilmore Kirkwood–Bethe 𝜏geo, 𝜏𝑒 considered NASG EoS**

APECSS

is reached by the pulse with a time delay of 𝜏geo = 𝐷∕𝑐𝐿, where 𝐷 is the distance between the bubble centers and 𝑐𝐿 is the speed of sound in the liquid. Model 1c also assumes a finite propagation velocity of the emitted pressure between the two bubbles, with a delay of 𝜏𝑒= 𝐷∕𝑐𝐿. Although the two types of delays are numerically the same for a row of bubbles, this equivalence does not hold for arbitrary bubble arrangements. This definition emphasizes the two different modeling aspects: the geometric time delay 𝜏geo considers the finite speed of the ultrasonic pulse, and that it reaches the bubbles at different times due to the geometry of the bubble layout, and the emission time delay 𝜏𝑒 considers the delay of the pressure emitted by one bubble reaching the other bubble. Model 2 employs quasi-acoustic pressure emission with Lagrangian wave tracking. In this case, both 𝜏geo and 𝜏𝑒 are considered. Finally, Model 3 uses the Gilmore equation to solve the radial dynamics, and the Kirkwood–Bethe hypothesis is applied to model the pressure emission. In this model, both the Gilmore equation and the Kirkwood–Bethe hypothesis utilize the same Noble–Abel Stiffened Gas equation of state (NASG EoS) [96,97] for the liquid.

2.1. Multiphase flow simulations with ALPACA

We employ the open-source ALPACA multiphase flow solver [88, 89] to carry out DNS simulations of a single bubble in 1D spherical symmetry and bubble pairs in axial symmetry. Using ALPACA, we solve the governing equations of compressible flow for each phase separately [88]:

𝜕𝜌


$$
𝜕𝑡= −∇⋅𝜌𝒖,
(1)
$$

𝜕𝜌𝒖


$$
𝜕𝑡= −∇⋅(𝜌𝒖⊗𝒖−𝛱),
(2)
$$

𝜕𝜌𝐸


$$
= −∇⋅(𝜌𝐸𝒖−𝛱𝒖−𝒒),
(3)
$$

where 𝜌 is the density, 𝒖 is the velocity, 𝐸 is the specific total energy and 𝛱 is the stress tensor. The effect of gravity is neglected. The stress tensor is


$$
𝛱= −𝑝𝑰+ 𝜇1
∇⊗𝒖+ (∇⊗𝒖)tr −2
$$


$$
3𝑰∇⋅𝒖
+ 𝜇2𝑰∇⋅𝒖,
(4)
$$

where 𝑝 is the pressure, 𝜇1 = 𝜇𝑠 is the shear viscosity, 𝜇2 = 𝜇𝑏−2∕3𝜇𝑠 where 𝜇𝑏 is the bulk viscosity and 𝑰 is the identity matrix. The heat flux is


$$
𝒒= 𝜅⋅∇𝑇,
(5)
$$

where 𝜅 is the heat conduction coefficient.

To calculate the convective fluxes between cells, the Roe Riemann solver is employed [98]. WENO5 reconstruction is applied in the solution method to accurately capture discontinuities, such as interface jumps and shocks [90,99]. A sharp level-set method is used to capture the gas-liquid interface [100–102]. ALPACA has been utilized previously for many bubble dynamics investigations, including bubble jetting near walls [103,104], acoustically excited bubbles [78, 93], shock-induced bubble collapses [105], and laser-induced bubble jets [106,107]. The interested reader is referred to the original software publication by Hoppe et al. [88] for more information about the software. Comprehensive convergence studies were previously carried out in ALPACA for single bubbles [93] and bubble pairs [78].

2.1.1. Noble Abel stiffened gas equation of state (NASG EoS)

To close the governing equations, the connection between pressure, density, and specific internal energy is provided by an equation of state. In this study, we employ the Noble-Abel Stiffened Gas (NASG) Equation of State (EoS) [96]:


$$
𝑝(𝑣, 𝑒) = (𝑛−1) 𝑒−𝑞
$$


$$
𝑣−𝑏−𝑛𝐵,
(6)
$$

where 𝑝 is the pressure, 𝑒 is the specific internal energy, and 𝑣= 1∕𝜌 is the specific volume. The parameters include the polytropic exponent 𝑛, co-volume 𝑏, pressure constant 𝐵, and energy constant 𝑞, which must be specified separately for the two phases. When 𝑏= 0 and 𝑞= 0, the NASG EoS simplifies to the stiffened gas EoS [108,109]. It further simplifies to the ideal gas EoS when 𝐵= 0.

The NASG parameters are readily available in the literature for water [64,110], while parameters for glycerol can be fitted based on guidelines published in [96]. For sulfuric acid, a complete parameter fitting was not possible due to the lack of measurement data; thus, the accuracy of the EoS is limited to lower pressures. However, this limitation does not affect the comparison of the models. The parameter fitting is described in detail in Appendix A. The fitted NASG parameters together with the rest of the parameters are summarized in Table A.4 in Appendix.

2.2. Spherical bubble models

This subsection introduces the Keller–Miksis and the Gilmore models. The equations are written in general form for an ensemble with 𝑁 bubbles.

2.2.1. Keller–Miksis equation

The Keller–Miksis equation describing the radial dynamics 𝑅𝑖(𝑡) of the 𝑖th bubble, located at (𝑥𝑖, 𝑦𝑖) is written as [43]:


$$
( 1 − ̇𝑅𝑖 𝑐𝐿
$$

) 𝑅𝑖̈𝑅𝑖+ ( 1 − ̇𝑅𝑖 3𝑐𝐿

) 3 2 ̇𝑅2 𝑖

= ( 1 + ̇𝑅𝑖 𝑐𝐿 + 𝑅𝑖

𝑐𝐿

d d𝑡


$$
) 𝑝𝐿,𝑖(𝑅𝑖, 𝑡) −𝑝∞(𝑥𝑖, 𝑦𝑖, 𝑡) −𝑝B,𝑖(𝑡)
$$


$$
𝜌𝐿 , (7)
$$

where 𝑖= 1 … 𝑁, 𝑐𝐿 is the speed of sound in the liquid, 𝜌𝐿 is the density of the liquid, 𝑝∞(𝑥, 𝑦, 𝑡) is the ultrasonic excitation pressure, and 𝑝B,𝑖(𝑡) is the pressure contribution from neighboring bubbles on the 𝑖th bubble. The liquid pressure at the bubble interface is given as


$$
𝑝𝐿,𝑖(𝑅𝑖, 𝑡) = 𝑝𝐺,𝑖(𝑡) + 𝑝𝑉−4𝜇𝐿 ̇𝑅𝑖(𝑡) 𝑅𝑖(𝑡) − 2𝜎 𝑅𝑖(𝑡) , (8)
$$

where 𝑝𝑉 is the vapor pressure, 𝜇𝐿 is the viscosity of the liquid, and 𝜎 is the coefficient of surface tension. To facilitate comparison with the multiphase flow simulations, the vapor pressure is neglected, i.e., 𝑝𝑉= 0. The gas pressure inside the 𝑖th bubble is given as

𝑝𝐺,𝑖(𝑡) = ( 2𝜎 𝑅𝐸,𝑖 + 𝑝0 −𝑝𝑉

) ( 𝑅𝐸,𝑖

𝑅𝑖(𝑡)


$$
)3𝑛𝐺 , (9)
$$

where 𝑅𝐸,𝑖 is the equilibrium radius of the 𝑖th bubble, 𝑝0 is the ambient pressure, and 𝑛𝐺 is the polytropic exponent of the gas inside the bubble.

Ultrasonics Sonochemistry 123 (2025) 107651

3

D. Nagy and F. Hegedűs

2.2.2. Gilmore model

The radial dynamics of a spherical bubble governed by the Gilmore model is given by [46]

( 1 − ̇𝑅𝑖 𝑐𝐿,𝑖

) 𝑅𝑖̈𝑅𝑖+ ( 1 − ̇𝑅𝑖 3𝑐𝐿,𝑖

) 3 2 ̇𝑅2 𝑖= ( 1 + ̇𝑅𝑖 𝑐𝐿,𝑖

) 𝐻𝑖+ ( 1 − ̇𝑅𝑖 𝑐𝐿,𝑖

) 𝑅𝑖̇𝐻𝑖

𝑐𝐿,𝑖 ,


$$
(10)
$$

where 𝐻𝑖= ℎ(𝑝𝐿,𝑖) −ℎ(𝑝∞,𝑖) is the enthalpy difference, ℎ(𝑝𝐿,𝑖) is the liquid enthalpy at the bubble wall, ℎ(𝑝∞,𝑖) is the enthalpy in the far field that also contains the contribution terms from neighboring bubbles, and 𝑐𝐿,𝑖 is the local speed of sound at the bubble wall. The liquid enthalpy and speed of sound are calculated using the NASG EoS [96] as

ℎ= 𝑛 𝑛−1 𝑝𝐿+ 𝐵

𝜌𝐿 − 𝑛 𝑛−1 𝑏(𝑝𝐿+ 𝐵) + 𝑏𝑝𝐿+ 𝑞, (11)

𝑐𝐿=


$$
√
$$

𝑛 𝑝𝐿+ 𝐵 𝜌(1 −𝑏𝜌𝐿) , (12)

where 𝑛, 𝑏, 𝐵, and 𝑞 are parameters of the NASG EoS. Note that the value of the energy constant 𝑞 is irrelevant in the Gilmore model, as it cancels out in the enthalpy difference 𝐻; however, it still influences the results of multiphase flow simulations. In this study, the Gilmore equation with the NASG EoS is solved using the APECSS library [111].

2.3. Modeling the pressure emission

This section describes the modeling of pressure emissions from bubbles. The general system of equations for 𝑁 bubbles is introduced. Later, the equations are used for a bubble pair with 𝑁= 2, with bubbles placed along a vertical symmetry axis with an inter-bubble distance 𝐷. The simplified equations solved for a bubble pair are included in Appendix B.

2.3.1. Model 1: Incompressible pressure emission

The incompressible pressure emission model is widely used [28, 30–33]. The acoustic pressure emitted by the 𝑗th bubble, under the assumption of an incompressible liquid, is given by

𝑝ac,𝑗(𝑥, 𝑦, 𝑡) = 𝜌𝐿 𝑟𝑗(𝑥, 𝑦)

( 2 ̇𝑅2 𝑗(𝑡)𝑅𝑗(𝑡) + 𝑅2 𝑗(𝑡) ̈𝑅𝑗(𝑡) ) , (13)

where 𝑟𝑗(𝑥, 𝑦) = √

(𝑥−𝑥𝑗)2 + (𝑦−𝑦𝑗)2 denotes the distance from the 𝑗th bubble. The pressure contribution from neighboring bubbles to the 𝑖th bubble is then computed as

𝑝B,𝑖(𝑡) =

𝑁 ∑


$$
𝑗=1 𝑗≠𝑖
(14)
$$

This term is then incorporated into the Keller–Miksis equation as given in Eq. (7). The incompressible pressure emission model is solved numerically using Wolfram Mathematica. This model is used only in the case of the Keller–Miksis equation; see Table 1.

Model 1a: No-delay

The simplest variant of the incompressible pressure emission models is the no-delay model. In this model, the acoustic pressure emission is described by Eq. (13) and the pressure contribution to the 𝑖th bubble by Eq. (14). In this simplified model, the time delay associated with the finite speed of sound is neglected, i.e., it is assumed that the ultrasonic pulse reaches both bubbles simultaneously and the pressure emissions of the bubbles reach each other immediately. Consequently, the farfield pressure is assumed to depend solely on time for each bubble:


$$
𝑝∞(𝑥𝑖, 𝑦𝑖, 𝑡) = 𝑝∞(𝑡) = 𝑝0 −𝑝𝐴sin (2𝜋𝑓𝑡) .
(15)
$$

This configuration is equivalent to the case where the two bubbles are aligned horizontally along the 𝑥-axis.

This is a widely used model to this day to study the dynamics of bubble clusters [32,112–115], mainly due to its simplicity and computational efficiency. It requires the solution of ordinary differential equations, for which many efficient approaches exist already [116– 119]. This computational efficiency comes at the cost of neglecting the time delay, that can lead to inaccuracies as demonstrated later.

Model 1b: Geometric time delay

Similarly to Model 1a, the pressure contribution from the neighboring bubble is calculated using Eq. (14). However, in this model, the geometric time delay associated with the propagation of the ultrasonic wave between the bubbles is taken into account. Specifically, the farfield pressure acting on each bubble is evaluated at its respective position:


$$
𝑝∞(𝑥𝑖, 𝑦𝑖, 𝑡) = 𝑝0 −𝑝𝐴sin ( 2𝜋𝑓 ( 𝑡−𝑦𝑖
$$

𝑐


$$
)) . (16)
$$

This model is rarely used, and the geometric time delay is often neglected when bubble clusters are simulated, although it is similarly efficient from a computational perspective as Model 1a, and also considers the bubble layout more accurately.

Model 1c: Both time delays

Although assuming a finite propagation velocity in an incompressible model is not physically rigorous, it can be introduced as an approximation to study the role of time delays in pressure coupling [120]. Then, Eq. (14) must be modified to:

𝑝B,𝑖(𝑡) =


$$
𝑁 ∑
$$


$$
𝑗=1 𝑗≠𝑖
(17)
$$

where 𝜏𝑒 is the emission time delay between the 𝑖th and 𝑗th bubbles. The far-field pressure 𝑝∞ is given by Eq. (16) as in the previous model. The resulting system of delay differential equations (DDE) based on Eq. (7) and (17) is solved using Wolfram Mathematica in this study.

Note that the emission time delay in pressure coupling between the bubbles, denoted by 𝜏𝑒, and the geometric time delay due to the bubbles being reached at different times by the pulse, denoted by 𝜏geo, have the same value when two bubbles lie along the symmetry axis, as both represent the time it takes for information to travel between the two bubbles. However, we use different notations to distinguish these two effects. Moreover, if the bubbles are not aligned along the symmetry axis, 𝜏𝑒 and 𝜏geo will generally have different values.

This model has some application [82]. Although this is still based on the incompressible assumption, it reconstructs the pressure field qualitatively well and can capture outgoing pressure waves from collapsing bubbles. However, it must be noted that solving DDEs numerically is generally more difficult and less computationally effective due to the irregular memory operations required for handling the delayed terms. This is especially painful for massively parallel architectures (e.g., GPUs) where large caches for a single system are not available.

2.3.2. Model 2: Quasi-acoustic assumption

The quasi-acoustic model assumes an incompressible liquid with finite propagation velocity, determined by the speed of sound in the liquid [50,51,64]. Following the approach of Coulombel and Denner [51], the pressure contribution to the 𝑖th bubble from all neighboring bubbles is expressed as

𝑝B,𝑖(𝑡) = 𝜌𝐿


$$
𝑁 ∑
$$


$$
𝑗=1,𝑗≠𝑖
$$


$$
𝑔𝑗(𝑡−𝜏𝑒)
$$

𝛥𝑟𝑖𝑗 + 𝜌𝐿

2

⎡ ⎢ ⎢⎣


$$
𝑁 ∑
$$


$$
𝑗=1,𝑗≠𝑖
$$


$$
𝜙𝑗(𝑡−𝜏𝑒)
$$

𝛥𝑟2 𝑖𝑗 + 1

𝑐𝐿


$$
𝑁 ∑
$$


$$
𝑗=1,𝑗≠𝑖
$$


$$
𝑔𝑗(𝑡−𝜏𝑒)
$$

𝛥𝑟𝑖𝑗

⎤ ⎥ ⎥⎦

2


$$
, (18)
$$


$$
where 𝛥𝑟𝑖𝑗= √
$$

(𝑥𝑖−𝑥𝑗)2 + (𝑦𝑖−𝑦𝑗)2 is the distance between 𝑖th and 𝑗th bubble, and 𝜏𝑒= 𝛥𝑟𝑖𝑗∕𝑐𝐿 accounts for the finite propagation speed. The invariants 𝜙𝑗 and 𝑔𝑗 are defined as [51]:


$$
𝜙𝑗(𝑡−𝜏𝑒) = 𝑅𝑗(𝑡−𝜏𝑒)2 ̇𝑅𝑗(𝑡−𝜏𝑒) −𝑅𝑗(𝑡−𝜏𝑒)
𝑔𝑗(𝑡−𝜏𝑒)
$$


$$
𝑐𝐿 , (19)
$$

Ultrasonics Sonochemistry 123 (2025) 107651

4

D. Nagy and F. Hegedűs


$$
𝑔𝑗(𝑡−𝜏𝑒) = 𝑅𝑗(𝑡−𝜏𝑒)
$$

[ 𝑝𝐿,𝑗(𝑡−𝜏𝑒) −𝑝∞(𝑥𝑗, 𝑦𝑗, 𝑡−𝜏𝑒) −𝑝B,𝑗(𝑡−𝜏𝑒)


$$
𝜌𝐿 + ̇𝑅𝑗(𝑡−𝜏𝑒)2
$$

2

]

.


$$
(20)
$$

Eqs. (7), (18)–(20) together form a system of delay differential–algebraic equations (DDAEs). An alternative to directly solving such DDAEs is the Lagrangian wave tracking method proposed by Denner and Schenke [64], in which emission nodes are propagated along outgoing characteristic lines with velocity 𝑐𝐿. To solve Model 2 using this method, we utilize the APECSS library developed by Denner and Schenke [111]. This emission model is used together with the Keller–Miksis equation as given in Table 1.

The main advantage of this model compared to the incompressible pressure emission is the physically more consistent modeling of the pressure emissions with a finite speed of sound. It captures the interaction more accurately as the pressure disturbances travel between bubbles at a finite speed, interact nonlinearly as given in Eq. (19) and (20), and shape future pressure emissions as well. The numerical solution of DDAEs is complicated, and high-performance solvers do not exist. Lagrangian wave tracking uses a straightforward algorithm: in each time step, pressure emissions are propagated along outgoing characteristics at the speed of sound. However, since bubble emissions are continuous over time, the number of propagating wave fronts increases, requiring the integration of an accumulating set of emission events. This is computationally much slower than the previous methods, and it requires a significant amount of memory, that can be problematic for large bubble clusters.

2.3.3. Model 3: Kirkwood–Bethe hypothesis

The Kirkwood–Bethe hypothesis [52,53] provides a physically consistent framework for modeling wave propagation in a fully compressible liquid. According to this hypothesis, the velocity of information propagation along an outgoing characteristic is not merely the local speed of sound, but rather the sum of the speed of sound and the local particle velocity [54]. The equations below are given in the local spherical coordinate system of a bubble with radial coordinate 𝑟. The radial position of the outgoing pressure wave is described by

d𝑟(𝑡)

d𝑡 = 𝑐(𝑟, 𝑡) + 𝑢(𝑟, 𝑡), (21)

where 𝑐(𝑟, 𝑡) is the local speed of sound and 𝑢(𝑟, 𝑡) is the particle velocity in the liquid at position 𝑟 and time 𝑡. This formulation reflects the fact that acoustic disturbances travel faster when carried by a moving fluid, and is essential for capturing nonlinear wave propagation near rapidly collapsing bubbles [64]. In this model, the particle velocity 𝑢(𝑟, 𝑡) can be explicitly computed as [111]:

𝑢(𝑟, 𝑡) = 𝑅(𝑡𝐸)2 ̇𝑅(𝑡𝐸)

𝑟(𝑡)2 − 𝑅(𝑡𝐸)𝑔(𝑡𝐸)

𝑟(𝑡)2(𝑐𝐿(𝑡𝐸) + ̇𝑅(𝑡𝐸)) + 𝑔(𝑡𝐸)


$$
𝑟(𝑡)(𝑐(𝑟, 𝑡) + 𝑢(𝑟, 𝑡)), (22)
$$

where 𝑡𝐸 denotes the emission time, i.e., the time at which the wavefront was emitted from the bubble wall and began propagating outward. The term 𝑔 is computed as

𝑔(𝑡𝐸) = 𝑅(𝑡𝐸) ( ℎ(𝑝𝐿(𝑡𝐸)) −ℎ(𝑝∞(𝑡𝐸)) + ̇𝑅(𝑡𝐸)2

2


$$
) , (23)
$$

with ℎ(𝑝) denoting the enthalpy as a function of pressure, calculated using the NASG EoS (see Section 2.2.2). Finally, the enthalpy is found as


$$
ℎ(𝑟, 𝑡) = ℎ(𝑝∞(𝑡)) + 𝑔(𝑡𝐸)
$$


$$
𝑟(𝑡) −𝑢(𝑟, 𝑡)2
$$

2 . (24)

For an ensemble of bubbles, the pressure field is computed by superimposing the contributions from all bubbles, where the enthalpy associated with each emission is evaluated according to Eq. (24) and then expressed and added up in the global coordinate system. The tracking of outgoing characteristics under the Kirkwood–Bethe assumption is implemented using Lagrangian wave tracking in the APECSS library [111], which is used later in this study.

The Kirkwood–Bethe hypothesis in Model 3 is always used together with the Gilmore equation, with both using the same EoS. As Lagrangian wave tracking is employed for the computation, it suffers from high compute costs as Model 2. The advantage of this model lies in its accuracy: it can predict the outgoing acoustic pressure accurately, and even the formation of shock fronts.

2.4. Error analysis

To quantitatively compare the models in the remainder of the article, the DNS carried out with the ALPACA solver is taken as a reference. The error of the coupled-spherical-bubble models is calculated based on their deviation from these simulations. Specifically, the RMS error in the bubble radius of a selected coupled-spherical-bubble model is estimated as

𝐸RMS =


$$
√
$$


$$
1 𝑇∫
$$

𝑇

0


$$
( 𝑅(𝑡) −𝑅DNS(𝑡)
$$

𝑅DNS(𝑡)


$$
)2 d𝑡, (25)
$$

where 𝑅(𝑡) is the bubble radius from the spherical model, 𝑅DNS(𝑡) is the equivalent radius from the DNS, and 𝑇 is the first collapse time. From a chemical perspective, however, the most crucial stage is bubble collapse. For this reason, the error in the minimal bubble radius is additionally calculated as

𝐸min = ||||||

𝑅min −𝑅DNS min 𝑅DNS min


$$
|||||| , (26)
$$

where 𝑅min is the minimal bubble radius during collapse according to the coupled-spherical-bubble model, and 𝑅DNS min  is the minimal equivalent bubble radius in the DNS. These two error measures are used in the following sections to systematically evaluate the accuracy of the coupled-spherical-bubble models against DNS.

3. Spherical bubble dynamics

Bubbles rarely collapse spherically in sonochemistry. Nonetheless, understanding the limitations of spherical bubble models in this hypothetical case is crucial. As discussed earlier, the main limitations of spherical models are in the simplified modeling of liquid compressibility and viscosity. In this section, different spherical bubble models (Keller–Miksis and Gilmore) and DNS simulations are validated by measurements from Ketterling and Apfel [121,122]. Subsequently, simulations are conducted over a wide range of pressure amplitudes in three different liquids, for fixed bubble radius (𝑅𝐸= 20 μm) and excitation frequency (𝑓= 100 kHz). These parameters are selected since they are relevant for various physical processes, including wastewater treatment [123] and ammonia production [5]. At the same time, DNS simulations can also be run in a reasonable time with sufficiently high resolution [78].

The bubble in the following simulations is excited by the changing background pressure:


$$
𝑝∞(𝑡) = 𝑝0 −𝑝𝐴sin(2𝜋𝑓𝑡),
(27)
$$

where 𝑝0 is the ambient pressure, 𝑓 is the excitation frequency and 𝑝𝐴 is the pressure amplitude. In the DNS simulations in ALPACA, the bubble is placed in a spherical standing wave, where the velocity, pressure and density are initialized based on Eqs. (C.11)–(C.13) derived in Appendix C. The bulk viscosity is neglected.

3.1. Experimental validation of spherical models

Experiments on single-bubble dynamics were conducted by Ketterling and Apfel [121,122], and their data are used here for model validation. They measured the oscillation of a single bubble with an equilibrium radius of approximately 𝑅𝐸= 4.6 μm in an acoustic field with a driving frequency of 𝑓= 33.4 kHz and a pressure amplitude

Ultrasonics Sonochemistry 123 (2025) 107651

5

D. Nagy and F. Hegedűs


> **Fig. 1. Validation of spherical bubble simulations against experimental data from Ketterling and Apfel [121]. The results of the spherical models are shown after the fifth acoustic period, with the phase in the simulations aligned to the measurements. The first collapse and subsequent rebounds are shown in detail.**

of 𝑝𝐴= 0.123 MPa. The measurements had high temporal resolution (0.2 μs) and excellent spatial accuracy, with the maximum bubble radius determined within 1.1 μm. Their reported radius–time curve represents an average over several oscillation periods. To ensure comparison with spherical models, the numerical simulations of the single-bubble models were also analyzed after a transient of five acoustic periods, after which the 𝑅(𝑡) curve was periodic.


> **Fig. 1 compares the experimental data with the predictions of the spherical models. The results show that the predictions of the spherical models are nearly identical, exhibiting no qualitative differences. Moreover, there is very good agreement with the experimental measurements following the initial transient: the first collapse and subsequent rebound are captured with remarkable accuracy. Following the first rebound, several rapid rebound oscillations are observed with an increasing frequency. During this phase, the small-amplitude oscillations are less pronounced in the experimental data than in the simulations, which may be attributed to the temporal averaging of multiple oscillation cycles. Nevertheless, the mean bubble radius remains in good agreement with the experimental results, and the onset of the subsequent growth phase is also accurately captured by the spherical models.**

These results demonstrate that the spherical models reproduce the experimental measurements with excellent accuracy, this validates their applicability under the tested conditions. However, it should be noted that the available experimental data lack sufficient temporal resolution to capture the dynamics near the bubble collapse. For instance, in Fig. 1, the minimum radius following the first collapse at approximately 11 μs cannot be determined with confidence. The bubble behavior during this collapse is particularly critical from a sonochemical perspective, as the extreme temperature and pressure inside the bubble govern the rate of chemical reactions. Since many theoretical studies on sonochemistry rely on spherical bubble models [1,5,124], it is essential to further investigate the bubble dynamics near collapse using DNS simulations. Therefore, in the subsequent sections, we focus on parameters relevant to sonochemistry and compare the spherical models exclusively with DNS simulations, while omitting experimental data that lack the temporal resolution required. The comparisons are then also repeated for liquids of higher viscosity, including sulfuric acid and glycerol.

3.2. Verification of the spherical models in water

The effect of the pressure amplitude is investigated in water for fixed bubble radius (𝑅𝐸 = 20 μm) and excitation frequency (𝑓 = 100 kHz). The viscosity of water is low, as such it is not expected to have a significant influence on the results. The collapse strength of

the bubbles is controlled via the pressure amplitude, and it allows for testing the applicability limit of the different models with respect to compressibility. To get a quick picture of the collapse strength, the compression ratio is calculated for each case as

𝛾= 𝑅𝐸


$$
𝑅min , (28)
$$

where 𝑅min is the minimum bubble radius during collapse and 𝑅𝐸 is the equilibrium bubble radius. The compression ratio varies between 1 and 20 in the simulations as shown in Table 2. The minimum radius 𝑅min and the maximum gas pressure inside the bubble 𝑝𝐺,max are also recorded for each case in Table 2. The results are plotted in Fig. 2a– b. For lower pressure amplitudes, all models predict similar values. The bubble radius as a function of time is illustrated in Fig. 3 for a bubble excited with a 𝑝𝐴= 0.15 MPa pressure amplitude, for which the compression ratio is 𝛾= 5.17. Clear agreement between the models can be observed, as expected.

With increasing pressure amplitude, the discrepancy in the minimum radius between the models becomes more pronounced; see Fig. 2a. For the case with 𝑝𝐴= 1 MPa, the bubble radius as a function of time is illustrated in Fig. 4, with a compression ratio of 𝛾= 16.56. The models agree well up until the very end of the bubble collapse. In the Keller– Miksis equation, a larger error can be found near the collapse, with a temporal shift and larger minimum radius. Similar conclusions can be drawn from the maximum bubble pressure in Fig. 2b. The Gilmore equation is more accurate at large pressure amplitudes as well.

The difference in the results at strong bubble collapse can be explained by the Mach number of the bubble wall during collapse, which is given by [125,126]


$$
Ma = ̇𝑅 𝑐𝐿,𝐵 , (29)
$$

where 𝑐𝐿,𝐵 is the speed of sound in the liquid at the bubble wall. The Keller–Miksis equation assumes a constant speed of sound at the bubble wall (𝑐𝐿,𝐵= const.). It also neglects the second-order terms with the Mach number; thus, it is expected to be accurate only for small Mach numbers. Second-order terms are considered in the Gilmore equation, and the speed of sound is calculated from the NASG EoS. The Mach number for various pressure amplitudes is illustrated in Fig. 2c. The Mach number varies between 0 and 1. Good agreement in the bubble radius for low pressure amplitudes 𝑝𝐴≤0.2 MPa corresponds to the low Mach number cases with around Ma ≤0.2. For higher pressure amplitudes, the Keller–Miksis equation becomes inaccurate due to the high Mach number. In such cases, the Gilmore equation also shows inaccuracies, however, it remains closer to the DNS results even for Ma approaching one.

Ultrasonics Sonochemistry 123 (2025) 107651

6

D. Nagy and F. Hegedűs


> **Table 2 Summary of maximum bubble compression for different pressure amplitudes. The compression ratio (𝛾) and Mach number (Ma) are determined from the ALPACA simulations. The 𝑅AL min, 𝑅KM min and 𝑅G min values denote the minimum bubble radius during collapse for the ALPACA, Keller–Miksis and Gilmore simulations, respectively. Similarly, 𝑝AL G,max, 𝑝KM G,max and 𝑝G G,max denote the maximum bubble pressure for the ALPACA, Keller–Miksis and Gilmore simulations, respectively. 𝑝𝐴 𝛾 Ma 𝑅AL min 𝑅KM min 𝑅G min 𝑝AL G,max 𝑝KM G,max 𝑝G G,max [MPa] [μm] [μm] [μm] [MPa] [MPa] [MPa] 0.01 1.0667 0.0006 18.748 18.734 18.734 0.131 0.131 0.131 0.03 1.2578 0.0027 15.900 15.878 15.878 0.263 0.263 0.263 0.06 1.7610 0.0087 11.357 11.336 11.336 1.083 1.085 1.085 0.1 3.0047 0.0275 6.656 6.654 6.661 10.221 10.165 10.125 0.15 5.1659 0.0988 3.871 3.890 3.907 99.995 96.935 95.110 0.2 7.1149 0.1908 2.811 2.829 2.84 411.94 369.37 362.40 0.3 10.1890 0.3319 1.962 2.076 2.019 1751 1353 1523 0.5 14.6994 0.5160 1.360 1.651 1.478 8613 3537 5637 0.7 17.2191 0.5842 1.161 1.525 1.304 17 049 4943 9536 1 16.5577 0.5633 1.207 1.555 1.355 13 540 4554 8117 1.5 20.2819 0.7758 0.986 1.257 0.890 37 236 11 118 47 389**


> **Fig. 2. (a–c) Minimum bubble radius, maximum internal gas pressure and maximum Mach number of the bubble wall at the first collapse as a function of the pressure amplitude. (d) The error of the Keller–Miksis (𝐸KM) and Gilmore equations (𝐸G) compared to the DNS based on the RMS error (𝐸RMS) and the relative error of the minimum radius (𝐸min). The parameters are 𝑅𝐸= 20 μm and 𝑓= 100 kHz.**

The accuracy of the spherical models is quantitatively investigated in Fig. 2d based on the RMS error in the first expansion-collapse cycle (𝐸RMS) according to Eq. (25) and on the deviation of the bubble radius during collapse (𝐸min) defined in Eq. (26). The analysis confirms the previous findings: for both the Keller–Miksis and Gilmore equations, the deviations remain limited for 𝑝𝐴≤0.2 MPa, corresponding to Ma ≤ 0.2. Although both models describe the first expansion-collapse cycle relatively well at large pressure amplitudes, with 𝐸RMS < 0.05, this hides the fact that discrepancies accumulate during collapse. The error

in the minimal radius reaches 0.17 for the Gilmore model and 0.64 for the Keller–Miksis equation as the maximum Mach number of the bubble wall approaches unity. As expected, the error of the Gilmore model, which considers the liquid compressibility more accurately, is significantly lower compared to the Keller-Miksis equation.

Nonetheless, the error of the Gilmore model must be explained. Spherical bubble models such as the Gilmore assume homogeneous pressure inside the bubble. However, this assumption breaks down in the case of a substantial bubble collapse, and the gas inside the bubble

Ultrasonics Sonochemistry 123 (2025) 107651

7

D. Nagy and F. Hegedűs


> **Fig. 3. Comparison of bubble radius predicted by different models in water. The parameters are 𝑅𝐸= 20 μm, 𝑓= 100 kHz and 𝑝𝐴= 0.15 MPa.**


> **Fig. 4. Comparison of bubble radius predicted by different models in water. The parameters are 𝑅𝐸= 20 μm, 𝑓= 100 kHz and 𝑝𝐴= 1 MPa.**

becomes inhomogeneous in the DNS. Fig. 5 illustrates a time series of a bubble near its minimum radius. At 12.2402 μs, the bubble reaches its minimal size, and the pressure inside the bubble reaches 30 GPa, while near the bubble wall it remains around 10 GPa. Additionally, the corresponding temperatures during collapse were estimated. The liquid temperature near the bubble wall remains below 700 K, which is well within the validity range of the Noble-Abel Stiffened Gas equation of state. The internal bubble temperature reaches 10 000 K where the ideal gas assumption is less accurate. However, this does not influence the comparison of models, as they all assume ideal gas. Furthermore, these temperatures can induce chemical reactions. Previous studies indicate that such reactions have negligible influence on the radial dynamics prior to collapse but may have some effect during the rebound phase, as directly shown by Wang et al. [127]. At the same time, the differences in the internal temperature and pressure of the bubble are much more significant. However, models that couple spherical bubble dynamics with chemical reactions [1,124] assume a spatially homogeneous distribution of temperature, pressure, and species within the bubble, which is invalid in the case of strong collapses. Therefore, these models must be employed with caution under such extreme conditions.

3.3. Effect of viscosity

The previously described simulations were repeated for sulfuric acid and glycerol, in order to explore the other main limiting factor,

the viscosity. The dynamic viscosity of sulfuric acid and glycerol is 0.025 Pa ⋅s and 1.468 Pa ⋅s, respectively. Figs. 6a and 6b show the minimum bubble radius during collapse in the two liquids. Similarly to the case with water, the maximum Mach number varies between 0 and 1 in the simulations, as seen in Figs. 6c and 6d. The errors in the bubble radius are calculated based on the RMS and minimum radius error and included in Figs. 6e and 6f for sulfuric acid and glycerol, respectively.

In sulfuric acid (moderate viscosity), there is relatively good agreement for low pressure amplitudes as exemplified by the minimal bubble radius in Fig. 6a. Fig. 7 shows the bubble radius with various models in sulfuric acid for 𝑝𝐴= 0.15 MPa, where a visible difference appears between DNS and the spherical models during the final stage of collapse. In glycerol (high viscosity), however, the discrepancies are much larger. Even at low pressure amplitudes, significant differences exist between the models, as depicted in Fig. 6b. Fig. 8 further demonstrates that, despite the maximum Mach numbers being of the same order of magnitude in this case (see Fig. 6d), the 𝑅(𝑡) curves reveal particularly large deviations.

The quantitative comparison of models in sulfuric acid (Fig. 6e) shows that at low pressure amplitudes the error in the minimum bubble radius is already about 0.05. The RMS error remains small, since most of the discrepancy accumulates during collapse, as illustrated in Fig. 7, which depicts the first oscillation cycle. With increasing pressure amplitude (and stronger bubble compression), the errors of the spherical models diverge. In glycerol (Fig. 6f), the spherical models appear more accurate at low pressure amplitudes, with errors close to zero. However, this is only because high viscosity suppresses bubble oscillations. Considering the error in the oscillation amplitude instead reveals substantial discrepancies: already at 𝑝𝐴= 0.01 MPa the amplitude error reaches 31%, and at 𝑝𝐴= 0.15 MPa (Fig. 8) it increases further to 33%. As the pressure amplitude rises, the errors diverge according to all definitions.

These results indicate that spherical bubble models (Keller–Miksis and Gilmore) have larger errors in high-viscosity liquids. This results from the simplified modeling, as viscosity is only considered at the bubble wall and not in the flow around the bubble. Around the bubble, potential flow is assumed, and the linear wave equation is solved [43], which neglects viscous terms. At high viscosity, however, the viscous effects could dominate the flow around the bubble. The Reynolds number gives the proportion of inertial and viscous forces; thus, we should expect the spherical bubble models to work for cases with Re ≫ 1. The Reynolds number for a bubble oscillation can be found as

Re = 𝜌𝑅̇𝑅


$$
𝜇 , (30)
$$

where we can approximate the velocity of the bubble wall with an average velocity as


$$
̇𝑅= 𝑅max −𝑅min
$$


$$
𝑇∕2
= 2 ⋅𝑓⋅(𝑅max −𝑅min),
(31)
$$

where 𝑇 is the period and 𝑓 is the frequency of the excitation. The Reynolds number for the 𝑝𝐴= 0.15 MPa case in water, sulfuric acid, and glycerol are Re = 152.5, 8.1, and 0.01, respectively. Thus, in water, the liquid inertia clearly dominates the flow around the bubble. In sulfuric acid, the effect of liquid inertia is still larger than the effect of viscosity. However, in glycerol, Re ≪1, indicating creeping flow, and thus the potential flow assumption completely breaks down.

3.4. Pressure emission

While previous sections focused on reproducing the radial dynamics of individual bubbles, an equally important aspect is reconstructing the pressure emitted by these bubbles. This is critical in multi-bubble systems, where the emitted pressure governs the interactions between bubbles and thus the collective dynamics of the cluster. Many studies use simplified multi-bubble models that assume spherical symmetry

Ultrasonics Sonochemistry 123 (2025) 107651

8

D. Nagy and F. Hegedűs


> **Fig. 5. The pressure distribution inside the bubble around the minimum bubble size with high temporal resolution. The parameters are 𝑓= 100 kHz, 𝑝𝐴= 1 MPa and 𝑅𝐸= 20 μm.**

and introduce pressure coupling through various approximations [28, 112–116,120]. However, it is not sufficient to accurately predict the radial dynamics; the emitted pressure field must also be well-resolved, as it directly determines the resulting inter-bubble interactions. In this section, we assess the quality of pressure emission modeling by comparing four approaches from Table 1:

• DNS using the ALPACA solver (reference), • the Keller–Miksis equation with incompressible pressure emissions but finite propagation speed (Model 1c). • the Keller–Miksis equation with quasi-acoustic coupling (Model 2), • the Gilmore equation coupled with the Kirkwood–Bethe hypothesis (Model 3).

Models with infinite propagation speed are not considered, as the resulting pressure emissions are qualitatively incorrect. These comparisons provide a clear picture of how well each method can capture the emitted pressure field and thus, their potential suitability for cluster-scale simulations.


> **Fig. 9 depicts a comprehensive comparison of the models for various cases. Fig. 9a shows the pressure profile at two time steps after bubble collapse in water. The bubble is excited with an acoustic standing wave with amplitude 𝑝𝐴= 0.15 MPa. In this case, the bubble collapse is moderate with 𝛾= 5.17, without the subsequent formation of a steep shock front. There is good agreement between the models in this case. The peak pressure as a function of distance from the origin is depicted in Fig. 9b, where, in each model, a similar damping of the pressure with 𝑝∝𝑥−0.87 is apparent.**

For a large pressure amplitude of 𝑝𝐴= 1 MPa, the compression ratio reaches 𝛾= 16.56 and a steep shock front forms, as observed in Fig. 9c. The Kirkwood–Bethe hypothesis (KB), with Lagrangian wave tracking introduced by Denner and Schenke [64], captures the shock wave remarkably well; the results are similar to those of the DNS simulation. The quasi-acoustic model (QA) cannot capture this shock front due to its assumption of a constant speed of sound. The same problem occurs with the incompressible emission model with finite propagation speed, and it also overpredicts the peak pressure. Since the speed of sound is constant, the higher pressure at the wavefront does not cause the wave to catch up with previous emissions. Although the propagation speed of the shock is higher in ALPACA than in the QA model, the wave appears to move faster in the QA model. This seeming contradiction is resolved by examining the bubble dynamics, as the bubble collapses earlier in the Keller–Miksis equation, as depicted in Fig. 4. The pressure decay in Fig. 9d agrees well between the Kirkwood– Bethe model and the DNS. The colored numbers indicate the steepness of the corresponding curves. The steepness in two selected points in Fig. 9d is also remarkably similar between the DNS and the Kirkwood–Bethe model. The pressure decay in the quasi-acoustic model is somewhat slower, while in the incompressible model the peak pressure decays exactly with 𝑥−1.

In liquids of higher viscosity, shock fronts smooth out due to the damping effect of viscosity [128], as the kinetic energy of the shock is dissipated. This is evident from the ALPACA simulations in Figs. 9e and 9g for sulfuric acid and glycerol, respectively. The Kirkwood–Bethe hypothesis does not account for the viscosity of the liquid; thus, the shock front remains steep, leading to a particularly large error in Fig. 9g for glycerol. The higher viscosity also increases the decay rate of the shock front pressure, as observed in Figs. 9f and 9h. The results indicate that spherical bubble models coupled with simplified models for pressure emission are accurate in low-viscosity liquids, such as water. The accuracy of the quasi-acoustic and incompressible models is limited to bubble oscillations with low compression ratios, as it cannot capture shock wave formation due to the assumption of constant propagation velocity. The Kirkwood–Bethe hypothesis overcomes this limitation by accurately modeling the propagation velocity using an EoS, thereby correctly predicting the formation of shocks.

4. Bubble pairs in an acoustic field

Investigating many-bubble systems using a coupled-spherical-bubble approach is compelling due to its simplicity. This section explores the limitations of that method for a simplified case: a pair of bubbles excited by an ultrasonic pulse in water, as depicted in Fig. 10. The parameters of the ultrasonic excitation (𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa) and the equilibrium radius of both bubbles (𝑅𝐸= 20 μm) are based on previous work [78]. This case is particularly interesting for analysis because the inter-bubble distance (𝐷) governs the jetting behavior of the bubbles. Three cases are examined in detail: for 𝐷= 100 μm, the bubbles jet toward each other; for 𝐷= 320 μm, the bottom bubble jets upward while the top bubble collapses without jetting (so-called jet suppression); and for 𝐷= 840 μm, both bubbles jet in the direction of the ultrasonic pulse — that is, upward.

4.1. Simulation setup

In our setup, the ultrasonic pulse reaches the bottom bubble at 𝑡= 0, while the top bubble is reached at 𝑡= 𝜏geo, where 𝜏geo is the geometric time delay caused by the finite propagation speed of the ultrasonic pulse. The simulation domain has a size of 0.5𝜆× 2𝜆, where 𝜆= 𝑐𝐿∕𝑓 is the wavelength. The boundary condition on the west side (𝑥= 0 line) is symmetry, corresponding to the symmetry axis. All other boundary conditions are zero-gradient. The ultrasonic pulse is initialized in the bottom half of the domain, as shown in Fig. 10b. After the simulation starts, the pulse travels upward through the domain. The parameters are summarized in the Appendix in Table B.5. Due to the lower compression in the simulations compared to the spherical simulations in Section 3, the use of the stiffened gas EoS (simplified version of the NASG EoS with co-volume 𝑏= 0) was sufficient. Its parameters are based on [78,109] and given in the Appendix in Table A.4. For more details of such simulations, the interested reader is referred to the previous work of the authors [78].

Ultrasonics Sonochemistry 123 (2025) 107651

9

D. Nagy and F. Hegedűs


> **Fig. 6. (a–d) Minimum bubble radius and Mach number of the bubble wall as a function of the pressure amplitude in sulfuric acid and glycerol. (e–f) The error of the Keller–Miksis (𝐸KM) and Gilmore equations (𝐸G) compared to the DNS based on the RMS error (𝐸RMS) and the relative error of the minimum radius (𝐸min). The parameters are: 𝑅𝐸= 20 μm and 𝑓= 100 kHz.**

For demonstration purposes, Fig. 11 shows the ultrasonic pulse traveling upward through the bubble pair with inter-bubble distance 𝐷= 100 μm. First, the negative part of the pulse reaches the bubbles, causing them to expand (see the bottom row of images in Figs. 11a– c). Then, during the positive part of the pulse, the bubbles collapse, as seen in Fig. 11d. During the collapse, bubble jets may form in the DNS simulation, with the direction of the jet depending on the distance between the bubbles. For example, in Fig. 11d the bubbles jet

toward each other. Finally, the pressure emission from bubble collapse is apparent in the top row of Fig. 11e.


> **Fig. 12 shows the results of the ALPACA simulation for an interbubble distance of 𝐷= 320 μm. The bubbles remain spherical throughout the expansion, as seen in Figs. 12a–c. During collapse, the bottom bubble forms an upward jet during collapse, while the top bubble remains compact as seen in Fig. 12d. Finally, in Fig. 12e, the pressure wave emission is visible in the top image, whereas in the bottom image, the top bubble can be seen to become more spherical during expansion.**

Ultrasonics Sonochemistry 123 (2025) 107651

10

D. Nagy and F. Hegedűs


> **Fig. 7. Comparison of the bubble radius obtained using different models in sulfuric acid. The parameters are 𝑅𝐸= 20 μm, 𝑓= 100 kHz and 𝑝𝐴= 0.15 MPa.**


> **Fig. 8. Comparison of the bubble radius obtained using different models in glycerol. The parameters are 𝑅𝐸= 20 μm, 𝑓= 100 kHz and 𝑝𝐴= 0.15 MPa.**

4.2. Radial dynamics with incompressible emission models

Due to their popularity in the literature, the incompressible emission models are investigated in detail in this section. The incompressible pressure emission within the model is considered in three different ways, as already introduced in Section 2.3 and Table 1. In the nodelay model (1a), both the geometric and the emission time delay are neglected. Then, the geometric time delay is considered in model 1b. In the final incompressible emission model both delays, the geometric and the emission time delays are considered (1c).

The radius of the top and bottom bubbles is compared with the ALPACA results in Fig. 13 for the three different incompressible models. In Figs. 13a–b (𝐷= 100 μm), the bubbles jet toward each other. While in Fig. 13c–d (𝐷= 320 μm), the bottom bubble jets upwards and the top bubble remains compact. This means that good agreement should be expected for the top bubble in the latter case between the spherical models and the ALPACA simulation. The bubble shapes in the corresponding ALPACA simulations are visible in Figs. 11–12.

The no-delay model (1a) does not account for the fact that the top bubble is reached later by the ultrasonic pulse; as a result, it collapses 𝜏geo = 𝐷∕𝑐𝐿 earlier, as observed in Figs. 13b and 13d. The model that considers the geometric time delay (1b) is more accurate and captures the collapse time of the top bubble. The model that considers both delays (1c) further improves the accuracy of the predicted bubble radii. This is especially apparent when observing the dynamics of the bottom bubble after collapse in Figs. 13a and 13c. The rebound phase is best described by model 1c, when both delays are considered.

These results indicate that, in the case of traveling waves, the geometric time delay in the excitation of the top bubble must be accounted for to achieve accurate predictions. While the effect of the emission time delay is less significant for the radial dynamics, it still enhances the overall accuracy, especially after bubble collapse. In the remainder of this work, we focus only on the incompressible model that considers both delays (1c), as this is the most accurate. It is important to note, however, that the 1b–1c type models perform well only in the absence of jet formation, and this holds primarily for radial dynamics. Significant discrepancies will emerge when reconstructing the pressure field, particularly in cases involving collapse asymmetries. Since pressure coupling strongly influences collective bubble behavior, such deviations may become critical in scenarios with many bubbles.

4.3. Radial dynamics of the quasi-acoustic and Kirkwood–Bethe emission models

Before continuing with the pressure field reconstruction, let us discuss the accuracy of the radial dynamics when using the quasiacoustic and the Kirkwood–Bethe emission models. The Keller–Miksis equation coupled with the quasi-acoustic pressure emission assumption (Model 2), and the Gilmore model combined with the Kirkwood–Bethe hypothesis (Model 3), are surprisingly accurate as long as the bubble does not jet. The bubble radius, internal pressure, and shape near collapse are shown for various cases in Fig. 14. The first column (Fig. 14a) presents the reference case of a single spherical bubble with an equilibrium radius 𝑅𝐸= 20 μm collapsing in an ultrasonic field with excitation parameters 𝑓= 100 kHz and 𝑝𝐴= 0.15 MPa, with the same parameter set presented in Fig. 13 (this case also corresponds to the spherical bubble simulations from Section 3). As expected, there is near-perfect agreement between all models, with the maximum bubble pressure reaching 100 MPa. Furthermore, the error in the minimum bubble radius is reported in Table 3. The RMS error of the coupledspherical-bubble models remains small for bubble pairs, as in water, these models are extremely accurate and errors only accumulate following the collapse. For a single expansion–collapse cycle, it is below 1%; therefore, its calculation is omitted in the following, and the analysis focuses instead on the error in the minimum bubble radius.

When two bubbles of the same size are placed vertically in an ultrasonic pulse, the collapse becomes non-spherical. In case the distance between the bubbles is 𝐷= 320 μm, the bottom bubble forms an upward jet, as shown in the bottom row of Fig. 14b. In the ALPACA simulation, the pressure inside the bottom bubble reaches only 40 MPa, while the spherical bubble models predict 55 MPa. This difference is due to jetting, which reduces compression. In contrast, jetting is suppressed in the top bubble, allowing it to remain compact, although, the shape is significantly distorted, as shown in the bottom row of Fig. 14c. The maximum pressure in the top bubble reaches 75 MPa in ALPACA and 89 MPa in the spherical models. Such an approximately 20% difference still seems considerable. The error in the minimal bubble radius remains around 2–3% as given in Table 3. However, keep in mind that the jetting bottom bubble has an influence on the top bubble, which also has a contribution to the difference. It is reasonable to assume that if both bubbles exhibit jet-suppression, the results would be even closer to the ALPACA simulation. In this case, the compression nearly matches that of the single spherical bubble (see Fig. 14a).

As the bubbles are placed closer together, both bubbles exhibit jetting: the bottom bubble jets upwards and the top bubble downwards, as shown in Figs. 14d and 14e. In this case, the pressure inside the bottom bubble reaches only 9 MPa in ALPACA, whereas the spherical models predict 34 MPa. This nearly fourfold overestimation is due to the particularly wide jet formed during collapse. For the top bubble, ALPACA yields a pressure of 34 MPa, while the spherical models predict 84 MPa. Compared to the single spherical bubble case, the maximum pressure is significantly lower in ALPACA, due to the proximity of the bubbles and the especially wide bubble jet. The error in the minimal

Ultrasonics Sonochemistry 123 (2025) 107651

11

D. Nagy and F. Hegedűs


> **Fig. 9. Pressure emission and peak shock wave pressure as a function of radial distance of (a–b) a weakly collapsing bubble in water, (c–d) a strongly collapsing bubble in water, (e–f) a strongly collapsing bubble in sulfuric acid, (g–h) a strongly collapsing bubble in glycerol. The numbers in the right column of panels indicate the steepness of the pressure decay curves. The coloring of the numbers corresponds to the coloring of the curves.**

bubble radius is also large, it is around 20% for the bottom and 10% for the top bubble.

Finally, a larger inter-bubble distance of 𝐷= 840 μm was also investigated. In this case, both bubbles jet upwards, and the ALPACA simulations yield maximum pressures of 55 MPa and 69 MPa for the bottom and top bubbles, respectively. Interestingly, these maximum

internal pressures are lower than in the 𝐷= 320 μm case, even though the bubbles are further apart. The spherical models predict 66 MPa and 86 MPa, corresponding to errors of 20% and 24.6%, respectively. At the same time, the error in the minimal bubble radius remains below 1% for the coupled-spherical-bubble models based on the quasiacoustic and Kirkwood–Bethe coupling. This apparent discrepancy in

Ultrasonics Sonochemistry 123 (2025) 107651

12

D. Nagy and F. Hegedűs


> **Fig. 10. (a) Layout of a bubble pair excited with an ultrasonic pulse. 𝑅1 and 𝑅2 are the equilibrium radii of the bottom and top bubble, respectively. The distance between the bubbles is 𝐷, the wavelength of the ultrasonic pulse is 𝜆, the pressure amplitude of the pulse is 𝑝𝐴, and the pulse propagates at the speed of sound 𝑐𝐿. (b) Real scale picture of the problem and initial conditions in the multiphase flow simulations.**


> **Fig. 11. Bubble pair excited by an ultrasonic pulse. The vertical black line depicts the symmetry axis, and the black contour shows the gas-liquid interface. The parameters are 𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa, 𝑅𝐸= 20 μm and 𝐷= 100 μm.**

the errors arises because the minimum radius is determined from the equivalent spherical volume, which the spherical-models calculate well, however the internal pressure is strongly influenced by jetting during collapse.

To summarize, the pressure inside the bottom and top bubble is overestimated by 277.8% and 147.1% for bubbles very close to each

other (Figs. 14d–e), by 37.5% and 18.7% in the case of jet-suppression of the top bubble (Figs. 14b–c), and finally, by 20% and 24.6% for farapart bubbles jetting in the same-direction. These differences correlate with the extent of jet formation: the more pronounced and wider the jet, the greater the overestimation of the internal pressure by the spherical models. Although a detailed analysis of this correlation lies

Ultrasonics Sonochemistry 123 (2025) 107651

13

D. Nagy and F. Hegedűs


> **Fig. 12. Bubble pair excited by an ultrasonic pulse. The vertical black line depicts the symmetry axis, and the black contour shows the gas-liquid interface. The parameters are 𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa, 𝑅𝐸= 20 μm and 𝐷= 320 μm.**


> **Fig. 13. Comparison of the incompressible pressure emission coupling for various inter-bubble distances under different assumptions. The parameters are 𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa and 𝑅𝐸= 20 μm.**

Ultrasonics Sonochemistry 123 (2025) 107651

14

D. Nagy and F. Hegedűs


> **Fig. 14. Comparison of the bubble radius and the bubble pressure in different cases with parameters 𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa and 𝑅𝐸= 20 μm. Top row: bubble radius near the minimum size. Middle row: average pressure inside the bubble. Bottom row: bubble shape after collapse. The depicted ALPACA simulations (solid black line) were already introduced in [78].**


> **Table 3 Error of the minimum bubble radius (𝐸min) at the first collapse for the top and bottom bubbles using different models for the coupling. The parameters are 𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa and 𝑅𝐸= 20 μm. 𝐷∕μ𝑚 Bubble Incompressible (1c) Quasi-acoustic Kirkwood–Bethe 100 Bottom 21.05% 19.66% 19.54% 100 Top 11.72% 9.42% 10.04% 320 Bottom 1.51% 2.25% 2.27% 320 Top 0.98% 3.47% 3.31% 840 Bottom 4.19% 0.96% 0.96% 840 Top 2.73% 0.76% 0.58%**

beyond the scope of this work, the trend suggests that jet morphology plays a critical role in determining the accuracy of spherical approximations. Interestingly, when the jetting is suppressed (as in Fig. 14c), the internal pressure can be predicted with reasonable accuracy even for closely spaced bubbles. This shows the coupled-spherical-bubble models can be applied in certain non-jetting scenarios, making them a computationally efficient alternative to DNS.

4.4. Pressure field reconstruction

Accurate reconstruction of the pressure field is essential for reliable coupled-spherical-bubble simulations. Incompressible pressure emission models assuming infinite propagation speed will fail to reproduce qualitatively accurate pressure fields. Therefore, only the finite propagation speed models, listed in Table 1, are investigated further for the bubble pair. Fig. 15 qualitatively depicts the pressure field after bubble collapse for the inter-bubble distance 𝐷= 100 μm. While the location of the emitted pressure waves is consistent across all models, the amplitude of the emitted wave is reduced in the DNS simulation in ALPACA. This discrepancy is due to the lower bubble pressure resulting from jetting, as seen in Figs. 14d and 14f for the same case. Because

the pressure inside the bubble remains lower in the ALPACA simulation, the amplitude of the emitted wave is also reduced.


> **Fig. 16 presents a similar qualitative comparison of the pressure field for the increased inter-bubble distance 𝐷= 320 μm, in which case the top bubble collapses without jetting. As before, the location of the emitted pressure waves remains consistent across all models. Notably, the amplitude of the wave emitted from the top bubble shows improved agreement between the ALPACA simulation and the spherical bubble models. In contrast, the pressure wave originating from the bottom bubble is again significantly reduced in the ALPACA simulation due to jetting of the bottom bubble, consistent with the behavior observed in Fig. 15.**

For a more accurate comparison, the pressure along the vertical axis is plotted in Fig. 17 at various distance between the bubbles in two time steps: one near the maximum bubble size and one after the collapse of both bubbles. Fig. 17a shows the pressure profile near maximal size at 𝑡= 6.0 μs. The gray region highlights the bubble locations, with their centers located at 𝑦= 0 and 𝑦= 100 μm, and both bubbles having radii of approximately 40 μm at this time step. In the DNS simulation, the bubbles remain mostly spherical, with only minor distortion. This so-called flattening [129] is due to the inertia of the liquid between the bubbles, which resists the expansion. Between the bubbles, around

Ultrasonics Sonochemistry 123 (2025) 107651

15

D. Nagy and F. Hegedűs


> **Fig. 15. Pressure field around the bubble pair after collapse at 𝑡= 9.1 μs, the parameters are 𝑓= 100 kHz, 𝑅𝐸= 20 μm and 𝐷= 100 μm.**


> **Fig. 16. Pressure field around the bubble pair after collapse at 𝑡= 9.1 μs, the parameters are 𝑓= 100 kHz, 𝑅𝐸= 20 μm and 𝐷= 320 μm.**

𝑦≈50 μm, the spherical models predict a significantly smaller pressure in the liquid. This discrepancy arises because the non-sphericity is not accounted for in the spherical models and liquid inertia is neglected in the inter-bubble coupling. However, far from the bubbles, all models give similar and accurate predictions.


> **Fig. 17b shows the pressure profile around the bubbles after jetting at 𝑡= 9.1 μs. Both bubbles emit a pressure wave during collapse. Since the pressure remains in the order of magnitude of 1 MPa, the**

speed of sound does not change significantly in the liquid. Thus, the propagation velocity of the pressure wave is approximately the same as the ultrasonic pulse. This means that the pressure wave emitted from the bottom bubble reaches the top bubble 𝜏geo = 𝐷∕𝑐𝐿 later, which corresponds to the delay in the collapse of the top bubble. The pressure wave is emitted from the top bubble exactly as the pressure wave from the bottom bubble reaches it. For 𝑦< 0, two separate pressure peaks are observed, with a distance of 𝛥𝑦= 2𝜏geo ⋅𝑐𝐿= 2𝐷= 200 μm between

Ultrasonics Sonochemistry 123 (2025) 107651

16

D. Nagy and F. Hegedűs


> **Fig. 17. Comparison of pressure along the 𝑦-axis using different models. The parameters are 𝑓= 100 kHz, 𝑝𝐴= 0.15 MPa and 𝑅𝐸= 20 μm. The distance between the bubbles and the time stamps are denoted in each panel. The gray region indicates the location of the bubble. The gray line is the ambient pressure without the contribution of the bubbles. The pictograms on the left of each panel show the bubble shapes observed in ALPACA.**

them. For 𝑦> 0, the pressure waves emitted from the bubbles add up. In general, as seen in Fig. 17b, the spherical models overpredict the pressure peak due to the overprediction of internal bubble pressure during collapse.

As the bubbles are farther apart, flattening does not occur during expansion, and the bubbles remain spherical. Figs. 17c and 17e show the pressure profile at maximum bubble expansion for bubbles separated by 𝐷= 320 μm and 𝐷= 840 μm, respectively. In these cases, the models agree well even between the bubbles. Fig. 17d shows the pressure profile after collapse in the 𝐷= 320 μm case. The top bubble

maintains its integrity (but non-spherical) in that case, while the bottom bubble jets in the upward direction. Three distinct pressure peaks are visible. The first peak, located around 𝑥≈−750 μm, corresponds to the pressure wave emitted during the collapse of the bottom bubble. Since this bubble undergoes jetting, there is a substantial discrepancy in amplitude between the DNS using ALPACA and the coupled-sphericalbubble models. The second peak, around 𝑥≈−100 μm, originates from the top bubble and propagates in the negative 𝑥-direction. A significant discrepancy remains here as well, primarily due to the interaction of this pressure wave with the jetting bottom bubble, which strongly

Ultrasonics Sonochemistry 123 (2025) 107651

17

D. Nagy and F. Hegedűs

influences the pressure field. The third peak, at 𝑥≈750 μm, is also part of the wave emitted from the top bubble but travels in the positive 𝑥-direction. In this case, the discrepancy is smaller compared to the previous two peaks, as the top bubble collapses without jetting, and the corresponding part of the pressure wave remains unaffected by the bottom bubble. Nevertheless, as the emitted pressure waves of both bubbles add-up, the large error caused by the bottom bubble is clearly visible.

Finally, Fig. 17e shows the pressure profile after collapse for the larger inter-bubble distance of 𝐷= 840 μm. Again, three pressure peaks are observed. The key difference is that the pressure emitted from the top bubble is significantly overpredicted by the spherical models — even in the third pressure peak at 𝑥≈1400 μm. This discrepancy arises because the top bubble jets in the DNS simulation, causing spherical models to overestimate the internal pressure. These results emphasize that coupled-spherical-bubble models can yield accurate predictions, but only when the bubbles remain compact (i.e., jet formation is suppressed). If the bubbles are far enough apart, and flattening does not occur, the pressure field around the bubbles is extremely accurate up until bubble collapse.

5. Discussion

In this study, spherical bubble models were examined in detail and compared against direct numerical simulations. As expected, these simplified models perform extremely well for low-viscosity liquids (e.g., water). Regarding the Mach number of the bubble wall, the Keller–Miksis equation is accurate until Ma ≈0.2, in this case the error in the simulations remains minimal. While the Gilmore model remains closer to the DNS results even for Ma approaching 1, as the RMS error stays under 5%. However, in very strong collapses, the interior of the bubble can become inhomogeneous, leading to small discrepancies between the Gilmore model and the DNS. As a result, the error in the minimal bubble radius can reach 20% during collapse. The simulations even match experimental data as demonstrated by a case where the maximum Mach number reaches Ma = 0.23. The minimum radius following bubble collapse cannot be determined from the experiments and error calculations are omitted; however, this case validates the spherical models and DNS simulations in water as the overall 𝑅(𝑡) curve is reproduced accurately. In high-viscosity liquids, the fundamental assumptions of the spherical models break down: the flow around the bubble is no longer potential, and liquid viscosity must be accounted for. Both the Keller–Miksis and Gilmore equations fail under these conditions. Our results indicate that large Reynolds numbers, i.e., Re = 𝜌𝑅̇𝑅∕𝜇≫1, are necessary for the accuracy of spherical models. Otherwise, even for weak bubble collapses, the error is significant.

The incompressible (with finite propagation speed), quasi-acoustic and Kirkwood–Bethe pressure emission models investigated are highly accurate for describing the pressure emissions of bubbles with limited compression. In cases of low bubble compression, pressure waves are emitted from the collapsing bubble without the formation of steep shock fronts. As the compression increases, the bubble collapse results in shock wave emissions. In these scenarios, the incompressible and quasi-acoustic models fail, as they assume a constant propagation velocity along the outgoing characteristics, which does not account for the dynamics of shock wave formation. On the other hand, the Kirkwood–Bethe hypothesis remains highly accurate in capturing shock wave emission. However, these pressure emission models do not incorporate the effects of viscosity. In high-viscosity liquids, this leads to inaccuracies, as viscosity causes significant dissipation around the shock front. As a result, these models underpredict the shock wave decay, especially in highly viscous liquids where the dissipation effects are more significant.

Spherical bubble models are generally highly accurate for cases involving low viscosity and limited Mach numbers, but their accuracy depends on the assumption that the bubbles remain spherical. In multi-bubble systems, this assumption is often violated due to the inhomogeneous pressure field around the bubbles, which can lead to significant inaccuracies. While coupling spherical models together may provide good predictions for radius-time curves up until collapse, the pressure field becomes distorted due to bubble interactions when the bubbles are close enough for flattening to occur. However, as the bubbles collapse, even small asymmetries in the pressure field can lead to bubble jetting, which reduces the compression of the bubbles. This effect causes coupled-spherical-bubble models to significantly overpredict the bubble pressure, leading to incorrect predictions for the pressure field and emitted wave amplitudes. These individual errors can accumulate in large bubble clusters, significantly altering the overall dynamics. Interestingly, the error in the bubble radius in the bubble pair is significantly smaller. For widely separated bubbles, it remains below 1%, whereas the error in the pressure exceeds 20%. Since the internal pressure is particularly important for chemical reactions, this discrepancy may be unacceptable in many scenarios, even though the radial dynamics would appear correct.

When jet suppression occurs during collapse, that is, when the bubble collapses without forming a liquid jet, spherical models remain more accurate for predicting the internal pressure. However, they still slightly overestimate the pressure, as the shape is rarely perfectly spherical, and some degree of surface distortion typically remains. In the context of cluster simulations, accurate modeling of pressure emission is crucial due to the numerous couplings between bubbles. When bubbles begin to jet, the pressure emissions become inaccurate, leading to a loss of overall simulation accuracy. Therefore, coupledspherical-bubble models should only be trusted when jet formation is suppressed. In this case, however, they offer a good balance between computational efficiency and predictive accuracy. That said, achieving jet suppression is not trivial and cannot be reliably enforced by simply adjusting the excitation parameters. Instead, it will require careful structuring of the bubble cluster geometry. The aforementioned limitations and overprediction of internal bubble pressure are particularly important for applications in sonochemistry, where accurate values for the internal bubble pressure are crucial to ensure reliable chemical calculations. As spherical models alone are insufficient for these cases due to their failure to account for jetting, future models should focus on predicting bubble jetting based on the pressure field around the bubble and consider its effects. This would improve the accuracy of coupled-spherical-bubble simulations.

This study has not investigated multi-bubble configurations in viscous liquids. However, single-bubble simulations indicate that both the radial dynamics and the pressure emissions become inaccurate even at moderately strong collapses. This suggests that coupled-sphericalbubble models are not sufficiently accurate for simulating multi-bubble systems at higher viscosities. Since the single-bubble dynamics already contain errors under such conditions, the resulting pressure fields will also be inaccurate. Because pressure emissions are the basis of coupling in these models, the overall simulation accuracy is expected to deteriorate even further in multi-bubble scenarios.

Currently, the main limitation of this study is the consideration of only simple multi-bubble layouts, where two bubbles lie along a symmetry axis. Future work could extend the analysis to more complex three-dimensional configurations. Additionally, investigations of dense bubble clouds with many interacting bubbles are needed, where the dissipation of the ultrasonic pulse may play a significant role. This study has also not considered the effects of the gas equation of state. At large collapses, the ideal gas assumption is not valid anymore, and the finite size of gas molecules (van der Waals hardcore radius) should be included. Furthermore, bulk viscosity, which is relevant for compressible flows and acts as further dissipation at the bubble surface, was

Ultrasonics Sonochemistry 123 (2025) 107651

18

D. Nagy and F. Hegedűs

neglected. In the future, modeling could be refined by incorporating these effects to improve the overall accuracy of the models.

6. Conclusion

Spherical bubble models provide accurate predictions for low-viscosity liquids at moderate Mach numbers; that is, the Keller– Miksis and Gilmore equations are extremely precise as long as Ma < 0.2. For Mach numbers approaching unity both models lose accuracy; however, the Gilmore model remains significantly closer to DNS results. Furthermore, their accuracy requires large Reynolds numbers, and these models will fail in high-viscosity liquids. For pressure emissions, incompressible and quasi-acoustic models are adequate for weak bubble collapses and can accurately track outward propagating pressure waves, while only the Kirkwood–Bethe hypothesis remains valid when shock waves form following a strong bubble collapse. In multi-bubble systems, spherical models reproduce radial bubble dynamics accurately with minimal error, but fail to capture the reduced pressure due to the non-spherical collapse, leading to a systematic overprediction of internal gas pressures and emitted pressure wave amplitudes. When bubble jetting is absent, coupled-spherical-bubble models are more accurate predicting the internal gas pressure, thus remain a computationally efficient alternative, although jet suppression requires careful design.

CRediT authorship contribution statement

Dániel Nagy: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Resources, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Ferenc Hegedűs: Writing – review & editing, Supervision, Project administration, Funding acquisition, Conceptualization.

Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Acknowledgments

The authors gratefully acknowledge Alexander Bußmann and Dr.- Ing. Stefan Adami (Technical University of Munich) for their valuable support with the ALPACA multiphase flow solver and assistance with software-related questions. Project No. 2025-2.1.2-EKÖP-KDP-202500005 has been implemented with the support provided by the Ministry of Culture and Innovation of Hungary, from the National Research, Development and Innovation Fund, financed under the EKÖP_KDP-251-BME-6 funding scheme. The authors acknowledge the financial support of the Hungarian National Research, Development and Innovation Office via NKFIH grant OTKA FK142376. Project no. TKP-6-6/PALY2021 has been implemented with the support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development and Innovation Fund, financed under the TKP2021-NVA funding scheme.

Appendix A. NASG parameters

A.1. Water

The NASG parameters for water are readily available in the literature [64,96,111]. The parameters used are the polytropic exponent 𝑛𝐿= 1.11, co-volume 𝑏𝐿= 6.8 × 10−4 m3∕kg, pressure constant 𝐵𝐿= 645.094 MPa and energy constant 𝑞𝐿= −1.178 MJ∕kg. These parameters are slightly modified from Denner and Schenke [64] to set the far field speed of sound to 𝑐𝐿= 1496 m∕s.

A.2. Glycerol

The reference temperature 𝑇ref = 298.15 K, pressure 𝑝0 = 0.1 MPa and density 𝜌0 = 1264.4 kg∕m3 are chosen. The parameters of the NASG equation of state is set based on the method of Le Métayer and Saurel [96], with the use tabulated data for the temperature, pressure and density relation (𝑇𝑖, 𝑝𝑖 and 𝜌𝑖) with 𝑖= 1 … 𝑁 where 𝑁 is the number of measurement points. Le Métayer and Saurel [96] uses an experimental ℎ𝑙(𝑇) curve as well; however, this is not readily available for glycerol. Instead, the liquid enthalpy ℎ𝑙= −7.2678 MJ∕kg is taken at the reference temperature based on [130]. This choice limits the accuracy of the equation of state to temperatures near the reference. Furthermore, the specific heat capacity of glycerol at the reference temperature at constant volume 𝐶𝑣= 2080 J∕kgK and at constant pressure 𝐶𝑝= 2394 J∕kgK are adopted from [131,132]. A set of measured values for (𝑇𝑖, 𝑝𝑖 and 𝜌𝑖) with 𝑖= 1 … 𝑁, 𝑁= 60 are taken from Ahmadi et al. [131] in the temperature range 𝑇∈[298, 423]K. The co-volume 𝑏 is calculated as [96]


$$
𝑏= 𝑣−(𝐶𝑝−𝐶𝑣)
$$

( 𝑇

𝑝

) , (A.1)

where 𝑣 and (𝑇∕𝑝) are averages from the measurements [96]:

𝑣= 1

𝑁


$$
𝑁 ∑
$$

𝑖=1

1 𝜌𝑖 and

( 𝑇

𝑝

) = 1

𝑁


$$
𝑁 ∑
$$

𝑖=1

𝑇𝑖 𝑝𝑖+ 𝐵. (A.2)

The pressure constant 𝐵 is found by solving the equation


$$
𝑝0 + 𝐵−
1 −
𝐶𝑝−𝐶𝑣
$$

𝐶𝑝

) 𝜌0𝑐2(1 −𝑏𝜌0) = 0. (A.3)

The energy constant is [96]


$$
𝑞= ℎ𝑙−𝐶𝑝𝑇−𝑏𝑝. (A.4)
$$

The ratio of specific heats is 𝛾= 𝐶𝑝∕𝐶𝑣. The calculated parameters are summarized in Table A.4. The NASG EoS is compared against measured values in Fig. A.18, the comparison shows that the speed of sound in the fitted EoS is less accurate for high temperatures.

A.3. Sulfuric acid

The liquid enthalpy ℎ𝑙= −8.2626 MJ∕kg is taken at the reference temperature 𝑇ref = 298.15 K [133]. The specific heat capacity is 𝐶𝑝= 1464.96 J∕kgK [134]. The reference speed of sound 𝑐ref = 1470 m∕s, reference density 𝜌ref = 1800 kg∕m3 and viscosity 𝜇= 0.025 Pas are adopted from [135]. Due to the lack of literature data on the density of sulfuric acid as a function of temperature and pressure the parameters cannot be fitted. Therefore, the parameters are assumed as 𝛾= 1.11, consistent with typical values for water, and 𝑏= 0.8∕𝜌ref, where the factor 0.8 is chosen to ensure 𝑏< 1∕𝜌ref, as required by the model, but remains an arbitrary estimate in the absence of better data. The pressure constant 𝐵 is fitted to match the speed of sound in the far field:

𝑐=


$$
√
$$

𝛾 𝑝0 + 𝐵 𝜌ref(1 −𝑏𝜌ref) . (A.5)

The energy constant is set based on Eq. (A.4) with 𝑇ref and 𝑝ref instead of measurement averages. The NASG parameters of sulfuric acid are summarized in Table A.4.

Appendix B. Details on bubble pair simulations

The coupled-spherical-bubble models are simplified for the case 𝑁= 2 with bubbles placed along a vertical symmetry axis, where the bubble center positions correspond to the configuration shown in Fig. 10a:

(𝑥1, 𝑦1) = (0, 0), and (𝑥2, 𝑦2) = (0, 𝐷), (B.1)

Ultrasonics Sonochemistry 123 (2025) 107651

19

D. Nagy and F. Hegedűs

Table A.4 NASG parameters for liquids. The column water+ denotes the parameters of the stiffened gas EoS for water, a simplified version of the NASG EoS. Liquid Water+ Water Glycerol Sulfuric acid Based on [78,109] [64,110] Present study Present study Reference density 𝜌ref∕(kg∕m3) 1000 1000 1264 1800 Reference speed of sound 𝑐ref∕(m∕s) 1496 1496 1920 1470 Reference pressure 𝑝ref∕MPa 0.1 0.1 0.1 0.1 Co-volume 𝑏∕(m3∕kg) 0 6.8 ⋅10−4 6.573 ⋅10−4 4.444 ⋅10−4 Pressure constant 𝐵∕MPa 508.5 645.1 683.7 700.7 Energy constant 𝑞∕(MJ∕kg) 0 −1.178 −8.148 −8.699 Ratio of specific heats 𝛾 4.4 1.110 1.151 1.110 Specific heat capacity 𝐶𝑣∕(J∕(kgK)) 3552 3552 2080 1320 Shear viscosity 𝜇∕Pas 0.001 0.001 1.468 0.025

Fig. A.18. Fitting of the Noble-Abel Stiffened Gas Equation of State for glycerol (continuous line) on the measurement data from [131] (points). Left: Pressure as function of density at various temperatures. Right: speed of sound as a function of pressure at various temperatures.

where 𝐷 is the distance between the bubble centers. To reduce the parameter space, the equilibrium radii of both bubbles are set equal, such that both bubbles have the same initial and equilibrium radii, i.e., 𝑅0,𝑖 = 𝑅𝐸 = 20 μm for 𝑖= 1, 2. The simulation parameters are summarized in Table B.5. The equations for the incompressible (Models 1a–1c) and the quasi-acoustic (Model 2) pressure emissions are simplified for a bubble pair below.

B.1. Equations for Model 1a (no-delay)

This model is based on the Keller–Miksis equation coupled with the incompressible pressure emission with no-delay. That is, the pressure emission from one bubble reaches the other bubble instantaneously; while, both bubbles excited in the same phase by the ultrasonic pulse. The Keller–Miksis equation for a bubble pair without delay is

( 1 − ̇𝑅1 𝑐𝐿

) 𝑅1 ̈𝑅1 + ( 1 − ̇𝑅1 3𝑐𝐿

) 3 2 ̇𝑅2 1

= ( 1 + ̇𝑅1 𝑐𝐿 + 𝑅1

𝑐𝐿

d d𝑡


$$
) 𝑝𝐿,1(𝑅1, 𝑡) −𝑝∞(0, 𝑦1, 𝑡) −𝑝B,1(𝑡)
$$

𝜌𝐿 , (B.2)

( 1 − ̇𝑅2 𝑐𝐿

) 𝑅2 ̈𝑅2 + ( 1 − ̇𝑅2 3𝑐𝐿

) 3 2 ̇𝑅2 2

= ( 1 + ̇𝑅2 𝑐𝐿 + 𝑅2

𝑐𝐿

d d𝑡


$$
) 𝑝𝐿,2(𝑅2, 𝑡) −𝑝∞(0, 𝑦2, 𝑡) −𝑝B,2(𝑡)
$$

𝜌𝐿 . (B.3)

In the no-delay case, the instantaneous pressure contribution from one bubble to the other simplifies to

𝑝B,1(𝑡) = 𝜌𝐿

𝐷 (2 ̇𝑅2 2(𝑡)𝑅2(𝑡) + 𝑅2 2(𝑡) ̈𝑅2(𝑡)) , (B.4)

𝑝B,2(𝑡) = 𝜌𝐿

𝐷 (2 ̇𝑅2 1(𝑡)𝑅1(𝑡) + 𝑅2 1(𝑡) ̈𝑅1(𝑡)) , (B.5)

where 𝐷 denotes the inter-bubble distance. Since the ultrasonic pulse reaches both bubbles simultaneously, the far-field pressure depends

solely on time:


$$
𝑝∞(0, 𝑦1, 𝑡) = 𝑝∞(0, 𝑦2, 𝑡) = 𝑝0 −𝑝𝐴sin (2𝜋𝑓𝑡) . (B.6)
$$

It must be noted, that this configuration is equivalent to two bubbles aligned horizontally along the 𝑥-axis.

B.2. Equations for Model 1b (geometric delay)

The geometric time delay, that is due to the finite propagation speed of the ultrasonic pulse, is considered in this model. The pressure emission between bubbles is still assumed to be instantaneous. The spherical dynamics is described by Eqs. (B.2)–(B.3) and the pressure contributions by Eqs. (B.4)–(B.5). Since the propagation of the ultrasonic pulse is accounted for, the far-field pressure depends on the location of the bubbles:


$$
𝑝∞(0, 0, 𝑡) = 𝑝0 −𝑝𝐴sin (2𝜋𝑓𝑡) , (B.7)
$$


$$
𝑝∞(0, 𝐷, 𝑡) = 𝑝0 −𝑝𝐴sin (2𝜋𝑓(𝑡−𝜏geo)) , (B.8)
$$

where 𝐷 is the center-to-center distance between the bubbles and 𝜏geo = 𝐷∕𝑐𝐿 is the time required for the ultrasonic pulse to travel from the bottom to the top bubble.

B.3. Equations for Model 1c (both delays)

This model considers both the geometric and emission time delays. The spherical dynamics of the bubbles are described by Eqs. (B.2)–(B.3) and the far-field pressure by Eqs. (B.7)–(B.8). The pressure contribution from the neighboring bubble assuming finite propagation speed within the incompressible model is given by

𝑝B,1(𝑡) = 𝜌𝐿


$$
𝐷 (2 ̇𝑅2 2(𝑡−𝜏𝑒)𝑅2(𝑡−𝜏𝑒) + 𝑅2 2(𝑡−𝜏𝑒) ̈𝑅2(𝑡−𝜏𝑒)) , (B.9)
$$

𝑝B,2(𝑡) = 𝜌𝐿


$$
𝐷 (2 ̇𝑅2 1(𝑡−𝜏𝑒)𝑅1(𝑡−𝜏𝑒) + 𝑅2 1(𝑡−𝜏𝑒) ̈𝑅1(𝑡−𝜏𝑒)) , (B.10)
$$

Ultrasonics Sonochemistry 123 (2025) 107651

20

D. Nagy and F. Hegedűs

Table B.5 Parameters of the bubble pair simulations. Parameter Symbol Value Speed of sound (liquid) 𝑐𝐿 1496 m∕s Density (liquid) 𝜌𝐿 1000 kg∕m3 Viscosity (liquid) 𝜇𝐿 0.001 Pa ⋅s Surface tension coefficient 𝜎 0.0728 N∕m Polytropic coefficient (gas) 𝑛𝐺 1.4 Ambient pressure 𝑝0 0.1 MPa Vapor pressure 𝑝𝑉 0 Pressure amplitude 𝑝𝐴 0.15 MPa

where 𝜏𝑒= 𝐷∕𝑐𝐿 is the propagation time of the acoustic pressure emission from one bubble to the other.

B.4. Equations for Model 2 (quasi-acoustic)

The quasi-acoustic model is coupled with the Keller–Miksis equation, as given in Eqs. (B.2)–(B.3). For the bubble pair, the pressure contributions to the bubbles, given in Eq. (18), simplify to

𝑝B,1(𝑡) = 𝜌𝐿

𝐷

[


$$
𝑔2(𝑡−𝜏𝑒) −1
$$

2


$$
(𝜙2(𝑡−𝜏𝑒)
$$


$$
𝐷 + 𝑔2(𝑡−𝜏𝑒)
$$

𝑐𝐿

)2]

, (B.11)

𝑝B,2(𝑡) = 𝜌𝐿

𝐷

[


$$
𝑔1(𝑡−𝜏𝑒) −1
$$

2


$$
(𝜙1(𝑡−𝜏𝑒)
$$


$$
𝐷 + 𝑔1(𝑡−𝜏𝑒)
$$

𝑐𝐿

)2]

, (B.12)

where 𝜏𝑒= 𝐷∕𝑐𝐿 is the propagation time of the acoustic pressure from one bubble to the other. The far-field pressure is given by Eqs. (B.7)–(B.8).

Appendix C. Spherical standing wave

The governing equations of compressible, inviscid flow under spherical symmetry are

𝜕𝜌

𝜕𝑡+ 1

𝑟2 𝜕 𝜕𝑟(𝑟2𝜌𝑢) = 0, (C.1)

𝜕𝑢


$$
𝜕𝑡+ 𝑢⋅𝜕𝑢
$$


$$
𝜕𝑟= −1
$$

𝜌 𝜕𝑝 𝜕𝑟, (C.2)

where 𝑟 is the radial coordinate. Expanding Eq. (C.1) results in

𝜕𝜌

𝜕𝑡+ 1

𝑟2

( 2𝑟𝜌𝑢+ 𝑟2𝑢𝜕𝜌

𝜕𝑟+ 𝑟2𝜌𝜕𝑢

𝜕𝑟

) = 0. (C.3)

Given that 𝑢𝜕𝜌∕𝜕𝑟≪𝜌𝜕𝑢∕𝜕𝑟 for acoustic waves, and 𝜌≈𝜌0 as the variation in the density is small, the above equation simplifies to

𝜕𝜌

𝜕𝑡+ 2𝜌0𝑢

𝑟 + 𝜌0 𝜕𝑢 𝜕𝑟= 0. (C.4)

Assuming that the particle velocity is smaller than the speed of sound, i.e., 𝑢≪𝑐, neglecting 𝑢⋅𝜕𝑢∕𝜕𝑟≪𝜕𝑢∕𝜕𝑡 is justified in Eq. (C.2):

𝜕𝑢


$$
𝜕𝑡= −1
$$

𝜌0

𝜕𝑝 𝜕𝑟. (C.5)

After taking the partial derivative of Eq. (C.4) with respect to time, Eq. (C.5) can be inserted:

𝜕2𝜌


$$
𝜕𝑡2 −2
$$

𝑟 𝜕𝑝 𝜕𝑟−𝜕2𝑝

𝜕𝑟2 = 0. (C.6)

Multiplying the equation with 𝑐2 = 𝜕𝑝∕𝜕𝜌 results in,

𝜕2𝑝


$$
𝜕𝑡2 −2𝑐2
$$

𝑟 𝜕𝑝 𝜕𝑟−𝑐2 𝜕2𝑝

𝜕𝑟2 = 0, (C.7)

that is a partial differential equation for the pressure 𝑝(𝑡, 𝑟), namely, the acoustic wave equation in spherical symmetry. We assume that the pressure changes sinusoidally in time for all 𝑟, that is

𝑝(𝑡, 𝑟) = 𝑝0 −𝑝𝐴⋅𝑔(𝑟) ⋅sin(𝜔𝑡), (C.8)

where 𝑔(𝑟) is an unknown function and 𝜔= 2𝜋𝑡. For 𝑟= 0, a bubble in the center of the spherical standing wave should experience a sinusoidal

pressure change, with 𝑝𝐴 pressure amplitude; thus, 𝑔(0) = 1. Inserting Eq. (C.8) into the acoustic wave equation in Eq. (C.7) and simplifying results a second order ordinary differential equation for 𝑔(𝑟),

𝜔2𝑔(𝑟) + 2𝑐2

𝑟𝑔′(𝑟) + 𝑐2𝑔′′(𝑟) = 0. (C.9)

The above equation can be solved via the d’Alembert reduction:

𝑔(𝑟) = 𝑐


$$
𝜔⋅ sin ( 𝑟𝜔
$$

𝑐

)

𝑟 , (C.10)

which solution satisfies 𝑟(0) = 1 in the limit. The velocity field can then be calculated from Eq. (C.4), and the density field from Eq. (C.5):


$$
𝑝(𝑡, 𝑟) = 𝑝0 −𝑝𝐴𝑐
$$


$$
𝜔𝑟⋅sin(𝜔𝑡) ⋅sin ( 𝑟𝜔
$$

𝑐

) , (C.11)

𝑢(𝑡, 𝑟) = −𝑝𝐴cos(𝜔𝑡)

𝑟2𝜌0𝜔2

[ 𝑟𝜔cos ( 𝑟𝜔

𝑐


$$
) −𝑐sin ( 𝑟𝜔
$$

𝑐

)] , (C.12)


$$
𝜌(𝑡, 𝑟) = 𝜌0 −𝑝𝐴
$$


$$
𝑐𝑟𝜔⋅sin(𝜔𝑡) ⋅sin (𝑟𝜔
$$

𝑐

) . (C.13)

The initial conditions in the spherical ALPACA simulations come from setting 𝑡= 0 in Eqs. (C.11)–(C.13).


## References

[1] C. Kalmár, K. Klapcsik, F. Hegedűs, Relationship between the radial dynamics

and the chemical production of a harmonically driven spherical bubble, Ultrason. Sonochem. 64 (2020) 104989. [2] S. Sochard, A. Wilhelm, H. Delmas, Modelling of free radicals production in a

collapsing gas-vapour bubble, Ultrason. Sonochem. 4 (2) (1997) 77–84. [3] S. Cho, S.H. Yun, Structure and optical properties of perovskite-embedded

dual-phase microcrystals synthesized by sonochemistry, Commun. Chem. 3 (1) (2020) 1–7. [4] A. Al-Awamleh, F. Hegedűs, Sono-hydrogen: a theoretical investigation of its

energy intensity, Period. Polytech. Mech. Eng. 68 (3) (2024) 254–263. [5] F. Kubicsek, Á. Kozák, T. Turányi, I.G. Zsély, M. Papp, A. Al-Awamleh, F.

Hegedûs, Ammonia production by microbubbles: A theoretical analysis of achievable energy intensity, Ultrason. Sonochem. 106 (2024) 106876. [6] P. Adamou, E. Harkou, S. Hafeez, G. Manos, A. Villa, S. Al-Salem, A. Con-

stantinou, N. Dimitratos, Recent progress on sonochemical production for the synthesis of efficient photocatalysts and the impact of reactor design, Ultrason. Sonochem. (2023) 106610. [7] K. Okitsu, F. Cavalieri, S.K. Bhangu, E. Colombo, M. Ashokkumar, Sonochemical

Production of Nanomaterials, Springer, 2018. [8] Z. He, F. Hou, Y. Du, C. Dai, R. He, H. Ma, Accelerating maturation of Chinese

rice wine by using a 20 L scale multi-sweeping-frequency mode ultrasonic reactor and its mechanism exploration, Ultrason. Sonochem. (2025) 107229. [9] D. Kulaga, A.K. Drabczyk, P. Zareba, J. Jaskowska, J. Chrzan, K.E. Greber, K.

Ciura, D. Plazuk, E. Wielgus, Green synthesis of 1, 3, 5-triazine derivatives using a sonochemical protocol, Ultrason. Sonochem. (2024) 106951. [10] S. Chakma, V.S. Moholkar, Intensification of wastewater treatment using sono-

hybrid processes: an overview of mechanistic synergism, Indian Chem. Eng. 57 (3–4) (2015) 359–381. [11] J. González-García, V. Sáez, I. Tudela, M.I. Díez-Garcia, M. Deseada Es-

clapez, O. Louisnard, Sonochemical treatment of water polluted by chlorinated organocompounds. a review, Water 2 (1) (2010) 28–74. [12] N. Bremond, M. Arora, S.M. Dammer, D. Lohse, Interaction of cavitation bubbles

on a wall, Phys. Fluids 18 (12) (2006). [13] T. Mason, E. Joyce, S. Phull, J. Lorimer, Potential uses of ultrasound in

the biological decontamination of water, Ultrason. Sonochem. 10 (6) (2003) 319–323. [14] M. Sakr, M.M. Mohamed, M.A. Maraqa, M.A. Hamouda, A.A. Hassan, J. Ali,

J. Jung, A critical review of the recent developments in micro–nano bubbles applications for domestic and industrial wastewater treatment, Alex. Eng. J. 61 (8) (2022) 6591–6612.

Ultrasonics Sonochemistry 123 (2025) 107651

21

D. Nagy and F. Hegedűs

[15] M. Jia, M.U. Farid, J.A. Kharraz, N.M. Kumar, S.S. Chopra, A. Jang, J. Chew,

S.K. Khanal, G. Chen, A.K. An, Nanobubbles in water and wastewater treatment systems: small bubbles making a big difference, Water Res. (2023) 120613. [16] Z. Eren, Dual-frequency ultrasonic oxidation of cyanobacterial toxins (MC-LR

and MC-RR) at drinking water resources: Assessment of analytical methods and ultrasonic reactor configuration, Ultrason. Sonochem. 113 (2025) 107203. [17] H. Ren, Y. Quan, S. Liu, J. Hao, Effectiveness of ultrasound (US) and

slightly acidic electrolyzed water (SAEW) treatments for removing Listeria monocytogenes biofilms, Ultrason. Sonochem. 112 (2025) 107190. [18] Y. Sun, J. Cao, X. Wang, C. Zhang, J. Luo, Y. Zeng, C. Zhang, Q. Li, Y. Zhang,

W. Xu, et al., Hypoxia-adapted sono-chemodynamic treatment of orthotopic pancreatic carcinoma using copper metal–organic frameworks loaded with an ultrasound-induced free radical initiator, ACS Appl. Mater. Interfaces 13 (32) (2021) 38114–38126. [19] M. Parsa, M.H. Entezari, A. Meshkini, Sono-synthesis approach improves an-

ticancer activity of ZnO nanoparticles: reactive oxygen species depletion for killing human osteosarcoma cells, Nanomedicine 16 (8) (2021) 657–671. [20] B.S. Gerstman, C.R. Thompson, S.L. Jacques, M.E. Rogers, Laser induced bubble

formation in the retina, Lasers Surg. Med. 18 (1) (1996) 10–21. [21] A. Nakagawa, T. Kumabe, Y. Ogawa, T. Hirano, T. Kawaguchi, K. Ohtani, T.

Nakano, C. Sato, M. Yamada, T. Washio, et al., Pulsed laser-induced liquid jet: evolution from shock/bubble interaction to neurosurgical application, Shock Waves 27 (2017) 1–14. [22] T.G. Van Leeuwen, E.D. Jansen, A.J. Welch, C. Borst, Excimer laser induced

bubble: dimensions, theory, and implications for laser angioplasty, Lasers Surg. Med. 18 (4) (1996) 381–390. [23] S. Freidank, A. Vogel, N. Linz, Mechanisms of corneal intrastromal laser

dissection for refractive surgery: ultra-high-speed photographic investigation at up to 50 million frames per second, Biomed. Opt. Express 13 (5) (2022) 3056–3079. [24] A. Holzhey, S. Sonntag, J. Rendenbach, J.S. Ernesti, V. Kakkassery, S. Grisanti,

F. Reinholz, S. Freidank, A. Vogel, M. Ranjbar, Development of a noninvasive, laser-assisted experimental model of corneal endothelial cell loss, J. Vis. Exp. 158 (2020) e60542. [25] A. Vogel, N. Linz, S. Freidank, G. Paltauf, Femtosecond-laser-induced nanocav-

itation in water: Implications for optical breakdown threshold and cell surgery, Phys. Rev. Lett. 100 (3) (2008) 038102. [26] J.M. Rosselló, C.-D. Ohl, Bullet jet as a tool for soft matter piercing and

needle-free liquid injection, Biomed. Opt. Express 13 (10) (2022) 5202–5211. [27] Z. Heidary, C.-D. Ohl, A. Mojra, Numerical analysis of ultrasound-mediated

microbubble interactions in vascular systems: Effects on shear stress and vessel mechanics, Phys. Fluids 36 (8) (2024). [28] Y. Shen, L. Zhang, Y. Wu, W. Chen, The role of the bubble–bubble interaction

on radial pulsations of bubbles, Ultrason. Sonochem. 73 (2021) 105535. [29] M. Adama Maiga, O. Coutier-Delgosha, D. Buisine, A new cavitation model

based on bubble-bubble interactions, Phys. Fluids 30 (12) (2018). [30] M. Ida, T. Naoe, M. Futakawa, Suppression of cavitation inception by gas bubble

injection: A numerical study focusing on bubble-bubble interaction, Phys. Rev. E 76 (4) (2007) 046309. [31] K. Yasui, J. Lee, T. Tuziuti, A. Towata, T. Kozuka, Y. Iida, Influence of the

bubble-bubble interaction on destruction of encapsulated microbubbles under ultrasound, J. Acoust. Soc. Am. 126 (3) (2009) 973–982. [32] Y. Fan, H. Li, J. Zhu, W. Du, A simple model of bubble cluster dynamics in an

acoustic field, Ultrason. Sonochem. 64 (2020) 104790. [33] K. Yasui, Multibubble sonoluminescence from a theoretical perspective,

Molecules 26 (15) (2021) 4624. [34] D. Fuster, A review of models for bubble clusters in cavitating flows, Flow

Turbul. Combust. 102 (3) (2019) 497–536. [35] D. Fuster, J.-M. Conoir, T. Colonius, Effect of direct bubble-bubble interactions

on linear-wave propagation in bubbly liquids, Phys. Rev. E 90 (6) (2014) 063010. [36] H. Chen, Z. Lai, Z. Chen, Y. Li, The secondary Bjerknes force between two

oscillating bubbles in Kelvin-Voigt-type viscoelastic fluids driven by harmonic ultrasonic pressure, Ultrason. Sonochem. 52 (2019) 344–352. [37] M.S. Plesset, The dynamics of cavitation bubbles, 1949. [38] M.S. Plesset, A. Prosperetti, Bubble dynamics and cavitation, Annu. Rev. Fluid

Mech. 9 (1977) 145–185. [39] A. Prosperetti, A generalization of the Rayleigh-Plesset equation of bubble

dynamics, Phys. Fluids 25 (3) (1982) 409–410. [40] L. Fu, X.-X. Liang, S. Wang, S. Wang, P. Wang, Z. Zhang, J. Wang, A. Vogel, C.

Yao, Laser induced spherical bubble dynamics in partially confined geometry with acoustic feedback from container walls, Ultrason. Sonochem. 101 (2023) 106664. [41] O. Louisnard, A simple model of ultrasound propagation in a cavitating liquid.

Part I: Theory, nonlinear attenuation and traveling wave generation, Ultrason. Sonochem. 19 (1) (2012) 56–65. [42] O. Louisnard, A simple model of ultrasound propagation in a cavitating liquid.

Part II: Primary Bjerknes force and bubble structures, Ultrason. Sonochem. 19 (1) (2012) 66–76. [43] J.B. Keller, M. Miksis, Bubble oscillations of large amplitude, J. Acoust. Soc.

Am. 68 (2) (1980) 628–633.

[44] K. Klapcsik, B. Gyires-Tóth, J.M. Rosselló, F. Hegedűs, Position control of an

acoustic cavitation bubble by reinforcement learning, Ultrason. Sonochem. 115 (2025) 107290. [45] K.D. Hattori, T. Yamamoto, Stability analysis of the effect of harmonic waves on

the shape stability of acoustic cavitation bubbles, Ultrason. Sonochem. (2025) 107444. [46] F.R. Gilmore, The Growth or Collapse of a Spherical Bubble in a Viscous

Compressible Liquid, Vol. 26, California Institute of Technology Pasadena, CA, USA, 1952. [47] F. Denner, The Gilmore-NASG model to predict single-bubble cavitation in

compressible liquids, Ultrason. Sonochem. 70 (2021) 105307. [48] D.B. Preso, D. Fuster, A.B. Sieber, D. Obreschkow, M. Farhat, Vapor compression

and energy dissipation in a collapsing laser-induced bubble, Phys. Fluids 36 (3) (2024). [49] A. Zhang, S.-M. Li, P. Cui, S. Li, Y.-L. Liu, A unified theory for bubble dynamics,

Phys. Fluids 35 (3) (2023). [50] L. Trilling, The collapse and rebound of a gas bubble, J. Appl. Phys. 23 (1)

(1952) 14–17. [51] P. Coulombel, F. Denner, Modeling time-delayed acoustic interactions of

cavitation bubbles and bubble clusters, Phys. Fluids 36 (12) (2024). [52] J. Kirkwood, H. Bethe, Basic propagation theory, Off. Sci. Res. Dev. (1942)

588–595. [53] R. Cole, Underwater explosions, 1948. [54] F. Denner, The Kirkwood–Bethe hypothesis for bubble dynamics, cavitation, and

underwater explosions, Phys. Fluids 36 (5) (2024). [55] S.H. Bryngelson, K. Schmidmayer, V. Coralic, J.C. Meng, K. Maeda, T. Colonius,

MFC: An open-source high-order multi-component, multi-phase, and multi-scale compressible flow solver, Comput. Phys. Comm. 266 (2021) 107396. [56] K. Vokurka, Comparison of Rayleigh’s, Herring’s, and Gilmore’s models of gas

bubbles, Acta Acust. 59 (3) (1986) 214–219. [57] K. Johansen, J.H. Song, P. Prentice, Validity of the keller-miksis equation for

‘‘non-stable’’ cavitation and the acoustic emissions generated, in: 2017 IEEE International Ultrasonics Symposium, IUS, IEEE, 2017, pp. 1–4. [58] R. Varga, R. Mettin, High dimensional parameter fitting of the Keller–Miksis

equation on an experimentally observed dual-frequency driven acoustic bubble, Period. Polytech. Mech. Eng. 63 (4) (2019) 326–335. [59] C.C. Church, Prediction of rectified diffusion during nonlinear bubble pulsations

at biomedical frequencies, J. Acoust. Soc. Am. 83 (6) (1988) 2210–2217. [60] S. Zhu, P. Zhong, Shock wave–inertial microbubble interaction: A theoretical

study based on the Gilmore formulation for bubble dynamics, J. Acoust. Soc. Am. 106 (5) (1999) 3024–3033. [61] W. Lauterborn, T. Kurz, R. Mettin, C. Ohl, Experimental and theoretical bubble

dynamics, Adv. Chem. Phys. 110 (1999) 295–380. [62] A. Vogel, S. Busch, U. Parlitz, Shock wave emission and cavitation bubble

generation by picosecond and nanosecond optical breakdown in water, J. Acoust. Soc. Am. 100 (1) (1996) 148–165. [63] X.-X. Liang, N. Linz, S. Freidank, G. Paltauf, A. Vogel, Comprehensive analysis

of spherical bubble oscillations and shock wave emission in laser-induced cavitation, J. Fluid Mech. 940 (2022) A5. [64] F. Denner, S. Schenke, Modeling acoustic emissions and shock formation of

cavitation bubbles, Phys. Fluids 35 (1) (2023). [65] H. Sternberg, W. Walker, Calculated flow and energy distribution following

underwater detonation of a pentolite sphere, Phys. Fluids 14 (9) (1971) 1869–1878. [66] G.H. Miller, T.J. Ahrens, Shock-wave viscosity measurement, Rev. Modern Phys.

63 (4) (1991) 919. [67] Y. Shen, K. Yasui, T. Zhu, M. Ashokkumar, A model for the effect of bulk

liquid viscosity on cavitation bubble dynamics, Phys. Chem. Chem. Phys. 19 (31) (2017) 20635–20640. [68] W. Lauterborn, C. Lechner, M. Koch, R. Mettin, Bubble models and real bubbles:

Rayleigh and energy-deposit cases in a Tait-compressible liquid, IMA J. Appl. Math. 83 (4) (2018) 556–589. [69] D. Fuster, S. Popinet, An all-Mach method for the simulation of bubble dynamics

problems in the presence of surface tension, J. Comput. Phys. 374 (2018) 752–768. [70] S.J. Shaw, Translation and oscillation of a bubble under axisymmetric

deformation, Phys. Fluids 18 (7) (2006) 072104. [71] S.J. Shaw, Shape distortion of an acoustically forced gas microbubble, Int. J.

Multiph. Flow 184 (2025) 105074. [72] S.J. Shaw, Controlled movement of a shape deforming bubble, Phys. Fluids 37

(7) (2025). [73] C. Lechner, W. Lauterborn, M. Koch, R. Mettin, Jet formation from bubbles

near a solid boundary in a compressible liquid: Numerical study of distance dependence, Phys. Rev. Fluids 5 (9) (2020) 093604. [74] C. Lechner, M. Koch, W. Lauterborn, R. Mettin, Fast jets from bubbles close

to solid objects: examples from pillars in water to infinite planes in different liquids, Tech. Mech. 43 (2023) 21–37. [75] X. Yang, J. Liang, Y. Qiao, Dynamics of three bubbles in a line driven by

ultrasound, J. Appl. Phys. 137 (24) (2025).

Ultrasonics Sonochemistry 123 (2025) 107651

22

D. Nagy and F. Hegedűs

[76] S. Terasaki, A. Kiyama, D. Kang, Y. Tomita, K. Sato, On the interaction of two

cavitation bubbles produced at different times: A jet from the primary bubble, Phys. Fluids 36 (1) (2024). [77] J. Shen, J. Ying, W. Liu, S. Zhang, Y. Zhang, The evolution of the bubble

collapse morphology between two cylinders within a confined space, Phys. Fluids 36 (10) (2024). [78] D. Nagy, F. Hegedűs, Suppressing the jet formation in a bubble pair excited

with an ultrasonic pulse, Ultrason. Sonochem. (2025) 107349. [79] K. Maeda, T. Colonius, Bubble cloud dynamics in an ultrasound field, J. Fluid

Mech. 862 (2019) 1105–1134. [80] B. Yang, M. Ye, S. Ren, L. Liu, Optimizing cavitation performance in bath-type

sonoreactor by numerical simulation combining the Keller-Miksis equation and nonlinear Helmholtz equation, Chem. Eng. J. (2025) 161696. [81] B. Sajjadi, A.A.A. Raman, S. Ibrahim, Influence of ultrasound power on acoustic

streaming and micro-bubbles formations in a low frequency sono-reactor: Mathematical and 3D computational simulation, Ultrason. Sonochem. 24 (2015) 193–203. [82] Y. Fan, H. Li, D. Fuster, Time-delayed interactions on acoustically driven bubbly

screens, J. Acoust. Soc. Am. 150 (6) (2021) 4219–4231. [83] Y. Wang, D. Chen, P. Wu, J. Li, Analysis of multi-bubble pulsations by the finite

element method and bubble dynamics equations, Phys. Fluids 36 (4) (2024). [84] S.-W. Ohl, J.M. Rosselló, D. Fuster, C.-D. Ohl, Finite amplitude wave

propagation through bubbly fluids, Int. J. Multiph. Flow 176 (2024) 104826. [85] S. Shaw, P. Spelt, Shock emission from collapsing gas bubbles, J. Fluid Mech.

646 (2010) 363–373. [86] H. Reese, U.J. Gutiérrez-Hernández, P. Pfeiffer, P.A. Quinto-Su, C.-D. Ohl,

Rayleigh wave induced cavitation bubble structures, Int. J. Multiph. Flow 184 (2025) 105114. [87] D. Fuster, T. Colonius, Modelling bubble clusters in compressible liquids, J.

Fluid Mech. 688 (2011) 352–389. [88] N. Hoppe, J.M. Winter, S. Adami, N.A. Adams, ALPACA - a level-set based

sharp-interface multiresolution solver for conservation laws, Comput. Phys. Comm. (ISSN: 0010-4655) 272 (2022) 108246. [89] N. Hoppe, S. Adami, N.A. Adams, A parallel modular computing environ-

ment for three-dimensional multiresolution simulations of compressible flows, Comput. Methods Appl. Mech. Engrg. 391 (2022) 114486. [90] L. Fu, X.Y. Hu, N.A. Adams, A family of high-order targeted ENO schemes for

compressible-fluid simulations, J. Comput. Phys. 305 (2016) 333–359. [91] J. Kaiser, S. Adami, I. Akhatov, N. Adams, A semi-implicit conservative sharp-

interface method for liquid-solid phase transition, Int. J. Heat Mass Transfer 155 (2020) 119800. [92] J. Kaiser, J. Winter, S. Adami, N. Adams, Investigation of interface deforma-

tion dynamics during high-Weber number cylindrical droplet breakup, Int. J. Multiph. Flow 132 (2020) 103409. [93] D. Nagy, S. Adami, F. Hegedűs, Direct numerical simulation of spherical and

non-spherical bubble dynamics using the ALPACA compressible multiphase flow solver, Int. J. Multiph. Flow (2025) 105287. [94] S. Shaw, The stability of a bubble in a weakly viscous liquid subject to an

acoustic traveling wave, Phys. Fluids 21 (2) (2009). [95] X. Huang, P.-B. Liu, G.-Y. Niu, H.-B. Hu, Experimental study on the translation

behavior of an in-situ bubble pair in the ultrasonic field, Ultrason. Sonochem. 112 (2025) 107188. [96] O. Le Métayer, R. Saurel, The Noble-Abel stiffened-gas equation of state, Phys.

Fluids 28 (4) (2016). [97] M. Radulescu, On the Noble-Abel stiffened-gas equation of state, Phys. Fluids

31 (11) (2019). [98] P.L. Roe, Approximate Riemann solvers, parameter vectors, and difference

schemes, J. Comput. Phys. 43 (2) (1981) 357–372. [99] G.-S. Jiang, C.-W. Shu, Efficient implementation of weighted ENO schemes, J.

Comput. Phys. 126 (1) (1996) 202–228. [100] X.Y. Hu, B. Khoo, N.A. Adams, F. Huang, A conservative interface method for

compressible flows, J. Comput. Phys. 219 (2) (2006) 553–578. [101] V. Bogdanov, F.S. Schranner, J.M. Winter, S. Adami, N.A. Adams, A level-set-

based sharp-interface method for moving contact lines, J. Comput. Phys. 467 (2022) 111445. [102] N. Fleischmann, J.M. Winter, S. Adami, N.A. Adams, High-order modeling of

interface interactions using level sets, GAMM-Mitt. 45 (2) (2022) e202200012. [103] A. Bußmann, F. Riahi, B. Gökce, S. Adami, S. Barcikowski, N.A. Adams,

Investigation of cavitation bubble dynamics near a solid wall by high-resolution numerical simulation, Phys. Fluids 35 (1) (2023). [104] J. Mur, A. Bußmann, T. Paula, S. Adami, N.A. Adams, C.-D. Ohl, et al., Micro-

jet formation induced by the interaction of a spherical and toroidal cavitation bubble, Ultrason. Sonochem. 112 (2025) 107185. [105] B. Biller, N. Hoppe, S. Adami, N.A. Adams, Jetting mechanisms in bubble-pair

interactions, Phys. Fluids 34 (7) (2022). [106] Y. Fan, A. Bußmann, F. Reuter, H. Bao, S. Adami, J.M. Gordillo, N. Adams, C.-D.

Ohl, Amplification of supersonic microjets by resonant inertial cavitation-bubble pair, Phys. Rev. Lett. 132 (10) (2024) 104004.

[107] A. Bußmann, S. Adami, N.A. Adams, A systematic calibration procedure for

bubble dynamics for laser ablation in liquids, Nanobubble Prod. 7 (2022) 72. [108] F.H. Harlow, A.A. Amsden, Fluid dynamics. a LASL monograph, Tech. rep., Los

Alamos National Lab.(LANL), Los Alamos, NM (United States), 1971. [109] J.W. Kaiser, N. Hoppe, S. Adami, N.A. Adams, An adaptive local time-stepping

scheme for multiresolution simulations of hyperbolic conservation laws, J. Comput. Phys. 4 (2019) 100038. [110] A. Chiapolino, R. Saurel, Extended Noble–Abel stiffened-gas equation of state

for sub-and-supercritical liquid-gas systems far from the critical point, Fluids 3 (3) (2018) 48. [111] F. Denner, S. Schenke, APECSS: A software library for cavitation bubble

dynamics and acoustic emissions, J. Open Source Softw. 8 (86) (2023) 5435. [112] A.A. Doinikov, Translational motion of two interacting bubbles in a strong

acoustic field, Phys. Rev. E 64 (2) (2001) 026301. [113] M. Arora, C.-D. Ohl, D. Lohse, Effect of nuclei concentration on cavitation

cluster dynamics, J. Acoust. Soc. Am. 121 (6) (2007) 3432–3436. [114] K. Pham, J.-F. Mercier, D. Fuster, J.-J. Marigo, A. Maurel, Scattering of acoustic

waves by a nonlinear resonant bubbly screen, J. Fluid Mech. 906 (2021) A19. [115] X. Shen, P. Wu, W. Lin, Numerical simulation of cavitation threshold in

water and viscoelastic medium based on bubble cluster dynamics, Ultrason. Sonochem. (2025) 107414. [116] M. Rodríguez, F. Blesa, R. Barrio, Opencl parallel integration of ordinary

differential equations: Applications in computational dynamics, Comput. Phys. Comm. 192 (2015) 228–236. [117] K. Klapcsik, GPU accelerated numerical investigation of the spherical stability

of an acoustic cavitation bubble excited by dual-frequency, Ultrason. Sonochem. 77 (2021) 105684. [118] D. Nagy, L. Plavecz, F. Hegedűs, The art of solving a large number of non-

stiff, low-dimensional ordinary differential equation systems on GPUs and CPUs, Commun. Nonlinear Sci. Numer. Simul. 112 (2022) 106521. [119] K. Klapcsik, Dataset of exponential growth rate values corresponding non-

spherical bubble oscillations under dual-frequency acoustic irradiation, Data Brief 40 (2022) 107810. [120] H. Haghi, M.C. Kolios, The role of primary and secondary delays in the

effective resonance frequency of acoustically interacting microbubbles, Ultrason. Sonochem. 86 (2022) 106033. [121] J.A. Ketterling, R.E. Apfel, Experimental validation of the dissociation hy-

pothesis for single bubble sonoluminescence, Phys. Rev. Lett. 81 (22) (1998) 4991. [122] F. Gaitan, Sonoluminescence and bubble stability, Phys. World 12 (3) (1999)

20. [123] D. Xia, J. Wu, K. Su, Influence of micron-sized air bubbles on sonochemical

reactions in aqueous solutions exposed to combined ultrasonic irradiation and aeration processes, J. Env. Chem. Eng. 10 (6) (2022) 108685. [124] D. Qin, S. Lei, B. Zhang, Y. Liu, J. Tian, X. Ji, H. Yang, Influence of interactions

between bubbles on physico-chemical effects of acoustic cavitation, Ultrason. Sonochem. 104 (2024) 106808. [125] K. Yasui, K. Yasui, Acoustic Cavitation, Springer, 2018. [126] X. Zheng, X. Wang, Y. Zhang, A single oscillating bubble in liquids with high

Mach number, Ultrason. Sonochem. 85 (2022) 105985. [127] C. Wang, H. Yan, R. Zhang, F. Chen, F. Liu, Numerical study of laser-induced

cavitation bubble with consideration of chemical reactions, Ultrason. Sonochem. 109 (2024) 107007. [128] J. Luo, G. Fu, W. Xu, Y. Zhai, L. Bai, J. Li, T. Qu, Experimental study

on attenuation effect of liquid viscosity on shockwaves of cavitation bubbles collapse, Ultrason. Sonochem. 111 (2024) 107063. [129] M. Postema, P. Marmottant, C.T. Lancée, S. Hilgenfeldt, N. De Jong, Ultrasound-

induced microbubble coalescence, Ultrasound Med. Biol. 30 (10) (2004) 1337–1344. [130] S.P. Verevkin, D.H. Zaitsau, V.N. Emel’yanenko, A.A. Zhabina, Thermodynamic

properties of glycerol: Experimental and theoretical study, Fluid Phase Equilib. 397 (2015) 87–94. [131] P. Ahmadi, A. Chapoy, R. Burgass, An investigation on the thermophysical

properties of glycerol, J. Chem. Thermodyn. 178 (2023) 106975. [132] M. Zábransk`y, Z. Kolská, V. Ruzicka, E.S. Domalski, Heat capacity of liquids:

critical review and recommended values. Supplement II, J. Phys. Chem. Ref. Data 39 (1) (2010). [133] F.J. Zeleznik, Thermodynamic properties of the aqueous sulfuric acid system to

350 K, J. Phys. Chem. Ref. Data 20 (6) (1991) 1157–1200. [134] J. Kunzler, W. Giauque, Aqueous sulfuric acid. Heat capacity. Partial specific

heat content of water at 25 and -20◦, J. Am. Chem. Soc. 74 (14) (1952) 3472–3476. [135] S.D. Hopkins, S.J. Putterman, B.A. Kappus, K.S. Suslick, C.G. Camara, Dynamics

of a sonoluminescing bubble in sulfuric acid, Phys. Rev. Lett. 95 (25) (2005) 254301.

Ultrasonics Sonochemistry 123 (2025) 107651

23

