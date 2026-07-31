Contents lists available at ScienceDirect


journal homepage: www.elsevier.com/locate/jcp


# Physically consistent formulations of split convective terms for turbulent compressible multi-component flows


## Ye Wang a,∗, Armin Wehrfritz b, Evatt R. Hawkes a

a School of Mechanical and Manufacturing Engineering, The University of New South Wales, Sydney, Australia b Department of Mechanical and Materials Engineering, University of Turku, Finland


### a r t i c l e  i n f o

Keywords: Split convective forms High-order finite-difference schemes Compressible multi-component flows Species mass fraction Temperature equilibrium Pressure equilibrium

a b s t r a c t

We analyse the properties and characteristics of kinetic-energy-preserving, entropy-preserving, and pressure-equilibrium-preserving split convective forms for compressible multi-component flows. The results show that such schemes offer improved pressure-equilibrium-preserving properties and numerical stability compared to most other existing schemes, but also that the preservation of pressure equilibrium is not guaranteed for flows with varying specific heats. Furthermore, for the convective terms in species mass fraction transport equations, some split forms may fail to preserve key physical properties discretely. We construct a formulation for the species convective terms that consistently maintains these key physical properties, including species mass conservation, uniform mass fraction preservation, and temperature-equilibrium preservation. The capability of the proposed scheme in maintaining these properties is demonstrated analytically and tested in one-dimensional advection problems. Last, the proposed scheme is compared with schemes that do not satisfy these properties in under-resolved simulations of a modified inviscid Taylor–Green vortex flow. The results show improved performance of the proposed scheme and highlight the importance of a convective scheme for the species mass fractions to be able to consistently preserve these physical properties in a discrete sense.

1.  Introduction

High-order and non-dissipative numerical schemes are advantageous for modelling compressible turbulence flows in direct numerical simulations and large-eddy simulations [1–4], due to their nature of small truncation errors [5,6] and low numerical diffusion [7] that enables high-fidelity turbulence simulations with reduced computational grid points. However, it is well known that such schemes lead to significant aliasing errors and may cause numerical instabilities [6,8], particularly for flows with high Reynolds numbers.

In high-Reynolds number turbulence simulations, aliasing errors arise when nonlinear convective terms (derivatives of multivariable products) are evaluated on a discrete grid, due to the under-resolved content above the Nyquist limit being misrepresented as (or aliased into) the resolved wavenumbers. Consequently, these errors often manifest as unphysical energy growth at higher wavenumbers, leading to inaccurate energy transfer and potential numerical instability [6,9]. In this situation, adding artificial dissipation such as high-order filters [10–12] becomes an option to remove the aliased energy and stabilise simulations; however, it is not always the first option [9]. As an alternative, recasting the convective terms in the ‘skew-symmetric’ split forms has been proposed to mitigate energy aliasing and nonlinear instability without needing artificial dissipation [6,9,13,14].

∗Corresponding author. E-mail addresses: ye.wang7@unswalumni.com (Y. Wang), armin.wehrfritz@utu.fi (A. Wehrfritz), evatt.hawkes@unsw.edu.au (E.R. Hawkes).

https://doi.org/10.1016/j.jcp.2025.114269 Received 25 June 2024; Received in revised form 29 July 2025; Accepted 1 August 2025

Journal of Computational Physics 540 (2025) 114269

Available online 5 August 2025 0021-9991/© 2025 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license ( http://creativecommons.org/licenses/by/4.0/ ).

Y. Wang, A. Wehrfritz and E.R. Hawkes

Blaisdell et al.[13] showed that a quadratic-split form using spectral methods reduced these aliasing errors, compared to the divergence form. Kennedy and Gruber[9] further developed stable cubically split forms by fully expanding the triple products for flows with large density variations (e.g., compressible reacting flows), which showed lower aliasing errors and improved stability relative to quadratic-split forms. Unlike the analysis using exact Fourier modes by Blaisdell et al.[13] and Kennedy and Gruber[9], Kuya and Kawai[15] used the modified wavenumber of finite-difference operators to analyse the spectral characteristics of split forms. This analysis [15] showed that in finite-difference methods, the more stable split forms exhibited larger aliasing errors compared to the divergence form. This observation indicated that reducing aliasing errors of split forms may not always be the primary mechanism by which to improve stability in finite-difference methods [15], as also noted by Honein and Moin[16]. Furthermore, Honein and Moin[16] and Kuya and Kawai[15] suggested that preserving secondary conservative variables, such as kinetic energy and entropy, could be more crucial for improving numerical stability by the split forms in finite-difference methods.

Given the vital role of the kinetic energy balance in turbulence development, it is important that the numerical scheme is capable of representing the correct evolution of kinetic energy. As defined by Jameson[17], in kinetic-energy-preserving (KEP) schemes for compressible flows, the discrete volume integral of kinetic energy is not changed by the convective terms but only by the pressure work, ignoring the viscous terms, boundary conditions, and time integration errors. A KEP scheme can be easily obtained from the consistency condition due to Jameson[17], which was derived for second-order finite-volume schemes. KEP schemes have been successfully used in incompressible [18] and compressible simulations [2,19,20] to suppress numerical instabilities. Feiereisen et al.[19] proposed a quadratic-split form that is KEP for compressible flows. Pirozzoli[2] found a cubic-split form from Kennedy and Gruber[9] is also KEP, which has subsequently been applied in other studies [21,22]. Comparisons, such as those by Kuya et al.[22] and Gassner et al.[4], have indicated that this cubic-split form [2,9] was a promising approach for compressible flows regarding numerical robustness and computational efficiency. Other notable KEP schemes for compressible flows are based on square-root density splittings, such as the formulations of Morinishi[1] and Rozema et al.[23], and recent works [24,25].

Despite its success, discrete conservation of kinetic energy is not a sufficient condition for numerical stability in under-resolved simulations of compressible flows, and additional conditions such as entropy preservation have been proposed [26]. It has been shown that for schemes achieving discrete conservation of entropy1, thermodynamic fluctuations converge and numerical instabilities are suppressed for high Reynolds number flows [16,21,22]. Following the concept by Tadmor[27], Chandrashekar[28] used the physical entropy as the mathematical entropy function to derive a second-order entropy-conservative scheme that is also KEP. However, Gassner et al.[4] reported its failure to preserve kinetic energy in a discontinuous Galerkin framework, which, due to the summation-by-parts property, shares many features with the finite-difference formulation.  Ranocha[29] furthermore constructed an improved scheme that is entropy-conservative according to Tadmor’s concept and preserves kinetic energy. A different approach was taken by Honein and Moin[16], who enforced exact conservation of the physical entropy by solving a modified internal energy equation that is numerically equivalent to the entropy equation. Similarly, Coppola et al.[21] achieved exact entropy conservation by directly solving the entropy equation. Interestingly, they also found that solving the standard internal energy equation in the cubic-split form nearly conserves entropy. Although a rigorous proof was not offered by Coppola et al.[21], this outcome may be linked to implicit entropy conservation via the Gibbs equation, which, as described by Subbareddy and Candler[20] and Kuya et al.[22], relates the continuity and internal energy equations with the entropy equation. Based on the Gibbs equation, consistency conditions were summarised in Refs. [3,22,30]; satisfying these conditions helps correctly represent local energy exchange between kinetic energy and internal energy, and thus supports entropy conservation. Particularly, the kinetic-energyand entropy-preserving (KEEP) scheme due to Kuya et al.[22] achieved discrete entropy conservation implicitly by solving the total energy equation with consistent split-form numerical fluxes. This approach is preferred as it also conserves total energy, which is important for compressible problems. Although this method does not conserve entropy exactly, the entropy conservation errors were small [22,30,31] and adjustable [32].

However, subsequent research by Shima et al.[31] found that despite conserving kinetic energy, entropy, and total energy, the split form by Kuya et al.[22] led to numerical instability in a resolved single-component flow with initially uniform pressure and velocity but varying density. This instability was attributed to spurious pressure oscillations induced by the cubic-split form for internal energy. To remedy this, Shima et al.[31] suggested a quadratic-split form that eliminated these oscillations, thus preserving pressure equilibrium and stabilising the simulation with a slight impact on entropy conservation. They described pressure equilibrium in inviscid, ideal-gas, single-component flows as the physical property where initially uniform pressure and velocity remain constant despite variations in density, and characterised their scheme, which consistently maintains this property in the discrete level, as pressureequilibrium-preserving (PEP). Ranocha and Gassner[33] subsequently found the entropy-conservative and KEP scheme developed by Ranocha[29] is also PEP. However, this scheme has stability issues due to the use of logarithmic means in the density flux [33] and is computationally expensive. Additionally, De Michele and Coppola recently confirmed the square-root splitting proposed by Rozema et al.[23] to be PEP [34] and to exhibit excellent entropy conservation [35], as further explored by Kawai and Kawai[36]. De Michele and Coppola[34] also constructed new PEP schemes through minor modifications to classical numerical fluxes of Pirozzoli[2] and Kuya et al.[22]. These modifications were shown to improve the overall performance of the original schemes without

1 Entropy in this article refers to the physical entropy, unless stated otherwise. Entropy conservation means that the discrete volume integral of the physical entropy is not changed by the convective terms and is preserved over time, assuming the viscous terms, boundary conditions and time integration errors are ignored.

Journal of Computational Physics 540 (2025) 114269

2

Y. Wang, A. Wehrfritz and E.R. Hawkes

increasing computational cost in ideal-gas, single-component flows. On the other hand, in multi-component flows, ensuring pressure equilibrium across interfaces, where jumps in specific heats may occur, presents extra challenges, particularly within a fully conservative system [37,38]. Despite many studies working on this issue [39–41], the application of split forms for maintaining pressure equilibrium had not been explored until the recent work by Fujiwara et al.[42]. They proposed a fully conservative split-form scheme that preserves pressure equilibrium at interfaces without jumps for calorically perfect gases, which reduces to the scheme by Shima et al.[31] for uniform concentrations. The entropy-conservative schemes in Tadmor’s framework (e.g., Ranocha[29]) have also recently been extended to multi-component, thermally perfect gas in a discontinuous Galerkin setting [43,44]. However, these do not satisfy a strict PEP property. Furthermore, Jain and Moin[30] derived a condition to preserve pressure equilibrium at two-phase flow interfaces, which requires identical split forms for internal energy and total mass, resulting in the same split forms as Shima et al.[31]. While different split forms were evaluated by Jain and Moin[30], a uniform specific heat ratio was assumed. In other related studies, more sophisticated PEP split forms have been constructed for real-fluid simulations to address challenges arising from highly nonlinear relations between pressure and internal energy [45,46]. Therefore, although various split forms of PEP schemes have been proposed over the past few years, discussion on most schemes in multi-component, compressible flows with varying specific heats remains limited. In particular, there is no thorough evaluation of the PEP schemes by Shima et al.[31] and Rozema et al.[23] in this context, leaving open questions about whether they still outperform other split forms, such as those that only preserve kinetic energy [2] or entropy [22], and how they compare to PEP schemes specifically designed for multi-component flows, such as the one by Fujiwara et al.[42].

For multi-component flows, the numerical schemes for the convective terms in the species mass transport equations require further careful consideration. The discrete conservation of species mass should first be preserved. This is particularly important in chemically reacting flows as a prerequisite for accurately evaluating reaction rates and other properties. When solving conservative species equations, Trisjono et al.[47] suggested discretely satisfying the advection forms of these equations, as it helps avoid an initially uniform species field being disturbed spuriously for flows with non-uniform density or velocity. Additionally, Terashima et al.[41] noted that properly treating the molar concentration, associated with species partial densities, is important to preserve temperature equilibrium at material interfaces. Although this challenge is typically encountered at material interfaces of multi-component flows [40,41], we will show that even with uniform specific heats and preserved pressure equilibrium, improper reconstructions of partial densities at flux points can cause spurious temperature oscillations. This was also discussed by Subbareddy et al.[48] for finite-volume methods. They advised against separating total density and species mass fraction when reconstructing partial densities at cell faces to prevent large temperature excursions. Otherwise, for systems solving equations for all species without the continuity equation, numerical schemes need to ensure that all partial densities sum to the mixture density [48]. Jain and Moin[30] implemented this condition in constructing split-form numerical fluxes for phase masses in multiphase flows. Nevertheless, very few studies have explored split forms for the convective terms in species mass transport equations, and to the best of our knowledge, the characteristics of different split forms for species convective terms have not been discussed, particularly regarding the key physical properties discussed above, including species mass conservation, uniform species field and temperature-equilibrium preservation. This gap remains despite the fact that some studies have applied split forms to species convective terms. For instance, Kennedy and Gruber[9] adopted split forms with better aliasing behaviour and computational efficiency, Pantano et al.[49] used a quadratic-split form to improve stability, Fujiwara et al.[42] employed a quadratic-split form for its compatibility with their pressure-equilibrium condition, and Terashima et al.[46] also chose a quadratic-split form when constructing their approximate PEP scheme. While Fujiwara et al.[42] discussed the PEP property of their quadratic-split form and a divergence form, their characteristics concerning the key physical properties mentioned above were not explored.

Therefore, the focus in the present study is on the development of a high-order, non-dissipative, accurate, stable and physically consistent2 finite-difference scheme for turbulent compressible multi-component flows. The first objective is to evaluate the performance and properties of schemes using the pressure-equilibrium-preserving split forms, including the schemes by Shima et al.[31] and Rozema et al.[23], in the context of multi-component flows. This evaluation includes a comparison with other existing schemes such as low-aliasing, kinetic-energy-preserving, or entropy-preserving schemes, and with the multi-component pressure-equilibrium-preserving scheme by Fujiwara et al.[42]. The properties of the schemes are investigated by analytical methods and their performance is tested in numerical experiments. A second objective is to construct a scheme for convective terms in the species mass transport equations based on the essential physical properties, including species mass conservation, uniform species field and temperature-equilibrium preservation. The capability of the proposed scheme in maintaining these physical properties is demonstrated analytically and verified in numerical tests. It is also compared to other possible schemes that do not satisfy these physical properties, to highlight the importance of using physically consistent forms in computing the mass fraction and temperature. It should be noted that we specifically aim at constructing high-order schemes for direct numerical simulations, and thus all formulations are derived for an arbitrary order of accuracy, and all numerical tests are carried out using eighth-order schemes.

The remainder of this article is organised as follows. Section 2 summarises the governing equations. Section 3 describes the conditions and formulations for the schemes which are in the numerical flux form, along with a summary of the schemes to be tested. Section 4 presents the numerical tests in terms of the two objectives, followed by concluding remarks in Section 5.

2 We say a scheme is physically consistent if the scheme consistently maintains relevant physical properties discretely.

Journal of Computational Physics 540 (2025) 114269

3

Y. Wang, A. Wehrfritz and E.R. Hawkes

2.  Governing equations

2.1.  Conservation equations

We are interested in solving the compressible, multi-component, chemically-reacting flow equations. However, given the current focus on numerical schemes for the convective terms, we consider the inviscid system, which contains the classical three-dimensional, compressible Euler equations for the mixture and additional species transport equations, as widely considered in the community [9, 40,49–53]:

𝜕𝜌

𝜕𝑡+ 𝜕𝜌𝑢𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq001.png)

𝜕𝜌𝑢𝑖

𝜕𝑡 + 𝜕𝜌𝑢𝑖𝑢𝑗

𝜕𝑥𝑗 + 𝜕𝑝


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq002.png)

𝜕𝜌𝐸

𝜕𝑡 + 𝜕𝜌𝐸𝑢𝑗

𝜕𝑥𝑗 + 𝜕𝑝𝑢𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq003.png)

𝜕𝜌𝑌𝛼

𝜕𝑡 + 𝜕𝜌𝑌𝛼𝑢𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq004.png)

where 𝑡 denotes time, 𝑥𝑖 are the spatial coordinates in a Cartesian coordinate system, 𝑢𝑖 is the velocity vector component in the 𝑖th

direction, 𝜌 is the mixture density, 𝑝 is the mixture total pressure, 𝑌𝛼 is the mass fraction of species 𝛼, and 𝑁𝑠 is the total number of species. Only 𝑁𝑠−1 species equations are solved and the mass fraction of species 𝑁𝑠 is determined from the constraint: ∑𝑁𝑠 𝛼=1 𝑌𝛼= 1. This approach ensures that the mass fractions of all species sum up to unity by design. The specific total energy, 𝐸= 𝑒+ 𝑘, is the sum of mixture specific internal energy, 𝑒, and specific kinetic energy, 𝑘= 𝑢𝑖𝑢𝑖∕2. Accordingly, as in Kuya et al.[22], we can rewrite the total energy equation (Eq. (3)) as

𝜕𝜌𝐸

𝜕𝑡 + 𝜕𝜌𝑒𝑢𝑗

𝜕𝑥𝑗 + 𝜕𝜌𝑘𝑢𝑗

𝜕𝑥𝑗 + 𝜕𝑝𝑢𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq005.png)

Similarly, as in Pirozzoli[2], using the mixture total specific enthalpy, 𝐻= 𝐸+ 𝑝∕𝜌, the total energy equation (Eq. (3)) becomes:

𝜕𝜌𝐸

𝜕𝑡 + 𝜕𝜌𝐻𝑢𝑗

𝜕𝑥𝑗 = 0. (6)

In the discussed equations, the conserved variables are solved directly, and the convective terms are expressed in the divergence form. This approach is commonly used in compressible solvers [52,54] to discretely conserve total mass, momentum, total energy, and species mass. However, as will be explored in Section 3, there are alternative convective formulations that are analytically equivalent to the divergence form. Despite being numerically different, these formulations may also achieve similar discrete conservation.

2.2.  Equation of state

We assume that the species are calorically perfect gases in thermal equilibrium, each with constant specific heat capacities 𝐶𝑝,𝛼 and 𝐶𝑣,𝛼. Consequently, the mean specific heats of their mixture alter with composition but remain constant with temperature changes. This simplifies the computation of thermodynamic relations by using mixing rules. It also allows for diverse spatial distributions of the mixture specific heats, a critical aspect of our study on pressure equilibrium in multi-component flows. The specific internal energy of species 𝛼 is


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq006.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq007.png)

and the mixture specific internal energy is given by

𝑒=

𝑁𝑠 ∑

𝛼=1 𝑌𝛼𝑒𝛼. (8)

Therein, 𝑇 is the temperature, 𝑅𝑢 is the universal gas constant, 𝑊𝛼 and 𝛾𝛼 are respectively the molar mass and the specific heat ratio of species 𝛼, the partial pressure is 𝑝𝛼= 𝑝𝑋𝛼 with 𝑋𝛼 denoting the species mole fraction, and the partial density 𝜌𝛼= 𝜌𝑌𝛼 denotes the density of species 𝛼. With the relation 𝑌𝛼∕𝑋𝛼= 𝑊𝛼∕𝑊, the mean molar mass of the mixture, 𝑊, is

𝑊=

𝑁𝑠 ∑

𝛼=1 𝑋𝛼𝑊𝛼=


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq008.png)

𝛼=1

𝑌𝛼 𝑊𝛼


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq009.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq010.png)

The mean specific heat ratio of the mixture is 𝛾= 𝐶𝑝∕𝐶𝑣, where 𝐶𝑝= ∑𝑁𝑠 𝛼=1 𝑌𝛼𝐶𝑝,𝛼 and 𝐶𝑣= ∑𝑁𝑠 𝛼=1 𝑌𝛼𝐶𝑣,𝛼 are the mean specific heats, and 𝛾 can be obtained from:

1 𝛾−1 =

𝑁𝑠 ∑

𝛼=1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq011.png)

Journal of Computational Physics 540 (2025) 114269

4

Y. Wang, A. Wehrfritz and E.R. Hawkes

Then, we obtain the thermodynamic equation of state for the mixture:

𝑒= 𝑝 𝜌(𝛾−1) , (11)

which closes the balance equations, and the mixture ideal-gas law:

𝑝= 𝜌𝑅𝑢

𝑊𝑇. (12)

3.  Numerical flux form of split convective terms

3.1.  Numerical flux form

For compressible flows that may develop discontinuities, it is beneficial to globally conserve the primary invariants (i.e., total mass, momentum, total energy, and species mass) in a discrete sense, since according to the Lax–Wendroff theorem [55], it assures the discrete solution converges to a unique weak solution upon grid convergence. The numerical flux form, which casts the discretisation of convective terms in the difference of numerical fluxes at adjacent intermediate nodes, ensures global conservation through the telescopic property [55]. It is also known as the local conservative form [14], and it achieves better numerical stability than forms without discrete conservation [21]. Using 𝐹 to represent a generic flux function, the spatial derivative in the 𝑗th direction at grid point 𝑚 is discretised using the numerical flux form:

𝜕𝐹𝑗 𝜕𝑥𝑗

|||||𝑚 ≈ ̂𝐹𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq012.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq013.png)

where ̂𝐹𝑗|𝑚± 1

2  denotes the numerical flux at the flux point 𝑚± 1

2 between grid points 𝑚 and 𝑚± 1, and Δ𝑥𝑗 is the grid spacing size.

Applying the generic numerical flux (Eq. (13)), the semi-discrete representation for Eqs. (1)–(4) can be expressed as:

𝜕𝜌

𝜕𝑡 ||||𝑚 + ̂𝐶𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq014.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq015.png)

𝜕𝜌𝑢𝑖

𝜕𝑡 ||||𝑚 + ̂𝑀𝑖𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq016.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq017.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq018.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq019.png)

𝜕𝜌𝐸

𝜕𝑡 ||||𝑚 + ̂𝐸𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq020.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq021.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq022.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq023.png)

𝜕𝜌𝑌𝛼

𝜕𝑡 ||||𝑚 + ̂𝑌𝛼,𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq024.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq025.png)

where ̂𝐶𝑗, ̂𝑀𝑖𝑗, ̂𝐺, ̂𝐸𝑗, ̂𝐷𝑗, and ̂𝑌𝛼,𝑗 are the total mass, momentum (in the 𝑖th direction), pressure-gradient, total energy, pressurediffusion, and species mass (for species 𝛼) numerical fluxes, respectively. Many compressible flow solvers (e.g., [52]) adopt the intuitive formulation of the energy equation (Eq. (16)), which was also used by Kennedy and Gruber[9]. An alternative formulation, based on separating the internal and kinetic energy (Eq. (18)), was used by Kok[3], Kuya et al.[22], Shima et al.[31], and Jain and Moin[30], whereas Jameson[17] and Pirozzoli[2] used the total enthalpy leading to Eq. (19):

𝜕𝜌𝐸

𝜕𝑡 ||||𝑚 + ̂𝐼𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq026.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq027.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq028.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq029.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq030.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq031.png)

𝜕𝜌𝐸

𝜕𝑡 ||||𝑚 + ̂𝐻𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq032.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq033.png)

where ̂𝐼𝑗, ̂𝐾𝑗, and ̂𝐻𝑗 are the internal energy, kinetic energy, and total enthalpy numerical fluxes, respectively. Coppola et al.[21] employed the internal energy equation:

𝜕𝜌𝑒

𝜕𝑡 ||||𝑚 + ̂𝐼𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq034.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq035.png)

̂𝑈𝑗|𝑚+ 1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq036.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq037.png)

where ̂𝑈𝑗 is the numerical flux for 𝜕𝑢𝑗∕𝜕𝑥𝑗 in the pressure-dilation term (𝑝𝜕𝑢𝑗∕𝜕𝑥𝑗).

3.2.  ‘Skew-symmetric’ split form

To simplify the notation, we will introduce the split forms in one dimension. The extension to two and three dimensions is trivial. The generic flux function can be expressed as 𝐹= 𝑓𝑔 and 𝐹= 𝑓𝑔ℎ respectively for quadratically and cubically nonlinear derivatives, where 𝑓, 𝑔, and ℎ are arbitrary functions of 𝑥. Quadratically nonlinear derivatives can be split as:

𝜕𝑓𝑔

𝜕𝑥= 𝛼𝑞 𝜕𝑓𝑔

𝜕𝑥+ (1 −𝛼𝑞 )( 𝑓𝜕𝑔

𝜕𝑥+ 𝑔𝜕𝑓

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq038.png)

Journal of Computational Physics 540 (2025) 114269

5

Y. Wang, A. Wehrfritz and E.R. Hawkes

where 𝛼𝑞 is an arbitrary coefficient. Ducros et al.[14] first showed that this split form with 𝛼𝑞= 1∕2 can be written in the numerical flux form for explicit central-difference approximations. Fisher et al.[56] proved that the telescopic property [55] is satisfied for any constant 𝛼𝑞 in conservation law equations discretised using any diagonal-norm split summation-by-part (SBP) operators. For cubically nonlinear derivatives, Kennedy and Gruber[9] derived the general split form:


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq039.png)

𝜕𝑥 = 𝛼𝑐 𝜕𝑓𝑔ℎ

𝜕𝑥 + 𝛽 ( 𝑓𝜕𝑔ℎ


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq040.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq041.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq042.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq043.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq044.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq045.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq046.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq047.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq048.png)

where 𝛼𝑐+ 𝛽+ 𝜅+ 𝛿+ 𝜁= 1. Eq. (22) contains all the possible combinations in Eq. (21) by letting the function ℎ= 1, the coefficients 𝛼𝑐+ 𝛿= 𝛼𝑞, and 𝛽+ 𝜅= 1 −𝛼𝑞. However, not all combinations of Eq. (22) can be cast into the numerical flux form (Eq. (13)) [2,21],

such as the formulation with 𝛼𝑐= 𝜁= 1

2 and 𝛽= 𝜅= 𝛿= 0, referred to here as the KG form:


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq049.png)

𝜕𝑥 = 1

2 𝜕𝑓𝑔ℎ

𝜕𝑥 + 1

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq050.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq051.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq052.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq053.png)

The KG form (Eq. (23)) demonstrates low aliasing errors and good numerical stability, as tested by Kennedy and Gruber[9], despite lacking conservation of primary quantities (except for total mass) and secondary quantities (kinetic energy and entropy). We include the KG form (Eq. (23)) in this study for comparative evaluation with the schemes introduced later in Section 3.3. For compressible multi-component flows, the governing equations split using the KG form read:

𝜕𝜌

𝜕𝑡+ 1

2

𝜕𝜌𝑢𝑗

𝜕𝑥+ 1

2

( 𝑢𝑗 𝜕𝜌 𝜕𝑥+ 𝜌 𝜕𝑢𝑗

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq054.png)

𝜕𝜌𝑢𝑖

𝜕𝑡 + 1

2

𝜕𝜌𝑢𝑖𝑢𝑗

𝜕𝑥 + 1

2

( 𝑢𝑖𝑢𝑗 𝜕𝜌 𝜕𝑥+ 𝜌𝑢𝑗 𝜕𝑢𝑖

𝜕𝑥+ 𝜌𝑢𝑖 𝜕𝑢𝑗

𝜕𝑥

) + 𝜕𝑝


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq055.png)

𝜕𝜌𝐸

𝜕𝑡 + 1

2

𝜕𝜌𝐸𝑢𝑗

𝜕𝑥 + 1

2

( 𝐸𝑢𝑗 𝜕𝜌 𝜕𝑥+ 𝜌𝑢𝑗 𝜕𝐸

𝜕𝑥+ 𝜌𝐸 𝜕𝑢𝑗

𝜕𝑥

) + 1

2

𝜕𝑝𝑢𝑗

𝜕𝑥+ 1

2

( 𝑢𝑗 𝜕𝑝 𝜕𝑥+ 𝑝 𝜕𝑢𝑗

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq056.png)

𝜕𝜌𝑌𝛼

𝜕𝑡 + 1

2

𝜕𝜌𝑌𝛼𝑢𝑗

𝜕𝑥 + 1

2

( 𝑌𝛼𝑢𝑗 𝜕𝜌 𝜕𝑥+ 𝜌𝑢𝑗 𝜕𝑌𝛼

𝜕𝑥+ 𝜌𝑌𝛼 𝜕𝑢𝑗

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq057.png)

On the other hand, Coppola et al.[21] showed that 𝜁= 0 is a sufficient condition for casting Eq. (22) into numerical flux form for explicit central-difference approximations, assuming all the coefficients are constants. With this constraint, we obtain four formulations:

Divergence form ∶𝜕𝑓𝑔ℎ


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq058.png)

Quadratic-split form ∶1

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq059.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq060.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq061.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq062.png)

Cubic-split form ∶1

4


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq063.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq064.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq065.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq066.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq067.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq068.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq069.png)

𝜕𝑥


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq070.png)

Product-rule form ∶𝑓𝜕𝑔ℎ


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq071.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq072.png)

where the terminology extends to the quadratically nonlinear derivatives by setting ℎ= 1. The quadratic-split (𝛼𝑐= 𝛽= 1

2; 𝜅= 𝛿= 𝜁=

0) and cubic-split (𝛼𝑐= 𝛽= 𝜅= 𝛿= 1

4 ; 𝜁= 0) forms are referred to as the ‘standard’ split forms by Pirozzoli[2] and are widely used [4, 22,30]. Additionally, we consider the square-root-based split form due to Rozema et al.[23] (see also De Michele and Coppola[34] and Kawai and Kawai[36]):

Square-root form ∶1

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq073.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq074.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq075.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq076.png)

𝜕𝑥

)

+ 1

2

(


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq077.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq078.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq079.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq080.png)

𝜕𝑥

)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq081.png)

Note that varying combinations of 𝑓, 𝑔, and ℎ may result in different discretisation schemes, each with different properties. For instance, the quadratic-split form by Feiereisen et al.[19] using 𝑓= 𝑢𝑖, 𝑔= 𝜌, and ℎ= 𝑢𝑗 is KEP, whereas the one due to Blaisdell et al.[13] with 𝑓= 𝑢𝑗, 𝑔= 𝜌, and ℎ= 𝑢𝑖 is not.

Applying the two-point discrete averaging operator from Pirozzoli[2] and Coppola et al.[21], and invoking the generic numerical flux (Eq. (13)), the numerical fluxes for these split forms in the 2𝐿th-order accuracy are expressed as:

̂𝐹|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq082.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq083.png)

where 𝑎𝐿,𝑙 is the coefficients in the standard explicit central-difference approximation of the first-order derivative with the formal order of accuracy of 2𝐿, and ̃ (𝑓, 𝑔, ℎ)𝑚−𝑘,𝑙 is the two-point discrete averaging operator. The corresponding operators to the forms in Eqs. (28)–(32) are:


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq084.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq085.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq086.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq087.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq088.png)

Journal of Computational Physics 540 (2025) 114269

6

Y. Wang, A. Wehrfritz and E.R. Hawkes


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq089.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq090.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq091.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq092.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq093.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq094.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq095.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq096.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq097.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq098.png)

All numerical fluxes in this study, except for the kinetic energy numerical flux, can be derived by substituting 𝑓, 𝑔, and ℎ with corresponding flow variables. Note that although we focus only on standard explicit schemes in this study, the split forms may also be discretised with compact stencils [57,58].

3.3.  Consistent forms for compressible multi-component flows

3.3.1.  Total mass and momentum numerical fluxes

The numerical fluxes for total mass, ̂𝐶𝑗, and momentum, ̂𝑀𝑖𝑗, are required to satisfy the kinetic-energy-preserving (KEP) property discretely. Jameson[17] derived a KEP condition:

̂𝑀𝑗|𝑚+ 1

2 = ̂𝐶𝑗|𝑚+ 1

2

𝑢𝑖|𝑚+ 𝑢𝑖|𝑚+1


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq099.png)

satisfying which results in a group of KEP schemes. However, this condition is confined to second-order accuracy and does not explicitly extend to high-order numerical fluxes. In Theorem 1, we extend the KEP condition to numerical fluxes in an arbitrary order of accuracy, with a proof provided in Appendix A.

Theorem 1. In the semi-discrete sense, the total kinetic energy is globally conserved, i.e., it is not changed by the convective terms but only changed by the pressure work and boundary terms, if the numerical fluxes in 2𝐿th-order accuracy for total mass,

̂𝐶𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq100.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq101.png)

and momentum,

̂ 𝑀𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq102.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq103.png)

satisfy the condition:


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq104.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq105.png)

where the two-point discrete averaging operators, ̃ (⋅, ⋅, ⋅), are defined in Eqs. (34)–(38).


## Proof. Appendix A. ∎

According to this condition, we can formulate KEP numerical fluxes with arbitrary order of accuracy. For instance, the standard quadratic-split and cubic-split forms are represented by

̂𝐶𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq106.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq107.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq108.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq109.png)

̂𝑀𝑖𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq110.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq111.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq112.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq113.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq114.png)

respectively. Their robustness and KEP property have been demonstrated in numerical tests [2,22,57]. Another group tested by Kuya et al.[22] is:

̂𝐶𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq115.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq116.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq117.png)

̂𝑀𝑖𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq118.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq119.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq120.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq121.png)

the divergence and quadratic-split formulations, respectively, which were shown to be less stable than Eqs. (43) and (44). Additionally, the square-root density splitting proposed by Rozema et al.[23] satisfies the KEP condition, and its excellent performance has recently been demonstrated by De Michele and Coppola[35] and Kawai and Kawai[36] in ideal-gas, single-component flows. The formulation is given by

̂𝐶𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq122.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq123.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq124.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq125.png)

Journal of Computational Physics 540 (2025) 114269

7

Y. Wang, A. Wehrfritz and E.R. Hawkes

̂𝑀𝑖𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq126.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq127.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq128.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq129.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq130.png)

In the present study, we adopt the two groups, Eqs. (43) and (44) and Eqs. (47) and (48), for their superior performance.

3.3.2.  Energy numerical fluxes 3.3.2.1.  Entropy conservation. The construction of the numerical fluxes in the energy equation should aim for discrete conservation of entropy. We adopt the method due to Kuya et al.[22], which implicitly conserves entropy by solving the total energy equation with consistent numerical fluxes. This method is grounded in the ‘conservation’ form of the Gibbs equation from Subbareddy and Candler[20]:

𝜕𝜌𝑠

𝜕𝑡+ 𝜕𝜌𝑠𝑢𝑗

𝜕𝑥𝑗 = 1

𝑇


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq131.png)

𝜌

)( 𝜕𝜌

𝜕𝑡+ 𝜕𝜌𝑢𝑗

𝜕𝑥𝑗

) + ( 𝜕𝜌𝑒

𝜕𝑡+ 𝜕𝜌𝑒𝑢𝑗

𝜕𝑥𝑗 + 𝑝 𝜕𝑢𝑗 𝜕𝑥𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq132.png)

where 𝑠= 𝐶𝑣ln(𝑝∕𝜌𝛾) is the mixture specific entropy. This equation (Eq. (49)) suggests that entropy is implicitly conserved upon satisfying the continuity and internal energy equations. The continuity equation is satisfied discretely by solving Eq. (14), while the satisfaction of the internal energy equation follows from satisfying the total energy and the kinetic energy equations, as per:

𝜕𝜌𝑒

𝜕𝑡+ 𝜕𝜌𝑒𝑢𝑗

𝜕𝑥𝑗 + 𝑝 𝜕𝑢𝑗 𝜕𝑥𝑗 = (𝜕𝜌𝐸

𝜕𝑡 + 𝜕𝜌𝐸𝑢𝑗

𝜕𝑥𝑗 + 𝜕𝑝𝑢𝑗

𝜕𝑥𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq133.png)

𝜕𝑡+ 𝜕𝜌𝑘𝑢𝑗

𝜕𝑥𝑗 + 𝑢𝑗 𝜕𝑝 𝜕𝑥𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq134.png)

Consistent numerical fluxes, i.e., fluxes satisfying the Analytical Relations3 proposed by Kuya et al.[22], are required for all convective terms in Eq. (50) to maintain correct local exchanges between internal energy and kinetic energy discretely and thus conserve entropy. Interested readers are referred to Refs. [22,57] for details.

In the present study, schemes are categorised as entropy preserving, as summarised later in Table 1, if they conserve entropy in the sense of Kuya et al.[22], that is, if they use consistent numerical fluxes that satisfy the Analytical Relations as defined by Kuya et al.[22]. However, this approach does not strictly guarantee exact entropy conservation, as it does not directly solve the entropy equation. Consequently, some entropy conservation errors are expected, and their magnitudes may vary significantly between schemes, as suggested by the error analysis in Tamaki et al.[32]. The actual entropy-conservation properties of the schemes considered will be assessed through the numerical tests in Section 4.1.

As per Refs. [22,57], for the total energy equation (Eq. (18)), the consistent kinetic energy numerical flux ̂𝐾𝑗, based on ̂𝐶𝑗 and ̂𝑀𝑖𝑗 from Eqs. (43) and (44), respectively, is formulated as:

̂𝐾𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq135.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq136.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq137.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq138.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq139.png)

The consistent internal energy numerical flux ̂𝐼𝑗 takes the cubic-split form:

̂𝐼𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq140.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq141.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq142.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq143.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq144.png)

However, relaxing the consistency condition in Analytical Relation 2 from Refs. [22,57] to

𝜕𝜌𝑘𝑢𝑗

𝜕𝑥𝑗 + 𝜕𝜌𝑒𝑢𝑗

𝜕𝑥𝑗 = 𝜕(𝜌𝑘+ 𝜌𝑒)𝑢𝑗


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq145.png)

allows for different splitting methods in ̂𝐼𝑗 (𝜌 and 𝑒) and ̂𝐾𝑗 (𝜌 and 𝑘), while still imposing the restriction that 𝑢𝑗 be consistently split in both. This flexibility permits the inclusion of the quadratic-split form:

̂𝐼𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq146.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq147.png)

2


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq148.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq149.png)

This is the pressure-equilibrium-preserving flux derived by Shima et al.[31] and Jain and Moin[30], and will be discussed in the next section. The consistent numerical flux for the pressure-diffusion term ̂𝐷𝑗 is in the product-rule form:

̂𝐷𝑗|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq150.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq151.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq152.png)

which is derived from the numerical flux for 𝜕𝑝∕𝜕𝑥𝑗 in Eq. (15):

̂𝐺|𝑚+ 1

2 = 2

𝐿 ∑

𝑙=1 𝑎𝐿,𝑙


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq153.png)

𝑘=0


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq154.png)


![Equation](images/[적용해볼것] PEP 관련 플럭스 스킴들 비교_eq155.png)

3 The Analytical Relations are: 1. the kinetic energy equation is derived from the continuity and momentum equations; 2. the kinetic and internal energies are convected by the same velocity; 3. the pressure-diffusion term is the sum of 𝑢𝑗𝜕𝑝∕𝜕𝑥𝑗 in the kinetic energy equation and 𝑝𝜕𝑢𝑗∕𝜕𝑥𝑗 in the internal energy equation [22].

Journal of Computational Physics 540 (2025) 114269

8

