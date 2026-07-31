Computers and Fluids 129 (2016) 134–145 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0001-01.png)


Contents lists available at ScienceDirect 

## ~~Computers and Fluids~~ 

journal homepage: www.elsevier.com/locate/compfluid 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0001-05.png)


## Assessment of SLAU2 and other flux functions with slope limiters in hypersonic shock-interaction heating 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0001-07.png)


## Keiichi Kitamura[∗] 

_Yokohama National University, 79-5 Tokiwadai, Hodogaya-ku, Yokohama, Kanagawa 240-8501, Japan_ 

a r t i c l e i n f o a b s t r a c t _Article history:_ Roles of flux functions (such as SLAU2 Kitamura and Shima, 2013), limiters, and reconstructed variables Received 24 September 2014 are thoroughly investigated in problems related to hypersonic heating issues, i.e., shock anomalous soRevised 1 January 2016 lutions (e.g., carbuncle phenomenon) and shock-interaction heating. Through numerical tests comparing Accepted 8 February 2016 Available online 18 February 2016 those different combinations, it is revealed that each of those factors has great impacts on the solutions at almost the same level. In particular, flux functions having at most one intermediate cell at the _Keywords:_ captured shock (e.g., AUSM[+] -up) show improved robustness against shock anomalies as the spatial acSLAU2 curacy increases, whereas those containing a few cells to represent the shock (e.g., SLAU2) tend to do AUSM[+] -up the opposite. Among many possible combinations, SLAU2, AUSM[+] -up, or AUSMPW+ along with _κ_ = -1, AUSMPW+ minmod-limited monotone upstream-centered schemes for conservation laws (MUSCL) interpolation for Flux limiter primitive variables show acceptable performance in the present study, as confirmed by the severe Type Hypersonic flow IV shock-interaction heating problem. In addition, conservation of mass flux across a shockwave is proven to be essential in accurate heating computations, indicating a possible, further modification of SLAU2. © 2016 Elsevier Ltd. All rights reserved. 

## **1. Introduction** 

In spite of maturity of the present computational fluid dynamics (CFD) technology, its reliability is still debated in hypersonic flows, particularly on shock anomalies [1–4] and heating prediction capabilities [5,6]. In a series of the authors’ past work [4–6], the following three properties (or called _“Hypersonic CFD Tips”_ hereafter) of Euler fluxes were found essential for accurate hypersonic heating computations: 

- (A) Robustness against shock anomalies (e.g., carbuncle phenomenon) 

- (B) Total enthalpy preserving (proved to be less critical than the other two, though) 

- (C) (Economical) boundary-layer resolving 

Although there found no methods _perfectly_ satisfying all the three items, the authors proposed promising candidates in [7] that possess them _in most cases_ , i.e., they are relatively robust against shock anomalies (property A). These anomalies include the carbuncle phenomenon [1–4] – a notorious problem of the Euler fluxes – which appears depending on many factors, such as flow conditions (Mach number, Reynolds number, specific heat ratio), computational grid (grid density, cell aspect ratio), and computational conditions (Euler flux, order of accuracy). Among 

> ∗ Corresponding author. Tel.: +81 45 339 3876. _E-mail address:_ kitamura@ynu.ac.jp 

these factors, we had focused on spatially first-order accurate performances of flux functions [4–7], based on the claim that “ _Carbuncle-like features are more evident in the plain first-order_ ” made by Pandolfi and D’Ambrosio in [2]. This statement is reasonable because jumps of variables at cell interfaces generally decrease with the order of accuracy in space. Remembering the fact that shock-capturing methods allow at least one intermediate cell inside the shock, and that the shock internal structure is expressed only numerically [4–11], anomalous solutions arising from such a numerically-defined zone are expected to be suppressed by thinning the region (sharper shock capturing) at a higher-order of accuracy. However, universality of this expectation is questionable for various flux functions and flux limiters available to date. This is partly because some flux functions (e.g., SLAU2 [7]) are designed to feed proper amount of dissipation to the captured shock at first-order spatial accuracy, and partly because strong limiters (e.g., minmod [12]) tend to yield first-order accuracy near discontinuities, whereas weak ones (e.g., superbee [12]) try to keep the second-order[1] as much as possible in expense of robustness (as widely known, higher-order accurate computations more likely oscillate at discontinuities). Thus, for stable and robust shock capturing, there may be appropriate combinations of a flux function and a limiter, which will be explored in the current work. 

> 1 In actuality, those spatial orders of accuracy are further reduced to one or even zero by non-differentiable limiters at discontinuities. From now, however, this explanation is omitted for brevity. 

http://dx.doi.org/10.1016/j.compfluid.2016.02.006 

0045-7930/© 2016 Elsevier Ltd. All rights reserved. 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

135 

A recent work by Tu et al. [13] extended the evaluation of flux functions in [4] to the fifth order weighted-compact-nonlinearscheme (WCNS). From their extensive survey, however, they concluded without reaching the concrete reasons that the high-order scheme may or may not be more stable than the low-order schemes depending on given computational conditions and grids. This revealed the need for further examinations on the relation between the shock anomalies and the spatial order of accuracy in a step-by-step manner, i.e., comparison between the first and second orders of accuracy (as will be done here). Coratekin et al. [14] conducted such a work a decade ago, but with only a few flux functions and limiters employed for limited test cases, and they recommended no combinations of a flux and a limiter, as opposed to the present work. Moreover, the work here will deal with the Edney’s type IV shock/shock interaction [15], which is known to yield a very severe surface heating, but whose best-suited flux function and/or limiter has not been clarified yet. 

Another important finding in the past work [7] and also by related researchers[2] is that even if the shock is smoothly captured (property A), and even if the chosen Euler flux is the one designed to satisfy properties B and C, the wall-heating profile may not be computed accurately. This odd behavior was observed only for a specific choice of a flux function (SLAU2 [7]), a limiter, and variables used for reconstruction. SLAU2 has been gaining its popularity recently: its low-speed performance has been studied in [16]; its variants have been suggested in [17]; its extension to multiphase flows has been conducted in [18]. Thus, the current discussion will be explored in depth not only from academic curiosity, but also for further improvements of SLAU2 and possibly other fluxes. 

The present paper will revisit the shock-robustness problem first, using several common or recently-proposed flux functions with (at most) _second-order_ accuracy. Then, we will compute heating profiles over a blunt-body with and without a shock/shock interaction. In each test case, method-to-method comparisons and discussions will be made for Euler fluxes, limiters, and reconstructed variables. Discussions will include the effect of a captured shockwave thickness, and introduce a new key element for accurate heating computations. 

## **2. Numerical methods** 

## _2.1. Governing equations_ 

The governing equations are the compressible Euler or Navier– Stokes equations as follows: 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0002-08.png)


- 2 Private communication with Hiroaki Nishikawa, National Institute of Aerospace, 

- Aug. 2010. 

where _ρ_ is the density, _ul_ velocity components in Cartesian coordinates, _E_ total energy per unit mass, _p_ pressure, _H_ total enthalpy ( _H_ = _E_ + ( _p_ / _ρ_ )), and _T_ temperature. The working gas is air approximated by the calorically perfect gas model with the specific heat ratio _γ_ = 1.4. The Prandtl number is Pr = 0.72. The molecular viscosity _μ_ and thermal conductivity _λ_ are related as _λ_ = _cpμ_ /Pr where _cp_ is specific heat at constant pressure. The viscosity _μ_ is calculated by Sutherland’s formula. These equations are discretized and solved with a finite-volume code. 

## _2.2. Numerical Methods_ 

Inviscid numerical fluxes at cell-interfaces are calculated by one of the following Euler fluxes: 

- Roe: Roe’s approximate Riemann solver (Flux-DifferenceSplitting, FDS) [19] 

- Roe (E-fix): Roe [19] with Harten’s entropy-fix (coefficient = 0.2) [20] 

- HLLE (Harten-Lax-van_Leer-Einfeldt) (Approximate Riemann Solver): HLL (Harten-Lax-van_Leer) [21] with Einfeldt’s wave estimation [22] 

- HLLC (Harten-Lax-van_Leer with Contact) (Approximate Riemann Solver) [23]: Contact-resolving extension of two-wave HLL [21] 

- Van Leer (Flux-Vector-Splitting, FVS) [24] 

- Hänel (Flux-Vector-Splitting, FVS) [25]: Total enthalpy preserving modification to Van Leer FVS [24] 

- AUSM[+] -up [26]: AUSM (Advection Upwind Splitting Method) family 

- SLAU2 [7]: Improved SLAU (Simple Low-dissipation AUSM) [27] 

- AUSM[+] -up2 [7]: Combination of AUSM[+] -up [26] mass flux and SLAU2 [7] pressure flux 

- AUSMPW+ [28]: AUSM-family featuring multidimensional pressure weighting function 

- SD-SLAU (Shock Detecting SLAU) [29]: SLAU [27] with multidimensional shock sensor 

- HLL-CPS-Zroe (HLL convective-pressure split Zha-Bilgen using Roe-averaged wave estimates) [30]: Hybrid of HLL [21] for mass flux and Zha-Bilgen FVS [31] for pressure flux 

Table 1 summarizes how each flux function satisfies the three properties mentioned in Introduction. Also, let us categorize those fluxes as to how thin or broad they capture shocks (a ‘thin’ shock stands for that having at most one intermediate cell inside, whereas a ‘broad’ shock is represented by a few cells) as listed below (and as also included in the Table 1): 

- ‘Thin’ shock capturing: Roe, Roe (E-fix), HLLC, AUSM[+] -up, and AUSMPW+ 

- ‘Broad’ shock capturing: HLLE, Van Leer, Hänel, SLAU2, AUSM[+] - up2, SD-SLAU, and HLL-CPS-Zroe 

Note that the term ‘broad’ does not necessarily stand for ‘too diffusive.’ It was reported in [7,29] that SLAU2, AUSM[+] -up2, and SD-SLAU have enough resolution both at shocks and boundarylayers (Appendix A shows that those fluxes well resolve the boundary layer, in contrast with Hänel and HLL-CPS-Zroe). 

The spatial accuracy guaranteed 2nd-order (by MUSCL with _κ_ = –1 [32] unless stated otherwise) at best. Either of minmod [12], Van Albada (coefficient = 10[−][6] ) [33], or superbee [12] flux limiter function (slope limiter) is employed, along with two-stage, 2ndorder Runge–Kutta or lower-upper symmetric Gauss-Seidel (LUSGS) for time integration. 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

136 

**Table 1** 

Three properties satisfied by flux functions. 

|Numerical fux functions|Property A: Robustness against|Property B: Total enthalpy|Property C: (Economical)|Captured shock|
|---|---|---|---|---|
||shock anomalies (at 1st-order)|preserving|boundary-layer resolving||
|Roe|Poor|No|Good|Thin|
|Roe (E-fx)|Poor|No|Good|Thin|
|HLLE|Fair|No|Poor|Broad|
|HLLC|Poor|No|Good|Thin|
|Van Leer|Good|No|Poor|Broad|
|Hänel|Good|Yes|Poor|Broad|
|AUSM + -up|Fair|Yes|Good|Thin|
|SLAU2|Good|Yes|Good|Broad|
|AUSM + -up2|Good|Yes|Good|Broad|
|AUSMPW+|Fair|Yes|Good|Thin|
|SD-SLAU|Poor|Yes|Good|Broad|
|HLL-CPS-Z roe|Fair|No|Poor|Broad|
||||(see Appendix A )||


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0003-05.png)


**Fig. 1.** Computational grid and conditions for 1.5D steady normal shock test. 

**Table 2** 

1.5D test results for various numerical flux functions, _1st-order_ both in time and space. 

|Numerical fux functions|Total shock-robustness scores|
|---|---|
|Roe [4,5]|8|
|Roe (E-fx) [4,5]|0|
|HLLE [4,5]|16|
|HLLC|8|
|Van Leer [5]|**20**|
|Hänel [5]|**20**|
|AUSM + -up [4,5]|16|
|SLAU2 [7]|**20**|
|AUSM + -up2 [7]|**20**|
|AUSMPW+ [4,5]|17|
|SD-SLAU|4|
|HLL-CPS-Z roe|17|


## **3. Numerical tests** 

## _3.1. Shock anomaly problem_ 

## _3.1.1. 1.5D normal shock (Euler Eqs.)_ 

This problem is called “1.5D (or 1-1/2-D) problem,” which was conducted in Ref. 4 to examine how schemes are robust in capturing a steady normal shock in a two-dimensional rectangular domain (Fig. 1). This setup mimics a close-up view of a hypersonic flow ahead of a stagnation point of a two-dimensional blunt-body, and hence, largely predicts results of such flow computations. In this paper we briefly review the problem and refer to Ref. 4 for details. 

As shown in Fig. 1, the computational grid comprises 50 × 25 cells evenly spaced without perturbations. A steady shock that includes an intermediate state is prescribed with initial conditions for left ( _L: i_ ≤ 12) and right ( _R: i_ ≥ 14) following the RankineHugoniot conditions across the normal shock. The internal shock conditions ( _M: i_ = 13) are as follows: 

- (1) The density is given as 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0003-16.png)


where the shock-position parameter _ε_ = 0.0, 0.1, ..., 0.9. The initial shock is imposed exactly on the cell-interface when _ε_ = 0.0, for instance, and at the cell-center when _ε_ = 0.5. 

- (2) The other variables are calculated based on _ρM_ so that all variables lie on the Hugoniot curve. The shock is aligned in one direction in the two-dimensional field, with the freestream Mach number _M_ ∞ = 6.0. No perturbations are introduced to the initial condition either. 

The computations are conducted for 40,000 steps with CFL = 0.5. If a scheme is stable for all the shock positions of _ε_ , the scheme can be labeled as 1.5D stable. 

Typical solutions are shown in Fig. 2. In Fig. 2, as stated in [4], 

- ‘2’ denotes a stable and symmetric solution with at least three orders of (L2-norm of) density residual reduction (Fig. 2a). 

- ‘1’ denotes an asymmetry and/or oscillation of the shock confined within two cells of the shock normal direction (Fig. 2b and c). 

- ‘0’ denotes an unstable solution usually associated with total breakdown of the shock (“carbuncle”) (Fig. 2d). 

These points introduced in Ref. 5 are used in Table 2 for the first order (both in space and time) results: given in the right column are the total points that indicate degrees of shock robustness for each scheme (maximum: 20 points). The following observations are new: 

- HLLC behaved like Roe, as expected (but not actually tested thus far). 

- SD-SLAU, which can eliminate multidimensional oscillations by shock-detecting function, unfortunately yielded carbuncle under some circumstances (0.0 ≤ ɛ ≤ 0.1, 0.4 ≤ ɛ ≤0.9). 

- HLL-CPS-Zroe showed slightly better performance (17 points) than its ingredient HLLE (16 points). 

The other results presented in this table were already reported in the past work [4,5,7], but repeated again here for reference (since they are essential for comparison with the 2nd-order results presented later). From these results, the following five flux functions will be selected and used in the rest of the paper. 

- ‘Thin’ shock capturing fluxes: Roe (E-fix), AUSM[+] -up, and AUSMPW+ 

- – ‘Broad’ shock capturing fluxes: Hänel and SLAU2 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

137 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0004-02.png)


**Fig. 2.** Typical solutions for 1.5D steady shock test: (a) 2 (Good: Stable), (b) 1 (Fair: Oscillatory), (c) 1 (Fair: Asymmetry), and (d) 0 (Poor: Carbuncle) [5]. 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0004-04.png)


**Fig. 3.** Computational grids, (a) Hypersonic, viscous, 2D blunt-body problem (160 × 160; every other grid lines are shown), and (b) Type IV shock interaction heating (320 × 480; every other four grid lines are shown). 

**Table 3** 1.5D test results for various numerical flux functions, _2nd-order_ both in time and space (by _minmod, prm_ ) ( _κ_ = –1 MUSCL). 

|space (by _minmod,_|_prm_ ) ( _κ_= –1 MUSCL).|||
|---|---|---|---|
|Numerical fux|Total shock-robustness|(1st-order)|(Captured shock)|
|functions|scores|||
|Roe (E-fx)<br>Hänel<br>AUSM + -up<br>SLAU2<br>AUSMPW+|8<br>17<br>20<br>17<br>18|(0)<br>(20)<br>(16)<br>(20)<br>(17)|(Thin)<br>(Broad)<br>(Thin)<br>(Broad)<br>(Thin)|


Roe (E-fix) is the most carbuncle-prone example; Hänel is known to be relatively robust against shocks but its boundary-layer resolution is poor; the other three have good performances in capturing shocks and boundary-layers both. Results using fluxes other than those five are expanded in [34]. 

_3.1.1.1. Effects of 2nd-order extension._ In Table 3, the corresponding 2nd-order MUSCL results are summarized (minmod-limited _κ_ = – 1 MUSCL with primitive-variable reconstruction), along with the 1st-order results and shockwave thicknesses repeated for ease of reference: 

- Compared with the 1st-order cases, _‘thin’ shock capturing fluxes_ [Roe (E-fix), AUSM[+] -up, and AUSMPW+] _showed improved robustness against shock anomalies_ (following the claim by Pan- 

dolfi and D’Ambrosio in [2]), _whereas the ‘broad’ counterparts_ [Hänel and SLAU2] _did the opposite_ (on the contrary to [2]). – At 2nd-order, AUSM[+] -up marked the highest score (20 points), followed by AUSMPW+ (18 points). 

We stated in [7] that the proper amount of dissipation was fed to SLAU2 (and AUSM[+] -up2); This is true for 1st-order spatial accuracy, but does not seem so for 2nd-order from the current results. This is understandable remembering the fact that a higher-order spatial accuracy generates a thinner shock in general, which apparently goes against the strategy taken in [7] where shock anomalies were suppressed by the dissipation addition (which usually widens the shock). This suggests that it would be preferred to _add different amount of dissipation to a flux function (specifically, a ‘broad’ shock capturing flux function) depending on spatial order of accuracy_ . 

_3.1.1.2. Effects of flux limiters and reconstructed variables._ Then Tables 4 and 5 show the results of selected fluxes with different limiters and different reconstructed variables. When primitive variables are interpolated (denoted as “prm”), 

- Different limiters resulted in different total points. The minmod limiter showed the highest scores, and in most cases, the Van Albada the next, and the superbee the last, in the order of the strength of the limiter. 

- Differences in the results due to limiters can be larger than those due to fluxes. For instance, SLAU2 scored 10 to 17 by different limiters, but the difference from AUSM[+] -up’s score (20) is only 3 for the same (minmod) limiter cases. 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

138 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0005-02.png)


**Fig. 4.** Hypersonic, viscous, 2D blunt-body problem results (Mach number contours at 100,000 steps) of minmod set, (a) Roe (E-fix), (b) Hänel, (c) AUSM[+] -up, (d) SLAU2, and (e) AUSMPW+. 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0005-04.png)


**Fig. 5.** Surface pressure and heating profiles of minmod set results, (a) Roe (E-fix), (b) Hänel, (c) AUSM[+] -up, (d) SLAU2, and (e) AUSMPW+. 

These tendencies hold for both ‘thin’ (e.g., AUSM[+] -up) and ‘broad’ (e.g., SLAU2) shock capturing flux functions. This is against our anticipation from the findings in Table 3 that the effects of flux limiters would have been different depending on to which group the flux function belongs, i.e., ‘thin’ shock capturing fluxes (e.g., AUSM[+] -up) would favor superbee (the weakest limiter – closest to 2nd-order) while ‘broad’ ones (e.g., SLAU2) would prefer minmod (the strongest limiter – closest to 1st-order). Thus, it is interpreted that _flux limiters have almost the same (or even higher) level of influ-_ 

_ence on the behaviors of captured shocks as flux functions do_ . A combination of a ‘thin’ shock capturing flux and a weak limiter appears to have created insufficient dissipation at shocks, leading to, as often reported in literature ([27], for instance), over/undershoots. 

Furthermore, when we focus on the effects of variables for reconstruction in the same table, 

- Reconstruction using conservative variables (denoted as “csv”) rather than primitive ones (“prm”) generally destabilized solutions, regardless of which ‘thin’ or ‘broad’ shock capturing 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

139 

## **Table 4** 

1.5D test results for various limiters and reconstructed variables (SLAU2) ( _2ndorder_ both in time and space) ( _κ_ = –1 MUSCL). 

|_order_ both in time and space) ( _κ_= –1 MUSCL)|.|
|---|---|
|Numerical fux functions (limiter, variable)|Total shock-robustness scores|
|SLAU2 (minmod, prm)|17|
|SLAU2 (Van Albada, prm)|10|
|SLAU2 (superbee, prm)|10|
|SLAU2 (minmod, csv+p)|13|
|SLAU2 (Van Albada, csv+p)|10|
|SLAU2 (superbee, csv+p)|1|
|SLAU2 (minmod, csv)|0|
|SLAU2 (Van Albada, csv)<br>SLAU2 (superbee, csv)|0<br>0|


## **Table 5** 

1.5D test results for various limiters and reconstructed variables (AUSM[+] -up) 

( _2nd-order_ both in time and space) ( _κ_ = _–1_ MUSCL). 

|( _2nd-order_ both in time and space) ( _κ_= _–1_ MU|SCL).|
|---|---|
|Numerical Flux Functions (limiter, variable)|Total Shock-Robustness Scores|
|AUSM + -up (minmod, prm)|20|
|AUSM + -up (Van Albada, prm)|11|
|AUSM + -up (superbee, prm)|14|
|AUSM + -up (minmod, csv+p)|17|
|AUSM + -up (Van Albada, csv+p)|11|
|AUSM + -up (superbee, csv+p)|11|
|AUSM + -up (minmod, csv)|16|
|AUSM + -up (Van Albada, csv)|9|
|AUSM + -up (superbee, csv)|2|


## **Table 6** 

1.5D test results for various numerical flux functions, limiters, and reconstructed variables ( _2nd-order_ both in time and space) ( _κ_ = 1/3 MUSCL). 

|variables ( _2nd-order_ both in time and space) ( _κ_|= 1/3 MUSCL).|
|---|---|
|Numerical Flux Functions (limiter, variable)|Total Shock-Robustness Scores|
|AUSM + -up (minmod, prm, _κ_= 1/3)|20|
|AUSM + -up (Van Albada, prm, _κ_= 1/3)|16|
|AUSM + -up (Van Albada, csv+p, _κ_= 1/3)|15|
|SLAU2 (minmod, prm, _κ_= 1/3)|17|
|SLAU2 (Van Albada, prm, _κ_= 1/3)|10|
|SLAU2 (Van Albada, csv+p, _κ_= 1/3)|11|


is used. Use of density, momentum, and pressure (denoted as “csv+p”) falls in the middle. The full score marked by AUSM[+] - up is guaranteed only for the minmod-prm combination. 

## **Table 7** 

2D blunt-body flow conditions [5,7] ( _Rer_ : Reynolds number based on the radius). 

|_M_ ∞|8.1|
|---|---|
|_Re_ _r_|1.31 ×10 5|
|_P_ ∞ [Pa]|370.6|
|_T_ ∞ [K]|63.73|


|**Table** **8**||
|---|---|
|Type IV shock/shock interaction fow conditions [ 37 ]||
|( _Re_ _r_ : Reynolds number based on the radius).||
|_M_ ∞|8.03|
|_Re_ _r_|2.57 ×10 5|
|_P_ ∞ [Pa]|985.01|
|_T_ ∞ [K]|111.56|
|Shock angle [deg.]|18.1114|


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0006-17.png)


**Fig. 6.** Hypersonic, viscous, 2D blunt-body problem results (Mach number contours at 100,000 steps) of Van Albada set, (a) SLAU2, and (b) SLAU2 ( _g_ = 1). 

## _3.2. Viscous heating problems_ 

_3.1.1.3. Effects of MUSCL parameter._ In addition, the parameter _κ_ in MUSCL is changed from –1 to 1/3 (from 2ndto 3rd-order in 1D, smooth flows). The results in Table 6 show that 

- AUSM[+] -up with minmod-prm combination still achieved 20 points. This combination seems to have produced proper amount of dissipation. 

Thus, the reconstructed variables and the formal order of accuracy in space have as much impacts as the flux limiters on the shock-robustness. 

Therefore, we should aware that _each flux function has its favorite combination of limiter and reconstructed variables at each formal spatial accuracy_ , and such a combination can be sought by numerical experiments as done here. In the rest of the paper, out of many possible combinations, the minmod-prm combination with _κ_ = –1 (denoted as ‘minmod set’ here after) is employed as the default, and sometimes compared with Van Albada-(csv+p) with _κ_ = 1/3 (denoted as ‘Van Albada set’). We will then proceed to more realistic, 2D heating problems next. 

## _3.2.1. Blunt-body, hypersonic heating problem (Navier–Stokes Eqs.)_ 

Now we consider hypersonic heating on the blunt-body wall. The freestream conditions (Table 7) and the model radius ( _r_ = 20 mm) are the same as in [5,7]. The wall temperature is prescribed as _Tw_ = 300 K (isothermal wall condition), and the grid lines are clustered to the wall so that the cell Reynolds number (Reynolds number based on the minimum spacing _�_ min) _Recell_ = 1.3. The grid has 160 × 160 cells and shown in Fig. 3a as well as the coordinates, whereas the grid shown in Fig. 3b will be used later in the shock interaction case. 

The numerical fluxes compared here are Roe (E-fix), AUSM[+] - up, and AUSMPW+ (‘thin’ shock capturing), and Hänel and SLAU2 (‘broad’ shock capturing). Results using other fluxes are found in [34]. Spatially second-order at maximum is guaranteed by MUSCL reconstruction ( _κ_ = –1) for primitive variables with minmod limiter (minmod set) or _κ_ = 1/3 with Van Albada – (csv + p) (Van Albada set) for the inviscid term, while central difference is used for the viscous term. As for time integration, LU-SGS is used with CFL = 200, and the computations were conducted for 100,000 steps 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

140 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0007-02.png)


**Fig. 7.** Surface pressure and heating profiles of Van Albada set results, (a) SLAU2, and (b) SLAU2 (g = 1). 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0007-04.png)


**Fig. 8.** Total enthalpy and mass flux profiles in _x_ -direction of minmod set results, (a) Roe (E-fix), (b) Hänel, (c) AUSM[+] -up, (d) SLAU2, and (e) AUSMPW+. 

to achieve approximately three orders reduction of the density residual. 

The computed flowfields by the minmod set are displayed in Fig. 4, along with the corresponding surface pressure and heating profiles (standardized by stagnation values [35]) in Fig. 5. All the fluxes showed similar flowfields with slight differences near the 

shock, and no clear evidence of carbuncles is observed. Thus, at least for these given flow conditions and the grid, those methods selected are free from shock anomalies, and hence, effects of the shock thickness is not discussed here. Smooth surface pressure is obtained by any flux used, but as for surface heating, only Roe (Efix) showed asymmetric and wavy patterns; Hänel underpredicted 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

141 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0008-02.png)


**Fig. 9.** Total enthalpy and mass flux profiles in _x_ -direction of Van Albada set results, (a) SLAU2, and (b) SLAU2 ( _g_ = 1). 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0008-04.png)


**Fig. 10.** Type IV shock/shock interaction solutions (minmod set), a) Roe (E-fix), b) Hänel, c) AUSM[+] -up, d) SLAU2, and e) AUSMPW+. 

the values. These results are largely consistent with the results reported in [5,7] in which the Van Albada set was applied, except for SLAU2. 

The SLAU2 results using the Van Albada set (as in [7]) is shown in Figs. 6a, and 7a. These results demonstrate that the SLAU2 favors the minmod set. These confirm the 1.5D shock robustness results in Table 5, i.e., SLAU2 preferred the minmod set (17 points) to the Van Albada set (10 points). 

Now, SLAU2 flux is modified by fixing the function “ _g_ ” [in Eq. (A.2d) in Appendix B] as unity so that full upwinding is realized (Private communication with Dr. Eiji Shima, JAXA, Jan. 16, 2013) across the shock; in other words, the role of “ _g_ ” is tested here. The effect is evident in Figs. 6b and 7b. Although the flowfields look somewhat contaminated, surface pressure was not affected considerably, and the smoothness of heating profile was greatly improved even with Van Albada set. 

In order to get deeper insights, we extracted total enthalpy and ( _x_ -directional) mass flux along the centerline, as shown in Figs. 8 (minmod set) and 9 (Van Albada set). In Fig. 8, all the fluxes tested showed mass flux jump at the shock, prominent in SLAU2 (Fig. 8d); Roe (E-fix) showed both total enthalpy and mass flux jumps at the shock; the other fluxes were designed to keep the total enthalpy constancy and actually behaved so at the shock. 

When the Van Albada set was applied (Fig. 9), however, 

- SLAU2 showed huge deviations both in mass flux (79%) and total enthalpy (57%), and the total enthalpy behind the shock was 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0008-12.png)


**Fig. 11.** Type IV shock/shock interaction solutions (Van Albada set), (a) Roe (E-fix), and (b) AUSM[+] -up. 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

142 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0009-02.png)


**Fig. 12.** Type IV Shock/Shock Interaction problem (close-up views), Van Albada set, (a) Roe (E-fix), (b) Roe (E-fix) with streamlines, and (c) AUSM[+] -up. 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0009-04.png)


**Fig. 13.** Type IV Surface pressure and heating profiles (minmod set), (a) Roe (E-fix), (b) Hänel, (c) AUSM[+] -up, (d) SLAU2, and (e) AUSMPW+. 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 143 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0010-01.png)


**Fig. 14.** Type IV Surface heating profiles (minmod set) summary. 

slightly reduced (–2%) (Fig. 9a), whereas they were cured when _g_ = 1 (Fig. 9b). 

Once errors are introduced to the mass flux, they have impacts on all the variable-vector components used in AUSM family (i.e., mass, momentum, and total enthalpy conservation), as seen in Eqs. (A.1a), (A.1b) in Appendix B. This is why SLAU2 without modification showed both the total enthalpy and mass flux errors (Fig. 9a), and they were suppressed by setting _g_ = 1 (Fig. 9b), leading to better representation of surface heating profile (Fig. 7b). Note that this symptom did not appear when the (unmodified) SLAU2 and minmod limiter were combined, because the minmod limiter can suppress such spurious oscillations from the mass flux. Moreover, Hänel, AUSM[+] -up, and AUSMPW+ had been designed to keep the mass flux. Therefore, _preservation of mass flux, which has not drawn particular attentions yet, must be carefully taken into account_ when a flux function and a limiter are chosen or developed, specifically when hypersonic heating is concerned. Then, the following item should be added to the _“Hypersonic CFD Tips”_ in Section 1: 

(B-2) Mass flux preserving 

## _3.2.2. Hypersonic type IV shock/shock interaction and heating problem (Navier–Stokes Eqs.)_ 

Finally, methods discussed above are applied to another wellknown test involving a shock/shock interaction [36-38], classified as type IV according to Edney [15]. Although the type IV interaction is known to be unsteady [37], we conducted steady computations with CFL = 1000 for 100,000 timesteps because we are interested in only the final, time-averaged solutions. Nevertheless, the present case involves the most severe heating known to date, and hence, is considered to represent complex shockinteracting flows in reality (e.g., [39–41]). The computational grid consists of 320 (circumferential) × 480 (wall-normal) cells clustered to the wall ( _Recell_ = 1) but not specifically adapted to the shock location (Fig. 3b). The freestream conditions (Table 8) and the model radius ( _r_ = 38.1 mm) are the same as in [37]. The wall temperature is prescribed as _Tw_ = 294.44 K (isothermal wall condition). 

Figs. 10–12 show overviews and blowup views of selected results (pressure contours). From Roe (E-fix) results both in Figs. 10a and 11a (in which different limiter sets are used), the carbuncle is observed ahead of the cylinder nose, even when the type IV shock interaction is formed (This may be the first time to report the carbuncle/type-IV interaction). The other cases successfully reproduced the type IV interaction structure, including a supersonic jet emanating from a shock/shock interaction point, a jet shock near the wall, and resulting drastic pressure rise (Fig. 12c). Indeed, those results looked very similar, although SLAU2 needed to start the 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0010-09.png)


**Fig. 15.** Type IV surface pressure and heating profiles (Van Albada set), (a) Roe (Efix), and (b) AUSM[+] -up. 

computation from the SLAU result as the initial condition. Thus, again, it is not discussed here whether the shock is ‘thin’ or ‘broad.’ 

The carbuncle in Roe (E-fix) is more clearly observed from the enlarged view in Fig. 12a: Streamlines passing through the carbuncle region are affected by the deformed shock (Fig. 12b), although the key structure of the type IV is maintained. AUSM[+] -up, on the other hand, showed no carbuncle, as seen in Fig. 12c. From those results, it is demonstrated that _the carbuncle can appear even in such complex shock/shock interacting flows: There may be ‘hidden’ carbuncle(s) in 3D, further complex flow simulations_ (in which it is almost impossible to identify carbuncles by human eyes [39-41]), _unless a shock-robust method is carefully chosen and employed_ . 

Then, surface pressure and heating rates are compared for different methods and also with measured data in Figs. 13–15. The numerical and experimental heating values are standardized by the corresponding undisturbed stagnation values for each, as done by Thareja et al. in [38]. For the computed heating results, the final solutions (after 100,000 timesteps) as well as averaged values over the last 50,000 timesteps are shown. For each case, agreement between the final and averaged solutions is observed around a peak region ( _φ_ ≈ –20°), indicating fair convergence. At the region upward ( _φ >_ 30°), however, results except for Hänel showed some variations. In this region, unsteady, upward disturbance (which is beyond the scope of the present work) was already reported in unsteady simulations by Zhong [37]. Thus, we will limit our discussions to the fairly converged zone ( _φ <_ 0°) when comparing the computed and experimental results. 

As for pressure, all the computed results attained the peak value ( _p_ / _p_ 0 ≈ 8) at nearly the right position ( _φ_ ≈ –20°.). The peak location of heating ( _q_ / _q_ 0 ≈ 13 at, again, _φ_ ≈ –20°) was also reproduced by any method, but its value is scattered among them. When the minmod set is used, AUSM[+] -up (Fig. 13c) showed the closest peak value to the measured one, followed by Roe (E-fix) 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 

144 

(Fig. 13a), showing slightly smaller peaks. Hänel, as anticipated from the previous test (Fig. 5b), poorly underpredicted the heating (Fig. 13b). SLAU2 and AUSMPW+ also showed smaller heating than the measured data, but discrepancies are not as wide as that in Hänel. They are summarized in Fig. 14. 

For the cases of the Van Albada set (Fig. 15), AUSM[+] -up exhibited a much higher peak ( _q_ / _q_ 0 ≈ 20) (Fig. 15b), although still in reasonable agreement with the experimental data in its profile. In contrast, Roe (E-fix) showed a lower peak heating ( _q_ / _q_ 0≈9) (Fig. 15a), partly due to the stronger influence of the carbuncle as seen in Fig. 12a and b. Other fluxes behaved similarly to the cases of minmod set, and thus, omitted. 

The results and discussions above highlighted importance of the roles of limiters, formal accuracy, and reconstructed variables as well as those of flux functions. The best combination may vary depending on problems, but from the present results, the following methods can be recommended for hypersonic heating computations: SLAU2, AUSM[+] -up, or AUSMPW+ along with _κ_ = –1, minmod limited MUSCL interpolation for primitive variables (minmod set). AUSM[+] -up seems the most robust and accurate at this stage, but a further improved SLAU2 may substitute it after the mass flux conservation is enforced in a careful manner near future. 

## **4. Conclusions** 

We have surveyed the roles of flux functions, limiters, and reconstructed variables in problems related to shock anomalies (e.g., carbuncle phenomenon), and shock-interaction heating. From numerical tests, it has been revealed that the limiters and the reconstructed variables have great impacts on the solutions as much as flux functions do. In particular, flux functions having at most one intermediate cell at the captured shock show improved robustness against shock anomalies as the spatial accuracy increases (e.g., Roe, AUSM[+] -up, and AUSMPW+), whereas those containing a few cells to represent the shock tend to do the opposite (e.g., Hänel and SLAU2). Among many possible combinations, the following set showed acceptable performance in the present study: 

- SLAU2, AUSM[+] -up, or AUSMPW+ flux along with _k_ = –1, minmod-limited MUSCL interpolation for primitive variables 

These combinations achieved satisfactory robustness against shock anomalies, and once the shock was captured well, the resulting surface heating over a blunt body was also predicted accurately, even when severe shock-interactions were present. 

In addition, the following item B-2) has been newly added to the _“Hypersonic CFD Tips”_ that are now completed as: 

- (A) Robustness against shock anomalies (e.g., carbuncle phenomenon) 

- (B-1) Total enthalpy preserving 

- (B-2) Mass flux preserving 

- (C) (Economical) boundary-layer resolving 

In fact, 

- Heating errors in SLAU2 can be cured when the mass flux is conserved across shocks (which has not drawn particular attentions yet) either by a modification to the flux (g=1) or by carefully choosing a limiter and reconstructed variables. 

Thus, incorporation of the bullet B-2, as well as the formal spatial accuracy and the reconstructed variables, will be the immediate next modification to SLAU2. 

Finally, the Roe flux yielded the carbuncle even in the type IV shock/shock interacting flow. In such a complex flowfield, the carbuncle is hardly detected, specifically in 3D, and thus the need of shock-robust methods is confirmed. 

## **Acknowledgments** 

This work was partially supported by Japan Society for the Promotion of Science (JSPS) KAKENHI [Grant-in-Aid for Young Scientists (B)] Grant Number 25820409. Most of the work presented herein was conducted while the author was at Nagoya University, Japan. Eiji Shima and Taku Nonomura, JAXA, Japan, Philip L. Roe at University of Michigan, Ann Arbor, MI, Yoshiaki Nakamura at Nagoya University, Japan, and Meng–Sing Liou at NASA Glenn Research Center, Cleveland, OH, gave us valuable comments. We are grateful to all their cooperation. 

## **Appendix A. Laminar boundary-layer over flat plate (Navier–Stokes Eqs.)** 

A _M_ ∞ = 0.2 flow over a flat plate is solved as in [42] to investigate boundary-layer resolutions of the flux functions (Fig. A1). The computation was carried out for 50,000 time steps with CFL = 0.5 for each case. In most cases the density residual dropped at least three orders. The results showed that SLAU (representing SLAU2, AUSM[+] -up2, and SD-SLAU) and other most fluxes reproduced Blasius’ analytical velocity profile, whereas Hänel, one of notoriously dissipative solvers, did not. HLL-CPS-Zroe solution is close to Hänel, although this flux preserves contact discontinuity in 1D [30]. Thus, the performance of HLL-CPS-Zroe, developed very recently, is represented by Hänel in a large portion of the paper. 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0011-22.png)


**Fig. A1.** Computed laminar boundary-layers over flat plate (Uniform flow: Mach 0.2), _κ_ = 1/3, (csv+p). 

_K. Kitamura / Computers and Fluids 129 (2016) 134–145_ 145 

## **Appendix B. SLAU2 flux formulation** 

Liou has developed the AUSM-family numerical fluxes (e.g., [26]), commonly expressed as: 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-03.png)


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-04.png)


SLAU2 scheme [7], one of AUSM-family schemes, is now briefly explained. The mass flux is: 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-06.png)


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-07.png)


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-08.png)


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-09.png)


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-10.png)


where _g_ is a switching function to a fully upwind scheme at a strong expansion, and the speed of sound, being arithmetic mean of the both side values for this flux, is: 


![](images/2016_Kitamura_assessment_SLAU2_flux_limiters_hypersonic_heating.pdf-0012-12.png)


**References** 

- [1] Peery, K.M. and Imlay, S.T., “Blunt-body flow simulations,” AIAA Paper 88– 2904, 1988. 

- [2] Pandolfi M, D’Ambrosio D. Numerical instabilities in upwind methods: analysis and cures for the “carbuncle” phenomenon. J Comput Phys 2001;166(2):271– 301. 

- [3] Henderson, S.J. and Menart, J.A., “Grid study on blunt bodies with the carbuncle phenomenon,” AIAA Paper 2007–3904, 2007. 

- [4] Kitamura K, Roe P, Ismail F. Evaluation of Euler fluxes for hypersonic flow computations. AIAA J 2009;47(1):44–53. 

- [5] Kitamura K, Shima E, Nakamura Y, Roe P. Evaluation of Euler fluxes for hypersonic heating computations. AIAA J 2010;48(4):763–76. 

- [6] Kitamura K, Shima E, Roe P. Carbuncle phenomena and other shock anomalies in three dimensions. AIAA J 2012;50(12):2655–69. doi:10.2514/1.J051227. 

- [7] Kitamura K, Shima E. Towards shock-stable and accurate hypersonic heating computations: A new pressure flux for AUSM-family schemes. J Comput Phys 2013;245:62–83. doi:10.1016/j.jcp.2013.02.046. 

- [8] Barth, T.J., “Some notes on shock-resolving flux functions Part 1: Stationary characteristics,” NASA TM-101087, 1989. 

- [9] Chauvat Y, Moschetta JM, Gressier J. Shock wave numerical structure and the carbuncle phenomenon. Int J Numer Methods Fluids 2005;47(8-9):903–9. doi:10.1002/fld.916. 

- [10] Roe PL. Fluctuations and signals—A framework for numerical evolution problems. In: Morton KW, Baines MJ, editors. Numerical methods for fluid dynamics. New York: Academic Press; 1982. p. 232–6. 

- [11] Zaide DW, Roe PL. A second-order finite volume method that reduces numerical shockwave anomalies in one dimension. In: Proceedings of the 21st AIAA computational fluid dynamics conference; 2013. p. 2013–699. AIAA Paper. 

- [12] Roe PL. Characteristic-based schemes for the Euler equations. Ann Rev Fluid Mech 1986;18:337–65. 

- [13] Tu G, Zhao X, Mao M, Chen J, Deng X, Liu H. Evaluation of Euler fluxes by a high-order CFD scheme: shock instability. Int J Comput Fluid Dyn 2014;28(5):1–16. doi:10.1080/10618562.2014.911847. 

- [14] Coratekin T, van Keuk J, Ballmann J. Performance of upwind schemes and turbulence models in hypersonic flows. AIAA J 2004;42(5):945–57. doi:10.2514/1. 9588. 

- [15] Edney B. Anomalous heat transfer and pressure distributions on blunt bodies at hypersonic speeds in the presence of an impinging shock, 115, Stockholm, Sweden: Aeronautical Research Institute of Sweden; 1968. FFA Rept. 

- [16] Feng QU, Chao YAN, Jian YU, Di SUN. A study of parameter-free shock capturing upwind schemes on low speeds’ issues. Sci China Technol Sci 2014;57(6):1183–90. 

- [17] Chakravarthy Kalyana, Chakraborty Debasis. Modified SLAU2 scheme with enhanced shock stability. Comput Fluids 2014;100:176–84. 

- [18] Kitamura K, Liou M-S, Chang C-H. Extension and comparative study of ausmfamily schemes for compressible multiphase flow simulations. Commun Comput Phys 2014;16(3):632–74. 

- [19] Roe PL. Approximate Riemann solvers, parameter vectors, and difference schemes. J Comput Phys 1981;43:357–72. 

- [20] Harten A. High resolution schemes for hyperbolic conservation laws. J Comput Phys 1983;49:357–93. 

- [21] Harten A, Lax PD, Van Leer B. On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. SIAM Rev 1983;25(1):35–61. 

- [22] Einfeldt B. On Godunov-type methods for gas dynamics. SIAM J Numer Anal 1988;25(2):294–318. doi:10.1137/0725021. 

- [23] Toro EF, M Spruce, Speares W. Restoration of the contact surface in the HLL Riemann solver. Shock Waves 1994;4:25–34. 

- [24] Van Leer B. Flux vector splitting for the Euler equations. Lecture Notes in Physics, vol 170. Springer Berlin Heidelberg; 1982. p. 507–12. 

- [25] Hänel, D, Schwane, R., and Seider, G., “On the accuracy of upwind schemes for the solution of the Navier–Stokes equations,” AIAA Paper 1987–1105, 1987. 

- [26] Liou MS. A sequel to AUSM, Part II: AUSM[+] -up for All Speeds. J Comput Phys 2006;214:137–70. 

- [27] Shima E, Kitamura K. Parameter-free simple low-dissipation AUSM-family scheme for all speeds. AIAA J 2011;49:1693–709. doi:10.2514/1.55308. 

- [28] Kim SS, Kim C, Rho OH, Hong SK. Methods for the accurate computations of hypersonic flows I. AUSMPW+ scheme. J Comput Phys 2001;174:38–80. 

- [29] Shima E, Kitamura K. Multidimensional Numerical Noise from Captured Shockwave and Its Cure. AIAA J 2013;51(4):992–8. doi:10.2514/1.J052046. 

- [30] Mandal JC, Panwar V. Robust HLL-type Riemann solver capable of resolving contact discontinuity. Comput Fluids 2012;63:148–64. 

- [31] GC Zha, Bilgen E. Numerical solutions of Euler equations by using a new flux vector splitting scheme. Int J Numer Meth Fluids 1993;17:115–44. 

- [32] Van Leer B. Towards the ultimate conservative difference scheme. V. A secondorder sequel to Godunov’s Method. J Comput Phys 1979;32:101–36. 

- [33] Van Albada GD, Van Leer B, Roberts WW Jr. A comparative study of computational methods in cosmic gas dynamics. Astron Astrophys 1982;108:76–84. 

- [34] Kitamura K. A further survey of shock capturing methods on hypersonic heating issues. In: Proceedings of the 21st AIAA Computational Fluid Dynamics Conference, AIAA Paper 2013-2698„ San Diego, CA; Jun. 24-27, 2013. 

- [35] Fay JA, Riddell FR. Theory of stagnation point heat transfer in dissociated air. J Aeronaut Sci 1958;25:73–85. 

- [36] Wieting AR, Holden MS. Experimental shock-wave interference heating on a cylinder at mach 6 and 8. AIAA J 1989;27(11):1557–65. 

- [37] Zhong X. Application of essentially nonoscillatory schemes to unsteady hypersonic shock-shock interference heating problems. AIAA J 1994;32(8):1606–16. 

- [38] Thareja RR, Stewart JR, Hassan O, Morgan K, Peraire J. A point implicit unstructured grid solver for the Euler and Navier-Stokes equations. Int J Numer Methods Fluids 1989;9(4):405–25. 

- [39] Gnoffo, P., Buck, G., Moss, J., Nielsen, E., Berger, K., Jones, W.T., and Rubavsky, R., “Aerothermodynamic analyses of towed ballutes,” AIAA Paper 2006–3771, 2006. 

- [40] Esquivel A, Raga AC, Cantó J, Rodríguez-González A, López-Cámara D, Velázquez PF, De Colle F. Model of Mira’s Cometary Head/Tail Entering the Local Bubble. Astrophys J 2010;725:1466–75. doi:10.1088/0004-637X/725/2/1466. 

- [41] Kitamura K, Nonaka S, Kuzuu K, Aono J, Fujimoto K, Shima E. Numerical and Experimental Investigations of Epsilon Launch Vehicle Aerodynamics at Mach 1.5. J Spacecr Rocket 2013;50(4):896–916. doi:10.2514/1.A32284. 

- [42] Nishikawa H, Kitamura K. Very simple, carbuncle-free, boundary-layer resolving, rotated-Hybrid Riemann solvers. J Comput Phys 2008;227 pp. 2560–2581. 

