JOURNAL OF COMPUTATIONAL PHYSICS 100, 335354 (1992)


# A Continuum Method for Modeling Surface Tension*

J. U. BRACKBILL, D. B. KOTHE, AND C. ZEMACH

Theoretical Division, Los Alamos National Laboratory, Los Alamos, New Mexico, 87545

Received August 30, 1990; revised July 29, 1991

A new method for modeling surface tension effects on fluid motion has been developed. Interfaces between fluids of different properties, or “colors,” are represented as transition regions of finite thickness, across which the color variable varies continuously. At each point in the transition region, a force density is defined which is proportional to the curvature of the surface of constant color at that point. It is normalized so that the conventional description of surface tension on an interface is recovered when the ratio of local transition region thickness to local radius of curvature approaches zero. The continuum method eliminates the need for interface reconstruction, simplifies the calculation of surface tension, enables accurate modeling of twoand three-dimensional fluid flows driven by surface forces, and does not impose any modeling restrictions on the number, complexity, or dynamic evolution of fluid interfaces having surface tension. Computational results for twodimensional flows are given to illustrate the properties of the method.

,o 1992 Academrc Press. Inc

I. INTRODUCTION

Liquid surfaces are in a state of tension, as though they possessed an elastic skin, because fluid molecules at or near the surface experience uneven molecular forces of attraction [ 11. Since abrupt changes in molecular forces occur when fluid properties change discontinuously, surface tension is an inherent characteristic of material interfaces. Surface tension results in a microscopic, localized “surface force” that exerts itself on fluid elements at interfaces in both the normal and tangential directions. Fluid interfacial motion induced by surface tension plays a fundamental role in many natural and industrial phenomena. Examples can be found in the studies of capillarity [2, 31, low-gravity fluid flow [4, 51, hydrodynamic stability [6], surfactant behavior [7, 81, cavitation [9], and droplet dynamics [2] in clouds [lo] and in fuel sprays used in internal combustion engines [ 111. Detailed analyses of these processes typically involve the use of numerical models to aid in understanding the resulting nonlinear fluid flows. Previous

* This work, supported in part by the NASA Lewis Research Center under NASA Lewis Interagency # C-32008-K, was performed under the auspices of the United States Department of Energy by the Los Alamos National Laboratory under Contract W-7405-Eng-36.

methods have suffered, however, from difficulties in modeling topologically complex interfaces having surface tension. We present a numerical method for modeling surface tension that alleviates these interface topology constraints, which we call the continuum surface force (CSF) model. The model interprets surface tension as a continuous, three-dimensional effect across an interface, rather than as a boundary value condition on the interface. The motivation for the CSF model presented in this paper is the accurate solution of Laplace’s formula, Eq. (9) below, for surface pressures occurring at transient fluid interfaces of arbitrary and time-dependent topology. The CSF model appears applicable to a general class of fluid phenomena influenced by interfacial surface tension. It has been applied successfully to model incompressible fluid flow in lowgravity environments, capillarity, and droplet dynamics. Consider, first, the effect of surface tension on a fluid interface. The surface stress boundary condition at an interface between two fluids (labeled 1 and 2) is [ 121

where (T is the fluid surface tension coefficient (in units of force per unit length), pa is the pressure in fluid CI for c1= 1, 2, 7,ik is the viscous stress tensor, ii is the unit normal (into fluid 2) at the interface, and K is the local surface curvature, R;' + R; ‘, where R, and R, are the principal radii of curvature of the surface. (Because 6, in the formulation of Ref. [12], is defined only on the interfacial surface, it can only have a surface gradient; this would be, perhaps, more clearly indicated by replacing &/ax, in (1) by (6, -@i/J as/ax,.) The gradient along a direction normal to the interface, V N? is


![Equation](images/1992_brackbill_csf_eq001.png)

The surface tension, 0, may vary along the interface and

581/100/‘2-9

335 0021-9991/92 S5.00 Copyright 0 1992 by Academic Press, Inc. All rights of reproduction in any form reserved.

where ~1, is the molecular viscosity in fluid a and u is the II. THE CONTINUUM METHOD fluid velocity. The normal and tangential projections of (1) then lead to scalar pressure boundary conditions at the A. A Volume Reformulation of Surface Tension

interface as given by [ 12, 133: In its standard form, surface tension contributes a surface pressure (9) that is the normal force per unit interfacial area

for the normal direction. and

where K(x,) is the curvature, taken positive if the center of curvature is in fluid 2 (see Appendix), and fi(x,) is the unit

for a tangential direction. The surface derivative is normal to A at x,, assumed to point into fluid 2. Consider two fluids, fluid 1 and fluid 2, separated by an a -=i.V, as (7) interface at time t. The two fluids are distinguished by some characteristic function, c(x),

and the normal derivative is

1

Cl, in fluid 1;


![Equation](images/1992_brackbill_csf_eq002.png)

cc> = (Cl + c2)/2, at the interface,

that changes discontinuously at the interface. For example,

While the normal stress boundary condition can be satisfied one may wish to predict the motion of the interface between

at the interface between two fluids that are at rest, the two incompressible fluids that are distinguished by different

tangential stress boundary condition requires that the fluid densities, p L, p2. The points x, on the interface A are given

be in motion. It is evident from (5) that surface tension by

336 BRACKBILL, KOTHE, AND ZEMACH

its gradient tangent to the interface is defined using the In this paper we address the accurate modeling of the differential surface operator, V,, normal boundary condition for interfaces between inviscid incompressible fluids (p = 0), where the surface tension v,=v-v,. (3) coefficient is constant. This condition reduces to Laplace’s formula [ 121 for the surface pressure pS, the fluid pressure Projecting (1) along the unit normal fi and tangent i jump across an interface under surface tension [2], results in scalar boundary conditions for the fluid pressure in directions normal and tangent to the interface, respecps-p2-p, =cTIc. (9) tively. For example, when the fluid on both sides of the interface is incompressible, the viscous stress tensor is given Surface pressure is therefore proportional to the curvature by (K) of the interface. The higher pressure is in the fluid medium on the concave side of the interface, since surface tension results in a net normal force directed toward the center of curvature of the interface (see Appendix).

(5) A at points x, on A, ps(x,)= IF~~‘(x,)l, where Fii) is the normal force component of the total surface force, F,, = F’!” + F(‘). Here, we consider interfaces between inviscid flzds hating a constant surface tension coefficient, where F = F’“‘. The surface force per unit interfacial area can thsean be’kritten as


![Equation](images/1992_brackbill_csf_eq003.png)


![Equation](images/1992_brackbill_csf_eq004.png)

manifests itself in the normal direction as a force, OK, that drives fluid surfaces toward a minimal energy state characP(XJ = (P >. (12) terized by a configuration of minimum surface area. From (6), it follows that spatial variations in the surface tension Consider replacing the discontinuous characteristic funccoefficient along the interface (&/as) cause fluid to flow tion by a smooth variation of fluid color F(x) from c1 to c2 from regions of lower to higher surface tension [2]. Examover a distance of Lo(h), where h is a length comparable to ples of variable surface tension flows can be found in the resolution afforded by a computational mesh with Refs. [ 14181. spacing Ax. This replaces the boundary-value problem at

computational grid \

transition

region

Y

SURFACE TENSION 331

for an inviscid fluid in the presence of surface tension then becomes

fluid 2

‘interface

fluid 1


> **FIG. 1. Contours of the color function ? separate fluids with color values c1 and c2. The transition region (unshaded) has width h. Normals, given by ri = V?/V?l, are calculated at vertices of computational cells lying in the interface region. Surface tension forces, F,,, are calculated at cell centers from the divergence of A.**

the interface by an approximate continuous model and mimics the problem specification in a numerical calculation, where one specifies the values of c at grid points and interpolates between. Contours of constant values of the color shown in Fig. 1 illustrate the properties of the smoothed variation produced by interpolation. In Fig. 1, the interface is replaced by a transition region that is not aligned with the grid. Within the transition region, there are nested contours of Z(x), where c, < ? d c2, whose curvature K varies slowly from contour to contour when oh < 1.


# c Y(x) d’x = h3, (17) v

that Y have bounded support,


![Equation](images/1992_brackbill_csf_eq005.png)

that Y be differentiable, and that Y decrease monotonically with increasing 1 x /. An example of an interpolation function with these properties is the B-spline, which is described by de Boor [ 191. The interface where the fluid changes from fluid 1 to fluid 2 discontinuously is therefore replaced by a continuous transition. It is no longer appropriate to apply a pressure jump induced by surface tension at an interface. Rather, surface tension should act everywhere within the transition region.

The interpolation function is defined so that the mollified color, E(x), approaches the characteristic function, c(x), as the scale length h -+ 0,

lim i’(x) = c(x). (19) h - 0

Consider a volume force, F,,(x), that gives the correct surface tension force per unit interfacial area, F,,(x,), as h + 0,

Further, F is differentiable because Y is, and

v?(x) = $1 c(x’) VY(x -x’) d3x’, I/ lim s F,,(x) d3x = 1 F,,(x,) dA (13) h-0 A” AA

where the area integral is over the portion dA of the interface lying within the small volume of integration d V. The volume AV is constructed so that its edges are normal to the surface, and its thickness, h, is small compared with the radius of curvature of the interface A. In addition to (13), we also require that F,,(x) be localized so that it is zero outside the interface region:


### F,,(x) = 0 for Iti( (x-x,)1 3 h. (14)

Including F,, , the Lagrangean fluid momentum equation


![Equation](images/1992_brackbill_csf_eq006.png)

where p is the density, u the velocity, and p the pressure. To formulate the volume force, one begins by defining a mollified color, F(x), that varies smoothly over a thickness h across the interface by convolving the characteristic function, c(x), with an interpolation function, ,4”,


![Equation](images/1992_brackbill_csf_eq007.png)

We require a normalization for Y in terms of the positive constant h,


![Equation](images/1992_brackbill_csf_eq008.png)

where the volume of integration contains the support of Y. Using Gauss’ theorem and noting that c(x) is constant within each fluid, one can convert the volume integral to an integral over the interface A,

vi(x)=Fjri(x,)Y(x-x,)dA, (21) A

where [c] is the jump in color, [c] = c2 - c,. From the integral in (21), one computes a weighted mean of the surface normal. Since Y has bounded support, the portion of the surface contributing to the integral is Lo(h’). Define x,~ as the point on A from which the normal

338 BRACKBILL, KOTHE, AND ZEMACH

direction to A, fi(x,,,), passes through x. Then xSO is the B. Properties of F,,(x) surface point closest to x. The integral in (21) is, approximately, In summary, the volume force, F,,(x), has the following

lr properties: I 72 J ri(x,)Y(x-x,)dA 1. A The volume force in the transition region, where the color varies smoothly from c1 to c2, is designed to simulate 1 = 2 i(%o) s h 2

A

Y(x-x,)dA+Q z (( >> 3 (22) the surface pressure on the interface between the fluids. Thus, the line integral of F,,(x) across the transition region, e.g., from P, to P, in Fig. 1, is approximately equal to the where R is the radius of curvature of the surface at xSO. One conventional surface pressure (x, is the interface point on can bound the integral in (22) by the line P, P,):

1

h2 I Wx -x,) dA <9(x -x,,,). A

As h + 0, 9(x - x,,,) is zero everywhere but x = xSO, and d?(x)

the corresponding limit of the integral of V?(x) across the CI Mx) fi(x) ccl

interface is given by = mc(x,) fi(x,) for h>O. (29)

lim h-0 s 6(x,,) -W(x) dx= [cl. (24) 2. In the limit that the width of the transition region in a direction normal to the interface goes to zero (h + 0), the

Thus, the limit h -+ 0 of V?(x) can be written volume force becomes

lim VC(x)=A[c] S[ri.(x-x,)] =Vc(x). (25) lim F,,(x) = F,,(x) X%x,). (x -x,)1, (30) h-0 h-0

This delta function can be used to rewrite F,,(x,) as a which yields the conventional surface pressure at x, given

volume integral for h = 0: by (9).

= s F,,(x) b[fi(x,) . (x - xS)] d3x V

= s m(x) ii(x) cS[iz(x,). (x - x,)] d3x. (26) V

The delta function converts the integral of F,,(x) over a volume V containing the interface A to an integral over A of F,,(x,) evaluated at that surface. The integral relation in (26), an identity for discontinuous interfaces (h = 0), can be used to approximate interfaces having a finite thickness h when (25) is substituted for the delta function. Upon substituting (25) into (26), we find that I F,,(x,) dA = lim s Wx) d3X. 4x) ccl (27) A h-0 y

By comparing (27) with (13), we identify the volume force, F,,(x), for finite h as


![Equation](images/1992_brackbill_csf_eq009.png)

III. NUMERICAL IMPLEMENTATION

A. Choice of the Color Function

For cases of incompressible flow, there is a natural alternative to the definition of Z(a) by (16) in terms of an interpolation function. Instead we set


### S(x) = P(X) (31)

at grid points, with p(x) derived from the evolution equation (15). The volume force is still given by (28). The transition region thickness is then of the order of the grid spacing and, at points outside the transition region, F(x) has the values p, , p2 in fluids 1, 2, respectively. The interface between the fluids is given by the surface P(xs)=th +Pz)‘<P). One can multiply the integrand on the right side of (27) by a function g(x) = Z(x)/(c) without changing the value of the integral in the limit h + 0, since at the interface x =x, and g(xS) = 1. If, for incompressible flow, we use E(x) = p(x), then g(x) is given by


### g(x) = P(X)/(P), (32)

339

and the volume force in (28) when multiplied becomes

F,,(X) = OK(X) F$J

3UKrACC

by g(x),


![Equation](images/1992_brackbill_csf_eq010.png)

With this modification, fluid acceleration due to surface tension modeled as a volume force density depends only on density gradients, not on the value of the density itself. Thus, if F,,(x) in (33) is substituted into (15)


![Equation](images/1992_brackbill_csf_eq011.png)

When the acceleration due to surface tension is independent of the density, neighboring contours in the transition region tend to remain a constant distance apart under the action of surface tension. Denser fluid elements in the transition region experience the same acceleration as lighter fluid elements when g(x) is included in F,,(x). Otherwise, the interface tends to thicken when F,,(x) is directed toward the fluid having the smaller density, and to thin when F,,(x) is directed toward the fluid having the larger density.

B. Evaluation of Curvature

The curvature of a surface A at xs, K, is calculated from


![Equation](images/1992_brackbill_csf_eq012.png)

where ri is the unit normal to the surface. A derivation is included in an appendix for completeness (see also Ref. C6, p. 23, Eq. (5.411). In the CSF model, the interface is replaced by nested surfaces of constant color, whose normals are gradients of the mollified color function,


![Equation](images/1992_brackbill_csf_eq013.png)

The unit normal is thus


![Equation](images/1992_brackbill_csf_eq014.png)

It follows that the expression, u V?, which is needed to evaluate the surface volume force, is given by


![Equation](images/1992_brackbill_csf_eq015.png)

where we have used (36) and (37). Since Vi: is nonzero only in the transition region, the surface volume force also is nonzero only in the transition region.

C. Quadrature

Because the contributions to the surface tension force come from the small portion of the computation mesh in the neighborhood of the interface, difhculty in formulating sufficiently accurate finite difference expressions might be expected. It turns out that low-order approximations may be used, provided one begins with a form of the volume force that emphasizes the region of maximum gradient. This allows one to apply boundary conditions with no more difficulty than with other terms, such as pressure. It is straightforward to approximate the volume force density in (28). The expression K V? in (38) is the product of a first-order and second-order spatial derivative of c”. The coefficients e and [c] are parameters. Modeling surface tension requires some special considerations, since the effects of surface tension should be confined to the neighborhood of the interface. To simplify the application of boundary conditions and to localize the domain of dependence of the volume force, an approximation with compact support is sought. To maintain the integrity of the transition region, the volume force should not change sign along the radius of curvature. These requirements are met by the MAC [21] and ALE [22] formulations described below. Let us consider how to compute surface effects in the MAC and ALE methods. To model surface tension with the CSF model at interfaces separating incompressible fluids computed with either the MAC or ALE methods, the color function, as in (3 1 ), is chosen to be the fluid density, which resides at cell centers in both methods. The curvature K therefore also will be cell-centered. We choose to locate F,, at cell centers in the MAC method and at cell vertices in the ALE method. The normal vectors at cell centers must be interpolated from nearby cell vertices in the ALE method and cell faces in the MAC method. Let a two-dimensional computational domain be partitioned in Cartesian geometry into a regular, orthogonal mesh with mesh spacing Ax and Ay. The center of each cell is located at (x ,,,, -vi, j), where xi,, = i dx, 1 < if A’, and yi,, = j Ay, 1 < j 6 M. Cell vertices and faces are located at odd multiples of Ax/2 and Ay/2, i.e., at (x,, ,ir,, + ,,2,

-Vi+ 112,1+ 112 1 and (x,+ I,2,,2 yj+ ,iz. ,), respectively, where .Y;+ I,>,j = x,, i + Ax/2 and Y;,, + 1,2 = yip, + Ay/2. Densities pi,, and pressures pi, j are stored at cell centers, (xi, j, y,,,). Since cell-centered density is used as the color function, set

c",,j= Pi.,.

1. ALE-like Scheme: Indirect Differentiation qf the Unit Normal

From (28) and (38) the volume force density at a vertex is given by


![Equation](images/1992_brackbill_csf_eq016.png)

340 BRACKBILL, KOTHE, AND ZEMACH

The curvature in (35) also can be written


![Equation](images/1992_brackbill_csf_eq017.png)

Finite difference approximations to (40) result in division by the arithmetic mean of Inl. Similar approximations to (35) result in division by the harmonic mean of Inl. As a result, the principal contributions to (35) come from the edges of the transition region rather than the center. Thus, numerical approximations to (40) for which the principal contributions are where the gradient of the color is maximum, give better results in practice. Vertex-centered normal vectors are obtained by differentiating the color function in the-four surrounding cells. For example, the normal vector at the top right vertex of cell (i, j ) is given by

~;+I,j+C"i+l,j+r-~i.j-~,,j+I

2Ax 1

c";,j+~+~;+l,j+l-cli,j-zli+l,j


![Equation](images/1992_brackbill_csf_eq018.png)

Other vertex-centered normal vectors can be found in a similar fashion by translating the i andj indices in the above expression. The curvature in (40) is calculated at cell centers. The divergence of n for cell (i, j) is calculated from the vertex-centered normals and is given by

= & [In, if l/2,, + l/2 + ?xi+ l/2.,- l/2

-n.~i~,~2,,,~f2--n,i~,i2,j-1/2 1

1

+- Cnvi+ l/2,/+ l/2 + ny i1/2,j+ l/2 2Ay


![Equation](images/1992_brackbill_csf_eq019.png)

The derivative of the magnitude of the normal vector, (nl, in the direction of the cell-centered unit normal, n, j, is given

by


# ( > 3L.v InI

Ini.jl

where the cell-centered normal is the average of vertex normals,

~i.i=~(~i+1/2,~+1/2+~i+1~2,,~1/2


![Equation](images/1992_brackbill_csf_eq020.png)

Vertex-centered values of F,, are needed for the computation of fluid acceleration due to surface tension in the ALE method, since the ALE acceleration is calculated at cell vertices. The required vertex-centered values are obtained by interpolating the curvature, evaluated using (42)(44), from the four neighboring cell-centered values, and then multiplying by the vertex normal in (41).

2. MAC-like Scheme: Direct Differentiation of the Unit Normal

In the MAC scheme [21], the xand y-components of the normals reside at cell faces on lines of constant i and j, respectively, in direct correspondence to the xand y-components of the velocity field. The components of the normal are not co-located as in the ALE scheme, but are instead located on separate faces of cell (i,j), with the x-component at the right face, (xi+ ,,2,,, Y~,~), and the y-component at the top face, (xi, j, yi,, + ,,,2). The cell-centered normal vector, n,,,, is computed from linear interpolation of nearby face-centered components,

“i,,=~.t(n.~i+,,2.,+n.~i~1/2,,}


![Equation](images/1992_brackbill_csf_eq021.png)

and a cell-centered curvature, (V . fi)i, j, is derived from the divergence of face-centered unit normals,

AY lni.,+1,21 - lni.,-1O [

n,, i. , + 1/2 n., i,,- 112 1 ’


![Equation](images/1992_brackbill_csf_eq022.png)

Face-centered normal vector magnitudes are given by, for example,

/ni+,,2,jl=(n2,i+,,2.,+nti+~,‘2.,)1’2

for the right face (i+ i,j) of cell (i,j), and


![Equation](images/1992_brackbill_csf_eq023.png)


![Equation](images/1992_brackbill_csf_eq024.png)

for the top face (i, j + f) of cell (i,j). Components of the normal vector are trivial in the xand y-directions at faces of constant i and j, respectively. For example,


![Equation](images/1992_brackbill_csf_eq025.png)

SURFACE

is the x-component at the right face (i + i,j) of cell (i, j),

i:

n,i,j+112=

l,J+ 1 -‘i,J


![Equation](images/1992_brackbill_csf_eq026.png)

is the y-component at the top face (i, j + 1) of cell (i,j). Other components must be obtained by interpolation, such as in computing the x-component at faces of constant j, where

n,, j+ 1,2 = i(nri+ 1/2,j+ si+ 1/2,/+ I


![Equation](images/1992_brackbill_csf_eq027.png)

is the x-component at the top face (i, j + $) of cell (i, j). Similarly, the y-component at faces of constant i must also be interpolated, such as the y-component at the right face (i+ $,j) of cell (i,j):

n,.,+,i2,,=~(n.,i,,+1,2+n,i+,,J+1,2


![Equation](images/1992_brackbill_csf_eq028.png)

Examination of the example finite difference expressions in (45)-(52) reveals that a MAC-based, cell-centered formulation of the volume force density in (28) can be computed solely from normal vector components n, and ny stored at the right and top faces of each computational cell, respectively. The computational storage and expense is the same as in the ALE scheme, with only one, mesh-wide vector array (n, j) required. Face-centered values of F,, are needed for the computation of fluid acceleration due to surface tension in the MAC method, since MAC acceleration is calculated at cell faces. The required face-centered values are obtained by interpolating from the two nearest cell-centered values, F,, I, j.

D. Boundary Conditions: Wall Adhesion

The effects of wall adhesion at fluid interfaces in contact with rigid boundaries in equilibrium can be estimated easily within the framework of the CSF model in terms of tIeq, the equilibrium contact angle between the fluid and wall. The angle Beq is called the static contact angle because it is experimentally measured when the fluid is at rest. The equilibrium contact angle is not simply a material property of the fluid. It depends also on the smoothness and geometry of the wall [23]. The normal to the interface at points x, on the wall is

n = n,,,, cos O,, + ri, sin Beq, (53)

where h, lies in the wall and is normal to the contact line

TENSION 341

between the interface and the wall at x,, and fiwal, is the unit wall normal directed into the wall. The unit normal ri, is computed using (36) with the fluid color I? reflected at the wall. Wall adhesion boundary conditions are more complex when the contact line is in motion, i.e., when the fluid in contact with the wall is moving relative to the wall [23]. The equilibrium wall adhesion boundary condition in (53) may have to be generalized by replacing O,, with a dynamic contact angle, edr that depends on conditions. locai fluid and wall

E. Stability

In the CSF model, the Lagrangian velocity field of fluid elements in the presence of surface tension is advanced in time with the momentum equation in (15). We consider the stability of a first-order, time discretization of ( 15) by which the velocity field u is advanced from time t = n At to t=(n+l)At,


![Equation](images/1992_brackbill_csf_eq029.png)

where the density, p, and volume force density, F,,, are taken at the old time (n) and pressure, p, is taken at the advanced time (n + 1). The finite difference expression in (54) can be written as

At U n+‘=fiLyvpn+l,

P

where the tilde velocity, V, is equal to the old velocity II”, plus the change in velocity resulting from surface tension forces,


# +f+&F”

P" sv'

Upon taking the divergence of (55), one finds that


![Equation](images/1992_brackbill_csf_eq030.png)


![Equation](images/1992_brackbill_csf_eq031.png)

where we have made use of the requirement that for incompressible flow the advanced-time velocity field must be solenoidal:


![Equation](images/1992_brackbill_csf_eq032.png)

From (57), it is evident that a solution of a Poisson equation for the pressure gives the advanced-time pressure field

342 BRACKBILL, KOTHE, AND ZEMACH

in the presence of surface tension. This is an equation whose solution is already well within the framework of the MAC method [21] and many other incompressible flow methods. The CSF model for surface tension therefore does not place new or special requirements on conventional algorithms used for finding the pressure field in incompressible flow. Surface tension is treated in the pressure solution no different from any other body force, i.e., gravity. There is a stability condition in solving for the advancedtime pressure field with (56) and (57). The explicit treatment of surface tension is stable when the time step resolves the propagation of capillary waves,


![Equation](images/1992_brackbill_csf_eq033.png)

where c( is the capillary wave phase velocity [24]:

ok [ 1


![Equation](images/1992_brackbill_csf_eq034.png)

The phase velocity in (60) of a capillary wave on an interface depends upon the wavenumber k, the parameter 0, and the fluid densities, p, and p2, on both sides of the interface. The conservative value of $ on the RHS of (59) guards against the case of two oppositely-moving capillary waves entering the same cell from opposite sides. The maximum allowed time step can be estimated using the maximum phase velocity. From (60), this occurs for the wavenumber k Inax = n/Ax in a finite-difference scheme, corresponding to the minimum resolvable wavelength 2Ax, where Ax is the grid spacing. Upon substituting (60) into (59) with k = k,,, and using (p) = i(p, + p2), the result is


![Equation](images/1992_brackbill_csf_eq035.png)

for the time step constraint imposed by an explicit treatment of surface tension. The surface tension time step constraint in (61) can limit Ar to small values, since At, cc (Ax)~‘~. An implicit treatment of surface tension would remove this constraint.

IV. NUMERICAL RESULTS

We now present the results for several standard static and dynamic problems with surface tension to illustrate the flexibility and accuracy of the CSF model. These examples supplement further comparison with theory and other numerical models in Ref. [27]. The application of the CSF model to a liquid drop problem as implemented in a VOF (volume-of-fluid) code [ 271 is illustrated by an equilibrium and oscillating drop calculation. (In this calculation, inter-

face reconstruction using the VOF method [26] maintains the integrity of the interface.) The application of the CSF model to the motion of a Rayleigh-Taylor unstable interface with surface tension as implemented in an ALE code [3 1 ] is described. (In this calculation, there is no interface reconstruction.) The imposition of the contact angle boundary condition is illustrated by two initial, boundary-value problems at extreme contact angles. Finally, a low-gravity spacecraft propellant reorientation and mixing problem demonstrates that CSF can model surface tension at fluid interfaces with complex topology. This illustrates a capability to follow automatically the breakup of interfaces that has been absent from other numerical methods [20].

A. Equilibrium Rod

In the absence of viscous, gravitational, or other external forces, surface tension causes a static liquid drop to become spherical. Laplace’s formula for an infinite cylinder surrounded by a background fluid at zero pressure, (9) gives the internal drop pressure, pdrop, to be


![Equation](images/1992_brackbill_csf_eq036.png)

where R is the drop radius. Results from the CSF model in Cartesian geometry using a two-dimensional 6 x 6 (cm) computational domain are compared with (62). We use a regular, orthogonal grid with uniform mesh spacing Ax and Ay, which partitions the domain into either a 30 x 30 low resolution (Ax = Ay = 0.20 cm) or a 60 x 60 high resolution (Ax = Ay =O.lO cm) mesh. A fluid drop with radius R = 2 (cm), density p = 1 (g/cm’), background density p = 4 (g/cm3), and surface tension coefficient CJ = 23.61 (dynes/cm), is centered at the point (3, 3). From Laplace’s formula, (62), the pressure jump is 11.805 (dynes/cm2). This value is compared with the mean computed drop pressure obtained with the CSF model, (p), defined as

where the sum is over the N, computational cells lying within the drop that have density p > 0.99. From the mean computed drop pressure in (63), a mean drop curvature, (K ), can be computed as


![Equation](images/1992_brackbill_csf_eq037.png)

and compared with the prescribed radius. Another measure of the relative error between the theoretical and computed drop pressure is given by the rms error,


# L c;j= 1 (P,,, - &k,p)2 ‘I2 2


![Equation](images/1992_brackbill_csf_eq038.png)

SUKPALC -- *-- ’ -- TENSION 343

TABLE I

Computed Mean Drop Pressure (p) and Mean Square Pressure Error as a Function of Mesh Spacing (R/Ax) and Finite Difference Scheme Defined in the Text

R/Ax <P YPdrop Mean square error

Unsmoothed ALE (P = p)

10 10* 20

0.996 3.09 x 1o-2 0.957 1.01 x 10-l 0.982 2.84 x lo-*

Smoothed ALE (c”= 0)

10 20 1.034 5.56 x lo-* 1.016 2.82 x lo-*

Unsmoothed MAC (F = p)

10 20 20’

2.938 1.97 x loo 4.68 1 3.73 x loo 1.230 2.75 x 10-l

Smoothed MAC (f = fi)

10 1.153 1.65 x 10-l 20 1.043 5.14 x lo-*

Note. The equilibrium planar drop, with density p= 1 and radius R = 2, has a surface tension coefficient e = 23.61 at the interface with a background fluid of density p = i. The asterisk (*) denotes a result using Eq. (35) for the curvature instead of Eq. (40). The dagger (+) denotes a result using Eq. (28) for the volume force instead of Eq. (33).

A

One consequence of the formulation of the CSF model, in which the volume force F,, is given by anK/[c], is that one can use a E to calculate K different from the c” used to calculate n. The coefficient n in (39) has its greatest magnitude in the center of the transition and falls to zero outside. It emphasizes the values of K that are local to the fluid interface. Thus, one should use data to calculate an n that is as peaked as possible in the interface to localize the surface forces. On the other hand, a smoother Z, i.e., one that changes more slowly through the interface, can be used to obtain a smoother K. The term nkwill then still be localized to the neighborhood of the interface. The drop results in Table I and Fig. 227 are obtained with n computed using c”= p, and K computed using either ? = p or ? = p, a smoothed density. The smoothed density is computed by convolving the density with a B-spline of degree I [19] #‘)(1X’-x1.h) with I= 2, where Y(l) # 0 only for Ix’-;1 < (I+ l)h/; =;h/2. At mesh points, the smoothed density is given by

,8i,j= i pi,,j.Y(‘)(x;,,j~-x,,j; h)

i’,.i’= 1


![Equation](images/1992_brackbill_csf_eq039.png)

where pi,j is the value of p at xi,j = ,-Z-i Ax + jj Ay and the sum gathers contributions from the nine values of pi,i (in 2D) within the support of Yc2). Table I lists computed results obtained with the CSF model, using for the volume force F,, at the drop interface.

B


> **FIG. 2. (A) Contours of constant density (p) mark the transition between a drop and a background fluid. With surface tension, o = 23.61, the surface tension force vectors (F,,) shown in (B) are calculated with the CSF model using the density as the color F. The surface forces in (B) are largest near the (p ) = 0.75 contour line. The computation is performed using the ALE scheme on a 30 x 30 mesh with Ax = Ay = 0.2.**

344 BBACKBILL, KOTHE, AND ZEMACH

The results consist of mean drop pressure (p) and mean square error as a function of mesh spacing (expressed as R/Ax), the mollified color ? used for the calculation of rc, and the finite difference scheme used for F,,, i.e., either MAC or ALE as discussed in Section IILC. Several important points need to be made about the results in Table I. First, it appears that the MAC scheme, which differentiates ri directly, leads in general to greater rms errors than the ALE scheme, which differentiates ri indirectly using (40). Normalizing n to ri before differentiation introduces the possibility of order unity errors in computing V. ri, as seen by the relatively high errors for the MAC scheme with c” = p. Second, using ? = i?, the smoothed density, decreases the rms error in almost all cases. Computed values of ( p ) and (K ) in those cases are all within 5 % of theory, with some high resolution cases being within less than 1% of theory. Third, there are large errors with the

MAC scheme when moderate to coarse zoning is used to resolve the interface. Fourth, the inclusion of the function g(x) = p(x)/(p) in the expression (33) for F,, appears to result in a small increase in error in the ALE scheme, but a large increase in poorly resolved MAC calculations. The quantities listed in Table I represent results generated in the first computational cycle, could easily (and typically do) change by a few percent when tabulated after a few hundred cycles. Each finite difference scheme computes a slightly different drop curvature because the finite difference approximation to K, derived from identical density data, is not the same for the MAC and ALE schemes. The value of K appears to depend on small displacements of the interface relative to the mesh. To reach equilibrium, the drop adjusts its shape until the forces are balanced. The drop results are illustrated in Fig. 2-7. The density contours in Fig. 2A illustrate the finite thickness interface


> **FIG. 3. The smoothness of curvature, rc, depends upon the smoothness of the color function, P. Choosing f =p (the drop density), as in (A), results in the curvature shown in (C); choosing P=p (the smoothed drop density), as in (B), results in the smoother curvature shown in (D). The computed curvature is plotted in (C) and (D) as a product of curvature and radius, since KR = 1 for a planar drop. The computation is performed using the ALE scheme on a 30 x 30 mesh with Ax = Ay = 0.2.**

SURFACE TENSION 345

surrounding the drop, resulting in surface forces that are largest near the (p) contour line shown in Fig. 2B. Surface plots of i: = p and c” = p are shown in Figs. 3A and B, respectively. The smoothing of the density increases the thickness of the transition region between the drop and background fluid to about 4 of the drop radius. The computed KR, which should be equal to 1, is shown in Fig. 3C and D for the ALE scheme. The smoother K in Fig. 3D results when a smoothed ,8 is used as the color function, giving a result of KR nearly equal to 1 everywhere except along the diagonals to the mesh where relative errors are 20 %. Without smoothing p, the errors in K are larger. Figures 3-7 allow one to compare the computed pressure and the surface force within the interface region for the MAC and ALE schemes using p and D as the color function. Without smoothing, the surface forces vary as shown in Fig. 4A, but the resulting pressure in Fig. 4C is smooth nearly everywhere. Smoothing p yields slightly smoother pressure, Fig. 4D. One can explain the insensitivity of the

pressure to variations in the surface force by noting that the force is a source term in the Poisson’s equation from which the pressure is computed. Solving Poisson’s equation for the pressure requires integrating the surface force twice, which considerably reduces the effect of small scale variations in its value. Figures 5 (with a p curvature color function) and 6 (with a p curvature color function) illustrate the effect of smoothing on the variation of ?, Inl, K, and IF,, 1 along the drop equator (y = 3). Drop curvature in Fig. 6C is smoother than the curvature in Fig. 5C. The curvature K in Fig. 6C significantly varies (KR is constant) because the drop interface thickness is comparable to the radius of the drop. The normals in Fig. 5B and 6B, however, are the same, since the normal is defined using p in both cases. The net surface forces in Figs. 5D and 6D are virtually identical, because the normal vector weights more heavily those values of K near the interface. The results with MAC suggest that smoothing is


> **FIG. 4. Whether the color is smoothed affects the surface tension forces IF,, 1, as shown in (A) and (B) corresponding to Figs. 3A and B. The drop pressure in (C) and (D), however, is not sensitive to smoothing. (The drop pressure should be o/R = 11.805.) The computation is performed using the ALE scheme on a 30 x 30 mesh with Ax = Ay = 0.2.**

346

1.0

0' 75 0

0.5

1 .o 5

E

b Z

0.0

a,

5 -+J u 0.5 > L

(3

0.0

1.5

8 L 0 LL

0.0

BRACKBILL. KOTHE. AND ZEMACH

r I I I I I I

* 3 2 1 0 1 2 3 Drop Radius


> **FIG. 5. Plots of (A) the mollified color function (E=p); (B) the magnitude of the normal vector (ml); (C) the curvature (K); and (D) the magnitude of the surface tension force (IF,, I) in a cut along the midline of the drop shown in Fig. 2. The computation is performed using the ALE scheme on a 30 x 30 mesh with Ax = Ay = 0.2.**

necessary to obtain an accurate, uniform drop pressure. Without smoothing, relative errors in the curvature listed in Table I are nearly lOO%, but with smoothing, the computed surface forces and drop pressure, shown in Figs. 7A and 7C, look almost identical to the results obtained with the ALE scheme. Convergence of the CFS model to theory is demonstrated by the results in Figs. 7B and D for the drop computed with fi and the ALE scheme on a 60 x 60 mesh. Increasing resolution localizes the surface forces and yields a uniform ring around the circumference of the drop. The pressure in Fig. 6D is apparently uniform with a mean value of ( p ) = 11.991, compared with a theoretical value of 11.805. The mesh interval is 5 % of the drop radius.

B. Nonequilibrium Rod

When a rod or cylindrical drop is deformed, capillary waves are induced that cause the drop surface to oscillate about its equilibrium shape. In the numerical calculations,

1.0

6 z 0

0.5

1.0 _

g

Z

0.0

al

5 -4-J 0 0.5 >

5 0

0.0

1.5

E

b LL


# 0.0 e

3 2 1 0 1 2 3 Drop Radius


> **FIG. 6. Plots of (A) the mollified color function (?=p); (B) the magnitude of the normal vector (ml); (C) the curvature (K); and (D) the magnitude of the surface tension force (IF,, I) in a cut along the midline of the drop shown in Fig. 2. The computation is performed using the ALE scheme on a 30 x 30 mesh with Ax = Ay = 0.2.**

the oscillations damp because of numerical dissipation. This behavior is observed in a numerical calculation when an initially square drop responds to unbalanced surface tension forces. The results are computed with a 2D, incompressible hydrocode [27] using the VOF method to describe the free surface [26]. On a 30 x 30 grid with Ax = Ay = 0.25 cm, a square ethanol drop with p = 0.79788 g/cm’ and cr = 23.61 dynes/cm evolves as shown in Fig. 8. At a sequence of times, t = 0, 0.05, 0.1, 0.2, 1.0, and 2.0 s, the oscillations of the surface of the drop are apparent. At t = 2.0 s, the drop is nearly circular in cross section, and the amplitude of the capillary waves is a fraction of its maximum value. The small, residual deviation from the equilibrium circular shape is due to numerical error (R/Ax - 6). Large deviations in the curvature that were apparent in the equilibrium calculation result in small, bounded variations in the equilibrium shape. Disturbingly, despite the dissipation in the numerical calculations, the kinetic energy does not decay to zero even after long times. The kinetic energy oscillates with constant

347 SURFACE TENSION


> **FIG. 7. Computed planar drop results, analogous to those of Fig. 4, using the MAC scheme in (A) and (C) and the ALE scheme on a 60 x 60 mesh with Ax = Ay = 0.1 in (B) and (D). In both cases the smoothed density, p, is used for the curvature calculation.**


> **FIG. 8. Surface force vectors are shown on an initially-square ethanol drop in zero gravity at times of (A) 0.0, (B) 0.05, (C) 0.10, (D) 0.20, (E) 1.0, and (F) 2.0 s. The initially square shape of the drop results in very strong surface forces at the high-curvature corners, setting the drop into oscillation. The period of oscillation is approximately 0.40. Numerical dissipation eventually damps the oscillation, causing the drop to approach an equilibrium spherical shape**

amplitude for many wave transit times. Since energy is dissipated, there must be a source that supplies enough energy to sustain the oscillations at constant amplitude. Animations of computational results indicate that the VOF treatment itself provides the driving force. One can see in the animations that small displacements of the surface cause discontinuous changes in the location and orientation of the interface with VOF. These occur in phase with the oscillations of the surface and provide a small impulse through the surface tension to sustain the oscillations.

C. Rayleigh-Taylor Instability

When a heavy fluid is supported against gravity by a light fluid, a Rayleigh-Taylor (R-T) instability develops in which perturbations of the interface grow exponentially in time as exp(nt) for small amplitudes. With surface tension, the growth rate n is given by [6]

k20

dP I + P2).


![Equation](images/1992_brackbill_csf_eq040.png)

581/100/2-10

348 BRACKBILL, KOTHE, AND ZEMACH

where k is the wave number of the perturbation, g is the gravitational acceleration perpendicular to the interface, and A = (pz - p,)/(pi + p2). From (67), one can compute a critical surface tension, oc, for which n* = 0. Following Daly [29], a stability parameter @ can be defined


![Equation](images/1992_brackbill_csf_eq041.png)

such that when @ > 1, n is imaginary and perturbations of the interface oscillate. In the results that follow, we express time and velocity in units of r and gr, respectively, where z is the R-T growth time, n ~ ‘, for CJ = 0. The Atwood number is A = 0.6. The instability is initiated with a perturbation of the form 0.035 cos(kx) given to the vertical velocity component at the interface. The wavenumber k is set equal to $. The calculations shown in Fig. 9 are performed on a uniform, rectilinear 30 x 90 zone mesh with PLUTO, an ALE code for flow in two dimensions [31]. The boundary conditions for the pressure equation which yields the incompressible, inviscid flow solution are Neumann on all boundaries. In PLUTO. there is no interface tracking or


# $

A

D

:f

B

E F

C


## 1


> **FIG. 9. The Rayleigh-Taylor unstable interface between a heavy and light fluid is shown. The deformation of the interface, defined as the (p ) contour line, is shown at a time of 10 for the stability parameter @ equal to (A) 0.0, (B) 0.29, (C) 0.58, (D) 0.88, (E) 1.02, and (F) 1.17.**

reconstruction as there is in VOF [26]. Instead, the density gradient in the transition region is convectively transported similarly to all other dependent variables using a third order, monotonicity-preserving scheme developed by Meltz [32]. Consequently, there is some numerical diffusion of the interface. Comparison of the computed R-T growth rates with theory provides a delicate test of the accuracy of the CSF model. One can ask whether perturbations are stable when @ > 1. Stability requires that the surface tension force counteract the gravitational force in detail. In Fig. 9, the interface, defined as the (p) contour, is depicted at a time of 10 for values of @ equal to 0.0, 0.29, 0.58, 0.88, 1.02, and 1.17 on a computation mesh with 30 zones in the x-direction. As @ increases, the deformation decreases. With @ = 0.29, the bubble and spike that are prominent with @ =0 have not yet developed. With Cp = 1.17, there is no apparent deformation of the interface. However, the numerically computed linear growth rate as shown in Fig. 10 is not zero even for @ > 1. The @ = 1.17 growth rate, as computed from the variation of the kinetic energy with time, is still 10% of the value with @ = 0. For comparison, results with 15 zones in the x direction for @ = 0.58 and @ = 1.17 are shown in Fig. 10. The results with the coarser grid are less accurate. Three additional points in Fig. 10 correspond to calculations in which diffusion is added to the mass continuity equation. Adding mass diffusion to the case @ = 1.17 with

1.00

075

c

2

A?

5 0.50 3

2 c7 c I w 0.25

0.00 0.25 0.50 0.75 1.00 1.25 Stability Parameter (G)


> **FIG. 10. Theoretical Rayleigh-Taylor instability growth rate curve (line) and computed growth rates (points) are plotted as a function of the stability parameter @. Points designated with diamonds and squares correspond to 15 and 30 zones along a wavelength, respectively. The solid diamond points, which result from adding a mass diffision term in the continuity equation, show that diffusion is stabilizing for @ < 1 and destabilizing for @ > 1.**

SURFACE TENSION 349


### B


> **FIG. 11. (A) Normal vectors n, (B) surface forces F,,, and (C) pressure contours after one computational cycle in a 4-cm deep pool of water which has been given given a wall adhesion boundary condition of 0,, = 5” on the cylindrical tank walls. The computation is performed on a 20 x 40 mesh with Ar = AZ = 0.25 cm.**

diffusivity /3 = 9.0 x 10P4g2r3 causes the growth rate to increase from 0.18 to 0.25. (Doubling the diffusivity to b = 1.8 x 10P3g2r3 increases the growth rate further, but by a smaller amount.) Adding mass diffusion to the case @ = 0.58 with /I = 6.0 x 10-3g2r3 causes the growth rate to decrease from 0.54 to 0.46. Linear dispersion theory predicts that mass diffusion decreases the growth rate for @ < 1 and increases the growth rate for @ > 1 [33], just as observed in the numerical calculations. One can conjecture that mass diffusion is the cause for the unstable flow for @ > 1. Other results not shown indicate how this may occur. There is persistent circulating flow, even for @ > 1. This causes numerical diffusion in the calculation of convective transport. Diffusion of the heavier fluid downward is a source of free energy that drives the flow, just as bulk motion of the heavier fluid drives the flow in the R-T instability in the absence of diffusion.

D. Flow Induced by Wall Adhesion

As an example of the effects of wall adhesion computed with the CSF model, consider a shallow pool of water located at the bottom of a cylindrical tank. Assume that the water interface wants to attain a specified contact angle 8,, with the tank wall, different from the initial angle of 90” (a horizontal interface). Two different cases are computed in Fig. 11-13, one in which the water wets the wall (0,, < 90”), and one in which the water does not wet the wall (0,, > 90”). The results are computed in cylindrical geometry on a 20 x 40 mesh (dr = AZ = 0.25 cm) with a 2D, incompressible hydrocode [27] using the VOF method to describe the free surface [26]. For the wetting case, the pool of water is 4 cm deep and 0,, = 5”. For the non-wetting case, the pool of water is 2 cm deep and Be4 = 175”. All external forces such as gravity are set to zero. Figure 11 displays plots of n, F,,, and pressure contours

after one computational cycle for 8,, = 5”. The wall adhesion boundary condition of (53) is applied to the normal vectors n along the tank wall in Fig. llA, resulting in the upward surface forces F,, at the identical location as shown in Fig. 11B. These wall adhesion forces, the only forces in the calculation, in turn generate a pressure field in the proximity of the wall (Fig. 11C). From the sequence of times in Fig. 12, t = 0.0,0.2,0.4,0.6, 0.8, and 1.0 s, it is evident that the flow field due to the wall adhesion forces for 8,, = 5” causes the water to move up the tank walls until the boundary condition in (53) is satisfied. For times later than 1 s the water oscillates about the meniscus position in Fig. 12F. This is a consequence of the kinetic energy in the flow, which builds up as a result of the calculation not being initialized with (53) satisfied at the walls. It is interesting that a “ball” of water evolves from an initial shallow pool when the wall adhesion forces are specified on every boundary with 8,, = 175”. The water


> **FIG. 12. Fluid flow vectors corresponding to Fig. 11 at times of (A) 0.0, (B) 0.2, (C) 0.4, (D) 0.6, (E) 0.8, and (F) 1.0 s.**

350 BBACKBILL, KOTHE. AND ZEMACH

1 c


> **FIG. 13. Time sequence of fluid flow vectors in a 2-cm deep pool of water which has been given a wall adhesion boundary condition of 19,~ = 175” on the cylindrical tank walls. The times shown are (A) 0.0, (B) 0.3, (C) 0.5, (D) 0.7, (E) 0.9, and (F) 1.3 s. The computation is performed on a 20 x 40 mesh with Ar = AZ = 0.25 cm.**

behaves in this case like mercury, wanting to separate itself from the walls since 8,, is obtuse. The flow field is displayed at times oft = 0.0,0.3,0.5,0.7,0.9, and 1.3 s in Fig. 13. A net upward momentum, evident in Fig. 13F, is imparted to the water ball by the wall adhesion forces, eventually causing it to collide with the top wall.

E. Low-Gravity Fluid Flows

The success of future long-term space missions depends in part upon the safe and efficient handling and storage of large quantities of liquid cryogens and propellants. Large tanks (approaching l&l00 m3 in volume) with complex internal structures will be used to store, handle, and transport the liquids. Surface tension is typically the dominant force governing the fluid behavior in the absence of gravity. Modeling these fluid flows poses a significant challenge, because an accurate treatment of surface tension is constrained by the complex interface topologies that arise in the inherent three-dimensionality of the flow, the multiple length scales that are present, and the complicated internal and external boundaries of the tanks themselves. Computed results using the CSF model for surface tension are shown for two types of low-gravity fluid flows, jet-induced tank mixing and liquid reorientation, that are likely to occur in future space missions.

1. Jet-Induced Tank Mixing

Jet-induced mixing of fluid stored in a partially-filled tank can prevent excessive thermal stratification of the fluid that may occur when the tank walls are exposed to the sun. A large radiative heat flux might also lead to boil-off of the liquid, depleting the storage volume and possibly resulting in an overpressure of the tank. An internal jet at

A B C D


> **FIG. 14. The results of a calculation using the VOF method to advect fluid interfaces and the CSF model for surface tension of a partially-tilled (50% full of LH, with a 5” contact angle) cylindrical tank (radius 210 cm, height 1020 cm), being mixed at zero gravity by a 1 cm/s axial jet located at the tank bottom. Fluid velocity vectors are shown at times of (A) 0, (B) 1000, (C) 2000, and (D) 3000 s. The surface tension prevents the jet from penetrating the free-surface.**

the bottom of the tank is proposed to induce mixing and to keep the fluid at a nearly uniform temperature. An optimal jet design is one with a velocity large enough to maximize fluid recirculation without penetrating or disrupting the free surface (which destroys the recirculation and mixing). The Weber number (ratio of inertial to surface tension forces) characterizes the competition between jet inertia and surface tension at the free surface. Figures 14 and 15 display the computed laminar flow field induced by an axial jet located at the bottom of a cylindrical tank having elliptical end caps. The results are computed

;, i ,l --<

A B C D


> **FIG. 15. The results of a calculation using the VOF method to advect fluid interfaces and a VOF interface reconstruction mode1 for surface tension of a partially-filled (50% full of LH, with a 5” contact angle) cylindrical tank (radius, 210 cm, height, 1020 cm), being mixed at zero gravity by a l-cm/s axial jet located at the tank bottom. Fluid velocity vectors are shown at times of (A) 0, (B) 1000, (C) 2000, and (D) 2900 s. The free-surface is disrupted due to surface pressure graininess shortly after 2900 s.**

SURFACE TENSION 351

with a 2D, incompressible hydrocode [27] using the VOF .::.:..

method to describe the free surface [26]. The jet velocity :. :I; j; ; j 1,. ..:‘. and radius are 1 cm/s and 10 cm, respectively, and the fluid ,,,,,.. B,,III#.II . ,,,,...1%! l(,..,l,,, .?,// is liquid hydrogen (LH,), occupying 50 % of the tank ,;,+ /I ’ volume. The free surface is initialized in an equilibrium meniscus position with tIeq = 5”. The computation coarsely resolves the tank, which has a radius and height of 210 cm and 1020 cm, respectively, with a 14 x 34 mesh that is refined along the tank axis of symmetry and wall. The Weber number of this system is approximately 8. 1 The computations of Figs. 14 and 15 are identical except A

that surface tension is computed in Fig. 14 with the CSF model, and in Fig. 15 with an interface reconstruction method [ 301 in which the VOF function is used to calculate interface curvature instead of the CSF model. The surface tension in Fig. 15 is then applied as an explicit pressure boundary condition. From Fig. 14 it is evident that a steady state, recirculating flow field is induced as a result of the jet. The jet does not penetrate the free surface, and the-free surface retains its equilibrium shape. A free surface disruption, on the other hand, occurs in Fig. 15 due to the variations in the surface pressure computed from the VOF reconstruction. A relatively crude reconstructed interface is used to estimate the curvature. This algorithm sometimes fails even to give the correct sign of K, leading to surface forces that tend to pull the free surface apart, causing a numericallyinduced disruption as seen in Fig. 15D. The CSF model makes more accurate use of the same VOF data, leading to surface forces that cause the free surface to seek a minimum energy configuration.

2. Liquid Reorientation

Reorientation of liquids stored in partially-filled tanks might be necessary in a microgravity environment before pumping and transfer can occur if the bulk liquid internal to the tank is not located in the neighborhood of the outlet pipe. One scenario proposed to alleviate this problem is a temporary, impulsive acceleration of the tank (provided, for example, by thrusters) to reorient the fluid around the outlet. Key parameters in this process are the minimum magnitude and duration of the thrust necessary to efficiently reorient the fluid. Computed reorientation results are shown in Fig. 16, where ethanol tilling 33 % of a cylindrical tank (radius 2 cm, height 9 cm) with elliptical end caps is reoriented over a period of 5 s by an impulsive z acceleration of the tank equal to 29.4 cm/s’. The ethanol is initially in an inverted equilibrium meniscus position (e,, = 5”) at the top of the tank. Fluid flow is resolved on a 13 x 36 computational mesh, having tiner zones along the tank axis of symmetry and wall. The results are computed with the CSF model for surface tension in the VOF code of Ref. [27]. From Fig. 16, it is evident that the applied acceleration is

E F G H


> **FIG. 16. The results of a calculation using the VOF method to advect the fluid interfaces and the CSF model for surface tension of an ethanol reorientation in a cylindrical tank (2.0 cm radius, 9.0 cm height) are shown. The ethanol, initially inverted and filling 33 % of the tank, is subjected to a z-directed acceleration of -29.4 cm/s2. Fluid velocity vectors are shown at times of (A) 0.1, (B) 0.7, (C) 1.0, and (D) 1.4, (E) 1.9, (F) 2.5, (G) 4.0, and (H) 5.0 s.**

enough to reorient the ethanol at the tank bottom, where the pipe outlet is assumed to be located. This calculation is very difficult for other numerical treatments of surface tension (i.e., Lagrangian, boundary integral, interface tracking, etc.) because of the complex interface topologies. The CSF model has no difficulty with this complexity, since interface geometries are estimated solely from the unit interface normal, which is computed easily from the gradient of VOF data.

V. CONCLUSIONS

The CSF model for surface tension has been described. In the CSF model, a surface force is formulated to model numerically surface tension effects at fluid interfaces having finite thickness. The method is ideally suited for Eulerian interfaces that are not in general aligned with the computational grid. The CSF model has been validated on both static and dynamic interfaces having surface tension. It alleviates previous topology constraints on modeling interfaces having surface tension without sacrificing accuracy and has been applied successfully to a number of fluid flows driven by surface tension.

352 BUCKBILL, KOTHE, AND ZEMACH

Extensions of the CSF model to include, for example, spatially varying surface tension, dynamic contact angle treatment of wall adhesion, and the implementation of the model in three dimensions appear to be straightforward and would extend the method to many new and physically interesting problems. Clearly, there can be improvements in the numerical implementation of the method. The explicit time step constraint, (61), is often more restrictive that any other constraint. The continuum formulation does not increase the severity of this constraint, but may make it easier to formulate implicit equations. An implicit formulation that removes this constraint would decrease the cost of calculations with surface tension by enormous factors. However, the complexity and nonlinearity of the surface tension equations makes this a challenging problem. As we discovered in the drop and Rayleigh-Taylor calculations, both the VOF interface reconstruction and the high-order approximation of convection contribute to errors in numerical calculations with the CSF model. These errors might be eliminated using a particle-in-cell treatment, where the contact discontinuity at interfaces is preserved by the Lagrangian particles [34]. This possibility will be explored. Finally, the reformulation of a discontinuous, interface problem as a continuum problem, which has demonstrably aided in describing surface tension, probably has applications to other problems with similar mathematical structure. We suggest that a fresh look at numerical computations on a grid from the point of view expressed here may yield new ways of modeling discontinuities.

APPENDIX: CURVATURE AND SURFACE TENSION

In this appendix we show that the effects of surface tension are easily derived by summing the tensile forces acting on an interfacial fluid element. The net tensile force, or surface force, is then automatically given as a sum of forces normal and tangential to the interface, with the normal force containing an expression for the vector curvature needed for the CSF model. Consider the surface tension on an interface S in three dimensions. At each point x, on S, one can define a set of orthonormal basis vectors (i , , i,, ri), where i, and i, are in the tangent plane and i2 is the unit normal to S. Two curves on S, s, and s2, can be associated with this coordinate system, with s, along i, and s2 along i,. Since by definition the vector curvature is the change in the unit tangent vector to a curve with respect to arc length s along the curve, one can compute vector curvatures


### K,+(il.V)il 1 (AlI

for curve 1 and


# K,=~=(i,.v)i,

2 (AZ)

for curve 2. The curvature, JC, is then defined as the geometric sum of the two curvatures K, = IK, 1 and

K2 = IK2 1 [20]:

0,) = PI + K2(%). b43)

The sum ICY + K~ may be shown to equal the sum of the reciprocals of the principal radii of curvature, R, and R,. (Note that since K # Jm, the curvature vector K is not the vector sum of K~ and K~, i.e., K # K~ + ~2.) It is difficult to use the expression in (A3) for estimating the curvature of a 3D surface in a numerical model, since it requires an algorithm for finding and choosing intelligently the two most optimal (in the sense of computational ease and expense) curves, s, and s2. There are, however, some numerical models for surface tension that attempt to optimally choose curves s, and s2, thereby making use of (A3) for an estimate of 3D surface curvature [28]. Fortunately, there is an alternative and computationally much simpler expression for K [6], which one can also derive by considering the net surface force per unit area, F,, , on any given element of the surface S [9]. Consider, as in Fig. Al, an element of area 6A = riGA about the point x, on S which is enclosed by a curve C having elemental arc length ds. The surface force exerted on the material in 6A by the material outside of 6A across the elemental line element ds, from Fig. A 1, is equal to aZ ds, where t^ is the unit tangent to S that is perpendicular to arc length vector ds (i ds = ds x ri)

FIG. Al. The surface S represents an interface separating two fluids in three dimensions. The tangential (FF’) and normal (FT’) components of the force on the surface element &A =fisA due to surface tension on its perimeter C are found by summing the tensile force elements oids=adsxrialongC.

SURFACE TENSION 353

at a point along C. The net surface force on element 6A, so that the vector curvature, K, is given by rirc. The use of the F,,6A, is found by summing all forces ai ds exerted on each gradient operator V, rather than V is appropriate when fi is element of arc length ds, defined only on a single surface. When, as in the text, we have a set of contours Z(x) = constant in a region of finite

F,,6A=$~F\d~=$-$d~ thickness, and G(x) =Vc”(x)/lc”(x)l within the region, V. ri = (V, + V,) . ri also has meaning. However,

= d.sxria= dA(rixV)xfio s s V,.ri=ri.(ui.V)ri=~(ri.V)(ri.~)=O, (A14)

=&4[(rixV)xrio] for 6A + 0, (A4) so that V, .ii in (A13) can be replaced by V .ri. The simple yet general expression for the curvature in

where we have used Stokes theorem. In the limit that (A13) has a number of computational advantages over the

6A + 0, we can from (A4) identify F,,(x,) as expression in (A3). First, the curvature can be obtained solely from the one unique property of a surface, the unit F,,(x,) = (ri x V) x ria, (A51 normal ri, which is given in terms of the color function in (36). Second, the differential surface operator automatically which, upon letting the differential operator work on both incorporates the effects of arbitrary coordinate systems. For ri and 0, becomes example, cylindrically symmetric, 2D estimations of curvature using (A3) require two radii of curvature, R,, and F,,(x,) = a[(fi XV) x ri] + [ri x (Vo)] x vi. (A6) RCYl, whereas V, . fi properly accounts for the additional cylindrical force simply with the cylindrical form of V,.

The differential operator can be written as the sum of Third, the extension to three dimensions is trivial, because

surface and normal operators, V = Vs + V,, where V, the unit normal has the same definition in any dimension.

and V, are defined in (2) and (3) respectively, so that Fourth, there is no need to find and determine two optimal curves s, and s2 from which to obtain a surface curvature,

(A7) since the unit normal ri uniquely parameterizes any 3D surface.

since li x V, = 0. Furthermore, by using the identities

ACKNOWLEDGMENTS

(ii x V,) x ti = $V,(ri ii) - rip,. ri)

= -qv, .ri) The authors would like to acknowledge fruitful conversation with our (‘48) colleagues R. Betti, T. D. Butler, R. C. Mjolsness, R. M. Rauenzahn, and D. L. Sulsky. and


## [fix (VCJ)] xii=vo-qF?.v)o=v,cT, (A9) REFERENCES

(A6) can be rewritten as

F,,(x,) = -rio(V, . fi) +V,CJ, (AlO)

from which we can identify

F;:‘(x,) = -fio(V, . fi) (All)

as the normal component of the surface force, and

F”‘(x ) = V sa s so (A121

as the tangential component of the surface force. The curvature can now be identified from (Al 1) as

ICE -(V,.i?), (A13)

1. M. Nelleon, Mechanics and Properties qf Matter (Heineman, London, 1952), Chap. VI.

2. V. G. Levich, Physicochemical Hydrodynamics (Prentice-Hall, Englewood Cliffs, NJ, 1962).

3. H. Lamb, Hydrodynamics, 6th ed. (Cambridge Univ. Press, Cambridge, UK, 1932).

4. S. Ostrach, Annu. Rev. Fluid. Mech. 14, 313 (1982).

5. A. D. Myshkis, V. G. Babskii, N. D. Kopachevskii, L. A. Slobozhanin, and A. D. Tyuptsov, Low-Gravity Fluid Mechanics (Springer-Verlag, New York, 1987).

6. P. G. Drazin and W. H. Reid, Hydrodynamic Stability (Cambridge Univ. Press, Cambridge, UK, 1981).

7. H. N. Oguz and S. S. Sadhal, J. Fluid. Mech. 194, 563 (1988).

8. D. P. Gaver III and J. B. Grotberg, J. Fluid. Mech. 213, 127 (1990).

9. G. K. Batchelor, An Introduction to Fluid Dynamics (Cambridge Univ. Press, Cambridge, UK, 1967).

10. H. R. Pruppacher and J. D. Klett, Microphysics of Clouds and Precipitation (Reidel, Dordrecht, 1978).

354 BRACKBILL, KOTHE, AND ZEMACH

11. E. S. Oran and J. P. Boris, Numerical Simulation of Reactive Flow (Elsevier, New York, 1987).

12. L. D. Landau and E. M. Lifshitz, Fluid Mechanics (Pergamon, New York, 1959).

13. V. G. Levich and V. S. Krylov, Annu. Rev. Fluid Mech. 1, 293 (1969).

14. D. B. R. Kenning, Appl. Mech. Rev. 21, 1101 (1968).

15. C.-S. Yih, Phys. Fluids 11, 477 (1968).

16. C.-S. Yih, Phys. Fluids 12, 1982 (1969).

17. J. Adler and L. Sowerby, J. FIuid Mech. 42, 549 (1970).

18. H. P. Greenspan, Stud. Appl. Math. 57,45 (1977).

19. C. de Boor, A Practical Guide to Sp/ines (Springer-Verlag. New York, 1967).

20. N. G. Mansour and T. S. Lundgren, Phys. Fluids A 2, 1141 (1990).

21. F. H. Harlow and J. E. Welch, Phys. Fluids 8, 2182 (1965).

22. C. W. Hirt, A. A. Amsden, and J. L. Cook, J. Comput. Phys. 14, 227

(1974).

23. F. Y. Kafka and E. B. Dussan V., J. Fluid Mech. 95, 539 (1979).

24. W. C. Elmore and M. A. Heald, Physics of Waves (McGraw-Hill, New York, 1969).

25. B. D. Nichols, C. W. Hirt, and R. S. Hotchkiss, “SOLA-VOF: A

Solution Algorithm for Transient Fluid Flow with Multiple Free Boundaries,” LA-8355, Los Alamos National Laboratory, 1980 (unpublished).

26. C. W. Hirt and B. D. Nichols, J. Compur. Phys. 39, 201 (1981).

27. D. B. Kothe, R. C. Mjolsness, and M. D. Torrey, “RIPPLE: A Computer Program for Incompressible Flows with Free Surfaces,” LA-12007-M& Los Alamos National Laboratory, 1991 (unpublished).

28. M. D. Torrey, R. C. Mjolsness, and L. R. Stein, “NASA-VOF3D: A Three-Dimensional Computer Program for Incompressible Flows with Free Surfaces,” LA-l 1009-MS, Los Alamos National Laboratory, 1987 (unpublished).

29. B. J. Daly, Phys. Fluids 12, 1340 (1969).

30. R. S. Hotchkiss, “Simulation of Tank Draining Phenomena with the NASA SOLA-VOF Code,” LA-8163-M& Los Alamos National Laboratory, 1979 (unpublished).

31. J. U. Brackbill, Methods Compuf. Phys. 16, 1 (1976).

32. B. J. A. Meltz, J. Cornput Phys., submitted.

33. R. Betti, Los Alamos National Laboratory, private communication,

(1990).

34. J. U. Brackbill and H. M. Ruppel, J. Compur. Phys. 65, 314 (1986).

