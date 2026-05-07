1


# Automatic Differentiation using Operator Overloading (ADOO) for implicit resolution of hyperbolic single phase


# and two-phase flow models

François FRAYSSEa and Richard Saurela,b

a: RS2N, Recherche Scientifique et Simulation Numérique, St Zacharie, France b: LMA, Laboratoire de Mécanique et d’Acoustique, UMR 7031 AMU - CNRS - Centrale Marseille, Marseille,

France

Abstract: Implicit time integration schemes are widely used in computational fluid dynamics numerical codes to speed-up computations. Indeed, implicit schemes usually allow for less stringent time-step stability constraints than their explicit counterpart. The derivation of an implicit scheme is however a challenging and time-consuming task, increasing substantially with the model equations complexity since this method usually requires a fairly accurate evaluation of the spatial scheme’s matrix Jacobian. This article presents a flexible method to overcome the difficulties associated to the computation of the derivatives, based on the forward mode of automatic differentiation using operator overloading. Flexibility and simplicity of the method are illustrated through implicit resolution of various flow models of increasing complexity such as the compressible Euler equations, a two-phase flow model in full equilibrium (Le Martelot, et al., 2014) and a symmetric variant (Saurel , et al., 2003) of the two-phase flow model of (Baer & Nunziato, 1986) dealing with mixtures in total disequilibrium.

Keywords: automatic differentiation, implicit, two-phase, finite volume, unstructured meshes


### I. Introduction

In the early days of numerical simulation, the industry growing needs for fast and accurate fluid flow predictions lead to numerous developments in the field of Computational Fluid Dynamics (CFD). Among them, in the aerospace community for instance, implicit time integrators arose from the motivation of decreasing the computational time required to reach steady-state, first for the Euler equations (Mulder & Van Leer, 1983) and later for the laminar compressible Navier-Stokes equations (Venkatakrishnan & Barth, 1989) and for the Reynolds-Averaged Navier-Stokes equations (Barth & Linton, 1995). Towards a better fidelity and thanks to the improvement in computational resources, the development of fast numerical schemes for unsteady flows became necessary and feasible. Taking advantage of the numerous acceleration techniques developed for the explicit time integrators solving steady-state flows, discrete unsteady equations have been solved as successive stationary problems at each time-step (Jameson, 1991). However for problems where the physical time scale is greater than the spatial scale divided by the eigenvalue, fully implicit methods are known to be more efficient (Pulliam, 1993) (Dubuc , et al., 1998).

Implicit methods for unsteady flows are known to suffer from higher diffusive and dispersive errors. Nevertheless, developments of high-order implicit methods such as those based on backward Taylor series or implicit Runge-Kutta (Gottlieb, et al., 2001) opened the way to implicit schemes applied to higher fidelity situations including direct numerical simulation (Martin & Candler, 2006) (Liu, et al., 2016).

One of the major difficulties with implicit schemes is that they require the derivatives of the spatial numerical scheme with respect to the flow variables. The complexity of the involved expressions rapidly grows with the model equations and spatial scheme sophistication such as for example, Riemann problem-based methods. This fact motivated the development of various methods requiring only a crude approximation of the Jacobian

2

matrix such as for example the alternating direction implicit (Briley & McDonald, 1977) where the implicit operator is dimensionally split (restricted to structured grids) or the lower upper symmetric Gauss-Seidel method where the operator is decomposed in lower and upper dominant factors (Jameson & Turkel, 1981). These approximations while decreasing the development time usually severely degrade the stability of the implicit scheme yielding effective time-steps much smaller than the full unfactored form or very slow convergence of the unsteady residual.

While appealing, approximation of the matrix Jacobian through finite differentiation is an expensive strategy and subject to round-off errors not easily controlled. Krylov subspace methods for solving linear systems (Saad & Schultz, 1986) lead to matrix-free implicit algorithms known as Newton-Krylov-matrix-free. In this approach, the matrix Jacobian is not explicitly built, indeed the Krylov methods such as for example the generalized minimal residual method (GMRes) mostly only involves matrix Jacobian-vector products which can directly be approximated through finite differentiation. Nevertheless, it is well-known that the convergence rate of the GMRes method can be highly improved by the application of a preconditioner. However, forming a good preconditioner without the explicit form of the matrix is not a trivial task and the whole method is still subject to numerical error affecting both the linear and the non-linear convergence rate.

Automatic Differentiation (AD) (Wengert, 1964) (Griewank & Walther, 2000) is an algorithmic method for evaluating derivatives up to any order by propagating the chain rule to a high-level language computer program. AD combines the main advantages of symbolic and numerical differentiation without their major drawbacks. In contrast to numerical differentiation it provides exact derivatives (up to round-off error) and is less computationally demanding. AD performs on computer algorithms directly and as such is able to differentiate any complex function as well as numerical procedures such as root-finding methods. Two main approaches to AD are usually considered: AD based on source code transformation (ADSCT) and AD based on operator overloading (ADOO), the key point of the present contribution.

 ADSCT is a procedure that takes as an argument a function or a routine of a computer code, parses all basic operations and intrinsic functions, and returns another function or routine containing the tangent or adjoint derivatives which can be compiled together with the original code. This method is quite easy to apply to any programming language. The generated code complexity does not increase with respect to the original code allowing efficient compile time optimizations. This method is however rather complex to implement. Examples of codes based on this approach include ADIFOR (Bischof, et al., 1996) and TAPENADE (Hascoet & Pascual, 2013).  ADOO determines the derivatives present in a computer code by appending a dual component to all the dependent variables. The dual variable contains the derivatives of the expression with respect to the independent variables and is propagated by applying elementary differentiation arithmetic based on the chain rule. ADOO benefits from object-oriented programming and more precisely from the ability of defining operators on user derived data-types, which is a computer language dependent feature. Programs based on this technique include ADOL-C and ADOL-F for C and FORTRAN programming language respectively (Walther & Griewank, 2012).

In this article, we will show how AD can be applied to the computation of complex matrix Jacobians and how it considerably simplifies the design of implicit schemes for sophisticated systems of equations.

The paper is organized as follows. In Section II, the derivation of a general implicit scheme targeting unsteady flows is presented. AD applied to the evaluation of the matrix Jacobian is detailed in Section III with practical examples in FORTRAN programming language. Then, various fluid flow models of increasing complexity are discretized implicitly and tested in Section IV, including the compressible Euler equations, a two-phase flow model in full equilibrium (Le Martelot, et al., 2014) and a symmetric variant (Saurel , et al., 2003) of the twophase flow model of (Baer & Nunziato, 1986), widely used to model non-equilibrium mixtures.

3


### II. Implicit time integration The fluid flow models considered in this article can be written under the following general differential form:


##        div F G div H 0 (1) t      


### Q Q Q Q


## where Q is the vector of conservative variables,     div F Q the conservative part and


##       G div H  Q Q the non-conservative part of the system. The discrete finite volume form is obtained by

first integrating (1) over a time-independent element i of boundary i and applying the Green-

Ostrogradski theorem to the conservative part:


##      i i i i

i i i dV dS G div H dV 0 (2) t

      


##           Q F Q n Q Q

Then, assuming second order quadrature rule for the surface and volume integrals, averaging the tensor


##  G Q over the domain i and introducing approximate Riemann fluxes * F and * H , system (2) can be

rewritten as:


#                   ij ij ij ij

ij i ij i

* * i

k k i k k i f f f f i j i j f f


## d V g ,g S g ,g S 0 (3) dt                      Q F Q Q n G Q H Q Q n

In the above system, i Q is the volume average of Q  defined as

i

i

i

i


### 1 dV V

    Q Q , ijf is a linear surface of

normal unit


### ijfn separating elements indexed by i and j  and pointing towards element j . Function  g is

an interpolation operator reconstructing left and right states at face ijf  satisfying a maximum principle. This


### function takes as argument a set of conservative vectors forming a compact stencil centered in cell i  for the


# set   k i Q and in cell j for the set   k j Q .

The numerical approximation (3) with respect to non-conservative terms in the present formulation is particularly simple, but appropriate for various two-phase flow models with non-equilibrium effects (Saurel, et al., 2009) (Furfaro & Saurel, 2015). As shown in these references appropriate Riemann solvers considering these terms have to be addressed.


### The implicit discrete temporal integration using Backward-Difference-Formula (BDF) under  form gives


# the following non-linear system for the unknown set of vectors          

ij

k k k i i i f  Q Q Q   forming the


### domain of dependence of cell i :

4


#            


#          

ij ij

ij i

ij ij

ij i

n 1 n n 1 n 1 n 1 * i i k k k i i f f i i j f

n 1 n 1 n 1 * i k k f f i j f

n n 1

i i i

1 V g ,g S t

g ,g S

V 0 (4) t

   



  





            

       

     


# 


# 

Q Q P Q F Q Q n

G Q H Q Q n

Q Q


## In the above system, the first order BDF method is obtained using the couple     , 1,0  and the second


## order BDF uses the couple     , 1,1/ 2  .

It can be shown that BDF of order 1 (BDF1) and 2 (BDF2) are L-stable, furthermore BDF1 is unconditionally Strong Stability Preserving (SSP) for the linear case (Total Variation Diminishing-stable). However, BDF2 and all other high-order methods (>1) are only conditionally SSP, e.g. it exists a constant   n 1 n n 1 s

EE c 0/ max ,..., ,for t c t      Q Q Q where EE t  is the maximum time step for TVD


### Explicit Euler integration. This constant c is not easily bounded in the general case, and often t is bounded by criteria based on relevant physics of the problem under study.

System (4) written for all cells of the mesh represents a coupled non-linear set of equations for the next time-

step unknown vector

n 1  Q . Its solution is approximated by the iterative Newton-Raphson (NR) method. An NR

method iteration is written as

r 1 r r   Q Q Q where the increment

r Q is given by the solution of the

linear system  

r r

r    


### P Q P Q Q

. At convergence of the NR method, e.g. when

r

NL n   Q


### Q

, where

NL  is a tolerance parameter, the solution at the next time-step is obtained: r

NL n

n 1 r 1 lim  

  

Q

Q


### Q Q .

The NR method thus requires a sequence of linear problems to solve each time-step involving the derivative of


### the function P with respect to the conservative variables r 


### 


### P


### Q

. In compact form, the Newton iteration is

obtained through the solution of the following linear system:


#        


#          

ij ij

ij i

ij ij

ij i

r r r * i

i k k d f f i j f

r r r * i k k f f i j f

s n n n 1

i i i i i i


## V 1 g ,g S t


## g ,g S


## 1 V V (5) t t








##                   


##        


##          


# 


# 


## I J Q F Q Q n


## G Q H Q Q n


## Q Q Q Q

where the matrix Jacobian J gathers all the conservative and non-conservative flux derivatives present in the

implicit operator r 


### 


### P


### Q

. In case of unstructured grids, J is a sparse non-symmetric block-matrix whose level of

5

fill-in depends on the union of all spatial domains of dependence which is equal to     k

i i card      Q  . Let us

consider     m k

i  Q Q , the block matrix im J of J is related to r 


### 


### P


### Q

through the following equalities:

       

   

     

ij ij ij ij r r r r r ij i ij i k k k k k i i j i j

ij r r r ij i k k k i i j

* * r i

i f f f f r r r f f m m m g ,g g ,g

im

* i

f r r f m m g ,g


## if m i, S S


## J


## if m i,

                                     

                  


##           


## 


##       


#  


# 

Q Q Q Q Q

Q Q Q


## P F H n G Q n Q Q Q


## P F n Q Q  

   


#        

ij ij ij r r ij i

k k i j

ij ij r ij i i

* r

i f f f r f m g ,g

r r * k k f f i d r i j f m Q

im i d


## S G S


## 1 g ,g S V


## 1 V

                  




##    


##             


##    


# 


# 

Q Q


## H Q n Q


## G H Q Q n I Q


## J I

Performing a single Newton iteration results in the so-called linearized implicit method, which requires only one linear system resolution per time-step. Such approximation allows a large gain in computational speed, but its accuracy is clearly restricted to the case where the magnitude of the unsteady residual is lower than the

temporal scheme’s truncation error:  

r 1

k n O t

    Q


### Q

. Nevertheless, this condition is fulfilled under the

following situations:

 Weak temporal non-linearity;  Accurate estimation of the initial guess in the Newton’s algorithm;  Accurate evaluation of the Jacobian matrix.

Backward difference formula of order two is not a self-starting method as it requires the solution at time n-1. Consequently, the numerical integration for the first time-step needs another discretization scheme. In this paper, the first-order backward difference formula is used to start the numerical integration.

The second implicit scheme considered in this paper is a semi-implicit scheme of the Runge-Kutta family. More precisely, the two-stage second-order Strong Stability Preserving Singly Diagonally Implicit Runge Kutta scheme (SSPSDIRK-2) (Kennedy & Carpenter, 2016). This is a multi-step method and as such is self-starting. The semiimplicit terminology refers to the fact that the Runge-Kutta stages are decoupled from each other, resulting in the present case in two sequential non-linear equations per time-step. This scheme can be written in the following compact form:

6


##                 ij ij ij ij

ij i ij i

r,1 i

r,1 r,1 r,1 r,1 r,1 r,1 * * i

i k k i k k d 11 11 f f f f i j i j f f

r,1 n

i i i

1,1 n r 1,1 r,1 r,1 1 r 1,1

i i i i i i i 0


## V 1 a a g ,g S g ,g S t


## V t


## with , , lim

 

 

 


##                                  


##   


##    


#  

Q

St


## I J Q F Q Q n G Q H Q Q n


## Q Q


## Q Q Q Q Q Q Q


##                


##       

ij ij ij ij

ij i ij i

r,2 r,2 r,2 r,2 r,2 r,2 * * i

i k k i k k d 22 11 f f f f i j i j f f

1 1 * k k 21 i j


## V 1 a a g ,g S g ,g S t


## a g ,g

 


##                                  


##  


#  

age1


## I J Q F Q Q n G Q H Q Q n


## F Q Q


## 


#         ij ij ij ij

ij i ij i

r,2 i

1 1 1 * i k k f f f f i j f f

r,2 n

i i i

1,2 1 r 1,2 r,2 r,2 2 r 1,2

i i i i i i 0

i


## S g ,g S


## V t


## with , , lim

 

 

 


##                      


##   


##    


#  

Q

Stage2


## n G Q H Q Q n


## Q Q


## Q Q Q Q Q Q Q


## Q


## 


##                ij ij ij ij

ij i ij i

2 s s s s n 1 n s * * i k k i k k s f f f f i j i j s 1 f f i


## t b g ,g S g ,g S V



  


##                              Q F Q Q n G Q H Q Q n

with


## 2 1 1 0 2 2 a ,b 1 2 2 1 1 2 2


##                      

For moderately and highly unsteady flows, forming an accurate Jacobian matrix is of primary importance to reach good convergence rate in the Newton’s method (up to quadratic).


### III. Evaluation of the Jacobian matrix

1. Introduction Two main strategies are usually followed to evaluate the Jacobian matrix of the spatial numerical scheme:

 Analytic differentiation. This approach involves both hand and symbolic differentiation of the numerical flux and overall scheme. The main advantage of this approach is the full determination and control of the elements of the matrix resulting generally in an accurate and efficient Jacobian evaluation. Various drawbacks are however present. First, the complexity of the differentials grows rapidly with the model and numerical scheme sophistication making this method very error-prone. The level of flexibility is very low since any slight modification in the model or in the numerical scheme needs an adaptation of the Jacobian matrix. Further difficulties arise when the numerical scheme

7

involves root-finding algorithms such as sophisticated boundary conditions and non-explicit equations of state (EOS). Symbolic differentiation is somehow more flexible but often at the cost of rather complicated expressions thus resulting in a larger computational overhead.  Numerical differentiation. This method is based on finite differencing, and as such is not an exact method. The aim is to approximate the implicit operator as

r r

FD r

FD


### ( + ) ( )      


### P Q P Q P


### Q

with FD  a numerical tolerance. While offering a maximum flexibility and ease of implementation,

this approach has two major drawbacks. First, the computational cost is relatively high as one evaluation of the function P is necessary for each degree of freedom. The second drawback lies in the

calibration of FD  . Indeed, its value is of fundamental importance as too high or too small values yield

inaccurate derivative approximations because of truncation and round-off errors respectively (Dennis & Schnabel, 1983). This approach gained however a lot of interest and success when used together with Krylov subspace methods as a solver for the linear problems (Brown & Saad, 1990). Indeed, in

these linear solvers the implicit operator only appears as a product with the solution increment

r Q .

Consequently, the full matrix does not need to be evaluated, instead the matrix-vector product is approximated through finite differencing:

r r r r FD r

FD


### ( + ) P( )         P Q Q Q P Q Q

While being attractive in terms of computational speed with respect to the full matrix approximation,

this method still suffers of non-trivial estimate of FD  . An inaccurate approximation of the implicit

operator can severely degrade the convergence rate of the Newton’s method or the accuracy of the linearized implicit method. Furthermore, it is well known that the convergence rate of the iterative Krylov subspace linear solvers degrades rapidly when the diagonal dominance decreases (e.g. when the time-step increases). Convergence rate of the linear solver is usually significantly increased by the application of a preconditioner which tends to decrease the spectral radius of the implicit operator. Efficient fully matrix-free preconditioning is nonetheless not trivial and is still an active research area.

2. Automatic differentiation As we shall see in the following, ADOO can be implemented in a straightforward manner into an existing FORTRAN code. Let us first consider the following simple example:


##  


## 

2

' f x sin x x


### f (x) cos x 2x


###  


###  


### The first step consists in replacing the real variables x and f by two objects AD x and AD f having two

components v dv      


### : v being the value of the variable itself and dv the value of its derivative. The FORTRAN 90

standard allows the construction of such objects, more commonly called derived data types (DDT). The

FORTRAN 2003 standard allows each component of a DDT, and in particular dv , to be an array which gives a greater flexibility in handling derivatives with respect to various independent variables. The evaluation of the derivative following the ADOO approach is performed in three steps:

8


## 


## 


##  


##  AD 2 2 2

AD AD AD AD AD AD

Initialization Operator overloading Propagation


### sin x x x x sin x x f x x x f (x ) sin cos x 1 1 1 cos x 2x f '(x) 2x


###                                                     

 Initialization. It consists in copying the value of x in the component v of DDT AD x and initializing the


### component dv to 1, indeed in this case dx dv 1 dx  

 Operator overloading. A DDT is a construct defined by the programmer. As such, no rules are predefined for operations involving one or more DDTs and they need to be coded. Fortunately, the number of basic operations and intrinsic functions is finite, and their coding can be done once and for

all. In the above example, rules must be defined for the operators 

AD 2 AD AD sin , ,  .

 Propagation. Once all needed operators and intrinsic functions are overloaded they can be applied to any DDT and the derivatives then propagate through the chain rule.

To better understand how ADOO works, let us consider the one-dimensional compressible Euler equations closed by the ideal gas equation of state:


##  


##  


##    

2 2 2 u


### 1 p 1 0 with u , u p and E e ,p u u t x 2 1 2 E u E p


###                                         


### F Q Q Q F Q


### where ,u,p,E,e,   denote the density, the velocity, the pressure, the total energy, the internal energy and

the polytropic gas constant respectively. Let us assume a BDF1 numerical scheme in time and a first order finite

volume discretization in space on a uniform grid indexed by i  and of spacing x  . A single Newton iteration (linearized implicit) simplifies the non-linear system (4) to the following tridiagonal linear system to be solved at each time-step:


#  

n n n n

n n

n

* * * * n n n i,i 1 i 1,i i 1,i i,i 1 * * i i 1 i 1 d i,i 1 i 1,i n n n n

i i i-1 i+1

n n n n+1 n * * i i+1 i i i i,i 1


## x


## t


## with , and (6)

       




##                               


##    


## F F F F I Q Q Q F F Q Q Q Q


## F F Q Q Q Q Q

Let us approximate the intercell fluxes by the two-waves Riemann solver of Rusanov (Rusanov, 1962) which has a relatively low algorithmic complexity:


#          


#  

n n n n n n * i i+1 i i+1 i+1 i MAX

n n n n n n i MAX i i i+1 i+1 i n i


### 1 , S 2


### p with S MAX u c , u c and c


###    


###      


### F Q Q F Q F Q Q Q

9

Let us focus on the evaluation of the local Jacobian matrix

n * i,i 1

n

i

 


### 


### F


### Q

appearing in (6) in the special case

n n n max i i i S u c ,u 0    . Omitting the time superscript n  and average symbol , the exact Jacobian

obtained by hand differentiation is given by:

           

      

n

2 i i i i i i 1 i i i 1 i i 1 i i i i i i i i i

* i,i 1 2 2 i i i i i 1 i 1 i i 1 i i i i i 1 i 1 i n

i i i i i i i

1 1 1 u c u 1 1 u E u 2 4 c 2 2 2 4 c 4 c

1 1 u 3 3 1 u u E u u u u c u u 4 4 c 2 2 2 4 c

  



    

                                                                  

F

Q

    

                 

i i 1 i 1 i i i i

3 2 i i i i i i i 2 i i i i 1 i 1 i i 1 i i 1 i 1 i i i 1 i 1 i i i i i i i i i i

1 1 u u u (7) 2 4 c

1 u u E 1 2 E 3 1 u 1 1 u c 1 u 1 u E E E u E E E E 2 4 c 2 4 2 4 c 2 4 c

 

      

                                                                 In order to compare this hand differentiated Jacobian to the one obtained by ADOO in FORTRAN language, the first step consists in defining a DDT as shown in Figure 1.


> **Figure 1: FORTRAN definition of a derived data type.**

Note the dimension of the component dv which corresponds to the total number of independent variables. In the case of one-dimensional Euler equations NCons represents the number of conservative variables which is equal to 3 and Nderiv the number of conservative vectors the intercell flux depends on. Nderiv is equal to 2 in


# the case of first order spatial discretization  

n n

i i+1 , Q Q .

The second step consists in overloading operators and intrinsic functions needed to compute the intercell flux in order to handle computations involving DDTs. Examples are given in Figure 2.


> **Figure 2: Examples of operator and function overloading. These functions allow the use of basic arithmetic with REAL type, AutoDiffType**

or a mix of both.

Let us focus on the function d_times_d described in Figure 2. This function takes as arguments two DDTs and

returns a DDT whose component v holds the product of the values and the component dv holds the conventional derivative of the product. Note the keyword ELEMENTAL, which is a FORTRAN 95 standard and allows to call this function with d1 and d2 being scalar DDTs or arrays of DDTs in a transparent way. The functions d_times_r and r_times_d are necessary when multiplying a DDT by a REAL constant. Note that similar rules may be defined for the product of an INTEGER by a DDT and vice versa.

10

At this point, all basic arithmetic operators and functions in the code involving DDTs would have to be rewritten to use the aforementioned routines. Fortunately, in FORTRAN, an automatic selection of the right function can be achieved through interfacing, see example in Figure 3.


> **Figure 3: Example of operator and function overloading. Interface blocks allowing the use of the symbols =,* and the intrinsic functions**

SIN,COS and EXP whether the arguments are of type REAL or AutoDiffType or a mix of both.

The selection is based on the type of the argument at compile time.

Once all required operators and functions are properly overloaded, the last step is to initialize the component


### dv of the independent variables to the identity matrix

n n

i i+1

d n n

i i+1


###              


### Q Q I Q Q

as shown in Figure 4.


> **Figure 4: Initialization of the derivatives of the left and right vectors of conservative variables prior application of the chain rule.**

Finally, a call to the routine computing the Rusanov flux (Figure 5) will automatically compute all the derivatives of the flux with respect to the left and to the right vectors of conservative variables: the v component of the

DDT variable “Flux(:)” holds all the necessary derivatives. For example,

L,R *

L


### F 


### 

is stored in

Flux(VC_Rho)%dv(VC_Rho,LEFT) and

L,R *

R


### F 


### 

is stored in Flux(VC_Rho)%dv(VC_Rho,RIGHT).

11


> **Figure 5: FORTRAN routine for the computation of the Rusanov flux as well as all its derivatives with respect to the left and right vectors**

of conservative variables. It is worth to note that except the declaration of the variables where TYPE(AutoDiffType) is inserted, the rest

of the subroutine corresponds to the conventional one used in the explicit version of the method.

Without loss of generality, let us detail the computation of

L,R *

L


### F 


### 

. After the initialization step detailed in


> **Figure 4, the left and right dv components of the conservative vectors**

L R c c V ,V  are filled with the following

values:

   

   

 

L R

L R

L R

L R

T T c c

L L R R

T T c c

L L

T c c

v: v:

V VC _ Rho V VC _ Rho L 1 0 0 L 0 0 0 dv : dv: R 0 0 0 R 1 0 0

v: u v : u

V VC _ RhoU V VC _ RhoU L 0 1 0 L 0 0 0 dv : dv: R 0 0 0 R 0 1 0

v: E

V VC _ RhoE V VC _ L 0 0 1 dv: R 0 0 0

                                                               

 

R R

T v: E

RhoE L 0 0 0 dv :R 0 0 1

           

In the Rusanov flux routine of Figure 5, the first instruction after the variables declarations (rhoL=VcL(VC_RHO)) is a DDT copy which is allowed since the assignment operator has been overloaded. This operation copies the

components v and dv of   L c V VC_ Rho   into the components v and dv of the variable L . The next line

extracts the left-face side value of the velocity L u  from the conservative vector. Overloading the division

operator gives the following value for the derivative of L u  with respect to L :

12

                 

   

 

 

L L L L L

L L

c c c c c L L 2 c c

L L L L L 2

L L

V VC_RhoU V VC_RhoU %dv 1,1 V VC_Rho %v V VC_RhoU %v V VC_Rho %dv 1,1 u u %dv 1,1 V VC_Rho V VC_Rho %v

0 u 1 u u %dv 1,1

     

      


### The same result is obtained by hand calculations. Denoting m u  , the velocity is obtained as m u 

.

Hence,

L

L L L L L L 2 2 L L L L L


### m u m u u


###                  

Proceeding the same way for the next lines and assuming MAX L L S u c   , the following intermediate

derivatives are obtained:

 

   

   

   

L L

L

2 L L

L L

L L L

L L MAX

L L L L

E E %dv 1,1

1 p %dv 1,1 u 2

1 c c %dv 1,1 4 c 2

1 u c S %dv 1,1 4 c 2

 

 

    

       

Finally, the derivative of the Rusanov flux with respect to the left density is obtained as:

   

* 2 L,R L L L L L

R L L L L L L

F 1 u u c u c 2 8 c 2 4

                 

which after some elementary algebraic manipulation gives the same expression as the one given in (7) obtained by hand differentiation.


### IV. Numerical examples

In this section numerical integration using time implicit schemes is performed on various flow models and spatial schemes to demonstrate the flexibility delivered by the automatic differentiation. One-dimensional and two-dimensional computations are considered using unstructured meshes decomposed in subdomains using Metis graph partitioning (Karypis & Kumar, 1998). Simulations are performed in parallel using the MPI protocol and the GNU-gfortran compiler. The linear systems arising from the implicit time discretization are solved by the PETSc libraries (Balay , et al., 2018) using block-Jacobi preconditioned GMRes solver (Saad & Schultz, 1986).

One-dimensional and two-dimensional Euler equations are addressed first on a shock-tube test problem followed by an unsteady transonic flow past a cylinder. Simulations of water-hammer using a two-phase flow model in equilibrium and a two-phase flow model in disequilibrium are carried out next.

13

1. Euler equations

a. 1D Sod shock-tube test-case The one-dimensional shock-tube test of Sod (1978) is performed first. The computational configuration is composed of a tube of 1m length with a membrane located at x=0.5m separating a chamber at atmospheric conditions on the left side and a low-pressure chamber on the right side. Air governed by the ideal gas

equation of state is considered with polytropic coefficient 1.4  . The pressure is set to 5 p 10 Pa  to the

left of the membrane and 4 p 10 Pa  to the right while the densities are set to 3 1kg.m  and

3 0.125kg.m  respectively. The computational domain is discretized with 10000 uniform cells. First-order

finite volume computations are carried out using explicit Euler integration scheme with a time-step controlled

by the CFL stability constraint t CFL 0.5 x    


### , being the fastest wave velocity over the domain at a

given time   n n MAX u c    . The time explicit numerical solution is compared to BDF1 with a single

Newton iteration for increasing CFL numbers: 10,50 and 100.

Four numerical flux functions are tested in the present study: the approximate two-waves Riemann solver of Rusanov (Rusanov, 1962), the flux splitting scheme AUSM+ (Liou, 1996), the three-wave approximate Riemann solver HLLC (Toro, et al., 1994) and the Godunov scheme based on the exact Riemann solution (Godunov, 1959). The matrix Jacobian of all schemes is built using the ADOO method presented in Section III.2. While the algorithmic complexity of the Rusanov and HLLC schemes allow a relatively easy symbolic differentiation (see Rinaldi, et al., (2014) for the implicit HLLC scheme), the AUSM+ (see Colonia, et al., (2014) for analytic derivatives ) and the Godunov scheme require a fairly amount of work. The AUSM+ scheme involves numerous conditional branches for the evaluation of the split Mach number and split pressure.

The Godunov scheme is based on the exact solution of the Riemann problem (Godunov, 1959) and involves a root-finding algorithm for the determination of the star pressure, where symbolic differentiation often requires some assumptions at this stage. In contrast, ADOO is able to differentiate the entire numerical scheme, propagating the derivatives inside the fixed-point iterative algorithm in a fully transparent way. The only precaution to be made in the automatic differentiation of root-finding algorithms of Newton type used in the Riemann solver is to set a tolerance allowing both function and function derivative to be converged, the latter generally having a slower rate.

Density plots comparing explicit Euler and BDF1 time integration schemes are presented in Figure 6, for the Rusanov, HLLC, AUSM+ and Godunov flux functions. Excellent stability for all schemes is obtained even for relatively high CFL number. Unconditional SSP property in the linear case for BDF1 together with exact Jacobians transpose very well in this non-linear case.

14


> **Figure 6: 1D Sod shock-tube problem, density plot. Comparison of explicit Euler and BDF1 numerical solutions using the Rusanov, HLLC,**

AUSMP and Godunov (left to right, top to bottom respectively) schemes with various CFL numbers.

BDF1 with HLLC scheme and CFL=100 results in a gain in computational time of about a factor 10. Note that in the present case, these 1D simulations have been carried out in the unstructured multi-dimensional code DALPHADT using the general GMRes linear solver provided by PETSc libraries. As such, the computational gain is representative of higher dimensions simulations.

ADOO time overhead has been measured with respect to analytic differentiation using the Rusanov flux function on a purely structured grid arrangement as well. Total simulation time has been recorded using both methods and use of ADOO showed negligible impact. Indeed, a time overhead lower than 0.1% with the fast block-tridiagonal Thomas algorithm has been observed. In multi-dimensional simulations with sparse nonsymmetric matrices and general linear solvers, this overhead is expected to become even smaller.

b. 2D transonic flow past a circular cylinder The second test-case using the compressible Euler equations is a 2D transonic flow around a circular cylinder of 1m diameter. The computational domain is rectangular (see Figure 7) with non-reflective boundary conditions (BC) assuming low-amplitude waves through algebraic acoustic relations. The circular cylinder BC assumes slip condition, consistent with the inviscid Euler equations. Free-stream BC as well as initial condition (IC) uses a Mach number (Ma) of 0.5 and an angle of attack of 0°. The domain is discretized with 50000 triangles.

A second-order finite volume spatial scheme is used. Cell-center gradients are computed using weighted leastsquares based on a stencil composed of all cells sharing a vertex with the control volume. Multi-dimensional limiting process is ensured using a vertex-based extension (Park, et al., 2010) of the Barth and Jespersen limiter (Barth & Jespersen, 1989). The implicit operator is built using ADOO assuming dependence on direct neighbors


# only, e.g. assuming the set     k

i Q composed of cells sharing a face with cell I (often referred as first-order

Jacobian in the literature). Second-order spatial accuracy is ensured through the right-hand-side of the linear system where fluxes are computed from reconstructed states. This assumption greatly reduces the memory space needed for the implicit operator as well as increases the convergence rate of the linear solver. Indeed, on

15

a regular 2D grid composed of equilateral triangles     k

i card 22       Q using weighted least-squares

gradient approximation together with vertex-based limiter of (Park, et al., 2010), resulting in 22 non-zero block


# matrices per row. Restriction of     k

i Q to direct neighbors only involves a Jacobian matrix composed of 4

non-zero blocks per row. This assumption nevertheless introduces a slight inconsistency between the Jacobian matrix and the right-hand-side resulting in the loss of quadratic convergence of the non-linear residual.


> **Figure 7: Single-phase transonic flow past a circular cylinder. Computational set-up.**

Under these operating conditions, the flow becomes transonic near the top of the cylinder and attached shockwaves form downstream, ahead of the rear stagnation point. Due to the important total pressure loss across the shock and the high curvature, vorticity is generated, and the flow detaches from the wall resulting in an unsteady periodic flow (Salas, 1983) (see solution snapshots in Figure 8).


> **Figure 8: Single-phase transonic flow past a circular cylinder. Contours of density gradients magnitude when periodic regime is reached.**

Solutions are displayed with a time interval of 0.01s when periodic regime is reached.

Explicit and implicit computations have been carried out using the Rusanov, AUSM+ and HLLC schemes for a

physical time of 0.3s. Ideal gas equation of state is used with a polytropic coefficient 1.4  . The explicit

scheme employs a second-order two-stage SSP-Explicit-Runge-Kutta method (SSPERGK-2) (Gottlieb, et al., 2001).

The lift coefficient is computed as

y

s L

2

w

pn.e dS C 1 u S 2

 






## 



, with ,u    the free-stream density and velocity and

2 w S 1m  the projected wetted surface. Lift coefficient is plotted as a function of time using AUSM+

16

approximate Riemann solver, SSPERGK-2 with CFL=0.5 and one-iteration BDF1 with CFL=10 and CFL=20 in Figure 9 (left).

The results agree with those of Pandolfi & Larocca, (1989).  The lift coefficient values obtained with the firstorder implicit scheme are very close to those given by the second-order explicit scheme despite a slight phase lag increasing with the numerical integration time-step. This phase lag can be highly decreased using the second-order BDF2 scheme at the cost of solving the non-linear problem up to a reasonable tolerance, see Figure 9 (right). The Newton-BDF2 scheme with second-order spatial reconstruction is in fact a quasi-Newton approach due to the first-order Jacobian assumption, and only linear convergence has been obtained. A decrease factor of 106 has been imposed to the magnitude of the non-linear residual with a maximum of 10 Newton iterations. Newton-BDF2 is however a rather costly method and no computational gain has been observed with respect to the explicit integration. On the other hand, a factor of 8 in computational time has been obtained using BDF1 (CFL=20) with respect to SSPERGK-2 on a 16-cores Intel Xeon E5-2687W-v2.

The simulation has been carried out using the Rusanov and HLLC fluxes and results are displayed in Figure 10. Explicit and implicit results are in overall good agreement.


> **Figure 9: 2D transonic flow past a circular cylinder. left: comparison of lift coefficient using explicit and implicit schemes based on**

AUSM+ spatial flux and right: BDF2 and Newton-BDF2 schemes compared to BDF1 and SSPERGK-2.


> **Figure 10: 2D transonic flow past a circular cylinder. Comparison of lift coefficient using explicit and implicit schemes based on Rusanov**

and HLLC spatial fluxes (left and right respectively).

All the tested implicit schemes are in good agreement with the explicit solution and results from the literature.

ADOO based implicit schemes are now considered for more sophisticated flow models for which derivation of numerical fluxes is a challenging task.

2. Two-phase flow models in full equilibrium and full disequilibrium Two compressible two-phase flow models are presented in this section. The implicit discretization using ADOO is applied and verified on the numerical simulation of water-hammer two phase flow.

17

a. Parent model in full disequilibrium The symmetric variant (Saurel , et al., 2003) of the compressible two-phase flow model of (Baer & Nunziato, 1986) in the absence of mass and heat transfer reads in the following 1D compact form:


##         (8) t x x


###         


### F Q H Q Q G Q S Q

with


## 


##    


##    


##  

1 I 1

1 1 1 1 1

2 1 1 1 1 I 1 1 1 1

1 1 1 1 1 I I 1 1 1 1

2 2 2 2 2

2

I 2 2 2 2 2 2 2

I I 2 2 2 2 2 2 2 2


### 0 u u 0 0 u p p u


### u E p p u , , , E


### 0 0 u


### p u u p


### p u E u E p

                                                                                                    


### Q F Q G Q H Q

2

2


### (9)


###                       

and


## 


##  


##  


##    


##  


##    

1 2

2 1

I 2 1 I 1 2

2 1

I 2 1 I 1 2


### p p


### 0


### u u


### u u u p p p (10)


### 0


### u u


### u u u p p p


###                                        


### S Q  


###  

In this system, the subscripts 1 and 2 refer to phase 1 and 2 respectively. k k k k k k , ,u ,p ,E ,T (k 1,2)   

are the volume fraction, density, velocity, pressure, total energy and temperature of each phase. I I u ,p  are the

interfacial velocity and pressure modeled as in (Saurel , et al., 2003):


##   1 1 2 2 1 2 1 2 1 1 2 1 1 2 I I 2 1 1 2 1 2 1 2 1 2


### Z u Z u p p Z p Z p Z Z u sgn , p sgn u u (11) Z Z x Z Z Z Z x Z Z                            

where k k k Z c  is the acoustic impedance of phase k with kc the associated sound speed. The

homogeneous part of this model considers each phase as compressible, evolving with its own velocity, pressure


## and temperature. The source term vector  S Q contains a closure law for pressure disequilibrium and drag


### effects, wherecontrols the rate at which pressure equilibrium is reached and  is the coefficient of friction.

The volume average pressure and interface velocity are given by:

1 1 2 2 2 1 1 2 I I 1 2 1 2


### Z u Z u Z p Z p u , p (12) Z Z Z Z        

18

The system is closed by an equation of state for each phase, stiffened-gas in this work:


##    

k k SG,k 2 2 k k k k k k k k


### p p 1 1 E e ,p u u (13) 2 1 2


###        

and by the saturation constraint:

1 2 1 (14) 

The compressible two-phase flow model (8)-(14) is a non-conservative hyperbolic system. The intercell flux functions arising from the spatial finite volume discretization are solved using an algebraic HLLC-type approximate Riemann solver detailed in (Furfaro & Saurel, 2015) and summarized in Appendix B.

Two 1D shock tube problems are solved using explicit and implicit time integrators. The first problem is a water-air shock tube without relaxations, and the second test is a shock tube in a water-air mixture with instantaneous pressure relaxation and drag effects.

Water-air shock tube without relaxations

Initially a 1m length tube is filled with pure water on the left side of a membrane located at x=0.8m and pure air on the right side. Water pressure is set to 0.2GPa while air pressure is set to 0.1MPa. Densities of water and air are initially set to 1000kg/m3 and 1 kg/m3 respectively. Water phase is governed by the stiffened gas EOS

with parameters SG,1 p 1GPa  and 2.35  and air is governed by the ideal gas law with 1.4  . A small

volume fraction of air is present in the water and vice versa ( 6 10  ). The mesh is composed of 2000 elements and first order spatial discretization is used. The simulation time is set to 276µs.

This test case in the absence of pressure and velocity relaxations is very challenging. Indeed, interface conditions across the two-phase contact of equal pressure and normal velocity are not trivial to achieve as they are solely satisfied by the consistent discretization of the non-conservative products present in the model.  This problem is thus a strong benchmark for numerical schemes applied to the two-phase flow model out of equilibrium.

Explicit Euler temporal integration with CFL=0.5 and implicit methods with CFL=20 using Newton-BDF1, Newton-BDF2 and Newton-SSPSDIRK schemes have been compared. Single-step BDF methods failed on this test case for CFL numbers greater than 5. Newton method tolerance has been set to 10-6 , reached in less than 10 iterations thanks to the Jacobian exactness yielding quadratic convergence.

Volume fraction of water is displayed in Figure 11. Explicit and implicit schemes are undistinguishable, and they all yield a perfectly bounded solution.

19


> **Figure 11: Water-air shock tube without relaxations. Volume fraction of water using explicit and implicit schemes.**

Mixture pressure and velocity as well as pressures and velocities of both phases are plot in Figure 12. It can be seen on the top graphs that BDF1 is more diffusive than the explicit Euler integration but is strongly stable. Newton-BDF2 and Newton-SSPSDIRK which are second order accurate give very similar solutions matching the explicit solution. A small oscillation is however present at the tail of the rarefaction wave using Newton-BDF2 scheme where Newton-SSPSDIRK remains stable. Second order implicit methods are only conditionally SSP, SSPSDIRK with a less stringent time-step limit.

Bottom graphs of  Figure 12 show both phases pressures and velocities. Only Newton-SSPSDIRK results are shown for the sake of conciseness, explicit solution can be found in (Furfaro & Saurel, 2015). Interface conditions are fulfilled: at the contact pressures and velocities of both phases match perfectly.


> **Figure 12: Water-air shock tube without relaxations. Comparison between explicit and implicit time integrators. Top: mixture pressure**

and mixture velocity. Bottom: pressures and velocities of both phases showing correct fulfillment of interface conditions.

Water-air mixture shock tube with pressure relaxation and drag effects

The second test case considered with the two-phase model out of equilibrium is a shock tube composed of a

water-air mixture. A 1m length tube is initially filled with water and air in proportion 0.1  and 0.9  . A


### membrane is placed at x 0.5m  separating a hot pressure chamber at 100bar to the left and 1b to the right.

20

Densities of water and air are set to 1000kg/m3 and 1kg/m3 respectively. Water phase is governed by the

stiffened gas EOS with parameters SG,1 p 1GPa  and 2.35  and air is governed by the ideal gas law with


### 1.4  . The mesh is composed of 2000 elements and first order spatial discretization is used. The simulation

time is set to 790µs.

Instantaneous pressure relaxation is applied following a time splitting procedure detailed in (Furfaro & Saurel,

2015). Drag effects are accounted for considering constant water bubble radius w R 5mm  . The exchange

interfacial area per unit volume ( I A ) dependence is thus simplified. It is assumed dependent only on water

volume fraction ( w  ) evolution which in turn is driven by pressure equilibrium. The total friction coefficient 

is then modelled as:

w 1 2 1 2 I 1 2 1 2 w


### 3 Z Z Z Z A Z Z Z Z R


###     

This finite rate momentum exchange is integrated following a first-order explicit time-splitting method.

A comparison between explicit Euler temporal integration using CFL=0.5 and Newton-SSPSDIRK scheme using CFL=10,20 and 30 is carried out. Pressure is displayed in Figure 13, showing overall good agreement between the explicit and the implicit solutions. The dominant error is due to the time-splitting strategy. Indeed, the difference between the explicit solution and SSPSDIRK with CFL=10 is much higher than the differences between the various implicit solutions using increasing CFL.


> **Figure 13: Water-air mixture shock tube with pressure relaxation and drag effects. Relaxed pressure using explicit and implicit time**

integrators.

Water and air velocities are displayed in Figure 14. Momentum exchanges driven by drag effects are quite stiff under these operating conditions as shown by the relatively small velocity drift. Overall good agreement between the explicit and the implicit solutions is visible.

21


> **Figure 14: Water-air mixture shock tube with pressure relaxation and drag effects. Water and air velocities using explicit and implicit**

time integrators.

b. Reduced model in full equilibrium Another two-phase flow model is considered as well for numerical experiments. An asymptotic analysis in the stiff limit of velocity, pressure and temperature of system (8)-(14) yields the following two-phase model in full equilibrium (Le Martelot, et al., 2014) (Chiapolino, et al., 2016):


##    

2

1 1


### u


### u u p 0, , (15) E u E p t x


### Y uY


###                               


###          


### F Q Q Q F Q

where 1 ,u,p,E,Y  denote the mixture density, velocity, pressure, total energy and mass fraction of phase 1

respectively. The system is closed by the saturation constraint:

1 2 Y Y 1 (16)  

and by a mixture equation of state satisfying the following principles of conservation of mixture internal energy and mixture specific volume:


##        

1 1 2 2

1 1 2 2


### e Ye p,T Y e p,T


### v Y v p,T Y v p,T (17)


###  


###  

Considering stiffened-gas equation of state for phase 1 and ideal gas equation of state for phase 2, system (17) admits a unique physical solution for the mixture pressure:

22


##    


##       1 2

1 2 1 2

2

1 2 SG,1 2 1 SG,1 1 2

1 1 v 2 2 v 1 SG,1 2 1 v 2 v 1 v 2 v


### 1 1 p A A p A A p A A 2 4 Y 1 c Y 1 c with A e p ,A e (18) Yc Y c Yc Y c


###       


###      

Equations (15),(16) and (18) form a conservative hyperbolic system. The sound speed associated to this system of equations is very well approximated by the simple Wood formula (Wood, 1930). The intercell flux function arising from the spatial finite volume discretization is solved using an algebraic HLLC-type approximate Riemann solver (Saurel, et al., 2006).

Two 1D problems are solved using explicit and implicit time integrators. The first test is a shock tube in a waterair mixture while the second problem is a double rarefaction wave. Multi-dimensional extension is addressed through a sonic jet problem and a gas-gas single mode Rayleigh-Taylor instability

Water-air mixture shock tube

A 1m length tube is filled with a mixture of water and air. Water mass fraction is set to 2%. Initially, a membrane located at x=0.5m separates a high-pressure chamber to a low-pressure one. On the left side, pressure is set to 0.2MPa while it is set to 0.1MPa on the right side. Temperature is imposed to 293K in all the

domain. Water phase is governed by the stiffened gas EOS with parameters SG,1 p 1GPa  and 2.35  and

air is governed by the ideal gas law with 1.4  . The mesh is composed of 10000 elements and first order

spatial discretization is used. The simulation time is set to 1ms.

Explicit and implicit integration is carried out. The explicit scheme uses first order Euler integration with CFL number set to 0.5, while the implicit scheme uses single-step BDF1 with CFL=10,20 and 40. Mixture density is displayed at the final time in Figure 15. Under these mild conditions single-step BDF1 remains stable even for a CFL number of 40, yielding a computational gain of a factor 10 with respect to explicit integration.


> **Figure 15: Water-air mixture shock tube. Mixture density at time t=1ms using explicit Euler and implicit BDF1 schemes.**

Double rarefaction wave

The second 1D test case considered using the two-phase model in mechanical and thermal equilibrium is a double rarefaction wave problem. This test is particularly interesting to benchmark numerical schemes as vacuum conditions are reached at the center.

A 1m length tube is considered filled with almost pure water (air mass fraction equal to 10-6) at atmospheric conditions p=0.1MPa and T=293K. Initially a membrane is located at x=0.5m. Velocity is set to -10m/s and 10m/s to the left and to the right of the membrane respectively. The domain is discretized into 10000 elements

23

and the simulation time is set to 1.5ms. Explicit first order Euler integration with CFL number equal to 0.5 and implicit single-step BDF1 scheme with CFL=10,20 and 40 are used with a first order spatial discretization.

Pressure and velocity at time t=1.5ms are displayed in Figure 16. Single-step BDF1 scheme remain stable in all cases without pressure positivity violation. At a CFL of 40, the gain in CPU time is about a factor 10 with respect to explicit Euler integration.


> **Figure 16: Double rarefaction wave. Velocity (left) and pressure (right) using explicit Euler and implicit BDF1 schemes.**

Multi-dimensional tests are now addressed. A 2D two-phase sonic jet simulation is carried out first, then a 2D two-phase single mode Rayleigh-Taylor instability is computed.

2D sonic jet

A 2D jet configuration is set up as described in Figure 17. Initially the domain is composed of a water-air mixture in equal mass proportion at atmospheric conditions. Water phase is governed by the stiffened gas EOS

with parameters SG,1 p 1GPa  and 2.35  and air is governed by the ideal gas law with 1.4  . At the

left side, a two-phase pressure tank-inlet boundary condition is imposed (see Appendix A). Tank pressure is set to 0.2MPa yielding a chocked flow through a convergent-divergent nozzle and a sonic jet develops.


> **Figure 17: 2D sonic jet computational set-up.**

The domain is discretized using an unstructured mesh composed of about 250000 triangles (the mesh edges length is multiplied by a factor 10 in Figure 17). Second order spatial discretization is applied using Barth and Jespersen limiter and the simulation time is set to 21ms. Explicit temporal integration uses second order SSPERGK scheme with CFL number equal to 0.5. Implicit schemes use single-step first order BDF and second order Newton-SSPSDIRK both using CFL equal to 20. In case of Newton-SSPSDIRK scheme, as quadratic

24

convergence cannot be achieved due to the first order Jacobian simplification, a limit of five iterations to reach a non-linear residual magnitude of 10-6 has been imposed.

Two-phase tank-inlet condition implies the solution to a non-linear equation to approximate the boundary flux. The Jacobian boundary contribution is computed in a straightforward manner using ADOO as detailed in Appendix A.

Contours of density at time 14ms and 21ms for the three temporal schemes are displayed in Figure 18. No significant diffusion is observed on the jet using BDF1 scheme, in contrast the shock wave is slightly smeared. At this CFL number, the higher order Newton-SSPSDIRK scheme predicts a sharper shock front but some nonamplifying oscillations are present.


> **Figure 18: 2D jet at time t=14ms (top) and t=21ms (bottom). Mixture density contours.**

Newton-SSPSDIRK scheme with an imposed maximum of five non-linear iterations per stage requires about 10 linear systems to be solved at each time step. At a CFL number of 20, the CPU gain with respect to two-stage explicit SSPERGK scheme is about 17%. Single-step BDF scheme which only requires one linear system solution per time step gives a CPU gain factor of about 5.

Air-Helium Rayleigh-Taylor instability

A 2D single-mode Rayleigh-Taylor configuration is set up as described in Figure 19. Initially the domain is composed of a pure light gas (helium) on the bottom and a denser gas on the top (air). An initial perturbed interface separates the two gases to trigger a single-mode instability driven by gravity effects. Gravity magnitude has been set to 1000m/s2. The computational domain is all bounded by slip wall boundary conditions. Air and helium are governed by the ideal gas law with polytropic coefficients 1.4 and 1.67 respectively.

25


> **Figure 19: 2D single-mode Rayleigh-Taylor instability computational set up.**

Final simulation time is set to 37.5ms. In the absence of viscous and surface tension effects, the interface is highly unstable and subject to chaotic behavior which can be triggered by small perturbations induced for example using distinct spatial numerical discretization (Liska & Wendroff, 2004). In our context, the spatial discretization is fixed using second-order HLLC scheme on a mesh composed of about 300000 triangles.

Explicit temporal integration uses second order SSPERGK scheme with CFL number equal to 0.5. Implicit schemes use single-step first order BDF and second order Newton-SSPSDIRK both using CFL equal to 100. In case of Newton-SSPSDIRK scheme, a limit of five iterations to reach a non-linear residual magnitude of 10-6 has been imposed. Contours of mixture density are shown for the three temporal schemes at two time intervals in Figure 20.

Single-step first order BDF scheme is more diffusive than the explicit scheme resulting in a slightly altered mushroom shape. In contrast, second order implicit SSPSDIRK scheme gives a remarkably similar solution as the explicit one.

The CPU speed-up using implicit SSPSDIRK scheme is about a factor 4 while it reaches a factor of about 27 using single-step first order BDF method.

26


> **Figure 20: Single mode Rayleigh-Taylor instability at time t=22.5ms (top) and t=37.5ms (bottom). Mixture density contours using explicit**

and implicit time integrators.


### V. Conclusion

A numerical procedure to evaluate the matrix Jacobian coming from the implicit discretization of hyperbolic systems has been presented. The approach is based on the forward mode of ADOO and showed numerous advantages:

 It is easily applicable to existing codes with languages compilers supporting user-defined DDT and operator overloading;

27

 It is highly flexible with models and numerical schemes adjustments as the user do not need to worry about the implementation of the derivatives;  It handles complex numerical algorithms such as root-finding methods or conditional branches differentiation in a straightforward manner,  The overhead associated to operations on DDTs is negligible in the present context as most of the computational time is spent in the linear solver.

The ADOO method has been applied to the implicit discretization of various flow models, involving root-finding methods and complex equations of state. In the test-cases needing unsteady residual resolution, the accurate Jacobian evaluation allowed very fast convergence of Newton’s method (always quadratic for first-order in space discretization).

ADOO allowed the implicit discretization of the compressible two-phase flow model of Baer and Nunziato. To the author’s knowledge, the present work corresponds to the first successful attempt with a fully implicit approach.

28


### Appendix A: Tank-inlet boundary condition

The tank-inlet boundary condition is obtained though the solution of a semi-Riemann problem described in Figure 21.


> **Figure 21: Tank-inlet boundary condition description.**

Assuming a stationary flow between the tank and the inlet, the integration of the energy equation associated to the Euler equations on volume V yields:


#      

mass cons° * * L 0 L 0 V


### . Hu dV=0 uHS uHS 0 H H        


###  

which gives a first relation linking the tank state to the left Riemann state.

The enthalpy is conserved across the left curved wave. Considering an isentropic flow in the tank, the speed of

sound 2

s


### p c        

is assumed constant across the left wave:

2 * * 2 L 0 L 0 * L 0


### p p c c   

. This gives a second relation

linking the tank state to the left Riemann state:

* * L 0 L 0 2 0


### p p c


###   . Acoustic relations are assumed through

the right wave:


#    

* * R R R R R R R R

* * * R R R R R R R


### p c u p c u


### p u p c u u


###   


###    

The middle wave being necessarily a contact, equality of pressure and normal velocity is imposed yielding

* * * * L R L R p p ,u u   . Putting all relations together yields to:


##  


##  


##  


##          

2 * * * * L L L L 0 * L

* * * L 0 L L 0 2 * 2 0 * * * * 0 * * * * R R R R R R R

* * L R

* * L R


### p 1 e p , u H 0 2


### p p f p c p 1 e p ,f p g p H h p 0 p p 2 f p u g p u c


### p p


### u u


###                           


###      

29

After application of the equation of state, this non-linear relation is solved numerically using Newton’s method. Classical sampling is then performed, and the flux is computed from the Riemann solution. ADOO is propagated to the Newton’s method in a transparent way, but with an additional convergence criterium based on the


## magnitude of the derivative of function   * h p .

Extension to the two-phase reduced model is straightforward considering the mixture equation of state. A prototype subroutine is given as example in Figure 22.


> **Figure 22: FORTRAN routine for the computation of tank-inlet boundary flux. The resulting data structure fluxbc contains the derivative**

blocks of the flux with respect to the vector of conservative variables.

30


### Appendix B: HLLC solver for the two-phase model in disequilibrium

The two-phase model in disequilibrium can be rewritten under the following compact form for a generic phase k:


##          


##  


##    


##  


##  

k I k I

k k k k k k k

2 k k k k k k k k I I k k

I I k k k k I I k


### u u u 0 0, , , (19) u p p u p t x x


### E p u E p u p u


###                                                                


### F Q H Q Q Q F Q H Q

The first step consists in approximating the interfacial terms I I u et p . This step takes its roots in the Discrete

Equations Method (Saurel , et al., 2003), given a 2D topology composed of independant channels as illustrated in Figure 23.


> **Figure 23: 2D topology showing two types of contacts at a given element boundary.**

On a given cell boundary, three types of contacts are admissible : two mono-phasic contacs 1-1 and 2-2 and one contact 2-1 if the phase 1 volume fraction is higher on the right side or 1-2 if it is lower. Interfacial pressure and velocity are approximated as the solution of the Riemann problem associated to the two-fluide Euler

equations. As the Riemann solution depends only on the left and on the right state, Iu and Ip become locally

constant during a time-step.

Focusing on a 2-1 contact, the wave pattern is shown in Figure 24.


> **Figure 24: Riemann wave pattern associated to contact 2-1.**

Wave speeds L,2 R,1 S ,S are computed following Davis approximation (Davis, 1988):

31

 


##  

R,1 L,1 L,1 R,1 R,1

L,2 L,2 L,2 R,2 R,2


### S Max u c ,u c


### S Min u c ,u c


###   


###   

The interfacial variables Iu and Ip are computed HLL (Harten, et al., 1983) and HLLC (Toro, et al., 1994)

approximations respectively:


#        


##    


##   

2 2

L,2 R,1 L,2 R,1 R,1 L,2 *,HLL I 21

L,2 L,2 R,1 R,1 R,1 L,2

* I R,21 R,1 R,1 R,1 R,1 I R,1


### u p u p S u S u u u u u S S


### p p u S u u p


###                


###     

In case of a 1-2 contact, a similar approach is followed (Furfaro & Saurel, 2015).

The interfacial vairbales being locally constant on an element boundary during a time-step, system (19) can be rewritten under the following conservative form:

       


##  


##    


##  

k I k

k k k k k

2 k k k

k k I k k

k k k k I I k


### u u 0, , u P p u t x


### E E p u p u


###                                                 


### F Q Q Q F Q

Local constancy of the interfacial variables also implies the decoupling of the full 7-wave Riemann problem in two 4-wave Riemann problems as shown in .


> **Figure 25: Main patterns of phase k Riemann problem: L,k M,k I R,k L,k I M,k R,k S S u S et S u S S      **

The 4-wave Riemann solution is based on the Rankine-Hugoniot relations, approximating the wave speeds

L,k S and R,k S by (Davis, 1988) and intermediate contact M,k S by HLL. Rankine-Hugoniot relations across the

various waves read :


##  


##  


##  


##  

* * L,k L,k L,k L,k L,k

* * R,k R,k R,k R,k R,k

** * ** * k R,k I k R,k

* ** * ** L,k k M,k L,k k


### S


### S


### u


### S


###   


###   


###   


###   


### F F Q Q


### F F Q Q


### F F Q Q


### F F Q Q

32

Explicit relations for

* * ** L,k R,k k , and Q Q Q are obtained as a functions of known states L,k R,k and Q Q .

Riemann solution state is obtained through a classical sampling. Once ccomputed, these states and fluxes are used in the non-conservative explicit or implicit Godunov scheme associated to system (19).

The whole numerical flux function is algorithmically rather complex. Nevertheless, the ADOO procedure is able to provide the exact Jacobian matrix in a straightfoward manner.

33


## References

Baer, M. R. & Nunziato, J. W., 1986. A two-phase mixture theory for the deflagration-to-detonation transition (ddt) in reactive granular materials. International Journal of Multiphase Flow, pp. 861-889.

Balay , S. et al., 2018. PETSc Users Manual, s.l.: Argonne National Laboratory.

Barth, T. & Jespersen, D., 1989. The design and application of upwind schemes on unstructured meshes. s.l., AIAA.

Barth, T. J. & Linton, S. W., 1995. An Unstructured Mesh Newton Solver for Compressible Fluid Flow and Its Parallel Implementation. AIAA Paper 95-0221.

Bischof, C., Khademi, P., Mauer, A. & Carle, A., 1996. Adifor 2.0: Automatic Differentiation of Fortran 77 Programs. IEEE Computational Science & Engineering, pp. 18-32.

Briley, W. R. & McDonald, H., 1977. Solution of the multi-dimensional compressible Navier-Stokes equations by a generalized implicit method. Journal of Computational Physics, pp. 372-397.

Brown, P. N. & Saad, Y., 1990. Hybrid Krylov Methods for Nonlinear Systems of Equations. SIAM Journal on Scientific and Statistical Computing, pp. 450-481.

Chiapolino, A., Boivin, P. & Saurel, R., 2016. A simple phase transition relaxation solver for liquid–vapor flows. International Journal for Numerical Methods in Fluids, 83(7), pp. 583-605.

Colonia, S., Steijl, R. & Barakos, G. N., 2014. Implicit implementation of the AUSMP and AUSMP-up schemes. International Journal for Numerical Methods in Fluids, p. 687–712.

Davis, S. F., 1988. Simplified second-order Godunov-type methods. SIAM Journal of Scientific and Statistical Computing, 9(3), p. 445–473.

Dennis, J. E. & Schnabel, R. B., 1983. Numerical Methods for Unconstrained Optimization and Nonlinear Equations. Englewood Cliffs: Prentice-Hall.

Dubuc , L. et al., 1998. Solution of the Unsteady Euler Equations Using an Implicit Dual Time Method. AIAA Journal, 36, pp. 1417-1424.

Furfaro, D. & Saurel, R., 2015. A simple HLLC-type Riemann solver for compressible non-equilibrium two-phase flows. Computers & Fluids, pp. 159-178.

Godunov, S., 1959. Finite difference method for numerical computation of discontinuous solutions of the equations of fluid dynamics. Matematicheskii Sbornik, Steklov Mathematical Institute of Russian Academy of Sciences, Volume 47, pp. 271-306.

Gottlieb, S., Shu, C.-W. & Tadmor, E., 2001. Strong Stability-Preserving High-Order Time Discretization Methods. SIAM Review, pp. 89-112.

Griewank, A. & Walther, A., 2000. Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation. s.l.:SIAM.

Harten, A., Lax, P. D. & van Leer, B., 1983. On upstreaming differencing and Godunov-type. SIAM Review, 25(1), pp. 35-61.

34

Hascoet, L. & Pascual, V., 2013. The Tapenade automatic differentiation tool: Principles, model, and specification. ACM Transactions on Mathematical Software.

Jameson, A., 1991. Time dependent calculations using multigrid, with applications to unsteady flows past airfoils and wings. AIAA Paper 91-1596.

Jameson, A. & Turkel, E., 1981. Implicit Scheme and LU-Decompositions. Mathematics of Computation, pp. 385397.

Karypis, G. & Kumar, V., 1998. A fast and high quality multilevel scheme for partitioning irregular graphs. SIAM Journal on Scientific Computing, 20(1), pp. 359-392.

Kennedy, C. A. & Carpenter, M. H., 2016. Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review, NASA Langley Research Center; Hampton, VA, United States: NASA/TM-2016-219173, L20470, L-20597, NF1676L-19716.

Le Martelot, S., Saurel, R. & Nkonga, B., 2014. Towards the direct numerical simulation of nucleate boiling flows. International Journal of Multiphase Flow, pp. 62-78.

Liou, M. S., 1996. A Sequel to AUSM: AUSMP. Journal of Computational Physics, 129(2), pp. 364-382.

Liu, X., Xia, Y., Luo, H. & Xuan, L., 2016. A Comparative Study of Rosenbrock-Type and Implicit Runge-Kutta Time Integration for Discontinuous Galerkin Method for Unsteady 3D Compressible Navier-Stokes equations. Communications in Computational Physics, pp. 1016-1044.

Martin, M. P. & Candler, G. V., 2006. A parallel implicit method for the direct numerical simulation of wallbounded compressible turbulence. Journal of Computational Physics, pp. 153-171.

Mulder, W. A. & Van Leer, B., 1983. Implicit upwind methods for the Euler equations. AIAA Paper 83-1930.

Pandolfi, M. & Larocca, F., 1989. Transonic flow about a circular cylinder. Computers & Fluids, 17(1*), pp. 205220.

Park, J. S., Yoon, S.-H. & Kim, C., 2010. Multi-dimensional limiting process for hyperbolic conservation laws on unstructured grids. Journal of Computational Physics, pp. 788-812.

Pulliam, T. H., 1993. Time Accuracy and the Use of Implicit Methods. AIAA Paper 93-3360.

Rinaldi, E., Pecnik, R. & Colonna, P., 2014. Exact Jacobians for implicit Navier–Stokes simulations of equilibrium real gas flows. Journal of Computational Physics, p. 459–477.

Rusanov, V. V., 1962. Calculation of interaction of non-steady shock waves with obstacles. USSR Computational Mathematics and Mathematical Physics, 1(2), pp. 304-320.

Saad, Y. & Schultz, M. H., 1986. GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems. SIAM Journal on Scientific and Statistical Computing, pp. 856-869.

Salas, M. D., 1983. Recent developments in transonic Euler flow over a circular cylinder. Mathematics and Computers in Simulation, 25(3), pp. 232-236.

Saurel , R., Gavrilyuk, S. & Renaud, F., 2003. A multiphase model with internal degrees of freedom: application to shock–bubble interaction. Journal of Fluid Mechanics, pp. 283-321.

Saurel, R., Boivin, P. & Le Métayer, O., 2006. A general formulation for cavitating, boiling and evaporating flows. Computers & Fluids, Volume 128, pp. 53-64.

35

Saurel, R., Petitpas, F. & Berry, R. A., 2009. Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures. Journal of Computational Physics, pp. 1678-1712.

Sod, A. S., 1978. A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. Journal of Computational Physics, 27(1), pp. 1-31.

Toro, E. F., Spruce, M. & Speares, W., 1994. Restoration of the contact surface in the HLL-Riemann solver. Shock Waves, 4(1), p. 25–34.

Venkatakrishnan, V. & Barth, T., 1989. Application of direct solvers to unstructured meshes for the Euler and Navier-Stokes equations using upwind schemes. AIAA Paper 89-0364.

Walther, A. & Griewank, A., 2012. Getting started with ADOL-C. In U. Naumann und O. Schenk, Combinatorial Scientific Computing, Chapman-Hall CRC Computational Science, pp. 181-202.

Wengert, R. E., 1964. A simple automatic derivative evaluation program. Communications of the ACM, pp. 463464.

Wood, A., 1930. A textbook of sound. London: G. Bell and Sons Ltd.

