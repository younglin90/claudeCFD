"""Symbolic (sympy) checks of the algebraic identities used by theorem L3-F (section 62):
 (1) mass-midpoint force correction  p_avg - p(x_mm) = c l^2/12 - s_p s_rho l^2/(8 rho) + O(l^3)
 (2) on a uniform grid the curvature fit of _curv_fit gives c = (p_+ - 2p + p_-)/(2h^2), so the
     curvature part of the correction is corr = Delta_2 p / 24, symbol sigma(k) = sin^2(kh/2)/6 in [0, 1/6]
 (3) lagged correction: Cayley amplification has |G| > 1 (grid modes grow);
     correction at the secant pressure: G is the Cayley transform of i*theta*[[0,1+sigma],[1,0]], |eig| = 1
 (4) node-mass lower bound M_I >= rho_min h/4 for an interface node (two zone lengths summing to h)
"""
import sympy as sp

x, l, h, rb, sr, cr, p0, spp, cp = sp.symbols("x l h rho_b s_rho c_rho p_0 s_p c_p", real=True)
ok = True

# (1) zone [-l/2, l/2], density rho = rb + sr x + cr (x^2 - l^2/12) (mean rb), pressure likewise
rho = rb + sr * x + cr * (x ** 2 - l ** 2 / 12)
p = p0 + spp * x + cp * (x ** 2 - l ** 2 / 12)
m = sp.integrate(rho, (x, -l / 2, l / 2))
# mass midpoint x_mm = d2 l^2 + d3 l^3 (ansatz), fixed by integral_{-l/2}^{x_mm} rho = m/2 to O(l^4)
d2, d3 = sp.symbols("d2 d3")
x_mm = d2 * l ** 2 + d3 * l ** 3
eq = sp.expand(sp.integrate(rho, (x, -l / 2, x_mm)) - m / 2)
co = sp.Poly(sp.series(eq, l, 0, 4).removeO(), l)
dsol = sp.solve([co.coeff_monomial(l ** 2), co.coeff_monomial(l ** 3)], [d2, d3], dict=True)[0]
x_mm = x_mm.subs(dsol)
pavg = sp.integrate(p, (x, -l / 2, l / 2)) / l
gap = sp.series(sp.simplify(pavg - p.subs(x, x_mm)), l, 0, 3).removeO()
claim = cp * l ** 2 / 12 - spp * sr * l ** 2 / (8 * rb)
r1 = sp.simplify(sp.expand(gap - claim))
print("(1) p_avg - p(x_mm) - [c_p l^2/12 - s_p s_rho l^2/(8 rho)] =", r1, "(to O(l^3))")
ok &= r1 == 0

# (2) uniform grid curvature fit, neighbour zones [h/2, 3h/2] and [-3h/2, -h/2], l = h
pm, pc_, pp = sp.symbols("p_m p_c p_p", real=True)
s_, c_ = sp.symbols("s c")
eqs = []
for av, a, b in ((pp, h / 2, 3 * h / 2), (pm, -3 * h / 2, -h / 2)):
    eqs.append(sp.Eq(s_ * (a + b) / 2 + c_ * ((b ** 3 - a ** 3) / (3 * (b - a)) - h ** 2 / 12), av - pc_))
sol = sp.solve(eqs, [s_, c_])
r2 = sp.simplify(sol[c_] - (pp - 2 * pc_ + pm) / (2 * h ** 2))
corr = sp.simplify(sol[c_] * h ** 2 / 12)
print("(2) c_fit - Delta_2 p/(2h^2) =", r2, "; corr =", corr)
ok &= r2 == 0 and sp.simplify(corr - (pp - 2 * pc_ + pm) / 24) == 0
kh = sp.Symbol("kh", real=True)
# Fourier symbol of -corr/p for p_j = e^{i j kh}: -(e^{ikh} - 2 + e^{-ikh})/24 = sin^2(kh/2)/6
sig = sp.simplify(-(sp.exp(sp.I * kh) - 2 + sp.exp(-sp.I * kh)) / 24)
r2b = sp.simplify(sp.expand_complex(sig - sp.sin(kh / 2) ** 2 / 6))
print("    sigma(k) - sin^2(kh/2)/6 =", r2b)
ok &= r2b == 0

# (3) amplification matrices (u = w sqrt(M), q = p sqrt(l/K), theta = acoustic CFL symbol)
th, sg = sp.symbols("theta sigma", positive=True)
I = sp.I
A_imp = sp.Matrix([[1, I * th * (1 + sg) / 2], [I * th / 2, 1]])
B_imp = sp.Matrix([[1, -I * th * (1 + sg) / 2], [-I * th / 2, 1]])
G_imp = A_imp.inv() * B_imp
W = sp.diag(1, 1 + sg)
r3 = sp.simplify(G_imp.H * W * G_imp - W)
print("(3) secant-pressure correction: G^H W G - W =", list(r3), "with W = diag(1, 1+sigma)  => |G|_W = 1")
ok &= all(sp.simplify(e) == 0 for e in r3)
A_lag = sp.Matrix([[1, I * th / 2], [I * th / 2, 1]])
B_lag = sp.Matrix([[1, -I * th / 2 - I * th * sg], [-I * th / 2, 1]])
G_lag = A_lag.inv() * B_lag
num = {th: 2, sg: sp.Rational(1, 6)}
rad = max(abs(complex(e.subs(num))) for e in G_lag.eigenvals().keys())
print(f"    lagged correction at theta=2, sigma=1/6: spectral radius = {rad:.6f} (> 1: grid mode grows)")
ok &= rad > 1
# (4) interface node mass: zones l1 + l2 = h with masses >= rho_min * l; M_I = (m1 + m2)/2
l1, rmin = sp.symbols("l1 rho_min", positive=True)
MI = (rmin * l1 + rmin * (h - l1)) / 2
print("(4) lower bound M_I = (rho_min l1 + rho_min (h - l1))/2 =", sp.simplify(MI), ">= rho_min h/4")
ok &= sp.simplify(MI - rmin * h / 2) == 0
# (5) general grid (block form, 3 zones periodic, symbolic M, H, symmetric HS): with
#     M w' = D_w^T S p,  H p' = -D_w w   (force -(p_b - p_{b-1}) = (D_w^T p)_b),  W = diag(M, H S)
#     W A + A^T W = 0, so the implicit midpoint (Cayley) step preserves w.Mw + p.(HS)p exactly
n3 = 3
Dw = sp.zeros(n3, n3)
for z in range(n3):
    Dw[z, (z + 1) % n3] += 1
    Dw[z, z] -= 1
Mm = sp.diag(*sp.symbols("M0:3", positive=True))
Hm = sp.diag(*sp.symbols("H0:3", positive=True))
a01, a02, a12, a00, a11, a22 = sp.symbols("a01 a02 a12 a00 a11 a22")
HS = sp.Matrix([[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]])      # any symmetric H S
Sm = Hm.inv() * HS
Amat = sp.BlockMatrix([[sp.zeros(n3, n3), Mm.inv() * Dw.T * Sm], [-Hm.inv() * Dw, sp.zeros(n3, n3)]]).as_explicit()
Wm = sp.diag(Mm, HS)
r5 = sp.simplify(Wm * Amat + Amat.T * Wm)
print("(5) W A + A^T W = 0 for symmetric HS:", r5 == sp.zeros(2 * n3, 2 * n3))
ok &= r5 == sp.zeros(2 * n3, 2 * n3)
print("ALL IDENTITIES OK" if ok else "IDENTITY FAILURE")
