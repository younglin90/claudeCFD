import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Do, Di, Lnoz = 260.0, 160.0, 300.0  # um
ro, ri = Do/2, Di/2
CW = "#808080"; CB = "#2b7bba"
T0, T1 = 0.0, 1.5*np.pi   # 3/4 cutaway (wedge open toward viewer)

def shear(X, Z, tl):       # tilt: lean +x with height
    return X + Z*tl

def surf(ax, X, Y, Z, color, alpha=1.0):
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True, shade=True)

def tube(ax, tl=0.0, blunt=0.0, bump=0.0):
    n = 64
    t = np.linspace(T0, T1, n)
    ztop = Lnoz-blunt if blunt > 0 else Lnoz
    zo = np.linspace(0, ztop, 2); To, Zo = np.meshgrid(t, zo)
    surf(ax, shear(ro*np.cos(To), Zo, tl), ro*np.sin(To), Zo, CW)       # outer wall
    z = np.linspace(0, Lnoz, 2); T, Z = np.meshgrid(t, z)
    surf(ax, shear(ri*np.cos(T), Z, tl), ri*np.sin(T), Z, CW)           # inner bore wall
    surf(ax, shear(ri*0.9*np.cos(T), Z, tl), ri*0.9*np.sin(T), Z, CB)   # bore liquid
    r2 = np.linspace(ri, ro, 2); Tb, Rb = np.meshgrid(t, r2)
    surf(ax, shear(Rb*np.cos(Tb), 0, tl), Rb*np.sin(Tb), np.zeros_like(Tb), CW)  # bottom rim
    if blunt <= 0:
        surf(ax, shear(Rb*np.cos(Tb), Lnoz, tl), Rb*np.sin(Tb), np.full_like(Tb, Lnoz), CW)  # flat top rim
    else:
        pp = np.linspace(0, np.pi, 14); TT, PP = np.meshgrid(t, pp)
        rc = 0.5*(ri+ro); br = 0.5*(ro-ri)
        R = rc + br*np.cos(PP); Zt = (Lnoz-br) + br*np.sin(PP)
        surf(ax, shear(R*np.cos(TT), Zt, tl), R*np.sin(TT), Zt, CW)     # rounded rim
    for th in (T0, T1):
        c, s = np.cos(th), np.sin(th)
        v = [[(shear(ri*c, 0, tl), ri*s, 0), (shear(ro*c, 0, tl), ro*s, 0),
              (shear(ro*c, Lnoz, tl), ro*s, Lnoz), (shear(ri*c, Lnoz, tl), ri*s, Lnoz)]]
        ax.add_collection3d(Poly3DCollection(v, facecolor=CW, edgecolor="none"))
    if bump > 0:
        rc = 0.5*(ri+ro)
        ax.bar3d(shear(rc, Lnoz, tl)-14, -14, Lnoz, 28, 28, bump, color="r", shade=True)

def setup(ax, zmax=Lnoz+170, box=(1, 1, 1.0)):
    ax.set_box_aspect(box); ax.set_xlim(-ro*1.5, ro*1.5); ax.set_ylim(-ro*1.5, ro*1.5)
    ax.set_zlim(-60, zmax); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()

fig = plt.figure(figsize=(16, 7.5))

ax = fig.add_subplot(1, 2, 1, projection="3d")
tube(ax)
ax.quiver(0, 0, Lnoz+40, 0, 0, 150, color="b", lw=1.5, arrow_length_ratio=0.18)
ax.text(0, 0, Lnoz+260, "cone-jet -> collector\n(1.5 mm above, not shown)", color="#345", ha="center", fontsize=10)
ax.text(0, ri+22, Lnoz*0.45, "bore\n(liquid feed)", color=CB, fontsize=10)
ax.text(ro+30, 0, Lnoz*0.5, "capillary wall\n= nozzle_wall", color="k", fontsize=10)
ax.text(0, 0, -85, "liquid_inlet (bore bottom)", color=CB, ha="center", fontsize=9)
ax.set_title("Resolved nozzle in 3D  (3/4 cutaway)\nhollow capillary tube + liquid bore, facing the collector", fontsize=11)
setup(ax, box=(1, 1, 1.15))

specs = [("sharp (baseline)", {}), ("D1  blunt  r_b=25um", dict(blunt=25.0)),
         ("D2  tilt 10deg", dict(tl=np.tan(np.radians(10)))), ("D3  bump 20um", dict(bump=20.0))]
for i, (name, kw) in enumerate(specs):
    a = fig.add_subplot(2, 4, 3+(i % 2)+4*(i//2), projection="3d")
    tube(a, tl=kw.get("tl", 0.0), blunt=kw.get("blunt", 0.0), bump=kw.get("bump", 0.0))
    a.set_title(name, fontsize=10); setup(a, box=(1, 1, 0.95))

fig.suptitle("Emitter geometry in 3D: a hollow capillary tube (gray) with a liquid bore (blue) facing the collector;  each defect modifies only the TIP", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/tmp/nozzle_3d.png", dpi=130, bbox_inches="tight")
print("saved /tmp/nozzle_3d.png")
