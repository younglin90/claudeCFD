import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import math

Do, Di = 260e-6, 160e-6
ri, ro, Lnoz = 0.5*Di, 0.5*Do, 300e-6
Lx = Lz = 4.0*Do
Ly = 1.5e-3
NX = NY = NZ = 20
dx, dy, dz = Lx/NX, Ly/NY, Lz/NZ
cx, cz = 0.5*Lx, 0.5*Lz
um = 1e6

C_BORE = "#2b7bba"   # liquid feed (bore inside the capillary)
C_WALL = "#5c5c5c"   # capillary wall (excluded solid -> nozzle_wall)
C_ATM  = "#eef4fb"   # atmosphere fluid
C_COLL = "#d98c3f"   # collector

def kind_xy(i, j):
    x = (i+0.5)*dx; y = (j+0.5)*dy
    r = abs(x - cx)
    if (ri < r < ro) and (y <= Lnoz):
        return "wall"
    if (r <= ri) and (y <= Lnoz):
        return "bore"         # liquid feed channel inside the capillary only
    return "atm"

def kind_xz(i, k):
    x = (i+0.5)*dx; z = (k+0.5)*dz
    r = ((x-cx)**2 + (z-cz)**2)**0.5
    if ri < r < ro: return "wall"
    if r <= ri:     return "bore"
    return "atm"

COL = {"wall": C_WALL, "bore": C_BORE, "atm": C_ATM}

fig = plt.figure(figsize=(13.5, 6.6))
ax1 = fig.add_axes([0.05, 0.12, 0.30, 0.78])   # full x-y
ax2 = fig.add_axes([0.40, 0.12, 0.26, 0.78])   # zoom x-y (nozzle)
ax3 = fig.add_axes([0.71, 0.30, 0.26, 0.46])   # x-z at nozzle

# ---- ax1: full x-y (true proportions) ----
for i in range(NX):
    for j in range(NY):
        k = kind_xy(i, j); col = COL[k]
        if j == NY-1 and k == "atm": col = C_COLL
        ax1.add_patch(Rectangle((i*dx*um, j*dy*um), dx*um, dy*um,
                                facecolor=col, edgecolor="#dde", linewidth=0.25))
ax1.add_patch(Rectangle((0, 0), Lx*um, (Lnoz+1.6*dy)*um, fill=False,
                        edgecolor="red", linewidth=1.4, linestyle="--"))
ax1.set_xlim(0, Lx*um); ax1.set_ylim(0, Ly*um)
ax1.set_xlabel("x [um]"); ax1.set_ylabel("y [um]")
ax1.set_title("full domain (x-y)\ninlet y=0 -> collector y=1500", fontsize=10)
ax1.text(cx*um, (Ly-0.5*dy)*um, "collector", ha="center", va="center", color="w",
         fontsize=8, fontweight="bold")
ax1.text(20, Ly*um*0.55, "atmosphere\n(outlet sides)", fontsize=8, color="#5a7")
ax1.text(Lx*um*0.62, 150*um*0.0+700, "<- red box =\n   nozzle zoom", fontsize=8, color="red")

# ---- ax2: zoom of the nozzle region (y 0..560 um) ----
ymax_zoom = 560e-6
for i in range(NX):
    for j in range(NY):
        if (j+0.5)*dy > ymax_zoom: continue
        k = kind_xy(i, j)
        ax2.add_patch(Rectangle((i*dx*um, j*dy*um), dx*um, dy*um,
                                facecolor=COL[k], edgecolor="#ccd", linewidth=0.5))
ax2.axhline(Lnoz*um, color="k", ls=":", lw=1)
ax2.text(Lx*um*0.97, Lnoz*um+8, "nozzle exit (y=Lnoz=300um)", ha="right", fontsize=8)
ax2.annotate("liquid_inlet\n(bore bottom, y=0)", xy=(cx*um, 6),
             xytext=(cx*um, 175*um*0.0+115), ha="center", fontsize=8.5, color=C_BORE,
             arrowprops=dict(arrowstyle="->", color=C_BORE))
ax2.annotate("nozzle_wall\n(capillary)", xy=((cx-0.5*(ri+ro))*um, 110),
             xytext=(15, 430), fontsize=8.5, color="#333",
             arrowprops=dict(arrowstyle="->", color="#333"))
ax2.text(cx*um, 470, "jet region\n(atmosphere\nabove exit)", ha="center", fontsize=8, color="#578")
ax2.set_xlim(0, Lx*um); ax2.set_ylim(0, ymax_zoom*um); ax2.set_aspect("equal")
ax2.set_xlabel("x [um]"); ax2.set_ylabel("y [um]")
ax2.set_title("nozzle zoom (x-y)\nbore feed + capillary walls", fontsize=10)

# ---- ax3: x-z section at the nozzle ----
for i in range(NX):
    for k in range(NZ):
        ax3.add_patch(Rectangle((i*dx*um, k*dz*um), dx*um, dz*um,
                                facecolor=COL[kind_xz(i, k)], edgecolor="#ccd", linewidth=0.4))
th = [t/120*6.2832 for t in range(121)]
ax3.plot([(cx+ri*math.cos(t))*um for t in th], [(cz+ri*math.sin(t))*um for t in th], "w-", lw=1.6)
ax3.plot([(cx+ro*math.cos(t))*um for t in th], [(cz+ro*math.sin(t))*um for t in th], "k-", lw=1.1)
ax3.set_xlim(0, Lx*um); ax3.set_ylim(0, Lz*um); ax3.set_aspect("equal")
ax3.set_xlabel("x [um]"); ax3.set_ylabel("z [um]")
ax3.set_title("x-z section at nozzle\nbore Di=160 / wall to Do=260 um", fontsize=10)

handles = [
    mlines.Line2D([], [], marker="s", ls="", mfc=C_BORE, mec="#ccd", ms=12, label="liquid_inlet / bore (liquid feed)"),
    mlines.Line2D([], [], marker="s", ls="", mfc=C_WALL, mec="#ccd", ms=12, label="nozzle_wall (capillary, excluded solid)"),
    mlines.Line2D([], [], marker="s", ls="", mfc=C_ATM,  mec="#ccd", ms=12, label="atmosphere (fluid; outlet on sides)"),
    mlines.Line2D([], [], marker="s", ls="", mfc=C_COLL, mec="#ccd", ms=12, label="collector"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Generated resolved-nozzle OpenFOAM mesh  -  7952 cells, named patches  (Candido Di=160 / Do=260 / Lnoz=300 um, collector 1.5 mm)",
             fontsize=12, y=0.985)
fig.savefig("/tmp/nozzle_mesh_figure.png", dpi=130, bbox_inches="tight")
print("saved /tmp/nozzle_mesh_figure.png")
