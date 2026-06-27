import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

Do, Di, Lnoz, H = 260e-6, 160e-6, 300e-6, 1.5e-3
ri, ro = Di/2, Do/2
um = 1e6
Rdom = 3*Do  # domain radius (half-width)

C_BORE="#2b7bba"; C_ATM="#eef4fb"; C_WALL="#888"; C_COLL="#d98c3f"

fig = plt.figure(figsize=(13.5, 7.2))
axF = fig.add_axes([0.05, 0.08, 0.34, 0.84])   # full domain
axT = fig.add_axes([0.48, 0.10, 0.48, 0.80])   # tip zoom

# ---------- FULL DOMAIN ----------
axF.add_patch(Rectangle((-Rdom*um, 0), 2*Rdom*um, H*um, fc=C_ATM, ec="k", lw=1.2))      # atmosphere box
axF.add_patch(Rectangle((-Rdom*um, (H-0.06e-3)*um), 2*Rdom*um, 0.06e-3*um, fc=C_COLL))  # collector strip
# capillary wall void (gray) two strips, bore (blue)
axF.add_patch(Rectangle((ri*um, 0), (ro-ri)*um, Lnoz*um, fc=C_WALL))
axF.add_patch(Rectangle((-ro*um, 0), (ro-ri)*um, Lnoz*um, fc=C_WALL))
axF.add_patch(Rectangle((-ri*um, 0), 2*ri*um, Lnoz*um, fc=C_BORE))
axF.plot([0,0],[Lnoz*um, H*um], "--", color="#39c", lw=1, alpha=0.6)  # jet axis
# red zoom box
axF.add_patch(Rectangle((-1.6*ro*um, (Lnoz-1.2*Lnoz)*um), 3.2*ro*um, 2.6*Lnoz*um, fill=False, ec="r", lw=1.6, ls="--"))
axF.text(0, (Lnoz+0.05e-3)*um, "tip zoom ->", color="r", ha="center", fontsize=8)
axF.set_xlim(-Rdom*um*1.05, Rdom*um*1.05); axF.set_ylim(-0.05e-3*um, H*um*1.04)
axF.set_xlabel("x [um] (radial)"); axF.set_ylabel("y [um]  (inlet y=0 -> collector y=H=1500)")
axF.set_title("Full domain (true scale)\nnozzle tiny vs 1.5 mm collector gap", fontsize=10)
axF.text(0, (H-0.03e-3)*um, "collector", ha="center", va="center", color="w", fontsize=8, fontweight="bold")
axF.text(Rdom*um*0.55, H*um*0.55, "atmosphere\n(gas)", fontsize=9, color="#578", ha="center")
axF.annotate("outlet\n(side)", xy=(-Rdom*um, H*um*0.5), xytext=(-Rdom*um*0.78, H*um*0.72),
             fontsize=8, color="#393", arrowprops=dict(arrowstyle="->", color="#393"))

# ---------- TIP ZOOM ----------
# regions
ax=axT
ax.add_patch(Rectangle((-1.6*ro*um, (Lnoz-1.2*Lnoz)*um), 3.2*ro*um, 2.6*Lnoz*um, fc=C_ATM, ec="k", lw=1))
ax.add_patch(Rectangle((ri*um, (Lnoz-1.2*Lnoz)*um), (ro-ri)*um, 1.2*Lnoz*um, fc=C_WALL))
ax.add_patch(Rectangle((-ro*um, (Lnoz-1.2*Lnoz)*um), (ro-ri)*um, 1.2*Lnoz*um, fc=C_WALL))
ax.add_patch(Rectangle((-ri*um, (Lnoz-1.2*Lnoz)*um), 2*ri*um, 1.2*Lnoz*um, fc=C_BORE))
ax.axhline(Lnoz*um, color="k", ls=":", lw=1)
# resolution hint: draw a fine grid over the tip band
for xx in np.arange(-1.6*ro, 1.6*ro, 11e-6):
    ax.plot([xx*um,xx*um],[(Lnoz-60e-6)*um,(Lnoz+120e-6)*um], color="#bbb", lw=0.3)
for yy in np.arange(Lnoz-60e-6, Lnoz+120e-6, 11e-6):
    ax.plot([-1.6*ro*um,1.6*ro*um],[yy*um,yy*um], color="#bbb", lw=0.3)
# labels
ax.annotate("liquid_inlet\n(bore feed, below)", xy=(0,(Lnoz-1.1*Lnoz)*um), xytext=(0,(Lnoz-0.95*Lnoz)*um),
            ha="center", color=C_BORE, fontsize=9, arrowprops=dict(arrowstyle="->", color=C_BORE))
ax.annotate("nozzle_wall\n(capillary: inner+outer\nwall + tip rim, no-slip electrode)",
            xy=((ri+ro)/2*um, (Lnoz-0.5*Lnoz)*um), xytext=(2.0*ro*um, (Lnoz-0.7*Lnoz)*um),
            color="#222", fontsize=9, arrowprops=dict(arrowstyle="->", color="#222"))
ax.text(0, (Lnoz+0.7*Lnoz)*um, "cone-jet forms\nhere (atmosphere)", ha="center", color="#578", fontsize=9)
ax.annotate("tip rim = the DEFECT site\n(D1 round it / D2 tilt / D3 bump)\nresolve wall >=3-4 cells (~10-12um)",
            xy=(ro*um, Lnoz*um), xytext=(0.2*ro*um, (Lnoz+0.45*Lnoz)*um), color="r", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="r"))
# dimension arrows
ax.annotate("", xy=(-ri*um, (Lnoz-1.12*Lnoz)*um), xytext=(ri*um, (Lnoz-1.12*Lnoz)*um),
            arrowprops=dict(arrowstyle="<->", color="k"))
ax.text(0, (Lnoz-1.18*Lnoz)*um, "Di=160um (bore)", ha="center", fontsize=8)
ax.annotate("", xy=(ri*um, (Lnoz-0.15*Lnoz)*um), xytext=(ro*um, (Lnoz-0.15*Lnoz)*um),
            arrowprops=dict(arrowstyle="<->", color="darkred"))
ax.text((ri+ro)/2*um, (Lnoz-0.08*Lnoz)*um, "wall 50um\n>=4 cells!", ha="center", fontsize=8, color="darkred")
ax.set_xlim(-1.6*ro*um, 1.6*ro*um); ax.set_ylim((Lnoz-1.2*Lnoz)*um, (Lnoz+1.4*Lnoz)*um)
ax.set_aspect("equal"); ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
ax.set_title("Tip zoom: regions + patches + resolution\n(gray = capillary WALL = void, surface is nozzle_wall)", fontsize=10)

fig.suptitle("Resolved-nozzle mesh construction:  fluid = bore + atmosphere;  capillary wall = void (its surface = nozzle_wall);  refine the tip", fontsize=12, y=0.99)
fig.savefig("/tmp/mesh_guide_figure.png", dpi=135, bbox_inches="tight")
print("saved /tmp/mesh_guide_figure.png")
