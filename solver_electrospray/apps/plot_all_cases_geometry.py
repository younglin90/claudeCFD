import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np, math

Do, Di, Lnoz = 260e-6, 160e-6, 300e-6
ri, ro = Di/2, Do/2
um = 1e6
C_WALL="#6b6b6b"; C_BORE="#2b7bba"; C_ATM="#eef4fb"
yb = Lnoz - 1.0*Lnoz   # bottom of drawn tip region
ytop_view = Lnoz + 0.55*Lnoz

def wall_top(r, B):
    t = Lnoz
    if B > 0:
        if r < ri + B:
            d = B*B - (r-(ri+B))**2; t = min(t, (Lnoz-B)+(math.sqrt(d) if d>0 else -1e9))
        if r > ro - B:
            d = B*B - (r-(ro-B))**2; t = min(t, (Lnoz-B)+(math.sqrt(d) if d>0 else -1e9))
        t = max(t, Lnoz-B)
    return t

def draw(ax, title, blunt=0.0, tilt_deg=0.0, bump=0.0):
    ax.add_patch(plt.Rectangle((-1.8*ro*um, yb*um), 3.6*ro*um, (ytop_view-yb)*um, fc=C_ATM, ec="k", lw=0.8))
    tl = math.tan(math.radians(tilt_deg))
    # bore (blue) leaning parallelogram
    bore = [(tl*yb-ri)*um, yb*um], [(tl*yb+ri)*um, yb*um], [(tl*Lnoz+ri)*um, Lnoz*um], [(tl*Lnoz-ri)*um, Lnoz*um]
    ax.add_patch(Polygon(bore, closed=True, fc=C_BORE))
    # wall strips (gray), filleted top, leaning
    for r in np.linspace(ri, ro, 46):
        wt = wall_top(r, blunt)
        for s in (1, -1):
            ax.plot([(tl*yb + s*r)*um, (tl*wt + s*r)*um], [yb*um, wt*um], color=C_WALL, lw=3.0, solid_capstyle="butt")
    # bump (D3): solid block on the +x rim
    if bump > 0:
        rb0, rb1 = ri+0.18*(ro-ri), ri+0.78*(ro-ri)
        ax.add_patch(plt.Rectangle((rb0*um, Lnoz*um), (rb1-rb0)*um, bump*um, fc=C_WALL, ec="r", lw=1.2))
    ax.axhline(Lnoz*um, color="k", ls=":", lw=0.6)
    ax.set_xlim(-1.8*ro*um, 1.8*ro*um); ax.set_ylim(yb*um, ytop_view*um)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)

fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.2))
# Row 1: baseline + D1 blunting sweep
draw(axes[0,0], "C0 sharp\n(baseline)")
draw(axes[0,1], "C1  D1 r_b=8um",  blunt=8e-6)
draw(axes[0,2], "C2  D1 r_b=15um", blunt=15e-6)
draw(axes[0,3], "C3  D1 r_b=20um", blunt=20e-6)
draw(axes[0,4], "C4  D1 r_b=25um", blunt=25e-6)
# Row 2: baseline ref + D2 tilt sweep
draw(axes[1,0], "C0 sharp (ref)")
draw(axes[1,1], "C5  D2 tilt 2deg", tilt_deg=2)
draw(axes[1,2], "C6  D2 tilt 5deg", tilt_deg=5)
draw(axes[1,3], "C7  D2 tilt 10deg", tilt_deg=10)
axes[1,4].axis("off")
# Row 3: baseline ref + D3 bump sweep
draw(axes[2,0], "C0 sharp (ref)")
draw(axes[2,1], "C8  D3 bump 5um",  bump=5e-6)
draw(axes[2,2], "C9  D3 bump 10um", bump=10e-6)
draw(axes[2,3], "C10 D3 bump 20um", bump=20e-6)
axes[2,4].axis("off")
# row family labels
axes[0,0].set_ylabel("D1 blunting\n(rim fillet, axisym.)", fontsize=10)
axes[1,0].set_ylabel("D2 tilt\n(off-axis, 3D)", fontsize=10)
axes[2,0].set_ylabel("D3 bump\n(protrusion, 3D)", fontsize=10)
for a in (axes[0,0],axes[1,0],axes[2,0]): a.set_yticks([]); a.yaxis.label.set_color("k")

fig.suptitle("All 11 emitter-tip cases (x-y tip cross-section).  gray = capillary wall (void), blue = bore.  Same Di/Do/Lnoz/H; only the tip differs.",
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("/tmp/cases11_figure.png", dpi=125, bbox_inches="tight")
print("saved /tmp/cases11_figure.png")
