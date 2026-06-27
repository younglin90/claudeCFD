import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Do, Di = 260e-6, 160e-6
ri, ro, Lnoz = 0.5*Di, 0.5*Do, 300e-6
um = 1e6

def wall_top(r, B):
    # top-of-wall y(r) for the capillary wall ri<r<ro, blunted with fillet radius B.
    if B <= 0:
        return Lnoz
    yt = Lnoz
    if r < ri + B:  # inner fillet
        d = B*B - (r-(ri+B))**2
        yt = min(yt, (Lnoz-B) + (d**0.5 if d > 0 else -1e9))
    if r > ro - B:  # outer fillet
        d = B*B - (r-(ro-B))**2
        yt = min(yt, (Lnoz-B) + (d**0.5 if d > 0 else -1e9))
    return max(yt, Lnoz - B)

fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
for ax, (B, title) in zip(axes, [(0, "sharp tip (baseline)"), (25e-6, "blunt25 (AO-eroded, fillet 25um)")]):
    rs = np.linspace(ri, ro, 200)
    yt = np.array([wall_top(r, B) for r in rs])
    # right wall strip: x = cx + r ; mirror for left. Use a local frame: x = r (relative to axis).
    # fill wall (gray) from y=150um up to yt(r), for both +r and -r strips.
    ax.fill_between(rs*um, 150, yt*um, color="#5c5c5c", label="nozzle_wall")
    ax.fill_between(-rs*um, 150, yt*um, color="#5c5c5c")
    # bore (blue) -ri..ri up to Lnoz
    ax.fill_between([-ri*um, ri*um], 150, Lnoz*um, color="#2b7bba", alpha=0.5, label="bore (liquid)")
    # atmosphere above tip
    ax.axhline(Lnoz*um, color="k", ls=":", lw=0.8)
    # field-concentration markers at the rim corners
    if B <= 0:
        ax.plot([ri*um, ro*um, -ri*um, -ro*um], [Lnoz*um]*4, "rv", ms=9)
        ax.text(0, (Lnoz+18e-6)*um, "sharp rim:\nstrong field E~1/r_edge", ha="center", color="r", fontsize=9)
    else:
        ax.text(0, (Lnoz+18e-6)*um, "rounded rim:\nrelaxed field", ha="center", color="#080", fontsize=9)
    ax.set_xlim(-ro*um*1.25, ro*um*1.25); ax.set_ylim(150, 360)
    ax.set_xlabel("radial position [um]"); ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
axes[0].set_ylabel("y [um]  (nozzle exit y=Lnoz=300)")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("D1 tip-blunting defect on the Candido capillary (x-y tip cross-section)\n"
             "AO erosion rounds the field-concentrating rim -> lower tip field -> altered cone-jet/plume",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/tmp/d1_tip_figure.png", dpi=140, bbox_inches="tight")
print("saved /tmp/d1_tip_figure.png")
