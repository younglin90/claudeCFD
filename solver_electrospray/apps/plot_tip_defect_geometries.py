import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle
import numpy as np, math

Do, Di, Lnoz = 260e-6, 160e-6, 300e-6
ri, ro = Di/2, Do/2
um = 1e6
C_WALL="#6b6b6b"; C_BORE="#2b7bba"; C_ATM="#eef4fb"

def base(ax, title):
    ax.add_patch(Rectangle((-1.7*ro*um,(Lnoz-1.3*Lnoz)*um),3.4*ro*um,2.6*Lnoz*um,fc=C_ATM,ec="k",lw=1))
    ax.set_xlim(-1.7*ro*um,1.7*ro*um); ax.set_ylim((Lnoz-1.3*Lnoz)*um,(Lnoz+1.3*Lnoz)*um)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])

def wallcol(ax, x0, x1, ytop):
    ax.add_patch(Rectangle((x0*um,(Lnoz-1.3*Lnoz)*um),(x1-x0)*um,(ytop-(Lnoz-1.3*Lnoz))*um,fc=C_WALL))

fig, axes = plt.subplots(1,4, figsize=(15,5))

# sharp
ax=axes[0]; base(ax,"sharp (baseline)\naxisymmetric")
wallcol(ax, ri, ro, Lnoz); wallcol(ax, -ro, -ri, Lnoz)
ax.add_patch(Rectangle((-ri*um,(Lnoz-1.3*Lnoz)*um),2*ri*um,1.3*Lnoz*um,fc=C_BORE))
ax.plot([ri*um,ro*um,-ri*um,-ro*um],[Lnoz*um]*4,"rv",ms=7)
ax.text(0,(Lnoz+0.9*Lnoz)*um,"square rim",ha="center",color="r",fontsize=8.5)

# D1 blunt
ax=axes[1]; base(ax,"D1 blunting\naxisymmetric (resolve fillet)")
B=25e-6
ang=np.linspace(0,math.pi/2,30)
# right wall with rounded inner+outer top -> dome
xs=np.linspace(ri,ro,60);
def yt(r):
    t=Lnoz
    if r<ri+B: t=min(t,(Lnoz-B)+math.sqrt(max(B*B-(r-(ri+B))**2,0)))
    if r>ro-B: t=min(t,(Lnoz-B)+math.sqrt(max(B*B-(r-(ro-B))**2,0)))
    return max(t,Lnoz-B)
for r in xs:
    ax.plot([r*um,r*um],[(Lnoz-1.3*Lnoz)*um, yt(r)*um],color=C_WALL,lw=1.6)
    ax.plot([-r*um,-r*um],[(Lnoz-1.3*Lnoz)*um, yt(r)*um],color=C_WALL,lw=1.6)
ax.add_patch(Rectangle((-ri*um,(Lnoz-1.3*Lnoz)*um),2*ri*um,(Lnoz-(Lnoz-1.3*Lnoz))*um,fc=C_BORE))
ax.text(0,(Lnoz+0.9*Lnoz)*um,"rounded rim\nr_b=0..25um",ha="center",color="#080",fontsize=8.5)

# D2 tilt
ax=axes[2]; base(ax,"D2 tilt (8 deg)\nBREAKS symmetry -> full 3D")
tl=math.radians(8)
for yy in np.linspace(Lnoz-1.3*Lnoz, Lnoz, 40):
    axx=math.tan(tl)*(yy)  # lean
    ax.plot([(axx+ri)*um,(axx+ro)*um],[yy*um,yy*um],color=C_WALL,lw=1.2)
    ax.plot([(axx-ro)*um,(axx-ri)*um],[yy*um,yy*um],color=C_WALL,lw=1.2)
    ax.plot([(axx-ri)*um,(axx+ri)*um],[yy*um,yy*um],color=C_BORE,lw=1.2)
ax.annotate("",xy=(math.tan(tl)*Lnoz*um*1.0,(Lnoz+0.9*Lnoz)*um),xytext=(0,(Lnoz+0.9*Lnoz)*um),arrowprops=dict(arrowstyle="->",color="purple"))
ax.text(0,(Lnoz+1.05*Lnoz)*um,"off-axis -> plume steer",ha="center",color="purple",fontsize=8)

# D3 bump
ax=axes[3]; base(ax,"D3 protrusion\nfull 3D + LOCAL refine")
wallcol(ax, ri, ro, Lnoz); wallcol(ax, -ro, -ri, Lnoz)
ax.add_patch(Rectangle((-ri*um,(Lnoz-1.3*Lnoz)*um),2*ri*um,1.3*Lnoz*um,fc=C_BORE))
# bump on right rim
ax.add_patch(Rectangle(((ri+0.2*(ro-ri))*um, Lnoz*um),(0.55*(ro-ri))*um,40e-6*um,fc=C_WALL,ec="r",lw=1.5))
ax.annotate("micro-bump ~5-20um\nLOCAL refine ~2-5um\nfield spike",xy=((ri+0.5*(ro-ri))*um,(Lnoz+40e-6)*um),
            xytext=(-1.6*ro*um,(Lnoz+0.6*Lnoz)*um),color="r",fontsize=8,arrowprops=dict(arrowstyle="->",color="r"))

fig.suptitle("Tip-defect geometries for the mesh (gray=capillary wall void, blue=bore).  D1 axisymmetric; D2/D3 break symmetry -> full 3D", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig("/tmp/defects4_figure.png", dpi=135, bbox_inches="tight")
print("saved /tmp/defects4_figure.png")
