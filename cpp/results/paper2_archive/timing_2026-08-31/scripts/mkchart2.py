# Reconstruction cost, tanh against GAUSS, forced THINC, 500 iterations.
#
# Log scale on both panels: the 3D cases run ~100x longer than the 2D ones, and on a linear axis
# the 2D bars vanish. The 2D numbers come from the single-core runs and the 3D from 24 threads --
# 500 forced steps of a 4M-cell mesh on one core is >15h -- so each group is annotated with its
# thread count and the panels are not compared across that boundary.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TWO = [("leveque","LeVeque"),("config3","cfg3"),("shockvortex","shock-vortex"),
       ("shockmixing","shock/mixing"),("mach3","Mach 3 step"),("doublemach","double Mach")]
THREE = [("deform3d","deformation 3D"),("twocyl3d","two-cylinder 3D"),("sphere3d","sphere 3D")]
D2, D3 = "/tmp/mbq/chart2D1T", "/tmp/mbq/chartF"

def read(d,c,s):
    x=open(f"{d}/{c}_{s}.log",encoding="utf-8",errors="replace").read()
    T=int(re.search(r"threads=(\d+)",x).group(1))
    if c.endswith("3d"):
        m=re.search(r"RECON breakdown \(summed (\d+)ms\): o2-LSQ (\d+)ms.*?cell-D-Newton (\d+)ms"
                    r".*?face-quad\+BVD (\d+)ms",x,re.S)
        _,lsq,cd,fb=(float(v)/1000 for v in m.groups()); rec=lsq+cd+fb
        k=re.search(r"kernels \(RDTSC\): D-solve ([0-9.]+)ms\s+face-value ([0-9.]+)ms",x)
        ds,fv=float(k.group(1))/1000,float(k.group(2))/1000
    else:
        m=re.search(r"MUSCL=([0-9.]+)s\s+THINC=([0-9.]+)s \(geom=([0-9.]+)s face=([0-9.]+)s\)"
                    r"\s+BVD_sel=([0-9.]+)s",x)
        mu,th,ge,fa,se=map(float,m.groups()); rec=mu+th+se
        ds=float(re.search(r"\[CHENG3_K\] D-solve ([0-9.]+)ms",x).group(1))/1000; fv=float('nan')
    ifr=re.search(r"interface=\d+ \(([0-9.]+)%\)",x)
    if not ifr:
        so=re.search(r"over (\d+) solves",x); ce=re.search(r"cells=(\d+)",x)
        nv=1 if c=="deform3d" else 5
        ifr=100*int(so.group(1))/(1500*nv*2*int(ce.group(1))) if so and ce else float('nan')
    else: ifr=float(ifr.group(1))
    return dict(T=T,recon=rec,dsolve=ds,faceval=fv,ifr=ifr)

rows=[(lab,read(D2,k,"s1"),read(D2,k,"s2")) for k,lab in TWO] + \
     [(lab,read(D3,k,"s1"),read(D3,k,"s2")) for k,lab in THREE]
labs=[r[0] for r in rows]; idx=np.arange(len(rows)); w=.38
fig,ax=plt.subplots(2,1,figsize=(13,9))

def panel(a, key, title, ylab):
    t=[r[1][key] for r in rows]; g=[r[2][key] for r in rows]
    a.bar(idx-w/2,t,w,label="tanh (quadrature + Newton)",color="#C0392B")
    a.bar(idx+w/2,g,w,label="GAUSS (moments + closed form)",color="#1F618D")
    for i,(x,y) in enumerate(zip(t,g)):
        a.text(i, max(x,y)*1.25, f"{x/y:.2f}x", ha="center", fontsize=9, fontweight="bold")
    a.set_yscale("log"); a.set_ylabel(ylab); a.set_title(title, fontsize=11)
    a.set_xticks(idx); a.set_xticklabels(
        [f"{l}\n({r[1]['T']}T, {r[1]['ifr']:.0f}%)" for l,r in zip(labs,rows)], fontsize=8)
    a.legend(fontsize=8); a.grid(axis="y", alpha=.3, which="both")
    a.axvline(5.5, color="k", lw=.8, ls="--", alpha=.5)
    a.text(2.5, a.get_ylim()[1]*.45, "2D", ha="center", fontsize=11, alpha=.5)
    a.text(7.0, a.get_ylim()[1]*.45, "3D", ha="center", fontsize=11, alpha=.5)

panel(ax[0], "recon",  "A  Reconstruction wall time, 500 iterations, forced THINC "
      "(x-labels: threads used, fraction of cells reconstructed)", "wall time (s, log)")
panel(ax[1], "dsolve", "B  Shift-D kernel alone (RDTSC, CPU time summed over threads). "
      "The ratio here does not depend on the thread count; the one in panel A does",
      "CPU time (s, log)")
fig.tight_layout()
out="/tmp/mbq/chartF/timing_breakdown.png"
fig.savefig(out,dpi=150)
print("Plot saved:",out)
print(f"{'case':<18}{'T':>3}{'if%':>7}{'recon_t':>10}{'recon_g':>10}{'rec x':>7}"
      f"{'Dsolve_t':>10}{'Dsolve_g':>10}{'D x':>7}")
for lab,a,b in rows:
    print(f"{lab:<18}{a['T']:>3}{a['ifr']:>7.1f}{a['recon']:>10.1f}{b['recon']:>10.1f}"
          f"{a['recon']/b['recon']:>7.2f}{a['dsolve']:>10.1f}{b['dsolve']:>10.1f}"
          f"{a['dsolve']/b['dsolve']:>7.1f}")
