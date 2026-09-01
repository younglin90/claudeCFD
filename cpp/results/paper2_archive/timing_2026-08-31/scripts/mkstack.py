# Reconstruction time split into the two kernels the closed form replaces and everything else.
#
# Three buckets, identical in 2D and 3D and summing exactly to the reconstruction time:
#   shift D-solve   : tanh = cell quadrature + Newton; GAUSS = moment contraction + closed form
#   face value      : tanh = face quadrature + tanh average; GAUSS = closed-form face moment
#   rest            : LSQ fit, limiter, interface detection, candidate assembly, BVD selection
# The split stops here on purpose. Going finer would need the 3D smooth candidate timed, and that
# is ~10 flops against a 40-cycle RDTSC pair -- the measurement would cost four times the thing
# measured. The 2D and 3D code also separate different stages natively (2D isolates MUSCL, 3D
# isolates the LSQ), so any finer correspondence would be an artefact of one implementation.
#
# Everything here is WALL time. The kernel timers read the invariant TSC, which advances with
# elapsed time rather than with issued instructions; summing it across threads would turn it into
# CPU time, but these runs use one thread, so there is nothing to sum and the reading is the
# elapsed time of the region. That is also what makes the three buckets add up to the
# reconstruction time. Forced THINC, so the kernels actually run.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "/tmp/mbq/stack"
TWO = [("leveque","LeVeque"),("config3","cfg3"),("shockvortex","shock-vortex"),
       ("shockmixing","shock/mixing"),("mach3","Mach 3 step"),("doublemach","double Mach")]
THREE = [("deform3d","deformation 3D"),("twocyl3d","two-cylinder 3D"),("sphere3d","sphere 3D")]

def read(c,s):
    x=open(f"{D}/{c}_{s}.log",encoding="utf-8",errors="replace").read()
    if c.endswith("3d"):
        m=re.search(r"RECON breakdown \(summed (\d+)ms\): o2-LSQ (\d+)ms.*?cell-D-Newton (\d+)ms"
                    r".*?face-quad\+BVD (\d+)ms",x,re.S)
        rec=sum(float(v)/1000 for v in m.groups()[1:])
        k=re.search(r"kernels \(RDTSC\): D-solve ([0-9.]+)ms\s+face-value ([0-9.]+)ms",x)
        ds,fv=float(k.group(1))/1000,float(k.group(2))/1000
        so=re.search(r"over (\d+) solves",x); ce=re.search(r"cells=(\d+)",x)
        st=int(re.search(r"steps=(\d+)",x).group(1)); nv=1 if c=="deform3d" else 5
        ifr=100*int(so.group(1))/(3*st*nv*2*int(ce.group(1))) if so else float('nan')
    else:
        m=re.search(r"MUSCL=([0-9.]+)s\s+THINC=([0-9.]+)s \(geom=([0-9.]+)s face=([0-9.]+)s\)"
                    r"\s+BVD_sel=([0-9.]+)s",x)
        mu,th,ge,fa,se=map(float,m.groups()); rec=mu+th+se
        k=re.search(r"\[CHENG3_K\] D-solve ([0-9.]+)ms\s+face-value ([0-9.]+)ms",x)
        ds,fv=float(k.group(1))/1000,float(k.group(2))/1000
        ifr=float(re.search(r"interface=\d+ \(([0-9.]+)%\)",x).group(1))
    return dict(recon=rec, dsolve=ds, faceval=fv, rest=max(rec-ds-fv,0.0), ifr=ifr)

COL = ["#C0392B", "#E67E22", "#AEB6BF"]
NAME = ["shift D-solve", "face value", "rest of reconstruction"]

def stack(a, cases, title, norm=False):
    labs=[]; xs=[]; x=0
    for key,lab in cases:
        for s,kern in (("s1","tanh"),("s2","GAUSS")):
            d=read(key,s)
            if d["ifr"]<0: d["ifr"]=read(key,"s1")["ifr"]
            tot=d["recon"] if not norm else d["recon"]/100.0
            vals=[d["dsolve"], d["faceval"], d["rest"]]
            if norm: vals=[v/d["recon"]*100 for v in vals]
            bot=0
            for v,c,n in zip(vals,COL,NAME):
                a.bar(x, v, .8, bottom=bot, color=c, label=n if x==0 else None)
                bot+=v
            labs.append(f"{lab}\n{kern}"); xs.append(x); x+=1
        x+=0.6
    a.set_xticks(xs); a.set_xticklabels(labs, fontsize=6.5, rotation=90)
    a.set_title(title, fontsize=10); a.grid(axis="y", alpha=.3)
    return a

fig = plt.figure(figsize=(15,10))
gs = fig.add_gridspec(2,3)
a1 = fig.add_subplot(gs[0,:2]); a2 = fig.add_subplot(gs[0,2]); a3 = fig.add_subplot(gs[1,:])
stack(a1, TWO,   "A  2D, 500 iterations, 1 thread, forced THINC"); a1.set_ylabel("recon wall time (s)")
stack(a2, THREE, "B  3D, 20 iterations, 1 thread");                a2.set_ylabel("recon wall time (s)")
stack(a3, TWO+THREE, "C  Same data as a share of the reconstruction "
      "(the two red/orange bands are what the closed form replaces)", norm=True)
a3.set_ylabel("share of reconstruction (%)"); a3.set_ylim(0,100)
a1.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out="/tmp/mbq/stack/recon_stack.png"; fig.savefig(out,dpi=150)
print("Plot saved:",out)
print(f"{'case':<17}{'kern':<7}{'if%':>6}{'recon':>9}{'Dsolve':>9}{'faceval':>9}{'rest':>9}"
      f"{'kern%':>7}")
for key,lab in TWO+THREE:
    for s,kern in (("s1","tanh"),("s2","GAUSS")):
        d=read(key,s)
        if d["ifr"]<0: d["ifr"]=read(key,"s1")["ifr"]
        print(f"{lab:<17}{kern:<7}{d['ifr']:>6.1f}{d['recon']:>9.1f}{d['dsolve']:>9.1f}"
              f"{d['faceval']:>9.1f}{d['rest']:>9.1f}{100*(d['dsolve']+d['faceval'])/d['recon']:>7.1f}")
