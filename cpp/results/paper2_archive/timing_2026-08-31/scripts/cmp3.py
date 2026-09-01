# Three-way comparison, to separate the gate from the thread count.
#   gated / 1 thread    (tblB)      -- the defensible configuration
#   forced / 16-24 thr  (chartF)    -- interface fraction raised, but bandwidth-bound
#   forced / 1 thread   (chart2D1T) -- interface fraction raised, kernel-bound
# Only one variable changes between each pair, so whichever moves the ratio is the cause.
import re, os
SETS = [("gated 1T", "/tmp/mbq/tblB"),
        ("forced MT", "/tmp/mbq/chartF"),
        ("forced 1T", "/tmp/mbq/chart2D1T")]
CASES = [("leveque","LeVeque"),("config3","cfg3"),("shockvortex","shock-vortex"),
         ("shockmixing","shock/mixing"),("mach3","Mach 3"),("doublemach","double Mach")]
def parse(d,c,s):
    f=f"{d}/{c}_{s}.log"
    if not os.path.exists(f): return None
    x=open(f,encoding="utf-8",errors="replace").read()
    m=re.search(r"MUSCL=([0-9.]+)s\s+THINC=([0-9.]+)s \(geom=([0-9.]+)s face=([0-9.]+)s\)\s+BVD_sel=([0-9.]+)s",x)
    if not m: return None
    mu,th,ge,fa,se=map(float,m.groups())
    w=[float(v) for l in x.splitlines() if "harness_wall" not in l
       for v in re.findall(r"wall=([0-9.]+)s", l)]
    k=re.search(r"\[CHENG3_K\] D-solve ([0-9.]+)ms",x)
    i=re.search(r"interface=\d+ \(([0-9.]+)%\)",x)
    T=re.search(r"threads=(\d+)",x)
    return dict(total=w[-1] if w else float('nan'), recon=mu+th+se, geom=ge, face=fa,
                dsolve=float(k.group(1))/1000 if k else None,
                ifr=float(i.group(1)) if i else float('nan'),
                T=int(T.group(1)) if T else 1)
print(f"{'case':<14}{'config':<12}{'T':>3}{'if%':>7}{'recon x':>9}{'total x':>9}{'D-solve x':>11}")
for key,lab in CASES:
    for name,d in SETS:
        a,b=parse(d,key,"s1"),parse(d,key,"s2")
        if not a or not b: print(f"{lab:<14}{name:<12}{'--- not available yet ---':>30}"); continue
        dx = f"{a['dsolve']/b['dsolve']:.1f}x" if (a['dsolve'] and b['dsolve']) else "  n/a"
        print(f"{lab:<14}{name:<12}{a['T']:>3}{a['ifr']:>7.1f}"
              f"{a['recon']/b['recon']:>9.2f}{a['total']/b['total']:>9.2f}{dx:>11}")
    print()
