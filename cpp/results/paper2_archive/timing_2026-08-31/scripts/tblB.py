import re, os
D="/tmp/mbq/tblB"
TWO=[("leveque","leveque"),("config3","config3"),("shockvortex","shock-vortex"),
     ("shockmixing","shock-mixing"),("mach3","Mach 3 step"),("doublemach","double Mach")]
THREE=[("deform3d","deformation 3D"),("sphere3d","sphere 3D"),("twocyl3d","two-cylinder 3D")]
def txt(c,s): return open(f"{D}/{c}_{s}.log",encoding="utf-8",errors="replace").read()
def total(c,s):
    v=[float(x) for l in txt(c,s).splitlines() if "harness_wall" not in l
       for x in re.findall(r"wall=([0-9.]+)s", l)]
    return v[-1]
def pro(c,s):
    m=re.search(r"prologue \(scratch re-init \+ cached builds\): ([0-9]+)ms", txt(c,s))
    return float(m.group(1))/1000 if m else 0.0
def p2(c,s):
    m=re.search(r"MUSCL=([0-9.]+)s\s+THINC=([0-9.]+)s \(geom=([0-9.]+)s face=([0-9.]+)s\)\s+BVD_sel=([0-9.]+)s", txt(c,s))
    mu,th,ge,fa,se=map(float,m.groups()); return ge, mu+th+se
def p3(c,s):
    m=re.search(r"RECON breakdown \(summed ([0-9]+)ms\): o2-LSQ ([0-9]+)ms.*?cell-D-Newton ([0-9]+)ms.*?face-quad\+BVD ([0-9]+)ms", txt(c,s), re.S)
    tot,lsq,cd,fb=(float(x)/1000 for x in m.groups()); return cd, cd+fb
def r(a,b): return f"{a/b:.2f}x"
print("=== TABLE 2 (2D 500 steps / 3D 200 steps, 1 thread, MOOD off; 3D total excludes one-time precompute)")
print(f"{'Case':<16}{'tanh tot':>10}{'GAUSS tot':>11}{'tot x':>7}"
      f"{'tanh sh':>9}{'GAUSS sh':>10}{'sh x':>7}{'tanh rec':>10}{'GAUSS rec':>11}{'rec x':>7}")
for key,lab in TWO+THREE:
    is2 = any(k==key for k,_ in TWO)
    f = p2 if is2 else p3
    t1,t2 = total(key,"s1")-pro(key,"s1"), total(key,"s2")-pro(key,"s2")
    s1,c1=f(key,"s1"); s2,c2=f(key,"s2")
    print(f"{lab:<16}{t1:>9.1f}{t2:>10.1f}{r(t1,t2):>8}{s1:>9.2f}{s2:>10.2f}{r(s1,s2):>8}"
          f"{c1:>10.2f}{c2:>11.2f}{r(c1,c2):>8}")
print()
print("=== interface census (2D, from the same 500-step runs)")
for key,lab in TWO:
    m=re.search(r"interface=([0-9]+) \(([0-9.]+)%\).*?Newton iters/solve=([0-9.]+)", txt(key,"s1"))
    j=re.search(r"flat\(no jump\)=[0-9]+ \(([0-9.]+)%\)", txt(key,"s1"))
    print(f"  {lab:<15} interface={m.group(2):>6}%   Newton/solve={m.group(3)}   flat={j.group(1) if j else '?':>6}%")
