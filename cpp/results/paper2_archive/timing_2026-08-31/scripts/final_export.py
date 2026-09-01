# Refresh the CSV with the 100%-coverage runs and rebuild the paper table from it.
#
# forced100 = every cell reconstructed, including those whose least-squares gradient is exactly
# zero (given an arbitrary plane there). That removes the cell-count difference between cases --
# all nine now do identical work -- so what the timings compare is the kernel and nothing else.
# It is NOT the published scheme: it reconstructs where THINC is undefined and it moves the
# answer, so the caption has to say so.
import re, os, csv, shutil, glob
ARC="/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_archive/timing_2026-08-31"
DST="/mnt/d/datas/paper2_timing_2026-09-01"
SETS=[("gated_1T","/tmp/mbq/tblB",500,200,"gated"),
      ("forced_MT","/tmp/mbq/chartF",500,500,"forced"),
      ("forced_1T","/tmp/mbq/chart2D1T",500,None,"forced"),
      ("forced100_1T","/tmp/mbq/stack",500,20,"forced100")]
C2=["leveque","config3","shockvortex","shockmixing","mach3","doublemach"]
C3=["deform3d","twocyl3d","sphere3d"]
def parse(d,c,s):
    f=f"{d}/{c}_{s}.log"
    if not os.path.exists(f): return None
    x=open(f,encoding="utf-8",errors="replace").read(); r={}
    m=re.search(r"threads=(\d+)",x); r["threads"]=int(m.group(1)) if m else 1
    w=[float(v) for l in x.splitlines() if "harness_wall" not in l
       for v in re.findall(r"wall=([0-9.]+)s",l)]
    r["total_wall_s"]=w[-1] if w else ""
    if c in C3:
        m=re.search(r"RECON breakdown \(summed (\d+)ms\): o2-LSQ (\d+)ms.*?cell-D-Newton (\d+)ms"
                    r".*?face-quad\+BVD (\d+)ms",x,re.S)
        if m:
            _,l,cd,fb=(float(v)/1000 for v in m.groups())
            r.update(o2_lsq_s=l,cellD_bucket_s=cd,facebvd_s=fb,recon_s=l+cd+fb)
        for k,pat in (("dsolve_s",r"D-solve ([0-9.]+)ms"),("faceval_s",r"face-value ([0-9.]+)ms"),
                      ("onetime_precompute_s",r"prologue[^:]*: (\d+)ms")):
            mm=re.search(pat,x)
            if mm: r[k]=float(mm.group(1))/1000
        for k,pat in (("newton_solves",r"over (\d+) solves"),("cells",r"cells=(\d+)")):
            mm=re.search(pat,x)
            if mm: r[k]=int(mm.group(1))
        mm=re.search(r"enstrophy=([0-9.]+)",x)
        if mm: r["enstrophy"]=float(mm.group(1))
        if r.get("newton_solves") and r.get("cells"):
            st=int(re.search(r"steps=(\d+)",x).group(1)); nv=1 if c=="deform3d" else 5
            r["interface_pct"]=100*r["newton_solves"]/(3*st*nv*2*r["cells"])
    else:
        m=re.search(r"MUSCL=([0-9.]+)s\s+THINC=([0-9.]+)s \(geom=([0-9.]+)s face=([0-9.]+)s\)"
                    r"\s+BVD_sel=([0-9.]+)s",x)
        if m:
            mu,th,ge,fa,se=map(float,m.groups())
            r.update(muscl_s=mu,thinc_s=th,geom_bucket_s=ge,face_bucket_s=fa,bvd_sel_s=se,
                     recon_s=mu+th+se)
        m=re.search(r"\[CHENG3_K\] D-solve ([0-9.]+)ms(?:\s+face-value ([0-9.]+)ms)?",x)
        if m:
            r["dsolve_s"]=float(m.group(1))/1000
            if m.group(2): r["faceval_s"]=float(m.group(2))/1000
        m=re.search(r"interface=(\d+) \(([0-9.]+)%\).*?D-solves=(\d+)\s+Newton iters/solve=([0-9.]+)",x)
        if m: r.update(interface_cells=int(m.group(1)),interface_pct=float(m.group(2)),
                       newton_solves=int(m.group(3)),newton_iters_per_solve=float(m.group(4)))
        m=re.search(r"no-gradient=(\d+) \(([0-9.]+)%\)",x)
        if m: r["rej_nogradient_pct"]=float(m.group(2))
    if all(k in r for k in ("recon_s","dsolve_s","faceval_s")):
        r["rest_s"]=max(r["recon_s"]-r["dsolve_s"]-r["faceval_s"],0.0)
        r["kernel_share_pct"]=100*(r["dsolve_s"]+r["faceval_s"])/r["recon_s"]
    return r
F=["config","mode","dim","case","kernel","threads","steps","total_wall_s","recon_s","dsolve_s",
   "faceval_s","rest_s","kernel_share_pct","o2_lsq_s","cellD_bucket_s","facebvd_s",
   "onetime_precompute_s","muscl_s","thinc_s","geom_bucket_s","face_bucket_s","bvd_sel_s",
   "cells","interface_cells","interface_pct","newton_solves","newton_iters_per_solve",
   "rej_nogradient_pct","enstrophy"]
rows=[]
for cfg,d,s2,s3,mode in SETS:
    for c in C2+C3:
        st=s2 if c in C2 else s3
        if st is None: continue
        for s,k in (("s1","tanh"),("s2","GAUSS")):
            r=parse(d,c,s)
            if not r: continue
            r.update(config=cfg,mode=mode,dim=2 if c in C2 else 3,case=c,kernel=k,steps=st)
            rows.append(r)
# GAUSS solves no Newton systems; the interface test is shared, so carry the tanh fraction across
for i,r in enumerate(rows):
    if r["kernel"]=="GAUSS" and not r.get("interface_pct"):
        for q in rows:
            if q["config"]==r["config"] and q["case"]==r["case"] and q["kernel"]=="tanh":
                r["interface_pct"]=q.get("interface_pct"); break
out=f"{ARC}/timing_measurements.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=F,extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)
print(f"{out}: {len(rows)} rows")
n=0
for cfg,d,*_ in SETS:
    for f in glob.glob(f"{d}/*.log"):
        shutil.copy(f,f"{ARC}/logs/{cfg}__{os.path.basename(f)}"); n+=1
for f in glob.glob("/tmp/mbq/stack/*.png"): shutil.copy(f,f"{ARC}/{os.path.basename(f)}")
for f in ("mkstack.py","re2d.sh","re3d.sh","final_export.py"):
    p=f"/tmp/mbq/{f}"
    if os.path.exists(p): shutil.copy(p,f"{ARC}/scripts/{f}")
if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(ARC,DST)
print(f"logs={n}; mirrored to {DST}")
