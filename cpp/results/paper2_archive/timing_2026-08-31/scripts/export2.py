# Extend the measurement CSV with the single-thread forced runs that carry BOTH kernel timers,
# then mirror the whole archive to the D: drive. Every row keeps the configuration that produced
# it -- the same case measured gated/forced or at 1/24 threads gives ratios from 0.95x to 2.4x,
# so a number without its conditions is not interpretable.
import re, os, csv, shutil, glob
ARC = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_archive/timing_2026-08-31"
DST = "/mnt/d/datas/paper2_timing_2026-09-01"
os.makedirs(f"{ARC}/logs", exist_ok=True)

SETS = [("gated_1T","/tmp/mbq/tblB",500,200,"gated",1),
        ("forced_MT","/tmp/mbq/chartF",500,500,"forced",None),
        ("forced_1T","/tmp/mbq/chart2D1T",500,None,"forced",1),
        ("stack_1T","/tmp/mbq/stack",500,20,"forced",1)]   # both kernel timers, wall time
CASES2 = ["leveque","config3","shockvortex","shockmixing","mach3","doublemach"]
CASES3 = ["deform3d","twocyl3d","sphere3d"]

def parse(d,c,s):
    f=f"{d}/{c}_{s}.log"
    if not os.path.exists(f): return None
    x=open(f,encoding="utf-8",errors="replace").read(); r={}
    m=re.search(r"threads=(\d+)",x); r["threads"]=int(m.group(1)) if m else 1
    w=[float(v) for l in x.splitlines() if "harness_wall" not in l
       for v in re.findall(r"wall=([0-9.]+)s",l)]
    r["total_wall_s"]=w[-1] if w else ""
    if c in CASES3:
        m=re.search(r"RECON breakdown \(summed (\d+)ms\): o2-LSQ (\d+)ms.*?cell-D-Newton (\d+)ms"
                    r".*?face-quad\+BVD (\d+)ms",x,re.S)
        if m:
            _,lsq,cd,fb=(float(v)/1000 for v in m.groups())
            r.update(o2_lsq_s=lsq,cellD_bucket_s=cd,facebvd_s=fb,recon_s=lsq+cd+fb)
        m=re.search(r"\(2a\)cand (\d+)ms.*?\(2b\)minTBV (\d+)ms.*?\(2c\)assemble (\d+)ms",x)
        if m: r.update(f2a_s=float(m.group(1))/1000,f2b_s=float(m.group(2))/1000,f2c_s=float(m.group(3))/1000)
        m=re.search(r"prologue \(scratch re-init \+ cached builds\): (\d+)ms",x)
        if m: r["onetime_precompute_s"]=float(m.group(1))/1000
        m=re.search(r"kernels \(RDTSC\): D-solve ([0-9.]+)ms\s+face-value ([0-9.]+)ms",x)
        if m: r.update(dsolve_s=float(m.group(1))/1000,faceval_s=float(m.group(2))/1000)
        for k,pat in (("newton_solves",r"over (\d+) solves"),("cells",r"cells=(\d+)")):
            mm=re.search(pat,x)
            if mm: r[k]=int(mm.group(1))
        mm=re.search(r"avg ([0-9.]+) Newton iters/solve",x)
        if mm: r["newton_iters_per_solve"]=float(mm.group(1))
        mm=re.search(r"enstrophy=([0-9.]+)",x)
        if mm: r["enstrophy"]=float(mm.group(1))
    else:
        m=re.search(r"MUSCL=([0-9.]+)s\s+THINC=([0-9.]+)s \(geom=([0-9.]+)s face=([0-9.]+)s\)"
                    r"\s+BVD_sel=([0-9.]+)s",x)
        if m:
            mu,th,ge,fa,se=map(float,m.groups())
            r.update(muscl_s=mu,thinc_s=th,geom_bucket_s=ge,face_bucket_s=fa,bvd_sel_s=se,recon_s=mu+th+se)
        m=re.search(r"\[CHENG3_K\] D-solve ([0-9.]+)ms(?:\s+face-value ([0-9.]+)ms)?",x)
        if m:
            r["dsolve_s"]=float(m.group(1))/1000
            if m.group(2): r["faceval_s"]=float(m.group(2))/1000
        m=re.search(r"cell-vars=(\d+)\s+interface=(\d+) \(([0-9.]+)%\)\s+D-solves=(\d+)"
                    r"\s+Newton iters/solve=([0-9.]+)",x)
        if m: r.update(cell_vars=int(m.group(1)),interface_cells=int(m.group(2)),
                       interface_pct=float(m.group(3)),newton_solves=int(m.group(4)),
                       newton_iters_per_solve=float(m.group(5)))
        m=re.search(r"flat\(no jump\)=(\d+) \(([0-9.]+)%\)\s+extremum\(D infinite\)=(\d+) \(([0-9.]+)%\)"
                    r"\s+no-gradient=(\d+) \(([0-9.]+)%\)",x)
        if m: r.update(rej_flat_pct=float(m.group(2)),rej_extremum_pct=float(m.group(4)),
                       rej_nogradient_pct=float(m.group(6)))
    if "recon_s" in r and "dsolve_s" in r and "faceval_s" in r:
        r["rest_s"]=max(r["recon_s"]-r["dsolve_s"]-r["faceval_s"],0.0)
    return r

FIELDS=["config","mode","dim","case","kernel","threads","steps","total_wall_s","recon_s",
        "dsolve_s","faceval_s","rest_s","o2_lsq_s","cellD_bucket_s","facebvd_s","f2a_s","f2b_s",
        "f2c_s","onetime_precompute_s","muscl_s","thinc_s","geom_bucket_s","face_bucket_s",
        "bvd_sel_s","cells","cell_vars","interface_cells","interface_pct","newton_solves",
        "newton_iters_per_solve","rej_flat_pct","rej_extremum_pct","rej_nogradient_pct","enstrophy"]
rows=[]
for cfg,d,st2,st3,mode,_ in SETS:
    for c in CASES2+CASES3:
        st=st2 if c in CASES2 else st3
        if st is None: continue
        for s,kern in (("s1","tanh"),("s2","GAUSS")):
            r=parse(d,c,s)
            if not r: continue
            r.update(config=cfg,mode=mode,dim=2 if c in CASES2 else 3,case=c,kernel=kern,steps=st)
            rows.append(r)
out=f"{ARC}/timing_measurements.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)
print(f"{out}: {len(rows)} rows")
n=0
for cfg,d,*_ in SETS:
    for f in glob.glob(f"{d}/*.log"):
        shutil.copy(f,f"{ARC}/logs/{cfg}__{os.path.basename(f)}"); n+=1
for f in glob.glob("/tmp/mbq/stack/*.png")+glob.glob("/tmp/mbq/chartF/*.png"):
    shutil.copy(f,f"{ARC}/{os.path.basename(f)}")
for f in ("mkstack.py","mkchart2.py","cmp3.py","stack2d.sh","stack3d.sh","chart2.sh","export2.py"):
    p=f"/tmp/mbq/{f}"
    if os.path.exists(p): shutil.copy(p,f"{ARC}/scripts/{f}")
print(f"logs={n}, figures + scripts copied")
if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(ARC,DST)
print("mirrored to", DST)
