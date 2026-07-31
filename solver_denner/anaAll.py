import csv, io, os, re, subprocess
env = dict(os.environ, DENNER_ACID="1")
V='./build-cpp/cpp/denner_1d/denner1d_validate'
D='./build-cpp/cpp/denner_1d/denner1d_dump'
cases=['01','02','04','05','07','13','14','15','24','25']
def metrics(c):
    out=subprocess.run([V,'--only',c],capture_output=True,text=True,env=env,timeout=300).stdout
    return dict(re.findall(r'"([a-z_0-9]+)":([^,}\s]+)',out))
def dump(c):
    try:
        out=subprocess.run([D,c],capture_output=True,text=True,env=env,timeout=300).stdout
        return list(csv.DictReader(io.StringIO(out)))
    except Exception:
        return []
def wig(vals):
    rng=max(vals)-min(vals)
    if rng<=0: return 0
    eps=1e-4*rng
    dp=[vals[i+1]-vals[i] for i in range(len(vals)-1)]
    s=[(1 if d>eps else (-1 if d<-eps else 0)) for d in dp]
    s=[x for x in s if x!=0]
    return sum(1 for a,b in zip(s,s[1:]) if a*b<0)
for c in cases:
    m=metrics(c)
    rows=dump(c)
    keep=('pass','linf_p','corr_p','corr_u','corr_rho','l2_p','l2_u','amp_ratio_p','hf_p','hf_u')
    ms=' '.join('%s=%s'%(k,m[k]) for k in keep if k in m)
    print('case%s | %s'%(c,ms))
    if not rows:
        print('   (no dump)'); continue
    x=[float(a['x']) for a in rows]
    parts=[]
    for k,kr in [('p','p_ref'),('u','u_ref'),('rho','rho_ref')]:
        try:
            v=[float(a[k]) for a in rows]; vr=[float(a[kr]) for a in rows]
        except Exception:
            continue
        rng=max(max(vr)-min(vr),1e-30)
        dev=[abs(a-b) for a,b in zip(v,vr)]
        mx=max(dev); i=dev.index(mx)
        parts.append('%s: maxdev=%5.2f%% @x=%.3f  wig=%d(ref %d)'%(k,100*mx/rng,x[i],wig(v),wig(vr)))
    print('   '+'\n   '.join(parts))
print('ANA_DONE')
