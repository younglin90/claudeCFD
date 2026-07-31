#!/usr/bin/env python3
import subprocess, os, math, io, csv, json

ROOT='/home/younglin90/work/claude_code/claudeCFD/solver_denner'
DUMP=ROOT+'/build-cpp/cpp/denner_1d/denner1d_dump'
os.chdir(ROOT)

def run_dump(case, extra_env):
    env=dict(os.environ); env['DENNER_ACID']='1'
    for k,v in extra_env.items(): env[k]=str(v)
    p=subprocess.run([DUMP, case], capture_output=True, text=True, env=env)
    rows=list(csv.DictReader(io.StringIO(p.stdout)))
    cols={k:[float(r[k]) for r in rows] for k in rows[0].keys()}
    return cols

def corr(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    sab=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    saa=sum((x-ma)**2 for x in a); sbb=sum((x-mb)**2 for x in b)
    if saa<=0 or sbb<=0: return 1.0 if (saa<1e-30 and sbb<1e-30) else 0.0
    return sab/math.sqrt(saa*sbb)

def amp_ratio(a,b):
    pa=max(a)-min(a); pb=max(b)-min(b)
    return pa/pb if pb>0 else float('nan')

def l2n(a,b):
    n=len(a); rng=max(b)-min(b) or 1.0
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(n))/n)/rng

def front_width(x, rho, rho_ref):
    # cells where solver rho is strictly between the two plateau values (transition band)
    lo=min(rho); hi=max(rho); span=hi-lo
    if span<=0: return 0
    lo5=lo+0.05*span; hi5=lo+0.95*span
    return sum(1 for v in rho if lo5<v<hi5)

def wake_p2p(x, p, p_ref, xa, xb):
    d=[p[i]-p_ref[i] for i in range(len(x)) if xa<=x[i]<=xb]
    return (max(d)-min(d)) if d else float('nan')

def tv_excess_pct(x, rho, rho_ref, xa, xb):
    idx=[i for i in range(len(x)) if xa<=x[i]<=xb]
    def tv(f): return sum(abs(f[idx[k+1]]-f[idx[k]]) for k in range(len(idx)-1))
    seg=[rho_ref[i] for i in idx]
    jump=max(seg)-min(seg) or 1.0
    return max(0.0, tv(rho)-tv(rho_ref))/jump*100.0

def iface_ip(x, p, p_ref, alpha):
    # interface region = where alpha crosses 0.5; linf of |p-p_ref| normalized by p_ref range
    # find alpha midpoint index
    ic=min(range(len(alpha)), key=lambda i: abs(alpha[i]-0.5))
    w=[i for i in range(len(x)) if abs(i-ic)<=8]
    rng=(max(p_ref)-min(p_ref)) or 1.0
    return max(abs(p[i]-p_ref[i]) for i in w)/rng

CONF={'prod':{}, 'NO_THINC':{'ACID_NO_THINC':1}, 'NO_TRBDF2':{'ACID_NO_TRBDF2':1}, 'NO_AJAC':{'ACID_NO_AJAC':1}}

print('=========== ABLATION (cases 02,07,14,25) ===========')
prod_ref={}
for cid, keys in [('02',['corr_rho','front']), ('07',['amp_p','amp_u','corr_p','wake']),
                  ('14',['corr_u','tv_exc']), ('25',['amp_p','corr_p','ip'])]:
    print('--- case %s ---'%cid)
    for cname, env in CONF.items():
        d=run_dump(cid, env)
        x,al,p,u,rho=d['x'],d['alpha'],d['p'],d['u'],d['rho']
        pr,ur,rr=d['p_ref'],d['u_ref'],d['rho_ref']
        m={}
        m['corr_p']=corr(p,pr); m['corr_u']=corr(u,ur); m['corr_rho']=corr(rho,rr)
        m['amp_p']=amp_ratio(p,pr); m['amp_u']=amp_ratio(u,ur)
        if cid=='02': m['front']=front_width(x,rho,rr)
        if cid=='07': m['wake']=wake_p2p(x,p,pr,0.55,0.95)
        if cid=='14': m['tv_exc']=tv_excess_pct(x,rho,rr,0.5,0.95)
        if cid=='25': m['ip']=iface_ip(x,p,pr,al)
        if cname=='prod': prod_ref[cid]=(p,u,rho)
        # for NO_AJAC report identical-ness
        idn=''
        if cname=='NO_AJAC':
            pp,uu,rrr=prod_ref[cid]
            dmax=max(abs(p[i]-pp[i]) for i in range(len(p)))
            idn=' maxdiff_vs_prod_p=%.2e'%dmax
        out=' '.join('%s=%s'%(k, ('%.5g'%m[k] if isinstance(m[k],float) else m[k])) for k in keys if k in m)
        print('  %-10s %s%s'%(cname, out, idn))

print()
print('=========== CFL ROBUSTNESS: case07 ===========')
for cflv in [0.2,0.45,0.6]:
    d=run_dump('07', {'ACID_STUDY_CFL':cflv})
    x,p,u=d['x'],d['p'],d['u']; pr,ur=d['p_ref'],d['u_ref']
    print('  cfl=%.2f amp_p=%.4f amp_u=%.4f corr_p=%.5f corr_u=%.5f wake_p2p=%.3f'%(
        cflv, amp_ratio(p,pr), amp_ratio(u,ur), corr(p,pr), corr(u,ur), wake_p2p(x,p,pr,0.55,0.95)))

print()
print('=========== GRID REFINEMENT (production scheme) ===========')
def refine(cid, scales, metric):
    print('--- case %s ---'%cid)
    prev=None
    for sc in scales:
        d=run_dump(cid, {'ACID_STUDY_NSCALE':sc})
        x,p,u,rho=d['x'],d['p'],d['u'],d['rho']; pr,ur,rr=d['p_ref'],d['u_ref'],d['rho_ref']
        N=len(x)
        vals={'l2_p':l2n(p,pr),'l2_u':l2n(u,ur),'amp_p':amp_ratio(p,pr),
              'corr_rho':corr(rho,rr),'corr_u':corr(u,ur)}
        v=vals[metric]
        order=''
        if prev is not None and metric.startswith('l2'):
            if prev>0 and v>0: order=' order=%.2f'%(math.log(prev/v)/math.log(2))
        prev=v
        print('  N=%-5d %s=%.5g%s   (amp_p=%.4f corr_u=%.5f corr_rho=%.5f)'%(
            N, metric, v, order, vals['amp_p'], vals['corr_u'], vals['corr_rho']))
refine('04',[1,2,4],'l2_p')
refine('05',[1,2],'l2_p')
refine('02',[1,2,4],'corr_rho')
refine('14',[1,2,4],'l2_u')
