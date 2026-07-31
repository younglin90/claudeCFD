#!/usr/bin/env python3
import subprocess, os, io, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT='/home/younglin90/work/claude_code/claudeCFD/solver_denner'
DUMP=ROOT+'/build-cpp/cpp/denner_1d/denner1d_dump'
os.chdir(ROOT)

def dump(case):
    env=dict(os.environ); env['DENNER_ACID']='1'
    p=subprocess.run([DUMP,case],capture_output=True,text=True,env=env)
    rows=list(csv.DictReader(io.StringIO(p.stdout)))
    return {k:[float(r[k]) for r in rows] for k in rows[0].keys()}

FLOOR=1e-18
def errfig(case, title, out):
    d=dump(case)
    x=d['x']
    ep=[max(abs(d['p'][i]-d['p_ref'][i]),FLOOR) for i in range(len(x))]
    eu=[max(abs(d['u'][i]-d['u_ref'][i]),FLOOR) for i in range(len(x))]
    er=[max(abs(d['rho'][i]-d['rho_ref'][i]),FLOOR) for i in range(len(x))]
    fig,ax=plt.subplots(1,3,figsize=(13.5,2.4))
    for a,(e,lab,col) in zip(ax,[(ep,'|p - p_ref|  [Pa]','tab:red'),
                                 (eu,'|u - u_ref|  [m/s]','tab:green'),
                                 (er,'|rho - rho_ref|  [kg/m3]','tab:blue')]):
        a.semilogy(x,e,color=col,lw=0.9)
        a.set_xlabel('x'); a.set_title(lab,fontsize=9)
        a.set_ylim(1e-17,max(1e-12,max(e)*3)); a.grid(True,alpha=0.3)
    fig.suptitle(title,fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(out,dpi=130); plt.close(fig)
    print('Plot saved:',out, '| max |dp|=%.2e |du|=%.2e |drho|=%.2e'%(max(ep),max(eu),max(er)))

errfig('01','case01 static interface -- pointwise error (machine precision)', ROOT+'/results_cpp/figs/case01_err.png')
errfig('02','case02 contact advection -- pointwise error', ROOT+'/results_cpp/figs/case02_err.png')
