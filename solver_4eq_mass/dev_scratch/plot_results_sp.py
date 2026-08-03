import csv, io, os, subprocess, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
env=dict(os.environ, DENNER_ACID="1")
DUMP='./build-cpp/cpp/denner_1d/denner1d_dump'
def dump(c):
    out=subprocess.run([DUMP,c],capture_output=True,text=True,env=env).stdout
    return list(csv.DictReader(io.StringIO(out)))
def C(r,k): return [float(a[k]) for a in r]
def X(r): return [float(a['x']) for a in r]
for c,name in [('25','case25  Mach-10 air-shock / water interface  (Denner 7.4.4)'),
               ('07','case07  air-water acoustic reflection/transmission  (Denner 7.3.2)')]:
    r=dump(c)
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    for j,(k,kr,lab) in enumerate([('p','p_ref','pressure p'),('u','u_ref','velocity u'),('rho','rho_ref','density rho')]):
        ax[j].plot(X(r),C(r,kr),'k--',lw=2,label='reference (exact)')
        ax[j].plot(X(r),C(r,k),color='tab:red',lw=1.1,label='ACID solver')
        ax[j].set_title(lab); ax[j].set_xlabel('x'); ax[j].grid(alpha=.3); ax[j].legend(fontsize=9)
    fig.suptitle(name, fontsize=13); fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig('results_cpp/figs/case%s.png'%c, dpi=120); plt.close(fig)
    print('saved case%s'%c)
