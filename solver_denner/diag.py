import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
def load(p):
    r=list(csv.DictReader(open(p)))
    return ([float(x['x']) for x in r],
            {k:[float(x[k]) for x in r] for k in ('p','u','rho','p_ref','u_ref','rho_ref')})

# case07 pressure (full) -- show the HF wiggle
x,d=load('/tmp/case_07.csv')
fig,ax=plt.subplots(1,2,figsize=(13,4))
ax[0].plot(x,d['p_ref'],'k--',lw=2,label='reference'); ax[0].plot(x,d['p'],'r-',lw=1,label='ACID')
ax[0].set_title('case07 pressure (full)'); ax[0].legend(); ax[0].grid(alpha=.3)
# zoom on the transmitted region (water side, x>0.5) where wiggle likely
ax[1].plot(x,d['p_ref'],'k--',lw=2); ax[1].plot(x,d['p'],'r.-',lw=1,ms=3)
ax[1].set_xlim(0.55,0.95); ax[1].set_title('case07 pressure zoom (water side)'); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig('/tmp/diag07.png',dpi=120); plt.close(fig)

# case24 p,u near shock -- compare cfl 0.30 vs 0.60
x3,d3=load('/tmp/case_24_cfl030.csv'); x6,d6=load('/tmp/case_24_cfl060.csv')
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
for j,(k,kr,lab) in enumerate([('p','p_ref','pressure'),('u','u_ref','velocity')]):
    ax[j].plot(x6,d6[kr],'k--',lw=2,label='reference')
    ax[j].plot(x3,d3[k],'-',color='tab:blue',lw=1.1,ms=3,label='cfl=0.30')
    ax[j].plot(x6,d6[k],'.-',color='tab:red',lw=1.0,ms=3,label='cfl=0.60')
    ax[j].set_xlim(0.6,0.95); ax[j].set_title('case24 %s zoom near shock'%lab); ax[j].legend(fontsize=8); ax[j].grid(alpha=.3)
fig.tight_layout(); fig.savefig('/tmp/diag24.png',dpi=120); plt.close(fig)
print('DIAG_PLOTS_DONE')
