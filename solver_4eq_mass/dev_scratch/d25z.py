import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=list(csv.DictReader(open('/tmp/case_25_full.csv')))
x=[float(a['x']) for a in r]
def col(k): return [float(a[k]) for a in r]
fig,ax=plt.subplots(2,1,figsize=(15,8))
for a,(k,kr,lab) in zip(ax,[('p','p_ref','pressure'),('u','u_ref','velocity')]):
    a.plot(x,col(kr),'k--',lw=2,label='reference'); a.plot(x,col(k),'r.-',lw=1.1,ms=4,label='ACID')
    a.set_xlim(0.18,0.52); a.set_ylabel(lab); a.grid(alpha=.3); a.legend(fontsize=9)
    a.axvline(0.25,color='b',ls=':',label='init shock 0.25')
ax[0].set_title('case25 AIR-region zoom (x 0.18-0.52): the left oscillation')
ax[-1].set_xlabel('x'); fig.tight_layout(); fig.savefig('/tmp/d25z.png',dpi=130); plt.close(fig)
print('done')
