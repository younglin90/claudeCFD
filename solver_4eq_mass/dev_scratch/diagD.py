import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=list(csv.DictReader(open('/tmp/case_24_full.csv')))
x=[float(a['x']) for a in r]
def dev(k,kr): return [float(a[k])-float(a[kr]) for a in r]
dp=dev('p','p_ref'); du=dev('u','u_ref'); dr=dev('rho','rho_ref')
fig,ax=plt.subplots(3,1,figsize=(15,10),sharex=True)
for a,(d,lab) in zip(ax,[(dp,'p'),(du,'u'),(dr,'rho')]):
    a.plot(x,d,'r-',lw=1,ms=2,marker='.'); a.axhline(0,color='k',lw=0.7)
    a.set_ylabel('ACID - ref  (%s)'%lab); a.grid(alpha=.3)
    a.axvline(0.1,color='b',ls=':',lw=1,label='initial disc. x=0.1')
    a.axvline(0.8,color='g',ls=':',lw=1,label='shock x=0.8'); a.legend(fontsize=8,loc='upper left')
ax[-1].set_xlabel('x'); ax[0].set_title('case24 deviation (ACID - reference) -- localizes the oscillation')
fig.tight_layout(); fig.savefig('/tmp/diag24D.png',dpi=120); plt.close(fig)
print('done')
