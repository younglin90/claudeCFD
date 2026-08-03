import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=list(csv.DictReader(open('/tmp/case_24_full.csv')))
x=[float(a['x']) for a in r]
def dev(k,kr): return [float(a[k])-float(a[kr]) for a in r]
fig,ax=plt.subplots(3,1,figsize=(14,9),sharex=True)
for a,(k,kr,lab) in zip(ax,[('p','p_ref','p'),('u','u_ref','u'),('rho','rho_ref','rho')]):
    d=dev(k,kr)
    a.plot(x,d,'r.-',lw=1,ms=4); a.axhline(0,color='k',lw=0.7)
    a.set_xlim(0.10,0.35); a.set_ylabel('ACID-ref (%s)'%lab); a.grid(alpha=.3)
    a.axvline(0.1,color='b',ls=':',label='init disc x=0.1'); a.legend(fontsize=8)
ax[0].set_title('case24 LEFT-region deviation zoom (x 0.1-0.35) -- oscillation or smooth bump?')
ax[-1].set_xlabel('x'); fig.tight_layout(); fig.savefig('/tmp/diag24Lz.png',dpi=130); plt.close(fig)
# also print: count sign changes of dp in [0.1,0.35] to quantify oscillation
import itertools
seg=[(float(a['x']),float(a['p'])-float(a['p_ref'])) for a in r if 0.1<=float(a['x'])<=0.35]
sc=sum(1 for (_,p1),(_,p2) in zip(seg,seg[1:]) if p1*p2<0)
print('left-region dp sign-changes:', sc, ' n=',len(seg), ' max|dp|=%.3e'%max(abs(p) for _,p in seg))
print('done')
