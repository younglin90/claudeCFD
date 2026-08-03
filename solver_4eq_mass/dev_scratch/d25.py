import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=list(csv.DictReader(open('/tmp/case_25_full.csv')))
x=[float(a['x']) for a in r]
def col(k): return [float(a[k]) for a in r]
P,Pr=col('p'),col('p_ref'); U,Ur=col('u'),col('u_ref'); R,Rr=col('rho'),col('rho_ref')
fig,ax=plt.subplots(3,2,figsize=(16,11))
for j,(s,sr,lab) in enumerate([(P,Pr,'p'),(U,Ur,'u'),(R,Rr,'rho')]):
    ax[j][0].plot(x,sr,'k--',lw=2,label='ref'); ax[j][0].plot(x,s,'r-',lw=1,label='ACID')
    ax[j][0].axvline(0.25,color='b',ls=':'); ax[j][0].axvline(0.5,color='g',ls=':')
    ax[j][0].set_ylabel(lab); ax[j][0].grid(alpha=.3); ax[j][0].legend(fontsize=7)
    ax[j][0].set_title('%s FULL (blue=init shock .25, green=interface .5)'%lab)
    # left (air) zoom x in [0.2,0.55]
    ax[j][1].plot(x,sr,'k--',lw=2); ax[j][1].plot(x,s,'r.-',lw=1,ms=3)
    ax[j][1].set_xlim(0.2,0.55); ax[j][1].axvline(0.5,color='g',ls=':')
    ax[j][1].set_title('%s LEFT/interface zoom (x 0.2-0.55)'%lab); ax[j][1].grid(alpha=.3)
fig.tight_layout(); fig.savefig('/tmp/d25.png',dpi=120); plt.close(fig)
# sign-change counts of (ACID-ref) per region
def sc(lo,hi,s,sr):
    seg=[s[i]-sr[i] for i in range(len(x)) if lo<=x[i]<=hi]
    c=sum(1 for a,b in zip(seg,seg[1:]) if a*b<0)
    return c,(max(abs(v) for v in seg) if seg else 0)
for nm,s,sr in [('p',P,Pr),('u',U,Ur),('rho',R,Rr)]:
    cl,ml=sc(0.0,0.49,s,sr); cr,mr=sc(0.51,1.0,s,sr)
    print('%s: LEFT(air x<0.49) signchg=%d max=%.3e | RIGHT(water x>0.51) signchg=%d max=%.3e'%(nm,cl,ml,cr,mr))
print('done')
