import csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=list(csv.DictReader(open('/tmp/case_24_full.csv')))
x=[float(a['x']) for a in r]
P=[float(a['p']) for a in r]; Pr=[float(a['p_ref']) for a in r]
U=[float(a['u']) for a in r]; Ur=[float(a['u_ref']) for a in r]
R=[float(a['rho']) for a in r]; Rr=[float(a['rho_ref']) for a in r]
fig,ax=plt.subplots(2,3,figsize=(16,8))
for j,(s,sr,lab) in enumerate([(P,Pr,'pressure'),(U,Ur,'velocity'),(R,Rr,'density')]):
    ax[0][j].plot(x,sr,'k--',lw=2,label='reference'); ax[0][j].plot(x,s,'r-',lw=1,label='ACID')
    ax[0][j].set_title('case24 %s FULL'%lab); ax[0][j].legend(fontsize=8); ax[0][j].grid(alpha=.3)
    ax[1][j].plot(x,sr,'k--',lw=2); ax[1][j].plot(x,s,'r.-',lw=1,ms=3)
    ax[1][j].set_xlim(0.0,0.25); ax[1][j].set_title('case24 %s LEFT zoom (x<0.25)'%lab); ax[1][j].grid(alpha=.3)
fig.tight_layout(); fig.savefig('/tmp/diag24L.png',dpi=120); plt.close(fig)
print('done')
