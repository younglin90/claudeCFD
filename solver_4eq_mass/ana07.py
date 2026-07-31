import csv, io, os, subprocess
env=dict(os.environ, DENNER_ACID="1")
out=subprocess.run(['./build-cpp/cpp/denner_1d/denner1d_dump','07'],capture_output=True,text=True,env=env).stdout
r=list(csv.DictReader(io.StringIO(out)))
x=[float(a['x']) for a in r]; p=[float(a['p']) for a in r]
# WIGGLE = local extrema: count slope-sign reversals of p (not vs reference)
def wig(lo,hi,nm):
    idx=[i for i in range(len(x)) if lo<=x[i]<=hi]
    pp=[p[i] for i in idx]
    dp=[pp[i+1]-pp[i] for i in range(len(pp)-1)]
    # count reversals of slope ignoring ~zero steps
    eps=1e-3
    sdp=[(1 if d>eps else (-1 if d<-eps else 0)) for d in dp]
    s=[v for v in sdp if v!=0]
    rev=sum(1 for a,b in zip(s,s[1:]) if a*b<0)
    ptp=max(pp)-min(pp)
    print('%-20s n=%3d  p2p=%6.3f Pa  slope-reversals(extrema)=%d'%(nm,len(idx),ptp,rev))
    return idx
print('flat wake region — is there a wiggle (slope reversals)?')
wig(0.45,0.95,'wake 0.45-0.95')
# print actual p values across a slice of the wake (every 5th cell)
idx=[i for i in range(len(x)) if 0.50<=x[i]<=0.65]
print('\nwake p samples (x, p-1e5 in Pa):')
for i in idx[::2]:
    print('  x=%.4f  p-1e5=%+.4f'%(x[i],p[i]-1e5))
