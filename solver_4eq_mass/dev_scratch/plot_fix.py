import csv, io, os, subprocess, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
env=dict(os.environ, DENNER_ACID="1")
DUMP='./build-cpp/cpp/denner_1d/denner1d_dump'
def dump(case, extra=None):
    e=dict(env, **(extra or {}))
    out=subprocess.run([DUMP,case],capture_output=True,text=True,env=e).stdout
    return list(csv.DictReader(io.StringIO(out)))
def C(r,k): return [float(a[k]) for a in r]
def X(r): return [float(a['x']) for a in r]

# case25: BEFORE = dhk off (cfl0.638+apadv only) ; AFTER = + dhat_scale 8
B25=dump('25',{'ACID_DHK':'1'}); A25=dump('25')
B07=dump('07',{'ACID_DHK':'1'}); A07=dump('07')
print('rows', len(B25), len(A25), len(B07), len(A07))

fig=plt.figure(figsize=(16,10))

ax=fig.add_subplot(2,2,1)
ax.plot(X(A25),C(A25,'p_ref'),'k--',lw=2,label='reference (exact)')
ax.plot(X(B25),C(B25,'p'),color='tab:orange',lw=1.3,marker='.',ms=4,label='before MWI-boost  amp=1.16 hf=0.63')
ax.plot(X(A25),C(A25,'p'),color='tab:red',lw=1.3,marker='.',ms=4,label='after  amp=1.09 hf=0.46')
ax.set_xlim(0.20,0.30); ax.set_ylim(0.6e8,1.15e8)
ax.set_title('case25 reflected-shock PRESSURE (zoom): overshoot down'); ax.set_ylabel('p [Pa]'); ax.grid(alpha=.3); ax.legend(fontsize=8)

ax=fig.add_subplot(2,2,2)
ax.plot(X(A25),C(A25,'u_ref'),'k--',lw=2,label='reference')
ax.plot(X(B25),C(B25,'u'),color='tab:orange',lw=1.3,marker='.',ms=4,label='before')
ax.plot(X(A25),C(A25,'u'),color='tab:red',lw=1.3,marker='.',ms=4,label='after')
ax.set_xlim(0.20,0.30); ax.set_ylim(-500,1000)
ax.set_title('case25 reflected-shock VELOCITY (zoom): ringing down'); ax.set_ylabel('u [m/s]'); ax.grid(alpha=.3); ax.legend(fontsize=8)

ax=fig.add_subplot(2,2,3)
ax.plot(X(A07),C(A07,'p_ref'),'k--',lw=2,label='reference')
ax.plot(X(B07),C(B07,'p'),color='tab:orange',lw=1.1,label='before  hf_p=0.36')
ax.plot(X(A07),C(A07,'p'),color='tab:red',lw=1.1,label='after  hf_p=0.28')
ax.set_xlim(0.35,1.05)
ax.set_title('case07 PRESSURE air-wake (x 0.35-1.05): ripple reduced'); ax.set_xlabel('x'); ax.set_ylabel('p [Pa]'); ax.grid(alpha=.3); ax.legend(fontsize=8)

ax=fig.add_subplot(2,2,4)
ax.plot(X(A07),C(A07,'p_ref'),'k--',lw=1.8,label='reference')
ax.plot(X(A07),C(A07,'p'),color='tab:red',lw=1.0,label='after (current)')
ax.set_title('case07 FULL pressure (air/water acoustic refl/trans)'); ax.set_xlabel('x'); ax.set_ylabel('p [Pa]'); ax.grid(alpha=.3); ax.legend(fontsize=8)

fig.suptitle('MWI dissipation boost (per-case dhat_scale): case25 overshoot 1.16->1.09 + ringing cut | case07 air-wake ripple 0.36->0.28 | 10/10 kept', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig('/tmp/fix_status.png',dpi=130); print('saved')
