#!/usr/bin/env python3
import subprocess, json, os, sys

os.chdir('/home/younglin90/work/claude_code/claudeCFD/solver_denner')
env = dict(os.environ)
env['DENNER_ACID'] = '1'
p = subprocess.run(['./build-cpp/cpp/denner_1d/denner1d_validate'],
                   capture_output=True, text=True, env=env)
rows = []
summary = None
for line in p.stdout.splitlines():
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
        try:
            rows.append(json.loads(line))
        except Exception as e:
            sys.stderr.write('parse fail: %s | %s\n' % (e, line[:80]))
    elif line.startswith('DENNER1D_CPP_METRIC'):
        summary = line

# order per paper figure/table narrative
order = ['01','02','04','05','07','13','14','15','24','25','26','27','28','30','31','33','34','35','36']
by_id = {r['case']: r for r in rows}

print('=== SUMMARY:', summary)
print('parsed cases:', len(rows))
print()
hdr = ['case','N','pass','l2_p','corr_p','corr_u','corr_rho','amp_p','amp_u','hf_p','linf_p','linf_u','linf_rho']
print('\t'.join(hdr))
for cid in order:
    r = by_id.get(cid)
    if not r:
        print(cid, 'MISSING'); continue
    def g(k):
        v = r.get(k)
        return ('%.5g' % v) if isinstance(v,(int,float)) else str(v)
    print('\t'.join([r['case'], str(r['N']), str(r['pass']),
                     g('l2_p'), g('corr_p'), g('corr_u'), g('corr_rho'),
                     g('amp_ratio_p'), g('amp_ratio_u'), g('hf_p'),
                     g('linf_p'), g('linf_u'), g('linf_rho')]))

# dump full json for the ones the paper narrates in detail
print()
print('=== FULL DETAIL (selected) ===')
for cid in ['01','02','14','25','26','30','31','35','07','13','24']:
    r = by_id.get(cid)
    if r: print(cid, json.dumps({k:r[k] for k in r if k not in ('case','N')}, default=str))

# write json for reuse
with open('/tmp/denner_metrics.json','w') as f:
    json.dump(by_id, f, indent=1)
print()
print('stderr tail:', p.stderr.strip()[-200:] if p.stderr else '(none)')
