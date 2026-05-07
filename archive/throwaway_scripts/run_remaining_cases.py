"""Run remaining cases (10-12, 15-26) with per-case signal-based timeout.
Independent execution so failures don't cascade."""
import signal, sys, os, time
import traceback

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD/results')
from run_all_26_force import (
    run_10, run_11, run_12, run_15, run_16, run_17, run_18, run_19, run_20,
    run_21, run_22, run_23, run_24, run_25, run_26)

R = '/home/younglin90/work/claude_code/claudeCFD/results'


class TimeoutError_(Exception):
    pass


def _handler(signum, frame):
    raise TimeoutError_("case timeout")


def run_with_timeout(fn, timeout=30):
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        res = fn()
        signal.alarm(0)
        return res
    except TimeoutError_:
        return ('TIMEOUT', float('nan'), float('nan'), float(timeout), 0.0)
    except Exception as e:
        signal.alarm(0)
        return (f'ERROR:{type(e).__name__}', float('nan'), float('nan'), 0.0, 0.0)


cases = [
    (10, run_10, 30), (11, run_11, 30), (12, run_12, 30),
    (15, run_15, 30), (16, run_16, 30), (17, run_17, 30),
    (18, run_18, 30), (19, run_19, 45), (20, run_20, 30),
    (21, run_21, 30), (22, run_22, 30), (23, run_23, 60),
    (24, run_24, 30), (25, run_25, 30), (26, run_26, 30),
]

t_all = time.time()
results = []
for num, fn, tmo in cases:
    print(f'\n=== Case {num:02d} (timeout={tmo}s) ===', flush=True)
    t0 = time.time()
    res = run_with_timeout(fn, tmo)
    wall = time.time() - t0
    print(f'  -> {res[0]} err_p={res[1]:.3e} err_u={res[2]:.3e} wall={wall:.1f}s', flush=True)
    results.append((num, res[0], res[1], res[2], wall, res[4]))

# Write summary for remaining
with open(f'{R}/remaining_cases_summary.md', 'w') as f:
    f.write(f'# Remaining cases (10-12, 15-26)\n\nTotal wall: {time.time()-t_all:.1f}s\n\n')
    f.write('| # | Status | err_p | err_u | wall(s) | t_final |\n|---|---|---|---|---|---|\n')
    for num, status, ep, eu, wall, tf in results:
        f.write(f'| {num:02d} | {status:15s} | {ep:.3e} | {eu:.3e} | {wall:5.1f} | {tf:.3e} |\n')

print(f'\nDone. wall={time.time()-t_all:.1f}s')
