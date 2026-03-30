"""V05, V07, V08 slow validation cases runner."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solver'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'solver', 'validate'))

# Import the validation module
import importlib.util
spec = importlib.util.spec_from_file_location(
    'run_all_1d_validations',
    os.path.join(os.path.dirname(__file__), 'solver', 'validate', 'run_all_1d_validations.py')
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print('\n' + '='*65)
print('  Running slow cases: V05, V07, V08')
print('='*65)

mod.v05_eoc_sinusoidal()
mod.v07_g1g2g3_pep(fast=False)
mod.v08_s1s2(fast=False)

print('\n' + '='*65)
print('  SLOW CASE SUMMARY')
print('='*65)
for name in ['V05_eoc_sinusoidal', 'V07_g1g2g3_pep', 'V08_s1s2']:
    if name in mod.results:
        ok, info = mod.results[name]
        mark = 'PASS' if ok else 'FAIL'
        print(f'  {mark}  {name:<35s} {info}')
print('='*65)
