# denner_1d EOS Validation Policy

Date: 2026-05-20
Scope: `solver_denner` Denner 1D validation workflow and solver work.

## Decision

The Denner 1D validation cases must use the NASG formulation for water, not the older SG/stiffened-gas water parameter set.

This policy applies at least to the current target validation set:

- `validation/1D`: `01`, `02`, `04`, `05`, `07`, `13`, `14`, `15`, `24`, `25`

Do not switch these validation cases from `WATER_NASG` to `WATER_SG` unless the user explicitly requests that change and a new decision record is written.

## Active EOS convention

Current validation driver definitions in `results/run_denner1d_17case.py`:

```python
AIR_IDEAL = {'gamma': 1.4, 'pinf': 0.0, 'b': 0.0, 'kv': 717.5, 'eta': 0.0}
WATER_NASG = {'gamma': 1.187, 'pinf': 7.028e8, 'b': 6.61e-4, 'kv': 3610.0, 'eta': -1.177788e6}
WATER_SG = {'gamma': 4.4, 'pinf': 6.0e8, 'b': 0.0, 'kv': 474.2, 'eta': 0.0}
```

`WATER_NASG` is the intended active water EOS for the target cases. `WATER_SG` is a legacy/compatibility constant only unless explicitly selected by future work.

`AIR_IDEAL` is acceptable as the air phase because it is the ideal-gas limit / degenerate NASG case with `pinf=0`, `b=0`, and `eta=0`.

## Factory behavior

`solver/denner_1d/eos/eos_class.py:create_eos()` defaults to NASG:

```python
eos_type = ph.get('eos_type', ph.get('type', 'nasg'))
if eos_type == 'stiffened':
    return StiffenedGasEOS(...)
return NasgEOS(...)
```

Therefore a phase dictionary is interpreted as NASG unless it explicitly sets `eos_type` or `type` to `"stiffened"`.

## Reporting and code-comment rule

Avoid loose wording such as “SG” when describing the active validation EOS. Use one of the following precise terms:

- `NASG` for water,
- `ideal-gas limit of NASG` for air,
- `legacy SG support` only for compatibility code that is not active in the target validations.

If comments mention “stiffened gas” generically, they must not imply that validation water was changed from NASG to SG.

## Verification commands

Use these checks before claiming EOS-related validation behavior:

```bash
grep -n "WATER_NASG\|WATER_SG" results/run_denner1d_17case.py
grep -n "ph1=.*WATER\|ph2=.*WATER\|WATER_NASG" results/run_denner1d_17case.py
grep -n "eos_type\|type.*stiffened\|create_eos" solver/denner_1d/eos/eos_class.py
```

Expected result: target validation cases reference `WATER_NASG`; `WATER_SG` should not be used for target cases.

## Prior audit note

A 2026-05-20 NASG derivative audit found that the implemented NASG thermodynamic derivatives matched finite-difference checks for phase and mixture quantities. The remaining case-14 density plateau/shock-location difficulty should not be assumed to be caused by switching to SG; the active setup remains NASG.
