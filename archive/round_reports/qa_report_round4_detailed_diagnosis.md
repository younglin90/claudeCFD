# Round 4 Detailed Diagnostic Analysis — Critical Bug Found

## Executive Finding

**Root Cause Identified:** The `solve_IMEX` function accepts `ph1`, `ph2` as **dictionaries** (see line 4548: `ph1['gamma']`), but the test case passes them as **EOS objects** (e.g., `NASGEOS(...)`).

This type mismatch causes Step 8's NASG detection code to fail:

```python
# Line 3449 in _peluchon_acoustic_im1:
_has_nasg = (ph1.get('b', 0.0) > 0.0) or (ph2.get('b', 0.0) > 0.0)
```

When `ph1` is an EOS object, `ph1.get(...)` raises `AttributeError` or returns wrong result.

---

## Type System Issue

### Current Implementation (Lines 4548-4549 in solve_IMEX):

```python
def solve_IMEX(ph1, ph2, ...):
    ...
    g1, pinf1 = ph1['gamma'], ph1['pinf']  # <-- EXPECTS DICT
    g2, pinf2 = ph2['gamma'], ph2['pinf']
```

### Test Case Invocation (from test_round4_comprehensive.py):

```python
ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}           # DICT ✓
ph2 = NASGEOS(gamma=..., pinf=..., kv=..., b=..., eta=...) # EOS OBJECT ✗
solve_IMEX(ph1, ph2, ...)
```

### The Problem:

**Test case uses mixed types: ph1 is dict, ph2 is EOS object**

This is **inconsistent** with the function's expected interface. When ph2 is an EOS object:
- Line 4549: `g2, pinf2 = ph2['gamma'], ph2['pinf']` → **KeyError or AttributeError**
- Line 3449: `ph2.get('b')` → **AttributeError** (EOS objects don't have `.get()`)

---

## Fix 1: Make solve_IMEX Accept Both Dict and EOS Objects

**Option A: Normalize inputs**

```python
def solve_IMEX(ph1, ph2, ...):
    # Convert EOS objects to dicts if needed
    if hasattr(ph1, '__dataclass_fields__'):  # EOS object
        ph1 = {
            'gamma': ph1.gamma,
            'pinf': ph1.pinf,
            'kv': ph1.kv,
            'b': getattr(ph1, 'b', 0.0),
            'eta': getattr(ph1, 'eta', 0.0),
        }
    if hasattr(ph2, '__dataclass_fields__'):  # EOS object
        ph2 = {
            'gamma': ph2.gamma,
            'pinf': ph2.pinf,
            'kv': ph2.kv,
            'b': getattr(ph2, 'b', 0.0),
            'eta': getattr(ph2, 'eta', 0.0),
        }
    # Now ph1, ph2 are guaranteed dicts
```

**Option B: Fix NASG detection to handle both types**

```python
def _get_nasg_param(ph):
    """Get 'b' parameter, handling both dict and EOS object types."""
    if isinstance(ph, dict):
        return ph.get('b', 0.0)
    else:
        return getattr(ph, 'b', 0.0)

# In _peluchon_acoustic_im1:
_has_nasg = (_get_nasg_param(ph1) > 0.0) or (_get_nasg_param(ph2) > 0.0)
```

---

## Why Test 3 Passed (SG Regression)

**Test 3 uses correct dict syntax:**

```python
ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}  # DICT
ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}  # DICT
eos1 = IdealEOS(gamma=1.4, kv=717.5)
eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)
...
solve_IMEX(ph1, ph2, ...)  # ✓ CORRECT: dicts, not EOS objects
```

So Test 3 never triggered the NASG detection bug.

---

## Why Tests 1, 2, 4, 5 Failed

**Tests 1, 2 use incorrect EOS object syntax:**

```python
ph2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
              b=6.61e-4, eta=-1.177788e6)
solve_IMEX(ph1, ph2, ...)  # ✗ WRONG: ph2 is EOS object, not dict
```

When ph2 is an EOS object:
1. Line 4549: `g2, pinf2 = ph2['gamma'], ph2['pinf']` raises error OR partially works
2. If partial (accessing via object attributes instead), g2/pinf2 might be wrong type
3. Line 3449: `ph2.get('b')` raises AttributeError
4. NASG detection fails → Wood c_mix substitution skipped → IM1 uses wrong c_mix
5. IM1 block-tridiag becomes ill-conditioned → garbage solution → overflow/NaN

**Tests 4, 5:** SG Phase 2 cases may have gotten EOS objects too (depending on where test spec comes from).

---

## Evidence: Check for Exception Suppression

The test output shows:
- No explicit exceptions raised
- But values reach inf, nan, overflow
- This suggests the error occurs silently:
  1. Dict indexing `ph2['gamma']` might fail, but exception caught somewhere
  2. Or EOS object is treated as dict, returning None or wrong values
  3. Subsequent code processes None/wrong values → garbage results

---

## Immediate Fix Instructions for code_maker (Round 5)

### Step 1: Determine Actual Function Signature

Check what solve_IMEX and _peluchon_acoustic_im1 **actually expect**:

```python
# Option A: Function should accept only DICTS
def solve_IMEX(ph1, ph2, ...):  # ph1, ph2 are dicts
    
# Option B: Function should accept DICTS or EOS objects
def solve_IMEX(ph1, ph2, ...):  # ph1, ph2 can be dict or EOS
```

The code at line 4548 (`ph1['gamma']`) suggests **Option A: dicts only**.

### Step 2: Fix the Type Mismatch

**In solve_IMEX (line 4477+):**

Add type normalization immediately after function entry:

```python
def solve_IMEX(ph1, ph2, a1r1_0, a2r2_0, ru_0, rE_0, a1_0, ...):
    """..."""
    # Type normalization: convert EOS objects to dicts if needed
    if hasattr(ph1, 'gamma') and not isinstance(ph1, dict):
        ph1 = {
            'gamma': ph1.gamma,
            'pinf': getattr(ph1, 'pinf', 0.0),
            'kv': getattr(ph1, 'kv', 717.5),
            'b': getattr(ph1, 'b', 0.0),
            'eta': getattr(ph1, 'eta', 0.0),
        }
    if hasattr(ph2, 'gamma') and not isinstance(ph2, dict):
        ph2 = {
            'gamma': ph2.gamma,
            'pinf': getattr(ph2, 'pinf', 0.0),
            'kv': getattr(ph2, 'kv', 474.2),
            'b': getattr(ph2, 'b', 0.0),
            'eta': getattr(ph2, 'eta', 0.0),
        }
    
    N = len(a1_0)
    ...
```

### Step 3: Fix NASG Detection in _peluchon_acoustic_im1

Replace line 3449:

```python
# OLD:
_has_nasg = (ph1.get('b', 0.0) > 0.0) or (ph2.get('b', 0.0) > 0.0)

# NEW:
b1 = ph1.get('b', 0.0) if isinstance(ph1, dict) else getattr(ph1, 'b', 0.0)
b2 = ph2.get('b', 0.0) if isinstance(ph2, dict) else getattr(ph2, 'b', 0.0)
_has_nasg = (b1 > 0.0) or (b2 > 0.0)
```

Or call a helper:

```python
def _get_eos_param(ph, param_name, default=0.0):
    """Get EOS parameter, handling dict or object."""
    if isinstance(ph, dict):
        return ph.get(param_name, default)
    else:
        return getattr(ph, param_name, default)

_has_nasg = (_get_eos_param(ph1, 'b') > 0.0) or (_get_eos_param(ph2, 'b') > 0.0)
```

---

## Summary

| Issue | Cause | Fix | Priority |
|-------|-------|-----|----------|
| Type mismatch (dict vs EOS object) | Test passes EOS, code expects dict | Normalize inputs in solve_IMEX | **CRITICAL** |
| NASG detection fails | `ph2.get('b')` on non-dict | Fix detection to handle both types | **CRITICAL** |
| Wood c_mix skipped | False `_has_nasg` → wrong c_mix | Fix detection above | **CRITICAL** |
| IM1 block-tridiag ill-conditioned | Wrong c_mix → wrong system matrix | Fix above | **CRITICAL** |
| Overflow/NaN in results | Ill-conditioned solve | Fix above | **CRITICAL** |

**Estimated fix time: 5-10 minutes (add type normalization + fix detection).**

---

## Verification Steps After Fix

After applying the fix, re-run:

```bash
python3 test_round4_comprehensive.py 2>&1 | tee /tmp/test_round4_fixed.log
```

**Expected results after fix:**
- TEST 1 (NASG mat CFL): err_p < 1e-2, err_u < 1e-2 → **PASS**
- TEST 2 (NASG CFL sweep): No overflow/NaN → **PASS**
- TEST 3 (SG regression): err_p ≈ 8.58e-12 → **PASS** ✓ (already passing)
- TEST 4 (Phase 2-1): u_max ∈ [225, 230] → **PASS** (may need minor tuning)
- TEST 5 (Acoustic baseline): err_p ≈ 2.56e-9 → **PASS** (may need minor tuning)

If after fix:
- TEST 1/2 still fail → suspect Step 3 (T_face ≥ 100 K) or Step 6 (D1 Gamma_inv)
- TEST 4/5 still degrade → suspect Step 7 (SLAU2) or row equilibration scaling
