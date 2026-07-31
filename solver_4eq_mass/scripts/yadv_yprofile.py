#!/usr/bin/env python3
"""TASK B, part 3 -- reconstruct the MASS fraction Y from the dumps of cases 24/33/34 and
locate the region of genuinely SHOCK-PROCESSED material, then compare only that region
against the Y-consistent Hugoniot (closure B).

Why this is needed: cases.cpp seeds x < 0.1 with the closure-(A) post-shock state
(alpha_post := alpha_pre), so the LEFT INFLOW carries a mass fraction Y_A that is hundreds of
times the pre-shock Y. Under Y-transport that is a material contact the alpha model does not
have (alpha is identical on both sides of it), so a naive post-shock window mixes inflow
material with shocked material.

NASG with b=0 for both phases (air: b=0; denner_water: b=0), so T is EXPLICIT from (p,rho,alpha):
    rho = (1/T) * [ alpha*p/(kv_a (ga-1)) + (1-alpha)*(p+pinf_w)/(kv_w (gw-1)) ]
"""
import os, re, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
GA, KVA, PIA = 1.4, 720.25, 0.0
GW, KVW, PIW = 4.1, 474.2, 4.4e8


def T_of(p, rho, al):
    num = al * (p + PIA) / (KVA * (GA - 1.0)) + (1.0 - al) * (p + PIW) / (KVW * (GW - 1.0))
    return num / rho


def rho_air(p, T):
    return (p + PIA) / (KVA * T * (GA - 1.0))


def dump(case, yadv):
    env = dict(os.environ, DENNER_ACID="1")
    if yadv:
        env["ACID_YADV"] = "1"
    else:
        env.pop("ACID_YADV", None)
    exe = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
    out = subprocess.run([exe, case], capture_output=True, text=True, env=env, cwd=W).stdout
    return [[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]]


hug = subprocess.run(["/tmp/yadv_hugoniot"], capture_output=True, text=True).stdout
cl, cur = {}, None
for ln in hug.splitlines():
    m = re.match(r"=+ case(\d+)", ln)
    if m:
        cur = m.group(1); cl[cur] = {}; continue
    m = re.match(r"pre-shock : p=([-\d.e+]+)\s+u=([-\d.e+]+)\s+T=([-\d.e+]+)\s+"
                 r"rho=([-\d.e+]+)\s+alpha=([-\d.e+]+)\s+Y=([-\d.e+]+)", ln)
    if m and cur:
        cl[cur]["pre"] = [float(g) for g in m.groups()]
    m = re.match(r"(p_post \[Pa\]|rho_post|u_post \[m/s\]|T_post \[K\]|alpha_post|Y_post)\s+"
                 r"([-\d.e+]+)\s+([-\d.e+]+)", ln)
    if m and cur:
        cl[cur][m.group(1).split()[0]] = (float(m.group(2)), float(m.group(3)))

for case in ("24", "33", "34"):
    c = cl[case]
    Ypre = c["pre"][5]
    YA = c["Y_post"][0]      # mass fraction carried by the closure-(A) inflow
    rows = dump(case, True)
    prof = []
    for r in rows:
        x, al, p, u, rho = r[0], r[1], r[2], r[3], r[4]
        T = T_of(p, rho, al)
        Y = al * rho_air(p, T) / rho
        prof.append((x, al, p, u, rho, T, Y))

    print(f"\n================ case{case} ================")
    print(f"IC left-inflow (closure A) Y = {YA:.6g}   |   undisturbed pre-shock Y = {Ypre:.6g}"
          f"   ratio = {YA/Ypre:.4g}")
    print(f"  -> the alpha field is IDENTICAL ({c['pre'][4]:.2f}) on both sides of x=0.1 in the IC,")
    print(f"     but Y jumps by {YA/Ypre:.4g}x there: a material contact the alpha model does not have.")

    # cells whose Y is still the pre-shock value (within 5%) = SHOCK-PROCESSED material
    shocked = [q for q in prof if abs(q[6] - Ypre) <= 0.05 * Ypre and q[2] > 2.0 * c["pre"][0]]
    # ...and cells carrying the inflow composition
    inflow = [q for q in prof if abs(q[6] - YA) <= 0.05 * YA]
    print(f"  cells carrying inflow composition Y~Y_A : {len(inflow)}"
          + (f"   x in [{inflow[0][0]:.3f}, {inflow[-1][0]:.3f}]" if inflow else ""))
    print(f"  cells with pre-shock composition Y~Y_pre AND p>2*p_pre (shock-processed): "
          f"{len(shocked)}" + (f"   x in [{shocked[0][0]:.3f}, {shocked[-1][0]:.3f}]" if shocked else ""))

    if shocked:
        # take the middle half of that band to avoid both edges
        k0, k1 = len(shocked) // 4, max(len(shocked) // 4 + 1, 3 * len(shocked) // 4)
        band = shocked[k0:k1]
        avg = lambda j: sum(q[j] for q in band) / len(band)
        spread = lambda j: (max(q[j] for q in band) - min(q[j] for q in band)) / (abs(avg(j)) + 1e-300)
        print(f"  --- shock-processed band, middle half ({len(band)} cells, "
              f"x in [{band[0][0]:.3f}, {band[-1][0]:.3f}]) vs closure (B) ---")
        print("  | qty | (B) Y-held | solver Y-path | rel err | band spread |")
        print("  |---|---|---|---|---|")
        for name, j, key in (("p", 2, "p_post"), ("u", 3, "u_post"),
                             ("rho", 4, "rho_post"), ("T", 5, "T_post"),
                             ("alpha", 1, "alpha_post")):
            B = c[key][1]
            v = avg(j)
            print(f"  | {name} | {B:.6g} | {v:.6g} | {(v-B)/abs(B):+.2%} | {spread(j):.2e} |")
