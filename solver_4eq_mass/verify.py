import subprocess, os, re, sys
env = dict(os.environ, DENNER_ACID="1")
V = "./build-cpp/cpp/denner_1d/denner1d_validate"
def run(args, extra=None):
    e = dict(env, **(extra or {}))
    return subprocess.run([V]+args, capture_output=True, text=True, env=e).stdout
allcases = "01,02,04,05,07,13,14,15,24,25"
out = run(["--only", allcases])
m = re.search(r"pass_count=\d+ total=\d+", out)
print("ALL:", m.group(0) if m else "NOTFOUND(len=%d)"%len(out))
for c in sys.argv[1:] or ["07","24"]:
    o = run(["--only", c])
    d = {}
    for k in ("hf_p","hf_u","corr_p","l2_p","amp_ratio_p","pass"):
        mm = re.search(r'"%s":[^,}]*'%k, o)
        if mm: d[k]=mm.group(0)
    print("case%s:"%c, " ".join(d.values()))
