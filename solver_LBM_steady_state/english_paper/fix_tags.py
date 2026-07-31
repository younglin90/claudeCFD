import re, pathlib
p = pathlib.Path("paper_en.md")
t = p.read_text(encoding="utf-8")
# pandoc's OMML writer drops \tag{}; render the number inline at equation end instead.
t2 = re.sub(r"\\tag\{(\d+)\}", r"\\qquad(\1)", t)
pathlib.Path("paper_en_fixed.md").write_text(t2, encoding="utf-8")
print("replaced:", t.count("\\tag{"), "-> remaining:", t2.count("\\tag{"), "qquad:", t2.count("\\qquad("))
