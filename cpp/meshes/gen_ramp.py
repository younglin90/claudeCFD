import gmsh, sys
h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
_out = sys.argv[2] if len(sys.argv) > 2 else "/home/younglin90/work/claude_code/claudeCFD/cpp/meshes/dmr_ramp.mesh2d"
tan30 = 0.5773502691896258
xa, xr, yt = 0.2, 3.0, 2.0
yramp = (xr - xa) * tan30
gmsh.initialize()
gmsh.model.add("dmr_ramp")
g = gmsh.model.geo
p1 = g.addPoint(0.0, 0.0,   0, h)
p2 = g.addPoint(xa,  0.0,   0, h)
p3 = g.addPoint(xr,  yramp, 0, h)
p4 = g.addPoint(xr,  yt,    0, h)
p5 = g.addPoint(0.0, yt,    0, h)
l1 = g.addLine(p1, p2)
l2 = g.addLine(p2, p3)
l3 = g.addLine(p3, p4)
l4 = g.addLine(p4, p5)
l5 = g.addLine(p5, p1)
cl = g.addCurveLoop([l1, l2, l3, l4, l5])
s = g.addPlaneSurface([cl])
g.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
ntags, ncoords, _ = gmsh.model.mesh.getNodes()
tag2idx = {}
xs, ys = [], []
for i, tg in enumerate(ntags):
    tag2idx[int(tg)] = len(xs)
    xs.append(ncoords[3*i]); ys.append(ncoords[3*i+1])
etypes, etags, enodess = gmsh.model.mesh.getElements(2)
tris = []
for et, enodes in zip(etypes, enodess):
    if et == 2:
        for k in range(0, len(enodes), 3):
            tris.append((tag2idx[int(enodes[k])], tag2idx[int(enodes[k+1])], tag2idx[int(enodes[k+2])]))
out = _out
with open(out, "w") as f:
    f.write("%d %d\n" % (len(xs), len(tris)))
    for x, y in zip(xs, ys):
        f.write("%.12g %.12g\n" % (x, y))
    for a, b, c in tris:
        f.write("%d %d %d\n" % (a, b, c))
gmsh.finalize()
print("nodes=%d tris=%d  domain x[%.3f,%.3f] y[%.3f,%.3f]  yramp@xr=%.5f" % (
    len(xs), len(tris), min(xs), max(xs), min(ys), max(ys), yramp))
