// Double-Mach reflection RAMP-frame domain (Dumbser 2007 JCP226 sec.7.5 / Cheng 2021 sec.4.2.4)
// Pentagon flow region above a 30-deg ramp (apex at x=0.2), top wall y=2, right outflow x=3.
// h = 1/200 uniform unstructured triangular mesh.
h = 0.005;                        // 1/200
tan30 = 0.5773502691896258;
xa = 0.2;                          // ramp apex x
xr = 3.0;                          // right boundary
yt = 2.0;                          // top wall
yramp = (xr - xa) * tan30;         // ramp height at x=xr = 2.8*tan30 = 1.61658...
Point(1) = {0.0, 0.0,   0, h};     // bottom-left
Point(2) = {xa,  0.0,   0, h};     // ramp apex
Point(3) = {xr,  yramp, 0, h};     // ramp meets right
Point(4) = {xr,  yt,    0, h};     // top-right
Point(5) = {0.0, yt,    0, h};     // top-left
Line(1) = {1, 2};                  // flat bottom  (x in [0,0.2], y=0)  -> symmetry/slip
Line(2) = {2, 3};                  // 30-deg ramp                        -> reflective slip
Line(3) = {3, 4};                  // right x=3                          -> outflow
Line(4) = {4, 5};                  // top y=2                            -> reflective wall
Line(5) = {5, 1};                  // left x=0                           -> post-shock inflow
Line Loop(1) = {1, 2, 3, 4, 5};
Plane Surface(1) = {1};
// force uniform size
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;
Mesh.Algorithm = 6;                // Frontal-Delaunay
