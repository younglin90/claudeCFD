// Parametric electrospray nozzle/collector geometry.
// Units are meters; OpenSCAD will display them as model units.
inner_diameter = 0.00016;
outer_diameter = 0.00026;
nozzle_length = 0.0003;
collector_distance = 0.0015;
collector_diameter = 0.005;
farfield_radius = 0.0025;
$fn = 128;

module nozzle_electrode() {
  translate([0, -nozzle_length, 0]) rotate([-90,0,0])
  difference() {
    cylinder(h=nozzle_length, d=outer_diameter);
    translate([0,0,-1e-9]) cylinder(h=nozzle_length + 2e-9, d=inner_diameter);
  }
}

module collector_ground() {
  translate([0, collector_distance, 0]) rotate([-90,0,0])
    cylinder(h=outer_diameter*0.08, d=collector_diameter, center=true);
}

module open_atmosphere_hint() {
  %translate([0, (-nozzle_length + collector_distance)/2, 0]) rotate([-90,0,0])
    cylinder(h=nozzle_length + collector_distance, r=farfield_radius);
}

nozzle_electrode();
collector_ground();
open_atmosphere_hint();
