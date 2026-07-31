// test_unstructured3d.cpp — verify build_unstructured_3d geometry on a gmsh mixed mesh.
#include "cfd/mesh_unstructured3d.hpp"
#include <cstdio>
#include <cmath>
#include <map>
using namespace cfd;

int main(int argc, char** argv) {
    const char* path = argc>1 ? argv[1] : "/tmp/mbq/cube_ext.umsh";
    bool ok=false; Mesh m = load_umsh_3d(path, &ok);
    if(!ok){ std::printf("FAIL: cannot read %s\n", path); return 1; }
    const int Nc=m.n_cells(), Nf=m.n_faces();

    std::map<int,int> typ; for(auto&c:m.cell_nodes) typ[(int)c.size()]++;
    double Vtot=0, vmin=1e300, amin=1e300; for(int c=0;c<Nc;++c){Vtot+=m.cell_volumes[c]; vmin=std::min(vmin,m.cell_volumes[c]);}
    for(int f=0;f<Nf;++f) amin=std::min(amin,m.face_areas[f]);
    // unit normals?
    double nerr=0; for(int f=0;f<Nf;++f){ double*n=&m.face_normals[3*f]; nerr=std::max(nerr,std::fabs(std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])-1.0)); }
    // closed-cell: sum of outward area*normal == 0
    std::vector<double> S(3*Nc,0.0);
    for(int f=0;f<Nf;++f){ double A=m.face_areas[f],*n=&m.face_normals[3*f]; int o=m.face_owner[f],nb=m.face_neighbour[f];
        S[3*o]+=A*n[0]; S[3*o+1]+=A*n[1]; S[3*o+2]+=A*n[2];
        if(nb>=0){ S[3*nb]-=A*n[0]; S[3*nb+1]-=A*n[1]; S[3*nb+2]-=A*n[2]; } }
    double clos=0; for(int c=0;c<Nc;++c){ double s=std::sqrt(S[3*c]*S[3*c]+S[3*c+1]*S[3*c+1]+S[3*c+2]*S[3*c+2]); clos=std::max(clos,s); }
    int nbnd=0; for(int f=0;f<Nf;++f) if(m.face_neighbour[f]<0) nbnd++;
    // centroids inside [0,1]?
    double cmin=1e300,cmax=-1e300; for(int c=0;c<Nc;++c)for(int d=0;d<3;++d){cmin=std::min(cmin,m.cell_centers[3*c+d]);cmax=std::max(cmax,m.cell_centers[3*c+d]);}

    std::printf("mesh=%s  cells=%d faces=%d boundary=%d\n", path, Nc, Nf, nbnd);
    std::printf("cell types:"); const char* nm[9]={0,0,0,0,"tet","pyr","prism",0,"hex"};
    for(auto&kv:typ) std::printf(" %s=%d", nm[kv.first]?nm[kv.first]:"?", kv.second); std::printf("\n");
    std::printf("Vtot=%.10g (expect domain volume)  vmin=%.3e amin=%.3e\n", Vtot, vmin, amin);
    std::printf("max |unit-normal-1| = %.2e  (want ~0)\n", nerr);
    std::printf("max closed-cell |sum A*n| = %.2e  (want ~0, divergence theorem)\n", clos);
    std::printf("centroid range [%.4f, %.4f]\n", cmin, cmax);
    bool pass = (vmin>0)&&(amin>0)&&(nerr<1e-12)&&(clos<1e-9)&&(Vtot>0);
    std::printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
