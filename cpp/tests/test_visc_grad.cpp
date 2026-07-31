// test_visc_grad.cpp — is the viscous cell-gradient (ReconCtx3D LSQ) accurate on an
// unstructured mesh? Linear field u=2x+3y-z must give grad=(2,3,-1) EXACTLY everywhere.
#include "cfd/mesh_unstructured3d.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/viscous3d.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace cfd;
int main(int argc,char**argv){
    Mesh m = load_umsh_3d(argc>1?argv[1]:"/tmp/mbq/vtube_tet.umsh");
    const int N=m.n_cells(); Euler3D eq; eq.gamma=1.4;
    ReconCtx3D c = build_recon_ctx_3d(m);
    std::vector<double> Wc((size_t)5*N);
    for(int i=0;i<N;++i){ double x=m.cell_centers[3*i],y=m.cell_centers[3*i+1],z=m.cell_centers[3*i+2];
        Wc[0*N+i]=1.0; Wc[1*N+i]=2*x+3*y-z; Wc[2*N+i]=0; Wc[3*N+i]=0; Wc[4*N+i]=1.0; }
    std::vector<double> gu,gv,gw,gT;
    viscous3d_cell_gradients(m,eq,c,Wc,1.0,gu,gv,gw,gT);
    double emax=0,emax_int=0,emax_bnd=0; int nbad=0;
    // boundary cell = has a boundary face
    std::vector<char> isb(N,0);
    for(int f=0;f<m.n_faces();++f) if(m.face_neighbour[f]<0) isb[m.face_owner[f]]=1;
    for(int i=0;i<N;++i){ double ex=std::fabs(gu[3*i]-2)+std::fabs(gu[3*i+1]-3)+std::fabs(gu[3*i+2]+1);
        emax=std::max(emax,ex); if(isb[i])emax_bnd=std::max(emax_bnd,ex); else emax_int=std::max(emax_int,ex);
        if(ex>1e-6)nbad++; }
    std::printf("mesh N=%d  grad err(linear field): max=%.3e  interior=%.3e  boundary=%.3e  #bad(>1e-6)=%d\n",
                N,emax,emax_int,emax_bnd,nbad);
    std::printf("%s\n", (emax<1e-6)?"PASS (gradient exact on linear)":"FAIL (gradient WRONG -> viscous flux garbage)");
    return 0;
}
