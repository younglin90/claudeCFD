// test_recon3d_unstr.cpp — sanity for the unstructured GAUSS-THINC reconstruction.
#include "cfd/mesh_unstructured3d.hpp"
#include "cfd/reconstruct3d_o2_unstr.hpp"
#include "cfd/reconstruct3d_unstr.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace cfd;

int main(int argc,char**argv){
    Mesh m=load_umsh_3d(argc>1?argv[1]:"/tmp/mbq/cube_ext.umsh");
    const int N=m.n_cells(), Nf=m.n_faces();
    ReconCtx3DO2 o2=build_recon_ctx_3d_o2_unstr(m);
    std::vector<double> WL,WR;
    auto cx=[&](int c){return m.cell_centers[3*c];}; auto cy=[&](int c){return m.cell_centers[3*c+1];}; auto cz=[&](int c){return m.cell_centers[3*c+2];};

    // (A) constant field -> face values exactly the constant (no interface, no overshoot)
    std::vector<double> W(5*N);
    for(int c=0;c<N;++c){ W[0*N+c]=2.0; W[1*N+c]=0;W[2*N+c]=0;W[3*N+c]=0;W[4*N+c]=3.0; }
    reconstruct3d_bvd_gauss_unstr(m,o2,W,5,WL,WR);
    double cerr=0; bool finA=true;
    for(int f=0;f<Nf;++f){ for(int v=0;v<5;++v){ double a=WL[(size_t)v*Nf+f],b=WR[(size_t)v*Nf+f];
        if(!std::isfinite(a)||!std::isfinite(b))finA=false; }
        cerr=std::max(cerr,std::fabs(WL[0*Nf+f]-2.0)); cerr=std::max(cerr,std::fabs(WL[4*Nf+f]-3.0)); }

    // (B) linear field rho=1+0.3(x+y+z) -> face values BOUNDED in stencil; o2 path near-exact
    for(int c=0;c<N;++c){ W[0*N+c]=1.0+0.3*(cx(c)+cy(c)+cz(c)); W[4*N+c]=2.0+0.5*cx(c); }
    reconstruct3d_bvd_gauss_unstr(m,o2,W,5,WL,WR);
    double rmin=1e300,rmax=-1e300,lerr=0; bool finB=true;
    for(int f=0;f<Nf;++f){ double a=WL[0*Nf+f]; if(!std::isfinite(a))finB=false;
        rmin=std::min(rmin,a); rmax=std::max(rmax,a);
        double fx=m.face_centers[3*f],fy=m.face_centers[3*f+1],fz=m.face_centers[3*f+2];
        lerr=std::max(lerr,std::fabs(a-(1.0+0.3*(fx+fy+fz)))); }

    std::printf("mesh N=%d Nf=%d  o2 max_nb=%d\n", N,Nf,o2.max_nb);
    std::printf("(A) constant: finite=%d  max|face-const|=%.2e  (want ~0)\n",(int)finA,cerr);
    std::printf("(B) linear:   finite=%d  rho face range=[%.4f,%.4f] (field [1,1.9])  max|face-exact|=%.3e\n",(int)finB,rmin,rmax,lerr);
    bool pass = finA&&finB&&(cerr<1e-9)&&(rmin>0.9)&&(rmax<2.0);
    std::printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
