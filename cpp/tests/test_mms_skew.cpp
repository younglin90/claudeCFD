// test_mms_skew.cpp — Method-of-Manufactured-Solutions accuracy on SKEWED / NON-ORTHOGONAL
// unstructured meshes. Loads a .umsh, measures (1) mesh-quality metrics (non-orthogonality
// angle, skewness), (2) cell P2-LSQ gradient error, (3) convective o2-quad face-value error,
// (4) VISCOUS face-gradient error for the CURRENT scheme (centroid-grad over-relaxed) vs the
// ENHANCED scheme (P2-reconstruction gradient evaluated at the TRUE face centroid x_f, i.e.
// skewness+non-orthogonality corrected via the o2 Hessian). Manufactured field is smooth/
// non-polynomial so the errors reflect real truncation, and distortion degradation is visible.
//
// Usage: ./test_mms_skew <mesh.umsh>
#include "cfd/mesh.hpp"
#include "cfd/mesh_unstructured3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include "cfd/reconstruct3d_o2_unstr.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace cfd;

// manufactured scalar + analytic gradient
static inline double phi_exact(double x,double y,double z){
    return std::sin(1.7*x+0.3)*std::sin(2.1*y+0.5)*std::sin(1.3*z+0.7);
}
static inline void grad_exact(double x,double y,double z,double g[3]){
    double sx=std::sin(1.7*x+0.3), cx=std::cos(1.7*x+0.3);
    double sy=std::sin(2.1*y+0.5), cy=std::cos(2.1*y+0.5);
    double sz=std::sin(1.3*z+0.7), cz=std::cos(1.3*z+0.7);
    g[0]=1.7*cx*sy*sz; g[1]=2.1*sx*cy*sz; g[2]=1.3*sx*sy*cz;
}

static double pct(std::vector<double>& v,double q){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t i=(size_t)(q*(v.size()-1)); return v[i];
}

int main(int argc,char**argv){
    const char* mesh = (argc>1)?argv[1]:"/tmp/mbq/cube_tet.umsh";
    std::vector<std::array<double,3>> X; std::vector<std::vector<int>> C;
    if(!read_umsh(mesh,X,C)){ std::printf("FAIL read %s\n",mesh); return 1; }
    Mesh m = build_unstructured_3d(X,C);
    ReconCtx3DO2 o2 = build_recon_ctx_3d_o2_unstr(m);
    const int N=m.n_cells(), Nf=m.n_faces();

    // cell field = phi(centroid)
    std::vector<double> W((size_t)N);
    for(int i=0;i<N;++i) W[i]=phi_exact(m.cell_centers[3*i],m.cell_centers[3*i+1],m.cell_centers[3*i+2]);
    std::vector<double> g; reconstruct3d_o2_coeffs(m,o2,W,1,0,g);   // g[9*i]: P2 coeffs

    // ---- (1) mesh-quality metrics on interior faces ----
    std::vector<double> nonOrtho, skew;
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f], n=m.face_neighbour[f]; if(n<0)continue;
        double dx=m.cell_centers[3*n]-m.cell_centers[3*o];
        double dy=m.cell_centers[3*n+1]-m.cell_centers[3*o+1];
        double dz=m.cell_centers[3*n+2]-m.cell_centers[3*o+2];
        double dl=std::sqrt(dx*dx+dy*dy+dz*dz);
        double nx=m.face_normals[3*f],ny=m.face_normals[3*f+1],nz=m.face_normals[3*f+2];
        double cth=std::fabs((dx*nx+dy*ny+dz*nz)/std::max(dl,1e-30));
        cth=std::min(1.0,cth); nonOrtho.push_back(std::acos(cth)*180.0/M_PI);
        // skewness vector m = (x_f - x_o) - [S.(x_f-x_o)/(S.d)] d ; S = area*n
        double cpx=m.face_centers[3*f]-m.cell_centers[3*o];
        double cpy=m.face_centers[3*f+1]-m.cell_centers[3*o+1];
        double cpz=m.face_centers[3*f+2]-m.cell_centers[3*o+2];
        double A=m.face_areas[f]; double Sx=A*nx,Sy=A*ny,Sz=A*nz;
        double SdotCp=Sx*cpx+Sy*cpy+Sz*cpz, Sdotd=Sx*dx+Sy*dy+Sz*dz;
        double t=(std::fabs(Sdotd)>1e-30)?SdotCp/Sdotd:0.0;
        double mx=cpx-t*dx,my=cpy-t*dy,mz=cpz-t*dz;
        skew.push_back(std::sqrt(mx*mx+my*my+mz*mz)/std::max(dl,1e-30));
    }

    // ---- helpers: P2 reconstruction gradient at an offset (dx,dy,dz) from cell center ----
    auto gradAt=[&](const double* gi,double dx,double dy,double dz,double out[3]){
        out[0]=gi[0]+gi[3]*dx+gi[6]*dy+gi[7]*dz;
        out[1]=gi[1]+gi[6]*dx+gi[4]*dy+gi[8]*dz;
        out[2]=gi[2]+gi[7]*dx+gi[8]*dy+gi[5]*dz;
    };
    auto valAt=[&](double phi0,const double* gi,double dx,double dy,double dz){
        return phi0+gi[0]*dx+gi[1]*dy+gi[2]*dz
              +0.5*gi[3]*dx*dx+0.5*gi[4]*dy*dy+0.5*gi[5]*dz*dz
              +gi[6]*dx*dy+gi[7]*dx*dz+gi[8]*dy*dz;
    };

    // ---- BJ limiter phi per cell (as the solver computes it: min over faces of allowed/dq) ----
    // + Venkatakrishnan smooth threshold eps^2=(K h)^3 variant (K=5) to gauge smooth-extremum clipping.
    std::vector<double> phiBJ(N,1.0), phiVK(N,1.0), phiU2(N,1.0);
    for(int i=0;i<N;++i){ double qbar=W[i]; double qmn=qbar,qmx=qbar;
        for(int k=0;k<o2.max_nb;++k){int nb=o2.nb[(size_t)i*o2.max_nb+k]; if(nb<0)continue;
            double wn=W[nb]; qmn=std::min(qmn,wn); qmx=std::max(qmx,wn);}
        const double* gi=&g[9*i]; double cx=m.cell_centers[3*i],cy=m.cell_centers[3*i+1],cz=m.cell_centers[3*i+2];
        double h=std::cbrt(m.cell_volumes[i]); double eps2=std::pow(5.0*h,3.0);
        double pbj=1.0,pvk=1.0;
        for(int fc:m.cell_faces[i]){ double dx=m.face_centers[3*fc]-cx,dy=m.face_centers[3*fc+1]-cy,dz=m.face_centers[3*fc+2]-cz;
            double dq=gi[0]*dx+gi[1]*dy+gi[2]*dz+0.5*gi[3]*dx*dx+0.5*gi[4]*dy*dy+0.5*gi[5]*dz*dz+gi[6]*dx*dy+gi[7]*dx*dz+gi[8]*dy*dz;
            double allow=dq>0?(qmx-qbar):(qmn-qbar);
            double r=(std::fabs(dq)>1e-30)?allow/dq:1.0; if(r<0)r=0; if(r<pbj)pbj=r;
            double D1=dq,D2=allow; double num=(D2*D2+2.0*D1*D2+eps2), den=(D2*D2+2.0*D1*D1+D1*D2+eps2);
            double pv=(std::fabs(den)>1e-30)?num/den:1.0; if(pv<pvk)pvk=pv;
        }
        phiBJ[i]=pbj<0?0:(pbj>1?1:pbj); phiVK[i]=pvk<0?0:(pvk>1?1:pvk);
        // U2 smooth-extremum spare: per-axis curvature (Hessian diag) sign-coherent over the stencil
        // => no Gibbs => smooth => DO NOT limit (phi=1). Else fall back to BJ. (MOOD-u2 a-priori, scale-free.)
        double rng=qmx-qmn, du2=h*h*h;
        bool smooth=(rng<du2);
        if(!smooth){ smooth=true;
            for(int ax=0;ax<3;++ax){ int c=3+ax; double Hlo=gi[c],Hhi=gi[c];
                for(int k=0;k<o2.max_nb;++k){int nb=o2.nb[(size_t)i*o2.max_nb+k]; if(nb<0)continue;
                    double Hn=g[9*nb+c]; if(Hn<Hlo)Hlo=Hn; if(Hn>Hhi)Hhi=Hn;}
                if(Hhi*Hlo< -du2){ smooth=false; break; }            // curvature sign flip on this axis
            }
        }
        phiU2[i]= smooth?1.0:phiBJ[i];
    }

    // ---- (2) cell gradient error (centroid) ----
    double e_grad=0,nrm_grad=0;
    for(int i=0;i<N;++i){ double ge[3]; grad_exact(m.cell_centers[3*i],m.cell_centers[3*i+1],m.cell_centers[3*i+2],ge);
        const double* gi=&g[9*i];
        for(int k=0;k<3;++k){ double er=gi[k]-ge[k]; e_grad+=er*er; nrm_grad+=ge[k]*ge[k]; } }

    // ---- (3) convective o2-quad face value error + (4) viscous face-grad CURRENT vs ENHANCED ----
    double e_conv=0,e_convBJ=0,e_convVK=0,nrm_conv=0;
    double e_vcur=0,e_venh=0,e_vavg=0,nrm_v=0;
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f], n=m.face_neighbour[f]; if(n<0)continue;
        double fx=m.face_centers[3*f],fy=m.face_centers[3*f+1],fz=m.face_centers[3*f+2];
        const double* go=&g[9*o]; const double* gn=&g[9*n];
        double phio=W[o], phin=W[n];
        // (3) owner-side o2-quad face value vs exact phi(x_f)
        double dxo=fx-m.cell_centers[3*o],dyo=fy-m.cell_centers[3*o+1],dzo=fz-m.cell_centers[3*o+2];
        double qf=valAt(phio,go,dxo,dyo,dzo); double pe=phi_exact(fx,fy,fz);
        e_conv+=(qf-pe)*(qf-pe); nrm_conv+=pe*pe;
        // limited face values (solver applies phi to the whole P2 increment): qbar + phi*(qf-qbar)
        double qbj=phio+phiBJ[o]*(qf-phio), qvk=phio+phiU2[o]*(qf-phio);
        e_convBJ+=(qbj-pe)*(qbj-pe); e_convVK+=(qvk-pe)*(qvk-pe);
        // exact face gradient
        double gfe[3]; grad_exact(fx,fy,fz,gfe);
        // centre-to-centre
        double dcx=m.cell_centers[3*n]-m.cell_centers[3*o];
        double dcy=m.cell_centers[3*n+1]-m.cell_centers[3*o+1];
        double dcz=m.cell_centers[3*n+2]-m.cell_centers[3*o+2];
        double Ld=std::sqrt(dcx*dcx+dcy*dcy+dcz*dcz); double ex=dcx/Ld,ey=dcy/Ld,ez=dcz/Ld;
        double dphi=phin-phio;
        // CURRENT: centroid gradients, over-relaxed along d
        double gco[3]={go[0],go[1],go[2]}, gcn[3]={gn[0],gn[1],gn[2]};
        double ax=0.5*(gco[0]+gcn[0]),ay=0.5*(gco[1]+gcn[1]),az=0.5*(gco[2]+gcn[2]);
        double corr=dphi/Ld-(ax*ex+ay*ey+az*ez);
        double gcur[3]={ax+corr*ex,ay+corr*ey,az+corr*ez};
        // ENHANCED: P2 gradients evaluated at the TRUE face centroid x_f (skew+non-ortho corrected)
        double gof[3],gnf[3];
        gradAt(go,dxo,dyo,dzo,gof);
        gradAt(gn,fx-m.cell_centers[3*n],fy-m.cell_centers[3*n+1],fz-m.cell_centers[3*n+2],gnf);
        double bx=0.5*(gof[0]+gnf[0]),by=0.5*(gof[1]+gnf[1]),bz=0.5*(gof[2]+gnf[2]);
        double corr2=dphi/Ld-(bx*ex+by*ey+bz*ez);
        double genh[3]={bx+corr2*ex,by+corr2*ey,bz+corr2*ez};
        // PURE-AVG: P2 gradients at x_f, simple average, NO compact correction (fully skew/non-ortho corrected)
        double gavg[3]={bx,by,bz};
        for(int k=0;k<3;++k){ e_vcur+=(gcur[k]-gfe[k])*(gcur[k]-gfe[k]);
            e_venh+=(genh[k]-gfe[k])*(genh[k]-gfe[k]);
            e_vavg+=(gavg[k]-gfe[k])*(gavg[k]-gfe[k]); nrm_v+=gfe[k]*gfe[k]; }
    }

    std::printf("MESH %s : N=%d Nf=%d\n",mesh,N,Nf);
    std::printf("  non-ortho(deg) mean=%.2f p95=%.2f max=%.2f | skewness mean=%.3f p95=%.3f max=%.3f\n",
        [&]{double s=0;for(double v:nonOrtho)s+=v;return s/std::max((size_t)1,nonOrtho.size());}(),
        pct(nonOrtho,0.95),pct(nonOrtho,1.0),
        [&]{double s=0;for(double v:skew)s+=v;return s/std::max((size_t)1,skew.size());}(),
        pct(skew,0.95),pct(skew,1.0));
    std::printf("  L2 cell-grad      = %.4e\n", std::sqrt(e_grad/std::max(1e-300,nrm_grad)));
    std::printf("  L2 conv o2-face   = %.4e  (BJ-limited=%.4e  U2spare-limited=%.4e)\n",
        std::sqrt(e_conv/std::max(1e-300,nrm_conv)), std::sqrt(e_convBJ/std::max(1e-300,nrm_conv)),
        std::sqrt(e_convVK/std::max(1e-300,nrm_conv)));
    std::printf("  L2 visc-facegrad  CURRENT=%.4e  ENH+orr=%.4e  PUREAVG(P2@face)=%.4e  cur/avg=%.2fx\n",
        std::sqrt(e_vcur/std::max(1e-300,nrm_v)), std::sqrt(e_venh/std::max(1e-300,nrm_v)),
        std::sqrt(e_vavg/std::max(1e-300,nrm_v)), std::sqrt(e_vcur/std::max(1e-300,e_vavg)));
    return 0;
}
