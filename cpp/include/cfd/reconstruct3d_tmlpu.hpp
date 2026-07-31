// cfd/reconstruct3d_tmlpu.hpp — 3D port of the 2D "CZL" reconstruction (reconstruct2d.hpp
// reconstruct_tmlpu_gated, the ungated TMLPU_TBV_ZEROBV path). P1-ONLY, component-wise,
// reconstruction-only. Deterministic. Behind env TMLPU_GATED.
//
// CZL recipe (mirror of the 2D ZEROBV branch, ZBVGATE NOT set):
//   1. per-cell P1 LSQ gradient over FACE-adjacent neighbours, inverse-distance weighted
//      (w_k = 1/|d_k|^2), grad = A^{-1} b, A = Σ w_k d_k d_k^T (3x3), b = Σ w_k d_k Δφ_k.
//   2. per-vertex min/max (vmn/vmx) of cell-averages over cells sharing each node
//      (reuses u3_build_nodeadj / U3NodeAdj CSR from reconstruct3d_unstr.hpp).
//   3. psi_lmp vertex-LMP cap (room-to-vertex-bound / vertex projection, capped at psi_cap).
//   4. per interior face: ASVL zero-BV pick on the [vanLeer, downwind]-LMP band; where the
//      two admissible side-bands overlap take the MIN-compressive shared value (BV=0,
//      single-valued face); no overlap -> discrete min-TBV BVD fallback. Euler positivity
//      floor on rho/p. Boundary face -> owner-side LMP-limited (psi<=1) P1 (old-BC recon).
//
// Env variants (mirror 2D): CZL = TMLPU_GATED=1 TMLPU_TBV_ZEROBV=1 TMLPU_ZBV_PICK=1
//   TMLPU_ZBV_DOWNWIND=1 ; VDB ablation = ... TMLPU_ZBV_BVDONLY=1 (skip psi* pick, always
//   discrete BVD fallback). Default (only TMLPU_GATED set) = CZL (pick + downwind).
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/reconstruct3d_unstr.hpp"   // U3NodeAdj, u3_build_nodeadj
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace cfd {

// van-Leer TVD limiter, EXACTLY as the 2D psi_van_leer (r>0 ? 2r/(1+r) : 0).
inline double u3_tmlpu_vl(double r) { if (r <= 0.0) return 0.0; return 2.0*r/(1.0+r); }

// Self-contained 3D CZL reconstruction. W is nvar*N (var-major, W[v*N+ci]); outputs
// W_L,W_R are nvar*Nf. psi_cap = 2.0. Builds LSQ gradient + node min/max internally.
inline void reconstruct_tmlpu_gated_3d(const Mesh& m, const std::vector<double>& W, int nvar,
                                       std::vector<double>& W_L, std::vector<double>& W_R,
                                       double psi_cap = 2.0) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double* cc = m.cell_centers.data();
    const double eps = 1e-30;
    // env variant flags (read once, deterministic). Default = CZL (pick + downwind).
    static const bool ZEROBV  = std::getenv("TMLPU_TBV_ZEROBV") != nullptr;
    static const bool BVDONLY = std::getenv("TMLPU_ZBV_BVDONLY") != nullptr;
    static const bool ZBVDW   = std::getenv("TMLPU_ZBV_DOWNWIND") != nullptr;
    static const int  ZBVPICK = []{ const char* e=std::getenv("TMLPU_ZBV_PICK"); return (e&&e[0])?std::atoi(e):0; }();
    // if only TMLPU_GATED is set (no ZEROBV knob), still run the CZL zero-BV core with
    // downwind upper band + MIN psi* pick (matches the "default downwind+pick" spec).
    const bool use_zerobv = ZEROBV || true;   // TMLPU_GATED already gated at the dispatch site
    const bool dw_band = ZBVDW || (!std::getenv("TMLPU_ZBV_SUPERBEE") && !std::getenv("TMLPU_ZBV_MINMOD"));
    const int  pick    = (std::getenv("TMLPU_ZBV_PICK")!=nullptr) ? ZBVPICK : 1;   // default MIN psi* (=1)
    const double posfloor = 0.1;

    // ---- (1) per-cell P1 LSQ gradient over FACE-adjacent neighbours (3x3 solve/cell) ----
    std::vector<double> grad((size_t)nvar*N*3, 0.0);
    #pragma omp parallel for schedule(dynamic,64)
    for (int ci = 0; ci < N; ++ci) {
        double cx=cc[3*ci], cy=cc[3*ci+1], cz=cc[3*ci+2];
        // symmetric 3x3 A = Σ w d dᵀ  (geometry only -> shared across vars)
        double A00=0,A01=0,A02=0,A11=0,A12=0,A22=0;
        // gather face neighbours once
        // (we accumulate b per var below, so recompute d per neighbour there)
        for (int fc : m.cell_faces[ci]) {
            int o=m.face_owner[fc], nn=m.face_neighbour[fc];
            int nb = (o==ci)?nn:o; if(nb<0) continue;
            double dx=cc[3*nb]-cx, dy=cc[3*nb+1]-cy, dz=cc[3*nb+2]-cz;
            double d2=dx*dx+dy*dy+dz*dz; if(d2<eps) continue; double wk=1.0/d2;
            A00+=wk*dx*dx; A01+=wk*dx*dy; A02+=wk*dx*dz;
            A11+=wk*dy*dy; A12+=wk*dy*dz; A22+=wk*dz*dz;
        }
        // inverse of symmetric 3x3 (cofactor/determinant). Fall back to zero gradient if singular.
        double c00=A11*A22-A12*A12, c01=A12*A02-A01*A22, c02=A01*A12-A11*A02;
        double det=A00*c00+A01*c01+A02*c02;
        double i00=0,i01=0,i02=0,i11=0,i12=0,i22=0;
        bool ok = std::fabs(det) > 1e-300;
        if (ok) {
            double idet=1.0/det;
            double c11=A00*A22-A02*A02, c12=A02*A01-A00*A12, c22=A00*A11-A01*A01;
            i00=c00*idet; i01=c01*idet; i02=c02*idet;
            i11=c11*idet; i12=c12*idet; i22=c22*idet;
        }
        for (int v=0; v<nvar; ++v) {
            double wc=W[(size_t)v*N+ci];
            double b0=0,b1=0,b2=0;
            if (ok) {
                for (int fc : m.cell_faces[ci]) {
                    int o=m.face_owner[fc], nn=m.face_neighbour[fc];
                    int nb=(o==ci)?nn:o; if(nb<0) continue;
                    double dx=cc[3*nb]-cx, dy=cc[3*nb+1]-cy, dz=cc[3*nb+2]-cz;
                    double d2=dx*dx+dy*dy+dz*dz; if(d2<eps) continue; double wk=1.0/d2;
                    double dphi=W[(size_t)v*N+nb]-wc;
                    b0+=wk*dx*dphi; b1+=wk*dy*dphi; b2+=wk*dz*dphi;
                }
            }
            double gx=i00*b0+i01*b1+i02*b2;
            double gy=i01*b0+i11*b1+i12*b2;
            double gz=i02*b0+i12*b1+i22*b2;
            grad[((size_t)v*N+ci)*3+0]=gx;
            grad[((size_t)v*N+ci)*3+1]=gy;
            grad[((size_t)v*N+ci)*3+2]=gz;
        }
    }

    // ---- (2) per-vertex min/max of cell-averages over cells sharing each node ----
    static U3NodeAdj NA;
    if (NA.mp!=&m || NA.N!=N) u3_build_nodeadj(m, NA);
    const int nn = NA.nn;
    std::vector<double> vmn((size_t)nvar*nn), vmx((size_t)nvar*nn);
    #pragma omp parallel for schedule(static)
    for (int p=0; p<nn; ++p) {
        int b=NA.off[p], e=NA.off[p+1];
        for (int v=0; v<nvar; ++v) {
            if (b==e) { vmn[(size_t)v*nn+p]=0; vmx[(size_t)v*nn+p]=0; continue; }
            double lo=W[(size_t)v*N+NA.cells[b]], hi=lo;
            for (int k=b+1;k<e;++k){ double q=W[(size_t)v*N+NA.cells[k]]; if(q<lo)lo=q; if(q>hi)hi=q; }
            vmn[(size_t)v*nn+p]=lo; vmx[(size_t)v*nn+p]=hi;
        }
    }
    const double* Xn = m.nodes.data();

    // ---- (3) psi_lmp vertex-LMP cap (uncapped room-to-vertex-bound / vertex projection) ----
    auto psi_lmp = [&](int ci, int v, double gx, double gy, double gz) -> double {
        double wc=W[(size_t)v*N+ci], p=psi_cap;
        double cx=cc[3*ci], cy=cc[3*ci+1], cz=cc[3*ci+2];
        for (int vn : m.cell_nodes[ci]) {
            if (vn<0 || vn>=nn) continue;
            double ox=Xn[3*vn]-cx, oy=Xn[3*vn+1]-cy, oz=Xn[3*vn+2]-cz;
            double proj=gx*ox+gy*oy+gz*oz;
            double allowed = (proj>=0.0) ? (vmx[(size_t)v*nn+vn]-wc) : (wc-vmn[(size_t)v*nn+vn]);
            double pk = (std::fabs(proj)>eps) ? std::max(allowed,0.0)/std::max(std::fabs(proj),eps) : psi_cap;
            if (pk<p) p=pk;
        }
        return p<0 ? 0 : (p>psi_cap ? psi_cap : p);
    };

    W_L.assign((size_t)nvar*Nf, 0.0);
    W_R.assign((size_t)nvar*Nf, 0.0);

    // ---- (4) per-face zero-BV CZL core ----
    #pragma omp parallel for schedule(dynamic,64)
    for (int f=0; f<Nf; ++f) {
        int o=m.face_owner[f], n=m.face_neighbour[f];
        double fx=m.face_centers[3*f+0], fy=m.face_centers[3*f+1], fz=m.face_centers[3*f+2];
        for (int v=0; v<nvar; ++v) {
            double wo=W[(size_t)v*N+o];
            double go0=grad[((size_t)v*N+o)*3+0], go1=grad[((size_t)v*N+o)*3+1], go2=grad[((size_t)v*N+o)*3+2];
            if (n < 0) {
                // boundary: owner-side LMP-limited (psi<=1) P1 recon to the face (old-BC convention)
                double inco=go0*(fx-cc[3*o])+go1*(fy-cc[3*o+1])+go2*(fz-cc[3*o+2]);
                double wb=wo + std::min(psi_lmp(o,v,go0,go1,go2), 1.0)*inco;
                W_L[(size_t)v*Nf+f]=wb; W_R[(size_t)v*Nf+f]=wb;
                continue;
            }
            double wn=W[(size_t)v*N+n];
            double gn0=grad[((size_t)v*N+n)*3+0], gn1=grad[((size_t)v*N+n)*3+1], gn2=grad[((size_t)v*N+n)*3+2];
            double inco=go0*(fx-cc[3*o])+go1*(fy-cc[3*o+1])+go2*(fz-cc[3*o+2]);
            double incn=gn0*(fx-cc[3*n])+gn1*(fy-cc[3*n+1])+gn2*(fz-cc[3*n+2]);
            double upo=go0*(cc[3*n]-cc[3*o])+go1*(cc[3*n+1]-cc[3*o+1])+go2*(cc[3*n+2]-cc[3*o+2]);
            double upn=gn0*(cc[3*o]-cc[3*n])+gn1*(cc[3*o+1]-cc[3*n+1])+gn2*(cc[3*o+2]-cc[3*n+2]);
            double ro=(std::fabs(upo)>eps)?(wn-wo)/upo:((wn-wo)*upo>=0?1e30:-1e30);
            double rn=(std::fabs(upn)>eps)?(wo-wn)/upn:((wo-wn)*upn>=0?1e30:-1e30);
            double plo=psi_lmp(o,v,go0,go1,go2), pln=psi_lmp(n,v,gn0,gn1,gn2);
            // lower band endpoint = van Leer; upper = downwind (r>0?2:0), Co-free (CZL).
            auto zlo = [&](double r){ return u3_tmlpu_vl(r); };
            auto zhi = [&](double r){ return dw_band ? (r>0.0?2.0:0.0) : u3_tmlpu_vl(r); };
            double WLvl =wo+std::min(zlo(ro),plo)*inco, WLcic=wo+std::min(zhi(ro),plo)*inco;
            double WRvl =wn+std::min(zlo(rn),pln)*incn, WRcic=wn+std::min(zhi(rn),pln)*incn;
            double WLlo=std::min(WLvl,WLcic), WLhi=std::max(WLvl,WLcic);
            double WRlo=std::min(WRvl,WRcic), WRhi=std::max(WRvl,WRcic);
            double ovlo=std::max(WLlo,WRlo), ovhi=std::min(WLhi,WRhi);
            double WLo=0.0, WRo=0.0;
            if (use_zerobv && ovlo<=ovhi && !BVDONLY) {           // zero-BV achievable
                double qmid=0.5*(wo+wn), qs;
                if (pick==1)      qs = (qmid<ovlo?ovlo:(qmid>ovhi?ovhi:qmid));    // MIN psi* (cell-midpoint clamp)
                else if (pick==2) qs = 0.5*(ovlo+ovhi);                          // overlap midpoint
                else              qs = (std::fabs(ovhi-qmid)>=std::fabs(ovlo-qmid))?ovhi:ovlo; // MAX psi*
                WLo=qs; WRo=qs;
            } else {                                              // fallback: discrete min-TBV BVD
                double bvvl=std::fabs(WLvl-WRvl), bvcic=std::fabs(WLcic-WRcic);
                if (bvcic<bvvl){ WLo=WLcic; WRo=WRcic; } else { WLo=WLvl; WRo=WRvl; }
            }
            // Euler positivity floor: rho (v==0) & p (v==nvar-1), component-wise.
            if (nvar>=4 && (v==0 || v==nvar-1)) {
                double fl=posfloor*std::min(wo,wn);
                if(WLo<fl)WLo=fl; if(WRo<fl)WRo=fl;
            }
            W_L[(size_t)v*Nf+f]=WLo; W_R[(size_t)v*Nf+f]=WRo;
        }
    }
}

} // namespace cfd
