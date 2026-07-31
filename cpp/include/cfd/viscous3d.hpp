// cfd/viscous3d.hpp — compressible Navier–Stokes viscous flux (3D).
//
// Adds the viscous contribution to the per-face flux array Fall used by
// euler3d_rhs. The compressible NS system is
//     ∂U/∂t + ∇·F_conv = ∇·G_visc
// so each face's TOTAL flux = F_conv − G_visc. Fall arrives holding F_conv·area
// (per face); this routine SUBTRACTS G_visc·area in place → Fall = (F_conv−G_visc)·area.
//
// Physics (μ const, ideal gas, R = 1 so T = p/rho):
//   τ_ij = μ ( ∂u_i/∂x_j + ∂u_j/∂x_i − (2/3) δ_ij div(u) )      (Newtonian, Stokes)
//   q_i  = −k ∂T/∂x_i,   k = μ cp / Pr,   cp = γ/(γ−1) R,   Pr = 0.72
//   G·n  = [ 0, (τ·n)_x, (τ·n)_y, (τ·n)_z, (τ·u − q)·n ]
//
// Gradients:
//   * per-cell ∇u,∇v,∇w,∇T via the FACE-neighbour weighted-LSQ (the SAME ReconCtx3D
//     ATA_inv/nb/d that drives the MUSCL gradient) — one 3-vector per field per cell.
//   * per-face gradient is the OVER-RELAXED / corrected average that avoids the
//     even-odd (checkerboard) decoupling of a plain centred average:
//         ∇φ_f = ½(∇φ_o+∇φ_n) + [ (φ_n−φ_o)/L_d − ½(∇φ_o+∇φ_n)·ê ] ê,
//     ê = d/|d|, d = centre-to-centre vector. (Minimum-correction = the d-aligned
//     part is replaced by the compact two-point difference; the rest is the average.)
//   * boundary faces: owner gradient + the BC ghost value for the normal difference,
//     so a no-slip wall gets ∂u/∂n ≠ 0 (the boundary layer); the centre-to-"centre"
//     vector is owner→face mirrored (2·(face−owner)).
//
// All math is additive; gated behind a caller flag (the inviscid Euler path never
// calls this). reconstruct3d.hpp (ReconCtx3D) is consumed read-only; mesh.hpp frozen.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/euler3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2.hpp"
#include <vector>
#include <cmath>
#include <functional>

namespace cfd {

// Viscous parameters. R = 1 (so T = p/rho). cp = gamma/(gamma-1). k = mu*cp/Pr.
struct ViscousParams {
    double mu = 0.0;     // dynamic viscosity (constant)
    double Pr = 0.72;    // Prandtl number
    double R  = 1.0;     // gas constant (R=1 ⇒ T = p/rho)
};

// Per-cell LSQ gradients of u,v,w,T (each N*3, layout [gx,gy,gz]) from the
// face-neighbour stencil in ReconCtx3D. Ghost handling: boundary faces are simply
// skipped here (the cell gradient uses only real neighbours — exactly as the MUSCL
// gradient does); the WALL normal-difference is injected later at the FACE level.
inline void viscous3d_cell_gradients(const Mesh& m, const Euler3D& eq,
                                     const ReconCtx3D& c, const std::vector<double>& Wc,
                                     double Rgas,
                                     std::vector<double>& gu, std::vector<double>& gv,
                                     std::vector<double>& gw, std::vector<double>& gT) {
    const int N = m.n_cells();
    gu.assign((size_t)N * 3, 0.0);
    gv.assign((size_t)N * 3, 0.0);
    gw.assign((size_t)N * 3, 0.0);
    gT.assign((size_t)N * 3, 0.0);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        const double* M = &c.ATA_inv[(size_t)ci * 9];
        double uc = Wc[(size_t)1 * N + ci], vc = Wc[(size_t)2 * N + ci], wc = Wc[(size_t)3 * N + ci];
        double rc = Wc[(size_t)0 * N + ci], pc = Wc[(size_t)4 * N + ci];
        double Tc = pc / (std::max(rc, 1e-30) * Rgas);
        double ru0=0,ru1=0,ru2=0, rv0=0,rv1=0,rv2=0, rw0=0,rw1=0,rw2=0, rT0=0,rT1=0,rT2=0;
        for (int k = 0; k < c.max_nb; ++k) {
            int nb = c.nb[(size_t)ci * c.max_nb + k];
            if (nb < 0) continue;
            double dx = c.d[((size_t)ci * c.max_nb + k) * 3 + 0];
            double dy = c.d[((size_t)ci * c.max_nb + k) * 3 + 1];
            double dz = c.d[((size_t)ci * c.max_nb + k) * 3 + 2];
            double du = Wc[(size_t)1 * N + nb] - uc;
            double dv = Wc[(size_t)2 * N + nb] - vc;
            double dw = Wc[(size_t)3 * N + nb] - wc;
            double rn = Wc[(size_t)0 * N + nb], pn = Wc[(size_t)4 * N + nb];
            double Tn = pn / (std::max(rn, 1e-30) * Rgas);
            double dT = Tn - Tc;
            ru0 += dx*du; ru1 += dy*du; ru2 += dz*du;
            rv0 += dx*dv; rv1 += dy*dv; rv2 += dz*dv;
            rw0 += dx*dw; rw1 += dy*dw; rw2 += dz*dw;
            rT0 += dx*dT; rT1 += dy*dT; rT2 += dz*dT;
        }
        gu[(size_t)ci*3+0] = M[0]*ru0 + M[1]*ru1 + M[2]*ru2;
        gu[(size_t)ci*3+1] = M[3]*ru0 + M[4]*ru1 + M[5]*ru2;
        gu[(size_t)ci*3+2] = M[6]*ru0 + M[7]*ru1 + M[8]*ru2;
        gv[(size_t)ci*3+0] = M[0]*rv0 + M[1]*rv1 + M[2]*rv2;
        gv[(size_t)ci*3+1] = M[3]*rv0 + M[4]*rv1 + M[5]*rv2;
        gv[(size_t)ci*3+2] = M[6]*rv0 + M[7]*rv1 + M[8]*rv2;
        gw[(size_t)ci*3+0] = M[0]*rw0 + M[1]*rw1 + M[2]*rw2;
        gw[(size_t)ci*3+1] = M[3]*rw0 + M[4]*rw1 + M[5]*rw2;
        gw[(size_t)ci*3+2] = M[6]*rw0 + M[7]*rw1 + M[8]*rw2;
        gT[(size_t)ci*3+0] = M[0]*rT0 + M[1]*rT1 + M[2]*rT2;
        gT[(size_t)ci*3+1] = M[3]*rT0 + M[4]*rT1 + M[5]*rT2;
        gT[(size_t)ci*3+2] = M[6]*rT0 + M[7]*rT1 + M[8]*rT2;
    }
}

// ROBUST cell gradients via the NODE-SHARING P2-LSQ (ReconCtx3DO2) — for UNSTRUCTURED
// meshes, where the face-neighbour linear LSQ above is ill-conditioned on boundary/skew
// tets (verified: boundary-cell grad error O(1e2) on a tet mesh -> garbage viscous flux).
// The o2 node-ring stencil (~26 nbrs) is well-conditioned; the gradient = first 3 of the
// 9 P2 coeffs. u,v,w from the primitive Wc; T = p/(rho R) built then differentiated.
inline void viscous3d_cell_gradients_o2(const Mesh& m, const ReconCtx3DO2& o2c,
                                        const std::vector<double>& Wc, double Rgas,
                                        std::vector<double>& gu, std::vector<double>& gv,
                                        std::vector<double>& gw, std::vector<double>& gT) {
    const int N = m.n_cells();
    gu.assign((size_t)N*3,0.0); gv.assign((size_t)N*3,0.0);
    gw.assign((size_t)N*3,0.0); gT.assign((size_t)N*3,0.0);
    std::vector<double> g9;
    reconstruct3d_o2_coeffs(m, o2c, Wc, 5, 1, g9);
    for(int i=0;i<N;++i){ gu[3*i]=g9[9*i]; gu[3*i+1]=g9[9*i+1]; gu[3*i+2]=g9[9*i+2]; }
    reconstruct3d_o2_coeffs(m, o2c, Wc, 5, 2, g9);
    for(int i=0;i<N;++i){ gv[3*i]=g9[9*i]; gv[3*i+1]=g9[9*i+1]; gv[3*i+2]=g9[9*i+2]; }
    reconstruct3d_o2_coeffs(m, o2c, Wc, 5, 3, g9);
    for(int i=0;i<N;++i){ gw[3*i]=g9[9*i]; gw[3*i+1]=g9[9*i+1]; gw[3*i+2]=g9[9*i+2]; }
    std::vector<double> Tarr((size_t)N);
    for(int i=0;i<N;++i) Tarr[i]=Wc[(size_t)4*N+i]/(std::max(Wc[(size_t)0*N+i],1e-30)*Rgas);
    reconstruct3d_o2_coeffs(m, o2c, Tarr, 1, 0, g9);
    for(int i=0;i<N;++i){ gT[3*i]=g9[9*i]; gT[3*i+1]=g9[9*i+1]; gT[3*i+2]=g9[9*i+2]; }
}

// FULL P2 coefficients (9 per cell per field) of u,v,w,T from the node-ring o2 LSQ — for the
// SKEWNESS/NON-ORTHOGONALITY-CORRECTED viscous face gradient. The face gradient is then the
// average of each adjacent cell's P2 gradient EVALUATED AT THE TRUE FACE CENTROID x_f
// (grad + Hessian·(x_f−x_cell)) instead of the cell-centroid gradient — which removes the
// O(h) skewness error of the centroid scheme (MMS: 1st→2nd order, distortion-robust, 2–7× lower
// error up to 90° non-orthogonality). Layout cu[9*i + {0,1,2}]=grad, {3,4,5}=∂²(xx,yy,zz),
// {6,7,8}=∂²(xy,xz,yz). T = p/(rho R).
inline void viscous3d_cell_coeffs_o2(const Mesh& m, const ReconCtx3DO2& o2c,
                                     const std::vector<double>& Wc, double Rgas,
                                     std::vector<double>& cu, std::vector<double>& cv,
                                     std::vector<double>& cw, std::vector<double>& cT) {
    const int N = m.n_cells();
    reconstruct3d_o2_coeffs(m, o2c, Wc, 5, 1, cu);
    reconstruct3d_o2_coeffs(m, o2c, Wc, 5, 2, cv);
    reconstruct3d_o2_coeffs(m, o2c, Wc, 5, 3, cw);
    std::vector<double> Tarr((size_t)N);
    for(int i=0;i<N;++i) Tarr[i]=Wc[(size_t)4*N+i]/(std::max(Wc[(size_t)0*N+i],1e-30)*Rgas);
    reconstruct3d_o2_coeffs(m, o2c, Tarr, 1, 0, cT);
}

// Subtract the viscous flux G·area from Fall for EVERY face (Fall holds F_conv·area
// on entry; (F_conv−G)·area on exit). `bc_ghost` produces the boundary ghost
// PRIMITIVE state for a boundary face (same signature/role as apply_bc3d but bound
// by the caller so this header stays solver-agnostic). For interior faces it is
// unused. Layout of Fall: Fall[v*Nf + f].
inline void viscous3d_add_face_flux(
        const Mesh& m, const Euler3D& eq, const ViscousParams& vp,
        const std::vector<double>& Wc,
        const std::vector<double>& gu, const std::vector<double>& gv,
        const std::vector<double>& gw, const std::vector<double>& gT,
        const std::function<void(int /*face*/, const double wL[5], double wR[5])>& bc_ghost,
        std::vector<double>& Fall) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double mu = vp.mu;
    const double cp = eq.gamma / (eq.gamma - 1.0) * vp.R;   // R=1 default
    const double kcond = mu * cp / vp.Pr;
    const double Rgas = vp.R;

    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double nx = m.face_normals[(size_t)f*3+0], ny = m.face_normals[(size_t)f*3+1], nz = m.face_normals[(size_t)f*3+2];
        double area = m.face_areas[f];

        // owner primitive
        double uo = Wc[(size_t)1*N+o], vo = Wc[(size_t)2*N+o], wo = Wc[(size_t)3*N+o];
        double ro = Wc[(size_t)0*N+o], po = Wc[(size_t)4*N+o];
        double To = po / (std::max(ro,1e-30) * Rgas);

        // owner gradients
        double guo[3] = {gu[(size_t)o*3+0], gu[(size_t)o*3+1], gu[(size_t)o*3+2]};
        double gvo[3] = {gv[(size_t)o*3+0], gv[(size_t)o*3+1], gv[(size_t)o*3+2]};
        double gwo[3] = {gw[(size_t)o*3+0], gw[(size_t)o*3+1], gw[(size_t)o*3+2]};
        double gTo[3] = {gT[(size_t)o*3+0], gT[(size_t)o*3+1], gT[(size_t)o*3+2]};

        // neighbour primitive + gradients (for boundary, from the BC ghost; the
        // ghost's gradient is taken = owner's so the average term is the owner grad,
        // and the normal difference carries the wall condition).
        double un, vn, wn, Tn;
        double gun[3], gvn[3], gwn[3], gTn[3];
        double dcx, dcy, dcz;   // centre-to-centre (or 2×owner→face for boundary)
        // velocity for the energy (τ·u) term — use the face-average velocity
        double uf, vf, wf;

        if (n >= 0) {
            un = Wc[(size_t)1*N+n]; vn = Wc[(size_t)2*N+n]; wn = Wc[(size_t)3*N+n];
            double rn = Wc[(size_t)0*N+n], pn = Wc[(size_t)4*N+n];
            Tn = pn / (std::max(rn,1e-30) * Rgas);
            for (int a = 0; a < 3; ++a) {
                gun[a] = gu[(size_t)n*3+a]; gvn[a] = gv[(size_t)n*3+a];
                gwn[a] = gw[(size_t)n*3+a]; gTn[a] = gT[(size_t)n*3+a];
            }
            dcx = m.cell_centers[(size_t)n*3+0] - m.cell_centers[(size_t)o*3+0];
            dcy = m.cell_centers[(size_t)n*3+1] - m.cell_centers[(size_t)o*3+1];
            dcz = m.cell_centers[(size_t)n*3+2] - m.cell_centers[(size_t)o*3+2];
            uf = 0.5*(uo+un); vf = 0.5*(vo+vn); wf = 0.5*(wo+wn);
        } else {
            // boundary: ghost primitive from the BC; ghost gradient = owner gradient.
            double wL[5] = {ro, uo, vo, wo, po}, wR[5];
            if (bc_ghost) bc_ghost(f, wL, wR); else for (int v=0; v<5; ++v) wR[v]=wL[v];
            un = wR[1]; vn = wR[2]; wn = wR[3];
            double rn = wR[0], pn = wR[4];
            Tn = pn / (std::max(rn,1e-30) * Rgas);
            for (int a = 0; a < 3; ++a) { gun[a]=guo[a]; gvn[a]=gvo[a]; gwn[a]=gwo[a]; gTn[a]=gTo[a]; }
            // owner→face vector, doubled to act as the owner→ghost centre-to-centre
            // (ghost mirrored across the wall face).
            dcx = 2.0*(m.face_centers[(size_t)f*3+0] - m.cell_centers[(size_t)o*3+0]);
            dcy = 2.0*(m.face_centers[(size_t)f*3+1] - m.cell_centers[(size_t)o*3+1]);
            dcz = 2.0*(m.face_centers[(size_t)f*3+2] - m.cell_centers[(size_t)o*3+2]);
            // wall velocity for the energy term: the VALUE AT THE WALL FACE, i.e.
            // ½(inner+ghost). For no-slip ghost=−inner ⇒ uf=0 (no work at the wall);
            // for a slip wall the tangential component survives.
            uf = 0.5*(uo+un); vf = 0.5*(vo+vn); wf = 0.5*(wo+wn);
        }

        double Ld2 = dcx*dcx + dcy*dcy + dcz*dcz;
        double Ld = std::sqrt(std::max(Ld2, 1e-300));
        double ex = dcx/Ld, ey = dcy/Ld, ez = dcz/Ld;

        // over-relaxed face gradient per field:
        //   gf = ½(go+gn) + [ (φn−φo)/Ld − ½(go+gn)·ê ] ê
        auto facegrad = [&](const double go[3], const double gn[3], double dphi, double gf[3]) {
            double ax = 0.5*(go[0]+gn[0]), ay = 0.5*(go[1]+gn[1]), az = 0.5*(go[2]+gn[2]);
            double adote = ax*ex + ay*ey + az*ez;
            double corr = dphi/Ld - adote;
            gf[0] = ax + corr*ex; gf[1] = ay + corr*ey; gf[2] = az + corr*ez;
        };
        double dudx[3], dvdx[3], dwdx[3], dTdx[3];
        facegrad(guo, gun, un-uo, dudx);
        facegrad(gvo, gvn, vn-vo, dvdx);
        facegrad(gwo, gwn, wn-wo, dwdx);
        facegrad(gTo, gTn, Tn-To, dTdx);

        // strain / stress (Newtonian, Stokes hypothesis)
        double div = dudx[0] + dvdx[1] + dwdx[2];
        double txx = mu*(2.0*dudx[0] - (2.0/3.0)*div);
        double tyy = mu*(2.0*dvdx[1] - (2.0/3.0)*div);
        double tzz = mu*(2.0*dwdx[2] - (2.0/3.0)*div);
        double txy = mu*(dudx[1] + dvdx[0]);
        double txz = mu*(dudx[2] + dwdx[0]);
        double tyz = mu*(dvdx[2] + dwdx[1]);

        // viscous flux through the face normal: G·n
        double Gmx = txx*nx + txy*ny + txz*nz;
        double Gmy = txy*nx + tyy*ny + tyz*nz;
        double Gmz = txz*nx + tyz*ny + tzz*nz;
        // energy: (τ·u)·n − q·n, with q = −k ∇T  ⇒ −q·n = +k ∇T·n
        double tun_x = txx*uf + txy*vf + txz*wf;   // (τ·u)_x
        double tun_y = txy*uf + tyy*vf + tyz*wf;
        double tun_z = txz*uf + tyz*vf + tzz*wf;
        double qdotn = -kcond*(dTdx[0]*nx + dTdx[1]*ny + dTdx[2]*nz);   // q·n
        double Gen = (tun_x*nx + tun_y*ny + tun_z*nz) - qdotn;

        // Fall = (F_conv − G)·area : subtract G·area.
        Fall[(size_t)1*Nf+f] -= Gmx*area;
        Fall[(size_t)2*Nf+f] -= Gmy*area;
        Fall[(size_t)3*Nf+f] -= Gmz*area;
        Fall[(size_t)4*Nf+f] -= Gen*area;
        // mass row (v=0) is unchanged (no viscous mass flux).
    }
}

// SKEWNESS/NON-ORTHOGONALITY-CORRECTED viscous face flux. Identical physics to
// viscous3d_add_face_flux but the face gradient is the average of the two adjacent cells'
// P2 reconstruction gradients EVALUATED AT THE TRUE FACE CENTROID x_f (cu/cv/cw/cT carry the
// 9 P2 coeffs from viscous3d_cell_coeffs_o2). This is fully skew+non-orthogonality corrected
// (it does NOT difference cell-centre values along d), giving 2nd-order face gradients on
// arbitrary distorted meshes (MMS-verified). Interior: ½(∇φ_o@f + ∇φ_n@f). Boundary: owner
// P2 gradient at x_f with the wall-normal component overridden by the BC ghost value (so the
// no-slip/adiabatic condition is still injected).
inline void viscous3d_add_face_flux_p2face(
        const Mesh& m, const Euler3D& eq, const ViscousParams& vp,
        const std::vector<double>& Wc,
        const std::vector<double>& cu, const std::vector<double>& cv,
        const std::vector<double>& cw, const std::vector<double>& cT,
        const std::function<void(int, const double wL[5], double wR[5])>& bc_ghost,
        std::vector<double>& Fall) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double mu = vp.mu;
    const double cp = eq.gamma / (eq.gamma - 1.0) * vp.R;
    const double kcond = mu * cp / vp.Pr;
    const double Rgas = vp.R;
    auto gradAt=[](const double* c,double dx,double dy,double dz,double g[3]){
        g[0]=c[0]+c[3]*dx+c[6]*dy+c[7]*dz;
        g[1]=c[1]+c[6]*dx+c[4]*dy+c[8]*dz;
        g[2]=c[2]+c[7]*dx+c[8]*dy+c[5]*dz;
    };
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        double nx=m.face_normals[(size_t)f*3+0], ny=m.face_normals[(size_t)f*3+1], nz=m.face_normals[(size_t)f*3+2];
        double area = m.face_areas[f];
        double fx=m.face_centers[(size_t)f*3+0], fy=m.face_centers[(size_t)f*3+1], fz=m.face_centers[(size_t)f*3+2];
        double xo=m.cell_centers[(size_t)o*3+0], yo=m.cell_centers[(size_t)o*3+1], zo=m.cell_centers[(size_t)o*3+2];
        double uo=Wc[(size_t)1*N+o], vo=Wc[(size_t)2*N+o], wo=Wc[(size_t)3*N+o];
        double ro=Wc[(size_t)0*N+o], po=Wc[(size_t)4*N+o]; double To=po/(std::max(ro,1e-30)*Rgas);
        // owner P2 gradients at the TRUE face centroid
        double guo[3],gvo[3],gwo[3],gTo[3];
        gradAt(&cu[(size_t)o*9],fx-xo,fy-yo,fz-zo,guo);
        gradAt(&cv[(size_t)o*9],fx-xo,fy-yo,fz-zo,gvo);
        gradAt(&cw[(size_t)o*9],fx-xo,fy-yo,fz-zo,gwo);
        gradAt(&cT[(size_t)o*9],fx-xo,fy-yo,fz-zo,gTo);
        double dudx[3],dvdx[3],dwdx[3],dTdx[3]; double uf,vf,wf;
        if (n >= 0) {
            // INTERIOR: pure average of both cells' P2 gradients at x_f (skew/non-ortho exact)
            double xn=m.cell_centers[(size_t)n*3+0], yn=m.cell_centers[(size_t)n*3+1], zn=m.cell_centers[(size_t)n*3+2];
            double gun[3],gvn[3],gwn[3],gTn[3];
            gradAt(&cu[(size_t)n*9],fx-xn,fy-yn,fz-zn,gun);
            gradAt(&cv[(size_t)n*9],fx-xn,fy-yn,fz-zn,gvn);
            gradAt(&cw[(size_t)n*9],fx-xn,fy-yn,fz-zn,gwn);
            gradAt(&cT[(size_t)n*9],fx-xn,fy-yn,fz-zn,gTn);
            for(int a=0;a<3;++a){ dudx[a]=0.5*(guo[a]+gun[a]); dvdx[a]=0.5*(gvo[a]+gvn[a]);
                dwdx[a]=0.5*(gwo[a]+gwn[a]); dTdx[a]=0.5*(gTo[a]+gTn[a]); }
            double un=Wc[(size_t)1*N+n], vn=Wc[(size_t)2*N+n], wn=Wc[(size_t)3*N+n];
            uf=0.5*(uo+un); vf=0.5*(vo+vn); wf=0.5*(wo+wn);
        } else {
            // BOUNDARY: owner P2 gradient at x_f, wall-normal component from the BC ghost value.
            double wL[5]={ro,uo,vo,wo,po}, wR[5];
            if (bc_ghost) bc_ghost(f,wL,wR); else for(int v=0;v<5;++v) wR[v]=wL[v];
            double un=wR[1], vn=wR[2], wn=wR[3]; double rn=wR[0], pn=wR[4]; double Tn=pn/(std::max(rn,1e-30)*Rgas);
            double dcx=2.0*(fx-xo), dcy=2.0*(fy-yo), dcz=2.0*(fz-zo);
            double Ld=std::sqrt(std::max(dcx*dcx+dcy*dcy+dcz*dcz,1e-300)); double ex=dcx/Ld,ey=dcy/Ld,ez=dcz/Ld;
            auto corrnorm=[&](double g[3],double dphi){ double c=dphi/Ld-(g[0]*ex+g[1]*ey+g[2]*ez); g[0]+=c*ex; g[1]+=c*ey; g[2]+=c*ez; };
            corrnorm(guo,un-uo); corrnorm(gvo,vn-vo); corrnorm(gwo,wn-wo); corrnorm(gTo,Tn-To);
            for(int a=0;a<3;++a){ dudx[a]=guo[a]; dvdx[a]=gvo[a]; dwdx[a]=gwo[a]; dTdx[a]=gTo[a]; }
            uf=0.5*(uo+un); vf=0.5*(vo+vn); wf=0.5*(wo+wn);
        }
        double div=dudx[0]+dvdx[1]+dwdx[2];
        double txx=mu*(2.0*dudx[0]-(2.0/3.0)*div), tyy=mu*(2.0*dvdx[1]-(2.0/3.0)*div), tzz=mu*(2.0*dwdx[2]-(2.0/3.0)*div);
        double txy=mu*(dudx[1]+dvdx[0]), txz=mu*(dudx[2]+dwdx[0]), tyz=mu*(dvdx[2]+dwdx[1]);
        double Gmx=txx*nx+txy*ny+txz*nz, Gmy=txy*nx+tyy*ny+tyz*nz, Gmz=txz*nx+tyz*ny+tzz*nz;
        double tun_x=txx*uf+txy*vf+txz*wf, tun_y=txy*uf+tyy*vf+tyz*wf, tun_z=txz*uf+tyz*vf+tzz*wf;
        double qdotn=-kcond*(dTdx[0]*nx+dTdx[1]*ny+dTdx[2]*nz);
        double Gen=(tun_x*nx+tun_y*ny+tun_z*nz)-qdotn;
        Fall[(size_t)1*Nf+f]-=Gmx*area; Fall[(size_t)2*Nf+f]-=Gmy*area;
        Fall[(size_t)3*Nf+f]-=Gmz*area; Fall[(size_t)4*Nf+f]-=Gen*area;
    }
}

} // namespace cfd
