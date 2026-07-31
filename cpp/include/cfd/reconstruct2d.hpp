// cfd/reconstruct2d.hpp — 2D limited-linear reconstruction: Barth-Jespersen
// multidimensional limiter on the vertex (Park-Yoon-Kim MLP-u) stencil with
// vertex min/max bounds, plus the optional T-MLP-u face LMP bound. This is the
// "T-MLP-u-L" core (mlp_u1 = BJ+vertex+vertex_bounds; mlp_u1_tmlpu adds the face
// bound). Port of _limited_linear_2d (solver/solve_T-MLP-u/reconstruction.py).
//
// Boundary faces fall back to first-order (owner value), exactly as the Python.
#pragma once
#include "cfd/mesh.hpp"
#include "cfd/io_vtk.hpp"   // tmlpu_branch_flag() global buffer for the TMLPU_FLAG diagnostic
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <cstdlib>
#include <string>

namespace cfd {

struct ReconCtx {
    int N = 0, Nn = 0, max_nb = 0, max_v = 0, max_v2c = 0;
    std::vector<int>    nb;        // N*max_nb (-1 pad)
    std::vector<double> d;         // N*max_nb*2 (offsets, zero where invalid)
    std::vector<double> w;         // N*max_nb (LSQ weights; 1 or IDW 1/dist^p)
    std::vector<double> ATA_inv;   // N*4
    std::vector<int>    sample_vid;// N*max_v (node id, -1 pad)
    std::vector<double> sample_off;// N*max_v*2
    std::vector<int>    v2c;       // Nn*max_v2c (-1 pad)
};

// Build vertex-stencil reconstruction context (vertex_bounds path).
//   idw_p > 0 weights the LSQ gradient by 1/dist^idw_p (T-MLP-u-L); 0 = uniform
//   (mlp_u1 / Barth-Jespersen).
inline ReconCtx build_recon_ctx(const Mesh& m, double idw_p = 0.0) {
    ReconCtx c;
    const int N = m.n_cells();
    c.N = N;
    c.Nn = (int)m.nodes.size() / 2;
    const double* cc = m.cell_centers.data();

    // vertex neighbours: cells sharing any node (1-ring).
    std::vector<std::vector<int>> vcells(c.Nn);
    for (int ci = 0; ci < N; ++ci)
        for (int v : m.cell_nodes[ci]) vcells[v].push_back(ci);
    std::vector<std::vector<int>> nbl(N);
    for (int ci = 0; ci < N; ++ci) {
        std::set<int> s;
        for (int v : m.cell_nodes[ci]) for (int c2 : vcells[v]) if (c2 != ci) s.insert(c2);
        nbl[ci].assign(s.begin(), s.end());
        c.max_nb = std::max(c.max_nb, (int)nbl[ci].size());
    }
    c.max_nb = std::max(c.max_nb, 1);
    c.nb.assign((size_t)N * c.max_nb, -1);
    c.d.assign((size_t)N * c.max_nb * 2, 0.0);
    c.w.assign((size_t)N * c.max_nb, 0.0);
    c.ATA_inv.assign((size_t)N * 4, 0.0);
    for (int ci = 0; ci < N; ++ci) {
        double a00 = 0, a01 = 0, a11 = 0;
        for (int k = 0; k < (int)nbl[ci].size(); ++k) {
            int nb = nbl[ci][k];
            c.nb[(size_t)ci * c.max_nb + k] = nb;
            double dx = cc[nb*2+0] - cc[ci*2+0], dy = cc[nb*2+1] - cc[ci*2+1];
            double wk = 1.0;
            if (idw_p > 0.0) {
                double dist = std::sqrt(dx*dx + dy*dy);
                wk = 1.0 / std::pow(std::max(dist, 1e-30), idw_p);
            }
            c.d[((size_t)ci * c.max_nb + k) * 2 + 0] = dx;
            c.d[((size_t)ci * c.max_nb + k) * 2 + 1] = dy;
            c.w[(size_t)ci * c.max_nb + k] = wk;
            a00 += wk*dx*dx; a01 += wk*dx*dy; a11 += wk*dy*dy;
        }
        double det = a00 * a11 - a01 * a01;
        if (std::fabs(det) > 1e-30) {
            c.ATA_inv[ci*4+0] =  a11 / det; c.ATA_inv[ci*4+3] =  a00 / det;
            c.ATA_inv[ci*4+1] = -a01 / det; c.ATA_inv[ci*4+2] = -a01 / det;
        }
    }
    // sample points = cell vertices.
    for (int ci = 0; ci < N; ++ci) c.max_v = std::max(c.max_v, (int)m.cell_nodes[ci].size());
    c.max_v = std::max(c.max_v, 1);
    c.sample_vid.assign((size_t)N * c.max_v, -1);
    c.sample_off.assign((size_t)N * c.max_v * 2, 0.0);
    for (int ci = 0; ci < N; ++ci) {
        const auto& vs = m.cell_nodes[ci];
        for (int k = 0; k < (int)vs.size(); ++k) {
            int v = vs[k];
            c.sample_vid[(size_t)ci * c.max_v + k] = v;
            c.sample_off[((size_t)ci * c.max_v + k) * 2 + 0] = m.nodes[v*2+0] - cc[ci*2+0];
            c.sample_off[((size_t)ci * c.max_v + k) * 2 + 1] = m.nodes[v*2+1] - cc[ci*2+1];
        }
    }
    // node -> cells (for vertex bounds).
    for (auto& vc : vcells) c.max_v2c = std::max(c.max_v2c, (int)vc.size());
    c.max_v2c = std::max(c.max_v2c, 1);
    c.v2c.assign((size_t)c.Nn * c.max_v2c, -1);
    for (int v = 0; v < c.Nn; ++v)
        for (int k = 0; k < (int)vcells[v].size(); ++k)
            c.v2c[(size_t)v * c.max_v2c + k] = vcells[v][k];
    return c;
}

// BJ admissible-ratio limiter (limiters._limiter_phi, 'bj' branch).
inline double bj_phi(double delta, double center, double lo, double hi) {
    const double eps = 1e-30;
    double allowed = delta >= 0.0 ? (hi - center) : (center - lo);
    double phi = (std::fabs(delta) > eps)
               ? std::max(allowed, 0.0) / std::max(std::fabs(delta), eps) : 1.0;
    return phi < 0.0 ? 0.0 : (phi > 1.0 ? 1.0 : phi);
}

// MLP-u2 (Venkatakrishnan) differentiable limiter on the SAME vertex bounds as
// BJ (MLP-u1) and the SAME linear reconstruction -> identical cost, less diffusive
// (smooth, no hard clip at smooth extrema). f_V = (dp^2 + 2 dp dm + eps2) /
// (dp^2 + dp dm + 2 dm^2 + eps2), dm = increment, dp = signed allowed room,
// eps2 = (K*h)^3 (K=0 ~ BJ-dissipative, K>>1 ~ limiter off). Liu et al. 2017.
inline double venk_phi(double delta, double center, double lo, double hi, double eps2) {
    const double eps = 1e-30;
    if (std::fabs(delta) <= eps) return 1.0;
    double dp = delta >= 0.0 ? (hi - center) : (lo - center);   // signed Delta+
    double dm = delta;                                          // Delta-
    double phi = (dp*dp + 2.0*dp*dm + eps2) / (dp*dp + dp*dm + 2.0*dm*dm + eps2);
    return phi < 0.0 ? 0.0 : (phi > 1.0 ? 1.0 : phi);
}

// TVD limiter functions psi(r) (r = far-upwind ratio). van_leer (smooth tier),
// cicsam/Hyper-C @ Co (sharp tier, compressive up to 2).
inline double psi_van_leer(double r) { if (r <= 0.0) return 0.0; return 2.0*r/(1.0+r); }
inline double psi_cicsam(double r, double Co) { if (r <= 0.0) return 0.0;
    double f = (1.0 - Co)/std::max(Co, 1e-10); double p = 2.0*r*f; return p > 2.0 ? 2.0 : p; }
// MSTACS (STACS family, Darwish-Moukalled): compressive interface limiter = SUPERBEE
// high-resolution branch bounded by the Hyper-C Courant cap (2r/Co). Less wrinkly than
// raw Hyper-C/CICSAM (superbee ramp at mid-r), still compressive up to 2.
inline double psi_mstacs(double r, double Co) { if (r <= 0.0) return 0.0;
    double sb = std::max(std::min(2.0*r, 1.0), std::min(r, 2.0));   // superbee HR
    double cap = 2.0*r/std::max(Co, 1e-10);                          // Hyper-C Courant cap
    double p = std::min(sb, cap); return p > 2.0 ? 2.0 : p; }
// C1 smoothstep: 0 below lo, 1 above hi, smooth cubic between. Continuous gate (no
// hard threshold) so the scheme does not switch discontinuously across mesh/regime.
inline double smoothstep(double lo, double hi, double x) {
    if (hi <= lo) return x >= hi ? 1.0 : 0.0;
    double t = (x - lo)/(hi - lo); t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return t*t*(3.0 - 2.0*t); }

// GENUINE T-MLP-u (linear, no quadrature): per cell an LSQ-residual smoothness
// indicator gates the limiter — SHARP cells (residual >= thr, e.g. a slot edge)
// use the compressive cicsam(r), SMOOTH cells (vortex/cone/hump) use van_leer ~
// MLP-u1. The vertex-LMP wrapper psi_LMP (room-to-vertex-bound / increment, NOT
// capped at 1 — that is the T-MLP-u key, allowing compression up to the bound)
// keeps it monotone. psi = min(psi_TVD(r), psi_LMP). Region-adaptive: compress at
// discontinuities, preserve vortices. (the user's intuition, faithful + linear.)
inline void reconstruct_tmlpu_gated(const Mesh& m, const ReconCtx& c,
                                    const std::vector<double>& W, int nvar,
                                    std::vector<double>& W_L, std::vector<double>& W_R,
                                    double thr, double Co, double psi_cap) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double* cc = m.cell_centers.data();
    static const bool FLAG = std::getenv("TMLPU_FLAG")!=nullptr;   // diagnostic: record per-cell density recon branch
    std::vector<double> grad((size_t)nvar*N*2, 0.0);
    std::vector<char> sharp((size_t)nvar*N, 0);
    std::vector<double> vmn((size_t)nvar*c.Nn), vmx((size_t)nvar*c.Nn);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) for (int v = 0; v < nvar; ++v) {
        double r0=0,r1=0, wc=W[(size_t)v*N+ci];
        for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double dphi=W[(size_t)v*N+nb]-wc, wk=c.w[(size_t)ci*c.max_nb+k];
            r0+=wk*c.d[((size_t)ci*c.max_nb+k)*2+0]*dphi; r1+=wk*c.d[((size_t)ci*c.max_nb+k)*2+1]*dphi; }
        double gx=c.ATA_inv[ci*4+0]*r0+c.ATA_inv[ci*4+1]*r1, gy=c.ATA_inv[ci*4+2]*r0+c.ATA_inv[ci*4+3]*r1;
        grad[((size_t)v*N+ci)*2+0]=gx; grad[((size_t)v*N+ci)*2+1]=gy;
        // SHARP-cell detector. GATE=residual: LSQ residual ||dphi-grad.d||/||dphi||.
        // GATE=gap: bimodality — sorted (self+neighbour) values, largest interior
        // GAP / range. A top-hat STEP (slot) is bimodal (big gap); a smooth
        // ramp/bump (cone/hump) is continuous (small gap) -> NOT flagged ->
        // shape preserved. GATE=both: residual AND gap (default: residual).
        // DEFAULT = both (residual AND gap-bimodality) — separates a true step
        // from smooth curvature, preserving cone/hump shapes while sharpening the
        // slot. TMLPU_GATE=residual or =gap to override.
        // GATE: 0 residual, 1 gap, 2 both(res&gap), 3 aniso (jump-tensor anisotropy
        // A=(l1-l2)/(l1+l2): directional EDGE high A vs radial-smooth low A), 4 res&aniso.
        static const char* GT = std::getenv("TMLPU_GATE");
        static const int gate = GT ? (std::string(GT)=="gap"?1:(std::string(GT)=="residual"?0:
            (std::string(GT)=="aniso"?3:(std::string(GT)=="resaniso"?4:2))) ) : 2;
        double num=0, den=0;
        for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double dphi=W[(size_t)v*N+nb]-wc;
            double pred=gx*c.d[((size_t)ci*c.max_nb+k)*2+0]+gy*c.d[((size_t)ci*c.max_nb+k)*2+1];
            double e=dphi-pred; num+=e*e; den+=dphi*dphi; }
        double sm = std::sqrt(num)/std::sqrt(std::max(den,1e-30));
        bool sh_res = (sm >= thr);
        bool sh_gap = false;
        if (gate >= 1) {
            double vals[64]; int nv=0; vals[nv++]=wc;
            for (int k=0;k<c.max_nb && nv<64;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                vals[nv++]=W[(size_t)v*N+nb]; }
            for (int a=1;a<nv;++a){ double t=vals[a]; int b=a-1; while(b>=0&&vals[b]>t){vals[b+1]=vals[b];--b;} vals[b+1]=t; }
            double rng = vals[nv-1]-vals[0], maxgap=0.0;
            for (int a=1;a<nv;++a){ double g=vals[a]-vals[a-1]; if(g>maxgap)maxgap=g; }
            sh_gap = rng>1e-30 && (maxgap/rng >= thr);
        }
        bool sh_aniso = false;
        if (gate >= 3) {
            // jump-moment tensor T = Σ_k (ΔW_k)² d_k d_kᵀ (2x2 symmetric).
            double Txx=0,Txy=0,Tyy=0;
            for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                double dphi=W[(size_t)v*N+nb]-wc, w2=dphi*dphi;
                double dx=c.d[((size_t)ci*c.max_nb+k)*2+0], dy=c.d[((size_t)ci*c.max_nb+k)*2+1];
                Txx+=w2*dx*dx; Txy+=w2*dx*dy; Tyy+=w2*dy*dy; }
            double tr=Txx+Tyy, dsc=std::sqrt(std::max((Txx-Tyy)*(Txx-Tyy)+4.0*Txy*Txy,0.0));
            double l1=0.5*(tr+dsc), l2=0.5*(tr-dsc);
            double A = (l1+l2>1e-30) ? (l1-l2)/(l1+l2) : 0.0;   // 1=1D edge, 0=isotropic/radial
            sh_aniso = (A >= thr);
        }
        bool s;
        switch(gate){ case 0: s=sh_res; break; case 1: s=sh_gap; break;
            case 3: s=sh_aniso; break; case 4: s=sh_res&&sh_aniso; break;
            default: s=sh_res&&sh_gap; }
        static const double PSHOCK_VETO = std::getenv("TMLPU_PSHOCK_VETO")
            ? std::max(0.0, std::atof(std::getenv("TMLPU_PSHOCK_VETO"))) : -1.0;
        if (s && PSHOCK_VETO >= 0.0 && nvar >= 4) {
            double pc = W[(size_t)(nvar-1)*N+ci];
            double pjump = 0.0;
            for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                double pn = W[(size_t)(nvar-1)*N+nb];
                double pj = std::fabs(pn-pc)/(std::fabs(pn)+std::fabs(pc)+1e-30);
                if (pj > pjump) pjump = pj; }
            if (pjump > PSHOCK_VETO) s = false;
        }
        sharp[(size_t)v*N+ci] = s ? 1 : 0;
    }
    // SHEAR-AWARE veto (Euler only): a slip line / vortex is rotation-dominated
    // (omega^2 >> theta^2); cicsam compression there over-stabilizes and suppresses
    // KH roll-up. Ducros sensor sd = omega^2/(omega^2+theta^2): ~1 shear, ~0 shock.
    // Where sd >= shear_thr, veto the sharp flag (keep the smooth=mlp_u1 tier) so
    // slip-stream vorticity grows; shocks (theta-dominated) stay compressed.
    // Velocity gradients already in grad[v=1 (u), v=2 (v)]. Off by default.
    static const char* SVe = std::getenv("TMLPU_SHEAR_VETO");
    static const double shear_thr = SVe ? std::atof(SVe) : -1.0;  // <0 => disabled
    if (shear_thr >= 0.0 && nvar >= 4) {
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double dudx=grad[((size_t)1*N+ci)*2+0], dudy=grad[((size_t)1*N+ci)*2+1];
            double dvdx=grad[((size_t)2*N+ci)*2+0], dvdy=grad[((size_t)2*N+ci)*2+1];
            double theta=dudx+dvdy, omega=dvdx-dudy;
            double sd = (omega*omega)/(omega*omega + theta*theta + 1e-30);
            if (sd >= shear_thr) for (int v=0;v<nvar;++v) sharp[(size_t)v*N+ci] = 0;
        }
    }
    // SHOCK veto (Euler, the COMBINED gate): residual+gap finds a real discontinuity,
    // but Ducros classifies it. Where sd < shock_thr (compression-dominated = SHOCK),
    // veto the sharp flag -> shocks are NOT cicsam-compressed (stay clean/monotone via
    // mlp_u1), while contacts/slip-lines (sd high = rotation/shear, OR low compression)
    // keep their sharp flag and get compressed. => compress contacts NOT shocks. Off by default.
    static const char* SkV = std::getenv("TMLPU_SHOCK_VETO");
    static const double shock_thr = SkV ? std::atof(SkV) : -1.0;  // <0 => disabled
    if (shock_thr >= 0.0 && nvar >= 4) {
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double dudx=grad[((size_t)1*N+ci)*2+0], dudy=grad[((size_t)1*N+ci)*2+1];
            double dvdx=grad[((size_t)2*N+ci)*2+0], dvdy=grad[((size_t)2*N+ci)*2+1];
            double theta=dudx+dvdy, omega=dvdx-dudy;
            double sd=(omega*omega)/(omega*omega + theta*theta + 1e-30);
            if (sd < shock_thr) for (int v=0;v<nvar;++v) sharp[(size_t)v*N+ci] = 0;
        }
    }
    // ===== T-MLP-u-C2 (robust unified): dimensionless sensors + continuous smoothstep
    // gates with SEPARATED responsibilities (Ducros+pressure-jump = shock PROTECT;
    // bimodality-gap = contact/interface; residual = smoothness). No hard threshold,
    // no all-sensor product. samt[ci] in [0,1] = continuous sharp-amount applied as a
    // van_leer<->MSTACS blend in the face loop. Off by default (TMLPU_C2). =====
    static const bool C2 = std::getenv("TMLPU_C2") != nullptr;
    static const char* C2PF = std::getenv("TMLPU_C2_POSFLOOR");
    const double posfloor = C2PF ? std::atof(C2PF) : 0.1;   // face rho,p >= posfloor*W_c
    static const bool C2_SMU1 = std::getenv("TMLPU_C2_SMOOTH_U1") != nullptr; // smooth tier=pt1.0(mlp_u1) vs van_leer
    static const bool C2_SHF  = std::getenv("TMLPU_C2_SHEARFREE") != nullptr; // DON'T compress velocity on shear cells
    static const double XREL = std::getenv("TMLPU_XREL") ? std::atof(std::getenv("TMLPU_XREL")) : 0.0; // extremum-relax velocity bound at smooth vortex (grow coherent rolls)
    static const bool XCAP1 = std::getenv("TMLPU_XREL_CAP1") != nullptr; // ψ<=1 pure un-clipping (MLP-u2, no anti-diffusion) vs ψ>1
    static const double TVBM = std::getenv("TMLPU_TVB") ? std::atof(std::getenv("TMLPU_TVB")) : 0.0; // resolution-consistent (M·h²) un-clip = bulletproof replacement for XREL bound-relax
    static const double VCOMP = std::getenv("TMLPU_VCOMP") ? std::atof(std::getenv("TMLPU_VCOMP")) : 0.0; // ψ∈[0,2] compress shear-layer velocity (bounded by LMP, monotone) -> thin shear -> KH grows
    // ===== NCC (Normal-Column Compression): skew-base B + non-orthogonal-purified normal jump J_n,
    // blended by β=θ+χ(1-θ), β bounded by TMLP-u (face value within neighbour min/max). TMLPU_NCC. =====
    static const bool NCC = std::getenv("TMLPU_NCC") != nullptr;
    static const double NCC_LAMT = std::getenv("NCC_LAMT") ? std::atof(std::getenv("NCC_LAMT")) : 1.0; // tangential skew-correction strength
    static const double NCC_LAMN = std::getenv("NCC_LAMN") ? std::atof(std::getenv("NCC_LAMN")) : 1.0; // non-orthogonal jump-purification strength
    static const double NCC_CHI  = std::getenv("NCC_CHI")  ? std::atof(std::getenv("NCC_CHI"))  : 1.0; // compression intent at sharp cells (0..1)
    static const double NCC_PJ   = std::getenv("NCC_PJ")   ? std::atof(std::getenv("NCC_PJ"))   : 0.1; // pressure-jump veto: compress only where p-jump < NCC_PJ (contacts, NOT shocks)
    static const bool UFORM = std::getenv("TMLPU_UFORM") != nullptr; // user formula: φ_L+ψ[0.5(φ_R−φ_L)+∇φ_L·(m_f−m_LR)] (central-jump normal + one-sided skew)
    static const bool GEN = std::getenv("TMLPU_GEN") != nullptr; // full genuine T-MLP-u: MLP α-limiter (vertex-bound) + ∇φ_fcorr=½(∇φ_L+∇φ_R)
    static const bool GENMS = std::getenv("TMLPU_GEN_MSTACS") != nullptr; // GEN ψ(r): default van_leer, mstacs if set
    static const bool GENPF = std::getenv("TMLPU_GEN_POSFLOOR") != nullptr; // GEN: enable rho,p positivity clamp (off=clean)
    static const char* GENVLE = std::getenv("TMLPU_GEN_VL");   // GEN ψ(r)=van_leer (force, overrides mstacs)
    static const bool GENDUC = std::getenv("TMLPU_GEN_DUCROS") != nullptr; // GEN: Ducros shock veto on ceiling (shock->psi<=1)
    static const bool GENPSAFE = std::getenv("TMLPU_GEN_PSAFE") != nullptr; // GEN: pressure acoustic safety (p ceiling<=1)
    static const bool GENPOS = std::getenv("TMLPU_GEN_POS") != nullptr;    // GEN: joint-theta positivity limiter (post-pass)
    static const bool GENCG = std::getenv("TMLPU_GEN_CGATE") != nullptr;   // GEN: contact-gated ceiling van_leer<->mstacs by samt (sharp contact only)
    static const bool GENCAP1 = std::getenv("TMLPU_GEN_CAP1") != nullptr;  // GEN diagnostic: cap ψ≤1 (test if ψ>1 anti-diffusion is the divergence cause)
    static const double GENPOSF = std::getenv("TMLPU_GEN_POSF") ? std::atof(std::getenv("TMLPU_GEN_POSF")) : 0.2; // pos floor frac of cell rho,p
    static const bool VCOMP_CONTACT_ONLY = std::getenv("TMLPU_VCOMP_CONTACT_ONLY") != nullptr; // reviewer-safe: compress velocity only on pressure-continuous density contacts
    static const bool VCOMP_VSHARP = std::getenv("TMLPU_VCOMP_VSHARP") != nullptr; // gate VCOMP by max(density-contact, velocity-shear-sharpness): compress sharp shears even w/o density jump, NOT smooth vortices
    static const double VCOMP_MIN_CONTACT = std::getenv("TMLPU_VCOMP_MIN_CONTACT")
        ? std::max(0.0, std::min(1.0, std::atof(std::getenv("TMLPU_VCOMP_MIN_CONTACT")))) : 0.0;
    // ===== T-MLP-u-C+ : physics-decomposed unified policy. Separated gates (shock /
    // contact / vortex), component-wise reconstruction (rho,p,u_n shock-capped; u_t
    // vortex-preserved), dimensionless sensors, psi=min(bound,tvd,shock_cap). TMLPU_CP. =====
    // T-MLP-u-DW (aggressive 2-tier): sharp cells -> PURE DOWNWIND (psi=2, max compression),
    // smooth cells -> van Leer. Gate = the binary `sharp` flag (residual+gap, +veto envs).
    // LMP wrapper psi=min(pt,pl) keeps it monotone/bounded. TMLPU_DW.
    static const bool DW = std::getenv("TMLPU_DW") != nullptr;
    static const bool CP = std::getenv("TMLPU_CP") != nullptr;
    static const bool CP_NT = std::getenv("TMLPU_CP_NT") != nullptr;
    static const bool CP_TCONTACT_ONLY = std::getenv("TMLPU_CP_TCONTACT_ONLY") != nullptr;
    static const bool CP_PROJ_LMP = std::getenv("TMLPU_CP_PROJ_LMP") != nullptr;
    static const bool CP_RHO_DW = std::getenv("TMLPU_CP_RHO_DW") != nullptr;
    static const bool CP_T_DW = std::getenv("TMLPU_CP_T_DW") != nullptr;
    static const bool CP_T_NOCAP = std::getenv("TMLPU_CP_T_NOCAP") != nullptr;
    static const char* CPCAP = std::getenv("TMLPU_CP_CAP");
    const double cap_str = CPCAP ? std::atof(CPCAP) : 0.5;   // shock psi cap: psi<=1-shock*cap_str
    static const bool CP_VSHARP = std::getenv("TMLPU_CP_VSHARP") != nullptr; // velocity compressive (MSTACS) not 1.0
    static const double CP_Q0 = std::getenv("TMLPU_CP_Q0") ? std::atof(std::getenv("TMLPU_CP_Q0")) : 0.0;
    static const double CP_Q1 = std::getenv("TMLPU_CP_Q1") ? std::atof(std::getenv("TMLPU_CP_Q1")) : 0.40;
    static const double CP_M0 = std::getenv("TMLPU_CP_M0") ? std::atof(std::getenv("TMLPU_CP_M0")) : 0.25;
    static const double CP_M1 = std::getenv("TMLPU_CP_M1") ? std::atof(std::getenv("TMLPU_CP_M1")) : 0.55;
    static const double CP_TSHARP = std::getenv("TMLPU_CP_TSHARP")
        ? std::max(0.0, std::min(1.0, std::atof(std::getenv("TMLPU_CP_TSHARP")))) : 1.0;
    static const double CP_VCONTACT_GAIN = std::getenv("TMLPU_CP_VCONTACT_GAIN")
        ? std::max(0.0, std::atof(std::getenv("TMLPU_CP_VCONTACT_GAIN"))) : 1.0;
    std::vector<double> cp_shock, cp_contact, cp_vortex;
    std::vector<double> samt, c2rot, vsh;
    if ((C2 || (GEN && GENCG)) && nvar >= 4) {
        samt.assign((size_t)N, 0.0); c2rot.assign((size_t)N, 0.0); vsh.assign((size_t)N, 0.0);
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double rc = W[(size_t)0*N+ci], pc = W[(size_t)(nvar-1)*N+ci];
            double gx=grad[((size_t)0*N+ci)*2+0], gy=grad[((size_t)0*N+ci)*2+1];
            // density bimodality gap normalized by local stencil range (dimensionless)
            double vals[64]; int nv=0; vals[nv++]=rc; double pjump=0.0;
            for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                if(nv<64) vals[nv++]=W[(size_t)0*N+nb];
                double pj=std::fabs(W[(size_t)(nvar-1)*N+nb]-pc); if(pj>pjump)pjump=pj; }
            for (int a=1;a<nv;++a){ double t=vals[a]; int b=a-1; while(b>=0&&vals[b]>t){vals[b+1]=vals[b];--b;} vals[b+1]=t; }
            double rng=vals[nv-1]-vals[0], maxgap=0.0;
            for (int a=1;a<nv;++a){ double g=vals[a]-vals[a-1]; if(g>maxgap)maxgap=g; }
            double G_hat = rng>1e-30 ? maxgap/rng : 0.0;
            // Ducros compression dominance theta^2/(theta^2+omega^2) (dimensionless ratio)
            double dudx=grad[((size_t)1*N+ci)*2+0], dudy=grad[((size_t)1*N+ci)*2+1];
            double dvdx=grad[((size_t)2*N+ci)*2+0], dvdy=grad[((size_t)2*N+ci)*2+1];
            double theta=dudx+dvdy, omega=dvdx-dudy;
            double D_comp=(theta*theta)/(theta*theta + omega*omega + 1e-30);
            // pressure jump relative to local pressure (dimensionless). slip-line/contact ~0,
            // strong shock >> 0 -> separates a rolling vortex's weak compression from a true shock.
            double P_hat = pjump/(std::fabs(pc)+1e-30);
            // separated continuous gates
            double shock_gate   = smoothstep(0.45,0.75,D_comp) * smoothstep(0.15,0.80,P_hat);
            double contact_gate = smoothstep(0.25,0.55,G_hat) * (1.0 - shock_gate);
            samt[ci] = contact_gate;   // sharpen contacts/slip-lines, NOT shocks (continuous)
            c2rot[ci] = (1.0 - D_comp) * (1.0 - shock_gate);  // rotation/shear dominance, non-shock
            // velocity-sharpness gate: bimodality gap of each velocity COMPONENT (u,v), take max.
            // A counter-streaming shear (u=±U) is BIMODAL in u even though speed |u|=U is uniform;
            // a slip-line has a bimodal tangential component. A SMOOTH vortex is unimodal in u and v.
            // -> ~1 at a sharp velocity shear/slip-line (compress), ~0 at a smooth vortex or shock,
            // even when density is uniform (no contact). Fixes the isentropic-vs-shear ambiguity.
            double Gv_hat = 0.0;
            for (int comp=1; comp<=2; ++comp) {
                double cvals[64]; int ncv=0; cvals[ncv++]=W[(size_t)comp*N+ci];
                for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                    if(ncv<64) cvals[ncv++]=W[(size_t)comp*N+nb]; }
                for (int a=1;a<ncv;++a){ double t=cvals[a]; int b=a-1; while(b>=0&&cvals[b]>t){cvals[b+1]=cvals[b];--b;} cvals[b+1]=t; }
                double crng=cvals[ncv-1]-cvals[0], cmaxgap=0.0;
                for (int a=1;a<ncv;++a){ double g=cvals[a]-cvals[a-1]; if(g>cmaxgap)cmaxgap=g; }
                double gh = crng>1e-30 ? cmaxgap/crng : 0.0; if(gh>Gv_hat)Gv_hat=gh; }
            vsh[ci] = smoothstep(0.25,0.55,Gv_hat) * (1.0 - shock_gate);  // sharp velocity shear, non-shock
            (void)rc;
        }
    }
    if (CP && nvar >= 4) {
        cp_shock.assign((size_t)N,0.0); cp_contact.assign((size_t)N,0.0); cp_vortex.assign((size_t)N,0.0);
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double rc=W[(size_t)0*N+ci], pc=W[(size_t)(nvar-1)*N+ci];
            // density gap + class-balance + max relative pressure jump (all dimensionless)
            double vals[64]; int nv=0; vals[nv++]=rc; double pjr=0.0;
            for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                if(nv<64) vals[nv++]=W[(size_t)0*N+nb];
                double pj=std::fabs(W[(size_t)(nvar-1)*N+nb]-pc)/(std::fabs(W[(size_t)(nvar-1)*N+nb])+std::fabs(pc)+1e-30);
                if(pj>pjr)pjr=pj; }
            for (int a=1;a<nv;++a){ double t=vals[a]; int b=a-1; while(b>=0&&vals[b]>t){vals[b+1]=vals[b];--b;} vals[b+1]=t; }
            double rng=vals[nv-1]-vals[0], maxgap=0.0; int ks=0;
            for (int a=1;a<nv;++a){ double g=vals[a]-vals[a-1]; if(g>maxgap){maxgap=g;ks=a;} }
            double G=rng>1e-30?maxgap/rng:0.0;
            double nL=ks, nR=nv-ks, Bbal=4.0*nL*nR/std::max((double)(nv*nv),1.0);
            double M=G*Bbal;                                    // bimodality with class balance
            // velocity gradients -> Ducros (compression-only) + Q-criterion (rotation vs strain)
            double dudx=grad[((size_t)1*N+ci)*2+0], dudy=grad[((size_t)1*N+ci)*2+1];
            double dvdx=grad[((size_t)2*N+ci)*2+0], dvdy=grad[((size_t)2*N+ci)*2+1];
            double theta=dudx+dvdy, omega=dvdx-dudy, nt=theta<0.0?-theta:0.0;
            double epsD=1e-30 + 1e-6*(theta*theta+omega*omega);
            double D_comp=(nt*nt)/(nt*nt + omega*omega + epsD);
            double Smag=2.0*dudx*dudx+2.0*dvdy*dvdy+(dudy+dvdx)*(dudy+dvdx), Wmag=omega*omega;
            double Qhat=(Wmag-Smag)/(Wmag+Smag+1e-30);          // >0 rotation/vortex dominant
            double P_hat=pjr;
            double shock = smoothstep(0.15,0.80,P_hat) * smoothstep(0.45,0.75,D_comp);
            cp_shock[ci]   = shock;
            cp_contact[ci] = (1.0-shock) * smoothstep(CP_M0,CP_M1,M) * (1.0 - smoothstep(0.15,0.80,P_hat));
            cp_vortex[ci]  = (1.0-shock) * smoothstep(CP_Q0,CP_Q1,Qhat);   // preserve tangential vel in vortex
        }
    }
    #pragma omp parallel for
    for (int vn=0; vn<c.Nn; ++vn) for (int v=0;v<nvar;++v){ double mn=1e300,mx=-1e300;
        for(int k=0;k<c.max_v2c;++k){ int ci=c.v2c[(size_t)vn*c.max_v2c+k]; if(ci<0)continue;
            double val=W[(size_t)v*N+ci]; if(val<mn)mn=val; if(val>mx)mx=val; }
        if(mn>mx){mn=0;mx=0;} vmn[(size_t)v*c.Nn+vn]=mn; vmx[(size_t)v*c.Nn+vn]=mx; }
    W_L.assign((size_t)nvar*Nf,0.0); W_R.assign((size_t)nvar*Nf,0.0);
    const double eps=1e-30;
    // per-cell uncapped LMP psi (room-to-vertex-bound / vertex projection), capped at psi_cap.
    auto psi_lmp = [&](int ci, int v, double gx, double gy) -> double {
        double wc=W[(size_t)v*N+ci], p=psi_cap;
        for(int k=0;k<c.max_v;++k){ int vn=c.sample_vid[(size_t)ci*c.max_v+k]; if(vn<0)continue;
            double dx=c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy=c.sample_off[((size_t)ci*c.max_v+k)*2+1];
            double proj=gx*dx+gy*dy;
            double allowed = proj>=0.0 ? (vmx[(size_t)v*c.Nn+vn]-wc) : (wc-vmn[(size_t)v*c.Nn+vn]);
            double pk = (std::fabs(proj)>eps) ? std::max(allowed,0.0)/std::max(std::fabs(proj),eps) : psi_cap;
            if(pk<p)p=pk; }
        return p<0?0:(p>psi_cap?psi_cap:p);
    };
    // TVB-consistent (resolution-consistent) un-clip for velocity: widen vertex bound by
    // r = M*h^2 (cell_vol ~ h^2/2) so a SMOOTH extremum (curvature <= M) is NOT clipped,
    // while a discontinuity (jump >> M h^2) stays limited. r->0 as h->0 => consistent
    // (Shu TVB / MLP-u2). Bulletproof replacement for the fixed-fraction XREL bound relax.
    auto psi_lmp_tvb = [&](int ci, int v, double gx, double gy, double Mtvb) -> double {
        double wc=W[(size_t)v*N+ci], p=psi_cap, r=Mtvb*2.0*m.cell_volumes[ci];
        for(int k=0;k<c.max_v;++k){ int vn=c.sample_vid[(size_t)ci*c.max_v+k]; if(vn<0)continue;
            double dx=c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy=c.sample_off[((size_t)ci*c.max_v+k)*2+1];
            double proj=gx*dx+gy*dy;
            double allowed = proj>=0.0 ? (vmx[(size_t)v*c.Nn+vn]+r-wc) : (wc-(vmn[(size_t)v*c.Nn+vn]-r));
            double pk = (std::fabs(proj)>eps) ? std::max(allowed,0.0)/std::max(std::fabs(proj),eps) : psi_cap;
            if(pk<p)p=pk; }
        return p<0?0:(p>psi_cap?psi_cap:p);
    };
    // Ducros shock sensor (kinematic: dilatation vs vorticity). D~1 at compressing shock, ~0 at shear/smooth.
    std::vector<double> gduc;
    if (GEN && GENDUC && nvar>=4) { gduc.assign(N,0.0);
        #pragma omp parallel for
        for (int ci=0; ci<N; ++ci){ double dudx=grad[((size_t)1*N+ci)*2+0],dudy=grad[((size_t)1*N+ci)*2+1];
            double dvdx=grad[((size_t)2*N+ci)*2+0],dvdy=grad[((size_t)2*N+ci)*2+1];
            double th=dudx+dvdy, om=dvdx-dudy, comp=th<0?-th:0.0;          // compression-gated dilatation
            gduc[ci]=comp*comp/(comp*comp+om*om+1e-30); } }                // Ducros: shock=1, shear=0
    // ===== T-MLP-u-W (WMU): characteristic-wave-aware one-sided MLP-u. Per face,
    // Roe-average -> project the cell-gradient face increment into the 4 Euler waves
    // (acoustic-, entropy/contact, shear, acoustic+) via the primitive eigenstructure,
    // apply a wave-FAMILY TVD limiter on the characteristic far-upwind ratio r_w
    // [acoustic = van_leer (dissipative, TVD shock capture), entropy = mstacs
    // (compressive, sharpen contacts), shear = central-biased (preserve KH/vortices)],
    // recompose, then clamp EACH physical variable to the one-sided MLP-u vertex bound
    // (psi <= psi_LMP, same sign) -> monotone, NO anti-diffusion beyond the vertex bound.
    // Combines WAU's exact wave separation with T-MLP-u's linear LMP limiter. A strong
    // shock has small entropy amplitude (dr ~ dp/c2) so compression auto-spares shocks
    // and targets true contacts. nvar==4 (Euler) only. Env: TMLPU_WAVE. =====
    static const bool WAVE = std::getenv("TMLPU_WAVE") != nullptr;
    if (WAVE && nvar == 4) {
        static const double Wg  = std::getenv("WAVE_GAMMA") ? std::atof(std::getenv("WAVE_GAMMA")) : 1.4;
        static const double Wen = std::getenv("WAVE_ENT")   ? std::atof(std::getenv("WAVE_ENT"))   : 0.6; // entropy compression [0,1]
        static const double Wsh = std::getenv("WAVE_SHEAR") ? std::atof(std::getenv("WAVE_SHEAR")) : 0.5; // shear central bias [0,1]
        const int RI=0, UI=1, VI=2, PI=nvar-1;
        #pragma omp parallel for
        for (int f=0; f<Nf; ++f){
            int o=m.face_owner[f], n=m.face_neighbour[f];
            double fx=m.face_centers[f*2+0], fy=m.face_centers[f*2+1];
            if (n<0){ for(int v=0;v<nvar;++v){ double wo=W[(size_t)v*N+o]; W_L[(size_t)v*Nf+f]=wo; W_R[(size_t)v*Nf+f]=wo; } continue; }
            double nx=m.face_normals[f*2+0], ny=m.face_normals[f*2+1];
            double nl=std::sqrt(nx*nx+ny*ny); if(nl>eps){nx/=nl;ny/=nl;}
            double tx=-ny, ty=nx;
            double ro=W[(size_t)RI*N+o], rn=W[(size_t)RI*N+n];
            double uo=W[(size_t)UI*N+o], uN=W[(size_t)UI*N+n];
            double vo=W[(size_t)VI*N+o], vN=W[(size_t)VI*N+n];
            double po=W[(size_t)PI*N+o], pn=W[(size_t)PI*N+n];
            double sro=std::sqrt(std::max(ro,1e-30)), srn=std::sqrt(std::max(rn,1e-30)), iss=1.0/(sro+srn);
            double uR=(sro*uo+srn*uN)*iss, vR=(sro*vo+srn*vN)*iss;
            double Ho=(Wg/(Wg-1.0))*po/std::max(ro,1e-30)+0.5*(uo*uo+vo*vo);
            double Hn=(Wg/(Wg-1.0))*pn/std::max(rn,1e-30)+0.5*(uN*uN+vN*vN);
            double HR=(sro*Ho+srn*Hn)*iss, q2=uR*uR+vR*vR;
            double c2=std::max((Wg-1.0)*(HR-0.5*q2),1e-12), cR=std::sqrt(c2);
            double rhoR=std::sqrt(std::max(ro*rn,1e-30));
            auto decomp=[&](double dr,double du,double dv,double dp,double a[4]){
                double dun=du*nx+dv*ny, dut=du*tx+dv*ty;
                a[0]=(dp-rhoR*cR*dun)/(2.0*c2); a[1]=dr-dp/c2; a[2]=dut; a[3]=(dp+rhoR*cR*dun)/(2.0*c2);
            };
            auto recompose=[&](const double a[4],double& dr,double& du,double& dv,double& dp){
                dr=a[0]+a[1]+a[3]; dp=c2*(a[0]+a[3]);
                double dun=(cR/rhoR)*(a[3]-a[0]), dut=a[2];
                du=dun*nx+dut*tx; dv=dun*ny+dut*ty;
            };
            auto fam=[&](double r,int w)->double{ double vl=psi_van_leer(r);
                if(w==1) return (1.0-Wen)*vl+Wen*psi_mstacs(r,Co);   // entropy: compress contacts
                if(w==2) return (1.0-Wsh)*vl+Wsh*1.0;               // shear: central (preserve KH)
                return vl; };                                       // acoustic: van_leer (TVD)
            double drx=cc[n*2+0]-cc[o*2+0], dry=cc[n*2+1]-cc[o*2+1];   // d_LR
            auto side=[&](int donor,double sgn,std::vector<double>& Wout){
                double gx[4],gy[4],wc[4];
                for(int v=0;v<nvar;++v){ wc[v]=W[(size_t)v*N+donor];
                    gx[v]=grad[((size_t)v*N+donor)*2+0]; gy[v]=grad[((size_t)v*N+donor)*2+1]; }
                double dxLR=sgn*drx, dyLR=sgn*dry;                       // d_LR (L) or d_RL (R)
                double a_up[4],a_dn[4],a_fc[4];
                decomp(gx[RI]*dxLR+gy[RI]*dyLR, gx[UI]*dxLR+gy[UI]*dyLR, gx[VI]*dxLR+gy[VI]*dyLR, gx[PI]*dxLR+gy[PI]*dyLR, a_up);
                decomp(sgn*(rn-ro), sgn*(uN-uo), sgn*(vN-vo), sgn*(pn-po), a_dn);   // downwind jump
                double dfx=fx-cc[donor*2+0], dfy=fy-cc[donor*2+1];
                decomp(gx[RI]*dfx+gy[RI]*dfy, gx[UI]*dfx+gy[UI]*dfy, gx[VI]*dfx+gy[VI]*dfy, gx[PI]*dfx+gy[PI]*dfy, a_fc);
                double a_lim[4];
                for(int w=0;w<4;++w){ double r=(std::fabs(a_dn[w])>eps)?a_up[w]/a_dn[w]:(a_up[w]*a_dn[w]>=0?1e30:-1e30);
                    a_lim[w]=fam(r,w)*a_fc[w]; }
                double dr,du,dv,dp; recompose(a_lim,dr,du,dv,dp);
                double dW[4]; dW[RI]=dr; dW[UI]=du; dW[VI]=dv; dW[PI]=dp;
                for(int v=0;v<nvar;++v){
                    double incu=gx[v]*dfx+gy[v]*dfy;                    // unlimited face increment
                    double mx=psi_lmp(donor,v,gx[v],gy[v])*incu;        // signed LMP-admissible increment
                    double d=dW[v];
                    if(incu>=0.0){ if(d<0.0)d=0.0; else if(d>mx)d=mx; }
                    else         { if(d>0.0)d=0.0; else if(d<mx)d=mx; }
                    Wout[(size_t)v*Nf+f]=wc[v]+d;
                }
            };
            side(o,+1.0,W_L);    // L: donor=owner, d_LR
            side(n,-1.0,W_R);    // R: donor=neighbour, d_RL = -d_LR, downwind jump = W_o-W_n
            (void)fx;(void)fy;
        }
        return;
    }
    // TMLPU_TBV (2026-07-04, user): replace the residual+gap sharp gate with a BOUNDARY-VARIATION
    // (TBV / BVD) sensor. Per cell, sum over faces |W_L - W_R| for the CICSAM candidate vs the
    // van-Leer candidate (both LMP-bounded, coherent 2-state), flag SHARP where CICSAM LOWERS the
    // total boundary variation by more than a margin (= a true interface being sharpened; a smooth
    // region should NOT prefer compression). Unifies the zero-BV/TBV theme with the coherent T-MLP-u.
    static const bool TBVGATE = std::getenv("TMLPU_TBV") != nullptr;
    static const double TBVMARG = std::getenv("TMLPU_TBV_MARGIN") ? std::atof(std::getenv("TMLPU_TBV_MARGIN")) : 0.0;
    if (TBVGATE) {
        std::vector<double> tvl((size_t)nvar*N, 0.0), tcic((size_t)nvar*N, 0.0);
        for (int f=0; f<Nf; ++f){ int o=m.face_owner[f], nb=m.face_neighbour[f]; if(nb<0) continue;
            double fx=m.face_centers[f*2+0], fy=m.face_centers[f*2+1];
            for (int v=0; v<nvar; ++v){
                double wo=W[(size_t)v*N+o], wn=W[(size_t)v*N+nb];
                double go0=grad[((size_t)v*N+o)*2+0], go1=grad[((size_t)v*N+o)*2+1];
                double gn0=grad[((size_t)v*N+nb)*2+0], gn1=grad[((size_t)v*N+nb)*2+1];
                double inco=go0*(fx-cc[o*2+0])+go1*(fy-cc[o*2+1]);
                double incn=gn0*(fx-cc[nb*2+0])+gn1*(fy-cc[nb*2+1]);
                double upo=go0*(cc[nb*2+0]-cc[o*2+0])+go1*(cc[nb*2+1]-cc[o*2+1]);
                double upn=gn0*(cc[o*2+0]-cc[nb*2+0])+gn1*(cc[o*2+1]-cc[nb*2+1]);
                double ro=(std::fabs(upo)>eps)?(wn-wo)/upo:((wn-wo)*upo>=0?1e30:-1e30);
                double rn=(std::fabs(upn)>eps)?(wo-wn)/upn:((wo-wn)*upn>=0?1e30:-1e30);
                double plo=psi_lmp(o,v,go0,go1), plnn=psi_lmp(nb,v,gn0,gn1);
                double qLvl =wo+std::min(psi_van_leer(ro),plo)*inco,  qLcic=wo+std::min(psi_cicsam(ro,Co),plo)*inco;
                double qRvl =wn+std::min(psi_van_leer(rn),plnn)*incn, qRcic=wn+std::min(psi_cicsam(rn,Co),plnn)*incn;
                double bvvl=std::fabs(qLvl-qRvl), bvcic=std::fabs(qLcic-qRcic);
                tvl[(size_t)v*N+o]+=bvvl; tvl[(size_t)v*N+nb]+=bvvl;
                tcic[(size_t)v*N+o]+=bvcic; tcic[(size_t)v*N+nb]+=bvcic;
            }
        }
        for (int i=0;i<N;++i) for (int v=0;v<nvar;++v)
            sharp[(size_t)v*N+i] = (tcic[(size_t)v*N+i] < tvl[(size_t)v*N+i]-TBVMARG) ? 1 : 0;
    }
    std::vector<signed char> fflag; if(FLAG) fflag.assign((size_t)Nf,-1);   // per-face density branch (v==0); -1 = branch not taken
    #pragma omp parallel for
    for (int f=0; f<Nf; ++f){ int o=m.face_owner[f], n=m.face_neighbour[f];
        double fx=m.face_centers[f*2+0], fy=m.face_centers[f*2+1];
        for (int v=0;v<nvar;++v){
            double wo=W[(size_t)v*N+o], go0=grad[((size_t)v*N+o)*2+0], go1=grad[((size_t)v*N+o)*2+1];
            // TMLPU_TBV_ZEROBV (2026-07-04, user): ASVL zero-BV pick on the [vanLeer,CICSAM]-LMP band,
            // on the COHERENT T-MLP-u base. Per side the admissible face value spans [vanLeer, CICSAM]
            // (both LMP-bounded). Where the two W-ranges OVERLAP, a zero-BV shared value q* (W_L=W_R)
            // exists (a whole set = "multiple psi*"); take the MOST-COMPRESSIVE one (overlap endpoint
            // furthest from the cell midpoint). No overlap -> fallback = discrete min-TBV (vanLeer or
            // CICSAM combo with smaller |W_L-W_R|). Bounded to the band -> milder than pure ASVL.
            static const bool ZEROBV = std::getenv("TMLPU_TBV_ZEROBV") != nullptr;
            // TMLPU_ZBV_GATE: apply the zero-BV pick ONLY at gap-gate SHARP cells (true steps); leave
            // SMOOTH cells (cone/hump) to the coherent T-MLP-u van-Leer path -> stops the smooth-body
            // over-compression the ungated zero-BV causes, while keeping the sharp-slot zero-BV win.
            static const bool ZBVGATE = std::getenv("TMLPU_ZBV_GATE") != nullptr;
            if (ZEROBV && n>=0 && (!ZBVGATE || sharp[(size_t)v*N+o] || sharp[(size_t)v*N+n])) {
                double wn=W[(size_t)v*N+n], gn0=grad[((size_t)v*N+n)*2+0], gn1=grad[((size_t)v*N+n)*2+1];
                double inco=go0*(fx-cc[o*2+0])+go1*(fy-cc[o*2+1]);
                double incn=gn0*(fx-cc[n*2+0])+gn1*(fy-cc[n*2+1]);
                double upo=go0*(cc[n*2+0]-cc[o*2+0])+go1*(cc[n*2+1]-cc[o*2+1]);
                double upn=gn0*(cc[o*2+0]-cc[n*2+0])+gn1*(cc[o*2+1]-cc[n*2+1]);
                double ro=(std::fabs(upo)>eps)?(wn-wo)/upo:((wn-wo)*upo>=0?1e30:-1e30);
                double rn=(std::fabs(upn)>eps)?(wo-wn)/upn:((wo-wn)*upn>=0?1e30:-1e30);
                double plo=psi_lmp(o,v,go0,go1), pln=psi_lmp(n,v,gn0,gn1);
                // TMLPU_ZBV_MINMOD (2026-07-04, user): use MINMOD (max(0,min(r,1)), <=1, no compression =
                // most diffusive) as the LOWER band endpoint instead of van Leer -> narrower/gentler band on
                // smooth flow -> less over-compression of the vortices (cone/hump wiggle) in the ungated zero-BV.
                static const bool ZBVMM = std::getenv("TMLPU_ZBV_MINMOD") != nullptr;
                auto zlo = [&](double r){ return ZBVMM ? std::max(0.0,std::min(r,1.0)) : psi_van_leer(r); };
                // TMLPU_ZBV_SUPERBEE (2026-07-04, user): use SUPERBEE (max(0,min(2r,1),min(r,2)), pure r,
                // NO Courant Co) as the compressive UPPER band endpoint instead of CICSAM(Co) -> band is
                // Co-free / parameter-free. With MIN psi* the upper endpoint matters little (pick is diffuse).
                static const bool ZBVSB = std::getenv("TMLPU_ZBV_SUPERBEE") != nullptr;
                static const bool ZBVDW = std::getenv("TMLPU_ZBV_DOWNWIND") != nullptr;  // psi=2 const (r>0) = most aggressive Co-free upper bound
                auto zhi = [&](double r){ return ZBVDW ? (r>0.0?2.0:0.0)
                                               : (ZBVSB ? std::max(0.0,std::max(std::min(2.0*r,1.0),std::min(r,2.0))) : psi_cicsam(r,Co)); };
                double WLvl=wo+std::min(zlo(ro),plo)*inco, WLcic=wo+std::min(zhi(ro),plo)*inco;
                double WRvl=wn+std::min(zlo(rn),pln)*incn, WRcic=wn+std::min(zhi(rn),pln)*incn;
                double WLlo=std::min(WLvl,WLcic), WLhi=std::max(WLvl,WLcic);
                double WRlo=std::min(WRvl,WRcic), WRhi=std::max(WRvl,WRcic);
                double ovlo=std::max(WLlo,WRlo), ovhi=std::min(WLhi,WRhi);
                // TMLPU_ZBV_BVDONLY (2026-07-04, user): SKIP the zero-BV psi* pick entirely; always use the
                // discrete BVD selection (whole vanLeer-recon vs whole downwind-recon, per cell, lower |W_L-W_R|).
                // Isolates what the zero-BV shared-value derivation contributes vs plain vanLeer/downwind BVD.
                static const bool BVDONLY = std::getenv("TMLPU_ZBV_BVDONLY") != nullptr;
                double WLo=0.0, WRo=0.0;
                if (ovlo <= ovhi && !BVDONLY) {                      // zero-BV achievable in the band
                    // TMLPU_ZBV_PICK (2026-07-04, user): which zero-BV shared value (the set = "multiple psi*").
                    // 0=MAX psi* (most compressive, furthest from midpoint) - sharpest slot but over-compresses
                    //   smooth vortices (overlap exists on smooth too -> spurious compression -> wiggle);
                    // 1=MIN psi* (LEAST compressive = clamp midpoint into overlap) - smooth -> ~central=coherent
                    //   (no over-compression, clean vortex) but softer slot; 2=overlap midpoint.
                    static const int ZBVPICK = []{ const char* e=std::getenv("TMLPU_ZBV_PICK"); return (e&&e[0])?std::atoi(e):0; }();
                    double qmid=0.5*(wo+wn), qs;
                    if (ZBVPICK==1)      qs = (qmid<ovlo?ovlo:(qmid>ovhi?ovhi:qmid));          // MIN psi* (cell-midpoint ref = central psi~1)
                    else if (ZBVPICK==2) qs = 0.5*(ovlo+ovhi);                                 // overlap midpoint
                    else if (ZBVPICK==3) { double vref=0.5*(WLvl+WRvl);                        // MIN psi* (vanLeer coherent ref = true least-compressive)
                                           qs = (vref<ovlo?ovlo:(vref>ovhi?ovhi:vref)); }
                    else                 qs = (std::fabs(ovhi-qmid)>=std::fabs(ovlo-qmid))?ovhi:ovlo; // MAX psi* (compressive)
                    WLo=qs; WRo=qs;
                    if(FLAG&&v==0) fflag[f]=0;   // branch 0: psi* zero-BV shared value
                } else {                                             // fallback: discrete min-TBV
                    double bvvl=std::fabs(WLvl-WRvl), bvcic=std::fabs(WLcic-WRcic);
                    if (bvcic<bvvl){ WLo=WLcic; WRo=WRcic; } else { WLo=WLvl; WRo=WRvl; }
                    if(FLAG&&v==0) fflag[f]=(bvcic<bvvl)?2:1;   // branch 2: downwind combo; branch 1: vanLeer combo
                }
                // Euler positivity: floor rho (v=0) & p (v=nvar-1) to posfloor*min(cell) (safety; the LMP
                // wrapper already bounds to positive neighbour averages). Component-wise, NO wave decomposition.
                if (nvar>=4 && (v==0||v==nvar-1)) { double fl=posfloor*std::min(wo,wn);
                    if(WLo<fl)WLo=fl; if(WRo<fl)WRo=fl; }
                W_L[(size_t)v*Nf+f]=WLo; W_R[(size_t)v*Nf+f]=WRo;
                continue;
            }
            if (GEN) {   // genuine T-MLP-u: φ_f = φ_donor + ψ[ 0.5Δ + ∇φ_fcorr·(d_df − 0.5 d_LR) ], ψ = min(ψ(r), LMP-cap bound/Δφ_V)
                if (n<0) { W_L[(size_t)v*Nf+f]=wo; W_R[(size_t)v*Nf+f]=wo; continue; }
                double wn=W[(size_t)v*N+n], gn0=grad[((size_t)v*N+n)*2+0], gn1=grad[((size_t)v*N+n)*2+1];
                double clx=cc[o*2+0],cly=cc[o*2+1], crx=cc[n*2+0],cry=cc[n*2+1];
                double rx=crx-clx, ry=cry-cly;                          // d_LR
                double gfx=0.5*(go0+gn0), gfy=0.5*(go1+gn1);            // ∇φ_fcorr (averaged)
                double djump=wn-wo;                                     // φ_R − φ_L
                // ----- L side (donor=L) -----
                double upL=go0*rx+go1*ry;                              // ∇φ_L·d_LR = φ_L − φ_LL
                double rL=upL/(djump+(djump>=0?eps:-eps));             // (φ_L−φ_LL)/(φ_R−φ_L+ε)
                // ψ(r): fixed geometric limiter (smooth→1, sharp→2). LMP vertex bound = monotone cap.
                // Contact-gated: blend van_leer(smooth-faithful)↔mstacs(compressive) by samt (=1 only at sharp contacts/slip-lines).
                double psiL;
                if (GENCG && nvar>=4) { double s=0.5*(samt[o]+samt[n]); psiL=(1.0-s)*psi_van_leer(rL)+s*psi_mstacs(rL,Co); }
                else psiL = GENVLE ? psi_van_leer(rL) : (GENMS ? psi_mstacs(rL,Co) : psi_van_leer(rL));
                if (GENDUC && nvar>=4) { double D=0.5*(gduc[o]+gduc[n]); psiL=(1.0-D)*psiL + D*std::min(1.0,psi_van_leer(rL)); } // shock veto: ψ≤1
                if (GENPSAFE && v==nvar-1 && psiL>1.0) psiL=1.0;          // pressure acoustic safety: no downwind compression
                for(int k=0;k<c.max_v;++k){ int vn=c.sample_vid[(size_t)o*c.max_v+k]; if(vn<0)continue;
                    double dx=c.sample_off[((size_t)o*c.max_v+k)*2+0], dy=c.sample_off[((size_t)o*c.max_v+k)*2+1];
                    double dphiV=0.5*djump + gfx*(dx-0.5*rx)+gfy*(dy-0.5*ry);
                    double bound=(dphiV>0)?(vmx[(size_t)v*c.Nn+vn]-wo):(vmn[(size_t)v*c.Nn+vn]-wo);
                    double capV=bound/(std::fabs(dphiV)>eps?dphiV:(dphiV>=0?eps:-eps)); // LMP: max ψ keeping vertex in [min,max]
                    if(capV<psiL)psiL=capV; }
                if(psiL<0)psiL=0; if(psiL>2.0)psiL=2.0;
                if(GENCAP1 && psiL>1.0)psiL=1.0;   // diagnostic: cap ψ≤1 (dissipative test)
                double dLfx=fx-clx, dLfy=fy-cly;
                W_L[(size_t)v*Nf+f] = wo + psiL*( 0.5*djump + gfx*(dLfx-0.5*rx)+gfy*(dLfy-0.5*ry) );
                // ----- R side (donor=R, acceptor=L, d_RL=-d_LR) -----
                double upR=-(gn0*rx+gn1*ry);                           // ∇φ_R·d_RL = φ_R − φ_LL_R
                double djR=-djump;                                     // φ_L − φ_R
                double rR=upR/(djR+(djR>=0?eps:-eps));
                double psiR;
                if (GENCG && nvar>=4) { double s=0.5*(samt[o]+samt[n]); psiR=(1.0-s)*psi_van_leer(rR)+s*psi_mstacs(rR,Co); }
                else psiR = GENVLE ? psi_van_leer(rR) : (GENMS ? psi_mstacs(rR,Co) : psi_van_leer(rR));
                if (GENDUC && nvar>=4) { double D=0.5*(gduc[o]+gduc[n]); psiR=(1.0-D)*psiR + D*std::min(1.0,psi_van_leer(rR)); }
                if (GENPSAFE && v==nvar-1 && psiR>1.0) psiR=1.0;
                for(int k=0;k<c.max_v;++k){ int vn=c.sample_vid[(size_t)n*c.max_v+k]; if(vn<0)continue;
                    double dx=c.sample_off[((size_t)n*c.max_v+k)*2+0], dy=c.sample_off[((size_t)n*c.max_v+k)*2+1];
                    double dphiV=0.5*djR + gfx*(dx+0.5*rx)+gfy*(dy+0.5*ry);   // d_RV − 0.5 d_RL = d_RV + 0.5 d_LR
                    double bound=(dphiV>0)?(vmx[(size_t)v*c.Nn+vn]-wn):(vmn[(size_t)v*c.Nn+vn]-wn);
                    double capV=bound/(std::fabs(dphiV)>eps?dphiV:(dphiV>=0?eps:-eps));
                    if(capV<psiR)psiR=capV; }
                if(psiR<0)psiR=0; if(psiR>2.0)psiR=2.0;
                if(GENCAP1 && psiR>1.0)psiR=1.0;   // diagnostic: cap ψ≤1 (dissipative test)
                double dRfx=fx-crx, dRfy=fy-cry;
                W_R[(size_t)v*Nf+f] = wn + psiR*( 0.5*djR + gfx*(dRfx+0.5*rx)+gfy*(dRfy+0.5*ry) );
                if (GENPF && (v==0 || v==nvar-1)) {   // optional positivity floor on rho, p (off by default)
                    double fL=posfloor*wo, fR=posfloor*wn;
                    if (W_L[(size_t)v*Nf+f]<fL) W_L[(size_t)v*Nf+f]=fL;
                    if (W_R[(size_t)v*Nf+f]<fR) W_R[(size_t)v*Nf+f]=fR;
                }
                continue;
            }
            if (UFORM) {   // φ_fL = φ_L + ψ_L[ 0.5(φ_R−φ_L) + ∇φ_L·(m_f − m_LR) ]   (central-jump normal + one-sided skew)
                if (n<0) { W_L[(size_t)v*Nf+f]=wo; W_R[(size_t)v*Nf+f]=wo; continue; }
                double wn=W[(size_t)v*N+n], gn0=grad[((size_t)v*N+n)*2+0], gn1=grad[((size_t)v*N+n)*2+1];
                double clx=cc[o*2+0],cly=cc[o*2+1], crx=cc[n*2+0],cry=cc[n*2+1];
                double rx=crx-clx, ry=cry-cly;                      // d_LR
                double sx=fx-0.5*(clx+crx), sy=fy-0.5*(cly+cry);    // m_f − m_LR (skewness)
                double upL=go0*rx+go1*ry;
                double rL=(std::fabs(upL)>eps)?(wn-wo)/upL:((wn-wo)*upL>=0?1e30:-1e30);
                double psiL=std::min(psi_van_leer(rL), psi_lmp(o,v,go0,go1));
                W_L[(size_t)v*Nf+f] = wo + psiL*( 0.5*(wn-wo) + (go0*sx+go1*sy) );
                double upR=-(gn0*rx+gn1*ry);
                double rR=(std::fabs(upR)>eps)?(wo-wn)/upR:((wo-wn)*upR>=0?1e30:-1e30);
                double psiR=std::min(psi_van_leer(rR), psi_lmp(n,v,gn0,gn1));
                W_R[(size_t)v*Nf+f] = wn + psiR*( 0.5*(wo-wn) + (gn0*sx+gn1*sy) );
                continue;
            }
            if (NCC) {   // hybrid: one-sided MUSCL base (stable, distinct L/R) blended -> NCC contact compression
                if (n<0) { W_L[(size_t)v*Nf+f]=wo; W_R[(size_t)v*Nf+f]=wo; continue; }
                double wn=W[(size_t)v*N+n], gn0=grad[((size_t)v*N+n)*2+0], gn1=grad[((size_t)v*N+n)*2+1];
                double clx=cc[o*2+0],cly=cc[o*2+1], crx=cc[n*2+0],cry=cc[n*2+1];
                double rx=crx-clx, ry=cry-cly;
                // --- one-sided MUSCL base (van Leer): each side own gradient -> Riemann dissipation -> STABLE ---
                double incLo=go0*(fx-clx)+go1*(fy-cly), upLo=go0*rx+go1*ry;
                double rLo=(std::fabs(upLo)>eps)?(wn-wo)/upLo:((wn-wo)*upLo>=0?1e30:-1e30);
                double WLos=wo + std::min(psi_van_leer(rLo), psi_lmp(o,v,go0,go1))*incLo;  // LMP-bounded (stable)
                double incRo=gn0*(fx-crx)+gn1*(fy-cry), upRo=-(gn0*rx+gn1*ry);
                double rRo=(std::fabs(upRo)>eps)?(wo-wn)/upRo:((wo-wn)*upRo>=0?1e30:-1e30);
                double WRos=wn + std::min(psi_van_leer(rRo), psi_lmp(n,v,gn0,gn1))*incRo;  // LMP-bounded (stable)
                // --- feature gate: compress only contacts (sharp + low pressure-jump), NOT shocks ---
                double loO=wo,hiO=wo,loN=wn,hiN=wn;
                double po=W[(size_t)(nvar-1)*N+o], pn=W[(size_t)(nvar-1)*N+n], pjO=0.0, pjN=0.0;
                for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)o*c.max_nb+k]; if(nb<0)continue;
                    double q=W[(size_t)v*N+nb]; loO=std::min(loO,q); hiO=std::max(hiO,q);
                    double pq=W[(size_t)(nvar-1)*N+nb]; pjO=std::max(pjO,std::fabs(pq-po)/(std::fabs(pq)+std::fabs(po)+eps)); }
                for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)n*c.max_nb+k]; if(nb<0)continue;
                    double q=W[(size_t)v*N+nb]; loN=std::min(loN,q); hiN=std::max(hiN,q);
                    double pq=W[(size_t)(nvar-1)*N+nb]; pjN=std::max(pjN,std::fabs(pq-pn)/(std::fabs(pq)+std::fabs(pn)+eps)); }
                // compress ONLY density (the contact variable); p, velocity stay one-sided (acoustic-stable)
                bool compv = (v==0);
                double chiL=(compv && sharp[(size_t)v*N+o] && pjO<NCC_PJ)?NCC_CHI:0.0;
                double chiR=(compv && sharp[(size_t)v*N+n] && pjN<NCC_PJ)?NCC_CHI:0.0;
                if (chiL<=0.0 && chiR<=0.0) { W_L[(size_t)v*Nf+f]=WLos; W_R[(size_t)v*Nf+f]=WRos; continue; }
                // --- NCC compressed value (skew base + purified normal jump, max compression in TMLP-u bound) ---
                double nx=m.face_normals[(size_t)f*2+0], ny=m.face_normals[(size_t)f*2+1];
                double nl=std::sqrt(nx*nx+ny*ny); if(nl>eps){nx/=nl;ny/=nl;}
                double dr=rx*nx+ry*ny, trx=rx-dr*nx, tr_y=ry-dr*ny;
                double jpure=0.5*(go0+gn0)*trx+0.5*(go1+gn1)*tr_y;
                double mLx=fx-clx,mLy=fy-cly, dmL=mLx*nx+mLy*ny, tLx=mLx-dmL*nx,tLy=mLy-dmL*ny;
                double corrL=go0*tLx+go1*tLy, roomL=corrL>=0?(hiO-wo):(wo-loO);
                double lamTL=(std::fabs(corrL)>eps)?std::min(NCC_LAMT,std::max(0.0,roomL)/std::fabs(corrL)):NCC_LAMT;
                double B_L=wo+lamTL*corrL, JnL=(wn-wo)-NCC_LAMN*jpure, betaTL;
                if(JnL>eps) betaTL=std::min(1.0,(hiO-B_L)/JnL); else if(JnL<-eps) betaTL=std::min(1.0,(loO-B_L)/JnL); else betaTL=1.0;
                betaTL=betaTL<0?0:(betaTL>1?1:betaTL);
                double WLc=B_L+betaTL*JnL; WLc=WLc<loO?loO:(WLc>hiO?hiO:WLc);
                W_L[(size_t)v*Nf+f]=(1.0-chiL)*WLos+chiL*WLc;
                double mRx=fx-crx,mRy=fy-cry, dmR=mRx*nx+mRy*ny, tRx=mRx-dmR*nx,tRy=mRy-dmR*ny;
                double corrR=gn0*tRx+gn1*tRy, roomR=corrR>=0?(hiN-wn):(wn-loN);
                double lamTR=(std::fabs(corrR)>eps)?std::min(NCC_LAMT,std::max(0.0,roomR)/std::fabs(corrR)):NCC_LAMT;
                double B_R=wn+lamTR*corrR, JnR=(wo-wn)+NCC_LAMN*jpure, betaTR;
                if(JnR>eps) betaTR=std::min(1.0,(hiN-B_R)/JnR); else if(JnR<-eps) betaTR=std::min(1.0,(loN-B_R)/JnR); else betaTR=1.0;
                betaTR=betaTR<0?0:(betaTR>1?1:betaTR);
                double WRc=B_R+betaTR*JnR; WRc=WRc<loN?loN:(WRc>hiN?hiN:WRc);
                W_R[(size_t)v*Nf+f]=(1.0-chiR)*WRos+chiR*WRc;
                continue;
            }
            double inc = go0*(fx-cc[o*2+0]) + go1*(fy-cc[o*2+1]);
            double pl = psi_lmp(o, v, go0, go1);
            double psi;
            if (n>=0){ double dpx=cc[n*2+0]-cc[o*2+0], dpy=cc[n*2+1]-cc[o*2+1];
                double up = go0*dpx + go1*dpy;          // upwind slope (grad . d_on)
                double dplus = W[(size_t)v*N+n]-wo;     // downwind difference
                double r = (std::fabs(up)>eps) ? dplus/up : (dplus*up>=0?1e30:-1e30);
                double pt, cap=psi_cap, plv=pl;   // C2/DW: pl is the real bound. plv = (XREL-relaxed) velocity bound.
                if (DW) { pt = sharp[(size_t)v*N+o] ? (r>0.0?2.0:0.0) : psi_van_leer(r); }  // sharp=downwind smooth=vanLeer
                else if (CP && nvar>=4) {
                    double gs=std::max(cp_shock[o],cp_shock[n]), gc=0.5*(cp_contact[o]+cp_contact[n]);
                    double gv=std::max(cp_vortex[o],cp_vortex[n]), vl=psi_van_leer(r);
                    if (v==0) { double rho_hi = CP_RHO_DW ? (r>0.0?2.0:0.0) : psi_mstacs(r,Co);
                                pt=(1.0-gc)*vl + gc*rho_hi; }                  // rho: contact sharpen
                    else if (v==nvar-1) pt=vl;                                  // p: TVD only
                    else { double gcv=std::min(1.0, CP_VCONTACT_GAIN*gc);
                           double gf=CP_TCONTACT_ONLY?gcv:std::max(gv,gcv);
                           double vsharp = CP_T_DW ? (r>0.0?2.0:0.0) : psi_mstacs(r,Co);
                           double vhi=CP_VSHARP?((1.0-CP_TSHARP)*vl + CP_TSHARP*vsharp):1.0;
                           pt=(1.0-gf)*vl + gf*vhi; }                          // u,v: preserve/sharpen tangential
                    cap=1.0 - gs*cap_str;                                       // shock psi cap (rho,p,u_n,u_t)
                }
                else if (C2 && nvar>=4) { double sa=samt[o];
                    if (C2_SHF && (v==1||v==2)) sa *= (1.0 - c2rot[o]);      // free shear velocity
                    double sm = C2_SMU1 ? 1.0 : psi_van_leer(r);            // smooth tier
                    pt = (1.0-sa)*sm + sa*psi_mstacs(r,Co);                  // sharp=MSTACS
                    if (VCOMP>0.0 && (v==1||v==2)) { double sgo = VCOMP_VSHARP ? std::max(samt[o], vsh[o]) : samt[o];
                        double cg = VCOMP_CONTACT_ONLY ? (VCOMP_MIN_CONTACT + (1.0 - VCOMP_MIN_CONTACT)*sgo) : 1.0;
                        double g=std::min(1.0, std::max(0.0, VCOMP*c2rot[o]*cg)); // ψ∈[0,2], ψ≤ψ_LMP: bounded contact/shear compression
                        pt = (1.0-g)*pt + g*psi_mstacs(r,Co); }              // ψ=min(pt,pl): bounded by LMP, monotone, NO bound-relax
                    if (TVBM>0.0 && (v==1||v==2)) plv=psi_lmp_tvb(o,v,go0,go1,TVBM*c2rot[o]); // resolution-consistent un-clip (bulletproof)
                    else if (XREL>0.0 && (v==1||v==2)) { double xr=XREL*c2rot[o]; // extremum-relax velocity at smooth vortex
                        if (!XCAP1) pt+=xr;                                  // ψ>1 anti-diffusion (off in un-clip mode)
                        plv=std::min(psi_cap, pl*(1.0+xr)); } }
                else pt = sharp[(size_t)v*N+o] ? psi_cicsam(r,Co) : (nvar==1?psi_van_leer(r):1.0);
                psi = std::min(std::min(pt, plv), cap);
            } else psi = std::min(1.0, pl);
            W_L[(size_t)v*Nf+f] = wo + psi*inc;
            if ((C2||CP) && (v==0||v==nvar-1) && W_L[(size_t)v*Nf+f] < posfloor*wo) W_L[(size_t)v*Nf+f]=posfloor*wo;
            if (n>=0){ double wn=W[(size_t)v*N+n], gn0=grad[((size_t)v*N+n)*2+0], gn1=grad[((size_t)v*N+n)*2+1];
                double incn = gn0*(fx-cc[n*2+0]) + gn1*(fy-cc[n*2+1]);
                double pln = psi_lmp(n, v, gn0, gn1);
                double dpx=cc[o*2+0]-cc[n*2+0], dpy=cc[o*2+1]-cc[n*2+1];
                double up=gn0*dpx+gn1*dpy, dplus=wo-wn;
                double r=(std::fabs(up)>eps)?dplus/up:(dplus*up>=0?1e30:-1e30);
                double pt, cap=psi_cap, plv=pln;   // C2/DW: pln is the real bound. plv = (XREL-relaxed) velocity bound.
                if (DW) { pt = sharp[(size_t)v*N+n] ? (r>0.0?2.0:0.0) : psi_van_leer(r); }  // sharp=downwind smooth=vanLeer
                else if (CP && nvar>=4) {
                    double gs=std::max(cp_shock[o],cp_shock[n]), gc=0.5*(cp_contact[o]+cp_contact[n]);
                    double gv=std::max(cp_vortex[o],cp_vortex[n]), vl=psi_van_leer(r);
                    if (v==0) { double rho_hi = CP_RHO_DW ? (r>0.0?2.0:0.0) : psi_mstacs(r,Co);
                                pt=(1.0-gc)*vl + gc*rho_hi; }
                    else if (v==nvar-1) pt=vl;
                    else { double gcv=std::min(1.0, CP_VCONTACT_GAIN*gc);
                           double gf=CP_TCONTACT_ONLY?gcv:std::max(gv,gcv);
                           double vsharp = CP_T_DW ? (r>0.0?2.0:0.0) : psi_mstacs(r,Co);
                           double vhi=CP_VSHARP?((1.0-CP_TSHARP)*vl + CP_TSHARP*vsharp):1.0;
                           pt=(1.0-gf)*vl + gf*vhi; }
                    cap=1.0 - gs*cap_str;
                }
                else if (C2 && nvar>=4) { double sa=samt[n];
                    if (C2_SHF && (v==1||v==2)) sa *= (1.0 - c2rot[n]);      // free shear velocity
                    double sm = C2_SMU1 ? 1.0 : psi_van_leer(r);            // smooth tier
                    pt = (1.0-sa)*sm + sa*psi_mstacs(r,Co);                  // sharp=MSTACS
                    if (VCOMP>0.0 && (v==1||v==2)) { double sgn = VCOMP_VSHARP ? std::max(samt[n], vsh[n]) : samt[n];
                        double cg = VCOMP_CONTACT_ONLY ? (VCOMP_MIN_CONTACT + (1.0 - VCOMP_MIN_CONTACT)*sgn) : 1.0;
                        double g=std::min(1.0, std::max(0.0, VCOMP*c2rot[n]*cg)); // ψ∈[0,2], ψ≤ψ_LMP: bounded contact/shear compression
                        pt = (1.0-g)*pt + g*psi_mstacs(r,Co); }              // ψ=min(pt,pln): bounded by LMP, monotone, NO bound-relax
                    if (TVBM>0.0 && (v==1||v==2)) plv=psi_lmp_tvb(n,v,gn0,gn1,TVBM*c2rot[n]); // resolution-consistent un-clip (bulletproof)
                    else if (XREL>0.0 && (v==1||v==2)) { double xr=XREL*c2rot[n]; // extremum-relax velocity at smooth vortex
                        if (!XCAP1) pt+=xr;                                  // ψ>1 anti-diffusion (off in un-clip mode)
                        plv=std::min(psi_cap, pln*(1.0+xr)); } }
                else pt = sharp[(size_t)v*N+n] ? psi_cicsam(r,Co) : (nvar==1?psi_van_leer(r):1.0);
                double psin=std::min(std::min(pt,plv),cap);
                W_R[(size_t)v*Nf+f] = wn + psin*incn;
                if ((C2||CP) && (v==0||v==nvar-1) && W_R[(size_t)v*Nf+f] < posfloor*wn) W_R[(size_t)v*Nf+f]=posfloor*wn;
            } else W_R[(size_t)v*Nf+f] = W_L[(size_t)v*Nf+f];
        }
    }
    // TMLPU_FLAG diagnostic: aggregate per-face density branch ids into a per-cell majority vote and
    // publish into the global tmlpu_branch_flag() buffer so the VTK writer can emit CELL_DATA branch_flag.
    if (FLAG){ auto& cf=cfd::tmlpu_branch_flag(); cf.assign((size_t)N,-1);
        std::vector<int> c0(N,0),c1(N,0),c2(N,0);
        for(int f2=0;f2<Nf;++f2){ signed char fl=fflag[f2]; if(fl<0)continue;
            int oo=m.face_owner[f2], nn=m.face_neighbour[f2];
            int* a=(fl==0)?c0.data():(fl==1?c1.data():c2.data()); a[oo]++; if(nn>=0)a[nn]++; }
        for(int i=0;i<N;++i){ int t=c0[i]+c1[i]+c2[i]; if(!t){cf[i]=-1;continue;}
            int best=0,val=c0[i]; if(c1[i]>val){val=c1[i];best=1;} if(c2[i]>val){val=c2[i];best=2;} cf[i]=(signed char)best; } }
    // Joint-θ positivity limiter (GEN): scale the WHOLE reconstruction increment by one θ∈[0,1]
    // per face-side so rho,p stay ≥ GENPOSF·cell. Preserves consistency (all vars same θ),
    // unlike per-variable clamp. Reduces compression only where positivity is at risk.
    if (GEN && GENPOS && nvar>=4) {
        #pragma omp parallel for
        for (int f=0; f<Nf; ++f){ int o=m.face_owner[f], n=m.face_neighbour[f]; if(n<0)continue;
            auto theta=[&](int ci, std::vector<double>& Wf)->double{ double th=1.0;
                for(int vv=0; vv<nvar; vv+=(nvar-1)){ // vv=0 (rho) and vv=nvar-1 (p)
                    double wc=W[(size_t)vv*N+ci], wf=Wf[(size_t)vv*Nf+f], fl=GENPOSF*wc;
                    if(wf<fl){ double dd=wf-wc; if(dd<-1e-300){ double t=(fl-wc)/dd; if(t<th)th=t; } } }
                return th<0?0:th; };
            double tL=theta(o,W_L); if(tL<1.0) for(int vv=0;vv<nvar;++vv){ double wc=W[(size_t)vv*N+o]; W_L[(size_t)vv*Nf+f]=wc+tL*(W_L[(size_t)vv*Nf+f]-wc); }
            double tR=theta(n,W_R); if(tR<1.0) for(int vv=0;vv<nvar;++vv){ double wc=W[(size_t)vv*N+n]; W_R[(size_t)vv*Nf+f]=wc+tR*(W_R[(size_t)vv*Nf+f]-wc); }
        }
    }
    if (CP && CP_NT && nvar >= 4) {
        #pragma omp parallel for
        for (int f = 0; f < Nf; ++f) {
            int o = m.face_owner[f], n = m.face_neighbour[f];
            double fx = m.face_centers[f*2+0], fy = m.face_centers[f*2+1];
            double nx = m.face_normals[f*2+0], ny = m.face_normals[f*2+1];
            double tx = -ny, ty = nx;

            auto rebuild_side = [&](int ci, int opp, double& uf, double& vf) {
                double uc = W[(size_t)1*N+ci], vc = W[(size_t)2*N+ci];
                double ucn = uc*nx + vc*ny;
                double uct = uc*tx + vc*ty;
                double gux = grad[((size_t)1*N+ci)*2+0];
                double guy = grad[((size_t)1*N+ci)*2+1];
                double gvx = grad[((size_t)2*N+ci)*2+0];
                double gvy = grad[((size_t)2*N+ci)*2+1];
                double gunx = nx*gux + ny*gvx, guny = nx*guy + ny*gvy;
                double gutx = tx*gux + ty*gvx, guty = tx*guy + ty*gvy;
                double dx = fx - cc[ci*2+0], dy = fy - cc[ci*2+1];
                double incn = gunx*dx + guny*dy;
                double inct = gutx*dx + guty*dy;
                double plu = psi_lmp(ci, 1, gux, guy);
                double plv = psi_lmp(ci, 2, gvx, gvy);
                double pl = std::min(plu, plv);
                auto psi_lmp_projected = [&](double wc, double gx, double gy, double ax, double ay) -> double {
                    double p = psi_cap;
                    for (int k = 0; k < c.max_v; ++k) {
                        int vn = c.sample_vid[(size_t)ci*c.max_v+k];
                        if (vn < 0) continue;
                        double dxv = c.sample_off[((size_t)ci*c.max_v+k)*2+0];
                        double dyv = c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                        double proj = gx*dxv + gy*dyv;
                        double mn = 1e300, mx = -1e300;
                        for (int kk = 0; kk < c.max_v2c; ++kk) {
                            int cj = c.v2c[(size_t)vn*c.max_v2c+kk];
                            if (cj < 0) continue;
                            double val = W[(size_t)1*N+cj]*ax + W[(size_t)2*N+cj]*ay;
                            if (val < mn) mn = val;
                            if (val > mx) mx = val;
                        }
                        if (mn > mx) continue;
                        double allowed = proj >= 0.0 ? (mx - wc) : (wc - mn);
                        double pk = (std::fabs(proj)>eps) ? std::max(allowed,0.0)/std::max(std::fabs(proj),eps) : psi_cap;
                        if (pk < p) p = pk;
                    }
                    return p < 0.0 ? 0.0 : (p > psi_cap ? psi_cap : p);
                };
                double pl_n = CP_PROJ_LMP ? psi_lmp_projected(ucn, gunx, guny, nx, ny) : pl;
                double pl_t = CP_PROJ_LMP ? psi_lmp_projected(uct, gutx, guty, tx, ty) : pl;
                double psi_n = std::min(1.0, pl_n);
                double psi_t = std::min(1.0, pl_t);
                if (opp >= 0) {
                    double uo = W[(size_t)1*N+opp], vo = W[(size_t)2*N+opp];
                    double oppn = uo*nx + vo*ny;
                    double oppt = uo*tx + vo*ty;
                    double dcx = cc[opp*2+0] - cc[ci*2+0];
                    double dcy = cc[opp*2+1] - cc[ci*2+1];
                    double upn = gunx*dcx + guny*dcy;
                    double upt = gutx*dcx + guty*dcy;
                    double rn = (std::fabs(upn)>eps) ? (oppn-ucn)/upn : ((oppn-ucn)*upn>=0?1e30:-1e30);
                    double rt = (std::fabs(upt)>eps) ? (oppt-uct)/upt : ((oppt-uct)*upt>=0?1e30:-1e30);
                    double gs = std::max(cp_shock[ci], cp_shock[opp]);
                    double gc = 0.5*(cp_contact[ci] + cp_contact[opp]);
                    double gv = std::max(cp_vortex[ci], cp_vortex[opp]);
                    double cap = 1.0 - gs*cap_str;
                    double vln = psi_van_leer(rn);
                    double vlt = psi_van_leer(rt);
                    double gcv = std::min(1.0, CP_VCONTACT_GAIN*gc);
                    double gf = CP_TCONTACT_ONLY ? gcv : std::max(gv, gcv);
                    double tsharp = CP_T_DW ? (rt>0.0?2.0:0.0) : psi_mstacs(rt, Co);
                    double t_hi = CP_VSHARP
                        ? ((1.0-CP_TSHARP)*vlt + CP_TSHARP*tsharp)
                        : std::max(1.0, vlt);
                    double pt = (1.0-gf)*vlt + gf*t_hi;
                    psi_n = std::min(std::min(vln, pl_n), cap);
                    psi_t = std::min(std::min(pt, pl_t), CP_T_NOCAP ? 1.0 : cap);
                }
                double un = ucn + psi_n*incn;
                double ut = uct + psi_t*inct;
                uf = un*nx + ut*tx;
                vf = un*ny + ut*ty;
            };

            double uL, vL;
            rebuild_side(o, n, uL, vL);
            W_L[(size_t)1*Nf+f] = uL;
            W_L[(size_t)2*Nf+f] = vL;
            if (n >= 0) {
                double uR, vR;
                rebuild_side(n, o, uR, vR);
                W_R[(size_t)1*Nf+f] = uR;
                W_R[(size_t)2*Nf+f] = vR;
            } else {
                W_R[(size_t)1*Nf+f] = uL;
                W_R[(size_t)2*Nf+f] = vL;
            }
        }
    }
}

// T-MLP-u-D directional box relaxation (added OUTSIDE the monotone clip). `base` is
// the already-limited (and optionally face-bound-clipped) face value = W_c+phi*(g.d).
// We split the gradient in the cell feature frame (n̂=layer normal, t̂=n̂⊥) and ADD
// extra anti-diffusion ONLY along the shear tangent, gated by the Ducros sensor s:
//   phi_t = phi + s*kappa*(1-phi);  extra = (phi_t-phi)*(g.t̂)(d.t̂);  val = base+extra.
// Reduces to `base` EXACTLY when kappa=0 or s=0 (=> bit-identical to MLP-u). For the
// positivity variables (rho,p) a hard floor val>=floorf*W_c keeps the FIXED HLLC
// positive (W_c>0). See docs/tmlpu_tensor_limiter_design.md.
inline double dir_relax_face(double base, double gx, double gy,
        double dx, double dy, double nx, double ny, double phi, double s,
        double kappa, double wc, bool pos_var, double floorf) {
    double phi_t = phi + s*kappa*(1.0 - phi);
    if (phi_t == phi) return base;            // kappa=0 or s=0 -> exact baseline
    double tx = -ny, ty = nx;                 // tangent = normal rotated 90 deg
    double inc_t = (gx*tx + gy*ty) * (dx*tx + dy*ty);
    double val = base + (phi_t - phi) * inc_t;
    if (pos_var && val < floorf*wc) val = floorf*wc;   // floor rho,p (wc>0)
    return val;
}

// Reconstruct W_L/W_R at face centres. W is nvar*N (var-major). Outputs nvar*Nf.
// face_bound = T-MLP-u face LMP clamp (mlp_u1_tmlpu). Boundary faces: first-order.
// shock_ctx (optional): if non-null and nvar>=4, cells flagged as shocks (large
// normalized |grad p|) use shock_ctx's weights+ATA_inv for the gradient (e.g.
// idw_p=0 Barth-Jespersen = sharp shock), while non-shock cells use c (e.g.
// idw_p=2 = rich vortices). c and shock_ctx MUST share the same vertex stencil
// (same nb/d/sample/v2c — only idw_p differs). p_thr = |grad p|/p shock threshold.
inline void reconstruct_bj_vertex(const Mesh& m, const ReconCtx& c,
                                  const std::vector<double>& W, int nvar,
                                  std::vector<double>& W_L, std::vector<double>& W_R,
                                  bool face_bound = false,
                                  const ReconCtx* shock_ctx = nullptr,
                                  double p_thr = 0.5) {
    const int N = m.n_cells(), Nf = m.n_faces();
    const double* cc = m.cell_centers.data();
    W_L.assign((size_t)nvar * Nf, 0.0);
    W_R.assign((size_t)nvar * Nf, 0.0);
    // init all faces to first-order
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        for (int v = 0; v < nvar; ++v) {
            W_L[(size_t)v*Nf+f] = W[(size_t)v*N+o];
            W_R[(size_t)v*Nf+f] = (n >= 0) ? W[(size_t)v*N+n] : W[(size_t)v*N+o];
        }
    }
    std::vector<double> grad((size_t)nvar * N * 2, 0.0);
    std::vector<double> phi((size_t)nvar * N, 1.0);
    // per-cell gradient (LSQ) + cell stencil min/max (for face bound)
    std::vector<double> smin((size_t)nvar * N), smax((size_t)nvar * N);
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        // shock-aware: pick weight/ATA set for this cell's gradient.
        const double* wW = c.w.data(); const double* AI = c.ATA_inv.data();
        if (shock_ctx && nvar >= 4) {
            double pc = W[(size_t)(nvar-1)*N+ci];
            double maxjump = 0.0; // max relative pressure jump to a neighbour (dimensionless)
            for (int k = 0; k < c.max_nb; ++k) { int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                double pj=std::fabs(W[(size_t)(nvar-1)*N+nb]-pc)/std::max(std::fabs(pc),1e-30);
                if (pj>maxjump) maxjump=pj; }
            if (maxjump > p_thr) { wW = shock_ctx->w.data(); AI = shock_ctx->ATA_inv.data(); }
        }
        for (int v = 0; v < nvar; ++v) {
            double r0 = 0, r1 = 0, wc = W[(size_t)v*N+ci];
            double mn = wc, mx = wc;
            for (int k = 0; k < c.max_nb; ++k) {
                int nb = c.nb[(size_t)ci*c.max_nb+k];
                if (nb < 0) continue;
                double dphi = W[(size_t)v*N+nb] - wc;
                double wk = wW[(size_t)ci*c.max_nb+k];
                r0 += wk * c.d[((size_t)ci*c.max_nb+k)*2+0] * dphi;
                r1 += wk * c.d[((size_t)ci*c.max_nb+k)*2+1] * dphi;
                if (W[(size_t)v*N+nb] < mn) mn = W[(size_t)v*N+nb];
                if (W[(size_t)v*N+nb] > mx) mx = W[(size_t)v*N+nb];
            }
            double gx = AI[ci*4+0]*r0 + AI[ci*4+1]*r1;
            double gy = AI[ci*4+2]*r0 + AI[ci*4+3]*r1;
            grad[((size_t)v*N+ci)*2+0] = gx; grad[((size_t)v*N+ci)*2+1] = gy;
            smin[(size_t)v*N+ci] = mn; smax[(size_t)v*N+ci] = mx;
        }
    }
    // vertex bounds: per node min/max of W over sharing cells.
    std::vector<double> vmin((size_t)nvar * c.Nn), vmax((size_t)nvar * c.Nn);
    #pragma omp parallel for
    for (int v = 0; v < c.Nn; ++v) {
        for (int var = 0; var < nvar; ++var) {
            double mn = 1e300, mx = -1e300;
            for (int k = 0; k < c.max_v2c; ++k) {
                int ci = c.v2c[(size_t)v*c.max_v2c+k];
                if (ci < 0) continue;
                double val = W[(size_t)var*N+ci];
                if (val < mn) mn = val; if (val > mx) mx = val;
            }
            if (mn > mx) { mn = 0; mx = 0; }
            vmin[(size_t)var*c.Nn+v] = mn; vmax[(size_t)var*c.Nn+v] = mx;
        }
    }
    // phi = min over cell vertices. BJ (MLP-u1) by default; MLP_U2=K env -> the
    // less-diffusive Venkatakrishnan f_V on the SAME bounds (cost-identical).
    static const char* U2 = std::getenv("MLP_U2");
    static const double U2K = U2 ? std::atof(U2) : 0.0;
    const bool u2 = U2K > 0.0;
    // MLP_S: shock-sensor gate. s = w^2/(w^2+th^2) in [0,1] (Ducros): ~1 in shear/
    // vortex, ~0 at shocks (compression). Blend phi_eff = (1-s)*phi_BJ + s*1 so the
    // limiter is FULL (sharp+monotone) at shocks but OFF (phi=1, no diffusion) in
    // vortices -> sharp shocks AND low diffusion. Needs velocity (nvar>=4).
    static const bool USE_S = std::getenv("MLP_S") != nullptr;
    static const char* SMT = std::getenv("MLP_S_THR");
    static const double S_THR = SMT ? std::atof(SMT) : 0.2;
    const bool s_scalar = USE_S && nvar == 1 && std::getenv("MLP_S_SCALAR") != nullptr;
    // MLP_S_VELONLY: apply the Ducros shear-relaxation ONLY to the velocity components
    // (var 1,2), leaving rho (0) and p (3) fully BJ/MLP-limited. Prevents the p/rho
    // un-limiting at s->1 that causes p_face<0 divergence (the safe velocity-shear fix
    // for KH billows: un-throttle the tangential velocity slope, keep thermodynamics
    // bounded). Default OFF (all-var relaxation, needs TMLPU_POS_FLOOR as backstop).
    static const bool S_VELONLY = std::getenv("MLP_S_VELONLY") != nullptr;
    std::vector<double> svtx(USE_S && (nvar>=4 || nvar==1) ? N : 0, 0.0);
    if (USE_S && nvar >= 4) {
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double dudx=grad[((size_t)1*N+ci)*2+0], dudy=grad[((size_t)1*N+ci)*2+1];
            double dvdx=grad[((size_t)2*N+ci)*2+0], dvdy=grad[((size_t)2*N+ci)*2+1];
            double th=dudx+dvdy, om=dvdx-dudy;
            svtx[ci] = (om*om)/(om*om + th*th + 1e-30);
        }
    } else if (s_scalar) {
        // scalar smoothness sensor: relax limiter where the field is SMOOTH (small
        // neighbour jump -> preserve smooth extrema like cone/hump peaks), keep it tight
        // at discontinuities (slot edge). s = 1 - clip(maxjump/(thr*range), 0, 1).
        double gmn=1e300, gmx=-1e300;
        for (int i=0;i<N;++i){ double w=W[i]; if(w<gmn)gmn=w; if(w>gmx)gmx=w; }
        double rng=std::max(gmx-gmn,1e-30);
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double wc=W[ci], mj=0.0;
            for (int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
                double d=std::fabs(W[nb]-wc); if(d>mj)mj=d; }
            double r=mj/(S_THR*rng); svtx[ci]=1.0-(r<0?0:(r>1?1:r));
        }
    }
    #pragma omp parallel for
    for (int ci = 0; ci < N; ++ci) {
        double eps2 = u2 ? std::pow(U2K * std::sqrt(2.0*m.cell_volumes[ci]), 3) : 0.0;
        for (int var = 0; var < nvar; ++var) {
            double wc = W[(size_t)var*N+ci], pmin = 1.0;
            double gx = grad[((size_t)var*N+ci)*2+0], gy = grad[((size_t)var*N+ci)*2+1];
            for (int k = 0; k < c.max_v; ++k) {
                int v = c.sample_vid[(size_t)ci*c.max_v+k];
                if (v < 0) continue;
                double sd = gx * c.sample_off[((size_t)ci*c.max_v+k)*2+0]
                          + gy * c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                double p = u2 ? venk_phi(sd, wc, vmin[(size_t)var*c.Nn+v], vmax[(size_t)var*c.Nn+v], eps2)
                              : bj_phi(sd, wc, vmin[(size_t)var*c.Nn+v], vmax[(size_t)var*c.Nn+v]);
                if (p < pmin) pmin = p;
            }
            if (USE_S && (nvar >= 4 || s_scalar) && (!S_VELONLY || var==1 || var==2)) { double s = svtx[ci]; pmin = (1.0-s)*pmin + s; }
            phi[(size_t)var*N+ci] = pmin;
        }
    }
    // T-MLP-u-D: per-cell feature frame (n̂ = top eigenvector of (grad u)^T(grad u)
    // = direction of strongest velocity variation = layer normal) + Ducros gate s.
    // Used to add tangential anti-diffusion (dir_relax_face) at the face loop. Needs
    // velocity (nvar>=4). Default OFF (env TMLPU_DIR_LIMIT) -> baseline unchanged.
    static const bool USE_DIR = std::getenv("TMLPU_DIR_LIMIT") != nullptr;
    static const char* DKA = std::getenv("TMLPU_DIR_KAPPA");
    static const double DKAPPA = DKA ? std::atof(DKA) : 1.0;
    static const char* DFL = std::getenv("TMLPU_DIR_FLOOR");
    static const double DFLOOR = DFL ? std::atof(DFL) : 0.2;
    // DIR_CLIP: re-impose the face-local monotone bound [smin,smax] on the COMBINED
    // directional value (fixes the opposite-sign inc_n/inc_t overshoot that breaks
    // monotonicity). Allows the tangential to use the full face-local admissible
    // range (looser than the vertex-min MLP bound) but no further.
    static const bool DIR_CLIP = std::getenv("TMLPU_DIR_CLIP") != nullptr;
    // POS_FLOOR: general primitive positivity floor on the reconstructed FACE state
    // (rho,p >= factor*cell). For PRIMITIVE reconstruction, rho_face>0 AND p_face>0 is
    // the full admissibility (independent of u,v), so this makes the FIXED HLLC keep
    // the cell average positive under CFL. Fixes the MLP_S divergence (unlimited p at
    // s=1 -> p_face<0). Applies to whatever variant is active. Default OFF.
    static const char* PFL = std::getenv("TMLPU_POS_FLOOR");
    static const double PFLOOR = PFL ? std::atof(PFL) : 0.0;
    static const bool USE_PFLOOR = PFL != nullptr;
    // RECOVERY prototype: blend the face value toward the CONTINUOUS recovered value
    // wrec = 0.5*(wL_unlimited + wR_unlimited) by factor g. g=1 => W_L=W_R=wrec (zero
    // face jump => FIXED HLLC gives the central physical flux, zero dissipation, in
    // smooth regions) WITHOUT un-limiting a gradient (no velocity overshoot). g=0 =>
    // baseline (bit-identical). Risk: g->1 = central = under-dissipative (ringing).
    // Flux UNTOUCHED — this only changes the reconstructed L/R states. Default OFF.
    static const char* REC = std::getenv("TMLPU_RECOVERY");
    static const double RECG = REC ? std::atof(REC) : 0.0;
    static const bool USE_REC = REC != nullptr;
    // REC_GATE: smoothness gate threshold. Recovery is turned OFF at faces with a
    // large normalized density jump (contacts/shocks) where central blending rings;
    // applied only in smooth regions. g_eff = g*max(0,1 - rjump/thr). 0 => no gate.
    static const char* RGT = std::getenv("TMLPU_REC_GATE");
    static const double REC_GATE = RGT ? std::atof(RGT) : 0.0;
    // REC_CLIP: after the recovery blend, clip the face value to the local neighbour
    // range [smin,smax] -> restores boundedness ([0,1] for scalar, positivity for
    // Euler) by capping the central-value overshoot at discontinuities; does NOT bite
    // in smooth regions (no overshoot there), so the diffusion win is preserved.
    static const bool REC_CLIP = std::getenv("TMLPU_REC_CLIP") != nullptr;
    // GENUINE T-MLP-u (port of solver/solve_T-MLP-u/reconstruction.py): the documented
    // compressive face value with the t* downwind term + non-orthogonal/skew correction
    // (beta/theta_min) for UNSTRUCTURED meshes:
    //   delta = comp*t*(W_n-W_o) + grad_corr.(m_f - c_o - t*(c_n-c_o))
    //   grad_corr = grad_bar - beta*(grad_bar.e_o - (W_n-W_o)/|d|) e_o   [non-orthogonal]
    //   grad_bar  = (1-t*)grad_o + t* grad_n                              [skew-interp]
    //   t*        = clip((m_f-c_o).n / (c_n-c_o).n, 0, 1)
    // TENSOR-LIMITER term `comp`: modulate the t* COMPRESSION by the feature frame. Where
    // the centroid line crosses the shear (Ducros s high AND e_o aligned with the shear-
    // normal n_feat), the t* compression DIFFUSES the oblique slip line, so reduce it:
    //   comp = 1 - TSTAR_GATE * s * (e_o . n_feat)^2
    // => re-directs the genuine compression to be feature-aware, fixing the t*-diffuses-KH
    // flaw ON skewed/non-orthogonal meshes (keeps compression at shocks s~0 and where the
    // face is tangential to the shear). TSTAR_GATE: 0 = pure genuine T-MLP-u, 1 = full
    // feature modulation. Default OFF (env TMLPU_GENUINE) -> baseline unchanged.
    static const bool USE_TMLPU = std::getenv("TMLPU_GENUINE") != nullptr;
    static const char* TSG = std::getenv("TMLPU_TSTAR_GATE");
    static const double TSTAR_GATE = TSG ? std::atof(TSG) : 1.0;
    const double THETA_MIN = 0.3;
    // SKEW-ONLY: the non-orthogonal/skew correction WITHOUT the t* compression (the
    // valuable part of genuine T-MLP-u for unstructured meshes, with t*'s smooth-flow
    // harm dropped). grad_corr = grad_o - beta*(grad_o.e_o - (W_n-W_o)/|d|) e_o applied
    // to the full face offset; reduces to baseline (grad_o.dx) for linear data
    // (corr=0) => consistent, 2nd-order. Default OFF.
    static const bool USE_SKEW = std::getenv("TMLPU_SKEW") != nullptr;
    // DIR2: proper per-variable shear-gated bound relaxation. For VELOCITY (u,v) in shear
    // (Ducros s high), relax the vertex-MLP bound toward the looser neighbour-range bound
    // [smin,smax] (Barth-style, still MONOTONE/bounded) to preserve the shear (vorticity);
    // rho,p stay tight (positivity, monotone shocks). Blend by s. Bounded => no overshoot
    // (unlike MLP_S which un-limits ALL vars to phi=1). K=relax strength. Default OFF.
    static const bool USE_DIR2 = std::getenv("TMLPU_DIR2") != nullptr;
    static const char* D2K = std::getenv("TMLPU_DIR2_K");
    static const double DIR2K = D2K ? std::atof(D2K) : 1.0;
    const bool feat = ((USE_DIR || USE_TMLPU || USE_DIR2) && nvar >= 4) || (USE_DIR && nvar == 1);
    std::vector<double> dnrm(feat ? (size_t)2*N : 0, 0.0);
    std::vector<double> dsen(feat ? (size_t)N : 0, 0.0);
    if (feat) {
        #pragma omp parallel for
        for (int ci = 0; ci < N; ++ci) {
            double e0, e1;
            if (nvar >= 4) {   // Euler: shear frame from velocity Jacobian + Ducros gate
                double ux=grad[((size_t)1*N+ci)*2+0], uy=grad[((size_t)1*N+ci)*2+1];
                double vx=grad[((size_t)2*N+ci)*2+0], vy=grad[((size_t)2*N+ci)*2+1];
                double th=ux+vy, om=vx-uy;
                dsen[ci] = (om*om)/(om*om + th*th + 1e-30);
                double a=ux*ux+vx*vx, b=ux*uy+vx*vy, cq=uy*uy+vy*vy;
                double tr=0.5*(a+cq), R=std::sqrt(std::max(0.25*(a-cq)*(a-cq)+b*b, 0.0));
                double lam=tr+R;
                if (std::fabs(b) > 1e-14) { e0=b; e1=lam-a; }
                else if (a >= cq) { e0=1.0; e1=0.0; } else { e0=0.0; e1=1.0; }
            } else {           // scalar: feature normal = grad direction (normal to iso-contour);
                e0=grad[((size_t)0*N+ci)*2+0]; e1=grad[((size_t)0*N+ci)*2+1]; dsen[ci]=1.0;
            }
            double nn=std::sqrt(e0*e0+e1*e1);
            if (nn > 1e-300) { dnrm[2*ci+0]=e0/nn; dnrm[2*ci+1]=e1/nn; }
            else { dnrm[2*ci+0]=1.0; dnrm[2*ci+1]=0.0; }
        }
    }
    // interior face values.
    #pragma omp parallel for
    for (int f = 0; f < Nf; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        if (n < 0) continue;
        double fx = m.face_centers[f*2+0], fy = m.face_centers[f*2+1];
        double dox = fx - cc[o*2+0], doy = fy - cc[o*2+1];
        double dnx = fx - cc[n*2+0], dny = fy - cc[n*2+1];
        // GENUINE T-MLP-u per-face geometry (variable-independent): t*, non-orthogonality
        // cosine, skew offsets, and the feature-aware compression factors comp_o/comp_n.
        double tso=0,tsn=0,eox=0,eoy=0,cosno=1,dfcox=0,dfcoy=0,dfcnx=0,dfcny=0;
        double co_o=1.0,co_n=1.0,dlen=1.0,dcx=0,dcy=0;
        if (USE_TMLPU || USE_SKEW) {
            dcx = cc[n*2+0]-cc[o*2+0]; dcy = cc[n*2+1]-cc[o*2+1];
            dlen = std::sqrt(std::max(dcx*dcx+dcy*dcy, 1e-30)); eox=dcx/dlen; eoy=dcy/dlen;
            double nx=m.face_normals[f*2+0], ny=m.face_normals[f*2+1];
            double nd=dcx*nx+dcy*ny, nds=std::fabs(nd)>1e-30?nd:(nd>=0?1e-30:-1e-30);
            tso=(dox*nx+doy*ny)/nds; tso=tso<0?0:(tso>1?1:tso);
            tsn=(dnx*(-nx)+dny*(-ny))/nds; tsn=tsn<0?0:(tsn>1?1:tsn);
            cosno=eox*nx+eoy*ny;
            dfcox=dox-tso*dcx; dfcoy=doy-tso*dcy;            // skew offset (owner)
            dfcnx=dnx-tsn*(-dcx); dfcny=dny-tsn*(-dcy);      // skew offset (neighbour)
            if (feat) {   // tensor-limiter: feature-aware compression modulation (needs dnrm)
                double a_o=eox*dnrm[2*o+0]+eoy*dnrm[2*o+1];
                co_o=1.0-TSTAR_GATE*dsen[o]*a_o*a_o;
                double a_n=eox*dnrm[2*n+0]+eoy*dnrm[2*n+1];
                co_n=1.0-TSTAR_GATE*dsen[n]*a_n*a_n;
            }
        }
        for (int v = 0; v < nvar; ++v) {
            double go0 = grad[((size_t)v*N+o)*2+0], go1 = grad[((size_t)v*N+o)*2+1];
            double gn0 = grad[((size_t)v*N+n)*2+0], gn1 = grad[((size_t)v*N+n)*2+1];
            double wo_ = W[(size_t)v*N+o], wn_ = W[(size_t)v*N+n];
            double wl, wr;
            if (USE_TMLPU) {
                double dplus = wn_ - wo_;
                double beta = std::min(1.0, std::max(cosno,0.0)/THETA_MIN);
                // owner side
                double gbx=(1.0-tso)*go0+tso*gn0, gby=(1.0-tso)*go1+tso*gn1;
                double cr=beta*((gbx*eox+gby*eoy) - dplus/dlen);
                double gcx=gbx-cr*eox, gcy=gby-cr*eoy;
                wl = wo_ + phi[(size_t)v*N+o]*(co_o*tso*dplus + gcx*dfcox+gcy*dfcoy);
                // neighbour side (e_n=-e_o, dplus_n=-dplus, same cosno/beta)
                double dpn=-dplus;
                double gbnx=(1.0-tsn)*gn0+tsn*go0, gbny=(1.0-tsn)*gn1+tsn*go1;
                double crn=beta*((gbnx*(-eox)+gbny*(-eoy)) - dpn/dlen);
                double gcnx=gbnx+crn*eox, gcny=gbny+crn*eoy;
                wr = wn_ + phi[(size_t)v*N+n]*(co_n*tsn*dpn + gcnx*dfcnx+gcny*dfcny);
            } else if (USE_SKEW) {
                // non-orthogonal/skew correction, NO t* compression (consistent, 2nd-order)
                double cs = (wn_ - wo_)/dlen;
                double beta = std::min(1.0, std::max(cosno,0.0)/THETA_MIN);
                double cro = beta*((go0*eox+go1*eoy) - cs);          // owner
                double gcox=go0-cro*eox, gcoy=go1-cro*eoy;
                wl = wo_ + phi[(size_t)v*N+o]*(gcox*dox+gcoy*doy);
                double crn = beta*((gn0*(-eox)+gn1*(-eoy)) + cs);    // neighbour (e_n=-e_o, cs_n=-cs)
                double gcnx=gn0+crn*eox, gcny=gn1+crn*eoy;
                wr = wn_ + phi[(size_t)v*N+n]*(gcnx*dnx+gcny*dny);
            } else {
                wl = wo_ + phi[(size_t)v*N+o]*(go0*dox+go1*doy);
                wr = wn_ + phi[(size_t)v*N+n]*(gn0*dnx+gn1*dny);
            }
            if (face_bound) {
                const double eps = 1e-30;
                double co = W[(size_t)v*N+o], del = wl - co;
                double al = del >= 0 ? smax[(size_t)v*N+o]-co : co-smin[(size_t)v*N+o];
                double th = std::fabs(del) > eps ? std::max(al,0.0)/std::max(std::fabs(del),eps) : 1.0;
                th = th < 0 ? 0 : (th > 1 ? 1 : th); wl = co + th*del;
                double cn = W[(size_t)v*N+n]; del = wr - cn;
                al = del >= 0 ? smax[(size_t)v*N+n]-cn : cn-smin[(size_t)v*N+n];
                th = std::fabs(del) > eps ? std::max(al,0.0)/std::max(std::fabs(del),eps) : 1.0;
                th = th < 0 ? 0 : (th > 1 ? 1 : th); wr = cn + th*del;
            }
            if (USE_DIR2 && feat && (v == 1 || v == 2)) {   // velocity: shear-gated bound relax
                double ll=smin[(size_t)v*N+o], hh=smax[(size_t)v*N+o];
                double loo = wo_ + (go0*dox+go1*doy); loo = loo<ll?ll:(loo>hh?hh:loo);
                wl += DIR2K*dsen[o]*(loo - wl);
                double lln=smin[(size_t)v*N+n], hhn=smax[(size_t)v*N+n];
                double lon = wn_ + (gn0*dnx+gn1*dny); lon = lon<lln?lln:(lon>hhn?hhn:lon);
                wr += DIR2K*dsen[n]*(lon - wr);
            }
            if (USE_DIR && feat) {
                bool pos = nvar >= 4 && (v == 0 || v == nvar-1);   // rho, p get the positivity floor (Euler only)
                wl = dir_relax_face(wl, go0, go1, dox, doy, dnrm[2*o+0], dnrm[2*o+1],
                                    phi[(size_t)v*N+o], dsen[o], DKAPPA,
                                    W[(size_t)v*N+o], pos, DFLOOR);
                wr = dir_relax_face(wr, gn0, gn1, dnx, dny, dnrm[2*n+0], dnrm[2*n+1],
                                    phi[(size_t)v*N+n], dsen[n], DKAPPA,
                                    W[(size_t)v*N+n], pos, DFLOOR);
                if (DIR_CLIP) {
                    double lo=smin[(size_t)v*N+o], hi=smax[(size_t)v*N+o];
                    wl = wl<lo?lo:(wl>hi?hi:wl);
                    double lon=smin[(size_t)v*N+n], hin=smax[(size_t)v*N+n];
                    wr = wr<lon?lon:(wr>hin?hin:wr);
                }
            }
            if (USE_REC) {
                double g = RECG;
                if (REC_GATE > 0.0) {   // smoothness gate: recovery OFF at discontinuities.
                    double ro=W[(size_t)0*N+o], rn=W[(size_t)0*N+n];
                    double rj=std::fabs(ro-rn)/(ro+rn+1e-30);   // density jump (contacts)
                    if (nvar >= 4) {                            // + pressure jump (shocks)
                        double po=W[(size_t)(nvar-1)*N+o], pn=W[(size_t)(nvar-1)*N+n];
                        double pj=std::fabs(po-pn)/(po+pn+1e-30); if (pj>rj) rj=pj;
                    }
                    double gate=1.0-rj/REC_GATE;
                    g *= (gate<0?0.0:(gate>1?1.0:gate));
                }
                double wL1 = W[(size_t)v*N+o] + (go0*dox+go1*doy);  // unlimited owner
                double wR1 = W[(size_t)v*N+n] + (gn0*dnx+gn1*dny);  // unlimited neighbour
                double wrec = 0.5*(wL1 + wR1);                      // continuous recovered value
                wl += g*(wrec - wl); wr += g*(wrec - wr);           // g=0 baseline, g=1 zero-jump
                if (REC_CLIP) {                                     // re-bound to neighbour range
                    double lo=smin[(size_t)v*N+o], hi=smax[(size_t)v*N+o];
                    wl = wl<lo?lo:(wl>hi?hi:wl);
                    double lon=smin[(size_t)v*N+n], hin=smax[(size_t)v*N+n];
                    wr = wr<lon?lon:(wr>hin?hin:wr);
                }
            }
            if (USE_PFLOOR && nvar >= 4 && (v == 0 || v == nvar-1)) {
                double lo_o = PFLOOR*W[(size_t)v*N+o]; if (wl < lo_o) wl = lo_o;
                double lo_n = PFLOOR*W[(size_t)v*N+n]; if (wr < lo_n) wr = lo_n;
            }
            W_L[(size_t)v*Nf+f] = wl; W_R[(size_t)v*Nf+f] = wr;
        }
    }
}

} // namespace cfd
