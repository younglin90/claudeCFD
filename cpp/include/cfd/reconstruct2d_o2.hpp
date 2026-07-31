// cfd/reconstruct2d_o2.hpp — order-2 (quadratic) WLSQ reconstruction.
// Per cell fit W ~ W_c + gx*dx + gy*dy + 0.5*hxx*dx^2 + 0.5*hyy*dy^2 + hxy*dx*dy
// over the (1-ring) vertex neighbour stencil (>=5 neighbours on criss-cross).
// The LSQ operator M = (A^T A)^-1 A^T (5 x maxnb) is geometry-only -> precomputed
// once; coeffs = M . (W_nb - W_c). Defining property (unit-tested): EXACT on any
// quadratic field. High-order base for a genuinely-convergent T-MLP-u (the MLP
// vertex limiter would wrap this next).
#pragma once
#include "cfd/mesh.hpp"
#include <vector>
#include <cmath>
#include <set>
#include <array>
#include <algorithm>

namespace cfd {

struct ReconCtxO2 {
    int N = 0, max_nb = 0, Nn = 0, max_v = 0, max_v2c = 0;
    std::vector<int>    nb;   // N*max_nb (-1 pad)
    std::vector<double> M;    // N*5*max_nb : coeff operator (gx,gy,hxx,hyy,hxy)
    std::vector<int>    sample_vid;   // N*max_v node ids (cell vertices)
    std::vector<double> sample_off;   // N*max_v*2 vertex offsets from cell centre
    std::vector<int>    v2c;          // Nn*max_v2c cells sharing each node
};

// invert a 5x5 (Gauss-Jordan, partial pivot); returns false if singular.
inline bool inv5(double A[5][5], double Inv[5][5]) {
    double M[5][10];
    for (int i=0;i<5;++i){ for(int j=0;j<5;++j){M[i][j]=A[i][j]; M[i][5+j]=(i==j)?1.0:0.0;} }
    for (int c=0;c<5;++c){
        int piv=c; double best=std::fabs(M[c][c]);
        for(int r=c+1;r<5;++r){ if(std::fabs(M[r][c])>best){best=std::fabs(M[r][c]);piv=r;} }
        if(best<1e-300) return false;
        if(piv!=c) for(int j=0;j<10;++j) std::swap(M[c][j],M[piv][j]);
        double d=M[c][c]; for(int j=0;j<10;++j) M[c][j]/=d;
        for(int r=0;r<5;++r){ if(r==c)continue; double f=M[r][c];
            for(int j=0;j<10;++j) M[r][j]-=f*M[c][j]; }
    }
    for(int i=0;i<5;++i)for(int j=0;j<5;++j) Inv[i][j]=M[i][5+j];
    return true;
}

inline ReconCtxO2 build_recon_ctx_o2(const Mesh& m) {
    ReconCtxO2 c; const int N=m.n_cells(); c.N=N;
    const double* cc=m.cell_centers.data();
    int Nn=(int)m.nodes.size()/2;
    std::vector<std::vector<int>> vcells(Nn);
    for(int ci=0;ci<N;++ci) for(int v:m.cell_nodes[ci]) vcells[v].push_back(ci);
    std::vector<std::vector<int>> nbl(N);
    for(int ci=0;ci<N;++ci){ std::set<int> s;
        for(int v:m.cell_nodes[ci]) for(int c2:vcells[v]) if(c2!=ci) s.insert(c2);
        nbl[ci].assign(s.begin(),s.end()); c.max_nb=std::max(c.max_nb,(int)nbl[ci].size()); }
    c.max_nb=std::max(c.max_nb,5);
    c.nb.assign((size_t)N*c.max_nb,-1);
    c.M.assign((size_t)N*5*c.max_nb,0.0);
    for(int ci=0;ci<N;++ci){
        int K=(int)nbl[ci].size();
        // weighted rows a_k = [dx,dy,.5dx^2,.5dy^2,dx*dy] * sqrt(w); w=1/dist^2
        std::vector<std::array<double,5>> a(K); std::vector<double> sw(K);
        for(int k=0;k<K;++k){ int nb=nbl[ci][k]; c.nb[(size_t)ci*c.max_nb+k]=nb;
            double dx=cc[nb*2+0]-cc[ci*2+0], dy=cc[nb*2+1]-cc[ci*2+1];
            double w=1.0/std::max(dx*dx+dy*dy,1e-30); double s=std::sqrt(w); sw[k]=s;
            a[k]={dx*s,dy*s,0.5*dx*dx*s,0.5*dy*dy*s,dx*dy*s}; }
        double ATA[5][5]={{0}}, Inv[5][5];
        for(int k=0;k<K;++k)for(int i=0;i<5;++i)for(int j=0;j<5;++j) ATA[i][j]+=a[k][i]*a[k][j];
        if(!inv5(ATA,Inv)) continue; // leave M=0 (degenerate -> first-order)
        // M = Inv * A^T (with weights): coeff_i = sum_k (Inv*a_k)_i * sw_k * dW_k_raw
        // store per neighbour: M[i,k] = (sum_j Inv[i][j]*a[k][j]) * sw[k]
        for(int k=0;k<K;++k)for(int i=0;i<5;++i){ double v=0;
            for(int j=0;j<5;++j) v+=Inv[i][j]*a[k][j];
            c.M[((size_t)ci*5+i)*c.max_nb+k]=v*sw[k]; }
    }
    // vertex data for the MLP limiter
    c.Nn=Nn;
    for(int ci=0;ci<N;++ci) c.max_v=std::max(c.max_v,(int)m.cell_nodes[ci].size());
    c.max_v=std::max(c.max_v,1);
    c.sample_vid.assign((size_t)N*c.max_v,-1);
    c.sample_off.assign((size_t)N*c.max_v*2,0.0);
    for(int ci=0;ci<N;++ci){ const auto& vs=m.cell_nodes[ci];
        for(int k=0;k<(int)vs.size();++k){ int v=vs[k]; c.sample_vid[(size_t)ci*c.max_v+k]=v;
            c.sample_off[((size_t)ci*c.max_v+k)*2+0]=m.nodes[v*2+0]-cc[ci*2+0];
            c.sample_off[((size_t)ci*c.max_v+k)*2+1]=m.nodes[v*2+1]-cc[ci*2+1]; } }
    for(auto& vc:vcells) c.max_v2c=std::max(c.max_v2c,(int)vc.size());
    c.max_v2c=std::max(c.max_v2c,1);
    c.v2c.assign((size_t)Nn*c.max_v2c,-1);
    for(int v=0;v<Nn;++v) for(int k=0;k<(int)vcells[v].size();++k)
        c.v2c[(size_t)v*c.max_v2c+k]=vcells[v][k];
    return c;
}

// MLP-limited order-2: scale the quadratic increment by phi in [0,1] so the
// reconstructed value at every cell vertex stays within the vertex min/max of
// vertex-sharing cells (no new extrema -> shock-capturing, bounded). phi=1 where
// smooth (stays exact on quadratics that respect the bounds).
// shear (size N, in [0,1]) + krelax: where shear is high (slip line / KH layer,
// NOT shocks), relax the MLP vertex bound by krelax*shear*(local range) so the
// quadratic is less clipped -> lower numerical diffusion on the shear layer while
// shocks (shear~0) keep the tight positivity-preserving bound. shear=nullptr ->
// classic strict MLP (unchanged).
// tvbM: Venkatakrishnan/TVB smoothness relaxation — the vertex bound is widened by
// tvbM*h^2 (h^2 ~ cell area), so smooth extrema (curvature <= tvbM) are NOT clipped
// -> recovers the quadratic's high order at smooth peaks (e.g. a vortex core) while
// genuine discontinuities (jump >> h^2) stay limited. Resolution-consistent.
inline void reconstruct_o2_limited(const Mesh& m, const ReconCtxO2& c,
                                   const std::vector<double>& W, int nvar, int vi,
                                   std::vector<double>& WLf, std::vector<double>& WRf,
                                   const double* shear = nullptr, double krelax = 0.0,
                                   double tvbM = 0.0, double venkatK = 0.0,
                                   bool hier = false) {
    const int N=m.n_cells(), Nf=m.n_faces();
    const double* cc=m.cell_centers.data();
    std::vector<double> g((size_t)N*5,0.0), phi(N,1.0), phi2(N,1.0);
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci){ double wc=W[(size_t)vi*N+ci]; double co[5]={0,0,0,0,0};
        for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double dW=W[(size_t)vi*N+nb]-wc;
            for(int i=0;i<5;++i) co[i]+=c.M[((size_t)ci*5+i)*c.max_nb+k]*dW; }
        for(int i=0;i<5;++i) g[(size_t)ci*5+i]=co[i]; }
    std::vector<double> vmn((size_t)c.Nn), vmx((size_t)c.Nn);
    #pragma omp parallel for
    for(int v=0;v<c.Nn;++v){ double mn=1e300,mx=-1e300;
        for(int k=0;k<c.max_v2c;++k){ int ci=c.v2c[(size_t)v*c.max_v2c+k]; if(ci<0)continue;
            double val=W[(size_t)vi*N+ci]; if(val<mn)mn=val; if(val>mx)mx=val; }
        if(mn>mx){mn=0;mx=0;} vmn[v]=mn; vmx[v]=mx; }
    const double eps=1e-30;
    const double venkat_eps2 = venkatK>0.0 ? std::pow(venkatK, 3) : 0.0;  // *h^3 per cell
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci){ double wc=W[(size_t)vi*N+ci]; const double* G=&g[(size_t)ci*5];
        double mrg = shear ? (krelax * shear[ci]) : 0.0;   // shear bound-relax fraction
        double rtvb = tvbM * 2.0 * m.cell_volumes[ci];     // TVB margin ~ tvbM*h^2
        double h = std::sqrt(2.0*m.cell_volumes[ci]);      // cell length scale
        double e2 = venkat_eps2 * h*h*h;                   // Venkatakrishnan eps^2=(K h)^3
        if (hier) {
            // Hierarchical MLP (high-order MLP-u2/u3 style): keep the gradient
            // (phi1) and add as much of the quadratic (phi2) as the MLP vertex
            // bound allows; if the linear part itself violates, drop quadratic and
            // limit the gradient. -> phi1=phi2=1 in smooth regions (3rd order),
            // phi2->0 then phi1->0 at discontinuities.
            bool lin_ok = true; double p1 = 1.0, p2 = 1.0;
            for(int k=0;k<c.max_v;++k){ int v=c.sample_vid[(size_t)ci*c.max_v+k]; if(v<0)continue;
                double dx=c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy=c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                double Lv=G[0]*dx+G[1]*dy;
                double lo=vmn[v]-wc, hi=vmx[v]-wc, tol=eps*(std::fabs(hi)+std::fabs(lo)+1.0);
                if(Lv>hi+tol || Lv<lo-tol){ lin_ok=false; break; } }
            if (lin_ok) {
                for(int k=0;k<c.max_v;++k){ int v=c.sample_vid[(size_t)ci*c.max_v+k]; if(v<0)continue;
                    double dx=c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy=c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                    double Lv=G[0]*dx+G[1]*dy, Qv=0.5*G[2]*dx*dx+0.5*G[3]*dy*dy+G[4]*dx*dy;
                    double lo=vmn[v]-wc, hi=vmx[v]-wc;
                    double lim = 1.0;                       // max phi2 keeping Lv+phi2*Qv in [lo,hi]
                    if (Qv > eps)       lim = (hi - Lv)/Qv;
                    else if (Qv < -eps) lim = (lo - Lv)/Qv;
                    if (lim < p2) p2 = lim; }
                p1 = 1.0; if(p2<0)p2=0; if(p2>1)p2=1;
            } else {
                p2 = 0.0;                                   // troubled: linear-only, limited
                for(int k=0;k<c.max_v;++k){ int v=c.sample_vid[(size_t)ci*c.max_v+k]; if(v<0)continue;
                    double dx=c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy=c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                    double Lv=G[0]*dx+G[1]*dy;
                    double allowed = Lv>=0.0 ? (vmx[v]-wc) : (wc-vmn[v]);
                    double pk = (std::fabs(Lv)>eps) ? std::max(allowed,0.0)/std::max(std::fabs(Lv),eps) : 1.0;
                    if(pk<p1) p1=pk; }
                if(p1<0)p1=0; if(p1>1)p1=1;
            }
            phi[ci]=p1; phi2[ci]=p2;
        } else {
            double p=1.0;
            for(int k=0;k<c.max_v;++k){ int v=c.sample_vid[(size_t)ci*c.max_v+k]; if(v<0)continue;
                double dx=c.sample_off[((size_t)ci*c.max_v+k)*2+0], dy=c.sample_off[((size_t)ci*c.max_v+k)*2+1];
                double delta=G[0]*dx+G[1]*dy+0.5*G[2]*dx*dx+0.5*G[3]*dy*dy+G[4]*dx*dy;
                double pk;
                if (venkatK > 0.0) {
                    double a = delta>=0.0 ? (vmx[v]-wc) : (vmn[v]-wc), b = delta;
                    pk = (std::fabs(b)>eps)
                       ? (a*a + 2.0*a*b + e2) / (a*a + a*b + 2.0*b*b + e2) : 1.0;
                } else {
                    double r = mrg * (vmx[v]-vmn[v]) + rtvb;
                    double allowed = delta>=0.0 ? (vmx[v]+r-wc) : (wc-(vmn[v]-r));
                    pk = (std::fabs(delta)>eps) ? std::max(allowed,0.0)/std::max(std::fabs(delta),eps) : 1.0;
                }
                if(pk<p) p=pk; }
            phi[ci]=p<0?0:(p>1?1:p); phi2[ci]=phi[ci];
        } }
    WLf.assign(Nf,0.0); WRf.assign(Nf,0.0);
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f], n=m.face_neighbour[f];
        double fx=m.face_centers[f*2+0], fy=m.face_centers[f*2+1];
        auto val=[&](int ci){ double dx=fx-cc[ci*2+0], dy=fy-cc[ci*2+1]; const double* G=&g[(size_t)ci*5];
            double Lp=G[0]*dx+G[1]*dy, Qp=0.5*G[2]*dx*dx+0.5*G[3]*dy*dy+G[4]*dx*dy;
            return W[(size_t)vi*N+ci]+phi[ci]*Lp+phi2[ci]*Qp; };
        WLf[f]=val(o); WRf[f]=(n>=0)?val(n):val(o); }
}

// reconstruct scalar var 'vi' (of nvar) face values into W_L/W_R (size Nf each).
inline void reconstruct_o2_scalar(const Mesh& m, const ReconCtxO2& c,
                                  const std::vector<double>& W, int nvar, int vi,
                                  std::vector<double>& WLf, std::vector<double>& WRf) {
    const int N=m.n_cells(), Nf=m.n_faces();
    const double* cc=m.cell_centers.data();
    std::vector<double> g((size_t)N*5,0.0);
    #pragma omp parallel for
    for(int ci=0;ci<N;++ci){ double wc=W[(size_t)vi*N+ci];
        double co[5]={0,0,0,0,0};
        for(int k=0;k<c.max_nb;++k){ int nb=c.nb[(size_t)ci*c.max_nb+k]; if(nb<0)continue;
            double dW=W[(size_t)vi*N+nb]-wc;
            for(int i=0;i<5;++i) co[i]+=c.M[((size_t)ci*5+i)*c.max_nb+k]*dW; }
        for(int i=0;i<5;++i) g[(size_t)ci*5+i]=co[i]; }
    WLf.assign(Nf,0.0); WRf.assign(Nf,0.0);
    for(int f=0;f<Nf;++f){ int o=m.face_owner[f], n=m.face_neighbour[f];
        double fx=m.face_centers[f*2+0], fy=m.face_centers[f*2+1];
        auto val=[&](int ci){ double dx=fx-cc[ci*2+0], dy=fy-cc[ci*2+1]; const double* G=&g[(size_t)ci*5];
            return W[(size_t)vi*N+ci]+G[0]*dx+G[1]*dy+0.5*G[2]*dx*dx+0.5*G[3]*dy*dy+G[4]*dx*dy; };
        WLf[f]=val(o); WRf[f]=(n>=0)?val(n):val(o); }
}

} // namespace cfd
