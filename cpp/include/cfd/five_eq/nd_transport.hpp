// Periodic prescribed-velocity 2-D alpha transport, port of nd_transport.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace cfd::five_eq {
struct AlphaGrid2 { int nx=0, ny=0; std::vector<double> value; };
struct FaceVelocity2 { std::vector<double> ux, uy; }; // ux:(nx+1)*ny, uy:nx*(ny+1)
struct AlphaTransportInfo { int steps=0; double dt_last=0.; };
inline int alpha_idx(int i,int j,int ny) { return i*ny+j; }
inline double alpha_limiter(double dm,double dp,bool superbee=true) {
    if(dm*dp<=0.) return 0.; const double s=dm>0.?1.:-1.;
    if(!superbee) return s*std::min(std::fabs(dm),std::fabs(dp));
    return s*std::max(std::min(2.*std::fabs(dm),std::fabs(dp)),std::min(std::fabs(dm),2.*std::fabs(dp)));
}
inline std::vector<double> alpha_rhs_periodic(const AlphaGrid2& a,const FaceVelocity2& v,double dx,double dy,bool superbee=true) {
    const int nx=a.nx,ny=a.ny; std::vector<double> sx(nx*ny),sy(nx*ny),fx((nx+1)*ny),fy(nx*(ny+1)),rhs(nx*ny);
    for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) { const int im=(i+nx-1)%nx,ip=(i+1)%nx,jm=(j+ny-1)%ny,jp=(j+1)%ny;
        sx[alpha_idx(i,j,ny)]=alpha_limiter(a.value[alpha_idx(i,j,ny)]-a.value[alpha_idx(im,j,ny)],a.value[alpha_idx(ip,j,ny)]-a.value[alpha_idx(i,j,ny)],superbee);
        sy[alpha_idx(i,j,ny)]=alpha_limiter(a.value[alpha_idx(i,j,ny)]-a.value[alpha_idx(i,jm,ny)],a.value[alpha_idx(i,jp,ny)]-a.value[alpha_idx(i,j,ny)],superbee); }
    for(int f=0;f<=nx;++f) for(int j=0;j<ny;++j) { const int il=(f+nx-1)%nx,ir=f%nx; const double al=a.value[alpha_idx(il,j,ny)]+.5*sx[alpha_idx(il,j,ny)], ar=a.value[alpha_idx(ir,j,ny)]-.5*sx[alpha_idx(ir,j,ny)];
        const double lo=std::min(a.value[alpha_idx(il,j,ny)],a.value[alpha_idx(ir,j,ny)]),hi=std::max(a.value[alpha_idx(il,j,ny)],a.value[alpha_idx(ir,j,ny)]); const double u=v.ux[f*ny+j]; fx[f*ny+j]=u*(u>=0.?std::clamp(al,lo,hi):std::clamp(ar,lo,hi)); }
    for(int i=0;i<nx;++i) for(int f=0;f<=ny;++f) { const int jl=(f+ny-1)%ny,jr=f%ny; const double al=a.value[alpha_idx(i,jl,ny)]+.5*sy[alpha_idx(i,jl,ny)], ar=a.value[alpha_idx(i,jr,ny)]-.5*sy[alpha_idx(i,jr,ny)];
        const double lo=std::min(a.value[alpha_idx(i,jl,ny)],a.value[alpha_idx(i,jr,ny)]),hi=std::max(a.value[alpha_idx(i,jl,ny)],a.value[alpha_idx(i,jr,ny)]); const double u=v.uy[i*(ny+1)+f]; fy[i*(ny+1)+f]=u*(u>=0.?std::clamp(al,lo,hi):std::clamp(ar,lo,hi)); }
    for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) rhs[alpha_idx(i,j,ny)]=-(fx[(i+1)*ny+j]-fx[i*ny+j])/dx-(fy[i*(ny+1)+j+1]-fy[i*(ny+1)+j])/dy;
    return rhs;
}
inline void alpha_clip_preserve_sum(std::vector<double>& a,double lo,double hi,double target) {
    for(auto& x:a) x=std::clamp(x,lo,hi);
    for(int it=0;it<8;++it) { const double d=target; double sum=0.; for(double x:a) sum+=x; const double diff=d-sum; if(std::fabs(diff)<=1.e-13*std::fmax(1.,std::fabs(target))) break;
        double cap=0.; for(double x:a) cap+=diff>0.?hi-x:x-lo; if(cap<=0.) break;
        for(auto& x:a) x=std::clamp(x+diff*(diff>0.?hi-x:x-lo)/cap,lo,hi); }
}
inline AlphaGrid2 solve_alpha_transport_2d(AlphaGrid2 a,double dx,double dy,double t_end,
    const std::function<FaceVelocity2(double)>& velocity,double cfl=.45,double dt_fixed=-1.,bool superbee=true,double lo=0.,double hi=1.,AlphaTransportInfo* info=nullptr) {
    double t=0.,last=0.,mass=0.; for(double x:a.value) mass+=x; int steps=0;
    auto add=[&](const std::vector<double>& x,const std::vector<double>& r,double s){std::vector<double> o=x; for(std::size_t i=0;i<o.size();++i)o[i]+=s*r[i]; return o;};
    while(t<t_end-1.e-15) { const auto v0=velocity(t); double vmax=1.e-14; for(double x:v0.ux)vmax=std::fmax(vmax,std::fabs(x)); for(double x:v0.uy)vmax=std::fmax(vmax,std::fabs(x)); const double dt=std::min(dt_fixed>0.?dt_fixed:cfl*std::min(dx,dy)/vmax,t_end-t);
        const auto k0=alpha_rhs_periodic(a,v0,dx,dy,superbee); AlphaGrid2 a1=a; a1.value=add(a.value,k0,dt); alpha_clip_preserve_sum(a1.value,lo,hi,mass);
        const auto k1=alpha_rhs_periodic(a1,velocity(t+dt),dx,dy,superbee); auto raw=add(a1.value,k1,dt); alpha_clip_preserve_sum(raw,lo,hi,mass); AlphaGrid2 a2=a; for(std::size_t q=0;q<raw.size();++q)a2.value[q]=.75*a.value[q]+.25*raw[q]; alpha_clip_preserve_sum(a2.value,lo,hi,mass);
        const auto k2=alpha_rhs_periodic(a2,velocity(t+.5*dt),dx,dy,superbee); raw=add(a2.value,k2,dt); alpha_clip_preserve_sum(raw,lo,hi,mass); for(std::size_t q=0;q<raw.size();++q)a.value[q]=a.value[q]/3.+2.*raw[q]/3.; alpha_clip_preserve_sum(a.value,lo,hi,mass); t+=dt;last=dt;++steps;
    } if(info)*info={steps,last}; return a;
}
} // namespace cfd::five_eq
