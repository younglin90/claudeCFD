// Per-cell material-block Schur factors for the ARS acoustic Newton solve.
#pragma once
#include <array>
#include <cmath>
#include "cfd/primitive.hpp"
namespace cfd::five_eq {
struct SchurBlocks { std::array<std::array<double,3>,3> a_inv{}; std::array<double,3> au{},ap{},ua{},pa{}; double uu=0.,up=0.,pu=0.,pp=0.,sigma_pp=0.; };
inline bool invert3(std::array<std::array<double,3>,3> a,std::array<std::array<double,3>,3>& out){for(int i=0;i<3;++i)out[i][i]=1.;for(int c=0;c<3;++c){int p=c;for(int r=c+1;r<3;++r)if(std::fabs(a[r][c])>std::fabs(a[p][c]))p=r;if(std::fabs(a[p][c])<1.e-30)return false;if(p!=c){std::swap(a[p],a[c]);std::swap(out[p],out[c]);}const double d=a[c][c];for(int j=0;j<3;++j){a[c][j]/=d;out[c][j]/=d;}for(int r=0;r<3;++r)if(r!=c){const double f=a[r][c];for(int j=0;j<3;++j){a[r][j]-=f*a[c][j];out[r][j]-=f*out[c][j];}}}return true;}
inline SchurBlocks schur_blocks(const PrimW&W,const EOS&e1,const EOS&e2){double J[5][5];dUdW_analytic(W,e1,e2,J);SchurBlocks b;const int rows[3]={0,1,4};const int cols[3]={0,1,2};std::array<std::array<double,3>,3>A{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)A[i][j]=J[rows[i]][cols[j]];const double reg=1.e-14*std::fmax(std::fmax(std::fabs(A[0][0]),std::fabs(A[1][1])),1.);for(int i=0;i<3;++i)A[i][i]+=reg;if(!invert3(A,b.a_inv))return b;for(int i=0;i<3;++i){b.au[i]=J[rows[i]][3];b.ap[i]=J[rows[i]][4];b.ua[i]=J[2][cols[i]];b.pa[i]=J[3][cols[i]];}auto dotinv=[&](const std::array<double,3>&x,const std::array<double,3>&y){double z=0.;for(int i=0;i<3;++i)for(int j=0;j<3;++j)z+=x[i]*b.a_inv[i][j]*y[j];return z;};b.uu=J[2][3]-dotinv(b.ua,b.au);b.up=J[2][4]-dotinv(b.ua,b.ap);b.pu=J[3][3]-dotinv(b.pa,b.au);b.pp=J[3][4]-dotinv(b.pa,b.ap);b.sigma_pp=b.pp-b.pu*b.up/std::fmax(std::fabs(b.uu),1.e-30);return b;}
} // namespace cfd::five_eq
