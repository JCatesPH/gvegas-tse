#pragma once

void gVegas(double& avgi, double& sd, double& chi2a);

#ifndef __MAIN_LOGIC
#define EXTERN extern
#else
#define EXTERN
#endif

const int ndim_max = 20;
const double alph = 1.5;
EXTERN double dx[ndim_max];
EXTERN double randm[ndim_max];
const int nd_max = 50;
EXTERN double xin[nd_max];

EXTERN double xjac;

EXTERN double xl[ndim_max],xu[ndim_max];
EXTERN double acc;
EXTERN int ndim, ncall, itmx, nprn;

EXTERN double xi[ndim_max][nd_max];
EXTERN double si, si2, swgt, schi;
EXTERN int ndo, it;

//EXTERN double alph;
EXTERN int mds;

EXTERN double calls, ti, tsi;

EXTERN int npg, ng, nd;
EXTERN double dxg, xnd;

EXTERN unsigned nCubes;

//----------------------------------
//  Set the parameters on the host.
//----------------------------------
EXTERN float mu_h;
EXTERN float hOmg_h;
EXTERN float a_h;
EXTERN float A_h;
EXTERN float rati_h;
EXTERN float eE0_h;
EXTERN float Gamm_h;
EXTERN float KT_h;
EXTERN float shift_h;
EXTERN float Gammsq_h;
EXTERN int   N_h;

EXTERN float V0_h;     
EXTERN float V2_h;    
EXTERN float Fac_h;   
EXTERN float qx_h;    
EXTERN float qy_h;     
//----------------------------------

#undef EXTERN

