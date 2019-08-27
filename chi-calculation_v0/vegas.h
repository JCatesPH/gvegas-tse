#ifndef VEGAS_H
#define VEGAS_H

void myVegas(float& avgi, float& sd, float& chi2a);

#ifndef __MAIN_LOGIC
#define EXTERN extern
#else
#define EXTERN
#endif

const int ndim_max = 20;
const float alph = 1.5;
EXTERN float dx[ndim_max];
EXTERN float randm[ndim_max];
const int nd_max = 50;
EXTERN double xin[nd_max];

EXTERN float xjac;

EXTERN float xl[ndim_max],xu[ndim_max];
EXTERN double acc;
EXTERN int ndim, ncall, itmx, nprn;

EXTERN float xi[ndim_max][nd_max];
EXTERN double si, si2, swgt, schi;
EXTERN int ndo, it;

//EXTERN double alph;
EXTERN int mds;

EXTERN double calls, ti, tsi;
//EXTERN float ti, tsi;

EXTERN int npg, ng, nd;
EXTERN float dxg, xnd;

EXTERN unsigned nCubes;

//adding stuff, don't know if it's right...

//----------------------------------
//  Set the parameters on the host.
//----------------------------------
EXTERN float mu_h;
EXTERN float hOmg_h;
EXTERN float hOmg2_h;
EXTERN float a_h;
EXTERN float A_h;
EXTERN float rati_h;
EXTERN float eE0_h;
EXTERN float Gamm_h;
EXTERN float KT_h;
EXTERN float shift_h;
EXTERN float Gammsq_h;
EXTERN int   N_h;
//----------------------------------

#undef EXTERN

#endif
