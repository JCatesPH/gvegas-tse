#ifndef VEGAS_H
#define VEGAS_H

void gVegas(float& avgi, float& sd, float& chi2a);

#ifndef __MAIN_LOGIC
#define EXTERN extern
#else
#define EXTERN
#endif

const int ndim_max = 20;
const double alph = 1.5;
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

EXTERN int npg, ng, nd;
EXTERN float dxg, xnd;

EXTERN unsigned nCubes;

#undef EXTERN

#endif
