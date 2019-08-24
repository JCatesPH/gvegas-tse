//  ======<< vegasconst.h >>======

#ifndef VEGASCONST_H
#define VEGASCONST_H

__device__ __constant__ int g_ndim;
__device__ __constant__ int g_ng;
__device__ __constant__ int g_npg;
__device__ __constant__ int g_nd;

__device__ __constant__ float g_xjac;
__device__ __constant__ float g_dxg;
__device__ __constant__ float g_xl[ndim_max];
__device__ __constant__ float g_dx[ndim_max];
__device__ __constant__ float g_xi[ndim_max][nd_max];

__device__ __constant__ unsigned g_nCubes;

//TEST FUNCTIONS VARIABLES
__device__ float move[ndim_max]; //Goes from 0 to 1 in every variable.
__device__ float offset[ndim_max]; //Goes from 0 to 1 in every variable but can be renormalized to change "difficulty".


/*-------- constants for chi ---------*/
__device__ __constant__ float mu     = 0.1f;
__device__ __constant__ float hOmg   = 0.3f;
__device__ __constant__ float a      = 3.6f;
__device__ __constant__ float A      = 4.f;
__device__ __constant__ float rati   = 0.1;
__device__ __constant__ float eE0    = 0.00711512; // rati * (hOmg * hOmg) / (2 * sqrt(A * mu))
__device__ __constant__ float Gamm   = 0.003;
__device__ __constant__ float KT     = 1e-6;
__device__ __constant__ float shift  = 0.00225; // A * (eE0 / hOmg) * (eE0 / hOmg)
__device__ __constant__ float Gammsq = 9e-6; // Gamm * Gamm
__device__ __constant__ int   N      = 3;


#endif

