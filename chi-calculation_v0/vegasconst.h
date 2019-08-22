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

#endif

