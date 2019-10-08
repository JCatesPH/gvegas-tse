#pragma once

__device__ __constant__ int g_ndim;
__device__ __constant__ int g_ng;
__device__ __constant__ int g_npg;
__device__ __constant__ int g_nd;

__device__ __constant__ double g_xjac;
__device__ __constant__ double g_dxg;
__device__ __constant__ double g_xl[ndim_max];
__device__ __constant__ double g_dx[ndim_max];
__device__ __constant__ double g_xi[ndim_max][nd_max];

__device__ __constant__ unsigned g_nCubes;


//----------------------------------
//  Set the parameters on the GPU.
//----------------------------------
__device__ __constant__ float mu;
__device__ __constant__ float hOmg;
__device__ __constant__ float a;
__device__ __constant__ float t0;
__device__ __constant__ float eA0a;
__device__ __constant__ float Gamm;
__device__ __constant__ float j0c;
__device__ __constant__ float j1c;
__device__ __constant__ float Gammsq;
__device__ __constant__ int   N;
__device__ __constant__ int   N2;

__device__ __constant__ float V0; 
__device__ __constant__ float V1; 
__device__ __constant__ float Fac; 
__device__ __constant__ float qx;  
__device__ __constant__ float qy; 
//----------------------------------