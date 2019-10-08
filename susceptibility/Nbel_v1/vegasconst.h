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
__device__ __constant__ float A;
__device__ __constant__ float rati;
__device__ __constant__ float eE0;
__device__ __constant__ float Gamm;
__device__ __constant__ float KT;
__device__ __constant__ float shift; // A * (eE0 / hOmg) * (eE0 / hOmg)
__device__ __constant__ float Gammsq; // Gamm * Gamm
__device__ __constant__ int   N;

__device__ __constant__ float V0; // eE0 * A / hOmg
__device__ __constant__ float V2; // A * (eE0 / hOmg) * (eE0 / hOmg)
__device__ __constant__ float Fac; // -(4.f * Gamm / (PI*PI))
__device__ __constant__ float qx;  //  0.01f + (PI / a) * i / 50.f
__device__ __constant__ float qy;  //  0.f
//----------------------------------