#include "vegasconst.h"
#include "vegas.h"

__global__
void gVegasCallFunc(float* gFval, int* gIAval)
{
   //--------------------
   // Check the thread ID
   //--------------------
   const unsigned int tIdx  = threadIdx.x;
   const unsigned int bDimx = blockDim.x;

   const unsigned int bIdx  = blockIdx.x;
   const unsigned int gDimx = gridDim.x;
   const unsigned int bIdy  = blockIdx.y;
   //   const unsigned int gDimy = gridDim.y;

   unsigned int bid  = bIdy*gDimx+bIdx;
   const unsigned int tid = bid*bDimx+tIdx;

   //   int ipg = tid%g_npg;
   int ig = tid/g_npg;

   unsigned nCubeNpg = g_nCubes*g_npg;

   if (tid<nCubeNpg) {

      unsigned ia[ndim_max];
      
      unsigned int tidRndm = tid;
      
      int kg[ndim_max];
      
      unsigned igg = ig;
      for (int j=0;j<g_ndim;j++) {
         kg[j] = igg%g_ng+1;
         igg /= g_ng;
      }
      
      //            randa(g_ndim,randm);
      float randm[ndim_max];
      fxorshift128(tidRndm, g_ndim, randm);
      
      float x[ndim_max];
      
      float wgt = g_xjac;
      for (int j=0;j<g_ndim;j++) {
         float xo,xn,rc;
         xn = (kg[j]-randm[j])*g_dxg+1.f;
         ia[j] = (int)xn-1;
         if (ia[j]<=0) {
            xo = g_xi[j][ia[j]];
            rc = (xn-(float)(ia[j]+1))*xo;
         } else {
            xo = g_xi[j][ia[j]]-g_xi[j][ia[j]-1];
            rc = g_xi[j][ia[j]-1]+(xn-(float)(ia[j]+1))*xo;
         }
         x[j] = g_xl[j]+rc*g_dx[j];
         wgt *= xo*(float)g_nd;
      }
      
      float f = wgt * func(x,wgt);
      
      //      gFval[tid] = (float)typeFinal[2];
      gFval[tid] = f;
      for (int idim=0;idim<g_ndim;idim++) {
         gIAval[idim*nCubeNpg+tid] = ia[idim];
      }
   }

}
