#include "vegasconst.h"

// This file contains the function you wish to be integrated with the Monte Carlo method.
__global__
void func(float* rx, double wgt)
{
   for (int i=0;i<g_ndim;i++) {
      value += 2.*rx[i];
   }

   return 0;
}