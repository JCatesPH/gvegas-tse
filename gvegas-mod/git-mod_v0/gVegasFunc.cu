#include "vegasconst.h"

__device__
double func(float* rx, double wgt)
{
   double value = 0.;
   for (int i=0;i<g_ndim;i++) {
      value += 2.*rx[i];
   }
   return value;

}


