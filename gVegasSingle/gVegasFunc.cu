#include "vegasconst.h"

__device__
float func(float* rx, float wgt)
{

   float value = 0.f;
   for (int i=0;i<g_ndim;i++) {
      value += rx[i];
   }
   return cosf(value);

}

/*
{

   float value = 1.f;
   for (int i=0;i<g_ndim;i++) {
      value *= 2.f*rx[i];
   }
   return value;

}
*/