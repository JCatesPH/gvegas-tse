#include "vegasconst.h"

__device__
float func(float* rx, float wgt)
{

   float value = 0.f;
   float cosval = 0.f;

   /*
   for (int i=0;i<7;i++) {
      value += rx[i];
   }
   */

   value = rx[0] + rx[1] + rx[2] + rx[3] + rx[4] + rx[5] + rx[6];
   cosval = cosf(value);

   return cosval;

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