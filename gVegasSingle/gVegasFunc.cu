#include "vegasconst.h"

__device__
float func(float* rx, float wgt)
{

   float value = 0.f;
   float cosval = 0.f;

   for (int i=0;i<7;i++) {
      value += rx[i];
   }
   
   cosval = __cosf(value);

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