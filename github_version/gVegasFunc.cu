#include "vegasconst.h"
#define CUDART_PI_F 3.141592654f

__device__
float func(float* rx, float wgt)
{

   float value = 0.f;
   for (int i=0;i<g_ndim;i++) {
      value += rx[i]*rx[i];
   }
   return value;
}

__device__
float oscillate(float* rx, float wgt, float* dillate, float* offset)
{
   float value = 0.f;
   for (int i = 0; i < g_ndim; i++) {
      value += dillate[i] * rx[i];
   }
   value += CUDART_PI_F * offset[0];
   value = cosf(value);
   return value;
}

__device__
float prodpeak(float* rx, float wgt, float* move, float* offset)
{
   float value = 1.f;
   for (int i = 0; i < g_ndim; i++) {
      value *= 1.f / ((rx[i]-move[i])*(rx[i]-move[i]) + (1.f/offset[i])*(1.f/offset[i]));
   }
   return value;
}

__device__
float cornerpeak(float* rx, float wgt, float* offset)
{
   float value = 1.f;
   for (int i = 0; i < g_ndim; i++) {
      value += offset[i] * rx[i];
   }
   value = 1.f / powf(value, (float)(g_ndim+1));
   return value;
}

__device__
float gaussian(float* rx, float wgt, float* move, float* offset)
{
   float value = 0.f;
   for (int i = 0; i < g_ndim; i++) {
      value = value - offset[i]*offset[i]*(rx[i]-move[i])*(rx[i]-move[i]);
   }
   value = expf(value);
   return value;
}

__device__
float czerocont(float* rx, float wgt, float* move, float* offset)
{
   float value = 0.f;
   for (int i = 0; i < g_ndim; i++) {
      value = value - offset[i]*fabsf(rx[i]-move[i]);
   }
   value = expf(value);
   return value;
}

__device__
float discont(float* rx, float wgt, float* limit, float* offset)
{
   float value = 0.f;
   for (int i = 0; i < g_ndim; i++) {
      value += offset[i]*rx[i];
   }
   value = expf(value) * (float)((1-(rx[0] > limit[0]))*(1-(rx[1] > limit[0])));
   return value;
}
