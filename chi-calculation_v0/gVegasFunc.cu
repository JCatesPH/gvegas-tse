#include "vegasconst.h"

#include <cuComplex.h> // Complex number module of cuda.

#define CUDART_PI_F 3.141592654f

/*-------- constants for chi ---------*/
#define mu      0.1f
#define hOmg    0.3f
#define a       4.f
#define A       4.f
#define rati    0.3
#define eE0     rati * (hOmg * hOmg) / (2 * sqrt(A * mu))
#define Gamm    0.003
#define KT      1e-6
#define shift   A * (eE0 / hOmg) * (eE0 / hOmg)
#define Gammsq  Gamm * Gamm
#define N       3

/*-------- helpful macros ---------*/
#define SQ(x)  (x * x) // Squares the argument
#define CB(x)  (x * x * x) // Cubes the argument

__device__
float chi(float* rx, float wgt)
{
   float dds = 0.f;
   // ds = 0  // UNUSED
   float ek;
   float ekq;
   float xk;
   float xkq;

   // ek = A * (sqrt((rx[0]) ** 2 + (rx[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
   ek = A * hypotf(rx[0], rx[1]) * hypotf(rx[0], rx[1]) + A * SQ(eE0 / hOmg);

   // ekq = A * (sqrt((rx[0] + qx) ** 2 + (rx[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
   ekq = A * hypotf(rx[0] + rx[2], rx[1]) * hypotf(rx[0] + rx[2], rx[1]) + A * SQ(eE0 / hOmg);

   // xk = 2 * A * eE0 * sqrt((rx[0]) ** 2 + (rx[1]) ** 2) / hOmg ** 2
   xk = 2 * A * eE0 * hypotf(rx[0], rx[1]) / SQ(hOmg);

   // xkq = 2 * A * eE0 * sqrt((rx[0] + qx) ** 2 + (rx[1] + 0) ** 2) / hOmg ** 2
   xkq = 2 * A * eE0 * hypotf(rx[0] + rx[2], rx[1]) / SQ(hOmg);

   // singmatrix = numba.cuda.shared.array((10,N),dtype=numba.types.complex128)
   int sizesing = 10 * N * sizeof(float);
   float* singmatrix;
   checkCudaErrors(cudaMalloc((void**)&singmatrix, sizesing));

   n = 0
   for (int j=-(N - 1)/2; i < ((N-1)/2+1)); i++) {
      singmatrix[0 + n * 10] = 2 * atan2f(Gamm, ek - hOmg / 2 + hOmg * i);
      singmatrix[1 + n * 10] = 2 * atan2f(Gamm, ekq - hOmg / 2 + hOmg * i);
      singmatrix[2 + n * 10] = 2 * atan2f(Gamm, ek + hOmg / 2 + hOmg * i);
      singmatrix[3 + n * 10] = 2 * atan2f(Gamm, ekq + hOmg / 2 + hOmg * i);

//========================================================================================================//
// HERE DOWN NEEDS MODIFICATION
      singmatrix[4 + n * 10] = complex(0, 1) * logf(Gammsq + SQ(ek - hOmg / 2 + hOmg * i));
      singmatrix[5 + n * 10] = complex(0, 1) * logf(Gammsq + SQ(ekq - hOmg / 2 + hOmg * i));
      singmatrix[6 + n * 10] = complex(0, 1) * logf(Gammsq + SQ(ek + hOmg / 2 + hOmg * i));
      singmatrix[7 + n * 10] = complex(0, 1) * logf(Gammsq + SQ(ekq + hOmg / 2 + hOmg * i));

      singmatrix[8 + n * 10] = cudahelpers.my_heaviside(mu - hOmg / 2 - hOmg * i);
      singmatrix[9 + n * 10] = cudahelpers.my_heaviside(mu + hOmg / 2 - hOmg * i);
      n = n + 1
   }

   size_dbl = 5
   dblmatrix = numba.cuda.shared.array((9,size_dbl),dtype=numba.types.complex128)

   n = 0
   for i in range(-(N - 1), N, 1):
       xi = hOmg * i
       zeta = ek - mu + xi
       eta = ekq - mu + xi

       zetasq = zeta ** 2
       etasq = eta ** 2

       dblmatrix[0,n] = 2 * atan2f(Gamm, zeta)
       dblmatrix[1,n] = 2 * atan2f(Gamm, eta)

       logged1 = logf(Gammsq + zetasq)
       logged2 = logf(Gammsq + etasq)

       dblmatrix[2,n] = complex(0, logged1)
       dblmatrix[3,n] = complex(0, logged2)

       dblmatrix[4,n] = cudahelpers.besselj(i, xk)
       dblmatrix[5,n] = cudahelpers.besselj(i, xkq)

       fac1i = ek - ekq + xi
       fac2i = complex(fac1i, 2 * Gamm)
       dblmatrix[6,n] = fac1i
       dblmatrix[7,n] = fac2i
       dblmatrix[8,n] = fac2i - ek + ekq
       n = n + 1

   #numba.cuda.syncthreads()

   for n in range(0, N):
       for alpha in range(0, N):
           for beta in range(0, N):
               for gamma in range(0, N):
                   for s in range(0, N):
                       for l in range(0, N):
                           p1p = dblmatrix[6,beta - gamma + N - 1] * (singmatrix[0,alpha] - dblmatrix[0,s + alpha] - singmatrix[4,alpha] + dblmatrix[2,s + alpha])
                           p2p = dblmatrix[7,alpha - gamma + N - 1] * (singmatrix[0,beta] - dblmatrix[0,s + beta] + singmatrix[4,beta] - dblmatrix[2,s + beta])
                           p3p = dblmatrix[8,alpha - beta + N - 1] * (-singmatrix[1,gamma] + dblmatrix[1,s + gamma] - singmatrix[5,gamma] + dblmatrix[3,s + gamma])

                           p1m = dblmatrix[6,beta - gamma + N - 1] * (singmatrix[2,alpha] - dblmatrix[0,s + alpha] - singmatrix[6,alpha] + dblmatrix[2,s + alpha])

                           p2m = dblmatrix[7,alpha - gamma + N - 1] * ( singmatrix[2,beta] - dblmatrix[0,s + beta] + singmatrix[6,beta] - dblmatrix[2,s + beta])

                           p3m = dblmatrix[8,alpha - beta + N - 1] * (-singmatrix[3,gamma] + dblmatrix[1,s + gamma] - singmatrix[7,gamma] + dblmatrix[3,s + gamma])

                           d1 = -2 * complex(0, 1) * dblmatrix[6,beta - gamma + N - 1] * dblmatrix[7,alpha - gamma + N - 1] * dblmatrix[8,alpha - beta + N - 1]

                           omint1p = singmatrix[8,s] * ((p1p + p2p + p3p) / d1)

                           omint1m = singmatrix[9,s] * ((p1m + p2m + p3m) / d1)

                           bess1 = dblmatrix[5,gamma - n + N - 1] * dblmatrix[5,gamma - l + N - 1] * dblmatrix[4,beta - l + N - 1] * dblmatrix[4,beta - s + N - 1] * dblmatrix[4,alpha - s + N - 1] * dblmatrix[4,alpha - n + N - 1]

                           grgl = bess1 * (omint1p - omint1m)

                           pp1p = dblmatrix[6,alpha - beta + N - 1] * (-singmatrix[1,gamma] + dblmatrix[1,s + gamma] - singmatrix[5,gamma] + dblmatrix[3,s + gamma])

                           pp2p = dblmatrix[7,alpha - gamma + N - 1] * (-singmatrix[1,beta] + dblmatrix[1,s + beta] + singmatrix[5,beta] - dblmatrix[3,s + beta])

                           pp3p = dblmatrix[8,beta - gamma + N - 1] * (singmatrix[0,alpha] - dblmatrix[0,s + alpha] - singmatrix[4,alpha] + dblmatrix[2,s + alpha])

                           pp1m = dblmatrix[6,alpha - beta + N - 1] * (-singmatrix[3,gamma] + dblmatrix[1,s + gamma] - singmatrix[7,gamma] + dblmatrix[3,s + gamma])

                           pp2m = dblmatrix[7,alpha - gamma + N - 1] * (-singmatrix[3,beta] + dblmatrix[1,s + beta] + singmatrix[7,beta] - dblmatrix[3,s + beta])

                           pp3m = dblmatrix[8,beta - gamma + N - 1] * (singmatrix[2,alpha] - dblmatrix[0,s + alpha] - singmatrix[6,alpha] + dblmatrix[2,s + alpha])

                           d2 = -2 * complex(0, 1) * dblmatrix[6,alpha - beta + N - 1] * dblmatrix[7,alpha - gamma + N - 1] * dblmatrix[8,beta - gamma + N - 1]

                           omint2p = singmatrix[8,s] * ((pp1p + pp2p + pp3p) / d2)

                           omint2m = singmatrix[9,s] * ((pp1m + pp2m + pp3m) / d2)

                           bess2 = dblmatrix[5,gamma - n + N - 1] * dblmatrix[5,gamma - s + N - 1] * dblmatrix[5,beta - s + N - 1] * dblmatrix[5,beta - l + N - 1] * dblmatrix[4,alpha - l + N - 1] * dblmatrix[4,alpha - n + N - 1]

                           glga = bess2 * (omint2p - omint2m)

                           dds = dds + Gamm * (grgl + glga)
   return -8 * dds.real / CB(CUDART_PI_F);
}
