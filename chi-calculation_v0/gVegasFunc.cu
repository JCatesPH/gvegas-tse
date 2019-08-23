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
   int sizesing = N * sizeof(float);
   float* singr, singi;
   checkCudaErrors(cudaMalloc((void**)&singr, 6 * sizesing));
   checkCudaErrors(cudaMalloc((void**)&singi, 4 * sizesing));

   n = 0
   for (int j=-(N - 1)/2; i < ((N-1)/2+1)); i++) {
      singr[0 + n * 6] = 2 * atan2f(Gamm, ek - hOmg / 2 + hOmg * i);
      singr[1 + n * 6] = 2 * atan2f(Gamm, ekq - hOmg / 2 + hOmg * i);
      singr[2 + n * 6] = 2 * atan2f(Gamm, ek + hOmg / 2 + hOmg * i);
      singr[3 + n * 6] = 2 * atan2f(Gamm, ekq + hOmg / 2 + hOmg * i);

      singr[4 + n * 6] = step(mu - hOmg / 2 - hOmg * i, 0.f); // 8 -> 4
      singr[5 + n * 6] = step(mu + hOmg / 2 - hOmg * i, 0.f); // 9 -> 5

      //------//

      singi[0 + n * 10] = logf(Gammsq + SQ(ek - hOmg / 2 + hOmg * i));  // 4 -> 0
      singi[1 + n * 10] = logf(Gammsq + SQ(ekq - hOmg / 2 + hOmg * i)); // 5 -> 1
      singi[2 + n * 10] = logf(Gammsq + SQ(ek + hOmg / 2 + hOmg * i));  // 6 -> 2
      singi[3 + n * 10] = logf(Gammsq + SQ(ekq + hOmg / 2 + hOmg * i)); // 7 -> 3

      n = n + 1
   }

   int sizedbl = (2*N-1) * sizeof(float);
   float* dblr, dbli;
   checkCudaErrors(cudaMalloc((void**)&dblr, 5 * sizedbl));
   checkCudaErrors(cudaMalloc((void**)&dbli, 2 * sizedbl));

   cuFloatComplex* dblz;
   checkCudaErrors(cudaMalloc((void**)&dblz, 2*(2*N-1)*sizeof(cuFloatComplex)));

   n = 0
   for (int i=-(N - 1); i < N; i++){
       dblr[0 + n * 5] = 2 * atan2f(Gamm, (ek - mu + hOmg * i));
       dblr[1 + n * 5] = 2 * atan2f(Gamm, (ekq - mu + hOmg * i));

       dblr[2 + n * 5] = jnf(i, xk); // Bessel function of order i
       dblr[3 + n * 5] = jnf(i, xkq);

       dblr[4 + n * 5] = ek - ekq + hOmg * i;

       //------//

       dbli[0 + n * 2] = logf(Gammsq + SQ(ek - mu + hOmg * i));
       dbli[1 + n * 2] = logf(Gammsq + SQ(ekq - mu + hOmg * i));

       //------//

       dblz[0 + n * 2] = make_cuFloatComplex(ek - ekq + hOmg * i, 2 * Gamm);
       dblz[1 + n * 2] = make_cuFloatComplex(hOmg * i, 2 * Gamm);
       n = n + 1
    }


   for (int n=0; n<N; n++){
       for (int alpha=0; alpha<N; alpha++){
           for (int beta=0; beta<N; beta++){
               for (int gamma=0; gamma<N; gamma++){
                   for (int s=0; s<N; s++){
                       for (int l=0; l<N; l++){
                           p1p = dblr[4,beta - gamma + N - 1] * (singmatrix[0,alpha] - dblr[0,s + alpha] - singmatrix[4,alpha] + dbli[0,s + alpha])
                           p2p = dblz[0,alpha - gamma + N - 1] * (singmatrix[0,beta] - dblr[0,s + beta] + singmatrix[4,beta] - dbli[0,s + beta])
                           p3p = dblz[1,alpha - beta + N - 1] * (-singmatrix[1,gamma] + dblr[1,s + gamma] - singmatrix[5,gamma] + dbli[1,s + gamma])

                           p1m = dblr[4,beta - gamma + N - 1] * (singmatrix[2,alpha] - dblr[0,s + alpha] - singmatrix[6,alpha] + dbli[0,s + alpha])

                           p2m = dblz[0,alpha - gamma + N - 1] * ( singmatrix[2,beta] - dblr[0,s + beta] + singmatrix[6,beta] - dbli[0,s + beta])

                           p3m = dblz[1,alpha - beta + N - 1] * (-singmatrix[3,gamma] + dblr[1,s + gamma] - singmatrix[7,gamma] + dbli[1,s + gamma])

                           d1 = -2 * complex(0, 1) * dblr[4,beta - gamma + N - 1] * dblz[0,alpha - gamma + N - 1] * dblz[1,alpha - beta + N - 1]

                           omint1p = singmatrix[8,s] * ((p1p + p2p + p3p) / d1)

                           omint1m = singmatrix[9,s] * ((p1m + p2m + p3m) / d1)

                           bess1 = dblr[3,gamma - n + N - 1] * dblr[3,gamma - l + N - 1] * dblr[2,beta - l + N - 1] * dblr[2,beta - s + N - 1] * dblr[2,alpha - s + N - 1] * dblr[2,alpha - n + N - 1]

                           grgl = bess1 * (omint1p - omint1m)

                           pp1p = dblr[4,alpha - beta + N - 1] * (-singmatrix[1,gamma] + dblr[1,s + gamma] - singmatrix[5,gamma] + dbli[1,s + gamma])

                           pp2p = dblz[0,alpha - gamma + N - 1] * (-singmatrix[1,beta] + dblr[1,s + beta] + singmatrix[5,beta] - dbli[1,s + beta])

                           pp3p = dblz[1,beta - gamma + N - 1] * (singmatrix[0,alpha] - dblr[0,s + alpha] - singmatrix[4,alpha] + dbli[0,s + alpha])

                           pp1m = dblr[4,alpha - beta + N - 1] * (-singmatrix[3,gamma] + dblr[1,s + gamma] - singmatrix[7,gamma] + dbli[1,s + gamma])

                           pp2m = dblz[0,alpha - gamma + N - 1] * (-singmatrix[3,beta] + dblr[1,s + beta] + singmatrix[7,beta] - dbli[1,s + beta])

                           pp3m = dblz[1,beta - gamma + N - 1] * (singmatrix[2,alpha] - dblr[0,s + alpha] - singmatrix[6,alpha] + dbli[0,s + alpha])

                           d2 = -2 * complex(0, 1) * dblr[4,alpha - beta + N - 1] * dblz[0,alpha - gamma + N - 1] * dblz[1,beta - gamma + N - 1]

                           omint2p = singmatrix[8,s] * ((pp1p + pp2p + pp3p) / d2)

                           omint2m = singmatrix[9,s] * ((pp1m + pp2m + pp3m) / d2)

                           bess2 = dblr[3,gamma - n + N - 1] * dblr[3,gamma - s + N - 1] * dblr[3,beta - s + N - 1] * dblr[3,beta - l + N - 1] * dblr[2,alpha - l + N - 1] * dblr[2,alpha - n + N - 1]

                           glga = bess2 * (omint2p - omint2m)

                           dds = dds + Gamm * (grgl + glga)
                        }
                    }
                }
            }
        }
    }
   return -8 * cuCrealf(dds) / CB(CUDART_PI_F);
}
