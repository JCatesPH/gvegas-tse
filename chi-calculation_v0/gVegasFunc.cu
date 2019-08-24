#include "vegasconst.h"

#include <cuComplex.h> // Complex number module of cuda.

#define CUDART_PI_F 3.141592654f

/*-------- constants for chi ---------*/
#define mu      0.1f
#define hOmg    0.3f
#define a       3.6f
#define A       4.f
#define rati    0.1
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
float heaviside(float x, float z)
{
    if (x < z)
    {
        return 0.f;
    }
    else
    {
        return 1.f;
    }
    
}

__device__
float chi(float* rx, float wgt)
{
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
   cuFloatComplex* sing;
   cudaMalloc((void**)&sing, N*sizeof(cuFloatComplex));

   int n = 0;
   for (int i=-(N-1)/2; i<((N-1)/2+1); i++) {
    sing[0 + n * 6] = make_cuFloatComplex(2 * atan2f(Gamm, ek - hOmg / 2 + hOmg * i), 0);
    sing[1 + n * 6] = make_cuFloatComplex(2 * atan2f(Gamm, ekq - hOmg / 2 + hOmg * i), 0);
    sing[2 + n * 6] = make_cuFloatComplex(2 * atan2f(Gamm, ek + hOmg / 2 + hOmg * i), 0);
    sing[3 + n * 6] = make_cuFloatComplex(2 * atan2f(Gamm, ekq + hOmg / 2 + hOmg * i), 0);

    sing[8 + n * 6] = make_cuFloatComplex(heaviside(mu - hOmg / 2 - hOmg * i, 0.f), 0); 
    sing[9 + n * 6] = make_cuFloatComplex(heaviside(mu + hOmg / 2 - hOmg * i, 0.f), 0); 

    sing[4 + n * 10] = make_cuFloatComplex(0, logf(Gammsq + SQ(ek - hOmg / 2 + hOmg * i)));  
    sing[5 + n * 10] = make_cuFloatComplex(0, logf(Gammsq + SQ(ekq - hOmg / 2 + hOmg * i))); 
    sing[6 + n * 10] = make_cuFloatComplex(0, logf(Gammsq + SQ(ek + hOmg / 2 + hOmg * i)));  
    sing[7 + n * 10] = make_cuFloatComplex(0, logf(Gammsq + SQ(ekq + hOmg / 2 + hOmg * i))); 

    n = n + 1;
   }

   cuFloatComplex* dbl;
   cudaMalloc((void**)&dbl, 9*(2*N-1)*sizeof(cuFloatComplex));

   n = 0;
   for (int i=-(N-1); i < N; i++)
   {
       dbl[0 + n * 5] = make_cuFloatComplex(2 * atan2f(Gamm, (ek - mu + hOmg * i)), 0);
       dbl[1 + n * 5] = make_cuFloatComplex(2 * atan2f(Gamm, (ekq - mu + hOmg * i)), 0);

       dbl[4 + n * 5] = make_cuFloatComplex(jnf(i, xk), 0); // Bessel function of order i
       dbl[5 + n * 5] = make_cuFloatComplex(jnf(i, xkq), 0);

       dbl[6 + n * 5] = make_cuFloatComplex(ek - ekq + hOmg * i, 0);

       dbl[2 + n * 2] = make_cuFloatComplex(0, logf(Gammsq + SQ(ek - mu + hOmg * i)));
       dbl[3 + n * 2] = make_cuFloatComplex(0, logf(Gammsq + SQ(ekq - mu + hOmg * i)));

       dbl[7 + n * 2] = make_cuFloatComplex(ek - ekq + hOmg * i, 2 * Gamm);
       dbl[8 + n * 2] = make_cuFloatComplex(hOmg * i, 2 * Gamm);

       n = n + 1;
    }

    cuFloatComplex I2 = make_cuFloatComplex(0, -2);

    cuFloatComplex omint1p;
    cuFloatComplex omint1m;
    cuFloatComplex bess1;
    cuFloatComplex omint2p;
    cuFloatComplex omint2m;
    cuFloatComplex bess2;
    cuFloatComplex dds;


   for (int n=0; n<N; n++){
       for (int alpha=0; alpha<N; alpha++){
           for (int beta=0; beta<N; beta++){
               for (int gamma=0; gamma<N; gamma++){
                   for (int s=0; s<N; s++){
                       for (int l=0; l<N; l++){
                            omint1p = cuCmulf(
                                sing[8+s*10], 
                                cuCdivf(
                                    cuCaddf(
                                        cuCaddf(
                                            cuCmulf(
                                                dbl[6+(beta - gamma + N - 1)*9], 
                                                cuCsubf(
                                                    cuCsubf(sing[alpha*10], dbl[(s+alpha)*9]), 
                                                    cuCaddf(sing[4+alpha*10], dbl[2+(s+alpha)*9])
                                                )
                                            ), 
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9], 
                                                cuCaddf(
                                                    cuCsubf(sing[beta*10], dbl[(s+beta)*9]), 
                                                    cuCsubf(sing[4+beta*10], dbl[2+(s+beta)*9])
                                                )
                                            )
                                        ), 
                                        cuCmulf(
                                            dbl[8+(alpha-beta+N-1)*9], 
                                            cuCsubf(
                                                cuCsubf(dbl[1+(s+gamma)*9], sing[1+gamma*10]), 
                                                cuCaddf(sing[5+gamma*10], dbl[3+(s+gamma)*9])
                                            )
                                        )
                                    ),  
                                    cuCmulf(
                                        I2, 
                                        cuCmulf(
                                            dbl[6 + (beta - gamma + N - 1) * 9], 
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9], 
                                                dbl[8+(alpha-beta+N-1)*9]
                                            )
                                        )
                                    )
                                )
                            );

                            omint1m = cuCmulf(
                                sing[9+s*10], 
                                cuCdivf(
                                    cuCaddf(
                                        cuCaddf(
                                            cuCmulf(
                                                dbl[6+(beta-gamma+N-1) * 9], 
                                                cuCsubf(
                                                    cuCsubf(sing[2+alpha*10], dbl[(s+alpha)*9]), 
                                                    cuCaddf(sing[6+alpha*10], dbl[2+(s+alpha)*9])
                                                )
                                            ),  
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9], 
                                                cuCsubf(
                                                    cuCsubf(sing[2+beta*10], dbl[(s+beta)*9]), 
                                                    cuCaddf(sing[6+beta*10], dbl[2+(s+beta)*9])
                                                )
                                            )
                                        ), 
                                        cuCmulf(
                                            dbl[8+(alpha-beta+N-1)*9], 
                                            cuCsubf(
                                                cuCsubf(dbl[1+(s+gamma)*9], sing[3+gamma*10]), 
                                                cuCaddf(sing[7+gamma*10], dbl[3+(s+gamma)*9])
                                            )
                                        )
                                    ),  
                                    cuCmulf(
                                        I2, 
                                        cuCmulf(
                                            dbl[6 + (beta - gamma + N - 1) * 9], 
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9], 
                                                dbl[8+(alpha-beta+N-1)*9]
                                            )
                                        )
                                    )
                                )
                            );

                            bess1 = cuCmulf(
                                dbl[5+(gamma-n+N-1)*9], 
                                cuCmulf(
                                    dbl[5+(gamma-l+N-1)*9], 
                                    cuCmulf(
                                        dbl[4+(beta-l+N-1)*9], 
                                        cuCmulf(
                                            dbl[4+(beta-s+N-1)*9], 
                                            cuCmulf(
                                                dbl[4+(alpha-s+N-1)*9], 
                                                dbl[4+(alpha-n+N-1)*9]
                                            )
                                        )
                                    )
                                )
                            );

                            omint2p = cuCmulf(
                                sing[8+s*10],
                                cuCdivf(
                                    cuCaddf(
                                        cuCmulf(
                                            dbl[6+(alpha-beta+N-1)*9], 
                                            cuCsubf(
                                                cuCsubf(dbl[1+(s+gamma)*9], sing[1+gamma*10]), 
                                                cuCaddf(sing[5+gamma*10], dbl[3+(s+gamma)*9])
                                            )
                                        ), 
                                        cuCaddf(
                                            cuCmulf(
                                                dbl[7+(alpha-beta+N-1)*9],
                                                cuCaddf(
                                                    cuCsubf(dbl[1+(s+beta)*9], sing[1+beta*10]), 
                                                    cuCsubf(sing[5+beta*10], dbl[3+(s+beta)*9])
                                                )
                                            ), 
                                            cuCmulf(
                                                dbl[8+(beta-gamma+N-1)*9],
                                                cuCsubf(
                                                    cuCsubf(sing[alpha*10], dbl[0 + (s+alpha) * 9]), 
                                                    cuCaddf(sing[4+alpha*10], dbl[2 + (s+alpha) * 9])
                                                )
                                            )
                                        )
                                    ), 
                                    cuCmulf(
                                        I2, 
                                        cuCmulf(
                                            dbl[6+(alpha-beta+N-1)*9],
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9],
                                                dbl[8+(beta-gamma+N-1)*9]
                                            )
                                        )
                                    )
                                )
                            );

                            omint2m = cuCmulf(
                                sing[9+s*10], 
                                cuCdivf(
                                    cuCaddf(
                                        cuCmulf(
                                            dbl[6+(alpha-beta+N-1)*9],
                                            cuCsubf(
                                                cuCsubf(dbl[1+(s+gamma)*9], sing[3+gamma*10]),
                                                cuCaddf(sing[7+gamma*10], dbl[3+(s+gamma)*9])
                                            )
                                        ), 
                                        cuCaddf(
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9], 
                                                cuCaddf(
                                                    cuCsubf(dbl[1+(s+beta)*9], sing[3+beta*10]), 
                                                    cuCsubf(sing[7+beta*10], dbl[3+(s+beta)*9])
                                                )
                                            ), 
                                            cuCmulf(
                                                dbl[8+(beta-gamma+N-1)*9],
                                                cuCsubf(
                                                    cuCsubf(sing[2+alpha*10], dbl[0 + (s+alpha) * 9]),
                                                    cuCaddf(sing[6+alpha*10], dbl[2 + (s+alpha) * 9])
                                                )
                                            )
                                        )
                                    ),
                                    cuCmulf(
                                        I2, 
                                        cuCmulf(
                                            dbl[6+(alpha-beta+N-1)*9],
                                            cuCmulf(
                                                dbl[7+(alpha-gamma+N-1)*9],
                                                dbl[8+(beta-gamma+N-1)*9]
                                            )
                                        )
                                    )
                                )
                            );

                            bess2 = cuCmulf(
                                dbl[5+(gamma-n+N-1)*9],
                                cuCmulf(
                                    dbl[5+(gamma-s+N-1)*9], 
                                    cuCmulf(
                                        dbl[5+(beta-s+N-1)*9],
                                        cuCmulf(
                                            dbl[5+(beta-l+N-1)*9],
                                            cuCmulf(
                                                dbl[4+(alpha-l+N-1)*9],
                                                dbl[4+(alpha-n+N-1)*9]
                                            )
                                        )
                                    )
                                )
                            );

                            dds = cuCaddf(
                                dds, 
                                // Gamm * (grgl + glga)
                                cuCaddf(
                                    make_cuFloatComplex(
                                        Gamm*cuCrealf(cuCmulf(
                                            bess1, 
                                            cuCsubf(omint1p, omint1m)
                                            )
                                        ), 
                                        Gamm*cuCimagf(cuCmulf(
                                            bess1, 
                                            cuCsubf(omint1p, omint1m))
                                            )
                                        ),
                                    make_cuFloatComplex(
                                        Gamm*cuCrealf(cuCmulf(
                                            bess2, 
                                            cuCsubf(omint2p, omint2m)
                                            )
                                        ), 
                                        Gamm*cuCimagf(cuCmulf(
                                            bess2, 
                                            cuCsubf(omint2p, omint2m))
                                        )
                                    )
                                )
                            );
                        }
                    }
                }
            }
        }
    }
   return -8 * cuCrealf(dds) / CB(CUDART_PI_F);
}
