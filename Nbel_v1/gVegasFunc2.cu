#include "vegasconst.h"

#include <cuComplex.h> // Complex number module of cuda.

#define CUDART_PI_F 3.141592654f
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

__device__
float func(double *rx, double wgt)
{
    float result;
    cuFloatComplex  vkqt;
    cuFloatComplex  vkqb;
    cuFloatComplex  vkt;
    cuFloatComplex  vkb;
    float N2 = (N - 1) / 2;

    vkqt = make_cuFloatComplex(-V0 * (rx[2] + qy), -V0 * (rx[1] + qx));
    vkqb = cuConjf(vkqt);

    vkt = make_cuFloatComplex(-V0 * (rx[2]), -V0 * (rx[1]));
    vkb = cuConjf(vkt);

    cuFloatComplex *thekq, *phikq, *thek, *phik, *Grkq, *Grk, *Gakq, *Gak, *pvkqt, *pvkqb, *pvkt, *pvkb, *mp;

    thekq = (cuFloatComplex*)malloc((N+1)*sizeof(cuFloatComplex));
    phikq = (cuFloatComplex*)malloc((N+2)*sizeof(cuFloatComplex));

    thek = (cuFloatComplex*)malloc((N+1)*sizeof(cuFloatComplex));
    phik = (cuFloatComplex*)malloc((N+2)*sizeof(cuFloatComplex));

    Grkq = (cuFloatComplex*)malloc(N*N*sizeof(cuFloatComplex));
    Grk = (cuFloatComplex*)malloc(N*N*sizeof(cuFloatComplex));
    Gakq = (cuFloatComplex*)malloc(N*N*sizeof(cuFloatComplex));
    Gak = (cuFloatComplex*)malloc(N*N*sizeof(cuFloatComplex));

    pvkqt = (cuFloatComplex*)malloc(N*sizeof(cuFloatComplex));
    pvkqb = (cuFloatComplex*)malloc(N*sizeof(cuFloatComplex));
    pvkt = (cuFloatComplex*)malloc(N*sizeof(cuFloatComplex));
    pvkb = (cuFloatComplex*)malloc(N*sizeof(cuFloatComplex));
    mp = (cuFloatComplex*)malloc(N*sizeof(cuFloatComplex));

    cuFloatComplex thzkq = make_cuFloatComplex(1.f, 0.f);
    cuFloatComplex thokq = make_cuFloatComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2) * hOmg, Gamm);

    cuFloatComplex thzk = make_cuFloatComplex(1.f, 0.f);
    cuFloatComplex thok = make_cuFloatComplex(rx[0] - A * ((rx[1]) * (rx[1]) + (rx[2]) * (rx[2])) - V2 - (N2) * hOmg, Gamm);

    cuFloatComplex phinpkq = make_cuFloatComplex(1.f, 0.f);
    cuFloatComplex phinkq = make_cuFloatComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2 - (N - 1)) * hOmg, Gamm);

    cuFloatComplex phinpk = make_cuFloatComplex(1.f, 0.f);
    cuFloatComplex phink = make_cuFloatComplex(rx[0] - A * ((rx[1]) * (rx[1]) + (rx[2]) * (rx[2])) - V2 - (N2 - (N - 1)) * hOmg, Gamm);

    thekq[0] = thzkq;
    thekq[1] = thokq;
    thek[0] = thzk;
    thek[1] = thok;

    phikq[0] = make_cuFloatComplex(0.0, 0.0);
    phikq[N+1] = phinpkq;
    phikq[N] = phinkq;

    phik[0] = make_cuFloatComplex(0.0, 0.0);
    phik[N+1] = phinpk;
    phik[N] = phink;

    cuFloatComplex vnkqt = make_cuFloatComplex(1.0, 0.0);
    pvkqt[0] = vnkqt;
    cuFloatComplex vnkqb = make_cuFloatComplex(1.0, 0.0);
    pvkqb[0] = vnkqb;
    cuFloatComplex vnkt = make_cuFloatComplex(1.0, 0.0);
    pvkt[0] = vnkt;
    cuFloatComplex vnkb = make_cuFloatComplex(1.0, 0.0);
    pvkb[0] = vnkb;

    cuFloatComplex mn = make_cuFloatComplex(1.0, 0.0);
    mp[0] = mn;

    //for ss in range(2, N + 1):
    for(int ss=2; ss < N+1; ss++) {
        int s = N - ss + 1;
        int n = ss - 1;

        cuFloatComplex theskq = make_cuFloatComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2 - (ss - 1)) * hOmg, Gamm); 
        theskq = cuCsubf(
            cuCmulf(theskq, thokq), 
            cuCmulf(vkqt, cuCmulf(vkqb, thzkq))
        );
        thzkq = thokq;
        thokq = theskq;

        thekq[ss] = thokq;

        cuFloatComplex phiskq = make_cuFloatComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2 - (s - 1)) * hOmg, Gamm);
        phiskq = cuCsubf(
            cuCmulf(phiskq, phinkq), 
            cuCmulf(vkqt, cuCmulf(vkqb, phinpkq))
        );
        phinpkq = phinkq;
        phinkq = phiskq;

        phikq[s] = phinkq;

        cuFloatComplex thesk = make_cuFloatComplex(rx[0] - A * (rx[1] * rx[1] + rx[2] * rx[2]) - V2 - (N2 - (ss - 1)) * hOmg, Gamm);
        thesk = cuCsubf(
            cuCmulf(theskq, thok), 
            cuCmulf(vkt, cuCmulf(vkb, thzk))
        );
        thzk = thok;
        thok = thesk;

        thek[ss] = thok;

        cuFloatComplex phisk = make_cuFloatComplex(rx[0] - A * (rx[1] * rx[1] + rx[2] * rx[2]) - V2 - (N2 - (s - 1)) * hOmg, Gamm);
        phisk = cuCsubf(
            cuCmulf(phisk, phink), 
            cuCmulf(vkt, cuCmulf(vkb, phinpk))
        );
        phinpk = phink;
        phink = phisk;

        phik[s] = phink;

        vnkqt = cuCmulf(vnkqt, vkqt);
        vnkt = cuCmulf(vnkt, vkt);
        vnkqb = cuCmulf(vnkqb, vkqb);
        vnkb = cuCmulf(vnkb, vkb);

        pvkqt[n] = vnkqt;
        pvkt[n] = vnkt;

        pvkqb[n] = vnkqb;
        pvkb[n] = vnkb;

        mn = cuCmulf(mn, make_cuFloatComplex(-1.0, 0.0));
        mp[n] = mn;
    }

    //for m in range(0, N):
    for(int m=0; m<N; m++) {
        //for n in range(m, N):
        for(int n=m; n<N; n++) {
            //if m == n:
            if(m==n){
                //Grkq[m, n] = thekq[m] * phikq[n + 2] / thekq[N]
                Grkq[IDX2C(m,n,N)] = cuCdivf(
                    cuCmulf(thekq[m], phikq[n + 2]), 
                    thekq[N]
                );
                //Gakq[IDX2C(m,n,N)] = complex(Grkq[IDX2C(m,n,N)].real, -Grkq[IDX2C(m,n,N)].imag);
                Gakq[IDX2C(m,n,N)] = cuConjf(Grkq[IDX2C(m,n,N)]);

                //Grk[IDX2C(m,n,N)] = thek[m] * phik[n + 2] / thek[N];
                Grk[IDX2C(m,n,N)] = cuCdivf(
                    cuCmulf(thek[m], phik[n + 2]), 
                    thek[N]
                );
                //Gak[IDX2C(m,n,N)] = complex(Grk[IDX2C(m,n,N)].real, -Grk[IDX2C(m,n,N)].imag);
                Gak[IDX2C(m,n,N)] = cuConjf(Grk[IDX2C(m,n,N)]);
            }
            //elif m < n:
            else if(m<n) {

                Grkq[IDX2C(m,n,N)] = cuCmulf(
                    cuCmulf(mp[n - m], pvkqt[n - m]),
                    cuCmulf(
                        thekq[m], 
                        cuCdivf(phikq[n + 2], thekq[N])
                    )
                );
                //Grkq[IDX2C(n,m,N)] = mp[n - m] * pvkqb[n - m] * thekq[m] * phikq[n + 2] / thekq[N];
                Grkq[IDX2C(n,m,N)] = cuCmulf(
                    cuCmulf(mp[n - m], pvkqb[n - m]),
                    cuCmulf(
                        thekq[m], 
                        cuCdivf(phikq[n + 2], thekq[N])
                    )
                );
                //Gakq[IDX2C(m,n,N)] = complex(Grkq[IDX2C(n,m,N)].real, -Grkq[IDX2C(n,m,N)].imag);
                //Gakq[IDX2C(n,m,N)] = complex(Grkq[IDX2C(m,n,N)].real, -Grkq[IDX2C(m,n,N)].imag);
                Gakq[IDX2C(m,n,N)] = cuConjf(Grkq[IDX2C(n,m,N)]);
                Gakq[IDX2C(n,m,N)] = cuConjf(Grkq[IDX2C(m,n,N)]);

                //Grk[IDX2C(m,n,N)] = mp[n - m] * pvkt[n - m] * thek[m] * phik[n + 2] / thek[N];
                Grk[IDX2C(m,n,N)] = cuCmulf(
                    cuCmulf(mp[n - m], pvkt[n - m]),
                    cuCmulf(
                        thek[m], 
                        cuCdivf(phik[n + 2], thek[N])
                    )
                );
                //Grk[IDX2C(n,m,N)] = mp[n - m] * pvkb[n - m] * thek[m] * phik[n + 2] / thek[N];
                Grk[IDX2C(n,m,N)] = cuCmulf(
                    cuCmulf(mp[n - m], pvkb[n - m]),
                    cuCmulf(
                        thek[m], 
                        cuCdivf(phik[n + 2], thek[N])
                    )
                );
                //Gak[IDX2C(m,n,N)] = complex(Grk[IDX2C(n,m,N)].real, -Grk[IDX2C(n,m,N)].imag);
                //Gak[IDX2C(n,m,N)] = complex(Grk[IDX2C(m,n,N)].real, -Grk[IDX2C(m,n,N)].imag);
                Gak[IDX2C(m,n,N)] = cuConjf(Grk[IDX2C(n,m,N)]);
                Gak[IDX2C(n,m,N)] = cuConjf(Grk[IDX2C(m,n,N)]);
            }
        }
    }

    cuFloatComplex chi1f = make_cuFloatComplex(0, 0);
//
    //for l in range(0, N):
    for(int l=0; l<N; l++) {
        //if (mu - x[0] + ((N - 1) / 2 - l) * hOmg) < 0:
        if((mu - rx[0] + (N2 - l) * hOmg) >= 0) {
            //for m in range(0, N):
            for(int m=0; m<N; m++) {
                //for n in range(0, N):
                for(int n=0; n<N; n++) {
                    //chi1f += Grkq[IDX2C(m,n,N)] * Grk[IDX2C(n,l,N)] * Gak[IDX2C(l,m,N)] + Grkq[IDX2C(n,l,N)] * Gakq[IDX2C(l,m,N)] * Gak[IDX2C(m,n,N)];
                    chi1f = cuCaddf(
                        chi1f,
                        cuCaddf(
                            cuCmulf(Grkq[IDX2C(m,n,N)], cuCmulf(Grk[IDX2C(n,l,N)], Gak[IDX2C(l,m,N)])),
                            cuCmulf(Grkq[IDX2C(n,l,N)], cuCmulf(Gakq[IDX2C(l,m,N)], Gak[IDX2C(m,n,N)]))
                        )
                    );
                }
            }
        }
    }

    result = Fac*cuCrealf(chi1f);

    free(thekq); 
    free(phikq); 
    free(thek); 
    free(phik); 
    free(Grkq); 
    free(Grk); 
    free(Gakq); 
    free(Gak); 
    free(pvkqt); 
    free(pvkqb); 
    free(pvkt); 
    free(pvkb); 
    free(mp);

    return result;
}

