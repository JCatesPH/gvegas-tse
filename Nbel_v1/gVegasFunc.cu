#include "vegasconst.h"

#include <cuComplex.h> // Complex number module of cuda.

#define CUDART_PI_F 3.141592654f
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

__device__
float func(double *rx, double wgt)
{
    float result;
    cuDoubleComplex  vkqt;
    cuDoubleComplex  vkqb;
    cuDoubleComplex  vkt;
    cuDoubleComplex  vkb;
    float N2 = (N - 1) / 2;

    vkqt = make_cuDoubleComplex(-V0 * (rx[2] + qy), -V0 * (rx[1] + qx));
    vkqb = cuConj(vkqt);

    vkt = make_cuDoubleComplex(-V0 * (rx[2]), -V0 * (rx[1]));
    vkb = cuConj(vkt);

    cuDoubleComplex *thekq, *phikq, *thek, *phik, *Grkq, *Grk, *Gakq, *Gak, *pvkqt, *pvkqb, *pvkt, *pvkb, *mp;

    thekq = (cuDoubleComplex*)malloc((N+1)*sizeof(cuDoubleComplex));
    phikq = (cuDoubleComplex*)malloc((N+2)*sizeof(cuDoubleComplex));

    thek = (cuDoubleComplex*)malloc((N+1)*sizeof(cuDoubleComplex));
    phik = (cuDoubleComplex*)malloc((N+2)*sizeof(cuDoubleComplex));

    Grkq = (cuDoubleComplex*)malloc(N*N*sizeof(cuDoubleComplex));
    Grk = (cuDoubleComplex*)malloc(N*N*sizeof(cuDoubleComplex));
    Gakq = (cuDoubleComplex*)malloc(N*N*sizeof(cuDoubleComplex));
    Gak = (cuDoubleComplex*)malloc(N*N*sizeof(cuDoubleComplex));

    pvkqt = (cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));
    pvkqb = (cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));
    pvkt = (cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));
    pvkb = (cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));
    mp = (cuDoubleComplex*)malloc(N*sizeof(cuDoubleComplex));

    cuDoubleComplex thzkq = make_cuDoubleComplex(1.f, 0.f);
    cuDoubleComplex thokq = make_cuDoubleComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2) * hOmg, Gamm);

    cuDoubleComplex thzk = make_cuDoubleComplex(1.f, 0.f);
    cuDoubleComplex thok = make_cuDoubleComplex(rx[0] - A * ((rx[1]) * (rx[1]) + (rx[2]) * (rx[2])) - V2 - (N2) * hOmg, Gamm);

    cuDoubleComplex phinpkq = make_cuDoubleComplex(1.f, 0.f);
    cuDoubleComplex phinkq = make_cuDoubleComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2 - (N - 1)) * hOmg, Gamm);

    cuDoubleComplex phinpk = make_cuDoubleComplex(1.f, 0.f);
    cuDoubleComplex phink = make_cuDoubleComplex(rx[0] - A * ((rx[1]) * (rx[1]) + (rx[2]) * (rx[2])) - V2 - (N2 - (N - 1)) * hOmg, Gamm);

    thekq[0] = thzkq;
    thekq[1] = thokq;
    thek[0] = thzk;
    thek[1] = thok;

    phikq[0] = make_cuDoubleComplex(0.0, 0.0);
    phikq[N+1] = phinpkq;
    phikq[N] = phinkq;

    phik[0] = make_cuDoubleComplex(0.0, 0.0);
    phik[N+1] = phinpk;
    phik[N] = phink;

    cuDoubleComplex vnkqt = make_cuDoubleComplex(1.0, 0.0);
    pvkqt[0] = vnkqt;
    cuDoubleComplex vnkqb = make_cuDoubleComplex(1.0, 0.0);
    pvkqb[0] = vnkqb;
    cuDoubleComplex vnkt = make_cuDoubleComplex(1.0, 0.0);
    pvkt[0] = vnkt;
    cuDoubleComplex vnkb = make_cuDoubleComplex(1.0, 0.0);
    pvkb[0] = vnkb;

    cuDoubleComplex mn = make_cuDoubleComplex(1.0, 0.0);
    mp[0] = mn;

    cuDoubleComplex theskq;
    cuDoubleComplex phiskq;
    cuDoubleComplex thesk;
    cuDoubleComplex phisk;

    //for ss in range(2, N + 1):
    for(int ss=2; ss < N+1; ss++) {
        int s = N - ss + 1;
        int n = ss - 1;

        theskq = make_cuDoubleComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2 - (ss - 1)) * hOmg, Gamm); 
        theskq = cuCsub(
            cuCmul(theskq, thokq), 
            cuCmul(vkqt, cuCmul(vkqb, thzkq))
        );
        thzkq = thokq;
        thokq = theskq;

        thekq[ss] = thokq;

        phiskq = make_cuDoubleComplex(rx[0] - A * ((rx[1] + qx) * (rx[1] + qx) + (rx[2] + qy) * (rx[2] + qy)) - V2 - (N2 - (s - 1)) * hOmg, Gamm);
        phiskq = cuCsub(
            cuCmul(phiskq, phinkq), 
            cuCmul(vkqt, cuCmul(vkqb, phinpkq))
        );
        phinpkq = phinkq;
        phinkq = phiskq;

        phikq[s] = phinkq;

        thesk = make_cuDoubleComplex(rx[0] - A * (rx[1] * rx[1] + rx[2] * rx[2]) - V2 - (N2 - (ss - 1)) * hOmg, Gamm);
        thesk = cuCsub(
            cuCmul(theskq, thok), 
            cuCmul(vkt, cuCmul(vkb, thzk))
        );
        thzk = thok;
        thok = thesk;

        thek[ss] = thok;

        phisk = make_cuDoubleComplex(rx[0] - A * (rx[1] * rx[1] + rx[2] * rx[2]) - V2 - (N2 - (s - 1)) * hOmg, Gamm);
        phisk = cuCsub(
            cuCmul(phisk, phink), 
            cuCmul(vkt, cuCmul(vkb, phinpk))
        );
        phinpk = phink;
        phink = phisk;

        phik[s] = phink;

        vnkqt = cuCmul(vnkqt, vkqt);
        vnkt = cuCmul(vnkt, vkt);
        vnkqb = cuCmul(vnkqb, vkqb);
        vnkb = cuCmul(vnkb, vkb);

        pvkqt[n] = vnkqt;
        pvkt[n] = vnkt;

        pvkqb[n] = vnkqb;
        pvkb[n] = vnkb;

        mn = cuCmul(mn, make_cuDoubleComplex(-1.0, 0.0));
        mp[n] = mn;
    }

    //for m in range(0, N):
    for(int m=0; m<N; m++) {
        //for n in range(m, N):
        for(int n=m; n<N; n++) {
            //if m == n:
            if(m==n){
                //Grkq[m, n] = thekq[m] * phikq[n + 2] / thekq[N]
                Grkq[IDX2C(m,n,N)] = cuCdiv(
                    cuCmul(thekq[m], phikq[n + 2]), 
                    thekq[N]
                );
                //Gakq[IDX2C(m,n,N)] = complex(Grkq[IDX2C(m,n,N)].real, -Grkq[IDX2C(m,n,N)].imag);
                Gakq[IDX2C(m,n,N)] = cuConj(Grkq[IDX2C(m,n,N)]);

                //Grk[IDX2C(m,n,N)] = thek[m] * phik[n + 2] / thek[N];
                Grk[IDX2C(m,n,N)] = cuCdiv(
                    cuCmul(thek[m], phik[n + 2]), 
                    thek[N]
                );
                //Gak[IDX2C(m,n,N)] = complex(Grk[IDX2C(m,n,N)].real, -Grk[IDX2C(m,n,N)].imag);
                Gak[IDX2C(m,n,N)] = cuConj(Grk[IDX2C(m,n,N)]);
            }
            //elif m < n:
            else if(m<n) {

                Grkq[IDX2C(m,n,N)] = cuCmul(
                    cuCmul(mp[n - m], pvkqt[n - m]),
                    cuCmul(
                        thekq[m], 
                        cuCdiv(phikq[n + 2], thekq[N])
                    )
                );
                //Grkq[IDX2C(n,m,N)] = mp[n - m] * pvkqb[n - m] * thekq[m] * phikq[n + 2] / thekq[N];
                Grkq[IDX2C(n,m,N)] = cuCmul(
                    cuCmul(mp[n - m], pvkqb[n - m]),
                    cuCmul(
                        thekq[m], 
                        cuCdiv(phikq[n + 2], thekq[N])
                    )
                );
                //Gakq[IDX2C(m,n,N)] = complex(Grkq[IDX2C(n,m,N)].real, -Grkq[IDX2C(n,m,N)].imag);
                //Gakq[IDX2C(n,m,N)] = complex(Grkq[IDX2C(m,n,N)].real, -Grkq[IDX2C(m,n,N)].imag);
                Gakq[IDX2C(m,n,N)] = cuConj(Grkq[IDX2C(n,m,N)]);
                Gakq[IDX2C(n,m,N)] = cuConj(Grkq[IDX2C(m,n,N)]);

                //Grk[IDX2C(m,n,N)] = mp[n - m] * pvkt[n - m] * thek[m] * phik[n + 2] / thek[N];
                Grk[IDX2C(m,n,N)] = cuCmul(
                    cuCmul(mp[n - m], pvkt[n - m]),
                    cuCmul(
                        thek[m], 
                        cuCdiv(phik[n + 2], thek[N])
                    )
                );
                //Grk[IDX2C(n,m,N)] = mp[n - m] * pvkb[n - m] * thek[m] * phik[n + 2] / thek[N];
                Grk[IDX2C(n,m,N)] = cuCmul(
                    cuCmul(mp[n - m], pvkb[n - m]),
                    cuCmul(
                        thek[m], 
                        cuCdiv(phik[n + 2], thek[N])
                    )
                );
                //Gak[IDX2C(m,n,N)] = complex(Grk[IDX2C(n,m,N)].real, -Grk[IDX2C(n,m,N)].imag);
                //Gak[IDX2C(n,m,N)] = complex(Grk[IDX2C(m,n,N)].real, -Grk[IDX2C(m,n,N)].imag);
                Gak[IDX2C(m,n,N)] = cuConj(Grk[IDX2C(n,m,N)]);
                Gak[IDX2C(n,m,N)] = cuConj(Grk[IDX2C(m,n,N)]);
            }
        }
    }

    cuDoubleComplex chi1f = make_cuDoubleComplex(0, 0);
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
                    chi1f = cuCadd(
                        chi1f,
                        cuCadd(
                            cuCmul(Grkq[IDX2C(m,n,N)], cuCmul(Grk[IDX2C(n,l,N)], Gak[IDX2C(l,m,N)])),
                            cuCmul(Grkq[IDX2C(n,l,N)], cuCmul(Gakq[IDX2C(l,m,N)], Gak[IDX2C(m,n,N)]))
                        )
                    );
                }
            }
        }
    }

    result = Fac*cuCreal(chi1f);

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

