import math
import time
import numpy as np
import numba
import ZMCIntegral
from numba import cuda

mu = 0.1  # Fermi-level
hOmg = 0.5 # Photon energy eV
a = 3.6  # Latice constant in AA (Cu)
A = 4.  # hbar^2/(2m)=4 evAA^2 (for free electron mass=~Cu)
rati = 0.1   # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * math.sqrt(A * mu))  # eE0
Gamm = 0.003  # Gamma in eV.
KT = 1. * 10. ** (-6)  #
shift = A * (eE0 / hOmg) ** 2  # Shift of the bands
V0 = eE0 * A / hOmg  # Paramiter of HF
V2 = A * (eE0 / hOmg) ** 2  # Pramiter of HF
Fac = -(4 * Gamm / (math.pi ** 2))
N=3
N1 = N + 1
N2 = N + 2
qy = 0.0



def setqx(qxi):
    global qx
    qx = qxi
    return



@cuda.jit(device=True)
def getqx():
    return qx



@cuda.jit(device=True)
def d(x):

    qx = getqx()

    vkqt = complex(-V0 * (x[2] + qy), -V0 * (x[1] + qx))
    vkqb = complex(vkqt.real, -vkqt.imag)

    vkt = complex(-V0 * (x[2]), -V0 * (x[1]))
    vkb = complex(vkt.real, -vkt.imag)

    thekq = numba.cuda.local.array(N+1, dtype=numba.types.complex64)
    phikq = numba.cuda.local.array(N+2, dtype=numba.types.complex64)

    thek = numba.cuda.local.array(N+1, dtype=numba.types.complex64)
    phik = numba.cuda.local.array(N+2, dtype=numba.types.complex64)

    Grkq = numba.cuda.local.array((N, N), dtype=numba.types.complex64)
    Grk = numba.cuda.local.array((N, N), dtype=numba.types.complex64)
    Gakq = numba.cuda.local.array((N, N), dtype=numba.types.complex64)
    Gak = numba.cuda.local.array((N, N), dtype=numba.types.complex64)

    pvkqt = numba.cuda.local.array(N, dtype=numba.types.complex64)
    pvkqb = numba.cuda.local.array(N, dtype=numba.types.complex64)
    pvkt = numba.cuda.local.array(N, dtype=numba.types.complex64)
    pvkb = numba.cuda.local.array(N, dtype=numba.types.complex64)
    mp = numba.cuda.local.array(N, dtype=numba.types.complex64)

    thzkq = complex(1.0, 0.0)
    thokq = complex(x[0] - A * ((x[1] + qx) ** 2 + (x[2] + qy) ** 2) - V2 - ((N - 1) / 2) * hOmg, Gamm)

    thzk = complex(1.0, 0.0)
    thok = complex(x[0] - A * ((x[1]) ** 2 + (x[2]) ** 2) - V2 - ((N - 1) / 2) * hOmg, Gamm)

    phinpkq = complex(1.0, 0.0)
    phinkq = complex(x[0] - A * ((x[1] + qx) ** 2 + (x[2] + qy) ** 2) - V2 - ((N - 1) / 2 - (N - 1)) * hOmg, Gamm)

    phinpk = complex(1.0, 0.0)
    phink = complex(x[0] - A * ((x[1]) ** 2 + (x[2]) ** 2) - V2 - ((N - 1) / 2 - (N - 1)) * hOmg, Gamm)
##
    thekq[0] = thzkq
    thekq[1] = thokq

    thek[0] = thzk
    thek[1] = thok

    phikq[0] = complex(0.0, 0.0)
    phikq[N+1] = phinpkq
    phikq[N] = phinkq

    phik[0] = complex(0.0, 0.0)
    phik[N+1] = phinpk
    phik[N] = phink

    vnkqt = complex(1.0, 0.0)
    pvkqt[0] = vnkqt
    vnkqb = complex(1.0, 0.0)
    pvkqb[0] = vnkqb
    vnkt = complex(1.0, 0.0)
    pvkt[0] = vnkt
    vnkb = complex(1.0, 0.0)
    pvkb[0] = vnkb

    mn = complex(1.0, 0.0)
    mp[0] = mn

    for ss in range(2, N + 1):
        s = int(N - ss + 1)
        n = int(ss - 1)

        theskq = complex(x[0] - A * ((x[1] + qx) ** 2 + (x[2] + qy) ** 2) - V2 - ((N - 1) / 2 - float(ss - 1)) * hOmg,
                         Gamm) * thokq - vkqt * vkqb * thzkq
        thzkq = thokq
        thokq = theskq

        thekq[ss] = thokq

        phiskq = complex(x[0] - A * ((x[1] + qx) ** 2 + (x[2] + qy) ** 2) - V2 - ((N - 1) / 2 - float(s - 1)) * hOmg,
                         Gamm) * phinkq - vkqt * vkqb * phinpkq
        phinpkq = phinkq
        phinkq = phiskq

        phikq[s] = phinkq

        thesk = complex(x[0] - A * ((x[1]) ** 2 + (x[2]) ** 2) - V2 - ((N - 1) / 2 - float(ss - 1)) * hOmg,
                        Gamm) * thok - vkt * vkb * thzk
        thzk = thok
        thok = thesk

        thek[ss] = thok

        phisk = complex(x[0] - A * ((x[1]) ** 2 + (x[2]) ** 2) - V2 - ((N - 1) / 2 - float(s - 1)) * hOmg,
                        Gamm) * phink - vkt * vkb * phinpk
        phinpk = phink
        phink = phisk

        phik[s] = phink

        vnkqt = vnkqt*vkqt
        vnkt = vnkt*vkt
        vnkqb = vnkqb*vkqb
        vnkb = vnkb*vkb

        pvkqt[n] = vnkqt
        pvkt[n] = vnkt

        pvkqb[n] = vnkqb
        pvkb[n] = vnkb

        mn = mn*complex(-1.0, 0.0)
        mp[n] = mn

    for m in range(0, N):
        for n in range(m, N):
            if m == n:
                Grkq[m, n] = thekq[m] * phikq[n + 2] / thekq[N]
                Gakq[m, n] = complex(Grkq[m, n].real, -Grkq[m, n].imag)
                Grk[m, n] = thek[m] * phik[n + 2] / thek[N]
                Gak[m, n] = complex(Grk[m, n].real, -Grk[m, n].imag)

            elif m < n:

                Grkq[m, n] = mp[n - m] * pvkqt[n - m] * thekq[m] * phikq[n + 2] / thekq[N]
                Grkq[n, m] = mp[n - m] * pvkqb[n - m] * thekq[m] * phikq[n + 2] / thekq[N]
                Gakq[m, n] = complex(Grkq[n, m].real, -Grkq[n, m].imag)
                Gakq[n, m] = complex(Grkq[m, n].real, -Grkq[m, n].imag)

                Grk[m, n] = mp[n - m] * pvkt[n - m] * thek[m] * phik[n + 2] / thek[N]
                Grk[n, m] = mp[n - m] * pvkb[n - m] * thek[m] * phik[n + 2] / thek[N]
                Gak[m, n] = complex(Grk[n, m].real, -Grk[n, m].imag)
                Gak[n, m] = complex(Grk[m, n].real, -Grk[m, n].imag)

    chi1f = complex(0.0, 0.0)
    for l in range(0, N):
        if (mu - x[0] + ((N - 1) / 2 - l) * hOmg) < 0:
            break
        else:
            for m in range(0, N):
                for n in range(0, N):
                    chi1f += Grkq[m, n] * Grk[n, l] * Gak[l, m] + Grkq[n, l] * Gakq[l, m] * Gak[m, n]

    return (Fac*chi1f).real


omi = -hOmg / 2
omf = hOmg / 2
kxi = -math.pi / a
kxf = math.pi / a
kyi = -math.pi / a
kyf = math.pi / a

spacing = 50

qqx = np.zeros(spacing)
resultArr = np.zeros(spacing)
errorArr = np.zeros(spacing)
timeArr = np.zeros(spacing)

tic = time.time()
j = 0
for i in range(0, 50, 1):
    qqx[j] = 0.01+ (math.pi/a)*i/spacing
    setqx(qqx[j])
    MC = ZMCIntegral.MCintegral(d, [[omi, omf], [kxi, kxf], [kyi, kyf]])
    # Setting the zmcintegral parameters
    MC.depth = 2
    MC.sigma_multiplication = 1000000
    MC.num_trials = 2
    start = time.time()
    result = MC.evaluate()
    print('Result for qx = ',qqx[j], ': ', result[0], ' with error: ', result[1])
    print('================================================================')
    end = time.time()
    print('Computed in ', end-start, ' seconds.')
    print('================================================================')
    resultArr[j] = result[0]
    errorArr[j] = result[1]
    timeArr[j] = end - start
    j = j + 1




print('================================================================')


j = 0
print('All values in csv format:')
print('qx,Integral,Error,Time')
for i in range(100, 200, 1):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (qqx[j],  resultArr[j], errorArr[j], timeArr[j]))
    j = j + 1

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc-tic, 'seconds.')

return complex(math.exp(-(a ** 2) * (x[3] ** 2 + x[4] ** 2) /(math.pi**2)))