#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <assert.h>

#include "helper_cuda.h"

#include "cublas_v2.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void mckernel(double& avgi, double& sd)
{
    checkCudaErrors(cudaMemcpyToSymbol(g_ndim, &ndim, sizeof(int)));
    return;
}