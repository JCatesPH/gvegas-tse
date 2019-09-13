#include "vegasconst.h"

#include <cuComplex.h> // Complex number module of cuda.

#define CUDART_PI_F 3.141592654f

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

// ztest10: Bessel function test
__device__
float func(float* rx, float wgt)
{
    return jnf(0, rx[0]+rx[1]);
}


/*

// ztest8-9: complex vectors using for-loop assignment and operations.
__device__
float func(float* rx, float wgt)
{
    cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
    cuFloatComplex *vector;
    vector = (cuFloatComplex*)malloc(4*sizeof(cuFloatComplex));

    //float fermi = heaviside(rx[0], 5);

    //vector[0] = make_cuFloatComplex(rx[0], rx[1]);
    //vector[1] = make_cuFloatComplex(rx[2], rx[3]);
    //vector[2] = make_cuFloatComplex(rx[4], rx[5]);
    //vector[3] = make_cuFloatComplex(rx[6], rx[7]);

    for(int i=0; i<4; i++)
    {
        vector[i] = make_cuFloatComplex(rx[2*i], rx[2*i+1]);
    }
    
    for(int i=0; i<4; i=i+2)
    {
        sum = cuCaddf(sum, cuCdivf(vector[i], vector[i+1]));
    }

    //sum = cuCaddf(vector[0], vector[1]);
    
    free(vector);

    return cuCrealf(sum);
}

*/

/*

// ztest7: Testing other device function calls.
__device__
float func(float* rx, float wgt)
{
    cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
    cuFloatComplex *vector;
    vector = (cuFloatComplex*)malloc(2*sizeof(cuFloatComplex));

    float fermi = heaviside(rx[0], 5);

    vector[0] = make_cuFloatComplex(fermi, rx[1]);
    vector[1] = make_cuFloatComplex(rx[2], rx[3]);
    
    sum = cuCaddf(vector[0], vector[1]);
    
    free(vector);

    return cuCrealf(sum);
}

*/

/*

// ztest6: Testing complex vectors.
__device__
float func(float* rx, float wgt)
{
    cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
    cuFloatComplex *vector;
    vector = (cuFloatComplex*)malloc(2*sizeof(cuFloatComplex));

    vector[0] = make_cuFloatComplex(rx[0], rx[1]);
    vector[1] = make_cuFloatComplex(rx[2], rx[3]);
    
    sum = cuCaddf(vector[0], vector[1]);
    
    free(vector);

    return cuCrealf(sum);
}

*/

/*

// ztest5, dbltest
__device__
float func(float* rx, float wgt)
{
    double sum = 0;
    double *vector; 
    vector = (double*)malloc(4*sizeof(double));

    for (int j=0; j<4; j++) {
        vector[j] = rx[j];
    }
    
    for (int j=0; j<4; j++) {
        sum += vector[j];
    }

    free(vector);

    return (float)sum;
}
*/

/*

// ztest2/3/4: Testing complex math operations.
__device__
float func(float* rx, float wgt)
{
    cuFloatComplex z1 = make_cuFloatComplex(rx[0], rx[1]);
    cuFloatComplex z2 = make_cuFloatComplex(rx[2], rx[3]);
    
    cuFloatComplex result;

    result = cuCdivf(z1, z2);

    return A * cuCrealf(result);
}

*/


/*

// ztest1
__device__
float func(float* rx, float wgt)
{
    cuFloatComplex sum = make_cuFloatComplex(0.f, 0.f);
    cuFloatComplex *vector;
    vector = (cuFloatComplex*)malloc(1*sizeof(cuFloatComplex));

    // int i = 0;
    for (int j=0; j<3; j++) {
        vector[j] = make_cuFloatComplex(rx[0], rx[1]);
    // i++;
    }

    for (int j=0; j<3; j++) {
        sum = cuCaddf(sum, vector[j]);
    }
    
    free(vector);

    return cuCrealf(sum);
}

*/
