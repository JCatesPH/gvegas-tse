#include "vegasconst.h"
#define CUDART_PI_F 3.141592654f

__device__
float sum(float* rx, int dim)
{
	float value = 0.f;
	for (int i = 0; i < dim; i++){
		value += rx[i];
	}
	value = 1.f / sqrt((float)dim/12.f) * (value - (float)dim / 2.f);
	return value;
}

__device__
float sqsum(float* rx, int dim)
{
	float value = 0.f;
	for (int i = 0; i < dim; i++){
		value += rx[i] * rx[i];
	}
	value = sqrtf(45.f / (4.f * (float)dim)) * (value - (float)dim / 3);
	return value;
}

__device__
float sumsqroot(float* rx, int dim)
{
	float value = 0.f;
	for (int i = 0; i < dim; i++){
		value += sqrtf(rx[i]);
	}
	value = sqrtf(18.f / (float)dim) * (value - 2.f/3.f * (float)dim);
	return value;
}

__device__
float prodones(float* rx, int dim)
{
	float value = 1.f;
	for (int i = 0; i < dim; i++){
		value *= copysignf(1.f, rx[i]-0.5f);
	}
	return value;
}

__device__
float prodexp(float* rx, int dim)
{
	float e = sqrtf((15.f * expf(15.f) + 15.f) / (13.f * expf(15.f) + 17.f));
	e = powf(e, float(dim) * 0.5f);
	float value = 1.f;
	for (int i = 0; i < dim; i++){
		value *= ((expf(30.f * rx[i] - 15.f)) - 1.f) / (expf(30.f * rx[i] - 15.f) + 1.f);		
	}
	value *= e;
	return value;
}

__device__
float prodcub(float* rx, int dim)
{
	float value = 1.f;
	for (int i = 0; i < dim; i++){
		value *= (-2.4f*sqrtf(7.f)*(rx[i]-0.5f)+8.f*sqrtf(7.f)*(rx[i]-0.5f)*(rx[i]-0.5f)*(rx[i]-0.5f));
	}
	return value;
}

__device__
//PRODX has a lot of extremes when dimensions are big, it's expected to not do well
float prodx(float* rx, int dim)
{
	float value = 1.f;
	for (int i = 0; i < dim; i++){
		value *= (rx[i] - 0.5f);
	}
	value *= powf(2.f*sqrtf(3.f), (float) dim);
	return value;
}

__device__
float sumfifj(float* rx, int dim)
{
	float value = 0.f;
	for (int i = 0; i < dim; i++){
		float aux = 0.f;
		for (int j = 0; j < i; j++){
			aux += copysignf(1.f,(1.f/6.f-rx[j])*(rx[j]-4.f/6.f));
		}
		value += copysignf(1.f,(1.f/6.f-rx[i])*(rx[i]-4.f/6.f))*aux;
	}
	value *= sqrtf(2.f/(float)(dim*(dim-1)));
	return value;
}

__device__
float sumfonefj(float* rx, int dim)
{
	float value = 0.f;
	for (int i = 1; i < dim; i++){
		value += 27.20917094*rx[i]*rx[i]*rx[i]-36.1925085*rx[i]*rx[i]+8.983337562*rx[i]+0.7702079855;
	}
	value *= (27.20917094*rx[0]*rx[0]*rx[0]-36.1925085*rx[0]*rx[0]+8.983337562*rx[0]+0.7702079855)/sqrtf((float)dim-1.f);
	return value;
}

__device__
float hellekalek(float* rx, int dim)
{
	float value = 1.f;
	for (int i = 0; i < dim; i++){
		value *= ((rx[i] - 0.5f)/sqrtf(12.f));
	}
	return value;
}

__device__
float roosarnoldone(float* rx, int dim)
{
	float value = 1.f/(float)dim;
	float aux = 0.f;
	for (int i = 0; i < dim; i++){
		aux += fabsf(4.f*rx[i]-2.f)-1.f;
	}
	value *= aux;
	return value;
}

__device__
//Can give huge error
float roosarnoldtwo(float* rx, int dim)
{
	float value = sqrtf(1.f/(powf(4.f/3.f, (float)dim)-1.f));
	for (int i = 0; i < dim; i++){
		value *= (fabsf(4.f*rx[i]-2.f) - 1.f);
	}
	return value;
}

__device__
float roosarnoldthree(float* rx, int dim)
{
	float value = 1.f/sqrtf(powf(CUDART_PI_F*CUDART_PI_F/8.f, (float)dim)-1.f);
	for (int i = 0; i < dim; i++){
		value *= (CUDART_PI_F/2.f*sinf(CUDART_PI_F*rx[i])-1.f);
	}
	return value;
}

__device__
//Choosing only RST1, since it's the most difficult.
float rst(float* rx, int dim)
{
	float value = 1.f/sqrtf(powf(1.f+1.f/12.f,(float)dim)-1.f);
	for (int i = 0; i < dim; i++){
		value *= ((fabsf(4.f*rx[i]-2.f)+1.f)/2.f-1.f);
	}
	return value;
}

__device__
float sobolprod(float* rx, int dim)
{
	float value = 1.f;
	for (int i = 0; i < dim; i++){
		value *= (1.f+1.f/((float)(3*(i+2)*(i+2))));
	}
	value = sqrtf(1.f/(value-1.f));
	for (int i = 0; i < dim; i++){
		value *= ((float)(i+1)+2.f*rx[i])/(float)(i + 2)-1.f;
	}
	return value;
}

__device__
//Choosing beta = 1 and alpha_i = 1 for every i.
float oscill(float* rx, int dim)
{
	float value = 2.f*CUDART_PI_F;
	float p = 1.f;
	for (int i = 0; i < dim; i++){
		value += rx[i];
		p *= sinf(0.5f);
	}
	value = cosf(value)-powf(2.f, (float)dim)*cosf(2.f*CUDART_PI_F+0.5f*(float)dim)*p;
	return value;
}

__device__
//Choosing beta_i = 0.5 and alpha_i = 1 for every i.
float prpeak(float* rx, int dim)
{
	float value = 1.f;
	float e = 1.f;
	for (int i = 0; i < dim; i++){
		value *= 1.f/(1+(rx[i]-0.5f)*(rx[i]-0.5f));
		e *= (atanf(0.5f)-atanf(-0.5f));
	}
	value += -e;
	return value;
}

//There are 4 functions missing from the document (CORPEAK, GAUSSIAN, C0 and DISCONT), but it gets really hard from here on to estimate numbers and I prefer stopping here. 17 is a good enough number of test functions.
