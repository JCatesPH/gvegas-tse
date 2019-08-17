#include "vegasconst.h"
#include "const.h"

__device__
double func(double* rx, double wgt)
{

   double value = 0;
   double result = 0;
   // double c = 0.01; // Constant for M-B Dis. that equals ratio : m / k_B T
   // double T = 20; // For the F-D Distribution
   // double mu = 1;
   // double sig = 1.0; // For the Gaussian Distribution
   // double mu  = 0.0;

   for (int i=0;i<g_ndim;i++) {
      // value += rx[i]; // Simple sum (for F-D)
      // value *= rx[i]; // Simple product
      // value += rx[i] * rx[i]; // Sum of squares (for M-B or Singular)
      // value += (rx[i] - mu) * (rx[i] - mu);
      value += cos(log(rx[i]) / rx[i]) / rx[i]
   }
   
   // Just return the sum the vector.
   result = value;

   // sin of vector's sum
   // result = sin(value);

   // Maxwell-Boltzmann Distribution
   // result = sqrt(2 * c * c * c / PI) * value * exp(-c / 2 * value);

   // Fermi-Dirac Distribution
   // result = 1 / ( exp((value - mu) / (T)) + 1 );

   // Gaussian Distribution
   // result = 1 / (sqrt(2 * PI * sig * sig)) * exp(-value / (2 * sig * sig));

   // Singular Example from Mathematica (Numerator determines spreading)
   // result = 1 / sqrt(value);

   return result;

}

/*
__device__
double func(double* rx, double wgt)
{
   double value = 1.;
   for (int i=0;i<g_ndim;i++) {
      value *= 2.*rx[i];
   }
   return value;

}
*/

