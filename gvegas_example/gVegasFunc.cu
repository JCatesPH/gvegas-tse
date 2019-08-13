#include "vegasconst.h"

__device__
double func(double* rx, double wgt)
{

   double value = 0;
   double result = 0;
   // double T = 20; // For the F-D Distribution
   // double mu = 1;
   double c = 0.01; // Constant for M-B Dis. that equals ratio : m / k_B T

   for (int i=0;i<g_ndim;i++) {
      value += rx[i] * rx[i];
   }
   
   // Just return the sum the vector.
   // result = value;

   // sin of vector's sum
   // result = sin(value);

   // Maxwell-Boltzmann Distribution
   result = sqrt(2 * c * c * c / 3.14159) * value * exp(-c / 2 * value);

   // Fermi-Dirac Distribution
   // result = 1 / ( exp((value - mu) / (T)) + 1 );

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

