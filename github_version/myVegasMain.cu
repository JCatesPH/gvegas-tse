#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

// includes, project
#include "helper_cuda.h"
// include initial files

#define __MAIN_LOGIC
#include "vegas.h"
#include "gvegas.h"
#undef __MAIN_LOGIC

#include "kernels.h"

int main(int argc, char* argv[])
{

   //------------------
   //  Initialization
   //------------------
   //
   // program interface:
   //   program -n "ncall0" -i "itmx0" -a "nacc" -b "nBlockSize0" -d "ndim0"
   //
   // parameters:
   //   ncall = 1024*ncall0 is the amount of function calls
   //   itmx  = itmx0 is the maximum iterations for the algorithm
   //   acc   = nacc*0.00001f is the desired accuracy
   //   nBlockSize = nBlockSize0 is the size of the CUDA block
   //   ndim = ndim0 is the dimension of the integration space

   int ncall0 = 0;
   int itmx0 = 10;
   int nacc  = 1;
   int nBlockSize0 = 256;
   int ndim0 = 6;
   int c;

   while ((c = getopt (argc, argv, "n:i:a:b:d:")) != -1)
       switch (c)
         {
         case 'n':
           ncall0 = atoi(optarg);
           break;
         case 'i':
           itmx0 = atoi(optarg);
           break;
         case 'a':
           nacc = atoi(optarg);
           break;
         case 'b':
           nBlockSize0 = atoi(optarg);
           break;
           case 'd':
             ndim0 = atoi(optarg);
             break;
         case '?':
           if (isprint (optopt))
             fprintf (stderr, "Unknown option `-%c'.\n", optopt);
           else
             fprintf (stderr,
                      "Unknown option character `\\x%x'.\n",
                      optopt);
           return 1;
         default:
           abort ();
         }

   ncall = (1 << ncall0)*1024;
   itmx = itmx0;
   acc = (float)nacc*0.000001f;
   nBlockSize = nBlockSize0;
   ndim = ndim0;

   assert(ndim <= ndim_max);

   mds = 1;

   ng = 0;
   npg = 0;

   for (int i=0;i<ndim;i++) { //Choose the box where to integrate
      xl[i] = 0.; //lower bound
      xu[i] = 1.; //upper bound
   }

//If nprn = 1 it prints the whole work, when nprn = 0, just the text in this code.
//If nprn = -1, we can get the grid update information.

  nprn = 1;
//  nprn = -1;
//  nprn = 0;

   double avgi = 0.;
   double sd = 0.;
   double chi2a = 0.;

   myVegas(avgi, sd, chi2a);

   //-------------------------
   //  Print out information
   //-------------------------
   std::cout.clear();
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# No. of Thread Block Size : "<<nBlockSize<<std::endl;
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# No. of dimensions        : "<<ndim<<std::endl;
   std::cout<<"# No. of func calls / iter : "<<ncall<<std::endl;
   std::cout<<"# No. of max. iterations   : "<<itmx<<std::endl;
   std::cout<<"# Desired accuracy         : "<<acc<<std::endl;
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# Answer                   : "<<avgi<<" +- "<<sd<<std::endl;
   std::cout<<"# Chisquare                : "<<chi2a<<std::endl;
   std::cout<<"#==========================="<<std::endl;

   //Print running times!
   std::cout<<"#==========================="<<std::endl;
   printf("# Function call time per iteration: %lf\n", timeVegasCallAndFill/(double)it);
   printf("# Refining time per iteration: %lf\n", timeVegasRefine/(double)it);
   std::cout<<"#==========================="<<std::endl;


    /* Instructions for performance measure
    char archivo[64];
    sprintf(archivo, "./datos/testtoolbox/prodexp.dat");
    FILE *f = fopen(archivo, "ab+");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(f, "%d %d %.9lf %.9lf %lf\n", ndim0, ncall0, avgi, sd, timeVegasCallAndFill+timeVegasRefine);
    fclose(f);
    */
   return 0;
}
