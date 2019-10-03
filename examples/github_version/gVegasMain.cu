#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

// includes, project
#include "helper_cuda.h"
// include initial files

#define __MAIN_LOGIC
#include "vegas.h"
#include "gvegas.h"
#undef __MAIN_LOGIC

#include "kernels.h"

int main(int argc, char** argv)
{

   //------------------
   //  Initialization
   //------------------
   //
   // program interface:
   //   program -n="ncall0" -i="itmx0" -a="nacc" -b="nBlockSize0"
   //
   // parameters:
   //   ncall = 1024*ncall0
   //   itmx  = itmx0
   //   acc   = nacc*0.00001f
   //   nBlockSize = nBlockSize0
   //

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

   for (int i=0;i<ndim;i++) {
      xl[i] = 0.;
      xu[i] = 1.;
   }
   //If nprn = 1 it prints the whole work, when nprn = 0, just the text in this code
   //If nprn = -1, we can get the grid update information.

   nprn = 1;
//   nprn = -1;
//  nprn = 0;

   double avgi = 0.;
   double sd = 0.;
   double chi2a = 0.;

   gVegas(avgi, sd, chi2a);

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

   cudaThreadExit();

   //Print running times!
   std::cout<<"#==========================="<<std::endl;
   std::cout<<"# Function call time per iteration: " <<timeVegasCall/(double)it<<std::endl;
   std::cout<<"# Values moving time per iteration: " <<timeVegasMove/(double)it<<std::endl;
   std::cout<<"# Filling (reduce) time per iteration: " <<timeVegasFill/(double)it<<std::endl;
   std::cout<<"# Refining time per iteration: " <<timeVegasRefine/(double)it<<std::endl;
   std::cout<<"#==========================="<<std::endl;

   /* Instructions for time measure
   
    int qth;
    qth = omp_get_max_threads();
    printf("%d \n", qth);
    char archivo[64];
    sprintf(archivo, "./datos/redtime/gVegas%d/red_d%dn%d.dat", qth, ndim0, ncall0);
    FILE *f = fopen(archivo, "ab+");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(f, "%lf\n", (timeVegasCall+timeVegasMove+timeVegasFill)/(double)it);
    fclose(f);
    */

   return 0;
}
