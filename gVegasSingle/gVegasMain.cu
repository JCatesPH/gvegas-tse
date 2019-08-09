#include <cstdlib>
#include <iostream>

#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

// includes, project
// #include <cutil_inline.h>
// include initial files

#define __MAIN_LOGIC
#include "vegas.h"
#include "gvegas.h"
#undef __MAIN_LOGIC

#include "kernels.h"

double getrusage_sec()
{
   struct rusage t;
   struct timeval tv;
   getrusage(RUSAGE_SELF, &t);
   tv = t.ru_utime;
   return tv.tv_sec + (double)tv.tv_usec*1e-6;
}

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

   int ncall0 = 256;
   int itmx0 = 10;
   int nacc  = 1;
   int nBlockSize0 = 256;

   // cutGetCmdLineArgumenti(argc, (const char**)argv, "n", &ncall0);
   // cutGetCmdLineArgumenti(argc, (const char**)argv, "i", &itmx0);
   // cutGetCmdLineArgumenti(argc, (const char**)argv, "a", &nacc);
   // cutGetCmdLineArgumenti(argc, (const char**)argv, "b", &nBlockSize0);

   ncall = ncall0*1024;
   itmx = itmx0;
   acc = (float)nacc*0.00001f;
   nBlockSize = nBlockSize0;

   cudaSetDevice(0);

   mds = 1;
   ndim = 8;
   
   ng = 0;
   npg = 0;

   for (int i=0;i<ndim;i++) {
      xl[i] = 0.;
      xu[i] = 1.;
   }
   
   nprn = 1;
//   nprn = -1;

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

   return 0;
}
