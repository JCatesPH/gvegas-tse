#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

// includes, project
//#include <cutil_inline.h>
// include initial files

#define __MAIN_LOGIC
#include "vegas.h"
#include "gvegas.h"
#undef __MAIN_LOGIC

#include "getrusage_sec.h"
#include "kernels.h"


int main(int argc, char** argv)
{

   //
   // program interface:
   //   program -ncall="ncall0" -itmx="itmx0" -acc="acc0" -b="nBlockSize0"
   //
   // parameters:
   //   ncall0 = "exxny"
   //   ncall = y*10^xx
   //   itmx  = itmx0
   //   acc   = 0.01*acc0
   //   nBlockSize = nBlockSize0
   //

   //------------------
   //  Initialization
   //------------------

   int itmx0 = 10;
   int nBlockSize0 = 256;
   int GPUdevice = 0;

   float acc0 = 0.0001f;

   ncall = 1024*32;
   itmx = itmx0;
   acc = 0.01*acc0;
   nBlockSize = nBlockSize0;

   cudaSetDevice(GPUdevice);

   mds = 1;
   ndim = 3;
   
   ng = 0;
   npg = 0;

   /*-------- Setting integration limits ---------*/
  /* for (int i=0;i<ndim;i++) {
    xl[i] = 0.;
    xu[i] = 1.;
  }*/

  // Based on original description of problem. 
  //"The integrand is Ds(kx,ky,qx,qy)/(2*pi)^3, and the limits of integration are kx=[-pi/a,pi/a],ky=[-pi/a,pi/a] , qx=[-pi/a,pi/a] and qy=[-pi/a,pi/a]."
  //"For qx and qy it is more efficient to use qx=[0.001,pi/a] and qy=0, because of the symmetry of the problem. kx and ky should be as we said before kx=[-pi/a,pi/a],ky=[-pi/a,pi/a]."

  xl[0] = -0.25;
  xu[0] = 0.25;

  xl[1] = -3.14159265358979 / 4.0; // kxi
  xu[1] = 3.14159265358979  / 4.0; // kxf

  xl[2] = -3.14159265358979 / 4.0; // kyi
  xu[2] = 3.14159265358979f / 4.0; // kyf
   
   nprn = 1;
   //   nprn = -1;
   /*----------------------------------------------*/

   double startTotal, endTotal, timeTotal;
   timeTotal = 0.;
   startTotal = getrusage_usec();

   timeVegasCall = 0.;
   timeVegasMove = 0.;
   timeVegasFill = 0.;
   timeVegasRefine = 0.;

   double avgi = 0.;
   double sd = 0.;
   double chi2a = 0.;

   gVegas(avgi, sd, chi2a);

   endTotal = getrusage_usec();
   timeTotal = endTotal - startTotal;

   //-------------------------
   //  Print out information
   //-------------------------
   std::cout.clear();
   std::cout<<std::setw(10)<<std::setprecision(6)<<std::endl;
   std::cout<<"#============================="<<std::endl;
   std::cout<<"# No. of Thread Block Size  : "<<nBlockSize<<std::endl;
   std::cout<<"#============================="<<std::endl;
   std::cout<<"# No. of dimensions         : "<<ndim<<std::endl;
   std::cout<<"# No. of func calls / iter  : "<<ncall<<std::endl;
   std::cout<<"# No. of max. iterations    : "<<itmx<<std::endl;
   std::cout<<"# Desired accuracy          : "<<acc<<std::endl;
   std::cout<<"#============================="<<std::endl;
   std::cout<<std::scientific;
   std::cout<<std::left<<std::setfill(' ');
   std::cout<<"# Result                    : "
            <<std::setw(12)<<std::setprecision(5)<<avgi<<" +- "
            <<std::setw(12)<<std::setprecision(5)<<sd<<" ( "
            <<std::setw(7)<<std::setprecision(4)
            <<std::fixed<<100.*sd/avgi<<"%)"<<std::endl;
   std::cout<<std::fixed;
   std::cout<<"# Chisquare                 : "<<std::setprecision(4)
            <<chi2a<<std::endl;
   std::cout<<"#============================="<<std::endl;
   std::cout<<std::right;
   std::cout<<"# Total Execution Time(sec) : "
            <<std::setw(10)<<std::setprecision(4)<<timeTotal<<std::endl;
   std::cout<<"#============================="<<std::endl;
   std::cout<<"# Time for func calls (sec) : "
            <<std::setw(10)<<std::setprecision(4)<<timeVegasCall
            <<" ( "<<std::setw(5)<<std::setprecision(2)
            <<100.*timeVegasCall/timeTotal<<"%)"<<std::endl;
   std::cout<<"# Time for data transf (sec): "
            <<std::setw(10)<<std::setprecision(4)<<timeVegasMove
            <<" ( "<<std::setw(5)<<std::setprecision(2)
            <<100.*timeVegasMove/timeTotal<<"%)"<<std::endl;
   std::cout<<"# Time for data fill (sec)  : "
            <<std::setw(10)<<std::setprecision(4)<<timeVegasFill
            <<" ( "<<std::setw(5)<<std::setprecision(2)
            <<100.*timeVegasFill/timeTotal<<"%)"<<std::endl;
   std::cout<<"# Time for grid refine (sec): "
            <<std::setw(10)<<std::setprecision(4)<<timeVegasRefine
            <<" ( "<<std::setw(5)<<std::setprecision(2)
            <<100.*timeVegasRefine/timeTotal<<"%)"<<std::endl;
   std::cout<<"#============================="<<std::endl;

   cudaThreadExit();

   return 0;
}
