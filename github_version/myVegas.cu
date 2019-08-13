#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <assert.h>

#include "helper_cuda.h"

#include "vegas.h"
#include "vegasconst.h"
#include "kernels.h"

#include "gvegas.h"

void myVegas(double& avgi, double& sd, double& chi2a)
{

   for (int j=0;j<ndim;j++) {
      xi[j][0] = 1.f;
      for (int i = 1; i < nd_max; i++) {
        xi[j][i] = 0.f;
      }
   }
   /*
   Original code doesn't account for the rest of xi, just assumes that when it
   declares the array, the rest will be set to zeroes, and that will not always
   happen.
   */
   // entry vegas1

   it = 0;

   // entry vegas2
   nd = nd_max;
   ng = 1;

   npg = 0;
   //std::cout<<"mds = "<<mds<<std::endl;
   if (mds!=0) {

      std::cout<<"ncall, ndim = "<<ncall<<", "<<ndim<<std::endl;
      ng = (int)pow((0.5*(double)ncall),1./(double)ndim);
      mds = 1;
      //      printf("ng = %d\n",ng);

      if (2*ng>=nd_max) {
         mds = -1;
         npg = ng/nd_max+1;
         nd = ng/npg;
         ng = npg*nd;
      }

   }
   std::cout<<"mds = "<<mds<<std::endl;
   //assert(mds == 1);

   //std::cout<<"ng = "<<ng<<std::endl;
   checkCudaErrors(cudaMemcpyToSymbol(g_ndim, &ndim, sizeof(int)));
   checkCudaErrors(cudaMemcpyToSymbol(g_ng,   &ng,   sizeof(int)));
   checkCudaErrors(cudaMemcpyToSymbol(g_nd,   &nd,   sizeof(int)));
   cudaThreadSynchronize(); // wait for synchronize

   nCubes = (unsigned)(pow(ng,ndim));
   checkCudaErrors(cudaMemcpyToSymbol(g_nCubes, &nCubes, sizeof(nCubes)));
   cudaThreadSynchronize(); // wait for synchronize

   npg = ncall/nCubes;
   if (npg<2) npg = 2;
   calls = (double)(npg*nCubes);

   unsigned nCubeNpg = nCubes*npg;

   //std::cout<<"nCubes= "<<nCubes<<std::endl;
   //std::cout<<"nCubeNpg= "<<nCubeNpg<<std::endl;

   if (nprn!=0) {
      // tsi = sqrt(tsi);
      std::cout<<std::endl;
      std::cout<<" << vegas internal parameters >>"<<std::endl;
      std::cout<<"            ng: "<<std::setw(5)<<ng<<std::endl;
      std::cout<<"            nd: "<<std::setw(5)<<nd<<std::endl;
      std::cout<<"           npg: "<<std::setw(5)<<npg<<std::endl;
      std::cout<<"        nCubes: "<<std::setw(12)<<nCubes<<std::endl;
      std::cout<<"    nCubes*npg: "<<std::setw(12)<<nCubeNpg<<std::endl;
   }

   dxg = 1.f/(float)ng;
   double dnpg = (double)npg;
   double dv2g = calls*calls*pow(dxg,ndim)*pow(dxg,ndim)/(dnpg*dnpg*(dnpg-1.));
   xnd = (float)nd;
   dxg *= xnd;
   xjac = 1.f/(float)calls;
   for (int j=0;j<ndim;j++) {
      dx[j] = xu[j]-xl[j];
      xjac *= dx[j];
   }

   checkCudaErrors(cudaMemcpyToSymbol(g_npg,  &npg,  sizeof(int)));
   checkCudaErrors(cudaMemcpyToSymbol(g_xjac, &xjac, sizeof(float)));
   checkCudaErrors(cudaMemcpyToSymbol(g_dxg,  &dxg,  sizeof(float)));
   cudaThreadSynchronize(); // wait for synchronize

   ndo = 1;

   if (nd!=ndo) {

      double rc = (double)ndo/xnd;

      for (int j=0;j<ndim;j++) {

         int k = -1;
         double xn = 0.;
         double dr = 0.;
         int i = k;
         k++;
         dr += 1.;
         double xo = xn;
         xn = xi[j][k];
         //         printf("xn = %g\n",xn);
         while (i<nd-1) {

            while (dr<=rc) {
               k++;
               dr += 1.;
               xo = xn;
               xn = xi[j][k];
               //printf("xn = %g\n",xn);
            }
            i++;
            dr -= rc;
            xin[i] = xn - (xn-xo)*dr;
         }

         for (int i=0;i<nd-1;i++) {
            xi[j][i] = (float)xin[i];
         }
         xi[j][nd-1] = 1.f;

      }
      ndo = nd;

   }

   checkCudaErrors(cudaMemcpyToSymbol(g_xl, xl, sizeof(xl)));
   checkCudaErrors(cudaMemcpyToSymbol(g_dx, dx, sizeof(dx)));
   checkCudaErrors(cudaMemcpyToSymbol(g_xi, xi, sizeof(xi)));
   cudaThreadSynchronize(); // wait for synchronize

   if (nprn!=0) {
      std::cout<<std::endl;
      std::cout<<" << input parameters for vegas >>"<<std::endl;
      std::cout<<"     ndim ="<<std::setw(3)<<ndim
               <<"   ncall ="<<std::setw(10)<<(int)calls<<std::endl;
      std::cout<<"     it   =  0"
               <<"   itmx ="<<std::setw(5)<<itmx<<std::endl;
      std::cout<<"     acc  = "<<std::fixed
               <<std::setw(9)<<std::setprecision(3)<<acc<<std::endl;
      std::cout<<"     mds  ="<<std::setw(3)<<mds
               <<"   nd = "<<std::setw(4)<<nd<<std::endl;
      for (int j=0;j<ndim;j++) {
         std::cout<<"    (xl,xu)= ( "<<std::setw(6)<<std::fixed
                  <<xl[j]<<" , "<<xu[j]<<" )"<<std::endl;
      }

   }

   // entry vegas3

   it = 0;
   si = 0.0f;
   si2 = 0.0f;
   swgt = 0.0f;
   schi = 0.0f;
   //   int iflag;
   // main integration loop

   //   std::cout<<"nBlockSize = "<<nBlockSize<<std::endl;
   //--------------------------
   //  Set up kernel variables
   //--------------------------
   //const int nGridSizeMax =  65535; //Original - Maximum size of grid in X for Fermi.
   const int nGridSizeMax = 1<<31 - 1; //This should be the one for current architectures.
   float hd[ndim_max][nd_max];

   dim3 ThBk(nBlockSize);

   int nGridSizeX, nGridSizeY;
   int nBlockTot = (nCubes-1)/nBlockSize+1;
   //std::cout<<"nBlockTot = "<<nBlockTot<<std::endl;
   nGridSizeY = (nBlockTot-1)/nGridSizeMax+1;
   nGridSizeX = (nBlockTot-1)/nGridSizeY+1;
   //std::cout<<"nGridSize (x,y) = "<<nGridSizeX<<", "<<nGridSizeY<<std::endl;
   dim3 BkGd(nGridSizeX, nGridSizeY);
   
   // Get a good grid for initzero()
   dim3 InitZeroTh(ndim,nd);

   if (nprn!=0) {
      std::cout<<std::endl;
      std::cout<<" << kernel parameters for CUDA >>"<<std::endl;
      std::cout<<"       Block size           ="<<std::setw(7)<<ThBk.x<<std::endl;
      std::cout<<"       Grid size            ="<<std::setw(7)<<BkGd.x
               <<" x "<<BkGd.y<<std::endl;
      int nThreadsTot = ThBk.x*BkGd.x*BkGd.y;
      std::cout<<"     Actual Number of calls ="<<std::setw(12)
               <<nThreadsTot*npg<<std::endl;
      std::cout<<"   Required Number of calls ="<<std::setw(12)
               <<nCubeNpg<<" ( "<<std::setw(6)<<std::setprecision(2)
               <<100.*(double)nCubeNpg/(double)(nThreadsTot*npg)<<"%)"<<std::endl;
      std::cout<<std::endl;
   }

   //By using the new GPU kernel we eliminate the need to move big stuff from
   //GPU to CPU and the need to run the Fill part.
   double startVegasCallAndFill, endVegasCallAndFill;
   double startVegasRefine, endVegasRefine;

   initzero<<<1,InitZeroTh>>>();
   getLastCudaError("initzero error");


   do {

      it++;

//      std::cout<<"call gVegasCallFunc: it = "<<it<<std::endl;
      startVegasCallAndFill = omp_get_wtime();

      // Initialize all values to zero, need to make a grid good enough to make everything faster...
      //initzero<<<1, 1>>>();
      // Now CallFilla will need a number of threads equal to the amount of cubes!
      myVegasCallFilla<<<BkGd, ThBk>>>(mds);
      getLastCudaError("myVegasCallFilla error");
      cudaThreadSynchronize(); // wait for synchronize
      checkCudaErrors(cudaMemcpyFromSymbol(&ti, doubleti, sizeof(double)));
      checkCudaErrors(cudaMemcpyFromSymbol(&tsi, doubletsi, sizeof(double)));
      checkCudaErrors(cudaMemcpyFromSymbol(&hd, d, sizeof(d)));
      //checkCudaErrors(cudaMemcpyFromSymbol(&hd, d, ndim_max*nd_max*sizeof(float)));

      endVegasCallAndFill = omp_get_wtime();
      timeVegasCallAndFill += endVegasCallAndFill-startVegasCallAndFill;

      //Initialize to zero before starting CPU computations to do everything at the same time
      initzero<<<1,InitZeroTh>>>();
      getLastCudaError("initzero error");

      tsi *= dv2g;
      double ti2 = (double)ti*(double)ti;
      double wgt = ti2/(double)tsi;
      si += ti*wgt;
      si2 += ti2;
      swgt += wgt;
      schi += ti2*wgt;
      avgi = si/swgt;
      sd = swgt*it/si2;
      chi2a = 0.;
      if (it>1) chi2a = sd*(schi/swgt-avgi*avgi)/((double)it-1.);
      sd = sqrt(1./sd);

      if (nprn!=0) {
         tsi = sqrt(tsi);
         std::cout<<std::endl;
         std::cout<<" << integration by vegas >>"<<std::endl;
         std::cout<<"     iteration no. "<<std::setw(4)<<it
                  <<"   integral=  "<<ti<<std::endl;
         std::cout<<"                          std dev  = "<<tsi<<std::endl;
         std::cout<<"     accumulated results: integral = "<<avgi<<std::endl;
         std::cout<<"                          std dev  = "<<sd<<std::endl;
	 if (it > 1) {
            std::cout<<"                          chi**2 per it'n = "
                     <<std::setw(10)<<std::setprecision(6)<<chi2a<<std::endl;
         }
         if (nprn<0) {
            for (int j=0;j<ndim;j++) {
               std::cout<<"   == data for axis "
                        <<std::setw(2)<<j<<" --"<<std::endl;
               std::cout<<"    x    delt i   convce";
               std::cout<<"    x    delt i   convce";
               std::cout<<"    x    delt i   convce"<<std::endl;

               for (int i=0;i<nd;i+=3) {
                  std::cout<<std::setw(6)<<std::setprecision(6)<<std::setfill(' ')
                           <<xi[j][i]<<" "<<hd[j][i]<<" "<<hd[j][i];
                  std::cout<<std::setw(6)<<std::setprecision(4)
                           <<xi[j][i+1]<<" "<<hd[j][i+1]<<" "<<hd[j][i+1];
                  std::cout<<std::setw(6)<<std::setprecision(4)
                           <<xi[j][i+2]<<" "<<hd[j][i+2]<<" "<<hd[j][i+2]
                           <<std::endl;
                           }

            }
         }
      }

      // refine grid

      startVegasRefine = omp_get_wtime();

      /*
      for (int ii=0;ii<ndim;ii++) {
         for (int jj=0;jj<nd;jj++) {
            std::cout<<"d["<<ii<<"]["<<jj<<"] = "<<std::scientific
                     <<d[ii][jj]<<std::endl;
         }
      }
      */

      double r[nd_max];
      double dt[ndim_max];
      for (int j=0;j<ndim;j++) {
         double xo = hd[j][0];
         double xn = hd[j][1];
         hd[j][0] = 0.5*(xo+xn);
         dt[j] = hd[j][0];
         for (int i=1;i<nd-1;i++) {
            hd[j][i] = xo+xn;
            xo = xn;
            xn = hd[j][i+1];
            hd[j][i] = (hd[j][i]+xn)/3.;
            dt[j] += hd[j][i];
         }
         hd[j][nd-1] = 0.5*(xn+xo);
         dt[j] += hd[j][nd-1];
      }

      for (int j=0;j<ndim;j++) {
         double rc = 0.;
         for (int i=0;i<nd;i++) {
            r[i] = 0.;
            if (hd[j][i]>0.) {
               double xo = dt[j]/hd[j][i];
               if (!isinf(xo))
                  r[i] = pow(((xo-1.)/xo/log(xo)),alph);
            }
            rc += r[i];
         }
         rc /= xnd;
         int k = -1;
         double xn = 0.;
         double dr = xn;
         int i = k;
         k++;
         dr += r[k];
         double xo = xn;
         xn = xi[j][k];

         do {

            while (dr<=rc) {
               k++;
               dr += r[k];
               xo = xn;
               xn = xi[j][k];
            }
            i++;
            dr -= rc;
            xin[i] = xn-(xn-xo)*dr/r[k];

         } while (i<nd-2);

         for (int i=0;i<nd-1;i++) {
            xi[j][i] = (float)xin[i];
         }
         xi[j][nd-1] = 1.f;

      }
      checkCudaErrors(cudaMemcpyToSymbol(g_xi, xi, sizeof(xi)));
      cudaThreadSynchronize(); // wait for synchronize

      endVegasRefine = omp_get_wtime();
      timeVegasRefine += endVegasRefine-startVegasRefine;

//      std::cout<<"The end of main loop: it, sd/avgi = "<<it<<", "
//               <<sd/fabs(avgi)<<std::endl;

   } while (it<itmx && acc*fabs(avgi)<sd);

}
