#include <iostream>
#include <iomanip>
#include <cmath>

//#include <cutil_inline.h>

#include "vegas.h" 
#include "vegasconst.h"
#include "kernels.h"

#include "gvegas.h"

#include "getrusage_sec.h"
#define PI          3.14159265358979f

void gVegas(double& avgi, double& sd, double& chi2a)
{

   for (int j=0;j<ndim;j++) {
      xi[j][0] = 1.;
   }

   // entry vegas1

   it = 0;

   // entry vegas2
   nd = nd_max;
   ng = 1;
   
   npg = 0;
   if (mds!=0) {
      
      ng = (int)pow((0.5*(double)ncall),1./(double)ndim);
      mds = 1;
      if (2*ng>=nd_max) {
         mds = -1;
         npg = ng/nd_max+1;
         nd = ng/npg;
         ng = npg*nd;
      }
      
   }
   cudaMemcpyToSymbol(g_ndim, &ndim, sizeof(int));
   cudaMemcpyToSymbol(g_ng,   &ng,   sizeof(int));
   cudaMemcpyToSymbol(g_nd,   &nd,   sizeof(int));
   cudaThreadSynchronize(); // wait for synchronize

   nCubes = (unsigned)(pow(ng,ndim));
   cudaMemcpyToSymbol(g_nCubes, &nCubes, sizeof(nCubes));
   cudaThreadSynchronize(); // wait for synchronize

   npg = ncall/nCubes;
   if (npg<2) npg = 2;
   calls = (double)(npg*nCubes);

   unsigned nCubeNpg = nCubes*npg;

   if (nprn!=0) {
      std::cout<<std::endl;
      std::cout<<" << vegas internal parameters >>"<<std::endl;
      std::cout<<"            ng: "<<std::setw(5)<<ng<<std::endl;
      std::cout<<"            nd: "<<std::setw(5)<<nd<<std::endl;
      std::cout<<"           npg: "<<std::setw(5)<<npg<<std::endl;
      std::cout<<"        nCubes: "<<std::setw(12)<<nCubes<<std::endl;
      std::cout<<"    nCubes*npg: "<<std::setw(12)<<nCubeNpg<<std::endl;
   }
   
   dxg = 1./(double)ng;
   double dnpg = (double)npg;
   double dv2g = calls*calls*pow(dxg,ndim)*pow(dxg,ndim)/(dnpg*dnpg*(dnpg-1.));
   xnd = (double)nd;
   dxg *= xnd;
   xjac = 1./(double)calls;
   for (int j=0;j<ndim;j++) {
      dx[j] = xu[j]-xl[j];
      xjac *= dx[j];
   }

   cudaMemcpyToSymbol(g_npg,  &npg,  sizeof(int));
   cudaMemcpyToSymbol(g_xjac, &xjac, sizeof(double));
   cudaMemcpyToSymbol(g_dxg,  &dxg,  sizeof(double));
   cudaThreadSynchronize(); // wait for synchronize

   //----------------------------------
   //  Set parameters in the integrand.
   //----------------------------------
   mu_h     = 0.1f;
   hOmg_h   = 0.5f;
   a_h      = 3.6f;
   A_h      = 4.f;
   rati_h   = 0.1f;
   eE0_h    = rati_h * (hOmg_h * hOmg_h) / (2 * sqrt(A_h * mu_h));
   Gamm_h   = 0.003f;
   KT_h     = 1e-6;
   shift_h  = A_h * (eE0_h / hOmg_h) * (eE0_h / hOmg_h);
   Gammsq_h = Gamm_h * Gamm_h;
   N_h      = 7;

   V0_h     = eE0_h * A_h / hOmg_h;
   V2_h     = A_h * (eE0_h / hOmg_h) * (eE0_h / hOmg_h);
   Fac_h    = -(4.f * Gamm_h / (PI*PI));
   qx_h     = 0.5; //0.01f + (PI / a_h) * 30.f / 50.f;
   qy_h     = 0.f;


   // Move the parameters to the GPU memory.
   cudaMemcpyToSymbol(mu,          &mu_h,  sizeof(float));
   cudaMemcpyToSymbol(hOmg,      &hOmg_h,  sizeof(float));
   cudaMemcpyToSymbol(a,            &a_h,  sizeof(float));
   cudaMemcpyToSymbol(A,            &A_h,  sizeof(float));
   cudaMemcpyToSymbol(rati,      &rati_h,  sizeof(float));
   cudaMemcpyToSymbol(eE0,        &eE0_h,  sizeof(float));
   cudaMemcpyToSymbol(Gamm,      &Gamm_h,  sizeof(float));
   cudaMemcpyToSymbol(KT,          &KT_h,  sizeof(float));
   cudaMemcpyToSymbol(shift,    &shift_h,  sizeof(float));
   cudaMemcpyToSymbol(Gammsq,  &Gammsq_h,  sizeof(float));
   cudaMemcpyToSymbol(N,            &N_h,  sizeof(int));
   
   cudaMemcpyToSymbol(V0,  &V0_h,  sizeof(float));
   cudaMemcpyToSymbol(V2,  &V2_h,  sizeof(float));
   cudaMemcpyToSymbol(Fac,  &Fac_h,  sizeof(float));
   cudaMemcpyToSymbol(qx,  &qx_h,  sizeof(float));
   cudaMemcpyToSymbol(qy,  &qy_h,  sizeof(float));
   cudaThreadSynchronize(); // wait for synchronize
   //----------------------------------

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

         while (i<nd-1) {

            while (dr<=rc) {
               k++;
               dr += 1.;
               xo = xn;
               xn = xi[j][k];
            }
            i++;
            dr -= rc;
            xin[i] = xn - (xn-xo)*dr;
         }
         
         for (int i=0;i<nd-1;i++) {
            xi[j][i] = (double)xin[i];
         }
         xi[j][nd-1] = 1.;

      }
      ndo = nd;
      
   }

   cudaMemcpyToSymbol(g_xl, xl, sizeof(xl));
   cudaMemcpyToSymbol(g_dx, dx, sizeof(dx));
   cudaMemcpyToSymbol(g_xi, xi, sizeof(xi));
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
   si = 0.;
   si2 = 0.;
   swgt = 0.;
   schi = 0.;

   //--------------------------
   //  Set up kernel vaiables
   //--------------------------
   const int nGridSizeMax =  65535;
   
   dim3 ThBk(nBlockSize);

   int nGridSizeX, nGridSizeY;
   int nBlockTot = (nCubeNpg-1)/nBlockSize+1;
   nGridSizeY = (nBlockTot-1)/nGridSizeMax+1;
   nGridSizeX = (nBlockTot-1)/nGridSizeY+1;
   dim3 BkGd(nGridSizeX, nGridSizeY);

   if (nprn!=0) {
      std::cout<<std::endl;
      std::cout<<" << kernel parameters for CUDA >>"<<std::endl;
      std::cout<<"       Block size           ="<<std::setw(7)<<ThBk.x<<std::endl;
      std::cout<<"       Grid size            ="<<std::setw(7)<<BkGd.x
               <<" x "<<BkGd.y<<std::endl;
      int nThreadsTot = ThBk.x*BkGd.x*BkGd.y;
      std::cout<<"     Actual Number of calls ="<<std::setw(12)
               <<nThreadsTot<<std::endl;
      std::cout<<"   Required Number of calls ="<<std::setw(12)
               <<nCubeNpg<<" ( "<<std::setw(6)<<std::setprecision(2)
               <<100.*(double)nCubeNpg/(double)nThreadsTot<<"%)"<<std::endl;
      std::cout<<std::endl;
   }
      
   // allocate Fval
   int sizeFval = nCubeNpg*sizeof(double);

   // CPU
   double* hFval;
   cudaMallocHost((void**)&hFval, sizeFval);
   memset(hFval, '\0', sizeFval);

   // GPU
   double* gFval;
   cudaMalloc((void**)&gFval, sizeFval);

   // allocate IAval
   int sizeIAval = nCubeNpg*ndim*sizeof(int);

   // CPU
   int* hIAval;
   cudaMallocHost((void**)&hIAval, sizeIAval);
   memset(hIAval, '\0', sizeIAval);

   // GPU
   int* gIAval;
   cudaMalloc((void**)&gIAval, sizeIAval);

   double startVegasCall, endVegasCall;
   double startVegasMove, endVegasMove;
   double startVegasFill, endVegasFill;
   double startVegasRefine, endVegasRefine;

   //==============================================================//
   // Attempt to expand heap size. Perhaps the memory allocation in the device is too large.
   int sizeComplex = (4 * N_h * N_h + 9 * N_h + 24 + 2) * sizeof(cuDoubleComplex);
   cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeComplex * 1024 * 1024);
   //==============================================================//

   do {
      
      it++;

      startVegasCall = getrusage_usec();
      gVegasCallFunc<<<BkGd, ThBk>>>(gFval, gIAval);
      cudaThreadSynchronize(); // wait for synchronize
      endVegasCall = getrusage_usec();
      timeVegasCall += endVegasCall-startVegasCall;

      startVegasMove = getrusage_usec();
      cudaMemcpy(hFval, gFval,  sizeFval,
                               cudaMemcpyDeviceToHost);

      cudaMemcpy(hIAval, gIAval,  sizeIAval,
                               cudaMemcpyDeviceToHost);
      endVegasMove = getrusage_usec();
      timeVegasMove += endVegasMove-startVegasMove;

// *****************         

      startVegasFill = getrusage_usec();

      ti = 0.;
      tsi = 0.;

      double d[ndim_max][nd_max];

      for (int j=0;j<ndim;++j) {
         for (int i=0;i<nd;++i) {
            d[j][i] = 0.;
         }
      }

      for (unsigned ig=0;ig<nCubes;ig++) {
         double fb = 0.;
         double f2b = 0.;
         for (int ipg=0;ipg<npg;ipg++) {
            int idx = npg*ig+ipg;
            double f = hFval[idx];
            double f2 = f*f;
            fb += f;
            f2b += f2;
         }
         f2b = sqrt(f2b*npg);
         f2b = (f2b-fb)*(f2b+fb);
         ti += fb;
         tsi += f2b;
         if (mds<0) {
            int idx = npg*ig;
            for (int idim=0;idim<ndim;idim++) {
               int iaj = hIAval[idim*nCubeNpg+idx];
               d[idim][iaj] += f2b;
            }
         }
      }

      if (mds>0) {
         for (int idim=0;idim<ndim;idim++) {
            int idimCube = idim*nCubeNpg;
            for (int idx=0;idx<nCubeNpg;idx++) {
               double f = hFval[idx];
               int iaj = hIAval[idimCube+idx];
               d[idim][iaj] += f*f;
            }
         }
      }

      endVegasFill = getrusage_usec();
      timeVegasFill += endVegasFill-startVegasFill;

      tsi *= dv2g;
      double ti2 = ti*ti;
      double wgt = ti2/tsi;
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
                  <<std::setw(10)<<std::setprecision(6)
                  <<"   integral=  "<<ti<<std::endl;
         std::cout<<"                          std dev  = "<<tsi<<std::endl;
         std::cout<<"     accumulated results: integral = "<<avgi<<std::endl;
         std::cout<<"                          std dev  = "<<sd<<std::endl;
	 if (it > 1) {
            std::cout<<"                          chi**2 per it'n = "
                     <<std::setw(10)<<std::setprecision(4)<<chi2a<<std::endl;
         }
         if (nprn<0) {
            for (int j=0;j<ndim;j++) {
               std::cout<<"   == data for axis "
                        <<std::setw(2)<<j<<" --"<<std::endl;
               std::cout<<"    x    delt i   convce";
               std::cout<<"    x    delt i   convce";
               std::cout<<"    x    delt i   convce"<<std::endl;
            }
         }
      }

      // refine grid

      startVegasRefine = getrusage_usec();
      
      double r[nd_max];
      double dt[ndim_max];
      for (int j=0;j<ndim;j++) {
         double xo = d[j][0];
         double xn = d[j][1];
         d[j][0] = 0.5*(xo+xn);
         dt[j] = d[j][0];
         for (int i=1;i<nd-1;i++) {
            d[j][i] = xo+xn;
            xo = xn;
            xn = d[j][i+1];
            d[j][i] = (d[j][i]+xn)/3.;
            dt[j] += d[j][i];
         }
         d[j][nd-1] = 0.5*(xn+xo);
         dt[j] += d[j][nd-1];
      }
      
      for (int j=0;j<ndim;j++) {
         double rc = 0.;
         for (int i=0;i<nd;i++) {
            r[i] = 0.;
            if (d[j][i]>0.) {
               double xo = dt[j]/d[j][i];
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
            xi[j][i] = (double)xin[i];
         }
         xi[j][nd-1] = 1.;

      }
      cudaMemcpyToSymbol(g_xi, xi, sizeof(xi));
      cudaThreadSynchronize(); // wait for synchronize

      endVegasRefine = getrusage_usec();
      timeVegasRefine += endVegasRefine-startVegasRefine;
      
   } while (it<itmx && acc*fabs(avgi)<sd);


   cudaFreeHost(hFval);
   cudaFree(gFval);

   cudaFreeHost(hIAval);
   cudaFree(gIAval);

}
