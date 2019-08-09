#include <iostream>
#include <iomanip>
#include <cmath>

#include "cutil_inline.h"

#include "vegas.h" 
#include "vegasconst.h"
#include "kernels.h"

#include "gvegas.h"

double getrusage_sec();

void gVegas(double& avgi, double& sd, double& chi2a)
{

   for (int j=0;j<ndim;j++) {
      xi[j][0] = 1.f;
   }

   // entry vegas1

   it = 0;

   // entry vegas2
   nd = nd_max;
   ng = 1;
   
   npg = 0;
   std::cout<<"mds = "<<mds<<std::endl;
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
   std::cout<<"ng = "<<ng<<std::endl;
   cutilSafeCall(cudaMemcpyToSymbol(g_ndim, &ndim, sizeof(int)));
   cutilSafeCall(cudaMemcpyToSymbol(g_ng,   &ng,   sizeof(int)));
   cutilSafeCall(cudaMemcpyToSymbol(g_nd,   &nd,   sizeof(int)));
   cudaThreadSynchronize(); // wait for synchronize

   nCubes = (unsigned)(pow(ng,ndim));
   cutilSafeCall(cudaMemcpyToSymbol(g_nCubes, &nCubes, sizeof(nCubes)));
   cudaThreadSynchronize(); // wait for synchronize

   npg = ncall/nCubes;
   if (npg<2) npg = 2;
   calls = (double)(npg*nCubes);

   unsigned nCubeNpg = nCubes*npg;
   
   //   std::cout<<"nCubes= "<<nCubes<<std::endl;
   //   std::cout<<"nCubeNpg= "<<nCubeNpg<<std::endl;

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

   cutilSafeCall(cudaMemcpyToSymbol(g_npg,  &npg,  sizeof(int)));
   cutilSafeCall(cudaMemcpyToSymbol(g_xjac, &xjac, sizeof(float)));
   cutilSafeCall(cudaMemcpyToSymbol(g_dxg,  &dxg,  sizeof(float)));
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

   cutilSafeCall(cudaMemcpyToSymbol(g_xl, xl, sizeof(xl)));
   cutilSafeCall(cudaMemcpyToSymbol(g_dx, dx, sizeof(dx)));
   cutilSafeCall(cudaMemcpyToSymbol(g_xi, xi, sizeof(xi)));
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
   //   int iflag;
   // main integration loop

   //   std::cout<<"nBlockSize = "<<nBlockSize<<std::endl;
   //--------------------------
   //  Set up kernel vaiables
   //--------------------------
   const int nGridSizeMax =  65535;
   
   dim3 ThBk(nBlockSize);

   int nGridSizeX, nGridSizeY;
   int nBlockTot = (nCubeNpg-1)/nBlockSize+1;
//   std::cout<<"nBlockTot = "<<nBlockTot<<std::endl;
   nGridSizeY = (nBlockTot-1)/nGridSizeMax+1;
   nGridSizeX = (nBlockTot-1)/nGridSizeY+1;
//   std::cout<<"nGridSize (x,y) = "<<nGridSizeX<<", "<<nGridSizeY<<std::endl;
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
   int sizeFval = nCubeNpg*sizeof(float);
//   std::cout<<"sizeFval = "<<sizeFval<<std::endl;

   // CPU
   float* hFval;
   cutilSafeCall(cudaMallocHost((void**)&hFval, sizeFval));
   memset(hFval, '\0', sizeFval);

   // GPU
   float* gFval;
   cutilSafeCall(cudaMalloc((void**)&gFval, sizeFval));

   // allocate IAval
   //   int sizeIAval = nCubeNpg*ndim*sizeof(unsigned short);
   int sizeIAval = nCubeNpg*ndim*sizeof(int);
//   std::cout<<"sizeIAval = "<<sizeIAval<<std::endl;

   // CPU
   //unsigned short* hIAval;
   int* hIAval;
   cutilSafeCall(cudaMallocHost((void**)&hIAval, sizeIAval));
   //unsigned short* hIAval =
   //  (unsigned short*)calloc(nCubeNpg*ndim, sizeof(unsigned short));
   memset(hIAval, '\0', sizeIAval);

   // GPU
   // unsigned short* gIAval;
   int* gIAval;
   cutilSafeCall(cudaMalloc((void**)&gIAval, sizeIAval));

   double startVegasCall, endVegasCall;
   double startVegasMove, endVegasMove;
   double startVegasFill, endVegasFill;
   double startVegasRefine, endVegasRefine;

   do {
      
      it++;

//      std::cout<<"call gVegasCallFunc: it = "<<it<<std::endl;
      startVegasCall = getrusage_sec();
      gVegasCallFunc<<<BkGd, ThBk>>>(gFval, gIAval);
      cudaThreadSynchronize(); // wait for synchronize
      endVegasCall = getrusage_sec();
      timeVegasCall += endVegasCall-startVegasCall;

      startVegasMove = getrusage_sec();
      cutilSafeCall(cudaMemcpy(hFval, gFval,  sizeFval,
                               cudaMemcpyDeviceToHost));

      cutilSafeCall(cudaMemcpy(hIAval, gIAval,  sizeIAval,
                               cudaMemcpyDeviceToHost));
      endVegasMove = getrusage_sec();
      timeVegasMove += endVegasMove-startVegasMove;

// *****************         

      startVegasFill = getrusage_sec();

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
            double f = (double)hFval[idx];
//            std::cout<<"idx,f = "<<idx<<", "<<std::scientific
//                     <<std::setw(10)<<std::setprecision(5)<<f<<std::endl;
            double f2 = f*f;
            fb += f;
            f2b += f2;
            /*
            for (int idim=0;idim<ndim;idim++) {
               int iaj = hIAval[idim*nCubeNpg+idx];
               d[idim][iaj] += f2;
            }
            */
         }
         f2b = sqrt(f2b*npg);
         f2b = (f2b-fb)*(f2b+fb);
         ti += fb;
         tsi += f2b;
         if (mds<0) {
            for (int idim=0;idim<ndim;idim++) {
               int idx = npg*ig;
               int iaj = hIAval[idim*nCubeNpg+idx];
               d[idim][iaj] += f2b;
            }
         }
      }

//      std::cout<<"mds = "<<mds<<std::endl;
      if (mds>0) {
         //         std::cout<<"ndim = "<<ndim<<std::endl;
         for (int idim=0;idim<ndim;idim++) {
            //            std::cout<<"idim = "<<idim<<std::endl;
            for (int idx=0;idx<nCubeNpg;idx++) {
               //               std::cout<<"idx = "<<idx<<std::endl;
               int iaj = hIAval[idim*nCubeNpg+idx];
               //               std::cout<<"iaj = "<<iaj<<std::endl;
               double f = (double)hFval[idx];
               //               std::cout<<"f = "<<f<<std::endl;
               double f2 = f*f;
               d[idim][iaj] += f2;
               //               std::cout<<"idim, iaj, idx, f = "<<idim<<", "<<iaj
               //                        <<", "<<idx<<", "<<f<<std::endl;
            }
         }
      }

      endVegasFill = getrusage_sec();
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
               /*
               for (int i=0;i<nd;i+=3) {
                  std::cout<<std::setw(6)<<std::setprecision(2)<<std::setfill(' ')
                           <<xi[j][i]<<" "<<di[j][i]<<" "<<d[j][i];
                  std::cout<<std::setw(6)<<std::setprecision(2)
                           <<xi[j][i+1]<<" "<<di[j][i+1]<<" "<<d[j][i+1];
                  std::cout<<std::setw(6)<<std::setprecision(2)
                           <<xi[j][i+2]<<" "<<di[j][i+2]<<" "<<d[j][i+2]
                           <<std::endl;
                           }
               */
            }
         }
      }

      // refine grid

      startVegasRefine = getrusage_sec();

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
            xi[j][i] = (float)xin[i];
         }
         xi[j][nd-1] = 1.f;

      }
      cutilSafeCall(cudaMemcpyToSymbol(g_xi, xi, sizeof(xi)));
      cudaThreadSynchronize(); // wait for synchronize

      endVegasRefine = getrusage_sec();
      timeVegasRefine += endVegasRefine-startVegasRefine;
      
//      std::cout<<"The end of main loop: it, sd/avgi = "<<it<<", "
//               <<sd/fabs(avgi)<<std::endl;
      
   } while (it<itmx && acc*fabs(avgi)<sd);


   cutilSafeCall(cudaFreeHost(hFval));
   cutilSafeCall(cudaFree(gFval));

   cutilSafeCall(cudaFreeHost(hIAval));
//   free(hIAval);
   cutilSafeCall(cudaFree(gIAval));

   //   std::cout<<"ng = "<<ng<<std::endl;
}
