#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "helper_cuda.h"

// This file contains the interface to the Monte Carlo program.


void main()
{
    /*
    The following parameters are set by the user:
        - tpb : CUDA Threads per Block.
        - maxit : Maximum number of times that the integration will be computed.
        - cpi : Number of function calls per iteration.
        - ndim : Number of dimensions in your integration.
    */ 

    int tpb = 512;
    int maxit = 20;
    int ndim = 5;
    int cpi = 1024 * 32

    assert(ndim <= ndim_max);

    // The limits of integration are set here. xl is lower bound. xu is upper bound.
    for (int i=0;i<ndim;i++) { 
        xl[i] = 1.; 
        xu[i] = 10.; 
    }

    // The result and standard deviation are instantiated.
    double avgi = 0.;
    double sd = 0.;
 
    // Integration function is called.
    time_t tic;

    MonteCarlo(avgi, sd);
 
    time_t toc;

    double seconds = difftime(toc, tic);

    //-------------------------
    //  Print out information
    //-------------------------
    std::cout.clear();
    std::cout<<"#==========================="<<std::endl;
    std::cout<<"# No. of Threads per Block : "<<tpd<<std::endl;
    std::cout<<"#==========================="<<std::endl;
    std::cout<<"# No. of dimensions        : "<<ndim<<std::endl;
    std::cout<<"# No. of func calls / iter : "<<cpi<<std::endl;
    std::cout<<"# No. of max. iterations   : "<<maxit<<std::endl;
    std::cout<<"#==========================="<<std::endl;
    std::cout<<"# Answer                   : "<<avgi<<" +- "<<sd<<std::endl;
    //std::cout<<"# Chisquare                : "<<chi2a<<std::endl;
    std::cout<<"#==========================="<<std::endl;
 
    //Print running times!
    std::cout<<"#==========================="<<std::endl;
    //printf("# Function call time per iteration: %lf\n", timeVegasCallAndFill/(double)it);
    //printf("# Refining time per iteration: %lf\n", timeVegasRefine/(double)it);
    std::cout<<"Time to integrate          :"<<seconds<<std::endl;
    std::cout<<"#==========================="<<std::endl;

}