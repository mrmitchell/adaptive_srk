#ifndef _ADAPTIVE_SRK_H_
#define _ADAPTIVE_SRK_H_

// Adaptive Runge-Kutta stochastic differential equation solver

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <malloc.h>
#include <memory>
#include <vector>

#ifdef USING_AMD
#include <acml.h>
#include <omp.h>
#else // Intel
#include "mkl.h"
#include "mkl_vsl.h"
#endif

#include "SdeFun.h"

#ifdef PROFILE
#include <boost/timer/timer.hpp>
#endif // PROFILE

#ifdef MATLAB_MEX_FILE
#include "mex.h"

extern "C" bool utIsInterruptPending();
#endif


#define PI 3.141592653589793
#define alignment 32

int solver(
        ////////////////////
        // Outputs
        ////////////////////
        // Vector of variable values at each timepoint
        std::vector<double *> * x_history,
        // Vector containing each timepoint
        std::vector<double> * t_history,
        // Vector containing each stepsize
        std::vector<double> * h_history,
        // Vector containing each error estimate
        std::vector<double> * err_history,
        ////////////////////
        // Inputs
        ////////////////////
        // Dimensionality of the system to be solved
        int * ndims,
        // Initial variable values
        double * x0,
        // Pointer to SdeFun class instance containing function to evaluate deterministic part of the dynamics
        SdeFun * sdefun,
        // Maximum timestep size
        double * dt_max,
        // Solver absolute error tolerance
        double * abstol,
        // Solver relative error tolerance
        double * reltol,
        // Final time
        double * tf,
        // Dimensionality of the Wiener process
        int * m,
        // Standard deviation of additive noise
        std::vector<double> * arg_sigma,
        // Number of Brownian paths to realize in parallel
        int * numpaths
        );

#endif // _ADAPTIVE_SKR_H_
