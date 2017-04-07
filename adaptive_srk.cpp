#include "adaptive_srk.h"

#ifdef USING_AMD
// Use standard malloc, free functions
#define STD_FREE
// ACML C interface takes ints, chars as size, stride, matrix order parameters; use the underlying Fortran calls that take pointers, for consistency with MKL
#define dcopy dcopy_
#define daxpy daxpy_
#define dger dger_
#define dgemv dgemv_
#define dgemm dgemm_
#define dscal dscal_
#define dnrm2 dnrm2_
#define ddot ddot_
#endif

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
        )
{
    #ifdef STD_FREE
    omp_set_num_threads(omp_get_max_threads());
    #endif
    #ifdef PROFILE
    boost::timer::cpu_timer solver_timer;
    #endif // PROFILE
    // Loop variables
    int i,j,k,l;

    // Function call return
    int status = -1;

    // Array lengths, for BLAS calls
    int n = 1;
    // Array stride, for BLAS calls
    int stride = 1;
    // Coefficients for BLAS calls
    double a = 1.;
    double a2 = 1.;
    double b = 1.;
    // Array transpose states, for BLAS calls
    char transa = 'N';
    char transb = 'N';

    // Number of terms in expansion for approximating multiple Wiener integrals
    int p = 10;

    // Working variables
    int noise_flag = 0; // Nonzero if any entries of sigma are nonzero
    int temp_int;
    int m_m = (*m) * (*m);
    int m_2_2p = (*m) * (2 + 2*p);
    int ndims_numpaths = (*ndims) * (*numpaths);
    double temp_double;
#ifdef STD_FREE
    double * temp_array_p = (double *) memalign( alignment, p * sizeof(double));
    double * temp_array_m = (double *) memalign( alignment, *m * sizeof(double));
    double * temp_array_m_m = (double *) memalign( alignment, m_m * sizeof(double));
    double * temp_array_ndims_1 = (double *) memalign( alignment, *ndims * sizeof(double));
    double * temp_array_ndims_2 = (double *) memalign( alignment, *ndims * sizeof(double));
    double * temp_array_ndims_m = (double *) memalign( alignment, *ndims * *m * sizeof(double));
    double * temp_array_ndims_numpaths = (double *) memalign( alignment, ndims_numpaths * sizeof(double));

    double * x_trial = (double *) memalign( alignment, ndims_numpaths * sizeof(double));
    double * x_trialhat = (double *) memalign( alignment, ndims_numpaths * sizeof(double));
#else
    double * temp_array_p = (double *) mkl_malloc( p * sizeof(double), alignment);
    double * temp_array_m = (double *) mkl_malloc( *m * sizeof(double), alignment);
    double * temp_array_m_m = (double *) mkl_malloc( m_m * sizeof(double), alignment);
    double * temp_array_ndims_1 = (double *) mkl_malloc( *ndims * sizeof(double), alignment);
    double * temp_array_ndims_2 = (double *) mkl_malloc( *ndims * sizeof(double), alignment);
    double * temp_array_ndims_m = (double *) mkl_malloc( *ndims * *m * sizeof(double), alignment);
    double * temp_array_ndims_numpaths = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);

    double * x_trial = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);
    double * x_trialhat = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);
#endif
    
    // Current solver time
    double t=0.;

    // sqrt(2) so we only need to compute it once
    double sqrt2 = sqrt(2.);

    // Currrent solver stepsize
    double h=*dt_max;

#ifdef STD_FREE
    double * meanerr = (double *) memalign( alignment, *ndims * sizeof(double) );
    double * prev_meanerr = (double *) memalign( alignment, *ndims * sizeof(double) );
#else
    // Vector to hold step error estimate, for adaptive stepping
    double * meanerr = (double *) mkl_malloc( *ndims * sizeof(double), alignment);
    // Previous step mean error
    double * prev_meanerr = (double *) mkl_malloc( *ndims * sizeof(double), alignment);
#endif

    // Error estimate
    double err;
    
    // Allow for sigma to be a sqrt(covariance) matrix, a vector of standard deviaions, or a scalar
    double * sigma;
    // If sigma==0, don't bother computing noise
    #pragma loop_count avg=2500
    #pragma vector aligned
    for ( i=0; i<arg_sigma->size(); i++ )
    {
        if ( (*arg_sigma)[i] != 0. )
        {
            noise_flag = 1;
        }
    }
    // If there's noise and arg_sigma isn't ndims x m, reshape it appropriately as a diagonal matrix
    if ( noise_flag != 0 )
    {
#ifdef STD_FREE
        sigma = (double *) memalign( alignment, *ndims * *m * sizeof(double) );
#else
        sigma = (double *) mkl_malloc( *ndims * *m * sizeof(double), alignment);
#endif

        // Fill sigma

        // Fill with zeros
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=2500
        #pragma vector aligned
        std::fill(sigma, sigma + ( (*m) * (*ndims) ), 0.);

        // arg_sigma is a single number
        if ( arg_sigma->size() == 1 )
        {
            #pragma parallel
            #pragma ivdep
            #pragma loop_count avg=50
            #pragma vector aligned
            for ( i=0; i<fmin(*m,*ndims); i++ )
            {
                sigma[i + i*(*m)] = (*arg_sigma)[0];
            }
        }
        
        // arg_sigma is a vector of length min(m,ndims)
        else if ( arg_sigma->size() == fmin(*m,*ndims) )
        {
            // Fill the diagonal with arg_sigma
            if (*m < *ndims)
            {
                #pragma parallel
                #pragma ivdep
                #pragma loop_count avg=50
                #pragma vector aligned
                for ( i=0; i<*m; i++ )
                {
                    sigma[i + i*(*m)] = (*arg_sigma)[i];
                }
            }
            else // *m > *ndims
            {
                #pragma parallel
                #pragma ivdep
                #pragma loop_count avg=50
                #pragma vector aligned
                for ( i=0; i<*ndims; i++ )
                {
                    sigma[i + i*(*m)] = (*arg_sigma)[i];
                }
            }
        }

        // If arg_sigma is ndims x m, copy arg_sigma to sigma
        else if ( ( arg_sigma->size() == *ndims * *m ) )
        {
            n = *ndims * *m;
            dcopy(&n, &arg_sigma->front(), &stride, sigma, &stride);
        }
        // Otherwise, error
        else
        {
            #ifndef MATLAB_MEX_FILE
            std::cout << "Invalid standard deviation argument to solver(); sigma must be 1 x 1, min(m,ndims) x 1, 1 x min(m,ndims), or ndims x m." << std::endl;
            #else
            mexPrintf("Invalid standard deviation argument has size %i.\nndims=%i m=%i\n",arg_sigma->size(),*ndims,*m);
            mexEvalString("drawnow;");
            #endif
            return 2;
        }
    }


    //////////////////////////////
    // Runge-Kutta Parameters
    //////////////////////////////
    // 3-step method
    int s = 3;

    // Parameters due to Kuepper, Lehn, and Roessler (2007)
    // TODO: Allow for time-depenence in sdefun

    // Order (3,2) coefficients, stored column-major
#ifdef STD_FREE
    double * A_00_00 = (double *) memalign( alignment, 9 * sizeof(double) );
#else
    double * A_00_00 = (double *) mkl_malloc( 9 * sizeof(double), alignment);
#endif
    A_00_00[0 + 0*3] = 0.;
    A_00_00[0 + 1*3] = 0.;
    A_00_00[0 + 2*3] = 0.;

    A_00_00[1 + 0*3] = 1.;
    A_00_00[1 + 1*3] = 0.;
    A_00_00[1 + 2*3] = 0.;

    A_00_00[2 + 0*3] = 0.25;
    A_00_00[2 + 1*3] = 0.25;
    A_00_00[2 + 2*3] = 0.;

#ifdef STD_FREE
    double * A_kk_00 = (double *) memalign( alignment, 9 * sizeof(double) );
#else
    double * A_kk_00 = (double *) mkl_malloc( 9 * sizeof(double), alignment);
#endif
    A_kk_00[0 + 0*3] = 0.;
    A_kk_00[0 + 1*3] = 0.;
    A_kk_00[0 + 2*3] = 0.;

    A_kk_00[1 + 0*3] = 1.;
    A_kk_00[1 + 1*3] = 0.;
    A_kk_00[1 + 2*3] = 0.;

    A_kk_00[2 + 0*3] = 1.;
    A_kk_00[2 + 1*3] = 0.;
    A_kk_00[2 + 2*3] = 0.;

#ifdef STD_FREE
    double * A_kl_00 = (double *) memalign( alignment, 9 * sizeof(double) );
#else
    double * A_kl_00 = (double *) mkl_malloc( 9 * sizeof(double), alignment);
#endif
    A_kl_00[0 + 0*3] = 0.;
    A_kl_00[0 + 1*3] = 0.;
    A_kl_00[0 + 2*3] = 0.;

    A_kl_00[1 + 0*3] = 0.;
    A_kl_00[1 + 1*3] = 0.;
    A_kl_00[1 + 2*3] = 0.;

    A_kl_00[2 + 0*3] = 0.;
    A_kl_00[2 + 1*3] = 0.;
    A_kl_00[2 + 2*3] = 0.;

#ifdef STD_FREE
    double * B_k_00_kk = (double *) memalign( alignment, 9 * sizeof(double) );
#else
    double * B_k_00_kk = (double *) mkl_malloc( 9 * sizeof(double), alignment);
#endif
    B_k_00_kk[0 + 0*3] = 0.;
    B_k_00_kk[0 + 1*3] = 0.;
    B_k_00_kk[0 + 2*3] = 0.;

    B_k_00_kk[1 + 0*3] = 0.6 - (0.4*sqrt(6.));
    B_k_00_kk[1 + 1*3] = 0.;
    B_k_00_kk[1 + 2*3] = 0.;

    B_k_00_kk[2 + 0*3] = 0.6 + (0.1*sqrt(6.));
    B_k_00_kk[2 + 1*3] = 0.;
    B_k_00_kk[2 + 2*3] = 0.;

#ifdef STD_FREE
    double * B_0_kk_kk = (double *) memalign( alignment, 9 * sizeof(double) );
#else
    double * B_0_kk_kk = (double *) mkl_malloc( 9 * sizeof(double), alignment);
#endif
    B_0_kk_kk[0 + 0*3] = 0.;
    B_0_kk_kk[0 + 1*3] = 0.;
    B_0_kk_kk[0 + 2*3] = 0.;

    B_0_kk_kk[1 + 0*3] = 1.;
    B_0_kk_kk[1 + 1*3] = 0.;
    B_0_kk_kk[1 + 2*3] = 0.;

    B_0_kk_kk[2 + 0*3] = -1.;
    B_0_kk_kk[2 + 1*3] = 0.;
    B_0_kk_kk[2 + 2*3] = 0.;

#ifdef STD_FREE
    double * B_0_kl_kl = (double *) memalign( alignment, 9 * sizeof(double) );
#else
    double * B_0_kl_kl = (double *) mkl_malloc( 9 * sizeof(double), alignment);
#endif
    B_0_kl_kl[0 + 0*3] = 0.;
    B_0_kl_kl[0 + 1*3] = 0.;
    B_0_kl_kl[0 + 2*3] = 0.;

    B_0_kl_kl[1 + 0*3] = 1.;
    B_0_kl_kl[1 + 1*3] = 0.;
    B_0_kl_kl[1 + 2*3] = 0.;

    B_0_kl_kl[2 + 0*3] = -1.;
    B_0_kl_kl[2 + 1*3] = 0.;
    B_0_kl_kl[2 + 2*3] = 0.;

#ifdef STD_FREE
    double * alpha_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * alpha_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    alpha_T[0] = 1./6.;
    alpha_T[1] = 1./6.;
    alpha_T[2] = 2./3.;

#ifdef STD_FREE
    double * gamma_k_kk_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gamma_k_kk_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gamma_k_kk_T[0] = 0.5;
    gamma_k_kk_T[1] = 0.25;
    gamma_k_kk_T[2] = 0.25;

#ifdef STD_FREE
    double * gamma_k_kl_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gamma_k_kl_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gamma_k_kl_T[0] = -0.5;
    gamma_k_kl_T[1] = 0.25;
    gamma_k_kl_T[2] = 0.25;
    
#ifdef STD_FREE
    double * gamma_kk_kk_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gamma_kk_kk_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gamma_kk_kk_T[0] = 0.;
    gamma_kk_kk_T[1] = 0.5;
    gamma_kk_kk_T[2] = -0.5;

#ifdef STD_FREE
    double * gamma_kl_kl_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gamma_kl_kl_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gamma_kl_kl_T[0] = 0.;
    gamma_kl_kl_T[1] = 0.5;
    gamma_kl_kl_T[2] = -0.5;

    // Embedded order (2,1) coefficients
#ifdef STD_FREE
    double * alphahat_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * alphahat_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    alphahat_T[0] = 0.5;
    alphahat_T[1] = 0.5;
    alphahat_T[2] = 0.;

#ifdef STD_FREE
    double * gammahat_k_kk_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gammahat_k_kk_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gammahat_k_kk_T[0] = 1.;
    gammahat_k_kk_T[1] = 0.;
    gammahat_k_kk_T[2] = 0.;

#ifdef STD_FREE
    double * gammahat_k_kl_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gammahat_k_kl_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gammahat_k_kl_T[0] = 0.;
    gammahat_k_kl_T[1] = 0.;
    gammahat_k_kl_T[2] = 0.;

#ifdef STD_FREE
    double * gammahat_kk_kk_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gammahat_kk_kk_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gammahat_kk_kk_T[0] = 0.;
    gammahat_kk_kk_T[1] = 0.;
    gammahat_kk_kk_T[2] = 0.;

#ifdef STD_FREE
    double * gammahat_kl_kl_T = (double *) memalign( alignment, 3 * sizeof(double) );
#else
    double * gammahat_kl_kl_T = (double *) mkl_malloc( 3 * sizeof(double), alignment);
#endif
    gammahat_kl_kl_T[0] = 0.;
    gammahat_kl_kl_T[1] = 0.;
    gammahat_kl_kl_T[2] = 0.;

    // s-dimensional arrays of gamma_k, gamma_kl, gammahat_k matrices with
    // diagonal entries gamma_k_kk_T[i] and off-diagonal entries gamma_k_kl_T[i]
    // diagonal entries gamma_kk_kk_T[i] and off-diagonal entries gamma_kl_kl_T[i]
    // diagonal entries gammahat_k_kk_T[i] and off-diagonal entries gammahat_k_kl_T[i]
    // for later convenience
    // Not allocating a gammahat_kl since it would be full of zeros...
    // Also not allocating gammahat_k[2,3] since they would be full of zeros...
#ifdef STD_FREE
    double ** gamma_k = (double **) memalign( alignment, s * sizeof(double *) );
    double ** gamma_kl = (double **) memalign( alignment, s * sizeof(double *) );
    double ** gammahat_k = (double **) memalign( alignment, 1 * sizeof(double *) );
    gammahat_k[0] = (double *) memalign( alignment, m_m * sizeof(double) );
#else
    double ** gamma_k = (double **) mkl_malloc( s * sizeof(double *), alignment);
    double ** gamma_kl = (double **) mkl_malloc( s * sizeof(double *), alignment);
    double ** gammahat_k = (double **) mkl_malloc( 1 * sizeof(double *), alignment);
    gammahat_k[0] = (double *) mkl_malloc( m_m * sizeof(double), alignment);
#endif
    #pragma loop_count avg=3
    #pragma vector aligned
    for ( i=0; i<s; i++ )
    {
#ifdef STD_FREE
        gamma_k[i] = (double *) memalign( alignment, m_m * sizeof(double) );
        gamma_kl[i] = (double *) memalign( alignment, m_m * sizeof(double) );
#else
        gamma_k[i] = (double *) mkl_malloc( m_m * sizeof(double), alignment);
        gamma_kl[i] = (double *) mkl_malloc( m_m * sizeof(double), alignment);
#endif
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=2500
        #pragma vector aligned
        std::fill( gamma_k[i], gamma_k[i] + m_m, gamma_k_kl_T[i] );
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=2500
        #pragma vector aligned
        std::fill( gamma_kl[i], gamma_kl[i] + m_m, gamma_kl_kl_T[i] );
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=50
        #pragma vector aligned
#pragma omp parallel for shared(gamma_k,gamma_k_kk_T,gamma_kk_kk_T,m,i) private(j) schedule(static)
        for ( j=0; j<*m; j++ )
        {
            (gamma_k[i])[j + j*(*m)] = gamma_k_kk_T[i];
            (gamma_kl[i])[j + j*(*m)] = gamma_kk_kk_T[i];
        }
    }
    // Only need i=0 for gammahat_k
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=2500
    #pragma vector aligned
    std::fill( gammahat_k[0], gammahat_k[0] + m_m, gammahat_k_kl_T[0] );
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=50
    #pragma vector aligned
#pragma omp parallel for shared(gammahat_k,gammahat_k_kk_T,m,i) private(j) schedule(static)
    for ( j=0; j<*m; j++ )
    {
        (gammahat_k[0])[j + j*(*m)] = gammahat_k_kk_T[0];
    }

     
    // Allocate supporting value arrays
#ifdef STD_FREE
    double * aH = (double *) memalign( alignment, ndims_numpaths * sizeof(double) );
#else
    double * aH = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);
#endif

#ifdef STD_FREE
    double ** H_00 = (double **) memalign( alignment, s * sizeof(double *) );
#else
    double ** H_00 = (double **) mkl_malloc( s * sizeof(double *), alignment);
#endif
    #pragma parallel
    #pragma loop_count 3
    for ( i=0; i<s; i++ )
    {
#ifdef STD_FREE
        H_00[i] = (double *) memalign( alignment, ndims_numpaths * sizeof(double) );
#else
        H_00[i] = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);
#endif
    }
    
    // Arrays that will only be used if noise is nonzero
    double ** randvars;
    double ** Jkl;
    if ( *m!=0 )
    { 
#ifdef STD_FREE
        randvars = (double **) memalign( alignment, *numpaths * sizeof(double *) );
        Jkl = (double **) memalign( alignment, *numpaths * sizeof(double *) );
#else
        randvars = (double **) mkl_malloc( *numpaths * sizeof(double *), alignment);
        Jkl = (double **) mkl_malloc( *numpaths * sizeof(double *), alignment);
#endif
        
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=10000
        #pragma vector aligned
        for ( i=0; i<*numpaths; i++ )
        {
#ifdef STD_FREE
            randvars[i] = (double *) memalign( alignment, *m * (2+2*p) * sizeof(double) );
            Jkl[i] = (double *) memalign( alignment, m_m * sizeof(double) );
#else
            // numpaths arrays to contain random variables for computing multiple Wiener integrals
                // Arrays contains m rows, (1 + 1 + p + p) columns corresponding
                // to ( xi | mu | eta_1 ... eta_p | zeta_1 ... zeta_p )
            randvars[i] = (double *) mkl_malloc( *m * (2+2*p) * sizeof(double), alignment);
            // Array to contain Stratonovich multiple integral during the calculation
            Jkl[i] = (double *) mkl_malloc( m_m * sizeof(double), alignment);
#endif
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Initialize the random number generator
    ////////////////////////////////////////////////////////////////////////////
    #ifdef USING_AMD
    int generator_id = 3;
    int subid = 0;  // Not referenced
    int lseed = 1;
    int * seed = (int *) memalign( alignment, lseed * sizeof(int) );
    time_t tseed;
    time(&tseed);
    seed[0] = (int)tseed;
    int lstate = 633;
    int * prng_state = (int *) memalign( alignment, lstate * sizeof(int) );
    int info;
    drandinitialize(generator_id, subid, seed, &lseed, prng_state, &lstate, &info);
    if ( info != 0 )
    {
        return 4;
    }
    #else // Intel
    time_t seed;
    time(&seed);
    VSLStreamStatePtr stream;
    vslNewStream ( &stream, VSL_BRNG_MT19937, seed );
    #endif

    // Maximum stepsize increase and decrease per step, to avoid thrashing
    double facmax = 2.;
    double facmin = 0.5;

    // Factor by which to scale h at each step
    double fac = 0.9;

    ////////////////////////////////////////////////////////////////////////////
    // Set up x_history and reserve space guaranteed to be needed in x_history,
    // t_history, h_history, err_history
    ////////////////////////////////////////////////////////////////////////////
    
    // Minimum number of steps that will be needed
    temp_int = (int) ceil( *tf / *dt_max );

    // Reserve vectors
    x_history->reserve(temp_int);
    t_history->reserve(temp_int);
    h_history->reserve(temp_int);
    err_history->reserve(temp_int);

    // Fill x_history[1] with x0
#ifdef STD_FREE
    x_history->push_back((double *) memalign( alignment, ndims_numpaths * sizeof(double) ) );
#else
    x_history->push_back((double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment));
#endif
    #pragma parallel
    #pragma loop_count avg=10000
    #pragma vector aligned
#pragma omp parallel for shared(numpaths,ndims,x0,stride,x_history) private(i) schedule(guided)
    for ( i=0; i<*numpaths; i++ )
    {
        dcopy(ndims, x0, &stride, x_history->back() + i*(*ndims), &stride);
    }

    // Set t_history[0] to 0
    t_history->push_back(0.);

    ////////////////////////////////////////////////////////////////////////////
    // Integrate the equations
    ////////////////////////////////////////////////////////////////////////////

    #ifdef PROFILE
    boost::timer::cpu_timer solver_loop_timer;
    #endif // PROFILE
    #ifdef DEBUG
    std::cout << "Entering solver loop" << std::endl;
    #endif
    while ( t < *tf )
    {
        #ifdef MATLAB_MEX_FILE_DEBUG
        int anyisnan = 0;
        mexPrintf("t=%f\n",t);
        mexEvalString("drawnow;");
        #endif
        #ifdef MATLAB_MEX_FILE
        if ( utIsInterruptPending() )
        {
            return 1;
        }
        #endif
        // Compute a trial step
        
        // Square root of h, so we don't have to repeatedly calculate it
        double sqrth = sqrt(h);

        if ( *m!=0 )
        {
            #pragma vector aligned
            #pragma loop_count avg=10000
            // TODO: Parallize this...
            for ( i=0; i<*numpaths; i++ )
            {
                // Generate random variables in randvars
                    // Need to compute double Wiener integrals I_kl with k != l
                    // xi_i = \Delta W_i / sqrt(h), so it is a standard normal
                #ifdef USING_AMD
                drandgaussian ( m_2_2p, 0., 1., prng_state, randvars[i], &info );
                if ( info != 0 )
                {
                    return 5;
                }
                #else // Intel
                vdRngGaussian ( VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, m_2_2p, randvars[i], 0., 1. );
                #endif

                // Zero out the elements of Jkl
                #pragma parallel
                #pragma ivdep
                #pragma loop_count avg=2500
                #pragma vector aligned
                std::fill( Jkl[i], Jkl[i] + m_m, 0. );
            }

            // Compute Ikl m x m matrix of double Wiener Ito integrals // Approximate Stratonovich integral J_kl
                // J_kl = I_kl + h/2 = \int \int Dw_k dW_l
                // using equation 5.3.31 in Platen and Bruti-Liberati
                // 'Numerical solution of stochastic differential equations with jumps in finance'
            #pragma parallel
            #pragma ivdep
            #pragma loop_count avg=10
            #pragma vector aligned
            for ( i=0; i<p; i++ )
            {
                temp_array_p[i] = 1./(i+1.);
            }
            double test_temp = dnrm2(&p, temp_array_p, &stride);
            double rho_p = (1./12.) - (1./(2.*PI*PI))*ddot(&p, temp_array_p, &stride, temp_array_p, &stride);
            double sqrt_rho_p = sqrt(rho_p);

            // h/(2*pi)*\sum_{r=1}^p ( (1/r) ( zeta_k,r*(sqrt(2)*xi_l + eta_l,r) - zeta_l,r*(sqrt(2)*xi_k + eta_k,r) ) ) + h*(xi_k*xi_l/2 + sqrt(rho_p)*(mu_k,p*xi_l - mu_l,p*xi_k))
            int r;
            // temp_array_m_m doesn't change in the loop; just fill it once
            #pragma parallel
            #pragma ivdep
            #pragma loop_count avg=2500
            #pragma vector aligned
            std::fill( temp_array_m_m, temp_array_m_m + m_m, (h/2.) );

            #pragma loop_count avg=10000
            #pragma vector aligned
#pragma omp parallel for shared(numpaths,p,m,randvars,stride,sqrt2,Jkl,m_m,h,temp_array_m_m) private(i,r,a) schedule(guided)
            for ( i=0; i<*numpaths; i++ )
            {
                // Local array, to make the loop threadsafe
#ifdef STD_FREE
                double * local_temp_array_m = (double *) memalign( alignment, *m * sizeof(double) );
#else
                double * local_temp_array_m = (double *) mkl_malloc( *m * sizeof(double), alignment);
#endif
                #pragma loop_count 10
                #pragma vector aligned
                for ( r=0; r<p; r++ )
                {
                    // temp_vector2 = sqrt(2)*xi_l + eta_l,r
                    dcopy(m, randvars[i] + (*m)*(2+r), &stride, local_temp_array_m, &stride);
                    a = sqrt2;
                    daxpy(m, &a, randvars[i], &stride, local_temp_array_m, &stride);
                    
                    // TODO: try dger and daxpy instead of two dger calls
                    // Jkl[k,l] += (1/r) * zeta[r][k] * temp_vector2[l]
                    a = 1./(r+1.);
//                if ( i== 0) std::cout << "Jkl[0]->data()[0]: " << Jkl[0]->data()[0] << std::endl;
                    dger(m, m, &a, randvars[i] + (*m)*(2+p+r), &stride, local_temp_array_m, &stride, Jkl[i], m);
//                if ( i == 0) std::cout << "Jkl[0]->data()[0]: " << Jkl[0]->data()[0] << std::endl;
//                if ( i == 0) {
//                    std::cout << "randvars | temp_vector2: ";
//                    for ( k=0; k<*m; k++ ) std::cout << randvars[0]->data()[k + (*m)*(2+p+r)] << " " << temp_vector2.data()[k] << " ";
//                    std::cout << std::endl;
//                }

                    // Jkl[k,l] -= (1/r) * temp_vector2[k] * zeta[r][l]
                    a = -1./(r+1.);
                    dger(m, m, &a, local_temp_array_m, &stride, randvars[i] + (*m)*(2+p+r), &stride, Jkl[i], m);
                }
                // Free the local array, since we don't need it anymore
#ifdef STD_FREE
                free(local_temp_array_m);
#else // Intel
                mkl_free(local_temp_array_m);
#endif

                // Jkl = Jkl/(2*PI)
                a = 1./(2.*PI);
                dscal( &m_m, &a, Jkl[i], &stride );
//                mkl_dimatcopy('C', 'N', *m, *m, a, Jkl[i]->data(), *m, *m);

                // Jkl += xi_k * xi_l / 2
                a = 0.5;
                dger(m, m, &a, randvars[i], &stride, randvars[i], &stride, Jkl[i], m);

                // Jkl += sqrt(rho_p) * mu_k,p * xi_l
                a = sqrt_rho_p;
                dger(m, m, &a, randvars[i] + (*m)*(1), &stride, randvars[i], &stride, Jkl[i], m);

                // Jkl -= sqrt(rho_p) * xi_k * mu_l,p
                a = -sqrt_rho_p;
                dger(m, m, &a, randvars[i], &stride, randvars[i] + (*m)*(1), &stride, Jkl[i], m);
                
                // Jkl *= h
                a = h;
                dscal(&m_m, &a, Jkl[i], &stride);

                // Ikl = Jkl - (h/2)
                a = -1.;
                daxpy( &n, &a, temp_array_m_m, &stride, Jkl[i], &stride );
            }
        }

        // Compute supporting values
        #pragma loop_count avg=3
        #pragma vector aligned
        for ( i=0; i<s; i++ )
        {
            // H_00[i] = x_history[end]
            dcopy(&ndims_numpaths, x_history->back(), &stride, H_00[i], &stride);

            // TODO: H_kl and H_kk are needed only if sigma depends on x.

            #pragma vector aligned
            for ( j=0; j<i; j++ )
            {
                // If the coefficient isn't 0:
                if ( A_00_00[i + 3*j] != 0. )
                {
                    // Apply sdefun to H_00
                    status = sdefun->eval_dynamics(H_00[j], aH);

                    // H_00 += aH * h * A_00_00[i][j]
                    a = A_00_00[i + 3*j] * h;
                    daxpy(&ndims_numpaths, &a, aH, &stride, H_00[i], &stride);
                }

                // If there's noise:
                if ( noise_flag != 0. )
                {
                    // If the coefficient isn't 0:
                    if ( B_k_00_kk[i + 3*j] != 0. )
                    {
                        // H_00 += (B_k_00_kk[i][j]*sqrth) * ( sigma * randvars[i]->data()[:,1])
                        a = B_k_00_kk[i + 3*j] * sqrth;
                        b = 1.;
                        transa = 'N';
                        transb = 'N';
                        #pragma loop_count avg=10000
                        #pragma vector aligned
#pragma omp parallel for shared(numpaths,transa,ndims,m,a,sigma,randvars,stride,b,H_00,i) private(k) schedule(guided)
                        for ( k=0; k<*numpaths; k++ )
                        {
                            #ifdef USING_AMD
                            dgemv(&transa, ndims, m, &a, sigma, ndims, randvars[k], &stride, &b, H_00[i] + k*(*ndims), &stride, 1);
                            #else // Intel
                            dgemv(&transa, ndims, m, &a, sigma, ndims, randvars[k], &stride, &b, H_00[i] + k*(*ndims), &stride);
                            #endif
                        }
                    }
                }
            }
        }

        // Compute a trial step
        dcopy( &ndims_numpaths, x_history->back(), &stride, x_trial, &stride );

        // Lower order step for error estimation
        // Consists of adding the single Wiener integral part for i=1; subsequent terms are all zero
        dcopy ( &ndims_numpaths, x_trial, &stride, x_trialhat, &stride );
        #ifdef MATLAB_MEX_FILE_DEBUG
        anyisnan = 0;
        for ( i=0; i<ndims_numpaths; i++ )
        {
            if ( isnan( x_trialhat[i] ) ) anyisnan += 1;
        }
        mexPrintf("1 any isnan x_trialhat: %i\n", anyisnan);
        mexEvalString("drawnow;");
        #endif

        // Only bother if there's noise
        // Second term of xhat_n+1, single Wiener integral part
        // x_trialhat += sum_{i=1:s} gamma_k_kl[i] * sigma*Ikl  with only s=1 term nontrivial
        if ( noise_flag != 0 )
        {
            transa = 'N';
            transb = 'N';
            #pragma loop_count avg=10000
            #pragma vector aligned
#pragma omp parallel for shared(m_m,ndims,m,gammahat_k,randvars,sqrth,transa,transb,sigma,stride,x_trialhat) private(j,k,l,a,b) schedule(guided)
            for ( j=0; j<*numpaths; j++ )
            {
                // Local arrays, to make the loop threadsafe
#ifdef STD_FREE
                double * local_temp_array_m_m = (double *) memalign( alignment, m_m * sizeof(double) );
                double * local_temp_array_ndims_m = (double *) memalign( alignment, *ndims * *m * sizeof(double) ); 
#else
                double * local_temp_array_m_m = (double *) mkl_malloc( m_m * sizeof(double), alignment );
                double * local_temp_array_ndims_m = (double *) mkl_malloc ( *ndims * *m * sizeof(double), alignment ); 
#endif
                #pragma loop_count avg=50
                #pragma vector aligned
                for ( k=0; k<*m; k++ )
                {
                    #ifdef USING_AMD
                    for ( l=0; l<*m; l++ )
                    {
                        local_temp_array_m_m[ l + k*(*m) ] = gammahat_k[0][ l + k*(*m) ] * randvars[j][l];
                    }
                    #else // Intel
                    vdMul(*m, gammahat_k[0] + k*(*m), randvars[j], local_temp_array_m_m + k*(*m));
                    #endif
                }
                a = sqrth;
                b = 0.;
                #ifdef USING_AMD
                dgemm( &transa, &transb, ndims, m, m, &a, sigma, ndims, local_temp_array_m_m, m, &b, local_temp_array_ndims_m, ndims, 1, 1);
                #else // Intel
                dgemm( &transa, &transb, ndims, m, m, &a, sigma, ndims, local_temp_array_m_m, m, &b, local_temp_array_ndims_m, ndims);
                #endif
                a = 1.;
                #pragma loop_count avg=50
                #pragma vector aligned
                for ( k=0; k<*m; k++ )
                {
                    daxpy(ndims, &a, local_temp_array_ndims_m + k*(*ndims), &stride, x_trialhat + j*(*ndims), &stride);
                }
    //            MKL_Domatadd( 'C', 'N', 'N', *ndims, 1, a, x_trial.data() + (*ndims)*i, *ndims, b, randvars[i]->data(), *ndims, x_trialhat.data() + (*ndims)*i, *ndims);
                // Free local arrays
#ifdef STD_FREE
                free(local_temp_array_m_m);
                free(local_temp_array_ndims_m);
#else // Intel
                mkl_free(local_temp_array_m_m);
                mkl_free(local_temp_array_ndims_m);
#endif
            }
            #ifdef MATLAB_MEX_FILE_DEBUG
            anyisnan = 0;
            for ( i=0; i<ndims_numpaths; i++ )
            {
                if ( isnan( x_trialhat[i] ) ) anyisnan += 1;
            }
            mexPrintf("2 any isnan x_trialhat: %i\n", anyisnan);
            mexEvalString("drawnow;");
            #endif
        }

        // Higher order step for the solution, and finish the lower order step
        #pragma loop_count 3
        #pragma vector aligned
        for ( i=0; i<s; i++ )
        {
            status = sdefun->eval_dynamics(H_00[i], temp_array_ndims_numpaths);
            if ( status != 0 )
            {
                return 3;
            }
            a = alpha_T[i] * h;
            daxpy(&ndims_numpaths, &a, temp_array_ndims_numpaths, &stride, x_trial, &stride);
            if ( alphahat_T[i] != 0. )
            {
                a = alphahat_T[i] * h;
                daxpy(&ndims_numpaths, &a, temp_array_ndims_numpaths, &stride, x_trialhat, &stride);
            }
            #ifdef MATLAB_MEX_FILE_DEBUG
            anyisnan = 0;
            for ( j=0; j<ndims_numpaths; j++ )
            {
                if ( isnan( x_trialhat[j] ) ) anyisnan += 1;
            }
            mexPrintf("3 any isnan x_trialhat: %i\n", anyisnan);
            mexEvalString("drawnow;");
            #endif



            // Only bother with this if there's noise
            if ( noise_flag != 0 )
            {
                transa = 'N';
                transb = 'N';
                #pragma loop_count avg=10000
                #pragma vector aligned
#pragma omp parallel for shared(numpaths,m_m,ndims,m,gamma_k,i,randvars,sqrth,transa,transb,sigma,stride,x_trial,Jkl,gamma_kl) private(j,k,l,a,b) schedule(guided)
                for ( j=0; j<*numpaths; j++ )
                {
#ifdef STD_FREE
                    double * local_temp_array_m_m = (double *) memalign( alignment, m_m * sizeof(double) );
                    double * local_temp_array_ndims_m = (double *) memalign( alignment, *ndims * *m * sizeof(double) );
#else // Intel
                    double * local_temp_array_m_m = (double *) mkl_malloc( m_m * sizeof(double), alignment );
                    double * local_temp_array_ndims_m = (double *) mkl_malloc( *ndims * *m * sizeof(double), alignment );
#endif
                    // Second term of x_n+1; single Wiener integral part
                    // sum_{k=1:m,l=1:m} gamma_k_kl * sigma(:,k) * I_k
                    #pragma loop_count avg=50
                    #pragma vector aligned
                    for ( k=0; k<*m; k++ )
                    {
                        #ifdef USING_AMD
                        for ( l=0; l<*m; l++ )
                        {
                            local_temp_array_m_m[l + k*(*m)] = gamma_k[i][l + k*(*m)] * randvars[j][l];
                        }
                        #else // Intel
                        vdMul(*m, gamma_k[i] + k*(*m), randvars[j], local_temp_array_m_m + k*(*m));
                        #endif
                    }
                    a = sqrth;
                    b = 0.;
                    #ifdef USING_AMD
                    dgemm(&transa, &transb, ndims, m, m, &a, sigma, ndims, local_temp_array_m_m, m, &b, local_temp_array_ndims_m, ndims, 1, 1);
                    #else // Intel
                    dgemm(&transa, &transb, ndims, m, m, &a, sigma, ndims, local_temp_array_m_m, m, &b, local_temp_array_ndims_m, ndims);
                    #endif
                    a = 1.;
                    #pragma loop_count avg=50
                    #pragma vector aligned
                    for ( k=0; k<*m; k++ )
                    {
                        daxpy(ndims, &a, local_temp_array_ndims_m + k*(*ndims), &stride, x_trial + j*(*ndims), &stride);
                    }
//                    // Debugging output
//                    for ( k=0; k<*ndims; k++ )
//                    {
//                        std::cout << x_trial.data()[k] << " ";
//                    }
//                    std::cout << std::endl;
//                    std::cout << Jkl[j]->data()[0] << std::endl;
//                    std::cout << gamma_kl[i]->data()[0] << std::endl;
//                    std::cout << sigma->data()[0] << std::endl;

                    // Third term of x_n+1; double Wiener integral part
                    // sum_{k=1:m, l=1:m} gamm_kl_kl * sigma(:,k) * Ikl
                    #ifdef USING_AMD
                    for ( l=0; l<m_m; l++ )
                    {
                        local_temp_array_m_m[l] = Jkl[j][l] * gamma_kl[i][l];
                    }
                    #else // Intel
                    vdMul( m_m, Jkl[j], gamma_kl[i], local_temp_array_m_m );
                    #endif
                    a = 1./sqrth;
                    b = 0.;
                    #ifdef USING_AMD
                    dgemm(&transa, &transb, ndims, m, m, &a, sigma, ndims, local_temp_array_m_m, m, &b, local_temp_array_ndims_m, ndims, 1, 1);
                    #else // Intel
                    dgemm(&transa, &transb, ndims, m, m, &a, sigma, ndims, local_temp_array_m_m, m, &b, local_temp_array_ndims_m, ndims);
                    #endif
                    a = 1.;
                    #pragma loop_count avg=50
                    #pragma vector aligned
                    for ( k=0; k<*m; k++ )
                    {
                        daxpy(ndims, &a, local_temp_array_ndims_m + k*(*ndims), &stride, x_trial + j*(*ndims), &stride);
                    }
//                    // Debugging output
//                    for ( k=0; k<*ndims; k++ )
//                    {
//                        std::cout << x_trial.data()[k] << " ";
//                    }
//                    std::cout << std::endl;
                    // Free local arrays
#ifdef STD_FREE
                    free(local_temp_array_m_m);
                    free(local_temp_array_ndims_m);
#else // Intel
                    mkl_free(local_temp_array_m_m);
                    mkl_free(local_temp_array_ndims_m);
#endif
                }
            }
        }

        #ifndef FIXED_STEP
        // Estimate error
        
        // Clear temp_vector
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=50
        #pragma vector aligned
        std::fill( temp_array_ndims_1, temp_array_ndims_1 + *ndims, 0. );

        #ifdef MATLAB_MEX_FILE_DEBUG
        anyisnan = 0;
        for ( i=0; i<ndims_numpaths; i++ )
        {
            if ( isnan( x_trial[i] ) ) anyisnan += 1;
        }
        mexPrintf("any isnan x_trial: %i\n", anyisnan);
        anyisnan = 0;
        for ( i=0; i<ndims_numpaths; i++ )
        {
            if ( isnan( x_trialhat[i] ) ) anyisnan += 1;
        }
        mexPrintf("any isnan x_trialhat: %i\n", anyisnan);
        mexEvalString("drawnow;");

        anyisnan = 0;
        for ( i=0; i<*ndims; i++ )
        {
            if ( isnan( temp_array_ndims_1[i] ) ) anyisnan += 1;
        }
        mexPrintf("any isnan temp_vector: %i\n", anyisnan);
        #endif
 
        // Compute error estimate
        a = 1./(*numpaths);
        a2 = -1./(*numpaths);
        #pragma loop_count avg=10000
        #pragma vector aligned
#pragma omp parallel for shared(numpaths,ndims,a,x_trial,stride,temp_array_ndims_1,a2,x_trialhat) private(i) schedule(guided)
        for ( i=0; i<*numpaths; i++ )
        {
            daxpy(ndims, &a, x_trial + i*(*ndims), &stride, temp_array_ndims_1, &stride);
            daxpy(ndims, &a2, x_trialhat + i*(*ndims), &stride, temp_array_ndims_1, &stride);
        }

        #ifdef MATLAB_MEX_FILE_DEBUG
        anyisnan = 0;
        for ( i=0; i<*ndims; i++ )
        {
            if ( isnan( temp_array_ndims_1[i] ) ) anyisnan += 1;
        }
        mexPrintf("any isnan temp_vector: %i\n", anyisnan);
        #endif
 
        #ifdef USING_AMD
#pragma omp parallel for shared(ndims,meanerr,temp_array_ndims_1) private(i) schedule(guided)
        for ( i=0; i<*ndims; i++ )
        {
            meanerr[i] = std::abs(temp_array_ndims_1[i]);
        }
        #else // Intel
        vdAbs(*ndims, temp_array_ndims_1, meanerr);
        #endif

        #ifdef MATLAB_MEX_FILE_DEBUG
        anyisnan = 0;
        for ( i=0; i<*ndims; i++ )
        {
            if ( isnan( temp_array_ndims_1[i] ) ) anyisnan += 1;
        }
        mexPrintf("any isnan temp_vector: %i\n", anyisnan);
        #endif
 
        // tolerance = max(prev_meanerr,meanerr)*reltol + abstol
        // temp_vector[i] = meanerr[i] / sqrt(tolerance[i])
        #pragma parallel
        #pragma ivdep
        #pragma loop_count avg=50
        #pragma vector aligned
#pragma omp parallel for shared(ndims,temp_array_ndims_1,meanerr,reltol,prev_meanerr,abstol) private(i) schedule(guided)
        for ( i=0; i<*ndims; i++ )
        {
            temp_array_ndims_1[i] = meanerr[i] / sqrt( (*reltol)*fmax(prev_meanerr[i], meanerr[i]) + (*abstol) );
        }

        #ifdef MATLAB_MEX_FILE_DEBUG
        anyisnan = 0;
        for ( i=0; i<*ndims; i++ )
        {
            if ( isnan( temp_array_ndims_1[i] ) ) anyisnan += 1;
        }
        mexPrintf("any isnan temp_vector: %i\n", anyisnan);
        #endif
 
        err = dnrm2( ndims, temp_array_ndims_1, &stride );

        #ifdef MATLAB_MEX_FILE_DEBUG
        mexPrintf("err = %f\n", err);
        mexEvalString("drawnow;");
        #endif

        #else // FIXED_STEP is true
        err = 0.5;
        #endif // FIXED_STEP

        err_history->push_back(err);
        h_history->push_back(h);

        #ifdef MATLAB_MEX_FILE_DEBUG
        mexPrintf("h = %f\n", h);
        mexPrintf("err = %f\n", err);
        mexEvalString("drawnow;");
        #endif
        if ( err <= 1. )    // Accept step
        {
            dcopy(ndims, meanerr, &stride, prev_meanerr, &stride);
#ifdef STD_FREE
            x_history->push_back( (double *) memalign( alignment, ndims_numpaths * sizeof(double) ) );
#else
            x_history->push_back( (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment) );
#endif
            dcopy(&ndims_numpaths, x_trial, &stride, x_history->back(), &stride);
            t += h;
            t_history->push_back(t);
            #ifdef DEBUG
            std::cout << "Accepted step" << std::endl;
            #endif
        }

        #ifndef FIXED_STEP
        h = h * fmin( facmax, fmax( facmin, fac*sqrt(1./err) ) );

        // If h>dt_max, default to dt_max
        if ( h>*dt_max )
        {
            h = *dt_max;
        }
        #endif // FIXED_STEP

        // If we'll pass tf, just step up to tf
        if ( (t+h) > *tf )
        {
            h = *tf - t;
        }
 

        #ifdef DEBUG
        std::cout << "t: " << t << std::endl;
        std::cout << "error: " << err << std::endl;
        std::cout << "h: " << h << std::endl;
        #endif

    }

    #ifdef PROFILE
    solver_timer.stop();
    solver_loop_timer.stop();

    std::cout << "Total function time:" << solver_timer.format() << std::endl;
    std::cout << "Loop time:" << solver_loop_timer.format() << std::endl;
    #endif // PROFILE

    // Free allocated arrays
    #ifdef USING_AMD
    free(seed);
    free(prng_state);
    #else // Intel
    mkl_free_buffers();
    #endif

#ifdef STD_FREE
    free(temp_array_p);
    free(temp_array_m);
    free(temp_array_m_m);
    free(temp_array_ndims_1);
    free(temp_array_ndims_2);
    free(temp_array_ndims_m);
    free(temp_array_ndims_numpaths);
    free(x_trial);
    free(x_trialhat);
    free(meanerr);
    free(prev_meanerr);
    if ( noise_flag != 0 )
    {
        free(sigma);
    }
    free(A_00_00);
    free(A_kk_00);
    free(A_kl_00);
    free(B_k_00_kk);
    free(B_0_kk_kk);
    free(B_0_kl_kl);
    free(alpha_T);
    free(gamma_k_kk_T);
    free(gamma_k_kl_T);
    free(gamma_kk_kk_T);
    free(gamma_kl_kl_T);
    free(alphahat_T);
    free(gammahat_k_kk_T);
    free(gammahat_k_kl_T);
    free(gammahat_kk_kk_T);
    free(gammahat_kl_kl_T);
    for ( i=0; i<s; i++ )
    {
        free(gamma_k[i]);
        free(gamma_kl[i]);
        free(H_00[i]);
    }
    free(gammahat_k[0]);
    free(gammahat_k);
    free(gamma_k);
    free(gamma_kl);
    free(H_00);
    free(aH);
    for ( i=0; i<*numpaths; i++ )
    {
        free(randvars[i]);
        free(Jkl[i]);
    }
    free(randvars);
    free(Jkl);
#else
    mkl_free(temp_array_p);
    mkl_free(temp_array_m);
    mkl_free(temp_array_m_m);
    mkl_free(temp_array_ndims_1);
    mkl_free(temp_array_ndims_2);
    mkl_free(temp_array_ndims_m);
    mkl_free(temp_array_ndims_numpaths);
    mkl_free(x_trial);
    mkl_free(x_trialhat);
    mkl_free(meanerr);
    mkl_free(prev_meanerr);
    if ( noise_flag != 0 )
    {
        mkl_free(sigma);
    }
    mkl_free(A_00_00);
    mkl_free(A_kk_00);
    mkl_free(A_kl_00);
    mkl_free(B_k_00_kk);
    mkl_free(B_0_kk_kk);
    mkl_free(B_0_kl_kl);
    mkl_free(alpha_T);
    mkl_free(gamma_k_kk_T);
    mkl_free(gamma_k_kl_T);
    mkl_free(gamma_kk_kk_T);
    mkl_free(gamma_kl_kl_T);
    mkl_free(alphahat_T);
    mkl_free(gammahat_k_kk_T);
    mkl_free(gammahat_k_kl_T);
    mkl_free(gammahat_kk_kk_T);
    mkl_free(gammahat_kl_kl_T);
    for ( i=0; i<s; i++ )
    {
        mkl_free(gamma_k[i]);
        mkl_free(gamma_kl[i]);
        mkl_free(H_00[i]);
    }
    mkl_free(gammahat_k[0]);
    mkl_free(gammahat_k);
    mkl_free(gamma_k);
    mkl_free(gamma_kl);
    mkl_free(H_00);
    mkl_free(aH);
    for ( i=0; i<*numpaths; i++ )
    {
        mkl_free(randvars[i]);
        mkl_free(Jkl[i]);
    }
    mkl_free(randvars);
    mkl_free(Jkl);
#endif
    // Don't free x_history[0:end], since we return that...

    return 0;
}
