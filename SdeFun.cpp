#include "SdeFun.h"

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

// SDE deterministic dynamics class

SdeFun::SdeFun(int modeval)
{
    ndims = -1;
    numpaths = -1;
    mode = modeval;
    a1 = NAN;
    a2 = NAN;
    b11 = NAN;
    b12 = NAN;
    b21 = NAN;
    b22 = NAN;
    beta = NAN;
    gamma = NAN;
    J = NULL;
    h = NULL;
    dynamics = NULL;

    stride = 1;
    transT = 'T';
    transN = 'N';
}

// dynamic 10
// dx/dt = 0 (+eta)
// Noise-only dynamics (Brownian motion)
// params: 
int SdeFun::brownian(double * x, double * dx)
{
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=500000
    #pragma vector aligned
    std::fill(dx, dx + ndims*numpaths, 0.);
    return 0;
}

// dynamic 11
// dx/dt = gamma (+eta)
// Brownian motion with drift
// params: gamma
int SdeFun::brownian_drift(double * x, double * dx)
{
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=500000
    #pragma vector aligned
    std::fill(dx, dx + ndims*numpaths, gamma);
    return 0;
}

// dynamic 20
// dx/dt = gamma*x
// params: gamma
int SdeFun::linear(double * x, double * dx)
{
    std::fill(dx, dx + ndims*numpaths, 0.);
    a = gamma;
    stride = 1;
    size = ndims*numpaths;
    daxpy(&size, &a, x, &stride, dx, &stride);
    return 0;
}

// dynamic 30
// 2-d sine
// dx/dt = y
// dy/dt = -x
// params:
int SdeFun::twoDsine(double * x, double * dx)
{
    std::fill(dx, dx + ndims*numpaths, 0.);
    stride = 2;
    size = numpaths;
    a = 1.;
    daxpy(&size, &a, x + 1, &stride, dx, &stride);
    a = -1.;
    daxpy(&size, &a, x, &stride, dx + 1, &stride);
    return 0;
}

// dynamic 40
// 2d system
// x1' = b11*a1 + b12*a2 - b11*x1 - b12*x2
// x2' = b21*a1 + b22*a2 - b21*x1 - b22*x2
// params: b11 b12 b21 b22
int SdeFun::twoDsystem(double * x, double * dx)
{
    #ifdef STD_FREE
    double * a1vec = (double *) memalign( alignment, numpaths * sizeof(double) );
    double * a2vec = (double *) memalign( alignment, numpaths * sizeof(double) );
    #else // Use mkl functions
    double * a1vec = (double *) mkl_malloc( numpaths * sizeof(double), alignment );
    double * a2vec = (double *) mkl_malloc( numpaths * sizeof(double), alignment );
    #endif
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=10000
    #pragma vector aligned
    std::fill(a1vec, a1vec + numpaths, a1);
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=10000
    #pragma vector aligned
    std::fill(a2vec, a2vec + numpaths, a2);

    std::fill(dx, dx + ndims*numpaths, 0.);

    a = b11;
    
    #pragma parallel
    #pragma ivdep
    #pragma loop_count avg=50
    #pragma vector aligned
    daxpy(&size, &a, a1vec, &stride, dx, &stride);
    a = b21;
    daxpy(&size, &a, a1vec, &stride, dx+1, &stride);

    a = b12;
    daxpy(&size, &a, a2vec, &stride, dx, &stride);
    a = b22;
    daxpy(&size, &a, a2vec, &stride, dx+1, &stride);

    a = -b11;
    daxpy(&size, &a, x, &stride, dx, &stride);
    a = -b21;
    daxpy(&size, &a, x, &stride, dx+1, &stride);

    a = -b12;
    daxpy(&size, &a, x+1, &stride, dx, &stride);
    a = -b22;
    daxpy(&size, &a, x+1, &stride, dx+1, &stride);

    // Free allocated arrays
    
#ifdef STD_FREE
    free(a1vec);
    free(a2vec);
#else
    mkl_free(a1vec);
    mkl_free(a2vec);
#endif

    return 0;
}

// dynamic 50
// Kaneko dynamics
// params: J gamma beta
int SdeFun::kaneko(double * x, double * dx)
{
#ifdef STD_FREE
    omp_set_num_threads(omp_get_max_threads());
    double * temp = (double *) memalign( alignment, ndims*numpaths * sizeof(double));
#else
    double * temp = (double *) mkl_malloc( ndims*numpaths * sizeof(double), alignment);
#endif
    // temp = beta * J*x
    a = beta;
    b = 0.;
    #ifdef USING_AMD
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, temp, &ndims, 1, 1);
    #else // Intel
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, temp, &ndims);
    #endif

    // dx = tanh( temp )
    #ifdef USING_AMD
    int i = 0;
    double ndims_numpaths = ndims*numpaths;
//#pragma omp parallel for shared(dx,ndims_numpaths,temp) private(i) schedule(guided)
    for ( i=0; i<ndims_numpaths; i++ )
    {
        dx[i] = tanh(temp[i]);
    }
    #else // Intel
    vdTanh(ndims*numpaths, temp, dx);
    #endif

    // dx = dx - x
    a = -1.;
    size = ndims*numpaths;
    stride = 1;
    daxpy(&size, &a, x, &stride, dx, &stride);

    // dx = gamma*dx
    a = gamma;
    size = ndims*numpaths;
    stride = 1;
    dscal(&size, &a, dx, &stride);

    // Free allocated arrays
    #ifndef USING_AMD
    mkl_free_buffers();
    #endif
#ifdef STD_FREE
    free(temp);
#else
    mkl_free(temp);
#endif

    return 0;
}

// dynamic 51
// Kaneko dynamics with the tanh replaced by sine
// gamma * ( ( beta * J*x )^3 - x)
// params: J gamma beta
int SdeFun::kaneko_sine(double * x, double * dx)
{
#ifdef STD_FREE
    omp_set_num_threads(omp_get_max_threads());
    double * temp = (double *) memalign( alignment, ndims*numpaths * sizeof(double));
#else
    double * temp = (double *) mkl_malloc( ndims*numpaths * sizeof(double), alignment);
#endif
    // temp = beta * J*x
    a = beta;
    b = 0.;
    #ifdef USING_AMD
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, temp, &ndims, 1, 1);
    #else // Intel
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, temp, &ndims);
    #endif

    // dx = sine( temp )
    int i = 0;
    double ndims_numpaths = ndims*numpaths;
//#pragma omp parallel for shared(dx,ndims_numpaths,temp) private(i) schedule(guided)
    for ( i=0; i<ndims_numpaths; i++ )
    {
        dx[i] = sin(temp[i]);
    }

    // dx = dx - x
    a = -1.;
    size = ndims*numpaths;
    stride = 1;
    daxpy(&size, &a, x, &stride, dx, &stride);

    // dx = gamma*dx
    a = gamma;
    size = ndims*numpaths;
    stride = 1;
    dscal(&size, &a, dx, &stride);

    // Free allocated arrays
    #ifndef USING_AMD
    mkl_free_buffers();
    #endif
#ifdef STD_FREE
    free(temp);
#else
    mkl_free(temp);
#endif

    return 0;
}

// dynamic 52
// Kaneko dynamics with no tanh
// gamma * ( beta * J*x - x)
// params: J gamma beta
int SdeFun::kaneko_notanh(double * x, double * dx)
{
    // dx = beta * J*x
    a = beta;
    b = 0.;
    #ifdef USING_AMD
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, dx, &ndims, 1, 1);
    #else // Intel
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, dx, &ndims);
    #endif

    // dx = dx - x
    a = -1.;
    size = ndims*numpaths;
    stride = 1;
    daxpy(&size, &a, x, &stride, dx, &stride);

    // dx = gamma*dx
    a = gamma;
    size = ndims*numpaths;
    stride = 1;
    dscal(&size, &a, dx, &stride);

    // Free allocated arrays
    #ifndef USING_AMD
    mkl_free_buffers();
    #endif

    return 0;
}


// dynamic 53
// Kaneko dynamics with an environmental field
// params: J gamma beta h
int SdeFun::kaneko_env_field(double * x, double * dx)
{
#ifdef STD_FREE
    omp_set_num_threads(omp_get_max_threads());
    double * temp = (double *) memalign( alignment, ndims*numpaths * sizeof(double));
#else
    double * temp = (double *) mkl_malloc( ndims*numpaths * sizeof(double), alignment);
#endif
    // temp = beta * J*x
    a = beta;
    b = 0.;
    #ifdef USING_AMD
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, temp, &ndims, 1, 1);
    #else // Intel
    dgemm(&transN, &transN, &ndims, &numpaths, &ndims, &a, J, &ndims, x, &ndims, &b, temp, &ndims);
    #endif

    // dx = tanh( temp )
    #ifdef USING_AMD
    int i = 0;
    double ndims_numpaths = ndims*numpaths;
//#pragma omp parallel for shared(dx,ndims_numpaths,temp) private(i) schedule(guided)
    for ( i=0; i<ndims_numpaths; i++ )
    {
        dx[i] = tanh(temp[i]);
    }
    #else // Intel
    vdTanh(ndims*numpaths, temp, dx);
    #endif

    // dx = dx - x
    a = -1.;
    size = ndims*numpaths;
    stride = 1;
    daxpy(&size, &a, x, &stride, dx, &stride);

    // dx = dx + h
    a = 1.;
    size = ndims*numpaths;
    stride = 1;
    daxpy(&size, &a, h, &stride, dx, &stride);

    // dx = gamma*dx
    a = gamma;
    size = ndims*numpaths;
    stride = 1;
    dscal(&size, &a, dx, &stride);

    // Free allocated arrays
    #ifndef USING_AMD
    mkl_free_buffers();
    #endif
#ifdef STD_FREE
    free(temp);
#else
    mkl_free(temp);
#endif

    return 0;
}

// dynamic 60
// x is an n x n matrix, J is an n x n matrix
// gamma * ( tanh( beta * x^T * J * x ) - x )
int SdeFun::tanh_quadratic(double * x, double * dx)
{
    int sqrt_ndims = (int) sqrt(ndims);
    int ndims_numpaths = ndims*numpaths;
#ifdef STD_FREE
    omp_set_num_threads(omp_get_max_threads());
    double * temp1 = (double *) memalign( alignment, ndims_numpaths * sizeof(double));
    double * temp2 = (double *) memalign( alignment, ndims_numpaths * sizeof(double));
#else
    double * temp1 = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);
    double * temp2 = (double *) mkl_malloc( ndims_numpaths * sizeof(double), alignment);
#endif
    // temp1 = beta * J*x
    // temp2 = 
    a = beta;
    c = 1.;
    b = 0.;
    int i = 0;
#pragma omp parallel for shared(x,temp1,temp2) private(i) schedule(guided)
    for (i=0; i<numpaths; i++)
    {
        #ifdef USING_AMD
        dgemm(&transN, &transN, &sqrt_ndims, &sqrt_ndims, &sqrt_ndims, &a, J, &sqrt_ndims, x+i*ndims, &sqrt_ndims, &b, temp1+i*ndims, &sqrt_ndims, 1, 1);
        dgemm(&transT, &transN, &sqrt_ndims, &sqrt_ndims, &sqrt_ndims, &c, x+i*ndims, &sqrt_ndims, temp1+i*ndims, &sqrt_ndims, &b, temp2+i*ndims, &sqrt_ndims, 1, 1);
        #else // Intel
        dgemm(&transN, &transN, &sqrt_ndims, &sqrt_ndims, &sqrt_ndims, &a, J, &sqrt_ndims, x+i*ndims, &sqrt_ndims, &b, temp1+i*ndims, &sqrt_ndims);
        dgemm(&transT, &transN, &sqrt_ndims, &sqrt_ndims, &sqrt_ndims, &c, x+i*ndims, &sqrt_ndims, temp1+i*ndims, &sqrt_ndims, &b, temp2+i*ndims, &sqrt_ndims);
        #endif
    }

    // dx = tanh( temp2 )
    #ifdef USING_AMD
//#pragma omp parallel for shared(dx,ndims_numpaths,temp2) private(i) schedule(guided)
    for ( i=0; i<ndims_numpaths; i++ )
    {
        dx[i] = tanh(temp2[i]);
    }
    #else // Intel
    vdTanh(ndims_numpaths, temp2, dx);
    #endif

    // dx = dx - x
    a = -1.;
    size = ndims_numpaths;
    stride = 1;
    daxpy(&size, &a, x, &stride, dx, &stride);

    // dx = gamma*dx
    a = gamma;
    size = ndims_numpaths;
    stride = 1;
    dscal(&size, &a, dx, &stride);

    // Free allocated arrays
    #ifdef STD_FREE
    free(temp1);
    free(temp2);
    #else
    mkl_free(temp1);
    mkl_free(temp2);
    #endif
    #ifndef USING_AMD
    mkl_free_buffers();
    #endif

    return 0;
}

// Getters
int SdeFun::get_ndims()
{
    return ndims;
}
int SdeFun::get_numpaths()
{
    return numpaths;
}
int SdeFun::get_mode()
{
    return mode;
}
double SdeFun::get_a1()
{
    return a1;
}
double SdeFun::get_a2()
{
    return a2;
}
double SdeFun::get_b11()
{
    return b11;
}
double SdeFun::get_b12()
{
    return b12;
}
double SdeFun::get_b21()
{
    return b21;
}
double SdeFun::get_b22()
{
    return b22;
}
double SdeFun::get_gamma()
{
    return gamma;
}
double SdeFun::get_beta()
{
    return beta;
}
double * SdeFun::get_J()
{
    return J;
}
double * SdeFun::get_h()
{
    return h;
}

// Setters
void SdeFun::set_ndims(int val)
{
    ndims = val;
}
void SdeFun::set_numpaths(int val)
{
    numpaths = val;
}
void SdeFun::set_a1(double val)
{
    a1 = val;
}
void SdeFun::set_a2(double val)
{
    a2 = val;
}
void SdeFun::set_b11(double val)
{
    b11 = val;
}
void SdeFun::set_b12(double val)
{
    b12 = val;
}
void SdeFun::set_b21(double val)
{
    b21 = val;
}
void SdeFun::set_b22(double val)
{
    b22 = val;
}
void SdeFun::set_gamma(double val)
{
    gamma = val;
}
void SdeFun::set_beta(double val)
{
    beta = val;
}
void SdeFun::set_J(double * val)
{
    // TODO: copy J to an aligned array
    J = val;
}
void SdeFun::set_h(double * val)
{
    // TODO: copy h to an aligned array
    h = val;
}

// Initialize solver; set pointer to appropriate dynamics function, check that all appropriate parameters have been set
int SdeFun::initialize_solver()
{ 
    // Check valid values of ndims and numpaths have been set.
    if (ndims == -1)
    {
        std::cout << "Error: System dimension ndims has not been set." << std::endl;
        return 1;
    }
    if (numpaths == -1)
    {
        std::cout << "Error: Number of Brownian paths to integrate numpaths has not been set." << std::endl;
        return 2;
    }
    if (ndims <= 0)
    {
        std::cout << "Error: Invalid ndims value " << ndims << std::endl;
        return 3;
    }
    if (numpaths <= 0)
    {
        std::cout << "Error: Invalid numpaths value " << numpaths << std::endl;
        return 4;
    }

    switch ( mode )
    {
        case 10:
            // Brownian dynamics
            //dynamics = std::bind(&SdeFun::brownian, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::brownian, this, _1, _2);
            break;
        case 11:
            // Brownian motion with drift
            if (isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 5;
            }
            //dynamics = std::bind(&SdeFun::brownian_drift, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::brownian_drift, this, _1, _2);
            break;
        case 20:
            // Linear dynamics
            if ( isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 6;
            }

            //dynamics = std::bind(&SdeFun::linear, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::linear, this, _1, _2);
            break;
        case 30:
            // 2-d sine system
            if ( ndims != 2 )
            {
                std::cout << "Error: Invalid number of dimensions for 2-d dynamics " << ndims << std::endl;
                return 7;
            }

            b = 0.;
            stride = 2;
            size = numpaths;

            //dynamics = std::bind(&SdeFun::twoDsine, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::twoDsine, this, _1, _2);
            break;
        case 40:
            // 2-d system
            if ( ndims != 2 )
            {
                std::cout << "Error: Invalid number of dimensions for 2-d dynamics " << ndims << std::endl;
                return 8;
            }

            if ( isnan(a1) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter a1 has not been set." << std::endl;
                return 9;
            }
            if ( isnan(a2) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter a2 has not been set." << std::endl;
                return 10;
            }
            if ( isnan(b11) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter b11 has not been set." << std::endl;
                return 11;
            }
            if ( isnan(b12) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter b12 has not been set." << std::endl;
                return 12;
            }
            if ( isnan(b21) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter b21 has not been set." << std::endl;
                return 13;
            }
            if ( isnan(b22) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter b22 has not been set." << std::endl;
                return 14;
            }

            stride = 2;
            size = numpaths;

            //dynamics = std::bind(&SdeFun::twoDsystem, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::twoDsystem, this, _1, _2);
            break;
        case 50:
            // Kaneko dynamics
            if ( isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 15;
            }
            if ( isnan(beta) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter beta has not been set." << std::endl;
                return 16;
            }
            if ( J == NULL )
            {
                std::cout << "Error: Required SdeFun dynamics parameter J has not been set." << std::endl;
                return 17;
            }

            //dynamics = std::bind(&SdeFun::kaneko, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::kaneko, this, _1, _2);
            break;
        case 51:
            // Kaneko sine dynamics
            if ( isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 15;
            }
            if ( isnan(beta) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter beta has not been set." << std::endl;
                return 16;
            }
            if ( J == NULL )
            {
                std::cout << "Error: Required SdeFun dynamics parameter J has not been set." << std::endl;
                return 17;
            }

            //dynamics = std::bind(&SdeFun::kaneko_sine, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::kaneko_sine, this, _1, _2);
            break;
        case 52:
            // Kaneko notanh dynamics
            if ( isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 15;
            }
            if ( isnan(beta) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter beta has not been set." << std::endl;
                return 16;
            }
            if ( J == NULL )
            {
                std::cout << "Error: Required SdeFun dynamics parameter J has not been set." << std::endl;
                return 17;
            }

            //dynamics = std::bind(&SdeFun::kaneko_notanh, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::kaneko_notanh, this, _1, _2);
            break;
        case 53:
            // Kaneko dynamics with an environmental field
            if ( isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 15;
            }
            if ( isnan(beta) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter beta has not been set." << std::endl;
                return 16;
            }
            if ( J == NULL )
            {
                std::cout << "Error: Required SdeFun dynamics parameter J has not been set." << std::endl;
                return 17;
            }
            if ( h == NULL )
            {
                std::cout << "Error: Required SdeFun dynamics parameter h has not been set." << std::endl;
                return 19;
            }

            //dynamics = std::bind(&SdeFun::kaneko, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::kaneko, this, _1, _2);
            break;
        case 60:
            // Quadratic form dynamics
            if ( isnan(gamma) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter gamma has not been set." << std::endl;
                return 15;
            }
            if ( isnan(beta) )
            {
                std::cout << "Error: Required SdeFun dynamics parameter beta has not been set." << std::endl;
                return 16;
            }
            if ( J == NULL )
            {
                std::cout << "Error: Required SdeFun dynamics parameter J has not been set." << std::endl;
                return 17;
            }

            //dynamics = std::bind(&SdeFun::tanh_quadratic, this, std::placeholders::_1, std::placeholders::_2);
            dynamics = boost::bind(&SdeFun::tanh_quadratic, this, _1, _2);
            break;
        default:
            std::cout << "Error: Invalid SDE solver mode specified: " << mode << std::endl;
            return 18;
    }

    return 0;
}

// Evaluate dynamics on input x, and set the value of dx to the output
int SdeFun::eval_dynamics(double * x, double * dx)
{
    int success;

    success = dynamics(x, dx);
    //success = (*dynamics)(x, dx);

    return success;
}
