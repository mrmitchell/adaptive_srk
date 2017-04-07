#include "Arithmetic.h"

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

#ifdef STD_FREE
#define mkl_free free
#endif

// Miscellaneous arithmetic functions

// Row- or column-wise mean of a 2-d array
int mean_2d(double * array, int m, int n, double * out, int dim, char order)
{
    // Check order is 'C' or 'R'
    if ( order!='C' && order!='R' )
    {
        return 1;
    }

    // Loop indices
    int i;

    // BLAS variables
    int stridea,strideb;
    double a,b;
    int step;

    // Zero out entries of out, which is expected to be of length
    #pragma parallel
    #pragma ivdep
    #pragma vector aligned
    std::fill(out, out + (dim==1 ? n : m), 0.);

    if ( dim==1 )
    {
        a = 1./m;
        stridea = order=='C' ? n : 1;
        strideb = 1;
        step = order=='C' ? 1 : n;
        for ( i=0; i<m; i++ )
        {
            daxpy( &n, &a, array + i*step, &stridea, out, &strideb );
        }
    }
    else if ( dim==2 )
    {
        a = 1./n;
        stridea = order=='C' ? 1 : m;
        strideb = 1;
        step  = order=='C' ? m : 1;
        for ( i=0; i<n; i++ )
        {
            daxpy( &m, &a, array + i*step, &stridea, out, &strideb );
        }
    }
    else
    {
        return 2;
    }

    return 0;
}


// Row- or column-wise variance estimate from a 2-d array
int var_2d(double * array, int m, int n, double * out, int dim, char order, double * mean)
{
    // Loop variable
    int i;

    // Vector length
    int size = (dim==1 ? n : m);
    int loopsize = (dim==1 ? m : n);

    // BLAS variables
    int stridea,strideb;
    double a,b;
    double a2,a3,b2,b3;
    int stridea2,stridea3,strideb2,strideb3;
    int step;

    // Zero out entries of out, which is expected to be of length size
    #pragma parallel
    #pragma ivdep
    #pragma vector aligned
    std::fill(out, out + size, 0.);

    // Check order is 'C' or 'R'
    if ( order!='C' && order!='R' )
    {
        return -1;
    }
    // Check dim is 1 or 2
    if ( dim!=1 && dim!=2 )
    {
        return -1;
    }

    // scratch vector
#ifdef ADDRESS_SANITIZER
    double * temp1 = (double *) malloc( size * sizeof(double) );
    double * temp2 = (double *) malloc( size * sizeof(double) );

    double * local_mean = (double *) malloc( size * sizeof(double) );
#else
    double * temp1 = (double *) mkl_malloc( size * sizeof(double), alignment );
    double * temp2 = (double *) mkl_malloc( size * sizeof(double), alignment );

    // Make a copy of mean
    double * local_mean = (double *) mkl_malloc( size * sizeof(double), alignment );
#endif

    if ( mean == NULL )
    {
        // Need to compute the mean
        int status = mean_2d(array, m, n, local_mean, dim, order);
        if ( status != 0 )
        {
            return -1;
        }
    }
    else
    {
        // Make a local copy of mean
        stridea = 1;
        strideb = 1;
        dcopy( &size, mean, &stridea, local_mean, &strideb );
    }

    // Compute variance
    if ( dim==1 )
    {
        if ( order=='C' )
        {
            stridea = m;
            strideb = 1;
            step = 1;
            
            stridea2 = 1;
            strideb2 = 1;
            a2 = -1.;

            stridea3 = 1;
            strideb3 = 1;
            a3 = 1./( (loopsize>1 ? loopsize-1 : loopsize) );
        }
        else // order=='R'
        {
            stridea = 1;
            strideb = 1;
            step = n;

            stridea2 = 1;
            strideb2 = 1;
            a2 = -1.;

            stridea3 = 1;
            strideb3 = 1;
            a3 = 1./( (loopsize>1 ? loopsize-1 : loopsize) );
        }
    }
    else // dim==2
    {
        if ( order=='C' )
        {
            stridea = 1;
            strideb = 1;
            step = m;

            stridea2 = 1;
            strideb2 = 1;
            a2 = -1.;

            stridea3 = 1;
            strideb3 = 1;
            a3 = 1./( (loopsize>1 ? loopsize-1 : loopsize) );
        }
        else // order=='R'
        {
            stridea = n;
            strideb = 1;
            step = 1;

            stridea2 = 1;
            strideb2 = 1;
            a2 = -1.;

            stridea3 = 1;
            strideb3 = 1;
            a3 = 1./( (loopsize>1 ? loopsize-1 : loopsize) );
        }
    }
    // Loop to compute variance
    #pragma parallel
    #pragma ivdep
    #pragma vector aligned
    for ( i=0; i<loopsize; i++ )
    {
        // Copy row of the array to a vector storing the entries contiguously in memory
        dcopy( &size, array + i*step, &stridea, temp1, &strideb );
        // row - mean
        daxpy( &size, &a2, local_mean, &stridea2, temp1, &strideb2 );
        // (row - mean)^2
        vdSqr( size, temp1, temp2 );
        
        // out += 1/(n-1) * (row-mean)^2
        daxpy( &size, &a3, temp2, &stridea3, out, &strideb3 );
    }

    // Free allocated arrays
    mkl_free_buffers();
    mkl_free(temp1);
    mkl_free(temp2);
    mkl_free(local_mean);

    return 0;
}
