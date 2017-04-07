// Python module wrapper function for adaptive_srk

#include "Python.h"
#include "numpy/arrayobject.h"

#include <cmath>
#include <iostream>
#include <vector>

#include <boost/timer/timer.hpp>

#ifdef USING_AMD
#include "acml.h"
#else // Intel
#include "mkl.h"
#include "mkl_vsl.h"
#define alignment 32
#endif

#include "SdeFun.h"         // Deterministic dynamics function class
#include "adaptive_srk.h"   // solver

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

// Python function syntax: output_tuple = sde_solver(x0, dt_max, abstol, reltol, tf, m, sigma, numpaths, dynamics_mode, params)
// output_tuple = (x_history, t_history, h_history, err_history, call_time)
static PyObject * solve(PyObject * self, PyObject * args)
{
    // Helper variables
    int status;
    int numel;
    int i;

    // BLAS helper variables
    int stride = 1;

    ////////////////////////////////////////////////////////////////////////////
    // Declare input variables
    ////////////////////////////////////////////////////////////////////////////
    npy_intp * ndims_ptr;
    int ndims;
    PyObject * x0_py;
    double dt_max;
    double abstol;
    double reltol;
    double tf;
    int m;
    PyObject * sigma_py;
    int numpaths;
    int dynamics_mode;
    PyObject * params_py;

    ////////////////////////////////////////////////////////////////////////////
    // Parse inputs
    ////////////////////////////////////////////////////////////////////////////
    //if (!PyArg_ParseTuple(args, "O&ddddiO&iiO&", PyArray_DescrConverter, &x0_py, &dt_max, &abstol, &reltol,
    //            &tf, &m, PyArray_DescrConverter, &sigma_py, &numpaths, &dynamics_mode, PyArray_DescrConverter,
    //            &params_py)) return NULL;
    if (!PyArg_ParseTuple(args, "O!ddddiO!iiO!", &PyArray_Type, &x0_py, &dt_max, &abstol, &reltol,
                &tf, &m, &PyArray_Type, &sigma_py, &numpaths, &dynamics_mode, &PyArray_Type,
                &params_py)) return NULL;

    ////////////////////////////////////////////////////////////////////////////
    // Pull local C variables/arrays out of the PyObject arguments
    ////////////////////////////////////////////////////////////////////////////
    // ndims
    if ( PyArray_NDIM(x0_py) != 1 )
    {
        PyErr_SetString(PyExc_ValueError,"Argument x0 shoule be 1-dimensional.");
        return NULL;
    }
    ndims_ptr = PyArray_DIMS(x0_py);
    ndims = (int)*ndims_ptr;

    // x0
#ifdef STD_FREE
    double * x0 = (double *) malloc( ndims * sizeof(double) );
#else
    double * x0 = (double *) mkl_malloc( ndims * sizeof(double), alignment );
#endif
    dcopy( &ndims, (double *)PyArray_DATA(x0_py), &stride, x0, &stride );

    // sigma
    std::vector<double> sigma((int)PyArray_SIZE(sigma_py));
    numel = sigma.size();
    dcopy(&numel, (double *)PyArray_DATA(sigma_py), &stride, sigma.data(), &stride);
    
    // params
#ifdef STD_FREE
    double * params = (double *) malloc ( PyArray_SIZE(params_py) * sizeof(double) );
#else
    double * params = (double *) mkl_malloc ( PyArray_SIZE(params_py) * sizeof(double), alignment );
#endif
    numel = (int) PyArray_SIZE(params_py);
    dcopy( &numel, (double *)PyArray_DATA(params_py), &stride, params, &stride);

    ////////////////////////////////////////////////////////////////////////////
    // Declare C++ output variables
    ////////////////////////////////////////////////////////////////////////////

    std::vector<double *> x_history;

    std::vector<double> t_history;

    std::vector<double> h_history;

    std::vector<double> err_history;

    ////////////////////////////////////////////////////////////////////////////
    // Initialize SdeFun object
    ////////////////////////////////////////////////////////////////////////////
    SdeFun sdefun(dynamics_mode);
    sdefun.set_ndims(ndims);
    sdefun.set_numpaths(numpaths);
    switch ( dynamics_mode )
    {
        case 10:
            break;
        case 11:
            sdefun.set_gamma(*params);
            break;
        case 20:
            sdefun.set_gamma(*params);
            break;
        case 30:
            break;
        case 40:
            sdefun.set_a1(*params);
            sdefun.set_a2(*(params + 1));
            sdefun.set_b11(*(params + 2));
            sdefun.set_b12(*(params + 3));
            sdefun.set_b21(*(params + 4));
            sdefun.set_b22(*(params + 5));
        case 50:
            sdefun.set_gamma(*params);
            sdefun.set_beta(*(params + 1));
            sdefun.set_J(params + 2);
            break;
        case 51:
            sdefun.set_gamma(*params);
            sdefun.set_beta(*(params + 1));
            sdefun.set_J(params + 2);
            break;
        case 52:
            sdefun.set_gamma(*params);
            sdefun.set_beta(*(params + 1));
            sdefun.set_J(params + 2);
            break;
        case 53:
            sdefun.set_gamma(*params);
            sdefun.set_beta(*(params + 1));
            sdefun.set_J(params + 2);
            sdefun.set_h(params + 2 + ndims*ndims);
            break;
        case 60:
            sdefun.set_gamma(*params);
            sdefun.set_beta(*(params + 1));
            sdefun.set_J(params + 2);
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError,"Invalid dynamics mode specified.");
            return NULL;
            break;
    }
    status = sdefun.initialize_solver();
    switch ( status )
    {
        case 0:
            // Success
            break;
        case 1:
            PyErr_SetString(PyExc_RuntimeError,"ndims must be set before initialize_solver can be called.");
            return NULL;
            break;
        case 2:
            PyErr_SetString(PyExc_RuntimeError,"numpaths must be set before initialize_solver can be called.");
            return NULL;
            break;
        case 3:
            PyErr_SetString(PyExc_RuntimeError,"Negative ndims value encountered. ndims must be positive.");
            return NULL;
            break;
        case 4:
            PyErr_SetString(PyExc_RuntimeError,"Negative numpaths value encountered. numpaths must be positive.");
            return NULL;
            break;
        case 5:
            PyErr_SetString(PyExc_RuntimeError,"gamma must be set for Brownian motion with drift (mode 11) dynamics.");
            return NULL;
            break;
        case 6:
            PyErr_SetString(PyExc_RuntimeError,"gamma must be set for linear (mode 20) dynamics.");
            return NULL;
            break;
        case 7:
            PyErr_SetString(PyExc_RuntimeError,"Invalid ndims value encountered. ndims must be 2 for the 2-d sine system (mode 30) dynamics.");
            return NULL;
            break;
        case 8:
            PyErr_SetString(PyExc_RuntimeError,"Invalid ndims value encountered. ndims must be 2 for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 9:
            PyErr_SetString(PyExc_RuntimeError,"a1 must be set before initialize_solver can be called for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 10:
            PyErr_SetString(PyExc_RuntimeError,"a2 must be set before initialize_solver can be called for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 11:
            PyErr_SetString(PyExc_RuntimeError,"b11 must be set before initialize_solver can be called for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 12:
            PyErr_SetString(PyExc_RuntimeError,"b12 must be set before initialize_solver can be called for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 13:
            PyErr_SetString(PyExc_RuntimeError,"b21 must be set before initialize_solver can be called for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 14:
            PyErr_SetString(PyExc_RuntimeError,"b22 must be set before initialize_solver can be called for the coupled 2-d system (mode 40) dynamics.");
            return NULL;
            break;
        case 15:
            PyErr_SetString(PyExc_RuntimeError,"gamma must be set for Kaneko (mode 50) dynamics.");
            return NULL;
            break;
        case 16:
            PyErr_SetString(PyExc_RuntimeError,"beta must be set for Kaneko (mode 50) dynamics.");
            return NULL;
            break;
        case 17:
            PyErr_SetString(PyExc_RuntimeError,"J must be set for Kaneko (mode 50) dynamics.");
            return NULL;
            break;
        case 18:
            PyErr_SetString(PyExc_RuntimeError,"Invalid solver mode specified for SdeFun.");
            return NULL;
            break;
        case 19:
            PyErr_SetString(PyExc_RuntimeError,"h must be set for Kaneko (mode 53) dynamics.");
            return NULL;
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError,"Undefined error status encountered in SdeFun::initialize_solver().");
            return NULL;
            break;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Call solver
    ////////////////////////////////////////////////////////////////////////////
    boost::timer::cpu_timer sdesolve_timer;

    status = solver(
            // Outputs
            &x_history, &t_history, &h_history, &err_history,
            // Inputs
            &ndims, x0, &sdefun, &dt_max, &abstol, &reltol, &tf, &m, &sigma, &numpaths);

    switch (status)
    {
        case 0:
            // success
            break;
        case 1:
            PyErr_SetString(PyExc_RuntimeError,"Interrupt signal caught during adaptive_srk::solver().");
            return NULL;
            break;
        case 2:
            PyErr_SetString(PyExc_RuntimeError,"Invalid standard deviation argument reported by adaptive_srk::solver().");
            return NULL;
            break;
        case 3:
            PyErr_SetString(PyExc_RuntimeError,"Error status reported by SdeFun::eval_dynamics() in adaptive_srk::solver().");
            return NULL;
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError,"Undefined error status reported by adaptive_srk::solver().");
            return NULL;
            break;
    }
    
    sdesolve_timer.stop();

    // Free allocated arrays
#ifdef STD_FREE
    free(x0);
    free(params);
#else
    mkl_free(x0);
    mkl_free(params);
#endif


    ////////////////////////////////////////////////////////////////////////////
    // Pack outputs in NumPy arrays
    ////////////////////////////////////////////////////////////////////////////
    int nd;
    npy_intp dims3[3];
    npy_intp dims1[1];

    // x_history
    nd = 3;
    dims3[0] = ndims;
    dims3[1] = numpaths;
    dims3[2] = x_history.size();
    PyObject * x_history_py = PyArray_New(&PyArray_Type, nd, dims3, NPY_DOUBLE, NULL, NULL, sizeof(double), 1, NULL);
    double * x_history_py_ptr = (double *)PyArray_DATA(x_history_py);
    numel = ndims*numpaths;
    for ( i=0; i<(int)x_history.size(); i++ )
    {
        dcopy( &numel, x_history[i], &stride, x_history_py_ptr + i*numel, &stride);
#ifdef STD_FREE
        free(x_history[i]);
#else
        mkl_free(x_history[i]);
#endif
    }

    // t_history
    nd = 1;
    numel = t_history.size();
    dims1[0] = numel;
    PyObject * t_history_py = PyArray_New(&PyArray_Type, nd, dims1, NPY_DOUBLE, NULL, NULL, sizeof(double), 1, NULL);
    double * t_history_py_ptr = (double *) PyArray_DATA(t_history_py);
    dcopy( &numel, t_history.data(), &stride, t_history_py_ptr, &stride);

    // h_history
    nd = 1;
    numel = h_history.size();
    dims1[0] = numel;
    PyObject * h_history_py = PyArray_New(&PyArray_Type, nd, dims1, NPY_DOUBLE, NULL, NULL, sizeof(double), 1, NULL);
    double * h_history_py_ptr = (double *) PyArray_DATA(h_history_py);
    dcopy(&numel, h_history.data(), &stride, h_history_py_ptr, &stride);

    // err_history
    nd = 1;
    numel = err_history.size();
    dims1[0] = numel;
    PyObject * err_history_py = PyArray_New(&PyArray_Type, nd, dims1, NPY_DOUBLE, NULL, NULL, sizeof(double), 1, NULL);
    double * err_history_py_ptr = (double *) PyArray_DATA(err_history_py);
    dcopy(&numel, err_history.data(), &stride, err_history_py_ptr, &stride);

    // call_time
    nd = 1;
    numel = 2;
    dims1[0] = 2;
    PyObject * call_time_py = PyArray_New(&PyArray_Type, nd, dims1, NPY_DOUBLE, NULL, NULL, sizeof(double), 1, NULL);
    double * call_time_py_ptr = (double *) PyArray_DATA(call_time_py);
    call_time_py_ptr[0] = (double)sdesolve_timer.elapsed().user;
    call_time_py_ptr[1] = (double)sdesolve_timer.elapsed().wall;

#ifndef STD_FREE
    mkl_free_buffers();
#endif

    ////////////////////////////////////////////////////////////////////////////
    // Pack output arrays in Numpy tuple and return it
    ////////////////////////////////////////////////////////////////////////////

    PyObject * output_tuple = Py_BuildValue("NNNNN",x_history_py,t_history_py,h_history_py,err_history_py,call_time_py);
//    Py_DECREF(x_history_py);
//    Py_DECREF(t_history_py);
//    Py_DECREF(h_history_py);
//    Py_DECREF(err_history_py);
//    Py_DECREF(call_time_py);
//    Py_DECREF(output_tuple);
//    Py_INCREF(output_tuple);

    return output_tuple;
}

static PyMethodDef sdepymethods[] = { 
    {"solve",solve,METH_VARARGS,"Create an SdeFun instance and use it to call adaptive_srk::solver() on the inputs."},
    {NULL,NULL,0,NULL}
};


extern "C" PyMODINIT_FUNC initsdepy(void)
{
    (void)Py_InitModule("sdepy",sdepymethods);
    import_array();
}
