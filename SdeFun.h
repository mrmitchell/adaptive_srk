#ifndef _SDEFUN_H_
#define _SDEFUN_H_
// SDE deterministic dynamics functions

#include <algorithm>
#include <cmath>
#include <iostream>
#include <malloc.h>
#include <vector>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#ifdef USING_AMD
#include <acml.h>
#include <omp.h>
#else // Intel
#include "mkl.h"
#include "mkl_vsl.h"
#endif

#define alignment 32

class SdeFun
{
private:
    // System parameters
    int ndims;
    int numpaths;

    // Dynamics mode
    int mode;

    // Dynamics parameters
    double a1;
    double a2;
    double b11;
    double b12;
    double b21;
    double b22;
    double gamma;
    double beta;
    double * J;
    double * h;

    // BLAS parameters
    int size;
    int stride;
    char transN;
    char transT;
    double a;
    double b;
    double c;

    // Dynamics function pointer
    //int (* dynamics)(int *, int *);
    //std::function<int(double*,double*)> dynamics;
    boost::function<int(double*,double*)> dynamics;

    // dynamic 10
    // dx/dt = 0 (+eta)
    // Noise-only dynamics (Brownian motion)
    // params: 
    int brownian(double * x, double * dx);

    // dynamic 11
    // dx/dt = gamma (+eta)
    // Brownian motion with drift
    // params: gamma
    int brownian_drift(double * x, double * dx);

    // dynamic 20
    // dx/dt = gamma*x
    // params: gamma
    int linear(double * x, double * dx);

    // dynamic 30
    // 2-d sine
    // dx/dt = y
    // dy/dt = -x
    // params:
    int twoDsine(double * x, double * dx);

    // dynamic 40
    // 2d system
    // x1' = b11*a1 + b12*a2 - b11*x1 - b12*x2
    // x2' = b21*a1 + b22*a2 - b21*x1 - b22*x2
    // params: b11 b12 b21 b22
    int twoDsystem(double * x, double * dx);

    // dynamic 50
    // Kaneko dynamics
    // gamma * ( tanh( beta * J*x ) - x)
    // params: J gamma beta
    int kaneko(double * x, double * dx);

    // dynamic 51
    // Kaneko dynamics with the tanh replaced by sine
    // gamma * ( sine ( beta * J*x ) - x)
    // params: J gamma beta
    int kaneko_sine(double * x, double * dx);

    // dynamic 52
    // Kaneko dynamics with no tanh
    // gamma * ( beta * J*x - x)
    // params: J gamma beta
    int kaneko_notanh(double * x, double * dx);

    // dynamic 53
    // Kaneko dynamics with an environmental field
    // gamma * ( tanh( beta * J*x ) - x + h)
    // params: J gamma beta h
    int kaneko_env_field(double * x, double * dx);

    // dynamic 60
    // x is an n x n matrix, J is an n x n matrix
    // gamma * ( tanh( beta * x^T * J * x ) - x )
    int tanh_quadratic(double * x, double * dx);

public:

    SdeFun(int modeval);

    int get_ndims();
    int get_numpaths();
    int get_mode();
    double get_a1();
    double get_a2();
    double get_b11();
    double get_b12();
    double get_b21();
    double get_b22();
    double get_gamma();
    double get_beta();
    double * get_J();
    double * get_h();

    void set_ndims(int val);
    void set_numpaths(int val);
    void set_a1(double val);
    void set_a2(double val);
    void set_b11(double val);
    void set_b12(double val);
    void set_b21(double val);
    void set_b22(double val);
    void set_gamma(double val);
    void set_beta(double val);
    void set_J(double * val);
    void set_h(double * val);

    // Initialize solver; set pointer to appropriate dynamics function, check that all appropriate parameters have been set
    int initialize_solver();

    // Evaluate dynamics on input x, and set the value of dx to the output
    int eval_dynamics(double * x, double * dx);

};

#endif // _SDEFUN_H_
