#ifndef _ARITHMETIC_H_
#define _ARITHMETIC_H_

// Miscellaneous arithmetic functions

#include <algorithm>
#include <vector>

#ifdef USING_AMD
#include "acml.h"
#else // Intel
#include "mkl.h"
#include "mkl_vsl.h"
#endif

#define alignment 32

// Row- or column-wise mean of a 2-d array
int mean_2d(double * array, int m, int n, double * out, int dim, char order);

// Row- or column-wise variance estimate from a 2-d array
int var_2d(double * array, int m, int n, double * out, int dim, char order, double * mean = NULL);


#endif // _ARITHMETIC_H_
