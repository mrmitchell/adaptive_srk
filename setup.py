"""Setup file to build adaptive SRK solver Python C-extension module."""
# Execute using
# python setup.py build

from distutils.core import setup, Extension

################################################################################
# Set options and paths
################################################################################
intel_or_amd = 'AMD' # Change to 'Intel' if appropriate

numpy_include = 'PATH_TO_NUMPY_INCLUDE'
boost_path = 'PATH_TO_BOOST'
# Path to compiled C++ solver libary libsde_solver.dylib/so
solver_path = 'PATH_TO_SDE_SOLVER'

# If AMD
acml_path = 'PATH_TO_ACML'
# If Intel
mkl_path = 'PATH_TO_MKL'

################################################################################
# Shouldn't need to edit
################################################################################
if intel_or_amd == 'AMD':
    def_macros = [('STD_FREE', None), ('USING_AMD', None)]
    inc_dirs = [numpy_include, acml_path+'/include', boost_path+'/include']
    lib_dirs = [acml_path+'/lib', boost_path+'/lib', solver_path]
    runlib_dirs = [acml_path+'/lib', boost_path+'/lib', solver_path]
    libs = ['sde_solver', 'acml_mp', 'm', 'gfortran', 'boost_timer', 'boost_system']
else:
    def_macros = []
    inc_dirs = [numpy_include, mkl_path+'/include', boost_path+'/include']
    lib_dirs = [mkl_path+'/lib', boost_path+'/lib', solver_path]
    runlib_dirs = [mkl_path+'/lib', boost_path+'/lib', solver_path]
    libs = ['sde_solver', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread', 'm']

module1 = Extension('sdepy',
                    define_macros=def_macros,
                    include_dirs=inc_dirs,
                    library_dirs=lib_dirs,
                    runtime_library_dirs=runlib_dirs,
                    libraries=libs,
                    sources=['sdepy.cpp'])

setup(name='PackageName',
      version='1.0',
      description='Python module wrapper for adaptive_srk.cpp stochastic differential equation solver library.',
      ext_modules=[module1])
