# 'make' to compile without debugging symbols but with Boost library assertions enables
# 'make profile' to compile with debugging/profiling symbols and with PROFILE defined, but with NDEBUG and without DEBUG defined
# 'make debug' to compile with debugging symbols and DEBUG defined
# 'make release' to compile without Boost assertions

################################################################################
# Set options and paths
################################################################################
intel_or_amd=# 'intel' or 'amd'

# Library name: *.so or *.dylib
dylib_ext=so

boost_path=#PATH_TO_BOOST
boostdir=#PATH_TO_BOOST

# Intel paths:
mkl_path=#PATH_TO_MKL

# AMD paths:
acml_path=#PATH_TO_ACML

################################################################################
# Shouldn't need to edit
################################################################################

ifeq ($(intel_or_amd),amd) # AMD
    CXX=g++
    CXXFLAGS=-DUSING_AMD -I$(acml_path)/include -ftree-parallelize-loops=64 -fopenmp -O3 -march=native
    LDFLAGS=-L$(acml_path)lib -Mcache_align -lacml_mp -lm -lgfortran

    dylib_flags=-shared

    boost_program_options=boost_program_options
    boost_date_time=boost_date_time
    boost_timer=boost_timer
    boost_system=boost_system

    boost_program_options_d=boost_program_options
    boost_date_time_d=boost_date_time
    boost_timer_d=boost_timer
    boost_system_d=boost_system
else # Intel
    CXX=icpc
    CXXFLAGS=-I$(mkl_path)/include -O3 -parallel -par-threshold50 -ipo -xHost -mavx -fno-alias
    LDFLAGS=-L$(mkl_path)/lib
    LDLIBS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

    dylib_flags=-dynamiclib

    boost_program_options=boost_program_options-mt
    boost_date_time=boost_date_time-mt
    boost_timer=boost_timer-mt
    boost_system=boost_system-mt

    boost_programyoptions_d=boost_program_options-mt-d
    boost_date_time_d=boost_date_time-mt-d
    boost_timer_d=boost_timer-mt-d
    boost_system_d=boost_system-mt-d
endif

CXXFLAGS+=-I$(boostdir)/include -g
LDFLAGS+=-g
LDLIBS+=-L$(boostdir)/lib -l$(boost_program_options) -l$(boost_date_time) -l$(boost_timer) -l$(boost_system)
SRCS=adaptive_srk.cpp SdeFun.cpp

main : $(SRCS:.cpp=.o)
	$(CXX) $(LDFLAGS) $(SRCS:.cpp=.o) -o ../bin/sdesolve.out $(LDLIBS)

dylib : $(SRCS:.cpp=.o)
	$(CXX) $(LDFLAGS) $(SRCS:.cpp=.o) -o ../bin/libsde_solver.$(dylib_ext) $(LDLIBS)

profile: CXXFLAGS+=-DPROFILE -DNDEBUG
profile: LDFLAGS+=-DPROFILE -DNDEBUG
profile: LDLIBS+=-l$(boost_timer) -l$(boost_system) -l$(boost_program_options) -l$(boost_date_time)
profile: main

debug: CXXFLAGS+=-g -DPROFILE -DDEBUG -pg
debug: LDFLAGS+=-g -DPROFILE -DDEBUG -pg
debug: LDLIBS+=-l$(boost_program_options_d) -l$(boost_date_time_d) -l$(boost_timer_d) -l$(boost_system_d)
debug: main

release: CXXFLAGS+=-DNDEBUG
release: LDFLAGS+=-DNDEBUG
release: LDLIBS+=-l$(boost_program_options) -l$(boost_date_time) -l$(boost_timer) -l$(boost_system)
release: main

library: CXXFLAGS+=-DNDEBUG -fPIC $(dylib_flags)
library: LDFLAGS+=-DNDEBUG -fPIC $(dylib_flags)
library: dylib

%.o : %.c
	$(CXX) $(CXXFLAGS) -c $<

srk_test.o : adaptive_srk.h SdeFun.h Arithmetic.h
adaptive_srk.o : adaptive_srk.h SdeFun.h
SdeFun.o : SdeFun.h

.PHONY : clean
clean :
	rm -rf $(SRCS:.cpp=.o)
