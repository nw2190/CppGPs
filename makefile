CC=gcc
CXX=g++
RM=rm -f
#FLAGS=-std=c++17 -fopenmp -floop-parallelize-all -ftree-parallelize-loops=4
#FLAGS=-std=c++17 -g -fopenmp -ftree-vectorize -mavx -ftree-vectorizer-verbose=2 -march=native

### STANDARD
#FLAGS=-std=c++17 -g -march=native -fopenmp

### GCC OPTIMIZED
FLAGS=-std=c++17 -g -march=native -fopenmp -O3

### GCC OPTIMIZED + AVX
#FLAGS=-std=c++17 -g -march=native -fopenmp -mavx -O3

### INTEL MKL
#OMPROOT=/opt/intel/compilers_and_libraries_2019.2.187/linux/compiler/lib
#LIBIOMP=/opt/intel/compilers_and_libraries_2019.2.187/linux/compiler/lib/intel64/libiomp5.a
#MKLROOT=/opt/intel/mkl
#MKLPATH=${MKLROOT}/lib/intel64_lin
#MKLINCLUDE=${MKLROOT}/include
#FLAGS=-L${MKLPATH} -I${MKLINCLUDE} ${MKLPATH}/libmkl_intel_lp64.a ${MKLPATH}/libmkl_intel_thread.a ${MKLPATH}/libmkl_core.a ${MKLPATH}/libmkl_intel_lp64.a ${MKLPATH}/libmkl_intel_thread.a ${MKLPATH}/libmkl_core.a ${LIBIOMP} -lpthread -lm -O3

# 
CFLAGS=-c -Wall 
HEADERS=GPs.h minimize.h utils.h
SOURCES=main.cpp GPs.cpp minimize.cpp utils.cpp
OBJECTS=$(SOURCES:.cpp=.o)
RUNFILE=Run

all: $(SOURCES) install

install: $(OBJECTS)
	$(CXX) $(FLAGS) -o $(RUNFILE) $(SOURCES)

main.o: main.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

GPs.o: GPs.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

minimize.o: minimize.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

utils.o: utils.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

clean:
	rm $(OBJECTS) $(RUNFILE)
