# Define compiler and remove commands
CXX=g++
RM=rm -f

# Specify path to Eigen headers
EIGENPATH=/usr/include/eigen3

### Optimize gcc compiler flags [ NO DEBUGGING ]
CXXFLAGS=-std=c++17 -I${EIGENPATH} -DNDEBUG -march=native -fopenmp -O3

### Optimize gcc compiler flags [ DEBUGGING ]
#CXXFLAGS=-std=c++17 -I${EIGENPATH} -g -march=native -fopenmp -O3

CFLAGS=-c -Wall

# Define all target list
all: main.cpp GPs.cpp install tests

# Install target list
install: main.o GPs.o
	$(CXX) $(CXXFLAGS) -o Run main.cpp GPs.cpp

# Test target list
tests: test1 test2 test3 test4

# Test targets
test1: tests/1D_example.o GPs.o
	$(CXX) $(CXXFLAGS) -o tests/1D_example tests/1D_example.cpp GPs.cpp

test2: tests/2D_example.o GPs.o
	$(CXX) $(CXXFLAGS) -o tests/2D_example tests/2D_example.cpp GPs.cpp

test3: tests/2D_multimodal.o GPs.o
	$(CXX) $(CXXFLAGS) -o tests/2D_multimodal tests/2D_multimodal.cpp GPs.cpp

test4: tests/1D_low_noise.o GPs.o
	$(CXX) $(CXXFLAGS) -o tests/1D_low_noise tests/1D_low_noise.cpp GPs.cpp

# Object files
main.o: main.cpp GPs.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) $< -o $@

GPs.o: GPs.cpp GPs.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) $< -o $@ 

# Test object files
tests/1D_example.o: tests/1D_example.cpp GPs.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) $< -o $@ 

tests/2D_example.o: tests/2D_example.cpp GPs.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) $< -o $@ 

tests/2D_multimodal.o: tests/2D_multimodal.cpp GPs.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) $< -o $@ 

tests/1D_low_noise.o: tests/1D_low_noise.cpp GPs.h
	$(CXX) $(CFLAGS) $(CXXFLAGS) $< -o $@ 

# Clean
clean:
	rm GPs.o main.o tests/1D_example.o tests/2D_example.o tests/2D_multimodal.o tests/1D_example tests/2D_example tests/2D_multimodal tests/1D_low_noise.o tests/1D_low_noise
