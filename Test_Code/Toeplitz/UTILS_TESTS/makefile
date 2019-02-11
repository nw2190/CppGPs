CC=gcc
CXX=g++
RM=rm -f
#FLAGS=-std=c++17 -fopenmp -floop-parallelize-all -ftree-parallelize-loops=4
FLAGS=-std=c++17 -g -fopenmp -ftree-vectorize -mavx -ftree-vectorizer-verbose=2
CFLAGS=-c -Wall 
HEADERS=utils.h
SOURCES=main.cpp utils.cpp
OBJECTS=$(SOURCES:.cpp=.o)
RUNFILE=Run

all: $(SOURCES) install

install: $(OBJECTS)
	$(CXX) $(FLAGS) -o $(RUNFILE) $(SOURCES)

main.o: main.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

utils.o: utils.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

clean:
	rm $(OBJECTS) $(RUNFILE)
