CC=gcc
CXX=g++
RM=rm -f
FLAGS=-std=c++17 -fopenmp
CFLAGS=-c -Wall 
HEADERS=GPs.h
SOURCES=main.cpp GPs.cpp
OBJECTS=$(SOURCES:.cpp=.o)
RUNFILE=Run

all: $(SOURCES) install

install: $(OBJECTS)
	$(CXX) $(FLAGS) -o $(RUNFILE) $(SOURCES)

main.o: main.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

GPs.o: GPs.cpp $(HEADER)
	$(CXX) $(CFLAGS) $(FLAGS) $< -o $@

clean:
	rm $(OBJECTS) $(RUNFILE)
