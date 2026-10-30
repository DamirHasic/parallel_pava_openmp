CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fopenmp -ffp-contract=off -march=native -DNDEBUG -Wall -Wextra

all: run compare

run: main.cpp ../three_phase_parallel_pava/pava.hpp
	$(CXX) $(CXXFLAGS) -o run main.cpp

compare: compare.cpp
	$(CXX) -O2 -std=c++17 -Wall -Wextra -o compare compare.cpp

.PHONY: all clean
clean:
	rm -f run compare
