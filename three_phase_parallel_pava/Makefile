CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra -march=native

all: libpava.so

libpava.so: pava_c.cpp pava.hpp
	$(CXX) $(CXXFLAGS) -fPIC -shared -o libpava.so pava_c.cpp

test: libpava.so
	python3 test_pava.py

clean:
	rm -f libpava.so

.PHONY: all test clean
