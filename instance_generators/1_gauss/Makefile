CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

gen_gauss: gen_gauss.cpp
	$(CXX) $(CXXFLAGS) -o gen_gauss gen_gauss.cpp

.PHONY: clean
clean:
	rm -f gen_gauss
