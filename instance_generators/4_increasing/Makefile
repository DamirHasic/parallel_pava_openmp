CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

gen_increasing: gen_increasing.cpp
	$(CXX) $(CXXFLAGS) -o gen_increasing gen_increasing.cpp

.PHONY: clean
clean:
	rm -f gen_increasing
