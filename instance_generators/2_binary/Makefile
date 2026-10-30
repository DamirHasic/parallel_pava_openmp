CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

gen_binary: gen_binary.cpp
	$(CXX) $(CXXFLAGS) -o gen_binary gen_binary.cpp

.PHONY: clean
clean:
	rm -f gen_binary
