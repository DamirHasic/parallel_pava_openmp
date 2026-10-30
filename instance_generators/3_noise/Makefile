CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

gen_noise: gen_noise.cpp
	$(CXX) $(CXXFLAGS) -o gen_noise gen_noise.cpp

.PHONY: clean
clean:
	rm -f gen_noise
