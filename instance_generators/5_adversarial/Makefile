CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

gen_adversarial: gen_adversarial.cpp
	$(CXX) $(CXXFLAGS) -o gen_adversarial gen_adversarial.cpp

.PHONY: clean
clean:
	rm -f gen_adversarial
