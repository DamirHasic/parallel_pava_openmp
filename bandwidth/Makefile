CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

run_stream: run_stream.cpp
	$(CXX) $(CXXFLAGS) -o run_stream run_stream.cpp

.PHONY: clean
clean:
	rm -f run_stream stream_*
