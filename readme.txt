Three-phase parallel PAVA for weighted L2 isotonic regression (C++17 / OpenMP)
==============================================================================

Code for the paper
  "Three-Phase Parallel PAVA for Weighted L2 Isotonic Regression: Analysis and OpenMP Implementation".

Problem: given y_1, ..., y_n (already sorted by the covariate) and weights w_i > 0, find
  x_hat = argmin sum_i w_i (y_i - x_i)^2   subject to   x_1 <= x_2 <= ... <= x_n.

The algorithms are in one header file, three_phase_parallel_pava/pava.hpp:
  pava::isotonic_seq          sequential PAVA (Algorithm 1 of the paper + expansion step)
  pava::isotonic_threephase   three-phase parallel PAVA with OpenMP (Algorithm 2 of the paper)


FOLDERS
-------
  three_phase_parallel_pava/              pava.hpp (header-only library), C API (pava_c.cpp), correctness tests
  experiments/          run:     timings of the sequential and the parallel PAVA on instance files
                        compare: checks that the parallel and the sequential solutions agree
  instance_generators/  generators of the five instance families of Section 5
                        (Gaussian, binary, pure noise, increasing, adversarial)
  bandwidth/            STREAM memory-bandwidth benchmark (Section 5, comparison to memory bandwidth)

Every folder has its own readme.txt.


REQUIREMENTS
------------
  Linux, g++ with C++17 and OpenMP (the paper used g++ 13.4.0), gcc (for STREAM), make.
  Memory: n = 10^8 needs about 4-7 GB of RAM.
  Disk: an instance file takes 8n bytes (800 MB for n = 10^8). One family (unit and random weights,
  n = 10^6, 10^7, 10^8, ten instances each) takes about 27 GB, and the saved solutions about 36 GB more.


USING THE LIBRARY
-----------------
  #include "three_phase_parallel_pava/pava.hpp"
  #include <vector>

  int main() {
      std::vector<double> y = {1, 3, 2, 4, 3, 5};
      std::vector<double> x_hat(y.size());
      std::vector<pava::Block> blk(y.size());               // work array, n blocks

      // unit weights: w = nullptr; 4 threads
      int64_t b = pava::isotonic_threephase(y.data(), nullptr, (int64_t)y.size(),
                                            x_hat.data(), blk.data(), 4);
      // x_hat = 1 2.5 2.5 3.5 3.5 5,  b = 4 blocks
  }

  Compile:  g++ -O2 -std=c++17 -fopenmp example.cpp
  Sequential version: pava::isotonic_seq(y, w, n, x_hat, blk). Details in three_phase_parallel_pava/readme.txt.


BUILD EVERYTHING
----------------
  make -C three_phase_parallel_pava              # libpava.so (needed only for the tests);  make -C three_phase_parallel_pava test  runs the tests
  make -C experiments          # run, compare
  for d in instance_generators/*/; do make -C "$d"; done
  make -C bandwidth            # run_stream

  The paper's compiler options -O2 -std=c++17 -fopenmp -march=native -ffp-contract=off are set in
  experiments/Makefile. No thread pinning was used.


REPRODUCING Section 5
---------------------
Example for the Gaussian family; the other families are the same with their own generator.

1. Instances (ten per n; unit weights in instances/, random weights in instances_w/):
     cd instance_generators/1_gauss
     mkdir -p instances instances_w          # the generators do not create folders
     for n in 1e6 1e7 1e8; do ./gen_gauss --n $n --instances 10; done
     for n in 1e6 1e7 1e8; do ./gen_gauss --n $n --instances 10 --weighted 1 --dir instances_w; done
     cd ../..
   For the adversarial family with random weights, add --c 2 (see instance_generators/readme.txt).

2. Timings and correctness check (5 timed runs per instance, P = 1, 2, 4, 8, 12 as in the paper):
     cd experiments
     ./run --algo seq --dir ../instance_generators/1_gauss/instances --out results/gauss/results.csv
     ./run --algo par --threads 1,2,4,8,12 --dir ../instance_generators/1_gauss/instances --out results/gauss/results.csv
     ./compare --dir results/gauss/yhat --out results/gauss/compare.csv
   and the same with ../instance_generators/1_gauss/instances_w and results/gauss_w/.
   The paper used --reps 50 for n = 10^6 (except the Gaussian family): put the n = 10^6 instances
   in a separate folder and run that folder with --reps 50.

3. Memory bandwidth:
     cd bandwidth
     ./run_stream



FILE FORMAT
-----------
  Instances and solutions are raw little-endian 8-byte doubles, no header, 8n bytes.
  Python: numpy.fromfile(name, dtype="<f8")


THIRD-PARTY CODE
----------------
  bandwidth/stream.c is the STREAM benchmark, version 5.10, Copyright 1991-2013 John D. McCalpin,
  included unchanged under its own license (see the header of the file).
