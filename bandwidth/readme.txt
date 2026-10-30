bandwidth: memory bandwidth with STREAM
=======================================

  stream.c        STREAM benchmark, version 5.10, Copyright 1991-2013 John D. McCalpin,
                  included unchanged under its own license (see the header of stream.c);
                  http://www.cs.virginia.edu/stream/
  run_stream.cpp  compiles stream.c for several array sizes and runs it for several thread counts

Build:  make          (or: g++ -O2 -std=c++17 -o run_stream run_stream.cpp)
Run:    ./run_stream
        ./run_stream --sizes 1e6,1e7,1e8 --threads 1,2,4,8,12 --ntimes 20 --out results/stream.csv

Defaults: n = 10^6, 10^7, 10^8;  P = 1, 2, 4, 8, 12;  20 repetitions per kernel (STREAM discards the
first one and reports the best of the others).

For every n, stream.c is compiled with
    gcc -O2 -fopenmp -march=native -mcmodel=medium -DSTREAM_ARRAY_SIZE=n -DNTIMES=ntimes stream.c -o stream_n
(-mcmodel=medium is needed because for n = 10^8 the three arrays take 2.4 GB) and run with
OMP_NUM_THREADS=P. Other OpenMP variables are taken from the shell, as for ../experiments/run.


OUTPUT
------
  results/stream.csv   one line per (n, P, kernel), appended:
                       date, n, threads, threads_counted, ntimes, kernel, bytes_per_elem, rate_GBps,
                       t_avg, t_min, t_max, ns_per_elem (= t_min / n, comparable with ns_per_point
                       of ../experiments/run), validates
  results/raw/         the raw STREAM output, stream_n<n>_P<P>.txt

Bytes per element as counted by STREAM (without write-allocate): Copy and Scale 16, Add and Triad 24.
Scale has the memory traffic of PAVA with unit weights (read y, write x_hat), and Add that of PAVA
with random weights (read y and w, write x_hat). The paper uses Scale with n = 10^8.
