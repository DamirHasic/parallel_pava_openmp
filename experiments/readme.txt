experiments: timings and correctness check
==========================================

  main.cpp      -> run       solves all instances in a folder with the sequential or the parallel PAVA
  compare.cpp   -> compare   compares the sequential and the parallel solutions

Build:  make
        (flags in the Makefile, as in the paper: -O2 -std=c++17 -fopenmp -ffp-contract=off -march=native -DNDEBUG)


run
---
  ./run --algo seq --dir DIR --out FILE
  ./run --algo par --threads 1,2,4,8,12 --dir DIR --out FILE

  Options
    --algo seq|par    sequential PAVA (Algorithm 1) or three-phase parallel PAVA (Algorithm 2)
    --threads LIST    thread counts for --algo par
                      (default: 1, 2, 4, ... up to the number of logical cores, plus that number)
    --dir PATH        folder with the instances
    --out FILE        CSV result file; lines are appended, the header is written if the file is new
    --reps R          timed runs per instance and thread count (default 5)
  Always give --dir and --out: the built-in defaults point to the binary family.

  Input
    Every file <type>_<k>_<n>.bin in DIR (e.g. gauss_3_1000000.bin), processed in the order type, n, k.
    If <type>_<k>_<n>.wbin is in the same folder, it holds the weights; otherwise unit weights.
    Files: raw little-endian 8-byte doubles, no header (see ../instance_generators).

  Output
    One CSV line per instance and thread count, with the columns
      date, instance, type, n, k, weighted, algo, threads, reps,
      t_min, t_median, t_mean, t_sd     seconds, over the R timed runs
      t_phase1, t_phase2, t_phase3      seconds, of the run with the median time (par only)
      blocks_phase1, blocks_final       b' (blocks after Phase 1) and b (blocks of the solution)
      ns_per_point                      t_median / n, in ns
      sse                               sum_i w_i (y_i - x_hat_i)^2
      t_read                            seconds to read the file (not part of the timings)
    The times do not include reading the input and writing x_hat.
    The CSV file is flushed after every line, so an interrupted run keeps its results.

    The solution x_hat is written once per instance, to <folder of FILE>/yhat/:
      <instance>.seq.yhat.bin  or  <instance>.par<P>.yhat.bin  (P = the largest thread count)


compare
-------
  ./compare --dir RESULTS/yhat --out RESULTS/compare.csv    all pairs <inst>.seq.yhat.bin / <inst>.par<P>.yhat.bin
  ./compare A.seq.yhat.bin B.par12.yhat.bin               one pair
  (defaults: --dir results/yhat, --out results/compare.csv; the output file is overwritten)

  Columns: seq_file, par_file, n,
           n_diff      number of i with different values (bitwise)
           max_abs     max_i |a_i - b_i|
           rel         max_abs / max_i |a_i|
           rms         sqrt((1/n) sum_i (a_i - b_i)^2)
           blocks_seq, blocks_par   number of blocks of the two solutions
           monotone    1 if the parallel solution is non-decreasing


EXAMPLE (Gaussian family)
-------------------------
  ./run --algo seq --dir ../instance_generators/1_gauss/instances --out results/gauss/results.csv
  ./run --algo par --threads 1,2,4,8,12 --dir ../instance_generators/1_gauss/instances --out results/gauss/results.csv
  ./compare --dir results/gauss/yhat --out results/gauss/compare.csv

  Random weights: the same with ../instance_generators/1_gauss/instances_w and results/gauss_w/.


  Reading the files in Python: numpy.fromfile(name, dtype="<f8")
