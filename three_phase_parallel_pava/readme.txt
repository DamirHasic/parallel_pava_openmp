three_phase_parallel_pava: the algorithms
=======================

  pava.hpp       header-only C++17/OpenMP library (namespace pava); this one file is all you need
  pava_c.cpp     C API (for Python ctypes or other languages), built into libpava.so
  test_pava.py   correctness tests (Python, through libpava.so)
  Makefile       make: libpava.so;  make test: runs the tests


Problem: minimize sum_i w_i (y_i - x_i)^2 subject to x_1 <= x_2 <= ... <= x_n,
where y is already sorted by the covariate and all w_i > 0.


FUNCTIONS (pava.hpp)
--------------------
  int64_t pava::isotonic_seq(const double* y, const double* w, int64_t n,
                             double* x_hat, pava::Block* blk, pava::Stats* st = nullptr);
      Sequential PAVA (Algorithm 1 of the paper) followed by the expansion step.

  int64_t pava::isotonic_threephase(const double* y, const double* w, int64_t n,
                                    double* x_hat, pava::Block* blk, int P, pava::Stats* st = nullptr);
      Three-phase parallel PAVA (Algorithm 2 of the paper) with P OpenMP threads:
        Phase 1: PAVA on P chunks of y, in parallel;
        Phase 2: sequential weighted PAVA on the blocks found in Phase 1;
        Phase 3: writing x_hat, in parallel.

  Arguments
    y       input, length n
    w       weights, length n, all > 0; nullptr means unit weights (faster: w is not read)
    n       number of points
    x_hat   output, length n
    blk     work array of n elements of type pava::Block (24 bytes each)
    P       number of threads (clamped to 1 <= P <= n)
    st      optional statistics (pava::Stats): number of blocks after Phase 1 (blocks_phase1, b' in
            the paper), number of blocks of the solution (blocks_final, b), times of the three phases
            and the total time in seconds (t_phase1, t_phase2, t_phase3, t_total)
  Return value: the number b of blocks of the solution (x_hat takes b distinct values).

  The lower-level function pava::Pava<Weighted>(y, w, l, r, L) is Algorithm 1 on the range [l, r).

  Notes
    - The code is 0-based, the paper is 1-based; the comment at the top of pava.hpp explains the mapping.
    - Phase 2 allocates 24 b' bytes in addition to the arrays above.
    - The two algorithms add the same numbers in a different order, so their solutions can differ
      by rounding (in the paper's experiments at most 1.6e-15 relative).
    - Compile with -fopenmp. The paper used -O2 -std=c++17 -fopenmp -march=native -ffp-contract=off.


EXAMPLE
-------
  #include "pava.hpp"
  #include <vector>

  int main() {
      std::vector<double> y = {1, 3, 2, 4, 3, 5};
      std::vector<double> w = {1, 1, 1, 1, 1, 1};
      std::vector<double> x_hat(y.size());
      std::vector<pava::Block> blk(y.size());
      pava::Stats st;

      int64_t b1 = pava::isotonic_seq(y.data(), w.data(), (int64_t)y.size(), x_hat.data(), blk.data());
      int64_t b2 = pava::isotonic_threephase(y.data(), nullptr, (int64_t)y.size(), x_hat.data(),
                                             blk.data(), 4, &st);
      // both: x_hat = 1 2.5 2.5 3.5 3.5 5, 4 blocks
  }

  g++ -O2 -std=c++17 -fopenmp example.cpp


C API (pava_c.cpp)
------------------
  int64_t pava_seq(const double* y, const double* w, int64_t n, double* yhat);
  int64_t pava_omp(const double* y, const double* w, int64_t n, double* yhat, int P, int64_t* blocks_phase1);

  w may be NULL (unit weights), blocks_phase1 may be NULL. Both return the number of blocks and
  allocate the work array themselves. Build: make  (-> libpava.so)


TESTS
-----
  make test          (builds libpava.so and runs python3 test_pava.py)

  Needs numpy; scikit-learn is optional (used as an extra reference if installed).
  For random, binary, tied, weighted and adversarial inputs it checks that
    - the sequential PAVA equals the brute-force min-max formula (small n),
    - the sequential PAVA equals scikit-learn's IsotonicRegression (if installed),
    - the three-phase PAVA equals the sequential PAVA for P = 1, 2, 3, 4, 7, 16, 64, including n < P,
    - the output is non-decreasing and b <= b'.
