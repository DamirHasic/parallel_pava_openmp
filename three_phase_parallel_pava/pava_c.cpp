// pava_c.cpp -- C API for ctypes / other languages.  Build: make libpava.so
#include "pava.hpp"
#include <cstdint>

extern "C" {

// Sequential PAVA.  w may be NULL (unit weights).  Returns the number of blocks.
int64_t pava_seq(const double* y, const double* w, int64_t n, double* yhat)
{
    std::vector<pava::Block> blk(n > 0 ? n : 1);
    return pava::isotonic_seq(y, w, n, yhat, blk.data());
}

// Three-phase OpenMP PAVA (Algorithm 2) with P threads.  Returns the number of blocks;
// *blocks_phase1 (if not NULL) receives B_1, the number of blocks after phase 1.
int64_t pava_omp(const double* y, const double* w, int64_t n, double* yhat, int P,
                 int64_t* blocks_phase1)
{
    std::vector<pava::Block> blk(n > 0 ? n : 1);
    pava::Stats st;
    const int64_t k = pava::isotonic_threephase(y, w, n, yhat, blk.data(), P, &st);
    if (blocks_phase1) *blocks_phase1 = st.blocks_phase1;
    return k;
}

}
