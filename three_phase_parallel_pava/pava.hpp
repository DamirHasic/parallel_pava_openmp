// pava.hpp -- weighted L2 isotonic regression (total order, sorted input)
//
// The code follows the manuscript, with the same names:
//   Pava(y, w, l, r, L)    Algorithm 1  (up-and-down-blocks PAVA with block sums and look-ahead)
//   isotonic_seq           Algorithm 1 on the whole range + the expansion step
//   isotonic_threephase    Algorithm 2  (three-phase parallel PAVA, OpenMP)
//
// Indexing.  The manuscript is 1-based, the code is 0-based: index i of the paper is i-1 here.
//   * The paper's range l..r (inclusive) is the half-open range [l-1, r) here, so the paper's
//     Pava(y, w, c_{p-1}+1, c_p) is Pava(y, w, c[p-1], c[p], .) and Pava(y', w', 1, B_1) is
//     Pava(y', w', 0, B1, .).
//   * The last index e of a block (1-based, inclusive) is the same number as the exclusive end
//     of the block in 0-based indexing, so e is stored unchanged.
//   * The stack position b of the paper (1-based) is L[b-1] here.
//
// Block (s, omega, e): weighted sum s = sum w_i y_i, weight omega = sum w_i, last index e.
// Value s/omega.  The pooling test v_b >= v is evaluated without division as
// s_b * omega >= s * omega_b (all weights > 0).
//
// Header-only, C++17.  Compile with -O2 -fopenmp.

#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace pava {

struct Block {
    double  s;       // weighted sum
    double  omega;   // weight
    int64_t e;       // last index (1-based) = exclusive end (0-based)
};

struct Stats {
    int     threads       = 1;   // P
    int64_t blocks_phase1 = 0;   // B_1 (== b for the sequential code)
    int64_t blocks_final  = 0;   // b: blocks of the solution
    double  t_phase1      = 0;   // seconds (three-phase only)
    double  t_phase2      = 0;
    double  t_phase3      = 0;
    double  t_total       = 0;
};

// ---------------------------------------------------------------------------
// Algorithm 1: Pava(y, w, l, r).  Blocks are written to L[0 .. b), b is returned.
// Weighted == false: w is ignored (unit weights), which saves memory traffic.
// ---------------------------------------------------------------------------
template <bool Weighted>
inline int64_t Pava(const double* __restrict y ,
                    const double* __restrict w ,
                    int64_t l ,
                    int64_t r ,
                    Block* __restrict L
                    )
{
    int64_t b = 0;
    int64_t i = l;
    while (i < r) {
        double s, omega;                                        // new block containing one point
        if constexpr (Weighted) { s = w[i] * y[i]; omega = w[i]; }
        else                    { s = y[i];        omega = 1.0;  }
        ++i;
        if (b > 0 && L[b - 1].s * omega >= s * L[b - 1].omega) {   // v_b >= v: down violation
            s += L[b - 1].s; omega += L[b - 1].omega;              // pool the top block into the new one
            --b;
            while (i < r && s >= y[i] * omega) {                   // look-ahead: v >= y_i
                if constexpr (Weighted) { s += w[i] * y[i]; omega += w[i]; }
                else                    { s += y[i];        omega += 1.0;  }
                ++i;
            }
            while (b > 0 && L[b - 1].s * omega >= s * L[b - 1].omega) {   // repair the down violations
                s += L[b - 1].s; omega += L[b - 1].omega;
                --b;
            }
        }
        ++b;
        L[b - 1] = Block{s, omega, i};                          // push; e_b = i-1 (1-based) = i (0-based end)
    }
    return b;
}

// ---------------------------------------------------------------------------
// Sequential PAVA: Algorithm 1 on 1..n, then the expansion step
// x_hat_i = s_k/omega_k for e_{k-1} < i <= e_k (e_0 = 0).
// L must have room for n blocks.  Returns the number of blocks b.
// ---------------------------------------------------------------------------
template <bool Weighted>
inline int64_t isotonic_seq(const double* y ,
                            const double* w ,
                            int64_t n ,
                            double* x_hat ,
                            Block* L ,
                            Stats* st = nullptr
                            )
{
    const double t0 = omp_get_wtime();
    const int64_t b = Pava<Weighted>(y, w, 0, n, L);
    int64_t e_prev = 0;                                         // e_0 = 0
    for (int64_t k = 0; k < b; ++k) {
        const double v = L[k].s / L[k].omega;
        for (int64_t i = e_prev; i < L[k].e; ++i) x_hat[i] = v;
        e_prev = L[k].e;
    }
    if (st) { st->threads = 1; st->blocks_phase1 = b; st->blocks_final = b;
              st->t_total = omp_get_wtime() - t0; }
    return b;
}

// ---------------------------------------------------------------------------
// Algorithm 2: three-phase parallel PAVA.
// blk is the shared block array of length n.  P is clamped to 1 <= P <= n.
// Returns the number of blocks b of the solution.
// ---------------------------------------------------------------------------
template <bool Weighted>
inline int64_t isotonic_threephase(const double* y, const double* w, int64_t n, double* x_hat,
                                   Block* blk, int P, Stats* st = nullptr)
{
    if (n <= 0) { if (st) *st = Stats{}; return 0; }
    if (P < 1) P = 1;
    if ((int64_t)P > n) P = (int)n;

    const double t0 = omp_get_wtime();

    // ---- Phase 1 ----
    std::vector<int64_t> c(P + 1);
    for (int p = 0; p <= P; ++p) c[p] = (int64_t)p * n / P;    // c_p = floor(pn/P)
    std::vector<int64_t> b_p(P + 1, 0);                         // b_p: number of blocks in L_p

    #pragma omp parallel for num_threads(P) schedule(static)
    for (int p = 1; p <= P; ++p)
        b_p[p] = Pava<Weighted>(y, w, c[p - 1], c[p], blk + c[p - 1]);   // L_p = blk[c_{p-1} ..)

    const double t1 = omp_get_wtime();

    // ---- Phase 2 ----
    int64_t B1 = 0;                                             // the total number of blocks
    for (int p = 1; p <= P; ++p) B1 += b_p[p];

    // concatenate L_1, ..., L_P:  (y'_k, w'_k, e'_k) = (s_j^p / omega_j^p, omega_j^p, e_j^p)
    std::unique_ptr<double[]>  y_prime(new double[B1]);
    std::unique_ptr<double[]>  w_prime(new double[B1]);
    std::unique_ptr<int64_t[]> e_prime(new int64_t[B1]);
    int64_t k = 0;
    for (int p = 1; p <= P; ++p) {
        const Block* L_p = blk + c[p - 1];
        for (int64_t j = 0; j < b_p[p]; ++j, ++k) {
            y_prime[k] = L_p[j].s / L_p[j].omega;
            w_prime[k] = L_p[j].omega;
            e_prime[k] = L_p[j].e;
        }
    }

    Block* L = blk;                                             // L overwrites L_1, ..., L_P (already copied)
    const int64_t b = Pava<true>(y_prime.get(), w_prime.get(), 0, B1, L);

    std::unique_ptr<int64_t[]> r(new int64_t[b]);
    for (int64_t t = 0; t < b; ++t)
        r[t] = e_prime[L[t].e - 1];                             // r_t = e'_{e_t}: last index of phase-1 block e_t

    const double t2 = omp_get_wtime();

    // ---- Phase 3 ----
    #pragma omp parallel for num_threads(P) schedule(static)
    for (int p = 1; p <= P; ++p) {
        // t = min{t : r_t >= c_{p-1}+1}  (binary search; 0-based: first t with r[t] > c[p-1])
        int64_t t = std::upper_bound(r.get(), r.get() + b, c[p - 1]) - r.get();
        double v = L[t].s / L[t].omega;                         // s_t/omega_t, recomputed only when t changes
        for (int64_t i = c[p - 1]; i < c[p]; ++i) {
            if (i >= r[t]) {                                    // paper: i > r_t (1-based)
                ++t;
                v = L[t].s / L[t].omega;
            }
            x_hat[i] = v;
        }
    }

    const double t3 = omp_get_wtime();
    if (st) {
        st->threads = P; st->blocks_phase1 = B1; st->blocks_final = b;
        st->t_phase1 = t1 - t0; st->t_phase2 = t2 - t1; st->t_phase3 = t3 - t2; st->t_total = t3 - t0;
    }
    return b;
}

// Convenience wrappers dispatching on w == nullptr (unit weights).
inline int64_t isotonic_seq(const double* y, const double* w, int64_t n, double* x_hat,
                            Block* L, Stats* st = nullptr)
{
    return w ? isotonic_seq<true>(y, w, n, x_hat, L, st) : isotonic_seq<false>(y, nullptr, n, x_hat, L, st);
}
inline int64_t isotonic_threephase(const double* y, const double* w, int64_t n, double* x_hat,
                                   Block* blk, int P, Stats* st = nullptr)
{
    return w ? isotonic_threephase<true>(y, w, n, x_hat, blk, P, st)
             : isotonic_threephase<false>(y, nullptr, n, x_hat, blk, P, st);
}

} // namespace pava
