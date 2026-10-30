// gen_adversarial.cpp -- generator of the adversarial test instances (Section 7, worst case of prop:worst)
//
//   y_i = i  (i = 1..n-1),   y_n = -c n^2      (default c = 1: the family y = (1, 2, ..., n-1, -n^2) of prop:worst)
//   optional weights w_i ~ U[0.5, 2]
//   Unit weights: the solution is one block, y_hat = (n(n-1)/2 - c n^2)/n (= -(n+1)/2 for c = 1).
//   In Phase 1 the first P-1 chunks are strictly increasing (one block per point) and the last chunk pools into
//   one block, so b' = c_{P-1} + 1 >= n/2 for P >= 2; Phase 2 then pools b' - 1 times.
//
// Weights.  prop:worst is stated for unit weights.  With weights the solution is one block if
//   sum_{i<n} w_i i < c w_n n^2                                                          (*)
// (then every block {j..n} has a negative mean, below y_{j-1} = j-1 >= 1).  With w_i in [0.5, 2], (*) holds for
// every draw when c >= 2 (0.5 * 2 n^2 > 2 * n(n-1)/2), but NOT always for c = 1 (it fails when w_n is below
// about 0.6).  The generator checks (*) for every instance and writes the result to the info file;
// use --c 2 for weighted instances.
//
// y does not depend on the seed: with unit weights all instances k are identical files (default --instances 1).
// The seed (seed + k for instance k) is used only for the weights: std::mt19937_64, std::uniform_real_distribution.
// y_i = i and -c n^2 are exact in double for n <= 9.4e7 (c = 1); for n = 1e8, n^2 = 1e16 is still exact.
//
// Writes K instances, one file each:   <dir>/adversarial_1_<n>.bin, ..., adversarial_K_<n>.bin
//   binary format: y_1, ..., y_n as raw 8-byte doubles (IEEE 754, little-endian), no header; size 8n bytes
//   -- the same format as the other instance families, so the solvers read them unchanged.
//   With --weighted 1 the weights go to a second file of the same format: adversarial_k_<n>.wbin
//   Reading in Python: numpy.fromfile("adversarial_1_1000000.bin", dtype="<f8")
// Plus <dir>/adversarial_<n>_info.txt with all parameters, the seed of every instance and the check (*).
//
//   make
//   ./gen_adversarial --n 1e6                               -> instances/adversarial_1_1000000.bin
//   ./gen_adversarial --n 1e7 --weighted 1 --c 2 --instances 10
//   options: --n --instances --c --seed --weighted 0|1 --dir
//
// File size: 8n bytes (n = 1e8 -> 800 MB per file).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

// writes one instance (and its weights); returns false on I/O error.
// sum_wi = sum_{i<n} w_i i and w_last = w_n (for the check (*)); unit weights: w = 1.
static bool write_instance(const std::string& ypath, const std::string& wpath, int64_t n, double c,
                           uint64_t seed, int weighted, long double& sum_wi, double& w_last) {
    FILE* fy = std::fopen(ypath.c_str(), "wb");
    FILE* fw = weighted ? std::fopen(wpath.c_str(), "wb") : nullptr;
    if (!fy || (weighted && !fw)) { if (fy) std::fclose(fy); if (fw) std::fclose(fw); return false; }
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    const double nd = (double)n;

    const int64_t C = 1 << 20;                       // write in pieces of 2^20 values (8 MB)
    std::vector<double> yb(C), wb(weighted ? C : 0);
    bool ok = true;
    sum_wi = 0; w_last = 1.0;
    for (int64_t lo = 1; lo <= n && ok; lo += C) {
        const int64_t m = std::min(C, n - lo + 1);
        for (int64_t j = 0; j < m; ++j) {
            const int64_t i = lo + j;                // 1-based index
            yb[j] = (i < n) ? (double)i : -c * nd * nd;
            const double wi = weighted ? unif(gen) : 1.0;
            if (weighted) wb[j] = wi;
            if (i < n) sum_wi += (long double)wi * (long double)i; else w_last = wi;
        }
        ok = std::fwrite(yb.data(), sizeof(double), (size_t)m, fy) == (size_t)m;
        if (ok && weighted) ok = std::fwrite(wb.data(), sizeof(double), (size_t)m, fw) == (size_t)m;
    }
    ok = (std::fclose(fy) == 0) && ok;
    if (fw) ok = (std::fclose(fw) == 0) && ok;
    return ok;
}

int main(int argc, char** argv) {
    int64_t n = 100000000; //instance size
    int K = 1;  //number of instances (unit weights: all identical, so 1 is enough)
    double c = 1.0;  //y_n = -c n^2 (c = 1: prop:worst)
    uint64_t seed = 12345; //used only for the weights
    int weighted = 0;  //not weighted
    std::string dir = "instances";
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); return 1; }
        const char* v = argv[++i];
        if      (s == "--n")         n = (int64_t)std::atof(v);
        else if (s == "--instances") K = std::atoi(v);
        else if (s == "--c")         c = std::atof(v);
        else if (s == "--seed")      seed = std::strtoull(v, nullptr, 10);
        else if (s == "--weighted")  weighted = std::atoi(v);
        else if (s == "--dir")       dir = v;
        else { fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
    }
    if (n < 2 || K <= 0 || c <= 0) { fprintf(stderr, "need n >= 2, --instances > 0, --c > 0\n"); return 1; }
    if (n > (int64_t(1) << 53)) { fprintf(stderr, "n > 2^53: y_i = i is not exact in double\n"); return 1; }

    const std::string ns = std::to_string((long long)n);
    const std::string info = dir + "/adversarial_" + ns + "_info.txt";
    FILE* fi = std::fopen(info.c_str(), "w");
    if (!fi) { fprintf(stderr, "cannot write %s (does the folder '%s' exist?)\n", info.c_str(), dir.c_str()); return 1; }
    const long double nl = (long double)n;
    fprintf(fi, "model: y_i = i (i = 1..n-1), y_n = -c n^2%s\n", weighted ? ", w_i ~ U[0.5,2]" : "");
    if (!weighted)
        fprintf(fi, "solution: one block, y_hat = (n(n-1)/2 - c n^2)/n = %.6Lf\n", (nl * (nl - 1) / 2 - (long double)c * nl * nl) / nl);
    fprintf(fi, "Phase 1 (P >= 2, chunks c_p = floor(pn/P)): b' = c_{P-1} + 1 >= n/2\n");
    if (weighted) fprintf(fi, "generator (weights only): std::mt19937_64, std::uniform_real_distribution (libstdc++)\n");
    fprintf(fi, "n = %lld\nc = %g\nweighted = %d\ninstances = %d\n", (long long)n, c, weighted, K);
    fprintf(fi, "k  file  seed  w_n  one_block(*)\n");

    int bad = 0;
    for (int k = 1; k <= K; ++k) {
        const uint64_t sk = seed + (uint64_t)k;
        const std::string base = dir + "/adversarial_" + std::to_string(k) + "_" + ns;
        const std::string path = base + ".bin";
        long double sum_wi = 0; double w_last = 1.0;
        if (!write_instance(path, base + ".wbin", n, c, sk, weighted, sum_wi, w_last)) { fprintf(stderr, "write error: %s\n", path.c_str()); return 1; }
        const bool one = sum_wi < (long double)c * (long double)w_last * nl * nl;     // condition (*)
        bad += !one;
        printf("%s  (seed %llu%s)  one block: %s\n", path.c_str(), (unsigned long long)sk, weighted ? "" : ", unused", one ? "yes" : "NO");
        fprintf(fi, "%d  %s  %llu  %.6f  %s\n", k, path.c_str(), (unsigned long long)sk, w_last, one ? "yes" : "NO");
    }
    fprintf(fi, "regenerate: ./gen_adversarial --n %g --instances %d --c %g --seed %llu --weighted %d\n",
            (double)n, K, c, (unsigned long long)seed, weighted);
    std::fclose(fi);
    printf("parameters and seeds: %s\n", info.c_str());
    if (bad) fprintf(stderr, "WARNING: %d instance(s) do not satisfy (*), the solution may have more than one block; use --c 2\n", bad);
    return 0;
}
