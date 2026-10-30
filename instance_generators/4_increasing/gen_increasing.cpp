// gen_increasing.cpp -- generator of the increasing test instances (Section 7, increasing experiment)
//
//   y_i = i,   i = 1..n      (strictly increasing, deterministic)
//   optional weights w_i ~ U[0.5, 2]
//   The data are already isotonic, so PAVA never pools: the solution is y itself (y_hat = y) with
//   B_1 = B_2 = n blocks -- the worst case for the memory traffic of phases 2 and 3.
//   The weights do not change the solution (y_hat = y for any positive weights); they only switch
//   the solver to its weighted code path.
//   y_i = i is exact in double precision for n <= 2^53.
//
// y does not depend on the seed: with unit weights all instances k are identical files, so one
// instance is enough (default --instances 1). The seed (seed + k for instance k) is used only for
// the weights: std::mt19937_64 with std::uniform_real_distribution from <random>.
//
// Writes K instances, one file each:   <dir>/increasing_1_<n>.bin, ..., increasing_K_<n>.bin
//   binary format: y_1, ..., y_n as raw 8-byte doubles (IEEE 754, little-endian), no header; size 8n bytes
//   -- the same format as the Gaussian, binary and noise instances, so the solvers read them unchanged.
//   With --weighted 1 the weights go to a second file of the same format: increasing_k_<n>.wbin
//   Reading in Python: numpy.fromfile("increasing_1_1000000.bin", dtype="<f8")
// Plus <dir>/increasing_<n>_info.txt with all parameters and the seed of every instance.
//
//   make
//   ./gen_increasing --n 1e6                            -> instances/increasing_1_1000000.bin
//   ./gen_increasing --n 1e7 --weighted 1               -> instances/increasing_1_10000000.{bin,wbin}
//   options: --n --instances --seed --weighted 0|1 --dir
//
// File size: 8n bytes (n = 1e8 -> 800 MB per file).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

// write one instance (and its weights); returns false on I/O error
static bool write_instance(const std::string& ypath, const std::string& wpath, int64_t n,
                           uint64_t seed, int weighted) {
    FILE* fy = std::fopen(ypath.c_str(), "wb");
    FILE* fw = weighted ? std::fopen(wpath.c_str(), "wb") : nullptr;
    if (!fy || (weighted && !fw)) { if (fy) std::fclose(fy); if (fw) std::fclose(fw); return false; }
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> unif(0.5, 2.0);

    const int64_t C = 1 << 20;                       // write in pieces of 2^20 values (8 MB)
    std::vector<double> yb(C), wb(weighted ? C : 0);
    bool ok = true;
    for (int64_t lo = 1; lo <= n && ok; lo += C) {
        const int64_t m = std::min(C, n - lo + 1);
        for (int64_t j = 0; j < m; ++j) {
            yb[j] = (double)(lo + j);                // y_i = i, i = lo + j
            if (weighted) wb[j] = unif(gen);
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
    uint64_t seed = 12345; //used only for the weights
    int weighted = 0;  //not weighted
    std::string dir = "instances";
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); return 1; }
        const char* v = argv[++i];
        if      (s == "--n")         n = (int64_t)std::atof(v);
        else if (s == "--instances") K = std::atoi(v);
        else if (s == "--seed")      seed = std::strtoull(v, nullptr, 10);
        else if (s == "--weighted")  weighted = std::atoi(v);
        else if (s == "--dir")       dir = v;
        else { fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
    }
    if (n <= 0 || K <= 0) { fprintf(stderr, "n and --instances must be positive\n"); return 1; }
    if (n > (int64_t(1) << 53)) { fprintf(stderr, "n > 2^53: y_i = i is not exact in double\n"); return 1; }

    const std::string ns = std::to_string((long long)n);
    const std::string info = dir + "/increasing_" + ns + "_info.txt";
    FILE* fi = std::fopen(info.c_str(), "w");
    if (!fi) { fprintf(stderr, "cannot write %s (does the folder '%s' exist?)\n", info.c_str(), dir.c_str()); return 1; }
    fprintf(fi, "model: y_i = i, i = 1..n%s\n", weighted ? ", w_i ~ U[0.5,2]" : "");
    fprintf(fi, "solution: y_hat = y, n blocks (no pooling)\n");
    if (weighted) fprintf(fi, "generator (weights only): std::mt19937_64, std::uniform_real_distribution (libstdc++)\n");
    fprintf(fi, "n = %lld\nweighted = %d\ninstances = %d\n", (long long)n, weighted, K);
    fprintf(fi, "k  file  seed\n");

    for (int k = 1; k <= K; ++k) {
        const uint64_t sk = seed + (uint64_t)k;
        const std::string base = dir + "/increasing_" + std::to_string(k) + "_" + ns;
        const std::string path = base + ".bin";
        if (!write_instance(path, base + ".wbin", n, sk, weighted)) { fprintf(stderr, "write error: %s\n", path.c_str()); return 1; }
        printf("%s  (seed %llu%s)\n", path.c_str(), (unsigned long long)sk, weighted ? "" : ", unused");
        fprintf(fi, "%d  %s  %llu\n", k, path.c_str(), (unsigned long long)sk);
    }
    fprintf(fi, "regenerate: ./gen_increasing --n %g --instances %d --seed %llu --weighted %d\n",
            (double)n, K, (unsigned long long)seed, weighted);
    std::fclose(fi);
    printf("parameters and seeds: %s\n", info.c_str());
    return 0;
}
