// gen_gauss.cpp -- generator of the Gaussian test instances (Section 7, Experiment 1)
//
//   y_i = a (t_i - 1/2) + sigma * eps_i,   t_i = (i - 1/2)/n,   eps_i ~ N(0,1) i.i.d.,   i = 1..n
//   optional weights w_i ~ U[0.5, 2]
//
// Random numbers: std::mt19937_64 (Mersenne Twister) with std::normal_distribution and
// std::uniform_real_distribution from <random>. Instance k uses the seed (seed + k).
//
// Writes K instances, one file each:   <dir>/gauss_1_<n>.bin, gauss_2_<n>.bin, ..., gauss_K_<n>.bin
//   binary format: y_1, ..., y_n as raw 8-byte doubles (IEEE 754, little-endian), no header; size 8n bytes.
//   With --weighted 1 the weights go to a second file of the same format: gauss_k_<n>.wbin
//   Reading in Python: numpy.fromfile("gauss_1_1000000.bin", dtype="<f8")
// Plus <dir>/gauss_<n>_info.txt with all parameters and the seed of every instance.
//
//   make
//   ./gen_gauss --n 1e6 --instances 10                  -> instances/gauss_1_1000000.bin ... gauss_10_1000000.bin
//   ./gen_gauss --n 1e7 --instances 10 --sigma 0.5 --weighted 1
//   options: --n --instances --slope (a) --sigma --seed --weighted 0|1 --dir
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
static bool write_instance(const std::string& ypath, const std::string& wpath, int64_t n, double a, double sigma,
                           uint64_t seed, int weighted) {
    FILE* fy = std::fopen(ypath.c_str(), "wb");
    FILE* fw = weighted ? std::fopen(wpath.c_str(), "wb") : nullptr;
    if (!fy || (weighted && !fw)) { if (fy) std::fclose(fy); if (fw) std::fclose(fw); return false; }
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.5, 2.0);

    const int64_t C = 1 << 20;                       // write in pieces of 2^20 values (8 MB)
    std::vector<double> yb(C), wb(weighted ? C : 0);
    bool ok = true;
    for (int64_t lo = 1; lo <= n && ok; lo += C) {
        const int64_t m = std::min(C, n - lo + 1);
        for (int64_t j = 0; j < m; ++j) {
            const int64_t i = lo + j;                // 1-based index
            const double t = (i - 0.5) / (double)n;
            yb[j] = a * (t - 0.5) + sigma * normal(gen);
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
    int K = 10;  //number of instances
    double a = 4.0, sigma = 1.0; //distribution parameters - see the manuscript
    uint64_t seed = 12345;
    int weighted = 0;  //not weighted
    std::string dir = "instances";
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); return 1; }
        const char* v = argv[++i];
        if      (s == "--n")         n = (int64_t)std::atof(v);
        else if (s == "--instances") K = std::atoi(v);
        else if (s == "--slope")     a = std::atof(v);
        else if (s == "--sigma")     sigma = std::atof(v);
        else if (s == "--seed")      seed = std::strtoull(v, nullptr, 10);
        else if (s == "--weighted")  weighted = std::atoi(v);
        else if (s == "--dir")       dir = v;
        else { fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
    }
    if (n <= 0 || K <= 0) { fprintf(stderr, "n and --instances must be positive\n"); return 1; }

    const std::string ns = std::to_string((long long)n);
    const std::string info = dir + "/gauss_" + ns + "_info.txt";
    FILE* fi = std::fopen(info.c_str(), "w");
    if (!fi) { fprintf(stderr, "cannot write %s (does the folder '%s' exist?)\n", info.c_str(), dir.c_str()); return 1; }
    fprintf(fi, "model: y_i = a (t_i - 1/2) + sigma eps_i, t_i = (i - 1/2)/n, eps_i ~ N(0,1)%s\n",
            weighted ? ", w_i ~ U[0.5,2]" : "");
    fprintf(fi, "generator: std::mt19937_64, std::normal_distribution (libstdc++)\n");
    fprintf(fi, "n = %lld\na = %g\nsigma = %g\nweighted = %d\ninstances = %d\n", (long long)n, a, sigma, weighted, K);
    fprintf(fi, "k  file  seed\n");

    for (int k = 1; k <= K; ++k) {
        const uint64_t sk = seed + (uint64_t)k;
        const std::string base = dir + "/gauss_" + std::to_string(k) + "_" + ns;
        const std::string path = base + ".bin";
        if (!write_instance(path, base + ".wbin", n, a, sigma, sk, weighted)) { fprintf(stderr, "write error: %s\n", path.c_str()); return 1; }
        printf("%s  (seed %llu)\n", path.c_str(), (unsigned long long)sk);
        fprintf(fi, "%d  %s  %llu\n", k, path.c_str(), (unsigned long long)sk);
    }
    std::fclose(fi);
    printf("parameters and seeds: %s\n", info.c_str());
    return 0;
}
