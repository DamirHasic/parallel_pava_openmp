// gen_binary.cpp -- generator of the binary test instances (Section 7, binary experiment)
//
//   y_i ~ Bernoulli(p_i),   p_i = sigma((s/n)(i - (n+1)/2)),   sigma(x) = 1/(1 + e^{-x}),   i = 1..n
//   independent; default s = 8, so p_i runs from about 0.018 (i = 1) to 0.982 (i = n);
//   optional weights w_i ~ U[0.5, 2].
//   Mimics a calibration set sorted by classifier score: y_i is the 0/1 label of the i-th smallest score.
//
// Random numbers: std::mt19937_64 (Mersenne Twister) with std::bernoulli_distribution and
// std::uniform_real_distribution from <random>. Instance k uses the seed (seed + k).
//
// Writes K instances, one file each:   <dir>/binary_1_<n>.bin, binary_2_<n>.bin, ..., binary_K_<n>.bin
//   binary format: y_1, ..., y_n as raw 8-byte doubles (values 0.0 and 1.0; IEEE 754, little-endian), no header;
//   size 8n bytes -- the same format as the Gaussian instances, so the solvers read them unchanged.
//   With --weighted 1 the weights go to a second file of the same format: binary_k_<n>.wbin
//   Reading in Python: numpy.fromfile("binary_1_1000000.bin", dtype="<f8")
// Plus <dir>/binary_<n>_info.txt with all parameters, the seed and the number of ones of every instance.
//
//   make
//   ./gen_binary --n 1e6 --instances 10                 -> instances/binary_1_1000000.bin ... binary_10_1000000.bin
//   ./gen_binary --n 1e7 --instances 10 --slope 4 --weighted 1
//   options: --n --instances --slope (s) --seed --weighted 0|1 --dir
//
// File size: 8n bytes (n = 1e8 -> 800 MB per file).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

// write one instance (and its weights); stores the number of ones in *ones; returns false on I/O error
static bool write_instance(const std::string& ypath, const std::string& wpath, int64_t n, double s,
                           uint64_t seed, int weighted, int64_t* ones) {
    FILE* fy = std::fopen(ypath.c_str(), "wb");
    FILE* fw = weighted ? std::fopen(wpath.c_str(), "wb") : nullptr;
    if (!fy || (weighted && !fw)) { if (fy) std::fclose(fy); if (fw) std::fclose(fw); return false; }
    std::mt19937_64 gen(seed);
    std::bernoulli_distribution bern;
    using bparam = std::bernoulli_distribution::param_type;
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    const double c = 0.5 * ((double)n + 1.0);       // centre (n+1)/2

    const int64_t C = 1 << 20;                       // write in pieces of 2^20 values (8 MB)
    std::vector<double> yb(C), wb(weighted ? C : 0);
    int64_t cnt = 0;
    bool ok = true;
    for (int64_t lo = 1; lo <= n && ok; lo += C) {
        const int64_t m = std::min(C, n - lo + 1);
        for (int64_t j = 0; j < m; ++j) {
            const int64_t i = lo + j;                // 1-based index
            const double x = s * ((double)i - c) / (double)n;
            const double p = 1.0 / (1.0 + std::exp(-x));
            const bool b = bern(gen, bparam(p));
            yb[j] = b ? 1.0 : 0.0;
            cnt += b;
            if (weighted) wb[j] = unif(gen);
        }
        ok = std::fwrite(yb.data(), sizeof(double), (size_t)m, fy) == (size_t)m;
        if (ok && weighted) ok = std::fwrite(wb.data(), sizeof(double), (size_t)m, fw) == (size_t)m;
    }
    ok = (std::fclose(fy) == 0) && ok;
    if (fw) ok = (std::fclose(fw) == 0) && ok;
    *ones = cnt;
    return ok;
}

int main(int argc, char** argv) {
    int64_t n = 100000000; //instance size
    int K = 10;  //number of instances
    double s = 8.0; //slope of the logistic curve - see the manuscript
    uint64_t seed = 12345;
    int weighted = 0;  //not weighted
    std::string dir = "instances";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); return 1; }
        const char* v = argv[++i];
        if      (a == "--n")         n = (int64_t)std::atof(v);
        else if (a == "--instances") K = std::atoi(v);
        else if (a == "--slope")     s = std::atof(v);
        else if (a == "--seed")      seed = std::strtoull(v, nullptr, 10);
        else if (a == "--weighted")  weighted = std::atoi(v);
        else if (a == "--dir")       dir = v;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (n <= 0 || K <= 0) { fprintf(stderr, "n and --instances must be positive\n"); return 1; }

    const std::string ns = std::to_string((long long)n);
    const std::string info = dir + "/binary_" + ns + "_info.txt";
    FILE* fi = std::fopen(info.c_str(), "w");
    if (!fi) { fprintf(stderr, "cannot write %s (does the folder '%s' exist?)\n", info.c_str(), dir.c_str()); return 1; }
    fprintf(fi, "model: y_i ~ Bernoulli(p_i), p_i = 1/(1 + exp(-(s/n)(i - (n+1)/2)))%s\n",
            weighted ? ", w_i ~ U[0.5,2]" : "");
    fprintf(fi, "generator: std::mt19937_64, std::bernoulli_distribution (libstdc++)\n");
    fprintf(fi, "n = %lld\ns = %g\nweighted = %d\ninstances = %d\n", (long long)n, s, weighted, K);
    fprintf(fi, "k  file  seed  ones\n");

    for (int k = 1; k <= K; ++k) {
        const uint64_t sk = seed + (uint64_t)k;
        const std::string base = dir + "/binary_" + std::to_string(k) + "_" + ns;
        const std::string path = base + ".bin";
        int64_t ones = 0;
        if (!write_instance(path, base + ".wbin", n, s, sk, weighted, &ones)) { fprintf(stderr, "write error: %s\n", path.c_str()); return 1; }
        printf("%s  (seed %llu, ones %lld)\n", path.c_str(), (unsigned long long)sk, (long long)ones);
        fprintf(fi, "%d  %s  %llu  %lld\n", k, path.c_str(), (unsigned long long)sk, (long long)ones);
    }
    fprintf(fi, "regenerate: ./gen_binary --n %g --instances %d --slope %g --seed %llu --weighted %d\n",
            (double)n, K, s, (unsigned long long)seed, weighted);
    std::fclose(fi);
    printf("parameters and seeds: %s\n", info.c_str());
    return 0;
}
