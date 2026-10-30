// compare.cpp -- compares the sequential and the parallel solutions yhat
//
//   make compare
//   ./compare                                   all pairs in results/yhat: <inst>.seq.yhat.bin vs <inst>.par<P>.yhat.bin
//   ./compare A.seq.yhat.bin B.par12.yhat.bin   one pair
//   options: --dir PATH (default results/yhat)   --out FILE (default results/compare.csv, overwritten)
//
// For each pair (a = sequential, b = parallel), n values each:
//   n_diff     number of i with a_i != b_i (bitwise)
//   max_abs    max_i |a_i - b_i|
//   rel        max_abs / max_i |a_i|
//   rms        sqrt( (1/n) sum_i (a_i - b_i)^2 )
//   blocks_seq, blocks_par   number of blocks (1 + number of i with x_i != x_{i-1})
//   monotone   1 if b_1 <= b_2 <= ... <= b_n
// Files: raw 8-byte doubles (as written by ./run).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool read_bin(const fs::path& p, std::vector<double>& x) {
    std::error_code ec;
    const uintmax_t bytes = fs::file_size(p, ec);
    if (ec || bytes % sizeof(double) != 0) return false;
    x.resize(bytes / sizeof(double));
    FILE* f = std::fopen(p.string().c_str(), "rb"); if (!f) return false;
    const bool ok = std::fread(x.data(), sizeof(double), x.size(), f) == x.size();
    std::fclose(f);
    return ok;
}

static long long count_blocks(const std::vector<double>& x) {
    if (x.empty()) return 0;
    long long b = 1;
    for (size_t i = 1; i < x.size(); ++i) b += (x[i] != x[i - 1]);
    return b;
}

// compares one pair, prints and writes one CSV line; returns false on error
static bool compare_pair(const fs::path& pa, const fs::path& pb, FILE* fo, std::vector<double>& a, std::vector<double>& b) {
    if (!read_bin(pa, a) || !read_bin(pb, b)) { fprintf(stderr, "cannot read %s or %s\n", pa.string().c_str(), pb.string().c_str()); return false; }
    if (a.size() != b.size()) { fprintf(stderr, "different lengths: %s (%zu) vs %s (%zu)\n", pa.filename().string().c_str(), a.size(), pb.filename().string().c_str(), b.size()); return false; }
    const size_t n = a.size();
    long long n_diff = 0; double max_abs = 0, max_a = 0, sum2 = 0; bool monotone = true;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (a[i] != b[i]) ++n_diff;
        max_abs = std::max(max_abs, d);
        max_a = std::max(max_a, std::fabs(a[i]));
        sum2 += d * d;
        if (i > 0 && b[i] < b[i - 1]) monotone = false;
    }
    const double rel = max_a > 0 ? max_abs / max_a : max_abs;
    const double rms = n ? std::sqrt(sum2 / n) : 0;
    const long long bs = count_blocks(a), bp = count_blocks(b);
    printf("%-34s vs %-36s n=%zu  n_diff=%lld  max_abs=%.3e  rel=%.3e  rms=%.3e  blocks=%lld/%lld  monotone=%d\n",
           pa.filename().string().c_str(), pb.filename().string().c_str(), n, n_diff, max_abs, rel, rms, bs, bp, monotone ? 1 : 0);
    fprintf(fo, "%s,%s,%zu,%lld,%.6e,%.6e,%.6e,%lld,%lld,%d\n", pa.filename().string().c_str(), pb.filename().string().c_str(),
            n, n_diff, max_abs, rel, rms, bs, bp, monotone ? 1 : 0);
    std::fflush(fo);
    return true;
}

int main(int argc, char** argv) {
    std::string dir = "results/yhat", out = "results/compare.csv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if ((s == "--dir" || s == "--out") && i + 1 < argc) { (s == "--dir" ? dir : out) = argv[++i]; }
        else files.push_back(s);
    }
    if (!files.empty() && files.size() != 2) { fprintf(stderr, "give two files, or none\n"); return 1; }

    FILE* fo = std::fopen(out.c_str(), "w");
    if (!fo) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    fprintf(fo, "seq_file,par_file,n,n_diff,max_abs,rel,rms,blocks_seq,blocks_par,monotone\n");

    std::vector<double> a, b;
    int pairs = 0;
    if (files.size() == 2) {
        pairs += compare_pair(files[0], files[1], fo, a, b);
    } else {
        // all <inst>.seq.yhat.bin, each with all matching <inst>.par<P>.yhat.bin
        std::vector<fs::path> all;
        for (const auto& e : fs::directory_iterator(dir)) if (e.is_regular_file()) all.push_back(e.path());
        std::sort(all.begin(), all.end());
        const std::string seq_tag = ".seq.yhat.bin";
        for (const fs::path& p : all) {
            const std::string name = p.filename().string();
            if (name.size() <= seq_tag.size() || name.compare(name.size() - seq_tag.size(), seq_tag.size(), seq_tag) != 0) continue;
            const std::string inst = name.substr(0, name.size() - seq_tag.size());
            bool found = false;
            for (const fs::path& q : all) {
                const std::string qn = q.filename().string();
                if (qn.rfind(inst + ".par", 0) == 0 && qn.size() > 9 && qn.compare(qn.size() - 9, 9, ".yhat.bin") == 0) {
                    pairs += compare_pair(p, q, fo, a, b); found = true;
                }
            }
            if (!found) fprintf(stderr, "no parallel result for %s\n", name.c_str());
        }
    }
    std::fclose(fo);
    printf("%d pairs compared, results in %s\n", pairs, out.c_str());
    return pairs > 0 ? 0 : 1;
}
