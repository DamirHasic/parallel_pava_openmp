// main.cpp -- runs sequential or parallel (three-phase) PAVA on all instances in a folder
//
//   make
//   ./run --algo seq --dir ../instance_generators/1_gauss/instances
//   ./run --algo par --threads 1,2,4,8 --dir ../instance_generators/1_gauss/instances
//
// For every instance file (*.bin), in the order type, n, k:
//   read it; for every number of threads P: run the algorithm --reps times and append one line to the
//   result file (--out, CSV); then go to the next file.
//   The solution yhat is written only once per instance, to results/yhat/<instance name>.<algo>.yhat.bin
//   (<algo> = seq, or par<P> for the largest P, e.g. gauss_1_1000000.par8.yhat.bin)
// The result file is flushed after every line, so an interrupted run keeps its results.
//
// Files are binary: raw 8-byte doubles (IEEE 754, little-endian), no header, size 8n bytes.
// Instance: <type>_<k>_<n>.bin (y), e.g. gauss_3_1000000.bin; weights, if any, in <type>_<k>_<n>.wbin.
// Reading in Python: numpy.fromfile(name, dtype="<f8")
//
// Options:
//   --algo seq|par        sequential PAVA (Algorithm 1) or three-phase parallel PAVA (Algorithm 2)
//   --threads 1,2,4,8     numbers of threads for --algo par (default: 1,2,4,... up to the number of logical cores,
//                         e.g. 1,2,4,8,12 on a 12-core machine)
//   --dir PATH            instance folder              (default ../instance_generators/1_gauss/instances)
//   --out FILE            result file, appended to     (default results/results.csv)
//   --reps R              timed runs per instance      (default 5)
//
// Result columns:
//   date, instance, type, n, k, weighted, algo, threads, reps,
//   t_min, t_median, t_mean, t_sd          (seconds, over the R timed runs)
//   t_phase1, t_phase2, t_phase3           (seconds, run with the median time; par only)
//   blocks_phase1, blocks_final            (B_1 and number of blocks of the solution)
//   ns_per_point                           (t_median / n, in ns)
//   sse                                    (objective sum w_i (y_i - yhat_i)^2)
//   t_read                                 (seconds to read the file; not part of the timings)

#include "../three_phase_parallel_pava/pava.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using pava::Block; using pava::Stats;

struct Instance { fs::path path; std::string type; int64_t n = 0; int k = 0; };

// name <type>_<k>_<n>.bin
static bool parse_name(const fs::path& p, Instance& in) {
    const std::string s = p.stem().string();
    const size_t b = s.rfind('_'); if (b == std::string::npos || b == 0) return false;
    const size_t a = s.rfind('_', b - 1); if (a == std::string::npos) return false;
    in.path = p; in.type = s.substr(0, a);
    in.k = std::atoi(s.substr(a + 1, b - a - 1).c_str());
    in.n = std::atoll(s.substr(b + 1).c_str());
    return in.k > 0 && in.n > 0;
}

// reads a binary file of doubles (size 8n bytes)
static bool read_bin(const fs::path& p, std::vector<double>& x) {
    std::error_code ec;
    const uintmax_t bytes = fs::file_size(p, ec);
    if (ec || bytes % sizeof(double) != 0 || bytes == 0) return false;
    x.resize(bytes / sizeof(double));
    FILE* f = std::fopen(p.string().c_str(), "rb"); if (!f) return false;
    const bool ok = std::fread(x.data(), sizeof(double), x.size(), f) == x.size();
    std::fclose(f);
    return ok;
}

// writes x as raw doubles (same format as the instances)
static bool write_bin(const fs::path& p, const std::vector<double>& x) {
    FILE* f = std::fopen(p.string().c_str(), "wb"); if (!f) return false;
    const bool ok = std::fwrite(x.data(), sizeof(double), x.size(), f) == x.size();
    return (std::fclose(f) == 0) && ok;
}

int main(int argc, char** argv) {
    std::string algo = "par", dir = "../instance_generators/2_binary/instances", out = "results/results_2_binary.csv";
    // list of thread numbers (par only); default 1, 2, 4, ..., up to the number of logical cores, plus that number
    std::vector<int> threads;
    for (int P = 1; P < omp_get_num_procs(); P *= 2) threads.push_back(P);
    threads.push_back(omp_get_num_procs());
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); return 1; }
        const std::string v = argv[++i];
        if      (s == "--algo")    algo = v;
        else if (s == "--threads") {
            threads.clear();
            size_t pos = 0;
            while (pos <= v.size()) {
                size_t c = v.find(',', pos); if (c == std::string::npos) c = v.size();
                const int P = std::atoi(v.substr(pos, c - pos).c_str());
                if (P > 0) threads.push_back(P);
                pos = c + 1;
            }
        }
        else if (s == "--dir")     dir = v;
        else if (s == "--out")     out = v;
        else if (s == "--reps")    reps = std::atoi(v.c_str());
        else { fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
    }
    if (algo != "seq" && algo != "par") { fprintf(stderr, "--algo must be seq or par\n"); return 1; }
    if (algo == "seq" || threads.empty()) threads = {1};
    const int P_save = *std::max_element(threads.begin(), threads.end());   // yhat is saved for this P
    if (reps < 1) reps = 1;

    // collect and sort the instances
    std::vector<Instance> inst;
    if (!fs::is_directory(dir)) { fprintf(stderr, "no folder %s\n", dir.c_str()); return 1; }
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".bin") continue;
        Instance in; if (parse_name(e.path(), in)) inst.push_back(in);
        else fprintf(stderr, "skipped (name is not <type>_<k>_<n>.bin): %s\n", e.path().filename().string().c_str());
    }
    std::sort(inst.begin(), inst.end(), [](const Instance& a, const Instance& b) {
        if (a.type != b.type) return a.type < b.type;
        if (a.n != b.n) return a.n < b.n;
        return a.k < b.k; });
    if (inst.empty()) { fprintf(stderr, "no instances in %s\n", dir.c_str()); return 1; }
    printf("%zu instances in %s, algo=%s, reps=%d\n", inst.size(), dir.c_str(), algo.c_str(), reps);

    // result file: header only if new or empty
    if (fs::path(out).has_parent_path()) fs::create_directories(fs::path(out).parent_path());
    const bool fresh = !fs::exists(out) || fs::file_size(out) == 0;
    FILE* fr = std::fopen(out.c_str(), "a");
    if (!fr) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    if (fresh) fprintf(fr, "date,instance,type,n,k,weighted,algo,threads,reps,t_min,t_median,t_mean,t_sd,"
                           "t_phase1,t_phase2,t_phase3,blocks_phase1,blocks_final,ns_per_point,sse,t_read\n");

    char date[32]; { std::time_t t = std::time(nullptr); std::strftime(date, sizeof date, "%Y-%m-%d %H:%M", std::localtime(&t)); }

    const fs::path ydir = fs::path(out).parent_path() / "yhat";
    fs::create_directories(ydir);
    std::vector<double> y, w, yhat;
    std::vector<Block> blk;
    for (const Instance& in : inst) {
        const double t0 = omp_get_wtime();
        fs::path wpath = in.path; wpath.replace_extension(".wbin");
        if (!read_bin(in.path, y) || (fs::exists(wpath) && (!read_bin(wpath, w) || w.size() != y.size()))) { fprintf(stderr, "cannot read %s\n", in.path.string().c_str()); continue; }
        const double t_read = omp_get_wtime() - t0;
        const int64_t n = (int64_t)y.size();
        if (n != in.n) fprintf(stderr, "warning: %s has %lld values, name says %lld\n", in.path.filename().string().c_str(), (long long)n, (long long)in.n);
        if (!fs::exists(wpath)) w.clear();
        const double* wp = w.empty() ? nullptr : w.data();

        yhat.assign(n, 0.0); blk.assign(n, Block{0, 0, 0});

        for (int P : threads) {
            auto run = [&](Stats& st) {
                if (algo == "seq") pava::isotonic_seq(y.data(), wp, n, yhat.data(), blk.data(), &st);
                else               pava::isotonic_threephase(y.data(), wp, n, yhat.data(), blk.data(), P, &st);
            };
            std::vector<Stats> sts(reps);
            for (int r = 0; r < reps; ++r) run(sts[r]);

            std::vector<double> ts(reps); for (int r = 0; r < reps; ++r) ts[r] = sts[r].t_total;
            std::vector<double> sorted = ts; std::sort(sorted.begin(), sorted.end());
            const double tmed = sorted[reps / 2], tmin = sorted[0];
            double mean = 0; for (double t : ts) mean += t; mean /= reps;
            double sd = 0; for (double t : ts) sd += (t - mean) * (t - mean); sd = reps > 1 ? std::sqrt(sd / (reps - 1)) : 0;
            size_t im = 0; for (int r = 0; r < reps; ++r) if (ts[r] == tmed) im = r;
            const Stats& st = sts[im];

            double sse = 0;
            for (int64_t i = 0; i < n; ++i) { const double d = y[i] - yhat[i]; sse += (wp ? wp[i] : 1.0) * d * d; }
            const bool par = (algo == "par");
            fprintf(fr, "%s,%s,%s,%lld,%d,%d,%s,%d,%d,%.6e,%.6e,%.6e,%.3e,", date, in.path.filename().string().c_str(), in.type.c_str(),
                    (long long)n, in.k, wp ? 1 : 0, algo.c_str(), par ? P : 1, reps, tmin, tmed, mean, sd);
            if (par) fprintf(fr, "%.6e,%.6e,%.6e,", st.t_phase1, st.t_phase2, st.t_phase3); else fprintf(fr, ",,,");
            fprintf(fr, "%lld,%lld,%.3f,%.10e,", (long long)st.blocks_phase1, (long long)st.blocks_final, tmed / n * 1e9, sse);
            fprintf(fr, "%.3f\n", t_read);
            std::fflush(fr);

            // solution yhat: only for one P per instance (saves disk space)
            std::string saved;
            if (P == P_save) {
                const std::string tag = par ? "par" + std::to_string(P) : "seq";
                const fs::path yf = ydir / (in.path.stem().string() + "." + tag + ".yhat.bin");
                if (!write_bin(yf, yhat)) fprintf(stderr, "cannot write %s\n", yf.string().c_str());
                saved = "  -> " + yf.filename().string();
            }

            printf("%-28s %s P=%-3d t_med=%.4fs  (%.2f ns/pt)  blocks=%lld%s\n", in.path.filename().string().c_str(), algo.c_str(),
                   par ? P : 1, tmed, tmed / n * 1e9, (long long)st.blocks_final, saved.c_str());
        }
    }
    std::fclose(fr);
    printf("results appended to %s\n", out.c_str());
    return 0;
}
