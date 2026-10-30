// run_stream.cpp -- runs McCalpin's STREAM benchmark (stream.c in this folder) for several array sizes
// and thread counts and collects the results in one CSV file.
//
// build:  g++ -O2 -std=c++17 -o run_stream run_stream.cpp
// usage:  ./run_stream                                   (sizes 1e6,1e7,1e8; threads 1,2,4,8,12; ntimes 20)
//         ./run_stream --sizes 1e8 --threads 1,2,4,8,12 --ntimes 20 --out results/stream.csv
//
// For every size n, stream.c is compiled once with the same flags as ../experiments/run
//     gcc -O2 -fopenmp -march=native -mcmodel=medium -DSTREAM_ARRAY_SIZE=n -DNTIMES=ntimes stream.c -o stream_n
// (-mcmodel=medium is needed because for n = 1e8 the three static arrays take 2.4 GB > 2 GB)
// and then run with OMP_NUM_THREADS=P for every P. Other OpenMP variables (OMP_PROC_BIND, OMP_PLACES)
// are inherited from the shell, as for ../experiments/run.
//
// STREAM runs every kernel ntimes times; the first run is discarded; it reports the best rate (from the
// minimum time) and the average/min/max time. Bytes per element counted by STREAM (no write-allocate):
//     Copy  c = a          16 B   = traffic of PAVA with unit weights   (read y,    write yhat)
//     Scale b = s*c        16 B
//     Add   c = a + b      24 B   = traffic of PAVA with random weights (read y, w, write yhat)
//     Triad a = b + s*c    24 B
//
// Output: CSV (appended; header written if the file is new), one line per (n, P, kernel):
//     date, n, threads, threads_counted, ntimes, kernel, bytes_per_elem, rate_GBps, t_avg, t_min, t_max,
//     ns_per_elem (= t_min / n, comparable with the ns per point of PAVA), validates
// plus the raw STREAM output in <out folder>/raw/stream_n<n>_P<P>.txt

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

struct Row { std::string kernel; int bytes; double rate_MBps, tavg, tmin, tmax; };

static std::vector<long long> parse_sizes(const std::string& s) {
    std::vector<long long> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::llround(std::strtod(t.c_str(), nullptr)));
    return v;
}
static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}
static void usage() {
    printf("usage: ./run_stream [--sizes 1e6,1e7,1e8] [--threads 1,2,4,8,12] [--ntimes 20] [--out results/stream.csv] [--src stream.c]\n");
}

int main(int argc, char** argv) {
    std::vector<long long> sizes = {1000000LL, 10000000LL, 100000000LL};
    std::vector<int> threads = {1, 2, 4, 8, 12};
    int ntimes = 20;
    std::string out = "results/stream.csv", src = "stream.c";
    for (int i = 1; i < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "-h" || k == "--help" || i + 1 >= argc) { usage(); return k == "-h" || k == "--help" ? 0 : 1; }
        const std::string v = argv[i + 1];
        if (k == "--sizes") sizes = parse_sizes(v);
        else if (k == "--threads") threads = parse_ints(v);
        else if (k == "--ntimes") ntimes = std::atoi(v.c_str());
        else if (k == "--out") out = v;
        else if (k == "--src") src = v;
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); usage(); return 1; }
    }
    if (ntimes < 2) ntimes = 2;
    if (!std::filesystem::exists(src)) { fprintf(stderr, "%s not found\n", src.c_str()); return 1; }

    namespace fs = std::filesystem;
    const fs::path outdir = fs::path(out).parent_path();
    const fs::path rawdir = (outdir.empty() ? fs::path(".") : outdir) / "raw";
    fs::create_directories(rawdir);

    char date[32]; { std::time_t now = std::time(nullptr); std::strftime(date, sizeof date, "%Y-%m-%d %H:%M", std::localtime(&now)); }
    const bool fresh = !fs::exists(out);
    FILE* fc = std::fopen(out.c_str(), "a");
    if (!fc) { fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    if (fresh) fprintf(fc, "date,n,threads,threads_counted,ntimes,kernel,bytes_per_elem,rate_GBps,t_avg,t_min,t_max,ns_per_elem,validates\n");

    const char* names[4] = {"Copy:", "Scale:", "Add:", "Triad:"};
    const int bytes[4] = {16, 16, 24, 24};

    for (long long n : sizes) {
        const std::string exe = "./stream_" + std::to_string(n);
        char cmd[1024];
        std::snprintf(cmd, sizeof cmd, "gcc -O2 -fopenmp -march=native -mcmodel=medium -DSTREAM_ARRAY_SIZE=%lld -DNTIMES=%d %s -o %s",
                      n, ntimes, src.c_str(), exe.c_str());
        printf("\n%s\n", cmd);
        if (std::system(cmd) != 0) { fprintf(stderr, "compilation failed\n"); return 1; }

        printf("n = %lld (%.1f MB per array), ntimes = %d\n", n, 8.0 * n / 1e6, ntimes);
        printf("   P   Copy GB/s  ns/elem    Add GB/s  ns/elem   Scale GB/s  Triad GB/s  valid\n");
        for (int P : threads) {
            std::snprintf(cmd, sizeof cmd, "OMP_NUM_THREADS=%d %s 2>&1", P, exe.c_str());
            FILE* pp = popen(cmd, "r");
            if (!pp) { fprintf(stderr, "cannot run %s\n", exe.c_str()); return 1; }
            std::string all; char line[1024];
            std::vector<Row> rows; bool valid = false; int counted = -1;
            while (std::fgets(line, sizeof line, pp)) {
                all += line;
                for (int q = 0; q < 4; ++q) {
                    const size_t len = std::strlen(names[q]);
                    if (std::strncmp(line, names[q], len) == 0) {
                        Row r; r.kernel = std::string(names[q], len - 1); r.bytes = bytes[q];
                        if (std::sscanf(line + len, "%lf %lf %lf %lf", &r.rate_MBps, &r.tavg, &r.tmin, &r.tmax) == 4) rows.push_back(r);
                    }
                }
                if (std::strstr(line, "Solution Validates")) valid = true;
                if (std::strstr(line, "Number of Threads counted")) { const char* e = std::strchr(line, '='); if (e) counted = std::atoi(e + 1); }
            }
            pclose(pp);

            const fs::path raw = rawdir / ("stream_n" + std::to_string(n) + "_P" + std::to_string(P) + ".txt");
            if (FILE* fr = std::fopen(raw.string().c_str(), "w")) { std::fputs(all.c_str(), fr); std::fclose(fr); }
            if (rows.size() != 4) { fprintf(stderr, "could not parse STREAM output, see %s\n", raw.string().c_str()); return 1; }
            if (counted != -1 && counted != P) fprintf(stderr, "warning: P = %d but STREAM counted %d threads\n", P, counted);

            for (const Row& r : rows)
                fprintf(fc, "%s,%lld,%d,%d,%d,%s,%d,%.3f,%.6f,%.6f,%.6f,%.4f,%d\n", date, n, P, counted, ntimes, r.kernel.c_str(),
                        r.bytes, r.rate_MBps / 1000.0, r.tavg, r.tmin, r.tmax, r.tmin * 1e9 / (double)n, valid ? 1 : 0);
            std::fflush(fc);
            printf("  %2d   %9.2f  %7.3f   %9.2f  %7.3f   %10.2f  %10.2f  %s\n", P,
                   rows[0].rate_MBps / 1000.0, rows[0].tmin * 1e9 / (double)n,
                   rows[2].rate_MBps / 1000.0, rows[2].tmin * 1e9 / (double)n,
                   rows[1].rate_MBps / 1000.0, rows[3].rate_MBps / 1000.0, valid ? "yes" : "NO");
        }
    }
    std::fclose(fc);
    printf("\nresults appended to %s, raw output in %s\n", out.c_str(), rawdir.string().c_str());
    return 0;
}
