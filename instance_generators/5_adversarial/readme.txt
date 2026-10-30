# 5_adversarial: adversarial instances (Section 7, worst case of prop:worst)

y_i = i (i = 1..n-1), y_n = -c n^2;  optional w_i ~ U[0.5, 2].   Default c = 1: y = (1, 2, ..., n-1, -n^2).
Unit weights: the solution is one block (value -(n+1)/2 for c = 1). Phase 1 gives b' = c_{P-1} + 1 >= n/2
blocks (one per point in the first P-1 chunks, one for the last chunk), and Phase 2 pools b' - 1 times.

Weights: prop:worst is for unit weights. With w_i in [0.5, 2] and c = 1 the solution is NOT always one block
(fails when w_n < ~0.6); c = 2 guarantees one block for every draw. The generator checks the condition
sum_{i<n} w_i i < c w_n n^2 for every instance (info file, column one_block). Use --c 2 with --weighted 1.

y does not depend on the seed, so with unit weights all instances are identical: 1 instance is the default.
The seed (seed + k for instance k; base 12345) is used only for the weights (std::mt19937_64).

    make
    mkdir -p instances                 # the generator does not create the folder
    ./gen_adversarial --n 1e6                     # -> instances/adversarial_1_1000000.bin
    ./gen_adversarial --n 1e7
    ./gen_adversarial --n 1e8
    ./gen_adversarial --n 1e8 --weighted 1 --c 2 --dir <other folder, created first>   # weighted variant (+ .wbin)
    options: --n --instances --c --seed --weighted 0|1 --dir

Output: binary, raw 8-byte doubles, 8n bytes (1e8 -> 800 MB per file); weights in adversarial_k_n.wbin.
Same format as the other families, so the solvers read it unchanged.
Python: numpy.fromfile("instances/adversarial_1_1000000.bin", dtype="<f8")
instances/adversarial_<n>_info.txt records the parameters, the seed of every instance and the one-block check.
