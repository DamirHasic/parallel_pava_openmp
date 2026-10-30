# 4_increasing: increasing instances (Section 7, increasing experiment)

y_i = i, i = 1..n;  optional w_i ~ U[0.5, 2].
Already isotonic: PAVA never pools, y_hat = y, B_1 = B_2 = n blocks
(worst case for the memory traffic of phases 2 and 3). Weights do not change the solution.
y does not depend on the seed, so with unit weights all instances are identical: 1 instance is the default.
The seed (seed + k for instance k; base 12345) is used only for the weights (std::mt19937_64).

    make
    mkdir -p instances                 # the generator does not create the folder
    ./gen_increasing --n 1e6          # -> instances/increasing_1_1000000.bin
    ./gen_increasing --n 1e7
    ./gen_increasing --n 1e8
    ./gen_increasing --n 1e8 --weighted 1 --dir <other folder, created first>   # weighted variant (+ .wbin)
    options: --n --instances --seed --weighted 0|1 --dir

Output: binary, raw 8-byte doubles, 8n bytes (1e8 -> 800 MB per file); weights in increasing_k_n.wbin.
Same format as 1_gauss, 2_binary and 3_noise, so the solvers read it unchanged.
Python: numpy.fromfile("instances/increasing_1_1000000.bin", dtype="<f8")
instances/increasing_<n>_info.txt records the parameters and the seed of every instance.
