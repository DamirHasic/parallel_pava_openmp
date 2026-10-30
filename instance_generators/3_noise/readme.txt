# 3_noise: pure-noise instances (Section 7, pure-noise experiment)

y_i = sigma * eps_i,  eps_i ~ N(0,1);  optional w_i ~ U[0.5, 2].
Constant signal: PAVA pools almost everything (about n pooling steps); the solution has only about
H_n ~ ln n + 0.58 blocks (n = 1e8 -> about 19).
Random numbers: std::mt19937_64 + std::normal_distribution; instance k uses seed + k.
With the same seed, noise_k_n uses the same random stream as gauss_k_n (gen_gauss --slope 0).
Defaults: sigma = 1, base seed = 12345, 10 instances.

    make
    mkdir -p instances                 # the generator does not create the folder
    ./gen_noise --n 1e6 --instances 10      # -> instances/noise_1_1000000.bin ... noise_10_1000000.bin
    ./gen_noise --n 1e7 --instances 10
    ./gen_noise --n 1e8 --instances 10
    options: --n --instances --sigma --seed --weighted 0|1 --dir

Output: binary, raw 8-byte doubles, 8n bytes (1e8 -> 800 MB per file); weights in noise_k_n.wbin.
Same format as 1_gauss and 2_binary, so the solvers read it unchanged.
Python: numpy.fromfile("instances/noise_1_1000000.bin", dtype="<f8")
instances/noise_<n>_info.txt records the parameters and the seed of every instance.
