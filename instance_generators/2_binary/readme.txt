# 2_binary: binary instances (Section 7, binary experiment)

y_i ~ Bernoulli(p_i),  p_i = sigma((s/n)(i - (n+1)/2)),  sigma(x) = 1/(1 + e^{-x});  optional w_i ~ U[0.5, 2].
Mimics the sorted scores and 0/1 labels of a calibration set (default s = 8: p_i from about 0.018 to 0.982).
Random numbers: std::mt19937_64 + std::bernoulli_distribution; instance k uses seed + k.
Defaults: s = 8, base seed = 12345, 10 instances.

    make
    mkdir -p instances                 # the generator does not create the folder
    ./gen_binary --n 1e6 --instances 10      # -> instances/binary_1_1000000.bin ... binary_10_1000000.bin
    ./gen_binary --n 1e7 --instances 10
    ./gen_binary --n 1e8 --instances 10
    options: --n --instances --slope --seed --weighted 0|1 --dir

Output: binary, raw 8-byte doubles (0.0 / 1.0), 8n bytes (1e8 -> 800 MB per file); weights in binary_k_n.wbin.
Same format as 1_gauss, so the solvers read it unchanged.
Python: numpy.fromfile("instances/binary_1_1000000.bin", dtype="<f8")
instances/binary_<n>_info.txt records the parameters, the seed and the number of ones of every instance.
