# 1_gauss: Gaussian instances (Section 7, Experiment 1)

y_i = a (t_i - 1/2) + sigma * eps_i,  t_i = (i - 1/2)/n,  eps_i ~ N(0,1);  optional w_i ~ U[0.5, 2].
Random numbers: std::mt19937_64 + std::normal_distribution; instance k uses seed + k.
Defaults: a = 4, sigma = 1, base seed = 12345, 10 instances.

    make
    mkdir -p instances                 # the generator does not create the folder
    ./gen_gauss --n 1e6 --instances 10      # -> instances/gauss_1_1000000.bin ... gauss_10_1000000.bin
    ./gen_gauss --n 1e7 --instances 10
    ./gen_gauss --n 1e8 --instances 10
    options: --n --instances --slope --sigma --seed --weighted 0|1 --dir

Output: binary, raw 8-byte doubles, 8n bytes (1e8 -> 800 MB per file); weights in gauss_k_n.wbin.
Python: numpy.fromfile("instances/gauss_1_1000000.bin", dtype="<f8")
instances/gauss_<n>_info.txt records the parameters and the seed of every instance.
