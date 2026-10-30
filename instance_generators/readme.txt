instance_generators: the five instance families of Section 5
============================================================

Each subfolder has one generator (.cpp), a Makefile and a readme.txt with all options.

  folder          program           y_i
  1_gauss         gen_gauss         4 (t_i - 1/2) + eps_i
  2_binary        gen_binary        Bernoulli(q_i),  q_i = 1 / (1 + exp(-8 (t_i - 1/2)))
  3_noise         gen_noise         eps_i
  4_increasing    gen_increasing    i                          (already isotonic: n blocks, no pooling)
  5_adversarial   gen_adversarial   (1, 2, ..., n-1, -c n^2)   (one block; worst case of Algorithm 2)

  t_i = (i - 1/2)/n,  eps_i ~ N(0, 1).
  With --weighted 1 the weights are w_i ~ U[0.5, 2]; otherwise unit weights.
  Random numbers: std::mt19937_64, seed 12345 + k for instance k.
  For the increasing and adversarial families y does not depend on the seed, so all instances
  with unit weights are identical.

Build:  make  in each subfolder, or from this folder:
          for d in */; do make -C "$d"; done


INSTANCES OF THE PAPER
----------------------
n = 10^6, 10^7, 10^8; ten instances each; unit weights and random weights.

  cd 1_gauss
  mkdir -p instances instances_w          # the generators do not create folders
  for n in 1e6 1e7 1e8; do ./gen_gauss --n $n --instances 10; done                                 # -> instances/
  for n in 1e6 1e7 1e8; do ./gen_gauss --n $n --instances 10 --weighted 1 --dir instances_w; done  # -> instances_w/

The same for gen_binary, gen_noise and gen_increasing. For the adversarial family use c = 2 with
random weights (c = 1 does not always give one block when the weights are random):

  cd 5_adversarial
  mkdir -p instances instances_w
  for n in 1e6 1e7 1e8; do ./gen_adversarial --n $n --instances 10; done
  for n in 1e6 1e7 1e8; do ./gen_adversarial --n $n --instances 10 --weighted 1 --c 2 --dir instances_w; done

Put the weighted instances in a different folder: their file names are the same, and ../experiments/run
uses weights whenever a .wbin file is next to the .bin file.


FILES
-----
  <type>_<k>_<n>.bin     y      raw little-endian 8-byte doubles, no header, 8n bytes (800 MB for n = 10^8)
  <type>_<k>_<n>.wbin    w      same format (weighted instances only)
  <type>_<n>_info.txt           parameters and seed of every instance

  Python: numpy.fromfile(name, dtype="<f8")

Disk space: one family with both weightings (n = 10^6, 10^7, 10^8, ten instances each) takes about 27 GB.
