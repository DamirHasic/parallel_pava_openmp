#!/usr/bin/env python3
"""Correctness tests for pava.hpp via the C API (libpava.so).

Checks, for random, binary, tied, weighted and adversarial inputs:
  1. sequential PAVA == brute-force min-max formula  (small n)
  2. sequential PAVA == scikit-learn IsotonicRegression  (if scikit-learn is installed)
  3. three-phase OpenMP PAVA == sequential PAVA for P in {1,2,3,4,7,16,64}, including n < P
  4. isotonicity of the output and block-count invariants (b <= B_1)
Tolerance: relative 1e-12 (the two algorithms accumulate the same sums in a different order).
"""
import ctypes, os, sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(here, "libpava.so"))
dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
lib.pava_seq.restype = ctypes.c_int64
lib.pava_seq.argtypes = [dp, ctypes.c_void_p, ctypes.c_int64, dp]
lib.pava_omp.restype = ctypes.c_int64
lib.pava_omp.argtypes = [dp, ctypes.c_void_p, ctypes.c_int64, dp, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]


def wptr(w):
    return None if w is None else w.ctypes.data_as(ctypes.c_void_p)


def pava_seq(y, w=None):
    y = np.ascontiguousarray(y, dtype=np.float64); yhat = np.empty_like(y)
    k = lib.pava_seq(y, wptr(w), len(y), yhat)
    return yhat, k


def pava_omp(y, w=None, P=4):
    y = np.ascontiguousarray(y, dtype=np.float64); yhat = np.empty_like(y)
    b1 = ctypes.c_int64(0)
    k = lib.pava_omp(y, wptr(w), len(y), yhat, P, ctypes.byref(b1))
    return yhat, k, b1.value


def minmax(y, w):
    """Brute-force min-max formula, O(n^3): yhat_i = max_{j<=i} min_{k>=i} mean(y[j..k])."""
    n = len(y); out = np.empty(n)
    cw = np.concatenate([[0.0], np.cumsum(w)]); cs = np.concatenate([[0.0], np.cumsum(w * y)])
    for i in range(n):
        best = -np.inf
        for j in range(i + 1):
            m = min((cs[k + 1] - cs[j]) / (cw[k + 1] - cw[j]) for k in range(i, n))
            best = max(best, m)
        out[i] = best
    return out


def close(a, b, tol=1e-12):
    scale = max(1.0, np.max(np.abs(b)))
    return np.max(np.abs(a - b)) <= tol * scale


def gen(kind, n, rng):
    if kind == "gauss":   y = 4 * (np.arange(n) / max(n, 1) - 0.5) + rng.standard_normal(n)
    elif kind == "binary": y = (rng.random(n) < 1 / (1 + np.exp(-8 * (np.arange(n) / max(n, 1) - 0.5)))).astype(float)
    elif kind == "ties":  y = rng.integers(0, 3, n).astype(float)
    elif kind == "adv":
        y = np.arange(1, n + 1, dtype=float)
        if n: y[-1] = -float(n) ** 2
    elif kind == "inc":   y = np.arange(n, dtype=float)
    elif kind == "dec":   y = -np.arange(n, dtype=float)
    else: raise ValueError(kind)
    return y


def main():
    rng = np.random.default_rng(0)
    kinds = ["gauss", "binary", "ties", "adv", "inc", "dec"]
    P_list = [1, 2, 3, 4, 7, 16, 64]
    nfail = 0

    # 1. brute force
    for t in range(300):
        n = rng.integers(1, 25); kind = kinds[t % len(kinds)]
        y = gen(kind, n, rng); w = np.ones(n) if t % 2 == 0 else rng.uniform(0.5, 2.0, n)
        ref = minmax(y, w); ys, _ = pava_seq(y, None if t % 2 == 0 else w)
        if not close(ys, ref): nfail += 1; print("FAIL brute force", kind, n)
    print("brute-force min-max check done")

    # 2. scikit-learn
    try:
        from sklearn.isotonic import IsotonicRegression
        for t in range(60):
            n = int(rng.integers(1, 20000)); kind = kinds[t % len(kinds)]
            y = gen(kind, n, rng); w = None if t % 2 == 0 else rng.uniform(0.5, 2.0, n)
            ref = IsotonicRegression(increasing=True).fit_transform(np.arange(n, dtype=float), y, sample_weight=w)
            ys, _ = pava_seq(y, w)
            if not close(ys, ref): nfail += 1; print("FAIL sklearn", kind, n)
        print("scikit-learn check done")
    except ImportError:
        print("scikit-learn not installed: skipped")

    # 3. three-phase == sequential, 4. invariants
    for t in range(400):
        n = int(rng.integers(0, 3000)) if t % 3 else int(rng.integers(0, 40)); kind = kinds[t % len(kinds)]
        y = gen(kind, n, rng); w = None if t % 2 == 0 else rng.uniform(0.5, 2.0, n)
        ys, ks = pava_seq(y, w)
        if np.any(np.diff(ys) < 0): nfail += 1; print("FAIL not isotonic", kind, n)
        for P in P_list:
            yo, ko, b1 = pava_omp(y, w, P)
            if n and not close(yo, ys): nfail += 1; print("FAIL three-phase", kind, n, P)
            if not (ko == ks and ko <= b1 <= n): nfail += 1; print("FAIL block counts", kind, n, P, ks, ko, b1)
    # a few large ones
    for kind in kinds:
        n = 2_000_000; y = gen(kind, n, rng)
        ys, ks = pava_seq(y); yo, ko, b1 = pava_omp(y, None, 16)
        ok = close(yo, ys) and ko == ks
        print(f"n={n} {kind:6s} blocks={ks:8d} B1(P=16)={b1:8d} {'ok' if ok else 'FAIL'}")
        if not ok: nfail += 1
    print("three-phase check done")
    print("ALL TESTS PASSED" if nfail == 0 else f"{nfail} FAILURES")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
