# Assignment 4: Matrix Multiplication Optimization

**Course:** Parallel Programming (PP-26), Åbo Akademi University

## Group members

- Behroz Karim: 2502071
- Hasnain Ajmal: 2502065
- Hugo Pérez Muñoz: 2501103
- Talha Rizwan: 2503934
- Max Nummila: 2202236

## Overview

Three single-precision dense matrix-multiplication implementations are provided in `main.cpp`:

- `naive_matmul`: textbook triple-nested loop, used as the baseline.
- `blocked_matmul`: cache-tiled with `i -> k -> j` inner ordering (`block_size = 32`).
- `parallel_matmul`: row-wise OpenMP parallelism with the same `i -> k -> j` inner ordering.

Note: For the full implementation explanation, including the optimization journey, the reasoning behind the `i -> k -> j` loop reordering, the choice of the block-size, and the compiler-flag exploration, please see **`writeup.pdf`**.

## Performance results

Measured on a 13th Gen Intel Core i5-13420H (6 P-cores / 12 logical threads) under WSL2 (Ubuntu 22.04), g++ 11.4.0, `OMP_NUM_THREADS=8`. Each value is the average of 5 back-to-back runs.

### Table 1. Supplied CMake build (no explicit optimization flags)

| Case | Dimensions (m × n × p) | Naive (s) | Blocked (s) | Parallel (s) | Blocked Speedup | Parallel Speedup |
|------|------------------------|-----------|-------------|--------------|-----------------|------------------|
| 0    | 64 × 64 × 64           | 0.000915  | 0.000913    | 0.000630     | 1.00×           | 1.45×            |
| 1    | 128 × 64 × 128         | 0.001891  | 0.001884    | 0.000504     | 1.00×           | 3.75×            |
| 2    | 100 × 128 × 56         | 0.001282  | 0.001333    | 0.000380     | 0.96×           | 3.38×            |
| 3    | 128 × 64 × 128         | 0.001970  | 0.001971    | 0.000889     | 1.00×           | 2.22×            |
| 4    | 32 × 128 × 32          | 0.000292  | 0.000249    | 0.000159     | 1.17×           | 1.84×            |
| 5    | 200 × 100 × 256        | 0.009277  | 0.009140    | 0.003738     | 1.01×           | 2.48×            |
| 6    | 256 × 256 × 256        | 0.032117  | 0.030844    | 0.011711     | 1.04×           | 2.74×            |
| 7    | 256 × 300 × 256        | 0.037422  | 0.035864    | 0.009984     | 1.04×           | **3.75×**        |
| 8    | 64 × 128 × 64          | 0.001006  | 0.000917    | 0.000295     | 1.10×           | 3.41×            |
| 9    | 256 × 256 × 257        | 0.031400  | 0.031574    | 0.008730     | 0.99×           | 3.60×            |

### Table 2. Compiler flags (`-O3 -march=native -ffast-math`)

| Case | Dimensions (m × n × p) | Naive (s) | Blocked (s) | Parallel (s) | Blocked Speedup | Parallel Speedup |
|------|------------------------|-----------|-------------|--------------|-----------------|------------------|
| 0    | 64 × 64 × 64           | 0.000237  | 0.000152    | 0.000169     | 1.56×           | 1.41×            |
| 1    | 128 × 64 × 128         | 0.000825  | 0.000370    | 0.000163     | 2.23×           | 5.08×            |
| 2    | 100 × 128 × 56         | 0.000462  | 0.000248    | 0.000136     | 1.86×           | 3.40×            |
| 3    | 128 × 64 × 128         | 0.000844  | 0.000449    | 0.000153     | 1.88×           | **5.51×**        |
| 4    | 32 × 128 × 32          | 0.000076  | 0.000048    | 0.000082     | 1.60×           | 0.94×            |
| 5    | 200 × 100 × 256        | 0.003272  | 0.001776    | 0.000863     | 1.84×           | 3.79×            |
| 6    | 256 × 256 × 256        | 0.012521  | 0.006108    | 0.003052     | 2.05×           | 4.10×            |
| 7    | 256 × 300 × 256        | 0.015248  | 0.007838    | 0.003625     | 1.95×           | 4.21×            |
| 8    | 64 × 128 × 64          | 0.000318  | 0.000181    | 0.000122     | 1.76×           | 2.60×            |
| 9    | 256 × 256 × 257        | 0.012761  | 0.005966    | 0.004022     | 2.14×           | 3.17×            |

## Notes on the results

- Under the supplied CMake build, the **parallel** implementation already delivers meaningful speedups (up to 3.75×), confirming that the `i -> k -> j` reordering and the row-wise OpenMP parallelism work as intended. The **blocked** implementation, however, hovers around 1×, the test matrices fit entirely in L2 cache, so blocking has essentially no capacity misses to eliminate at these sizes.
- With aggressive compiler flags, both kernels improve, blocked rises to ~2× and parallel reaches up to 5.51×. The `i -> k -> j` inner loop is an ideal target for the compiler's auto-vectorizer.
