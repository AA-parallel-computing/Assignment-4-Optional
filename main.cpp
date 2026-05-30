#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>

// Naive matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j].
void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// Blocked (cache-tiled) matrix multiplication.
//
// The outer loops (ii, jj, kk) walk over block_size x block_size tiles so that
// each tile's working set is intended to fit in cache. The inner loops use
// i-k-j ordering (rather than the pseudocode's i-j-k) so that the innermost
// loop accesses B[k*p+j] and C[i*p+j] with stride 1; this is friendly to both
// the hardware prefetcher and the compiler's auto-vectorizer, and lets
// A[i*n+k] be hoisted into a register as loop-invariant.
//
// std::min(...) clips the tile bounds so the function works for matrices
// whose dimensions are not a multiple of block_size (e.g. cases 2, 5, 7, 9).
// Because each tile accumulates into C across the kk loop, C must start at
// zero.
void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }

    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                uint32_t i_max = std::min(ii + block_size, m);
                uint32_t j_max = std::min(jj + block_size, p);
                uint32_t k_max = std::min(kk + block_size, n);
                for (uint32_t i = ii; i < i_max; ++i) {
                    for (uint32_t k = kk; k < k_max; ++k) {
                        float a_ik = A[i * n + k];
                        for (uint32_t j = jj; j < j_max; ++j) {
                            C[i * p + j] += a_ik * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP.
//
// The outer i loop is parallelized with #pragma omp parallel for. Each thread
// processes a disjoint set of rows of C (first zeroing them, then performing
// the i-k-j accumulation), so the kernel is data-race free without any
// atomics, locks, or reduction clauses. The i-k-j inner ordering is used for
// the same stride-1 / auto-vectorization reasons as in blocked_matmul.
void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            C[i * p + j] = 0.0f;
        }
        for (uint32_t k = 0; k < n; ++k) {
            float a_ik = A[i * n + k];
            for (uint32_t j = 0; j < p; ++j) {
                C[i * p + j] += a_ik * B[k * p + j];
            }
        }
    }
}

// Compare result_file against reference_file element-wise.
//
// Returns true if both files have the same dimensions and every value in
// result_file is within abs_tol + rel_tol * |expected| of the corresponding
// value in reference_file. The absolute tolerance accounts for the reference
// being printed to only two decimal places; the relative tolerance accounts
// for accumulated floating-point error in larger dot products.
bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream r(result_file), e(reference_file);
    if (!r.is_open() || !e.is_open()) {
        return false;
    }
    uint32_t r_rows, r_cols, e_rows, e_cols;
    r >> r_rows >> r_cols;
    e >> e_rows >> e_cols;
    if (r_rows != e_rows || r_cols != e_cols) {
        return false;
    }
    const uint32_t total = r_rows * r_cols;
    const float abs_tol = 1e-2f;
    const float rel_tol = 1e-3f;
    for (uint32_t i = 0; i < total; ++i) {
        float a, b;
        r >> a;
        e >> b;
        if (std::fabs(a - b) > abs_tol + rel_tol * std::fabs(b)) {
            return false;
        }
    }
    return true;
}

// Read a matrix:
//   first line: "<rows> <cols>"
//   followed by rows*cols whitespace-separated floats.
// Returns a newly allocated float[rows*cols];
static float *read_matrix(const std::string &path, uint32_t &rows, uint32_t &cols) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return nullptr;
    }
    in >> rows >> cols;
    float *data = new float[rows * cols];
    for (uint32_t i = 0; i < rows * cols; ++i) {
        in >> data[i];
    }
    return data;
}

// Write a matrix in the same format as read_matrix:
// header line "<rows> <cols>" followed by row-major values, one row per line.
static void write_matrix(const std::string &path, const float *data, uint32_t rows, uint32_t cols) {
    std::ofstream out(path);
    out << rows << " " << cols << "\n";
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (j > 0) out << " ";
            out << data[i * cols + j];
        }
        out << "\n";
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    // Read input0.raw (matrix A) and input1.raw (matrix B)
    uint32_t m = 0, n = 0, n_b = 0, p = 0;
    float *A = read_matrix(input0_file, m, n);
    float *B = read_matrix(input1_file, n_b, p);
    if (!A || !B || n != n_b) {
        std::cerr << "Failed to load inputs for case " << case_number << std::endl;
        delete[] A; delete[] B;
        return 1;
    }

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Average over a few runs to reduce timing variability (as suggested in the README).
    const int iters = 5;

    // Measure performance of naive_matmul
    double naive_total = 0.0;
    for (int it = 0; it < iters; ++it) {
        double start_time = omp_get_wtime();
        naive_matmul(C_naive, A, B, m, n, p);
        naive_total += omp_get_wtime() - start_time;
    }
    double naive_time = naive_total / iters;
    write_matrix(result_file, C_naive, m, p);
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    double blocked_total = 0.0;
    for (int it = 0; it < iters; ++it) {
        double start_time = omp_get_wtime();
        blocked_matmul(C_blocked, A, B, m, n, p, 32);
        blocked_total += omp_get_wtime() - start_time;
    }
    double blocked_time = blocked_total / iters;
    write_matrix(result_file, C_blocked, m, p);
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    double parallel_total = 0.0;
    for (int it = 0; it < iters; ++it) {
        double start_time = omp_get_wtime();
        parallel_matmul(C_parallel, A, B, m, n, p);
        parallel_total += omp_get_wtime() - start_time;
    }
    double parallel_time = parallel_total / iters;
    write_matrix(result_file, C_parallel, m, p);
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct) {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Blocked time: " << blocked_time << " seconds\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}