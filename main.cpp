#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < p; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    for (uint32_t i = 0; i < m * p; i++) {
        C[i] = 0.0f;
    }

    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                uint32_t i_end = std::min(ii + block_size, m);
                uint32_t j_end = std::min(jj + block_size, p);
                uint32_t k_end = std::min(kk + block_size, n);

                for (uint32_t i = ii; i < i_end; i++) {
                    for (uint32_t j = jj; j < j_end; j++) {
                        float sum = C[i * p + j];
                        for (uint32_t k = kk; k < k_end; k++) {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] = sum;
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    naive_matmul(C, A, B, m, n, p);
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream result(result_file);
    std::ifstream ref(reference_file);

    if (!result.is_open() || !ref.is_open()) {
        return false;
    }

    uint32_t rm, rp, fm, fp;
    result >> rm >> rp;
    ref >> fm >> fp;

    if (rm != fm || rp != fp) {
        return false;
    }

    for (uint32_t i = 0; i < rm * rp; i++) {
        float a, b;
        result >> a;
        ref >> b;
        if (std::fabs(a - b) > 1e-3f) {
            return false;
        }
    }

    return true;
}

void write_result(const std::string &result_file, float *C, uint32_t m, uint32_t p) {
    std::ofstream out(result_file);
    out << m << " " << p << "\n";
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < p; j++) {
            out << C[i * p + j];
            if (j + 1 < p) out << " ";
        }
        out << "\n";
    }
    out.close();
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

std::ifstream input0(input0_file);
if (!input0.is_open()) {
    std::cerr << "Failed to open " << input0_file << std::endl;
    return 1;
}

uint32_t m, n;
input0 >> m >> n;

float *A = new float[m * n];
for (uint32_t i = 0; i < m * n; i++) {
    input0 >> A[i];
}
input0.close();


std::ifstream input1(input1_file);
if (!input1.is_open()) {
    std::cerr << "Failed to open " << input1_file << std::endl;
    delete[] A;
    return 1;
}

uint32_t n2, p;
input1 >> n2 >> p;

if (n != n2) {
    std::cerr << "Matrix dimension mismatch: input0 columns != input1 rows" << std::endl;
    delete[] A;
    return 1;
}

float *B = new float[n * p];
for (uint32_t i = 0; i < n * p; i++) {
    input1 >> B[i];
}
input1.close();


    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    write_result(result_file, C_naive, m, p);


    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;

    write_result(result_file, C_blocked, m, p);


    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    write_result(result_file, C_parallel, m, p);


    // Validate parallel result
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