#include <bits/stdc++.h>
#include "fp8.hpp"
#include "linear.hpp"
int main() {
    const int N = 3; // Matrix dimension
    // const int M = 2; // Vector dimension

    // Define matrix A
    fp8 A[N][N] = {
        {fp32_to_fp8(1.0), fp32_to_fp8(2.0), fp32_to_fp8(3.0)},
        {fp32_to_fp8(4.0), fp32_to_fp8(5.0), fp32_to_fp8(6.0)},
        {fp32_to_fp8(7.0), fp32_to_fp8(8.0), fp32_to_fp8(9.0)}
    };

    // Define vector x
    fp8 x[N] = {fp32_to_fp8(1.0), fp32_to_fp8(2.0), fp32_to_fp8(3.0)};

    // Define vector y
    fp8 y[N];

    // Perform matrix-vector multiplication: y = A * x
    // cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, (double*)A, N, x, 1, 0.0, y, 1);
    gemv((uint8_t *)(&A[0][0]),N,N,(uint8_t *)x,(uint8_t *)y,0);

    // Display the result vector y
    std::cout << "Result vector y:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << fp8_to_fp32(y[i]) << std::endl;
    }

    return 0;
}