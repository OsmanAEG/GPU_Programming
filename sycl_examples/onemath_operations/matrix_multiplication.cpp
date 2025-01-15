///////////////////////////////////////////////////////////////////////
// This file is part of the Osman's GPU_Programming github repo
// It is licensed under the MIT licence.  A copy of
// this license, in a file named LICENSE.md, should have been
// distributed with this file.  A copy of this license is also
// currently available at "http://opensource.org/licenses/MIT".
//
// Unless explicitly stated, all contributions intentionally submitted
// to this project shall also be under the terms and conditions of this
// license, without any additional terms or conditions.
///////////////////////////////////////////////////////////////////////
/// \file
/// \brief Onemath example for matrix multiplication with sycl
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../sycl_tools/buffers.h"
#include "../sycl_tools/cpu_solutions.h"
#include "../sycl_tools/devices.h"
#include "../sycl_tools/validate.h"
#include "../sycl_tools/vectors.h"

///////////////////////////////////////////////////////////////////////
// onemath
///////////////////////////////////////////////////////////////////////
#include <oneapi/mkl.hpp>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////
// defining types
///////////////////////////////////////////////////////////////////////
using Scalar_T = double;
using Vector_T = std::vector<double>;

int main() {
  const int M = 123;
  const int K = 568;
  const int N = 399;

  const Scalar_T alpha = 1.0;
  const Scalar_T beta = 0.0;

  // input matrices
  const auto matrix_A = sycl_tools::make_random_vector<Scalar_T>(M*K, 0, 100);
  const auto matrix_B = sycl_tools::make_random_vector<Scalar_T>(K*N, 0, 100);

  // output matrix
  Vector_T matrix_C(M*N, 0.0);

  // answer computed on cpu
  const auto answer = sycl_tools::matrix_multiplication_cpu(matrix_A, matrix_B, M, K, N);

  // sycl scope
  {
    // sycl buffers
    auto buf_A = sycl_tools::make_buffer(matrix_A, M*K);
    auto buf_B = sycl_tools::make_buffer(matrix_B, K*N);
    auto buf_C = sycl_tools::make_buffer(matrix_C, M*N);

    auto Q = sycl_tools::get_device(0, 0);

    // general matrix multiplication
    oneapi::mkl::blas::column_major::gemm(
      Q,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      M, N, K,
      alpha,
      buf_A, M,
      buf_B, K,
      beta,
      buf_C, M
    );

    Q.wait();
  }

  sycl_tools::check_equal_vector(matrix_C, answer);

  return 0;
}