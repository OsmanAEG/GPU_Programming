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
#include "../sycl_tools/usm.h"
#include "../sycl_tools/validate.h"
#include "../sycl_tools/vectors.h"

///////////////////////////////////////////////////////////////////////
// onemath
///////////////////////////////////////////////////////////////////////
#include <oneapi/mkl.hpp>

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <algorithm>
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

///////////////////////////////////////////////////////////////////////
// matmul with buffers
///////////////////////////////////////////////////////////////////////
template<typename Queue_T, typename Vector_T>
auto matmul_with_buffers(
    Queue_T Q,
    const Vector_T& A,
    const Vector_T& B,
    Vector_T& C,
    const std::size_t& M,
    const std::size_t& K,
    const std::size_t& N,
    const double alpha = 1.0,
    const double beta = 0.0) {
  std::cout << "MATMUL WITH BUFFERS IN ONEMATH" << std::endl;

  const auto answer = sycl_tools::matrix_multiplication_cpu(A, B, M, K, N);

  // sycl scope
  {
    // sycl buffers
    auto buf_A = sycl_tools::make_buffer(A, M * K);
    auto buf_B = sycl_tools::make_buffer(B, K * N);
    auto buf_C = sycl_tools::make_buffer(C, M * N);

    // general matrix multiplication
    oneapi::mkl::blas::column_major::gemm(
        Q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        M,
        N,
        K,
        alpha,
        buf_A,
        M,
        buf_B,
        K,
        beta,
        buf_C,
        M);

    Q.wait();
  }

  sycl_tools::check_equal_vector(C, answer);
}

///////////////////////////////////////////////////////////////////////
// matmul with usm
///////////////////////////////////////////////////////////////////////
template<typename Queue_T, typename Vector_T>
auto matmul_with_usm(
    Queue_T Q,
    const Vector_T& A,
    const Vector_T& B,
    Vector_T& C,
    const std::size_t& M,
    const std::size_t& K,
    const std::size_t& N,
    const Scalar_T& alpha = 1.0,
    const Scalar_T& beta = 0.0) {
  std::cout << "MATMUL WITH USM IN ONEMATH" << std::endl;

  // answer computed on cpu
  const auto answer = sycl_tools::matrix_multiplication_cpu(A, B, M, K, N);

  auto usm_A = sycl_tools::make_usm(Q, A);
  auto usm_B = sycl_tools::make_usm(Q, B);
  auto usm_C = sycl_tools::make_usm(Q, C);

  // general matrix multiplication
  oneapi::mkl::blas::column_major::gemm(
      Q,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      M,
      N,
      K,
      alpha,
      usm_A,
      M,
      usm_B,
      K,
      beta,
      usm_C,
      M);

  Q.wait();

  Q.memcpy(C.data(), usm_C, M * N * sizeof(Scalar_T)).wait();

  sycl_tools::check_equal_vector(C, answer);
}

int main() {
  auto Q = sycl_tools::get_device(0, 0);

  const std::size_t M = 123;
  const std::size_t K = 568;
  const std::size_t N = 399;

  // input matrices
  const auto A = sycl_tools::make_random_vector<Scalar_T>(M * K, 0, 100);
  const auto B = sycl_tools::make_random_vector<Scalar_T>(K * N, 0, 100);

  // output matrix
  Vector_T C(M * N, 0.0);

  // execute matmul example with buffers
  std::cout << std::endl;
  matmul_with_buffers(Q, A, B, C, M, K, N);

  // reset vector C
  std::fill(C.begin(), C.end(), 0.0);

  // execute matmul example with usm
  std::cout << std::endl;
  matmul_with_usm(Q, A, B, C, M, K, N);

  return 0;
}