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
/// \brief Basic example of matrix addition with sycl
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
using Vector_T = std::vector<Scalar_T>;

int main() {
  const int M = 120;
  const int N = 500;

  // input matrices
  const auto matrix_A = sycl_tools::make_random_vector<Scalar_T>(M * N, 0, 100);
  const auto matrix_B = sycl_tools::make_random_vector<Scalar_T>(M * N, 0, 100);

  // output matrix
  Vector_T matrix_C(M * N, 0.0);

  // answer computed on cpu
  const auto answer = sycl_tools::vector_addition_cpu(matrix_A, matrix_B);

  // sycl scope
  {
    // sycl buffers
    auto buf_A = sycl_tools::make_buffer(matrix_A, M, N);
    auto buf_B = sycl_tools::make_buffer(matrix_B, M, N);
    auto buf_C = sycl_tools::make_buffer(matrix_C, M, N);

    auto Q = sycl_tools::get_device(0, 0);

    Q.submit([&](sycl::handler& h) {
      sycl::accessor acc_A{buf_A, h};
      sycl::accessor acc_B{buf_B, h};
      sycl::accessor acc_C{buf_C, h};

      h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> idx) {
        acc_C[idx] = acc_A[idx] + acc_B[idx];
      });
    });
  }

  sycl_tools::check_equal_vector(matrix_C, answer);

  return 0;
}
