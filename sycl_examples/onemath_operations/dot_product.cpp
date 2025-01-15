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
/// \brief Onemath example for vector dot product with sycl
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
  const int N = 2048;

  const int stride = 1;

  // input vectors
  const auto vec_A = sycl_tools::make_random_vector<Scalar_T>(N, 0, 100);
  const auto vec_B = sycl_tools::make_random_vector<Scalar_T>(N, 0, 100);

  // final result
  Scalar_T sum = 0;

  // answer computed on cpu
  const auto answer = sycl_tools::dot_product(vec_A, vec_B);

  // sycl scope
  {
    // sycl buffers
    auto buf_A   = sycl_tools::make_buffer(vec_A, N);
    auto buf_B   = sycl_tools::make_buffer(vec_B, N);
    auto sum_buf = sycl_tools::make_buffer(sum);

    auto Q = sycl_tools::get_device(0, 0);

    // vector dot product
    oneapi::mkl::blas::column_major::dot(
      Q,
      N,
      buf_A, stride,
      buf_B, stride,
      sum_buf
    );

    Q.wait();
  }

  sycl_tools::check_equal_value(sum, answer);

  return 0;
}