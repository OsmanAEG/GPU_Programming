#ifndef CPU_SOLUTIONS_H
#define CPU_SOLUTIONS_H

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
/// \brief Header file that provides helper functions
///        for cpu solutions.
///////////////////////////////////////////////////////////////////////

namespace sycl_tools {

///////////////////////////////////////////////////////////////////////
// dot product
///////////////////////////////////////////////////////////////////////
template <typename Vector_T>
auto
dot_product(Vector_T vec_A, Vector_T vec_B)
{
  using Scalar_T = typename Vector_T::value_type;
  Scalar_T sum = 0;

  for (int i = 0; i < vec_A.size(); ++i) {
    sum += vec_A[i] * vec_B[i];
  }

  return sum;
}

///////////////////////////////////////////////////////////////////////
// matrix multiplication
///////////////////////////////////////////////////////////////////////
template <typename Vector_T>
auto
matrix_multiplication_cpu(Vector_T vec_A, Vector_T vec_B, int M, int K, int N)
{
  Vector_T vec_C(M * N, 0);

  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        vec_C[j * M + i] += vec_A[k * M + i] * vec_B[j * K + k];
      }
    }
  }

  return vec_C;
}

///////////////////////////////////////////////////////////////////////
// vector addition
///////////////////////////////////////////////////////////////////////
template <typename Vector_T>
auto
vector_addition_cpu(Vector_T vec_A, Vector_T vec_B)
{
  Vector_T vec_C(vec_A.size(), 0.0);

  for (int i = 0; i < vec_A.size(); ++i) {
    vec_C[i] = vec_A[i] + vec_B[i];
  }

  return vec_C;
}

}  // namespace sycl_tools

#endif  // CPU_SOLUTIONS_H