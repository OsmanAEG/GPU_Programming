///////////////////////////////////////////////////////////////////////
// local
///////////////////////////////////////////////////////////////////////
#include "../sycl_tools/buffers.h"
#include "../sycl_tools/devices.h"
#include "../sycl_tools/validate.h"

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

int main() {
  int M = 120;
  int N = 500;

  // input matrices
  std::vector<Scalar_T> matrix_A(M*N, 2.5);
  std::vector<Scalar_T> matrix_B(M*N, 8.3);

  // output matrix
  std::vector<Scalar_T> matrix_C(M*N, 0.0);

  // sycl scope
  {
    // sycl buffers
    auto buf_A = sycl_tools::make_buffer<Scalar_T>(matrix_A, M, N);
    auto buf_B = sycl_tools::make_buffer<Scalar_T>(matrix_B, M, N);
    auto buf_C = sycl_tools::make_buffer<Scalar_T>(matrix_C, M, N);

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

  std::vector<Scalar_T> answer(M*N, 10.8);
  sycl_tools::check_equal(matrix_C, answer);
}
