#ifndef BUFFERS_H
#define BUFFERS_H

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
///        related to sycl buffers.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace sycl_tools {

// making a 0D buffer
template<typename Scalar_T>
auto make_buffer(Scalar_T& val) {
  sycl::buffer<Scalar_T, 1> buf(&val, sycl::range<1>(1));
  return buf;
}

// making a 1D buffer
template<typename Vector_T>
auto make_buffer(Vector_T& vec, int M) {
  using Scalar_T = typename Vector_T::value_type;
  sycl::buffer<Scalar_T, 1> buf(vec.data(), sycl::range<1>(M));
  return buf;
}

// making a 2D buffer
template<typename Vector_T>
auto make_buffer(Vector_T& vec, int M, int N) {
  using Scalar_T = typename Vector_T::value_type;
  sycl::buffer<Scalar_T, 2> buf(vec.data(), sycl::range<2>(M, N));
  return buf;
}

// making a 3D buffer
template<typename Vector_T>
auto make_buffer(Vector_T& vec, int M, int N, int K) {
  using Scalar_T = typename Vector_T::value_type;
  sycl::buffer<Scalar_T, 3> buf(vec.data(), sycl::range<3>(M, N, K));
  return buf;
}

} // namespace sycl_tools

#endif // BUFFERS_H
