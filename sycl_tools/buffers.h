///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace sycl_tools {

// making a 1D buffer
template<typename Scalar_T, typename Vector_T>
auto make_buffer(Vector_T& vec, int M) {
  sycl::buffer<Scalar_T, 1> buf(vec.data(), sycl::range<1>(M));
  return buf;
}

// making a 2D buffer
template<typename Scalar_T, typename Vector_T>
auto make_buffer(Vector_T& vec, int M, int N) {
  sycl::buffer<Scalar_T, 2> buf(vec.data(), sycl::range<2>(M, N));
  return buf;
}

// making a 3D buffer
template<typename Scalar_T, typename Vector_T>
auto make_buffer(Vector_T& vec, int M, int N, int K) {
  sycl::buffer<Scalar_T, 3> buf(vec.data(), sycl::range<3>(M, N, K));
  return buf;
}

} // namespace sycl_tools