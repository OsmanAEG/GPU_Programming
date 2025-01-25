#ifndef USM_H
#define USM_H

///////////////////////////////////////////////////////////////////////
// This file is part of Osman's GPU_Programming github repo
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
///        related to sycl usm allocations.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace sycl_tools {

// reserving usm memory
template<typename Queue_T, typename Vector_T>
auto make_usm(Queue_T& Q, Vector_T& vec) {
  using Scalar_T = typename Vector_T::value_type;
  const size_t N = vec.size();

  Scalar_T* usm_mem = sycl::malloc_shared<Scalar_T>(N, Q);

  Q.memcpy(usm_mem, vec.data(), N*sizeof(Scalar_T)).wait();

  return usm_mem;
}

} // namespace sycl_tools

#endif // USM_H