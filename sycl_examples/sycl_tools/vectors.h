#ifndef VECTORS_H
#define VECTORS_H

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
///        to create vectors.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <bits/stdc++.h>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace sycl_tools {

// generate random values
template<typename Scalar_T>
auto generate_random_vec(int N, double min, double max) {
  std::vector<Scalar_T> vec(N, 0.0);

  std::random_device random_device;
  std::mt19937 generator(random_device());

  if constexpr (std::is_integral<Scalar_T>::value) {
    std::uniform_int_distribution<Scalar_T> distribution(min, max);
    for (auto& val : vec)
      val = distribution(generator);
  } else {
    std::uniform_real_distribution<Scalar_T> distribution(min, max);
    for (auto& val : vec)
      val = distribution(generator);
  }

  return vec;
}

// making a vector of random values
template<typename Scalar_T>
auto make_random_vector(int N, double min, double max) {
  return generate_random_vec<Scalar_T>(N, min, max);
}

} // namespace sycl_tools

#endif // VECTORS_H
