#ifndef VALIDATE_H
#define VALIDATE_H

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
///        to validate code examples.
///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <assert.h>
#include <cmath>
#include <iostream>

namespace sycl_tools {

// check if two values are equivalent
template<typename Scalar_T>
void check_equal_value(Scalar_T a, Scalar_T b, double epsilon = 1.0e-6) {
  if constexpr (std::is_floating_point_v<Scalar_T>) {
    assert(std::fabs(a - b) <= epsilon);
  } else {
    assert(a == b);
  }

  std::cout << "The two values are equal!" << std::endl;
}

// check if two vectors are equivalent
template<typename Vector_T>
void check_equal_vector(Vector_T a, Vector_T b, double epsilon = 1.0e-6) {
  for (int i = 0; i < a.size(); ++i) {
    if constexpr (std::is_floating_point_v<typename Vector_T::value_type>) {
      assert(std::fabs(a[i] - b[i]) <= epsilon);
    } else {
      assert(a[i] == b[i]);
    }
  }

  std::cout << "The two vectors are equal!" << std::endl;
}

} // namespace sycl_tools

#endif // VALIDATE_H
