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
#include <iostream>

namespace sycl_tools {

// check if two vectors are equivalent
template<typename Vector_T>
void check_equal(Vector_T a, Vector_T b) {
  for(int i = 0; i < a.size(); ++i) {
    assert(a[i] == b[i]);
  }

  std::cout << "The two vectors are equal!" << std::endl;
}

} // namespace sycl_tools

#endif // VALIDATE_H