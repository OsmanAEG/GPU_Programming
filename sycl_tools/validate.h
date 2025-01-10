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