///////////////////////////////////////////////////////////////////////
// stl
///////////////////////////////////////////////////////////////////////
#include <iostream>

///////////////////////////////////////////////////////////////////////
// sycl
///////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>

namespace sycl_tools {

// selecting a sycl device
auto get_device(int platform_idx, int device_idx) {
  auto platforms = sycl::platform::get_platforms();
  auto devices = platforms[platform_idx].get_devices();

  auto Q = sycl::queue(devices[device_idx]);

  std::cout << "Selected device: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << std::endl;

  return Q;
}

} // namespace sycl_tools