#ifndef DEVICES_H
#define DEVICES_H

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
///        related to sycl devices.
///////////////////////////////////////////////////////////////////////

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
  auto devices   = platforms[platform_idx].get_devices();

  auto Q = sycl::queue(devices[device_idx]);

  std::cout << "Selected device: " << Q.get_device().template get_info<sycl::info::device::name>()
            << std::endl;

  return Q;
}

}  // namespace sycl_tools

#endif  // DEVICES_H