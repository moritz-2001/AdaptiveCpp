/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_SUBGROUP_HPP
#define HIPSYCL_SUBGROUP_HPP

#include <cstdint>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "detail/thread_hierarchy.hpp"
#include "id.hpp"
#include "range.hpp"
#include "memory.hpp"

#ifdef HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "sscp/builtins/subgroup.hpp"
#endif

#define HIERACHICAL
//#define RV

#if defined(RV)
#include "host/rv.h"
#endif

constexpr size_t SGSize = 32;

namespace hipsycl::sycl {

#if defined(HIERACHICAL) and not defined(RV)
class sub_group {
public:
  using id_type = sycl::id<1>;
  using range_type = sycl::range<1>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;

  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  explicit sub_group(size_t group_id, size_t num_subgroups, void* local_memory, size_t subgroup_id) :
  _group_id(group_id), _num_subgroups(num_subgroups), _local_memory(local_memory), _subgroup_id(subgroup_id) {}

  // Only exists to get code compiling
  explicit sub_group() : _group_id(0), _num_subgroups(0), _local_memory(nullptr) {
    assert(false);
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const {
    return id_type{get_local_linear_id()};
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const {
    return _subgroup_id;
  }

  // always returns the maximum sub_group size
  HIPSYCL_KERNEL_TARGET
  range_type get_local_range() const {
    return range_type{get_local_linear_range()};
  }

  // always returns the maximum sub_group size
  HIPSYCL_KERNEL_TARGET
  linear_range_type get_local_linear_range() const {
    return SGSize; // TODO wrong for incomplete subgroups
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_max_local_range() const {
    return range_type{SGSize};
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_group_id() const {
    return id_type{get_group_linear_id()};
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_group_linear_id() const {
    return _group_id;
  }

  HIPSYCL_KERNEL_TARGET
  linear_range_type get_group_linear_range() const {
    return _num_subgroups;
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_group_range() const {
    return range_type{get_group_linear_range()};
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  range_type get_max_group_range() const {
    return get_group_range();
  }

  HIPSYCL_KERNEL_TARGET
  bool leader() const {
    return get_local_linear_id() == 0;
  }

  HIPSYCL_KERNEL_TARGET
  void *get_local_memory_ptr() const {
    // TODO not offically supported
    return _local_memory;
  }
private:
  size_t _group_id;
  size_t _num_subgroups;
  void* _local_memory;
  size_t _subgroup_id;

};

#elif defined(RV)
class sub_group {
public:
  using id_type = sycl::id<1>;
  using range_type = sycl::range<1>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;

  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  sub_group() {}

  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const {
    return id_type{get_local_linear_id()};
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const {
    return rv_lane_id();
  }

  // always returns the maximum sub_group size
  HIPSYCL_KERNEL_TARGET
  range_type get_local_range() const {
    return range_type{get_local_linear_range()};
  }

  // always returns the maximum sub_group size
  HIPSYCL_KERNEL_TARGET
  linear_range_type get_local_linear_range() const {
    return rv_num_lanes(); //  TODO or SGSize?
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_max_local_range() const {
    return range_type{get_local_linear_range()};
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_group_id() const {
    return id_type{get_group_linear_id()};
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_group_linear_id() const {
    assert(false); // TODO implement
  }

  HIPSYCL_KERNEL_TARGET
  linear_range_type get_group_linear_range() const {
    // TODO WRONG
    assert(false);
    //return rv_num_lanes();
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_group_range() const {
    return range_type{get_group_linear_range()};
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  range_type get_max_group_range() const {
    return get_group_range();
  }

  HIPSYCL_KERNEL_TARGET
  bool leader() const {
    return rv_lane_id() == 0;
  }

  HIPSYCL_KERNEL_TARGET
  void *get_local_memory_ptr() const {
    return _local_memory;
  }
private:
  void* _local_memory;
};

#else
class sub_group {
public:
  using id_type = sycl::id<1>;
  using range_type = sycl::range<1>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;

  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;


  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const {
    return id_type{get_local_linear_id()};
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const {
    __acpp_backend_switch(
        return 0,
        return __acpp_sscp_get_subgroup_local_id(),
        return local_tid() & get_warp_mask(),
        return local_tid() & get_warp_mask());
  }

  // always returns the maximum sub_group size
  HIPSYCL_KERNEL_TARGET
  range_type get_local_range() const {
    return range_type{get_local_linear_range()};
  }

  // always returns the maximum sub_group size
  HIPSYCL_KERNEL_TARGET
  linear_range_type get_local_linear_range() const {
    __acpp_backend_switch(
        return 1,
        return __acpp_sscp_get_subgroup_size(),
        // TODO This is not actually correct for incomplete subgroups
        return __acpp_warp_size,
        return __acpp_warp_size);
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_max_local_range() const {
    __acpp_backend_switch(
        return range_type{1},
        return range_type{__acpp_sscp_get_subgroup_max_size()},
        return range_type{__acpp_warp_size},
        return range_type{__acpp_warp_size});
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_group_id() const {
    return id_type{get_group_linear_id()};
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_group_linear_id() const {
    __acpp_backend_switch(
        return 0, // TODO This is probably incorrect
        return __acpp_sscp_get_subgroup_id(),
        return local_tid() >> (__ffs(__acpp_warp_size) - 1),
        return local_tid() >> (__ffs(__acpp_warp_size) - 1));
  }

  HIPSYCL_KERNEL_TARGET
  linear_range_type get_group_linear_range() const {
    __acpp_backend_switch(
        return 1,
        return __acpp_sscp_get_num_subgroups(),
        return hiplike_num_subgroups(),
        return hiplike_num_subgroups());
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_group_range() const {
    return range_type{get_group_linear_range()};
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  range_type get_max_group_range() const {
    return get_group_range();
  }

  HIPSYCL_KERNEL_TARGET
  bool leader() const {
    return get_local_linear_id() == 0;
  }

#if !HIPSYCL_LIBKERNEL_IS_DEVICE_PASS
  HIPSYCL_KERNEL_TARGET
  void *get_local_memory_ptr() const {
    return _local_memory_ptr;
  }
#endif

private:
  int hiplike_num_subgroups() const {
    __acpp_if_target_hiplike(
        int local_range =
            __acpp_lsize_x * __acpp_lsize_y * __acpp_lsize_z;
        return (local_range + __acpp_warp_size - 1) / __acpp_warp_size;
    );
    return 0;
  }

  HIPSYCL_KERNEL_TARGET
  int local_tid() const {
    __acpp_if_target_device(
      int tid = __acpp_lid_x
              + __acpp_lid_y * __acpp_lsize_x
              + __acpp_lid_z * __acpp_lsize_x * __acpp_lsize_y;
      return tid;
    );
    return 0;
  }

  HIPSYCL_KERNEL_TARGET
  int get_warp_mask() const {
    // Assumes that __acpp_warp_size is a power of two
    __acpp_if_target_hiplike(
      return __acpp_warp_size - 1;
    );
    return 0;
  }

};
#endif

}


#endif