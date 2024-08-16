/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018,2019 Aksel Alpay
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

#ifndef HIPSYCL_ND_ITEM_HPP
#define HIPSYCL_ND_ITEM_HPP

#include <functional>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "id.hpp"
#include "item.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "multi_ptr.hpp"
#include "group.hpp"
#include "device_event.hpp"
#include "detail/mem_fence.hpp"

#include "detail/thread_hierarchy.hpp"
#include "detail/device_barrier.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {
#ifdef SYCL_DEVICE_ONLY
using host_barrier_type = void;
#else
using host_barrier_type = std::function<void()>;
#endif
}

class handler;

template <int Dimensions = 1>
struct nd_item
{
  /* -- common interface members -- */
  static constexpr int dimensions = Dimensions;
  // TODO

  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_global_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_id<Dimensions>() + (*_offset);
#else
    __acpp_if_target_sscp(return detail::get_global_id<Dimensions>() +
                                        (*_offset););
    return this->_global_id + (*_offset);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_id<Dimensions>(dimension) + _offset->get(dimension);
#else
    __acpp_if_target_sscp(
        return detail::get_global_id<Dimensions>(dimension) +
                          _offset->get(dimension););
    return this->_global_id[dimension] + (*_offset)[dimension];
#endif
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  size_t get_global(int dimension) const
  {
    return this->get_global_id(dimension);
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_global() const
  {
    return this->get_global_id();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_linear_id() const
  {
    __acpp_if_target_sscp(
        return __acpp_sscp_get_global_linear_id<Dimensions>(););

    return detail::linear_id<Dimensions>::get(get_global_id(),
                                              get_global_range());
  }

  HIPSYCL_KERNEL_TARGET friend bool operator ==(const nd_item<Dimensions>& lhs, const nd_item<Dimensions>& rhs)
  {
    // nd_item is not allowed to be shared across work items, so comparison can only be true
    return true;
  }

  HIPSYCL_KERNEL_TARGET friend bool operator !=(const nd_item<Dimensions>& lhs, const nd_item<Dimensions>& rhs)
  {
    return !(lhs==rhs);
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_local() const
  {
    return this->get_local_id();
  }

  [[deprecated]] 
  HIPSYCL_KERNEL_TARGET
  size_t get_local(int dimension) const
  {
    return this->get_local_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_local_id<Dimensions>(dimension););

    return this->_local_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_local_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_local_id<Dimensions>(););
    return this->_local_id;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_id() const
  {
    __acpp_if_target_sscp(
        return __acpp_sscp_get_local_linear_id<Dimensions>(););
    return detail::linear_id<Dimensions>::get(get_local_id(), get_local_range());
  }

  HIPSYCL_KERNEL_TARGET
  group<Dimensions> get_group() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return group<Dimensions>{};
#else
    return group<Dimensions>{
        _group_id,
        _local_range,
        _num_groups,
        static_cast<detail::host_barrier_type *>(_group_barrier),
        get_local_id(),
        _local_memory_ptr,
        _sub_local_memory_ptr,
         get_sub_group()
    };
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_group_id<Dimensions>(dimension););
    return this->_group_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::linear_id<Dimensions>::get(detail::get_group_id<Dimensions>(),
                                              detail::get_grid_size<Dimensions>());
#else
    __acpp_if_target_sscp(
        return __acpp_sscp_get_group_linear_id<Dimensions>(););
    return detail::linear_id<Dimensions>::get(this->_group_id, this->_num_groups);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  sub_group get_sub_group() const
  {
#if USE_RV and not HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
    return sub_group{static_cast<uint32_t>(get_local_linear_id()) / SGSize, (get_local_range().size() + (SGSize-1)) / SGSize, _sub_local_memory_ptr};
#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
    return sub_group{};
#else
    return sub_group{
        static_cast<uint32_t>(get_local_linear_id()) / SGSize,
        (get_local_range().size() + (SGSize-1)) / SGSize,
        _sub_local_memory_ptr,
      _subgroup_id
    };
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_global_size<Dimensions>();)
    return this->_num_groups * this->_local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_global_size<Dimensions>(dimension););
    return this->_num_groups[dimension] * this->_local_range[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_local_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_local_size<Dimensions>(););
    return this->_local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_local_size<Dimensions>(dimension););
    return this->_local_range[dimension];
#endif
  }
  
  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_group_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_grid_size<Dimensions>(););
    return this->_num_groups;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_grid_size<Dimensions>(dimension););
    return this->_num_groups[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_offset() const
  {
    return *_offset;
  }

  HIPSYCL_KERNEL_TARGET
  nd_range<Dimensions> get_nd_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return nd_range<Dimensions>{detail::get_global_size<Dimensions>(),
                                detail::get_local_size<Dimensions>(),
                                get_offset()};
#else
    __acpp_if_target_sscp(return nd_range<Dimensions>{
        detail::get_global_size<Dimensions>(),
        detail::get_local_size<Dimensions>(), get_offset()};);
    
    return nd_range<Dimensions>{
      this->_num_groups * this->_local_range,
      this->_local_range,
      this->get_offset()
    };
#endif
  }

  HIPSYCL_LOOP_SPLIT_BARRIER HIPSYCL_KERNEL_TARGET
  void barrier(access::fence_space space =
      access::fence_space::global_and_local) const
  {
    __acpp_if_target_device(
      detail::local_device_barrier(space);
    );
    __acpp_if_target_host(
        detail::host_barrier_type *barrier =
            static_cast<detail::host_barrier_type *>(_group_barrier);
        (*barrier)();
    );
  }

  template <access::mode accessMode = access::mode::read_write>
  HIPSYCL_KERNEL_TARGET
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    detail::mem_fence<accessMode>(accessSpace);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements) const
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
    return get_group().async_work_group_copy(dest,
                                      src, numElements, srcStride);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
    return get_group().async_work_group_copy(dest, src, numElements, destStride);
  }

  template <typename... eventTN>
  HIPSYCL_KERNEL_TARGET
  void wait_for(eventTN... events) const
  {
    get_group().wait_for(events...);
  }

  
#if defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  HIPSYCL_KERNEL_TARGET
  nd_item(const id<Dimensions>* offset)
    : _offset{offset}
  {}
#else
  HIPSYCL_KERNEL_TARGET
  nd_item(const id<Dimensions>* offset,
          id<Dimensions> group_id, id<Dimensions> local_id, 
          range<Dimensions> local_range, range<Dimensions> num_groups,
          detail::host_barrier_type* host_group_barrier = nullptr,
          void* local_memory_ptr = nullptr,
          void* sub_local_memory_ptr = nullptr,
          size_t subgroup_id = 0
          )
    : _offset{offset},
      _group_id{group_id},
      _local_id{local_id},
      _local_range{local_range},
      _num_groups{num_groups},
      _global_id{group_id * local_range + local_id},
      _local_memory_ptr(local_memory_ptr),
      _sub_local_memory_ptr(sub_local_memory_ptr),
      _subgroup_id(subgroup_id)
  {
    __acpp_if_target_host(
      _group_barrier = static_cast<void*>(host_group_barrier);
    );
  }
#endif

private:
  const id<Dimensions>* _offset;

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  const id<Dimensions> _group_id;
  const id<Dimensions> _local_id;
  const range<Dimensions> _local_range;
  const range<Dimensions> _num_groups;
  const id<Dimensions> _global_id;
  void *_local_memory_ptr;
  void *_sub_local_memory_ptr;
  size_t _subgroup_id;
#endif

#ifndef SYCL_DEVICE_ONLY
  // Store void ptr to avoid function pointer types
  // appearing in SSCP code
  void* _group_barrier;
#endif
};

} // namespace sycl
} // namespace hipsycl

#endif
