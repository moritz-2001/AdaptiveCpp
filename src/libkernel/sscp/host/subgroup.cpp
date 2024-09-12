/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/sycl/libkernel/host/rv.h"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/RV.h"
//
extern "C" size_t __hipsycl_cbs_local_id_subgroup;

// TODO RV

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_local_id() {
#if USE_RV
  return rv_lane_id();
#else
  return __hipsycl_cbs_local_id_subgroup;
#endif
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_size() {
#if USE_RV
  return __acpp_sscp_get_subgroup_max_size();
#else
  if (const bool lastGroup = __acpp_sscp_get_num_subgroups() - 1 == __acpp_sscp_get_subgroup_id();
      not lastGroup) {
    return __acpp_sscp_get_subgroup_max_size();
  }
  const auto wg_size = __acpp_sscp_get_local_size_x() * __acpp_sscp_get_local_size_y() *
                       __acpp_sscp_get_local_size_z();

  return wg_size - (__acpp_sscp_get_num_subgroups() - 1) * __acpp_sscp_get_subgroup_max_size();
#endif
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_max_size() {
  return hipsycl::compiler::SGSize;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_id() {
  const size_t local_tid = __acpp_sscp_get_local_id_x() +
                           __acpp_sscp_get_local_id_y() *
                               (__acpp_sscp_get_local_size_x() +
                                __acpp_sscp_get_local_id_z() * __acpp_sscp_get_local_size_x());

  const auto res = local_tid / __acpp_sscp_get_subgroup_max_size();

  #if USE_RV
    return rv_is_uniform(res);
  #else
    return res;
  #endif
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups() {
  const auto wg_size = __acpp_sscp_get_local_size_x() * __acpp_sscp_get_local_size_y() *
                       __acpp_sscp_get_local_size_z();
  // Round up
  return (wg_size + __acpp_sscp_get_subgroup_max_size() - 1) / __acpp_sscp_get_subgroup_max_size();
}
