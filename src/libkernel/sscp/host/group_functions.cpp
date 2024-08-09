#include "../../../../include/hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/host/host.h"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sub_group.hpp"

#include "hipSYCL/RV.h"
#include "hipSYCL/sycl/libkernel/host/cbs_intrinsics.h"
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "hipSYCL/sycl/libkernel/host/rv_shuffle.h"

// TODO use the correct functions
extern "C" [[clang::convergent]] void __acpp_cbs_sub_barrier();
extern "C" [[clang::convergent]] void __acpp_cbs_barrier();

extern "C" size_t __acpp_cbs_local_id_x;
extern "C" size_t __acpp_cbs_local_id_y;
extern "C" size_t __acpp_cbs_local_id_z;

extern "C" size_t __acpp_cbs_local_size_x;
extern "C" size_t __acpp_cbs_local_size_y;
extern "C" size_t __acpp_cbs_local_size_z;

// TODO CBS AND RV REDUCE MISSING OPS

// TODO for scan no impls in header?

// TODO SYCL incomplete sub-groups? CBS INTRINSICS support

// TODO shift_right 0 or group end ?

// TODO Implement reduction for floats


size_t get_local_linear_id() {
  size_t lid_x = __acpp_cbs_local_id_x;
  size_t lid_y = __acpp_cbs_local_id_z;
  size_t lid_z = __acpp_cbs_local_id_y;

  size_t lsize_x = __acpp_cbs_local_size_x;
  size_t lsize_y = __acpp_cbs_local_size_y;

  return lsize_x * lsize_y * lid_z + lsize_x * lid_y + lid_x;
}

size_t get_local_size() {
  size_t size_x = __acpp_cbs_local_size_x;
  size_t size_y = __acpp_cbs_local_size_y;
  size_t size_z = __acpp_cbs_local_size_z;

  return size_x * size_y * size_z;
}

#define ALL_VARIANTS(MACRO)                                                                        \
  MACRO(sub, int8, i8)                                                                             \
  MACRO(sub, int16, i16)                                                                           \
  MACRO(sub, int32, i32)                                                                           \
  MACRO(sub, int64, i64)                                                                           \
  MACRO(sub, uint8, u8)                                                                            \
  MACRO(sub, uint16, u16)                                                                          \
  MACRO(sub, uint32, u32)                                                                          \
  MACRO(sub, uint64, u64)                                                                          \
  MACRO(work, int8, i8)                                                                            \
  MACRO(work, int16, i16) MACRO(work, int32, i32) MACRO(work, int64, i64) MACRO(work, uint8, u8)   \
      MACRO(work, uint16, u16) MACRO(work, uint32, u32) MACRO(work, uint64, u64)

template <typename T> T work_broadcast(const int sender, T x) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  if (const size_t local_linear_id = get_local_linear_id(); sender == local_linear_id) {
    scratch[0] = x;
  }

  __acpp_cbs_barrier();
  T tmp = scratch[0];
  __acpp_cbs_barrier();

  return tmp;
}

template <typename T> T sub_broadcast(const int sender, T x) {
#if USE_RV
  return hipsycl::sycl::detail::extract_impl<T>(x, sender);
#else
  auto e = static_cast<uint64_t>(sender);
  __acpp_cbs_sub_barrier();
  auto t = __cbs_extract(x, e);
  __acpp_cbs_sub_barrier();
  return t;
#endif
}

constexpr ReduceOp reduce_op_map(__acpp_sscp_algorithm_op op) {
  switch (op) {
  case __acpp_sscp_algorithm_op::plus:
    return ReduceOp::PLUS;
  case __acpp_sscp_algorithm_op::multiply:
    return ReduceOp::MUL;
  case __acpp_sscp_algorithm_op::min:
    return ReduceOp::MIN;
  case __acpp_sscp_algorithm_op::max:
    return ReduceOp::MAX;
  case __acpp_sscp_algorithm_op::bit_and:
  case __acpp_sscp_algorithm_op::bit_or:
  case __acpp_sscp_algorithm_op::bit_xor:
  case __acpp_sscp_algorithm_op::logical_and:
  case __acpp_sscp_algorithm_op::logical_or:
    break;
  }
  return ReduceOp::NOT_SUPPORTED;
}

template <typename T> constexpr T binary_op(__acpp_sscp_algorithm_op op, T x, T y) {
  switch (op) {
  case __acpp_sscp_algorithm_op::plus:
    return x + y;
  case __acpp_sscp_algorithm_op::multiply:
    return x * y;
  case __acpp_sscp_algorithm_op::min:
    return std::min(x, y);
  case __acpp_sscp_algorithm_op::max:
    return std::max(x, y);
  case __acpp_sscp_algorithm_op::bit_and:
    return x & y;
  case __acpp_sscp_algorithm_op::bit_or:
    return x | y;
  case __acpp_sscp_algorithm_op::bit_xor:
    return x ^ y;
  case __acpp_sscp_algorithm_op::logical_and:
    return x and y;
  case __acpp_sscp_algorithm_op::logical_or:
    return x or y;
  }
  assert(false);
  return {};
}

template <typename T> T sub_reduce(__acpp_sscp_algorithm_op op, T x) {
  ReduceOp operation = reduce_op_map(op);
  if (operation == ReduceOp::NOT_SUPPORTED) {
    assert(false);
  }
#if USE_RV
  return rv_reduce(x, static_cast<int>(operation));
#else
  __acpp_cbs_sub_barrier();
  const T t = __cbs_reduce(x, static_cast<int>(operation));
  __acpp_cbs_sub_barrier();
  return t;
#endif
}

template <typename T> T work_reduce(__acpp_sscp_algorithm_op op, T x) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = get_local_linear_id();
  const auto local_range = get_local_size();

  T result = sub_reduce(op, x);

  // Collect results from sub-groups
  if (__acpp_sscp_get_subgroup_local_id() == 0) {
    // Use sg-id
    scratch[lid] = result;
  }

  __acpp_cbs_barrier(); // as this starts a new loop with CBS,
  // LLVM detects it's actually just one iteration and optimizes it

  // First work-item does reduction on the results of the sub-group reductions
  if (lid == 0) {
    T y{};
    for (uint32_t j = 0u; j < local_range; j += __acpp_sscp_get_subgroup_size()) {
      y = binary_op(op, y, scratch[j]);
    }
    scratch[0] = y;
  }

  __acpp_cbs_barrier();
  T endResult = scratch[0];
  __acpp_cbs_barrier();

  return endResult;
}

template <typename T> T sub_shift_left(T x, __acpp_uint32 delta) {
#if USE_RV
  return hipsycl::sycl::detail::shuffle_down_impl<T>(x, static_cast<int>(delta));
#else

  T *scratch = static_cast<T *>(sub_group_shared_memory);
  auto lid = __acpp_sscp_get_subgroup_local_id();
  auto target_lid = lid + delta;
  scratch[lid] = x;
  __acpp_cbs_sub_barrier();

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid >= hipsycl::compiler::SGSize)
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_cbs_sub_barrier();
  return x;

#endif
}

template <typename T> T work_shift_left(T x, __acpp_uint32 delta) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = get_local_linear_id();
  const auto local_range = get_local_size();
  auto target_lid = lid + delta;

  scratch[lid] = x;
  __acpp_cbs_barrier();

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid >= local_range)
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_cbs_barrier();

  return x;
}

template <typename T> T sub_shift_right(T x, __acpp_uint32 delta) {
#if USE_RV
  return hipsycl::sycl::detail::shuffle_up_impl(x, delta);
#else
  T *scratch = static_cast<T *>(sub_group_shared_memory);

  const auto lid = __acpp_sscp_get_subgroup_local_id();
  const auto local_range = lid + delta;
  auto target_lid = lid - delta;

  scratch[lid] = x;
  __acpp_cbs_sub_barrier();

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid >= hipsycl::compiler::SGSize || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_cbs_sub_barrier();
  return x;
  /*
  __acpp_cbs_sub_barrier();
  const auto pos =
      __acpp_sscp_get_subgroup_local_id() - delta >= __acpp_sscp_get_subgroup_max_size()
          ? 0
          : __acpp_sscp_get_subgroup_local_id() - delta;
  auto tmp = __cbs_shuffle(x, pos);
  __acpp_cbs_sub_barrier();
  return tmp;
  */
#endif
}

template <typename T> T work_shift_right(T x, __acpp_uint32 delta) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = get_local_linear_id();
  const auto local_range = get_local_size();
  auto target_lid = lid - delta;

  scratch[lid] = x;
  __acpp_cbs_barrier();

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid >= local_range || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  __acpp_cbs_barrier();

  return x;
}

template <typename T> T sub_select(T x, __acpp_uint32 delta) {
#if USE_RV
  return hipsycl::sycl::detail::shuffle_impl(x, delta);
#else
  T *scratch = static_cast<T *>(sub_group_shared_memory);
  auto lid = __acpp_sscp_get_subgroup_local_id();
  scratch[lid] = x;
  __acpp_cbs_sub_barrier();
  x = scratch[delta];
  __acpp_cbs_sub_barrier();
  return x;
#endif
}

template <typename T> T work_select(T x, __acpp_uint32 delta) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = get_local_linear_id();

  scratch[lid] = x;
  __acpp_cbs_barrier();
  x = scratch[delta];
  __acpp_cbs_barrier();

  return x;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_any(bool pred) {
  // TODO CHECK
  return work_reduce(__acpp_sscp_algorithm_op::max, static_cast<uint8_t>(pred)) > 0;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred) {
#if USE_RV
  return rv_any(pred);
#else
  auto v = static_cast<uint8_t>(pred);
  __acpp_cbs_sub_barrier();
  const auto t = __cbs_reduce(v, static_cast<int>(ReduceOp::MAX)) > 0;
  __acpp_cbs_sub_barrier();
  return t;
#endif
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool pred) {
  return work_reduce(__acpp_sscp_algorithm_op::min, static_cast<uint8_t>(pred)) > 0;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred) {
#if USE_RV
  return rv_all(pred);
#else
  auto v = static_cast<uint8_t>(pred);
  __acpp_cbs_sub_barrier();
  const auto t = __cbs_reduce(v, static_cast<int>(ReduceOp::MIN)) > 0;
  __acpp_cbs_sub_barrier();
  return t;
#endif
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool pred) { return __acpp_sscp_work_group_all(not pred); }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred) { return __acpp_sscp_sub_group_all(not pred); }

// TODO floats
#define REDUCE(LEVEL, T, TNAME)                                                                    \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_##T __acpp_sscp_##LEVEL##_group_reduce_##TNAME(           \
      __acpp_sscp_algorithm_op op, __acpp_##T x) {                                                 \
    return LEVEL##_reduce(op, x);                                                                  \
  };

#define BROADCAST(LEVEL, T, TNAME)                                                                 \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_##T __acpp_sscp_##LEVEL##_group_broadcast_##TNAME(        \
      __acpp_int32 sender, __acpp_##T x) {                                                        \
    return LEVEL##_broadcast(sender, x);                                                           \
  };

#define SHIFT_LEFT(LEVEL, T, TNAME)                                                                \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_##T __acpp_sscp_##LEVEL##_group_shl_##TNAME(              \
      __acpp_##T x, __acpp_uint32 delta) {                                                         \
    return LEVEL##_shift_left(x, delta);                                                           \
  };

#define SHIFT_RIGHT(LEVEL, T, TNAME)                                                               \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_##T __acpp_sscp_##LEVEL##_group_shr_##TNAME(              \
      __acpp_##T x, __acpp_uint32 delta) {                                                         \
    return LEVEL##_shift_right(x, delta);                                                          \
  };

// TODO PERMUTE

#define SELECT(LEVEL, T, TNAME)                                                                    \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_##T __acpp_sscp_##LEVEL##_group_select_##TNAME(           \
      __acpp_##T x, __acpp_uint32 delta) {                                                         \
    return LEVEL##_select(x, delta);                                                               \
  };

ALL_VARIANTS(BROADCAST)
ALL_VARIANTS(REDUCE)
ALL_VARIANTS(SHIFT_LEFT)
ALL_VARIANTS(SHIFT_RIGHT)
ALL_VARIANTS(SELECT)

// TODO SCAN