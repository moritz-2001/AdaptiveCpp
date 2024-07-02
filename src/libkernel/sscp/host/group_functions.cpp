#include "hipSYCL/sycl/libkernel/host/rv_shuffle.h"
#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/host/host.h"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sub_group.hpp"

#include <cassert>
#include <cstdint>
#include <type_traits>

#define RV

extern "C" [[clang::convergent]] void __acpp_cbs_sub_barrier();
extern "C" [[clang::convergent]] void __acpp_cbs_barrier();

// TODO TEST RV

// TODO acpp switch maps to host in generic case when using sub_grouo{}.get_local_linear_id()

// TODO problems with intrinsics (host code is still generated?)
// => linker error when compiling with generic

// TODO shift left => T trivally copyable ? , ....

// TODO core function: purpose of DIM?

// TODO CBS AND RV REDUCE MISSING OPS

// TODO ANY and ALL IMPS do not work (Also, fix them in host/group_functions)

// TODO for scan no impls in header?

// TODO SYCL incomplete sub-groups?
// -> TEST WITHOUT INTRINSICS; ATM intrinsics do not support incomplete sub-groups

// TODO why do I need this function impl; Shouln't this function be created by the optimizations
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_if_global_sizes_fit_in_int() { return false; }

#define ALL(MACRO)                                                                                 \
  MACRO(sub, int8, i8)                                                                             \
  MACRO(sub, int16, i16)                                                                           \
  MACRO(sub, int32, i32)                                                                           \
  MACRO(sub, int64, i64)                                                                           \
  MACRO(work, int8, i8) MACRO(work, int16, i16) MACRO(work, int32, i32) MACRO(work, int64, i64)

template <typename T> T work_broadcast(const int sender, T x) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  if (const size_t local_linear_id = __acpp_sscp_get_local_linear_id<3>();
      sender == local_linear_id) {
    scratch[0] = x;
  }

  __acpp_cbs_barrier();
  T tmp = scratch[0];
  __acpp_cbs_barrier();

  return tmp;
}

template <typename T> T sub_broadcast(const int sender, T x) {
  __acpp_cbs_sub_barrier();
  auto t = __cbs_extract(x, static_cast<uint64_t>(sender));
  __acpp_cbs_sub_barrier();
  return t;
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
#ifdef RV
  return rv_reduce(x, static_cast<int>(operation));
#else
  __acpp_cbs_sub_barrier();
  const auto t = __cbs_reduce(x, static_cast<int>(operation));
  __acpp_cbs_sub_barrier();
  return t;
#endif
}

template <typename T> T work_reduce(__acpp_sscp_algorithm_op op, T x) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = __acpp_sscp_get_local_linear_id<3>();
  const auto local_range = __acpp_sscp_get_local_size<3>();

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
#ifdef RV
  return hipsycl::sycl::detail::shuffle_down_impl<T>(x, static_cast<int>(delta));
#else
  __acpp_cbs_sub_barrier();
  const auto pos = __acpp_sscp_get_subgroup_local_id() + delta >= 32
                       ? 0
                       : __acpp_sscp_get_subgroup_local_id() + delta;
  auto tmp = __cbs_shuffle(x, pos);
  __acpp_cbs_sub_barrier();
  return tmp;
#endif
}

template <typename T> T work_shift_left(T x, __acpp_uint32 delta) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = __acpp_sscp_get_local_linear_id<3>();
  const auto local_range = __acpp_sscp_get_local_size<3>();
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
#ifdef RV
  return hipsycl::sycl::detail::shuffle_up_impl(x, delta);
#else
  __acpp_cbs_sub_barrier();
  const auto pos =
      __acpp_sscp_get_subgroup_local_id() - delta >= __acpp_sscp_get_subgroup_max_size()
          ? 0
          : __acpp_sscp_get_subgroup_local_id() - delta;
  auto tmp = __cbs_shuffle(x, pos);
  __acpp_cbs_sub_barrier();
  return tmp;
#endif
}

template <typename T> T work_shift_right(T x, __acpp_uint32 delta) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = __acpp_sscp_get_local_linear_id<3>();
  const auto local_range = __acpp_sscp_get_local_size<3>();
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
#ifdef RV
  return hipsycl::sycl::detail::shuffle_impl(x, delta);
#else
  __acpp_cbs_sub_barrier();
  auto tmp = __cbs_shuffle(x, delta);
  __acpp_cbs_sub_barrier();
  return tmp;
#endif
}

template <typename T> T work_select(T x, __acpp_uint32 delta) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = __acpp_sscp_get_local_linear_id<3>();

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
#ifdef RV
  return rv_any(pred);
#else
  __acpp_cbs_sub_barrier();
  const auto t = __cbs_reduce(static_cast<uint8_t>(pred), static_cast<int>(ReduceOp::MAX)) > 0;
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
#ifdef RV
  return rv_all(pred);
#else
  __acpp_cbs_sub_barrier();
  const auto t = __cbs_reduce(static_cast<uint8_t>(pred), static_cast<int>(ReduceOp::MIN)) > 0;
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
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_## T __acpp_sscp_## LEVEL## _group_reduce_## TNAME(           \
      __acpp_sscp_algorithm_op op, __acpp_## T x) {                                                 \
    return LEVEL## _reduce(op, x);                                                                  \
  };

#define BROADCAST(LEVEL, T, TNAME)                                                                 \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_## T __acpp_sscp_## LEVEL## _group_broadcast_## TNAME(        \
      __acpp_int32 sender, __acpp_## T x) {                                                         \
    return LEVEL## _broadcast(sender, x);                                                           \
  };

#define SHIFT_LEFT(LEVEL, T, TNAME)                                                                \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_## T __acpp_sscp_## LEVEL## _group_shl_## TNAME(              \
      __acpp_## T x, __acpp_uint32 delta) {                                                         \
    return LEVEL## _shift_left(x, delta);                                                           \
  };

#define SHIFT_RIGHT(LEVEL, T, TNAME)                                                               \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_## T __acpp_sscp_## LEVEL## _group_shr_## TNAME(              \
      __acpp_## T x, __acpp_uint32 delta) {                                                         \
    return LEVEL## _shift_right(x, delta);                                                          \
  };

// TODO PERMUTE

#define SELECT(LEVEL, T, TNAME)                                                                    \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_## T __acpp_sscp_## LEVEL## _group_select_## TNAME(           \
      __acpp_## T x, __acpp_uint32 delta) {                                                         \
    return LEVEL## _select(x, delta);                                                               \
  };

ALL(BROADCAST)
ALL(REDUCE)
ALL(SHIFT_LEFT)
ALL(SELECT)

// TODO SCAN