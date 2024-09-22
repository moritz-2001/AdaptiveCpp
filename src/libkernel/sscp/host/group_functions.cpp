#include "../../../../include/hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sub_group.hpp"
#include <cassert>
#include <cstdint>
#include <type_traits>

// TODO use the correct functions
extern "C" [[clang::convergent]] void __acpp_cbs_barrier();
extern "C" void* work_group_shared_memory;

extern "C" size_t __acpp_cbs_local_id_x;
extern "C" size_t __acpp_cbs_local_id_y;
extern "C" size_t __acpp_cbs_local_id_z;

extern "C" size_t __acpp_cbs_local_size_x;
extern "C" size_t __acpp_cbs_local_size_y;
extern "C" size_t __acpp_cbs_local_size_z;

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

#define ALL_F(MACRO) \
	MACRO(sub, f16, f16) \
MACRO(sub, f32, f32) \
MACRO(sub, f64, f64) \
MACRO(work, f16, f16) \
MACRO(work, f32, f32) \
MACRO(work, f64, f64)

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
	return x;
}

template <typename T> T sub_shift_left(T x, __acpp_uint32 delta) {
	return x;
}

enum class ReduceOp {
  NOT_SUPPORTED = -1,
  PLUS = 0,
  MUL = 1,
  MIN = 2,
  MAX = 3,
  BIT_AND = 4,
  BIT_OR = 5,
  BIT_XOR = 6,
};

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

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true> constexpr T binary_op(__acpp_sscp_algorithm_op op, T x, T y) {
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

template <typename T, std::enable_if_t<not std::is_integral_v<T>, bool> = true> constexpr T binary_op(__acpp_sscp_algorithm_op op, T x, T y) {
  switch (op) {
  case __acpp_sscp_algorithm_op::plus:
    return x + y;
  case __acpp_sscp_algorithm_op::multiply:
    return x * y;
  case __acpp_sscp_algorithm_op::min:
    return std::min(x, y);
  case __acpp_sscp_algorithm_op::max:
    return std::max(x, y);
  case __acpp_sscp_algorithm_op::logical_and:
    return x and y;
  case __acpp_sscp_algorithm_op::logical_or:
    return x or y;
  case __acpp_sscp_algorithm_op::bit_and:
  case __acpp_sscp_algorithm_op::bit_or:
  case __acpp_sscp_algorithm_op::bit_xor:
  }
  assert(false);
  return {};
}

template <typename T> T sub_reduce(__acpp_sscp_algorithm_op op, T x) {
  return x;
}

template <typename T> T work_reduce(__acpp_sscp_algorithm_op op, T x) {
  T *scratch = static_cast<T *>(work_group_shared_memory);

  const auto lid = get_local_linear_id();
  scratch[lid] = x;

  __acpp_cbs_barrier(); // as this starts a new loop with CBS,
  // LLVM detects it's actually just one iteration and optimizes it

  // First work-item does reduction on the results of the sub-group reductions
  if (lid == 0) {
    T y = scratch[0];
    for (uint32_t j = 1u; j < get_local_size(); j += 1) {
      y = binary_op(op, y, scratch[j]);
    }
    scratch[0] = y;
  }

  __acpp_cbs_barrier();
  T endResult = scratch[0];
  __acpp_cbs_barrier();

  return endResult;
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
	return x;
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
	return x;
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
  return work_reduce(__acpp_sscp_algorithm_op::max, static_cast<uint8_t>(pred)) > 0;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred) {
	return pred;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool pred) {
  return work_reduce(__acpp_sscp_algorithm_op::min, static_cast<uint8_t>(pred)) > 0;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred) {
	return pred;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool pred) { return __acpp_sscp_work_group_all(not pred); }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred) { return __acpp_sscp_sub_group_all(not pred); }


template <typename T> T sub_inclusive_scan(__acpp_sscp_algorithm_op op, T x) {
	return x;
}

template <typename T> T work_inclusive_scan(__acpp_sscp_algorithm_op op, T x) {
  T *scratch = static_cast<T *>(work_group_shared_memory);
  size_t lid = get_local_linear_id();
  scratch[lid] = x;

  __acpp_cbs_barrier();
  if (get_local_linear_id() == 0) {
    for (auto i = 1ul; i < get_local_size(); ++i) {
      scratch[i] = binary_op(op, scratch[i - 1], scratch[i]);
    }
  }
  __acpp_cbs_barrier();
  const auto res = scratch[lid];
  __acpp_cbs_barrier();
  return res;
}



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

#define INCLUSIVE_SCAN(LEVEL, T, TNAME)                                                                    \
HIPSYCL_SSCP_CONVERGENT_BUILTIN __acpp_##T __acpp_sscp_##LEVEL##_group_inclusive_scan_##TNAME(           \
__acpp_sscp_algorithm_op op, __acpp_##T x) {                                                 \
return LEVEL##_inclusive_scan(op, x);                                                                  \
};

ALL_VARIANTS(BROADCAST)
ALL_VARIANTS(REDUCE)
ALL_VARIANTS(SHIFT_LEFT)
ALL_VARIANTS(SHIFT_RIGHT)
ALL_VARIANTS(SELECT)
ALL_VARIANTS(INCLUSIVE_SCAN)


ALL_F(INCLUSIVE_SCAN)
ALL_F(REDUCE)