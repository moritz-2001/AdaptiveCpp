#ifndef RV_H
#define RV_H

#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"
#include <cstdint>

extern "C" bool rv_any(bool);
extern "C" bool rv_all(bool);
extern "C" std::uint32_t rv_ballot(bool);
extern "C" std::uint32_t rv_popcount(bool);
extern "C" std::uint32_t rv_index(bool);
extern "C" bool rv_mask();
extern "C" std::uint32_t rv_lane_id();
extern "C" std::uint32_t rv_num_lanes();

#ifdef RV
static constexpr bool isRV = true;
#else
static constexpr bool isRV = false;
#endif

template <typename T> T rv_shuffle(T, std::int32_t);

template <typename T> T rv_extract(T, std::uint32_t);

template <typename T> T rv_insert(T, std::uint32_t, T);

template <typename T> T rv_reduce(T, int);

enum class ReduceOp {
  NOT_SUPPORTED = -1,
  PLUS = 0,
  MUL = 1,
  MIN = 2,
  MAX = 3,
};
template <typename T> T __cbs_reduce(T, int);

template <typename T> T __cbs_shift_left(T, uint64_t i);

template <typename T> T __cbs_shift_right(T, uint64_t i);

template <typename T> T __cbs_shuffle(T, uint64_t i);

template <typename T> T __cbs_extract(T, uint64_t i);

// TODO REMOVE
#define ALL_INTEGRAL(MACRO)                                                                        \
  MACRO(uint8_t)                                                                                   \
  MACRO(uint16_t)                                                                                  \
  MACRO(uint32_t)                                                                                  \
  MACRO(uint64_t)                                                                                  \
  MACRO(int8_t)                                                                                    \
  MACRO(int16_t)                                                                                   \
  MACRO(int32_t)                                                                                   \
  MACRO(int64_t)                                                                                   \
  MACRO(__acpp_uint8)                                                                              \
  MACRO(__acpp_uint16)                                                                             \
  MACRO(__acpp_uint32)                                                                             \
  MACRO(__acpp_uint64)                                                                             \
  MACRO(__acpp_int8) MACRO(__acpp_int16) MACRO(__acpp_int32) MACRO(__acpp_int64)

#define CBS_EXTRACT(T) T __cbs_extract(T, uint64_t);

// ALL_INTEGRAL(CBS_EXTRACT)

#endif // RV_H
