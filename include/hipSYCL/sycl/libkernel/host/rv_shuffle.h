#ifndef RV_SHUFFLE_H
#define RV_SHUFFLE_H

#include "rv.h"

namespace hipsycl {
namespace sycl {

namespace detail {

// template <class T>
// T intrin_extract(T, std::uint32_t);

#define MANGLED_VARIANTS(TYPE, BackupType, MangleSuffix)                                   \
  HIPSYCL_FORCE_INLINE TYPE intrin_extract(TYPE vec, std::uint32_t idx) {                           \
    return static_cast<TYPE>(rv_extract_##MangleSuffix(static_cast<BackupType>(vec), idx));                                \
  }                                                                            \
  HIPSYCL_FORCE_INLINE TYPE intrin_insert(TYPE vec, std::uint32_t idx, TYPE val) {                  \
    return static_cast<TYPE>(rv_insert_##MangleSuffix(static_cast<BackupType>(vec), idx, val));                            \
  }

MANGLED_VARIANTS(float, float, f)
MANGLED_VARIANTS(double, double, d)
MANGLED_VARIANTS(std::int8_t, std::int8_t, i8)
MANGLED_VARIANTS(std::int16_t, std::int16_t, i16)
MANGLED_VARIANTS(std::int32_t, std::int32_t, i32)
MANGLED_VARIANTS(std::int64_t, std::int64_t, i64)
MANGLED_VARIANTS(long long, std::int64_t, i64)
MANGLED_VARIANTS(std::uint8_t, std::int8_t, i8)
MANGLED_VARIANTS(std::uint16_t, std::int16_t, i16)
MANGLED_VARIANTS(std::uint32_t, std::int32_t, i32)
MANGLED_VARIANTS(std::uint64_t, std::int64_t, i64)
MANGLED_VARIANTS(unsigned long long, std::int64_t, i64)

template <size_t SizeOfT>
void copy_bits(std::uint8_t *tgtPtr, const std::uint8_t *ptr) {
#pragma unroll
  for (int i = 0; i < SizeOfT; ++i)
    tgtPtr[i] = ptr[i];
}

template <typename T, size_t Words>
void copy_bits(std::array<float, Words> &words, T &&x) {
  copy_bits<sizeof(T)>(reinterpret_cast<std::uint8_t *>(words.data()),
                       reinterpret_cast<std::uint8_t *>(&x));
}

template <typename T, size_t Words>
void copy_bits(T &tgt, const std::array<float, Words> &words) {
  copy_bits<sizeof(T)>(reinterpret_cast<std::uint8_t *>(&tgt),
                       reinterpret_cast<const std::uint8_t *>(words.data()));
}

template <typename T, typename Operation> T apply_on_data(T x, Operation &&op) {
  // roundUp(sizeof(T) / sizeof(float))
  constexpr std::size_t words_no =
      (sizeof(T) + sizeof(float) - 1) / sizeof(float);

  std::array<float, words_no> words;
  copy_bits(words, x);

  for (int i = 0; i < words_no; i++)
    words[i] = std::forward<Operation>(op)(words[i]);

  T output;
  copy_bits(output, words);

  return output;
}

// implemented based on warp_shuffle_op in rocPRIM

// difference between shuffle_impl and extract_impl: id for extract must be
// uniform value.
template <typename T> HIPSYCL_FORCE_INLINE T shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](const float data) {
    float ret = data;
#pragma unroll
    for (int i = 0; i < rv_num_lanes(); ++i) {
      const int srcLane = rv_extract(id, i);
      const float v = rv_extract(data, srcLane);
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}

template <typename T> HIPSYCL_FORCE_INLINE T extract_impl(T x, int id) {
  return apply_on_data(x,
                       [id](const float data) { return rv_extract(data, id); });
}

template <typename T> HIPSYCL_FORCE_INLINE T shuffle_up_impl(T x, int offset) {
  return apply_on_data(x, [offset](const float data) {
    float ret = data;
#pragma unroll
    for (int i = 0; i < rv_num_lanes(); ++i) {
      const float v =
          rv_extract(data, (rv_num_lanes() - offset + i) % rv_num_lanes());
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}

template <typename T>
HIPSYCL_FORCE_INLINE T shuffle_down_impl(T x, int offset) {
  return apply_on_data(x, [offset](const float data) {
    float ret = data;
#pragma unroll
    for (int i = 0; i < rv_num_lanes(); ++i) {
      const float v = rv_extract(data, (offset + i) % rv_num_lanes());
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}

template <typename T> T shuffle_xor_impl(T x, int lane_mask) {
  T ret = x;
#pragma unroll
  for (int i = 0; i < rv_num_lanes(); ++i) {
    int idx = (lane_mask ^ i) & (rv_num_lanes() - 1);
    const T v = intrin_extract(x, idx);
    ret = intrin_insert(ret, i, v);
  }
  return ret;
}

} // namespace detail

} // namespace sycl
} // namespace hipsycl

#endif //RV_SHUFFLE_H
