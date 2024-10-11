#ifndef RV_SHUFFLE_H
#define RV_SHUFFLE_H

#include "rv.h"
#include <type_traits>
#include <array>
#include <cstring>


namespace hipsycl {
namespace sycl {

namespace detail {

template<class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) &&
    std::is_trivially_copyable_v<From> &&
    std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept
{
  static_assert(std::is_trivially_constructible_v<To>,
      "This implementation additionally requires "
      "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}


template <typename T, typename RvOperation>
T apply_on_data(T x, const RvOperation& op) {
  if constexpr (std::is_fundamental<T>::value) {
    return op(x);
  } else if constexpr (sizeof(T) == 8) {
    return bit_cast<T>(op(bit_cast<uint64_t>(x)));
  } else if constexpr (sizeof(T) == 4) {
    return bit_cast<T>(op(bit_cast<uint32_t>(x)));
  } else if constexpr (sizeof(T) == 2) {
    return bit_cast<T>(op(bit_cast<uint16_t>(x)));
  } else if constexpr (sizeof(T) == 1) {
    return bit_cast<T>(op(bit_cast<uint8_t>(x)));
  } else {
    constexpr std::size_t number_of_floats = (sizeof(T) + sizeof(float) - 1) / sizeof(float);
    std::array<float, number_of_floats> words{};
    std::memcpy(&words, &x, sizeof(T));

    for (auto i = 0ul; i < number_of_floats; ++i)
      words[i] = op(words[i]);

    T output;
    std::memcpy(&output, &words, sizeof(T));
    return output;
  }
}

template <typename T> T extract_impl(T x, int id) {
  return apply_on_data(x, [id](const auto data) { return rv_extract(data, id); });
}

// difference between shuffle_impl and extract_impl: id for extract must be
// uniform value.
template <typename T> T shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](const auto data) {
    auto ret = data;
#pragma unroll
    for (int i = 0; i < rv_num_lanes(); ++i) {
      const auto srcLane = rv_extract(id, i);
      const auto v = rv_extract(data, srcLane);
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}

template <typename T>
T shuffle_up_impl(T x, int offset) {
  return apply_on_data(x, [offset](const auto data) {
    auto ret = data;
   #pragma unroll
    for (int i = 0; i < rv_num_lanes(); ++i) {
      const auto v = rv_extract(data, (rv_num_lanes() - offset + i) % rv_num_lanes());
      ret = rv_insert(ret, i, v);
    }
    return ret;
  });
}

template <typename T> T shuffle_down_impl(T x, int offset) {
    return apply_on_data(x, [offset](const auto data) {
      auto ret = data;
 #pragma unroll
      for (int i = 0; i < rv_num_lanes(); ++i) {
        const auto v = rv_extract(data, (offset + i) % rv_num_lanes());
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
#endif // RV_SHUFFLE_H
