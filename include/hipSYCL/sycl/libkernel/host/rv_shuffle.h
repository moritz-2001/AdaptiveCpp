#ifndef RV_SHUFFLE_H
#define RV_SHUFFLE_H

#include "rv.h"
#include <type_traits>
#include <array>
#include <cstring>


namespace hipsycl {
namespace sycl {

namespace detail {

template <typename T, typename RvOperation>
T apply_on_data(T x, const RvOperation& op) {
  if constexpr (std::is_fundamental<T>::value) {
    return op(x);
  } else {
    static_assert(std::is_fundamental<T>::value);
    constexpr std::size_t number_of_floats = (sizeof(T) + sizeof(float) - 1) / sizeof(float);
    std::array<float, number_of_floats> words;
    std::memcpy(&words, &x, sizeof(T));
    for (auto& word : words) {
      word = op(word);
    }
    T output;
    std::memcpy(&output, &words, sizeof(T));
    return output;
  }
}

/*
template <typename T> T swap(T vector, uint32_t i, uint32_t j) {
  const auto ithLaneV = rv_extract(vector, i);
  const auto jthLaneV = rv_extract(vector, j);
  vector = rv_insert(vector, j, ithLaneV);
  vector = rv_insert(vector, i, jthLaneV);
  return vector;
}

template <typename T, typename Pred> uint32_t rv_partition(T& x, Pred pred) {
  size_t j = 0; // All element in [0, j) fullfill pred.
  T copyX = x;
#pragma unroll
  for (uint32_t i = 0; i < rv_num_lanes(); ++i) {
    const auto v = rv_extract(copyX, i);
    if (pred(v)) {
      x = swap(x, j, i);
      ++j;
    }
  }
  return j;
}

template <typename T, typename Pred> uint32_t rv_count_if(T x, Pred pred) {
  size_t j = 0;
#pragma unroll
  for (uint32_t i = 0; i < rv_num_lanes(); ++i) {
      const auto v = rv_extract(x, i);
      if (pred(v)) {
        ++j;
      }
  }
  return j;
}

template <typename T, typename Pred> int rv_find_if(T x, Pred pred) {
  int j = -1;
#pragma unroll
  for (int i = rv_num_lanes()-1; i >= 0; --i) {
    const auto v = rv_extract(x, i);
    if (pred(v)) {
      j = i;
    }
  }
  return j;
}
*/

template <typename T> T extract_impl(T x, int id) {
  return apply_on_data(x, [id](const auto data) { return rv_extract(data, id); });
}

// difference between shuffle_impl and extract_impl: id for extract must be
// uniform value.
template <typename T> T shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](const float data) {
    float ret = data;
#pragma unroll
    for (int i = 0; i < rv_num_lanes(); ++i) {
      const auto srcLane = rv_extract(id, i);
      const float v = rv_extract(data, srcLane);
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
#endif // RV_SHUFFLE_H
