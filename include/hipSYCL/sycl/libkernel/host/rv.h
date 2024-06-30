#ifndef RV_H
#define RV_H

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

template<typename T>
T rv_shuffle(T, std::int32_t);

template<typename T>
T rv_extract(T, std::uint32_t);

template<typename T>
T rv_insert(T, std::uint32_t, T);

template<typename T>
T rv_reduce(T, int);


enum class ReduceOp {
  PLUS = 0,
  MUL = 1,
  MIN = 2,
  MAX = 3,
};
template<typename  T>
T __cbs_reduce(T, int);

template<typename  T>
T __cbs_shift_left(T, uint64_t i);

template<typename  T>
T __cbs_shift_right(T, uint64_t i);

template<typename  T>
T __cbs_shuffle(T, uint64_t i);

template<typename T>
T __cbs_extract(T, uint64_t);



template<typename T>
T __reduce(T x, int i) {
  if constexpr (isRV) {
    return rv_reduce<T>(x, i);
  } else {
    return __cbs_reduce(x, i);
  }
}





#endif // RV_H
