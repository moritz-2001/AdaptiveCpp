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


template<typename T>
T rv_shuffle(T, std::int32_t);

template<typename T>
T rv_extract(T, std::uint32_t);

template<typename T>
T rv_insert(T, std::uint32_t, T);

template<typename T>
T rv_reduce(T, int);


template<typename T>
T extract(T, uint32_t);



// extern "C" void rv_align(void*, std::int32_t);

//template<typename T>
//extern "C" T rv_extract(T, std::uint32_t);

#define MANGLED_VARIANTS(Type, MangleSuffix)                                                       \
  extern "C" Type rv_extract_##MangleSuffix(Type, std::uint32_t);                                  \
  extern "C" Type rv_insert_##MangleSuffix(Type, std::uint32_t, Type);                             \
  extern "C" Type rv_shuffle_##MangleSuffix(Type, std::uint32_t);

MANGLED_VARIANTS(float, f)
MANGLED_VARIANTS(double, d)
MANGLED_VARIANTS(std::int8_t, i8)
MANGLED_VARIANTS(std::int16_t, i16)
MANGLED_VARIANTS(std::int32_t, i32)
MANGLED_VARIANTS(std::int64_t, i64)

#endif // RV_H