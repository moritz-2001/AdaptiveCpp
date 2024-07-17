#ifndef rv_H
#define rv_H

#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"
#include <cstdint>
#include "hipSYCL/RV.h"

extern "C" bool rv_any(bool);
extern "C" bool rv_all(bool);
extern "C" std::uint32_t rv_ballot(bool);
extern "C" std::uint32_t rv_popcount(bool);
extern "C" std::uint32_t rv_index(bool);
extern "C" bool rv_mask();
extern "C" std::uint32_t rv_lane_id();
extern "C" std::uint32_t rv_num_lanes();

template <typename T> T rv_shuffle(T, std::int32_t);

template <typename T> T rv_extract(T, std::uint32_t);

template <typename T> T rv_insert(T, std::uint32_t, T);

template <typename T> T rv_reduce(T, int);



// ALL_INTEGRAL(CBS_EXTRACT)

#endif // RV_H
