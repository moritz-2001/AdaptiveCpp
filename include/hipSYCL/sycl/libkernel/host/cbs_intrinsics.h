#ifndef CBS_INTRINSICS_H
#define CBS_INTRINSICS_H

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
// bit_and,
// bit_or,
// bit_xor,
// logical_and,
// logical_or

template <typename T> T __cbs_reduce(T, int);

template <typename T> T __cbs_shuffle(T, uint64_t i);

template <typename T> T __cbs_extract(T, uint64_t i);

#endif //CBS_INTRINSICS_H
