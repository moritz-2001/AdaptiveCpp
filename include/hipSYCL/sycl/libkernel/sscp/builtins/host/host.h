#ifndef HOST_H
#define HOST_H

#include "../builtin_config.hpp"

#include <cstdint>

extern "C" void* work_group_shared_memory;
extern "C" void* sub_group_shared_memory();

#endif //HOST_H
