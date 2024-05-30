// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %t | FileCheck %s

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

int main()
{
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i)
  {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 1, mode::read_write, target::local>{local_size, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
        cl::sycl::nd_range<1>{global_size, local_size},
        [=](cl::sycl::nd_item<1> item) noexcept {
          const auto lid = item.get_local_id(0);
          const auto groupId = item.get_group(0);

          scratch[lid] = acc[item.get_global_id()];

          for(int i = 0; i < 4; ++i)
          {
            if(groupId > 0)
            {
              item.barrier();
              if(groupId > 1)
              {
                scratch[lid] += scratch[lid + 1];
                item.barrier();
                scratch[lid] += scratch[lid + 1];
              }
              else if(lid < local_size)
                scratch[lid] += scratch[0];
            }
            else
            {
              scratch[lid] += 1;
              item.barrier();
              scratch[lid] += 1;
            }
          }

          if(lid == 0)
            acc[item.get_global_id()] = scratch[lid];
        });
    });
  }
  for(size_t i = 0; i < global_size / local_size; ++i)
  {
    // CHECK: 8
    // CHECK: 4096
    // CHECK: 132096
    // CHECK: 197632
    std::cout << host_buf[i * local_size] << "\n";
  }
}
