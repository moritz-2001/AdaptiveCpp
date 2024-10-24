// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 256*4;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for (size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch =
          cl::sycl::accessor<int, 1, mode::read_write , target::local>{32, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
          cl::sycl::nd_range<1>{global_size, local_size}, [=](cl::sycl::nd_item<1> item) noexcept {
            const auto g = item.get_group();

            auto val = acc[item.get_global_id()];
            acc[item.get_global_id()] = cl::sycl::reduce_over_group(g, val, cl::sycl::plus<int>());
          });
    });
  }
  for (size_t i = 0; i < (global_size + local_size-1) / local_size; ++i) {
    // CHECK: 32640
    // CHECK: 98176
    // CHECK: 163712
    // CHECK: 229248
    std::cout << host_buf[i * local_size] << "\n";
  }
}