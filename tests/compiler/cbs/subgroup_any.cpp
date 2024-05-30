// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O3
// RUN: %t | FileCheck %s

#include <CL/sycl.hpp>
#include <iostream>

int main() {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 256*2;

  std::vector<int> host_buf(global_size, 0);
  std::iota(host_buf.begin(), host_buf.end(), 0);
  cl::sycl::queue queue;

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 1, mode::read_write, target::local>{local_size, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
          cl::sycl::nd_range<1>{global_size, local_size}, [=](cl::sycl::nd_item<1> item) noexcept {
            const auto g = item.get_group();
            const auto sg = item.get_sub_group();

            auto val = acc[item.get_global_id()];
            acc[item.get_global_id()] = cl::sycl::any_of_group(sg, val == 8) ? 1 : 0;
          });
    });
  }

  // CHECK: 1
  std::cout << host_buf[0] << "\n";
  // CHECK: 1
  std::cout << host_buf[31] << "\n";

  // CHECK: 0
  std::cout << host_buf[32] << "\n";
}