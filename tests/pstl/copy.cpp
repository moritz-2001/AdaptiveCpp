/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_copy, enable_unified_shared_memory)

struct non_trivial_copy {
  non_trivial_copy(){}

  non_trivial_copy(int val)
  : x{val} {}

  non_trivial_copy(const non_trivial_copy& other)
  : x{other.x + 1} {}

  non_trivial_copy& operator=(const non_trivial_copy& other) {
    x = other.x + 1;
    return *this;
  }

  friend bool operator==(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x == b.x;
  }

  friend bool operator!=(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x != b.x;
  }

  int x;
};

template<class T>
void test_copy(std::size_t problem_size) {
  std::vector<T> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = T{i};
  }

  std::vector<T> dest_device(problem_size);
  std::vector<T> dest_host(problem_size);

  auto ret = std::copy(std::execution::par_unseq, data.begin(), data.end(),
                       dest_device.begin());
  std::copy(data.begin(), data.end(), dest_host.begin());

  BOOST_CHECK(ret == dest_device.begin() + problem_size);
  BOOST_CHECK(dest_device == dest_device);
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types::type) {
  test_copy<T>(0);
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types::type) {
  test_copy<T>(1);
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types::type) {
  test_copy<T>(1000);
}



BOOST_AUTO_TEST_SUITE_END()
