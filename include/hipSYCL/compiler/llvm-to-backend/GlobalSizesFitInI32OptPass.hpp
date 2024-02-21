/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2024 Aksel Alpay
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

#ifndef HIPSYCL_SSCP_GLOBAL_SIZES_FIT_IN_I32_OPT_HPP
#define HIPSYCL_SSCP_GLOBAL_SIZES_FIT_IN_I32_OPT_HPP

#include <llvm/IR/PassManager.h>


namespace hipsycl {
namespace compiler {

class GlobalSizesFitInI32OptPass : public llvm::PassInfoMixin<GlobalSizesFitInI32OptPass> {
public:
  GlobalSizesFitInI32OptPass(bool GlobalSizesFitInInt, int KnownGroupSizeX = -1,
                             int KnownGroupSizeY = -1, int KnownGroupSizeZ = -1);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  int KnownGroupSizeX;
  int KnownGroupSizeY;
  int KnownGroupSizeZ;
  bool GlobalSizesFitInInt;
};


// inserts llvm.assume calls to assert that x >= RangeMin && x < RangeMax.
bool insertRangeAssumptionForBuiltinCalls(llvm::Module &M, llvm::StringRef BuiltinName,
                                          long long RangeMin, long long RangeMax, bool MaxIsLessThanEqual = false);

}
}

#endif
