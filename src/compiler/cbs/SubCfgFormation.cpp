/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/cbs/UniformityAnalysis.hpp"

#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/LoopPass.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Regex.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/CodeExtractor.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include <cstddef>
#include <functional>
#include <llvm/Analysis/IVDescriptors.h>
#include <numeric>

#define DEBUG_SUBCFG_FORMATION
#define RV

namespace {
using namespace hipsycl::compiler;
using namespace hipsycl::compiler::cbs;

constexpr std::array<char, 3> DimName{'x', 'y', 'z'};

enum class HierarchicalLevel {
  CBS,
  H_CBS_GROUP,
  H_CBS_SUBGROUP,
};

// Reference type only!
struct HierarchicalSplitInfo {
  HierarchicalLevel Level;
  llvm::ArrayRef<llvm::Value *> OuterLocalSize;
  llvm::ArrayRef<llvm::Value *> OuterIndices;
  llvm::Value *ContiguousIdx;
  llvm::Value *SGIdArg;
  llvm::SmallDenseMap<llvm::Argument *, llvm::AllocaInst *, 8> *ArgsToAloca;

  llvm::Value *getContiguousIdx() {
    switch (Level) {
    case HierarchicalLevel::CBS:
    case HierarchicalLevel::H_CBS_GROUP:
      return ContiguousIdx;
    case HierarchicalLevel::H_CBS_SUBGROUP:
      return SGIdArg;
    }
    assert(false);
  }
};

// gets the load inside F from the global variable called VarName
llvm::Instruction *getLoadForGlobalVariable(llvm::Function &F, llvm::StringRef VarName) {
  auto SizeT = F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
  auto *GV = F.getParent()->getOrInsertGlobal(VarName, SizeT);
  for (auto U : GV->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->getFunction() == &F)
      return LI;
  }

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  return Builder.CreateLoad(SizeT, GV);
}

llvm::LoadInst *mergeGVLoadsInEntry(llvm::Function &F, llvm::StringRef VarName,
                                    llvm::Type *ty = nullptr) {
  auto SizeT = F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
  auto *GV = F.getParent()->getOrInsertGlobal(VarName, SizeT);

  llvm::LoadInst *FirstLoad = nullptr;
  llvm::SmallVector<llvm::LoadInst *, 4> Loads;
  for (auto U : GV->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->getFunction() == &F) {
      if (!FirstLoad)
        FirstLoad = LI;
      else
        Loads.push_back(LI);
    }
  }

  if (FirstLoad) {
    FirstLoad->moveBefore(&F.getEntryBlock().front());
    for (auto *LI : Loads) {
      LI->replaceAllUsesWith(FirstLoad);
      LI->eraseFromParent();
    }
    return FirstLoad;
  }

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  auto *Load = Builder.CreateLoad(ty ? ty : SizeT, GV, "cbs.load." + GV->getName());
  return Load;
}

// parses the range dimensionality from the mangled kernel name
std::size_t getRangeDim(llvm::Function &F) {
  auto FName = F.getName();
  // todo: fix with MS mangling
  llvm::Regex Rgx("iterate_nd_range_ompILi([1-3])E");
  llvm::SmallVector<llvm::StringRef, 4> Matches;
  if (Rgx.match(FName, &Matches))
    return std::stoull(static_cast<std::string>(Matches[1]));

  if (auto MD = F.getParent()->getNamedMetadata(SscpAnnotationsName)) {
    for (auto OP : MD->operands()) {
      if (OP->getNumOperands() == 3 &&
          llvm::cast<llvm::MDString>(OP->getOperand(1))->getString() == SscpKernelDimensionName) {
        if (&F == llvm::dyn_cast<llvm::Function>(
                      llvm::cast<llvm::ValueAsMetadata>(OP->getOperand(0))->getValue())) {
          auto ConstMD = llvm::cast<llvm::ConstantAsMetadata>(OP->getOperand(2))->getValue();
          if (auto CI = llvm::dyn_cast<llvm::ConstantInt>(ConstMD))
            return CI->getZExtValue();
          if (auto ZI = llvm::dyn_cast<llvm::ConstantAggregateZero>(ConstMD))
            return 0;
          if (auto CS = llvm::dyn_cast<llvm::ConstantStruct>(ConstMD))
            return llvm::cast<llvm::ConstantInt>(CS->getOperand(0))->getZExtValue();
        }
      }
    }
  }
  llvm_unreachable("[SubCFG] Could not deduce kernel dimensionality!");
}

// searches for llvm.var.annotation and returns the value that is annotated by it, as well the
// annotation instruction
std::pair<llvm::Value *, llvm::Instruction *>
getLocalSizeArgumentFromAnnotation(llvm::Function &F) {
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *UI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (UI->getCalledFunction()->getName().startswith("llvm.var.annotation")) {
          HIPSYCL_DEBUG_INFO << *UI << '\n';
          llvm::GlobalVariable *AnnotateStr = nullptr;
          if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(UI->getOperand(1));
              CE && CE->getOpcode() == llvm::Instruction::GetElementPtr) {
            if (auto *AnnoteStr = llvm::dyn_cast<llvm::GlobalVariable>(CE->getOperand(0)))
              AnnotateStr = AnnoteStr;
          } else if (auto *AnnoteStr =
                         llvm::dyn_cast<llvm::GlobalVariable>(UI->getOperand(1))) // opaque-ptr
            AnnotateStr = AnnoteStr;

          if (AnnotateStr) {
            if (auto *Data =
                    llvm::dyn_cast<llvm::ConstantDataSequential>(AnnotateStr->getInitializer())) {
              if (Data->isString() &&
                  Data->getAsString().startswith("hipsycl_nd_kernel_local_size_arg")) {
                if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(UI->getOperand(0)))
                  return {BC->getOperand(0), UI};
                return {UI->getOperand(0), UI};
              }
            }
          }
        }

  assert(false && "Didn't find annotated argument!");
  return {nullptr, nullptr};
}

// identify the local size values by the store to it
void fillStores(llvm::Value *V, int Idx, llvm::SmallVector<llvm::Value *, 3> &LocalSize) {
  if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(V)) {
    LocalSize[Idx] = Store->getOperand(0);
  } else if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(V)) {
    for (auto *BCU : BC->users()) {
      fillStores(BCU, Idx, LocalSize);
    }
  } else if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
    auto *IdxV = GEP->indices().begin() + (GEP->getNumIndices() - 1);
    auto *IdxC = llvm::cast<llvm::ConstantInt>(IdxV);
    for (auto *GU : GEP->users()) {
      fillStores(GU, IdxC->getSExtValue(), LocalSize);
    }
  }
}

// reinterpret single argument as array if neccessary and load scalar size values into LocalSize
void loadSizeValuesFromArgument(llvm::Function &F, int Dim, llvm::Value *LocalSizeArg,
                                const llvm::DataLayout &DL,
                                llvm::SmallVector<llvm::Value *, 3> &LocalSize, bool IsSscp) {
  // local_size is just an array of size_t's..
  auto SizeTSize = DL.getLargestLegalIntTypeSizeInBits();
  auto *SizeT = DL.getLargestLegalIntType(F.getContext());

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  llvm::Value *LocalSizePtr = nullptr;
  if (!LocalSizeArg->getType()->isArrayTy()) {
#if HAS_TYPED_PTR
    auto PtrTy = llvm::Type::getIntNPtrTy(F.getContext(), SizeTSize);
#else
    auto PtrTy = llvm::PointerType::get(F.getContext(), 0);
#endif
    LocalSizePtr = Builder.CreatePointerCast(LocalSizeArg, PtrTy, "local_size.cast");
  }
  for (unsigned int I = 0; I < Dim; ++I) {
    auto CurDimName = DimName[IsSscp ? Dim - I - 1 : I];
    if (LocalSizeArg->getType()->isArrayTy()) {
      LocalSize[I] =
          Builder.CreateExtractValue(LocalSizeArg, {I}, "local_size." + llvm::Twine{CurDimName});
    } else {
      auto *LocalSizeGep =
          Builder.CreateInBoundsGEP(SizeT, LocalSizePtr, Builder.getIntN(SizeTSize, I),
                                    "local_size.gep." + llvm::Twine{CurDimName});
      HIPSYCL_DEBUG_INFO << *LocalSizeGep << "\n";

      LocalSize[I] =
          Builder.CreateLoad(SizeT, LocalSizeGep, "local_size." + llvm::Twine{CurDimName});
    }
  }
}

// get the wg size values for the loop bounds
llvm::SmallVector<llvm::Value *, 3> getLocalSizeValues(llvm::Function &F, int Dim,
                                                       bool isSscpKernel) {
  if (isSscpKernel) {
    llvm::SmallVector<llvm::Value *, 3> LocalSize(Dim);
    for (int I = 0; I < Dim; ++I) {
      auto Load = getLoadForGlobalVariable(F, LocalSizeGlobalNames[Dim - I - 1]);
      Load->moveBefore(F.getEntryBlock().getTerminator());
      LocalSize[I] = Load;
    }
    return LocalSize;
  }

  auto &DL = F.getParent()->getDataLayout();
  auto [LocalSizeArg, Annotation] = getLocalSizeArgumentFromAnnotation(F);

  llvm::SmallVector<llvm::Value *, 3> LocalSize(Dim);
  HIPSYCL_DEBUG_INFO << *LocalSizeArg << "\n";

  if (!llvm::dyn_cast<llvm::Argument>(LocalSizeArg))
    for (auto *U : LocalSizeArg->users())
      fillStores(U, 0, LocalSize);
  else
    loadSizeValuesFromArgument(F, Dim, LocalSizeArg, DL, LocalSize, false);

  Annotation->eraseFromParent();
  return LocalSize;
}

std::unique_ptr<RegionImpl> getRegion(llvm::Function &F, const llvm::LoopInfo &LI,
                                      llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  return std::unique_ptr<RegionImpl>{new FunctionRegion(F, Blocks)};
}

llvm::SmallVector<llvm::CallInst *, 8> getExtractIntrinsics(llvm::Function &F) {
  llvm::SmallVector<llvm::CallInst *, 8> Intrinsics{};
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (utils::isExtractIntrinsic(CallInst->getCalledFunction()))
          Intrinsics.emplace_back(CallInst);
      }
    }
  }
  return Intrinsics;
}

// calculate uniformity analysis
VectorizationInfo getVectorizationInfo(llvm::Function &F, Region &R, llvm::LoopInfo &LI,
                                       llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                                       size_t Dim, HierarchicalSplitInfo HI) {
  VectorizationInfo VecInfo{F, R};
  // seed varyingness
  for (size_t D = 0; D < Dim - 1; ++D) {
    VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, LocalIdGlobalNames[D]), VectorShape::cont());
  }
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, LocalIdGlobalNames[Dim - 1]), VectorShape::cont());

  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, SgIdGlobalName), VectorShape::cont());

  VectorizationAnalysis VecAna{VecInfo, LI, DT, PDT};
  VecAna.analyze();
  return VecInfo;
}

// create the wi-loops around a kernel or subCFG, LastHeader input should be the load block,
// ContiguousIdx may be any identifyable value (load from undef)
void createLoopsAround(llvm::Function &F, llvm::BasicBlock *AfterBB,
                       const llvm::ArrayRef<llvm::Value *> &LocalSize, int EntryId,
                       llvm::ValueToValueMapTy &VMap,
                       llvm::SmallVector<llvm::BasicBlock *, 3> &Latches,
                       llvm::BasicBlock *&LastHeader, llvm::Value *&ContiguousIdx, bool IsSscp,
                       HierarchicalSplitInfo HI) {
  const auto &DL = F.getParent()->getDataLayout();
  auto *LoadBB = LastHeader;
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};

  const size_t Dim = LocalSize.size();
  const auto InnerMost = Dim - 1;
  constexpr auto OuterMost = 0;

  struct LocalIdNames {
    // e.g.: x
    char shortName;
    llvm::StringRef name;
  };

  std::array<LocalIdNames, 3> LocalIdsNames{};
  for (size_t D = 0; D < Dim; ++D) {
    LocalIdsNames[D].shortName = DimName[D];
    LocalIdsNames[D].name = LocalIdGlobalNames[D];
  }
  if (IsSscp) {
    std::reverse(LocalIdsNames.begin(), LocalIdsNames.begin() + Dim);
  }

  auto loopIncrement = [&HI, &InnerMost](const int D) -> size_t {
    if (D != InnerMost) {
      return 1;
    }
    switch (HI.Level) {
    case HierarchicalLevel::CBS:
    case HierarchicalLevel::H_CBS_SUBGROUP:
      return 1;
    case HierarchicalLevel::H_CBS_GROUP:
      return SGSize;
    }
    assert(false);
    return 0;
  };

  auto suffix = [&HI, &LocalIdsNames, &EntryId](int D) {
    return (llvm::Twine{
                HI.Level == HierarchicalLevel::H_CBS_SUBGROUP ? 's' : LocalIdsNames[D].shortName} +
            ".subcfg." + llvm::Twine{EntryId})
        .str();
  };

  // from innermost to outermost: create loops around the LastHeader and use AfterBB as dummy exit
  // to be replaced by the outer latch later
  llvm::SmallVector<llvm::PHINode *, 3> IndVars;
  for (int D = static_cast<int>(Dim) - 1; D >= 0; --D) {
    const std::string Suffix = suffix(D);

    auto *Header = llvm::BasicBlock::Create(LastHeader->getContext(), "header." + Suffix + "b",
                                            LastHeader->getParent(), LastHeader);
    Builder.SetInsertPoint(Header, Header->getFirstInsertionPt());
    auto *WIIndVar =
        Builder.CreatePHI(DL.getLargestLegalIntType(F.getContext()), 2, "indvar." + Suffix);
    WIIndVar->addIncoming(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 0),
                          &F.getEntryBlock());

    IndVars.push_back(WIIndVar);
    Builder.CreateBr(LastHeader);

    auto *Latch = llvm::BasicBlock::Create(F.getContext(), "latch." + Suffix + "b", &F);
    Builder.SetInsertPoint(Latch, Latch->getFirstInsertionPt());

    auto *IncIndVar = Builder.CreateAdd(
        WIIndVar, Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), loopIncrement(D)),
        "addInd." + Suffix, true, false);
    WIIndVar->addIncoming(IncIndVar, Latch);

    HIPSYCL_DEBUG_ERROR << "ICMP " << *LocalSize[D] << " " << *IncIndVar << "\n";
    llvm::Value *LoopCond = Builder.CreateICmpULT(IncIndVar, LocalSize[D], "exit.cond." + Suffix);

#ifndef RV
    if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
      assert(D == InnerMost);
      // TODO don't we need to multiply the local sizes?
      auto *ContCond = Builder.CreateICmpULT(ContiguousIdx, HI.OuterLocalSize.back(),
                                             "exit.cont_cond." + Suffix);
      LoopCond = Builder.CreateLogicalAnd(ContCond, LoopCond);
    }
#endif

    Builder.CreateCondBr(LoopCond, Header, AfterBB);
    Latches.push_back(Latch);
    LastHeader = Header;
  }

  std::reverse(Latches.begin(), Latches.end());
  std::reverse(IndVars.begin(), IndVars.end());

  for (size_t D = 1; D < Dim; ++D) {
    Latches[D]->getTerminator()->replaceSuccessorWith(AfterBB, Latches[D - 1]);
    IndVars[D]->replaceIncomingBlockWith(&F.getEntryBlock(), IndVars[D - 1]->getParent());
  }

  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP or HI.Level == HierarchicalLevel::CBS) {
    auto *MDWorkItemLoop = llvm::MDNode::get(
        F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::WorkItemLoop)});
    auto *LoopID =
        llvm::makePostTransformationMetadata(F.getContext(), nullptr, {}, {MDWorkItemLoop});
    Latches[InnerMost]->getTerminator()->setMetadata("llvm.loop", LoopID);
  }

  VMap[AfterBB] = Latches[InnerMost];

  // add contiguous ind var calculation to load block
  Builder.SetInsertPoint(IndVars[InnerMost]->getParent(), ++IndVars[InnerMost]->getIterator());
  llvm::Value *Idx = IndVars[OuterMost];
  for (size_t D = 1; D < Dim; ++D) {
    const std::string Suffix =
        (llvm::Twine{LocalIdsNames[D].shortName} + ".subcfg." + llvm::Twine{EntryId}).str();

    Idx = Builder.CreateMul(Idx, LocalSize[D], "idx.mul." + Suffix, true);
    Idx = Builder.CreateAdd(IndVars[D], Idx, "idx.add." + Suffix, true);
  }

  for (size_t D = 0; D < Dim; ++D) {
    VMap[mergeGVLoadsInEntry(F, LocalIdsNames[D].name)] = IndVars[D];
  }

  if (HI.Level == HierarchicalLevel::CBS) {
    Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
    VMap[mergeGVLoadsInEntry(F, SgIdGlobalName)] = Builder.CreateURem(
        IndVars.back(), llvm::ConstantInt::get(IndVars.back()->getType(), SGSize));
  } else if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    VMap[HI.SGIdArg] = Idx;
    Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
    auto StridedInner = llvm::cast<llvm::Instruction>(HI.OuterIndices.back())->getOperand(0);
    Idx = Builder.CreateAdd(Idx, StridedInner, "stride.inner.add.");
    // TODO ? fixme: this is not actually the contiguous index for multi-dim..
  }

  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP or HI.Level == HierarchicalLevel::CBS) {
    VMap[ContiguousIdx] = Idx; // ContiguousIdx = idx + sg_id
    ContiguousIdx = Idx;
  } else {
    assert(HI.Level == HierarchicalLevel::H_CBS_GROUP);
    Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
    llvm::outs() << "IND VAR: " << *IndVars.back() << "\n";
    llvm::outs() << "SgIdGlobal: " << *mergeGVLoadsInEntry(F, SgIdGlobalName) << "\n";

    VMap[mergeGVLoadsInEntry(F, LocalIdsNames[InnerMost].name)] =
        Builder.CreateAdd(IndVars.back(), mergeGVLoadsInEntry(F, SgIdGlobalName));
    VMap[ContiguousIdx] = ContiguousIdx;
  }

  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    llvm::ValueToValueMapTy VMap;
    // Old Idx becomes new Idx
    VMap[HI.ContiguousIdx] = Idx;
    VMap[mergeGVLoadsInEntry(F, LocalIdsNames[InnerMost].name)] = Idx;
    VMap[HI.SGIdArg] = IndVars[0];
    VMap[mergeGVLoadsInEntry(F, SgIdGlobalName)] = IndVars[0];
    llvm::SmallVector<llvm::BasicBlock *> Blocks{Latches.begin(), Latches.end()};
    Blocks.push_back(LoadBB);
    llvm::remapInstructionsInBlocks(Blocks, VMap);
  }
}

class SubCFG {
  using BlockVector = llvm::SmallVector<llvm::BasicBlock *, 8>;
  BlockVector Blocks_;
  BlockVector NewBlocks_;
  size_t EntryId_;
  llvm::BasicBlock *EntryBarrier_;
  llvm::SmallDenseMap<llvm::BasicBlock *, size_t> ExitIds_;
  llvm::AllocaInst *LastBarrierIdStorage_;
  llvm::BasicBlock *EntryBB_;
  llvm::BasicBlock *ExitBB_;
  llvm::BasicBlock *LoadBB_;
  llvm::BasicBlock *PreHeader_;
  size_t Dim;
  HierarchicalSplitInfo HI;
  llvm::SmallVector<llvm::Value *, 3> WIIndVars_;

  llvm::BasicBlock *
  createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                   llvm::BasicBlock *After, llvm::BasicBlock *TargetBB);

  void loadMultiSubCfgValues(
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
      llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  void loadUniformAndRecalcContValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
      llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  llvm::BasicBlock *createLoadBB(llvm::ValueToValueMapTy &VMap);

  llvm::BasicBlock *createUniformLoadBB(llvm::BasicBlock *OuterMostHeader);

public:
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
         const SplitterAnnotationInfo &SAA, size_t Dim, HierarchicalSplitInfo HI);

  SubCFG(const SubCFG &) = delete;

  SubCFG &operator=(const SubCFG &) = delete;

  SubCFG(SubCFG &&) = default;

  SubCFG &operator=(SubCFG &&) = default;

  BlockVector &getBlocks() noexcept { return Blocks_; }
  const BlockVector &getBlocks() const noexcept { return Blocks_; }

  BlockVector &getNewBlocks() noexcept { return NewBlocks_; }
  const BlockVector &getNewBlocks() const noexcept { return NewBlocks_; }

  size_t getEntryId() const noexcept { return EntryId_; }

  llvm::BasicBlock *getEntry() noexcept { return EntryBB_; }
  llvm::BasicBlock *getExit() noexcept { return ExitBB_; }
  llvm::BasicBlock *getLoadBB() noexcept { return LoadBB_; }

  const llvm::SmallVector<llvm::Value *, 3> &getWIIndVars() const noexcept { return WIIndVars_; }
  HierarchicalSplitInfo getHI() const noexcept { return HI; }

  void replicate(llvm::Function &F,
                 const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
                     &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                 llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, bool IsSscp,
                 llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  void arrayifyMultiSubCfgValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements,
      VectorizationInfo &VecInfo, llvm::Function &F);

  void fixSingleSubCfgValues(
      llvm::DominatorTree &DT,
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
      llvm::Value *ReqdArrayElements, VectorizationInfo &VecInfo,
      llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  void print() const;

  void removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const;

  llvm::SmallVector<llvm::Instruction *, 16>
  topoSortInstructions(const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const;
};

// create new exiting block writing the exit's id to LastBarrierIdStorage_
llvm::BasicBlock *
SubCFG::createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                         llvm::BasicBlock *After, llvm::BasicBlock *TargetBB) {
  HIPSYCL_DEBUG_INFO << "Create new exit with ID: " << BarrierPair.second << " at "
                     << After->getName() << "\n";

  auto *Exit = llvm::BasicBlock::Create(After->getContext(),
                                        After->getName() + ".subcfg.exit" +
                                            llvm::Twine{BarrierPair.second} + "b",
                                        After->getParent(), TargetBB);

  auto &DL = Exit->getParent()->getParent()->getDataLayout();
  llvm::IRBuilder Builder{Exit, Exit->getFirstInsertionPt()};
  Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), BarrierPair.second),
                      LastBarrierIdStorage_);
  Builder.CreateBr(TargetBB);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

// identify a new SubCFG using DFS starting at EntryBarrier
SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
               const SplitterAnnotationInfo &SAA, size_t Dim, HierarchicalSplitInfo HI)
    : EntryId_(BarrierIds.lookup(EntryBarrier)), EntryBarrier_(EntryBarrier),
      LastBarrierIdStorage_(LastBarrierIdStorage), EntryBB_(EntryBarrier->getSingleSuccessor()),
      LoadBB_(nullptr), PreHeader_(nullptr), Dim(Dim), HI(HI) {
  assert(HI.ContiguousIdx && "Must have found __acpp_cbs_local_id_{x,y,z}");

  llvm::SmallVector<llvm::BasicBlock *, 4> WL{EntryBarrier};
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();

    llvm::SmallVector<llvm::BasicBlock *, 2> Succs{llvm::succ_begin(BB), llvm::succ_end(BB)};
    for (auto *Succ : Succs) {
      if (std::find(Blocks_.begin(), Blocks_.end(), Succ) != Blocks_.end())
        continue;

      if (!(HI.Level == HierarchicalLevel::H_CBS_SUBGROUP ? utils::hasOnlySubBarrier(Succ, SAA)
                                                          : utils::hasOnlyBarrier(Succ, SAA))) {
        WL.push_back(Succ);
        Blocks_.push_back(Succ);
      } else {
        size_t BId = BarrierIds.lookup(Succ);
        assert(BId != 0 && "Exit barrier block not found in map");
        ExitIds_.insert({Succ, BId});
      }
    }
  }
}

void SubCFG::print() const {
  HIPSYCL_DEBUG_INFO << "SubCFG entry barrier: " << EntryId_ << "\n";
  HIPSYCL_DEBUG_INFO << "SubCFG block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : Blocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG exits: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto ExitIt
                                  : ExitIds_) {
    llvm::outs() << ExitIt.first->getName() << " (" << ExitIt.second << "), ";
  } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG new block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : NewBlocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
}

void addRemappedDenseMapKeys(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &OrgInstAllocaMap,
    const llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &NewInstAllocaMap) {
  for (auto &[Inst, Alloca] : OrgInstAllocaMap) {
    if (auto *NewInst = llvm::dyn_cast_or_null<llvm::Instruction>(VMap.lookup(Inst)))
      NewInstAllocaMap.insert({NewInst, Alloca});
  }
}

// clone all BBs of the subcfg, create wi-loop structure around and fixup values
void SubCFG::replicate(
    llvm::Function &F, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
    llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, bool IsSscp,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca) {
  llvm::ValueToValueMapTy VMap;

  // clone blocks
  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", &F);
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      if (auto ExitIt = ExitIds_.find(Succ); ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, AfterBB));
      }
    }
  }

  LoadBB_ = createLoadBB(VMap);

  VMap[EntryBarrier_] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  llvm::BasicBlock *LastHeader = LoadBB_;
  llvm::Value *Idx = HI.ContiguousIdx;

  createLoopsAround(F, AfterBB, LocalSize, EntryId_, VMap, Latches, LastHeader, Idx, IsSscp, HI);

  for (size_t D = 0; D < LocalSize.size(); ++D) {
    WIIndVars_.push_back(VMap[HI.Level == HierarchicalLevel::H_CBS_SUBGROUP
                                  ? HI.SGIdArg
                                  : mergeGVLoadsInEntry(F, LocalIdGlobalNames[D])]);
  }

  PreHeader_ = createUniformLoadBB(LastHeader);
  LastHeader->replacePhiUsesWith(&F.getEntryBlock(), PreHeader_);

  print();

  // TODO what is the purpose of RemappedInstAllocaMap, and why does it get filled before load...
  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap,
                        loadToAlloca);
  loadUniformAndRecalcContValues(BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap,
                                 loadToAlloca);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = Latches[0];
  HI.ContiguousIdx = Idx;
  HI.SGIdArg = WIIndVars_.back();
}

// remove incoming PHI blocks that no longer actually have an edge to the PHI
void SubCFG::removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const {
  for (auto *BB : BlocksToRemap) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> Predecessors{llvm::pred_begin(BB), llvm::pred_end(BB)};
    for (auto &I : *BB) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
        llvm::SmallVector<llvm::BasicBlock *, 4> IncomingBlocksToRemove;
        for (size_t IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues(); ++IncomingIdx) {
          auto *IncomingBB = Phi->getIncomingBlock(IncomingIdx);
          if (!Predecessors.contains(IncomingBB))
            IncomingBlocksToRemove.push_back(IncomingBB);
        }
        for (auto *IncomingBB : IncomingBlocksToRemove) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Remove incoming block " << IncomingBB->getName()
                             << " from PHI " << *Phi << "\n";
          Phi->removeIncomingValue(IncomingBB);
          HIPSYCL_DEBUG_INFO << "[SubCFG] Removed incoming block " << IncomingBB->getName()
                             << " from PHI " << *Phi << "\n";
        }
      }
    }
  }
}

// check if a contiguous value can be tracked back to only uniform values and the wi-loop indvar
// currently cannot track back the value through PHI nodes.
bool dontArrayifyContiguousValues(
    llvm::Instruction &I,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements, llvm::Value *IndVar,
    VectorizationInfo &VecInfo) {
  // is cont indvar
  if (VecInfo.isPinned(I))
    return true;

  /*
  for (auto *User : I.users()) {
    if (auto *FCall = llvm::dyn_cast<llvm::CallInst>(User)) {
      if (utils::isCBSIntrinsic(FCall->getCalledFunction())) {
        return false;
      }
    }
  }
  */

  llvm::SmallVector<llvm::Instruction *, 4> WL;
  llvm::SmallPtrSet<llvm::Instruction *, 8> UniformValues;
  llvm::SmallVector<llvm::Instruction *, 8> ContiguousInsts;
  llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;
  HIPSYCL_DEBUG_INFO << "[SubCFG] Cont value? " << I << " IndVar: " << *IndVar << "\n";
  WL.push_back(&I);
  while (!WL.empty()) {
    auto *WLValue = WL.pop_back_val();
    if (auto *WLI = llvm::dyn_cast<llvm::Instruction>(WLValue))
      for (auto *V : WLI->operand_values()) {
        HIPSYCL_DEBUG_INFO << "[SubCFG] Considering: " << *V << "\n";

        if (V == IndVar || VecInfo.isPinned(*V) || llvm::isa<llvm::Constant>(V))
          continue;
        // todo: fix PHIs
        if (LookedAt.contains(V))
          return false;
        LookedAt.insert(V);

        // collect cont and uniform source values
        if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
          if (VecInfo.getVectorShape(*OpI).isContiguousOrStrided()) {
            WL.push_back(OpI);
            ContiguousInsts.push_back(OpI);
          } else if (!UniformValues.contains(OpI))
            UniformValues.insert(OpI);
        }
      }
  }

  for (auto *UI : UniformValues) {
    HIPSYCL_DEBUG_INFO << "[SubCFG] UniValue to store: " << *UI << "\n";
    if (BaseInstAllocaMap.lookup(UI))
      continue;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Store required uniform value to single element alloca " << I
                       << "\n";
    auto *Alloca = utils::arrayifyInstruction(AllocaIP, UI, IndVar, nullptr);
    BaseInstAllocaMap.insert({UI, Alloca});
    VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
  }
  ContInstReplicaMap.insert({&I, ContiguousInsts});
  return true;
}

// creates array allocas for values that are identified as spanning multiple subcfgs
void SubCFG::arrayifyMultiSubCfgValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements,
    VectorizationInfo &VecInfo, llvm::Function &F) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> OtherCFGBlocks;
  for (auto &Cfg : SubCFGs) {
    if (&Cfg != this)
      OtherCFGBlocks.insert(Cfg.Blocks_.begin(), Cfg.Blocks_.end());
  }

  auto *ContiguousIdx = HI.getContiguousIdx();

  HIPSYCL_DEBUG_ERROR << "[SubCFG] ARRAIFY \n";

  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (&I == ContiguousIdx)
        continue;
      if (InstAllocaMap.lookup(&I)) {
        continue;
      }
      // if any use is in another subcfg
      if (utils::anyOfUsers<llvm::Instruction>(&I, [&OtherCFGBlocks, &I](auto *UI) {
            return (UI->getParent() != I.getParent() ||
                    UI->getParent() == I.getParent() && UI->comesBefore(&I)) &&
                   OtherCFGBlocks.contains(UI->getParent());
          })) {
        HIPSYCL_DEBUG_ERROR << "[SubCFG] USE in another subcfg \n";
        // load from an alloca, just widen alloca
        if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(&I))
          if (auto *Alloca = utils::getLoopStateAllocaForLoad(*LInst)) {
            InstAllocaMap.insert({&I, Alloca});
            continue;
          }

        HIPSYCL_DEBUG_ERROR << "arrayifyMultiSubCfgValues: " << I << "\n";
        // GEP from already widened alloca: reuse alloca
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (GEP->hasMetadata(MDKind::Arrayified)) {
            // If we are on the subgroup level, then the gep operand might be an argument
            // and not directly an alloca.
            auto *GepPointerOperand = GEP->getPointerOperand();
            if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
              if (auto *Argument = llvm::dyn_cast<llvm::Argument>(GepPointerOperand)) {
                GepPointerOperand = (*HI.ArgsToAloca)[Argument];
              }
            }
            InstAllocaMap.insert({&I, llvm::cast<llvm::AllocaInst>(GepPointerOperand)});
            continue;
          }

        auto Shape = VecInfo.getVectorShape(I);
        HIPSYCL_DEBUG_ERROR << "VECTOR INFO: " << Shape << "\n";

#ifndef HIPSYCL_NO_PHIS_IN_SPLIT
        // if value is uniform, just store to 1-wide alloca
        if (Shape.isUniform()) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Value uniform, store to single element alloca " << I
                             << "\n";
          auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContiguousIdx, nullptr);
          InstAllocaMap.insert({&I, Alloca});
          VecInfo.setVectorShape(*Alloca, VectorShape::uni());
          continue;
        }
#endif
        // if contiguous, and can be recalculated, don't arrayify but store
        // uniform values and insts required for recalculation
        if (Shape.isContiguousOrStrided()) {
          if (dontArrayifyContiguousValues(I, BaseInstAllocaMap, ContInstReplicaMap, AllocaIP,
                                           ReqdArrayElements, ContiguousIdx, VecInfo)) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying " << I << "\n";
            continue;
          }
        }

        // create wide alloca and store the value
        auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContiguousIdx, ReqdArrayElements);
        InstAllocaMap.insert({&I, Alloca});
        VecInfo.setVectorShape(*Alloca, Shape);
      }
    }
  }
}

void remapInstruction(llvm::Instruction *I, llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<llvm::Value *, 8> WL{I->value_op_begin(), I->value_op_end()};
  for (auto *V : WL) {
    if (VMap.count(V))
      I->replaceUsesOfWith(V, VMap[V]);
  }
  HIPSYCL_DEBUG_INFO << "[SubCFG] remapped Inst " << *I << "\n";
}

// inserts loads from the loop state allocas for varying values that were identified as
// multi-subcfg values
void SubCFG::loadMultiSubCfgValues(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &LoadToAlloca) {
  llvm::Value *NewContIdx = VMap[HI.getContiguousIdx()];

  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::IRBuilder Builder{LoadTerm};

  for (auto &[Inst, Alloca] : InstAllocaMap) {
    // If def not in sub CFG but a use of it is in the sub CFG
    if (std::find(Blocks_.begin(), Blocks_.end(), Inst->getParent()) == Blocks_.end()) {
      if (utils::anyOfUsers<llvm::Instruction>(Inst, [this](llvm::Instruction *UI) {
            return std::find(NewBlocks_.begin(), NewBlocks_.end(), UI->getParent()) !=
                   NewBlocks_.end();
          })) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Inst)) {
          if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            llvm::Value *ContIdx = VMap[HI.ContiguousIdx];
            auto *Type = GEP->getSourceElementType();
            auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                Type, GEP->getPointerOperand(), ContIdx, GEP->getName() + "c"));
            NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
            VMap[Inst] = NewGEP;
            continue;
          }
        }
        auto *IP = LoadTerm;
        if (!Alloca->isArrayAllocation())
          IP = UniformLoadTerm;
        HIPSYCL_DEBUG_ERROR << "[SubCFG] Load from Alloca " << *Alloca << " in "
                            << IP->getParent()->getName() << "\n";
        auto *Load = utils::loadFromAlloca(Alloca, NewContIdx, IP, Inst->getName());
        LoadToAlloca[Load] = Alloca;

        utils::copyDgbValues(Inst, Load, IP);
        VMap[Inst] = Load;
      }
    }
  }
}

// Inserts loads for the multi-subcfg values that were identified as uniform
// inside the wi-loop preheader. Additionally clones the instructions that were
// identified as contiguous \a ContInstReplicaMap inside the LoadBB_ to restore
// the contiguous value just from the uniform values and the wi-idx.
void SubCFG::loadUniformAndRecalcContValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &LoadToAlloca) {
  llvm::ValueToValueMapTy UniVMap;
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  auto ContiguousIdx = HI.getContiguousIdx();
  llvm::Value *NewContIdx = VMap[ContiguousIdx];
  UniVMap[ContiguousIdx] = NewContIdx;

  // copy local id load value to univmap
  for (size_t D = 0; D < this->Dim; ++D) {
    auto *Load = getLoadForGlobalVariable(*this->LoadBB_->getParent(), LocalIdGlobalNames[D]);
    UniVMap[Load] = VMap[Load];
  }

  // load uniform values from allocas
  for (auto &InstAllocaPair : BaseInstAllocaMap) {
    auto *IP = UniformLoadTerm;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Load base value from Alloca " << *InstAllocaPair.second
                       << " in " << IP->getParent()->getName() << "\n";
    auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewContIdx, IP,
                                       InstAllocaPair.first->getName());
    LoadToAlloca[Load] = InstAllocaPair.second;
    utils::copyDgbValues(InstAllocaPair.first, Load, IP);
    UniVMap[InstAllocaPair.first] = Load;
  }

  // get a set of unique contiguous instructions
  llvm::SmallPtrSet<llvm::Instruction *, 16> UniquifyInsts;
  for (auto &Pair : ContInstReplicaMap) {
    UniquifyInsts.insert(Pair.first);
    for (auto &Target : Pair.second)
      UniquifyInsts.insert(Target);
  }

  auto OrderedInsts = topoSortInstructions(UniquifyInsts);

  llvm::SmallPtrSet<llvm::Instruction *, 16> InstsToRemap;
  // clone the contiguous instructions to restore the used values
  for (auto *I : OrderedInsts) {
    if (UniVMap.count(I))
      continue;

    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction and operands of: " << *I << " to "
                       << LoadTerm->getParent()->getName() << "\n";
    auto *IClone = I->clone();
    IClone->insertBefore(LoadTerm);
    InstsToRemap.insert(IClone);
    UniVMap[I] = IClone;
    if (VMap.count(I) == 0)
      VMap[I] = IClone;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction: " << *IClone << "\n";
  }

  // finally remap the singular instructions to use the other cloned contiguous instructions /
  // uniform values
  for (auto *IToRemap : InstsToRemap)
    remapInstruction(IToRemap, UniVMap);
}

llvm::SmallVector<llvm::Instruction *, 16> SubCFG::topoSortInstructions(
    const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const {
  llvm::SmallVector<llvm::Instruction *, 16> OrderedInsts(UniquifyInsts.size());
  std::copy(UniquifyInsts.begin(), UniquifyInsts.end(), OrderedInsts.begin());

  auto IsUsedBy = [](llvm::Instruction *LHS, llvm::Instruction *RHS) {
    for (auto *U : LHS->users()) {
      if (U == RHS)
        return true;
    }
    return false;
  };
  for (int I = 0; I < OrderedInsts.size(); ++I) {
    int InsertAt = I;
    for (int J = OrderedInsts.size() - 1; J > I; --J) {
      if (IsUsedBy(OrderedInsts[J], OrderedInsts[I])) {
        InsertAt = J;
        break;
      }
    }
    if (InsertAt != I) {
      auto *Tmp = OrderedInsts[I];
      for (int J = I + 1; J <= InsertAt; ++J) {
        OrderedInsts[J - 1] = OrderedInsts[J];
      }
      OrderedInsts[InsertAt] = Tmp;
      --I;
    }
  }
  return OrderedInsts;
}

llvm::BasicBlock *SubCFG::createUniformLoadBB(llvm::BasicBlock *OuterMostHeader) {
  auto *LoadBB = llvm::BasicBlock::Create(OuterMostHeader->getContext(),
                                          "uniloadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          OuterMostHeader->getParent(), OuterMostHeader);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(OuterMostHeader);
  return LoadBB;
}

llvm::BasicBlock *SubCFG::createLoadBB(llvm::ValueToValueMapTy &VMap) {
  auto *NewEntry = llvm::cast<llvm::BasicBlock>(static_cast<llvm::Value *>(VMap[EntryBB_]));
  auto *LoadBB = llvm::BasicBlock::Create(NewEntry->getContext(),
                                          "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          NewEntry->getParent(), NewEntry);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(NewEntry);
  return LoadBB;
}

// if the kernel contained a loop, it is possible, that values inside a single
// subcfg don't dominate their uses inside the same subcfg. This function
// identifies and fixes those values.
void SubCFG::fixSingleSubCfgValues(
    llvm::DominatorTree &DT,
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
    llvm::Value *ReqdArrayElements, VectorizationInfo &VecInfo,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &LoadToAlloca) {

  auto *AllocaIP = LoadBB_->getParent()->getEntryBlock().getTerminator();
  auto *LoadIP = LoadBB_->getTerminator();
  auto *UniLoadIP = PreHeader_->getTerminator();
  llvm::IRBuilder Builder{LoadIP};

  llvm::DenseMap<llvm::Instruction *, llvm::Instruction *> InstLoadMap;
  llvm::Value *ContiguousIdx = HI.getContiguousIdx();

  for (auto *BB : NewBlocks_) {
    llvm::SmallVector<llvm::Instruction *, 16> Insts{};
    std::transform(BB->begin(), BB->end(), std::back_inserter(Insts), [](auto &I) { return &I; });
    for (auto *Inst : Insts) {
      auto &I = *Inst;
      for (auto *OPV : I.operand_values()) {
        // check if all operands dominate the instruction -> otherwise we have to fix it
        if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV); OPI && !DT.dominates(OPI, &I)) {
          if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(Inst)) {
            // if a PHI node, we have to check that the incoming values dominate the terminators
            // of the incoming block..
            bool FoundIncoming = false;
            for (auto &Incoming : Phi->incoming_values()) {
              if (OPV == Incoming.get()) {
                auto *IncomingBB = Phi->getIncomingBlock(Incoming);
                if (DT.dominates(OPI, IncomingBB->getTerminator())) {
                  FoundIncoming = true;
                  break;
                }
              }
            }
            if (FoundIncoming)
              continue;
          }
          HIPSYCL_DEBUG_ERROR << "Instruction not dominated " << I << " operand: " << *OPI << "\n";

          if (auto *Load = InstLoadMap.lookup(OPI))
            // if the already inserted Load does not dominate I, we must create another load.
            if (DT.dominates(Load, &I)) {
              I.replaceUsesOfWith(OPI, Load);
              continue;
            }

          if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(OPI))
            if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
              // TODO what should happen, if F is the sub group function and the gep indexes is a
              // work group "array". ContiguousIdx is the subgroup induction variable.
              auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                  GEP->getType(), GEP->getPointerOperand(), ContiguousIdx, GEP->getName() + "c"));
              NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
              I.replaceUsesOfWith(OPI, NewGEP);
              InstLoadMap.insert({OPI, NewGEP});
              continue;
            }

          llvm::AllocaInst *Alloca = nullptr;
          if (auto *RemAlloca = RemappedInstAllocaMap.lookup(OPI))
            Alloca = RemAlloca;
          if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(OPI))
            if (auto *NewAlloca = utils::getLoopStateAllocaForLoad(*LInst))
              Alloca = NewAlloca;
          if (!Alloca) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] No alloca, yet for " << *OPI << "\n";
            Alloca = utils::arrayifyInstruction(
                AllocaIP, OPI, ContiguousIdx,
                VecInfo.getVectorShape(I).isUniform() ? nullptr : ReqdArrayElements);
            VecInfo.setVectorShape(*Alloca, VecInfo.getVectorShape(I));
          }

          auto Idx = ContiguousIdx;
#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          // in split loop, OPI might be used multiple times, get the user, dominating this user
          // and insert load there
          llvm::Instruction *NewIP = &I;
          for (auto *U : OPI->users()) {
            if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U); UI && DT.dominates(UI, NewIP)) {
              NewIP = UI;
            }
          }
#else
          // doesn't happen if we keep the PHIs
          auto *NewIP = LoadIP;
          if (!Alloca->isArrayAllocation()) {
            NewIP = UniLoadIP;
            Idx = llvm::ConstantInt::get(ContiguousIdx->getType(), 0);
          }
#endif

          auto *Load = utils::loadFromAlloca(Alloca, Idx, NewIP, OPI->getName());
          LoadToAlloca[Load] = Alloca;

          utils::copyDgbValues(OPI, Load, NewIP);

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          I.replaceUsesOfWith(OPI, Load);
          InstLoadMap.insert({OPI, Load});
#else
          // if a loop is conditionally split, the first block in a subcfg might have another
          // incoming edge, need to insert a PHI node then
          const auto NumPreds = std::distance(llvm::pred_begin(BB), llvm::pred_end(BB));
          if (!llvm::isa<llvm::PHINode>(I) && NumPreds > 1 &&
              std::find(llvm::pred_begin(BB), llvm::pred_end(BB), LoadBB_) != llvm::pred_end(BB)) {
            Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
            auto *PHINode = Builder.CreatePHI(Load->getType(), NumPreds, I.getName());
            for (auto *PredBB : llvm::predecessors(BB))
              if (PredBB == LoadBB_)
                PHINode->addIncoming(Load, PredBB);
              else
                PHINode->addIncoming(OPV, PredBB);

            I.replaceUsesOfWith(OPI, PHINode);
            InstLoadMap.insert({OPI, PHINode});
          } else {
            I.replaceUsesOfWith(OPI, Load);
            InstLoadMap.insert({OPI, Load});
          }
#endif
        }
      }
    }
  }
}

llvm::BasicBlock *createUnreachableBlock(llvm::Function &F) {
  auto *Default = llvm::BasicBlock::Create(F.getContext(), "cbs.while.default", &F);
  llvm::IRBuilder Builder{Default, Default->getFirstInsertionPt()};
  Builder.CreateUnreachable();
  return Default;
}

// create the actual while loop around the subcfgs and the switch instruction to
// select the next subCFG based on the value in \a LastBarrierIdStorage
llvm::BasicBlock *generateWhileSwitchAround(llvm::BasicBlock *PreHeader, llvm::BasicBlock *OldEntry,
                                            llvm::BasicBlock *Exit,
                                            llvm::AllocaInst *LastBarrierIdStorage,
                                            std::vector<SubCFG> &SubCFGs) {
  auto &F = *PreHeader->getParent();
  auto &M = *F.getParent();
  const auto &DL = M.getDataLayout();

  auto *WhileHeader = llvm::BasicBlock::Create(PreHeader->getContext(), "cbs.while.header",
                                               PreHeader->getParent(), OldEntry);
  llvm::IRBuilder Builder{WhileHeader, WhileHeader->getFirstInsertionPt()};
  auto *LastID = Builder.CreateLoad(LastBarrierIdStorage->getAllocatedType(), LastBarrierIdStorage,
                                    "cbs.while.last_barr.load");
  auto *Switch = Builder.CreateSwitch(LastID, createUnreachableBlock(F), SubCFGs.size());
  for (auto &Cfg : SubCFGs) {
    Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), Cfg.getEntryId()),
                    Cfg.getEntry());
    Cfg.getEntry()->replacePhiUsesWith(PreHeader, WhileHeader);
    Cfg.getExit()->getTerminator()->replaceSuccessorWith(Exit, WhileHeader);
  }
  Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), ExitBarrierId), Exit);

  Builder.SetInsertPoint(PreHeader->getTerminator());
  Builder.CreateStore(
      llvm::ConstantInt::get(LastBarrierIdStorage->getAllocatedType(), EntryBarrierId),
      LastBarrierIdStorage);
  PreHeader->getTerminator()->replaceSuccessorWith(OldEntry, WhileHeader);
  return WhileHeader;
}

// drops all lifetime intrinsics - they are misinforming ASAN otherwise (and are
// not really fixable at the right scope..)
void purgeLifetime(SubCFG &Cfg) {
  llvm::SmallVector<llvm::Instruction *, 8> ToDelete;
  for (auto *BB : Cfg.getNewBlocks())
    for (auto &I : *BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction())
          if (CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
              CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_end)
            ToDelete.push_back(CI);

  for (auto *I : ToDelete)
    I->eraseFromParent();
}

// fills \a Hull with all transitive users of \a Alloca
void fillUserHull(llvm::Value *Alloca, llvm::SmallVectorImpl<llvm::Instruction *> &Hull) {
  llvm::SmallVector<llvm::Instruction *, 8> WL;
  std::transform(Alloca->user_begin(), Alloca->user_end(), std::back_inserter(WL),
                 [](auto *U) { return llvm::cast<llvm::Instruction>(U); });
  llvm::SmallPtrSet<llvm::Instruction *, 32> AlreadySeen;
  while (!WL.empty()) {
    auto *I = WL.pop_back_val();
    AlreadySeen.insert(I);
    Hull.push_back(I);
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!AlreadySeen.contains(UI))
          if (UI->mayReadOrWriteMemory() || UI->getType()->isPointerTy())
            WL.push_back(UI);
      }
    }
  }
}

// checks if all uses of an alloca are in just a single subcfg (doesn't have to be arrayified!)
bool isAllocaSubCfgInternal(llvm::Value *Alloca, const std::vector<SubCFG> &SubCfgs,
                            const llvm::DominatorTree &DT) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> UserBlocks;
  {
    llvm::SmallVector<llvm::Instruction *, 32> Users;
    fillUserHull(Alloca, Users);
    utils::PtrSetWrapper<decltype(UserBlocks)> Wrapper{UserBlocks};
    std::transform(Users.begin(), Users.end(), std::inserter(Wrapper, UserBlocks.end()),
                   [](auto *I) { return I->getParent(); });
  }

  for (auto &SubCfg : SubCfgs) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> SubCfgSet{SubCfg.getNewBlocks().begin(),
                                                       SubCfg.getNewBlocks().end()};
    if (std::any_of(UserBlocks.begin(), UserBlocks.end(),
                    [&SubCfgSet](auto *BB) { return SubCfgSet.contains(BB); }) &&
        !std::all_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet, Alloca](auto *BB) {
          if (SubCfgSet.contains(BB)) {
            return true;
          }
          HIPSYCL_DEBUG_INFO << "[SubCFG] BB not in subcfgset: " << BB->getName()
                             << " for alloca: ";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alloca->print(llvm::outs()); llvm::outs() << "\n";)
          return false;
        }))
      return false;
  }

  return true;
}

// Widens the allocas in the entry block to array allocas.
// Replace uses of the original alloca with GEP that indexes the new alloca with
// \a Idx.
void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT,
                     std::vector<SubCFG> &SubCfgs, llvm::Value *ReqdArrayElements,
                     VectorizationInfo &VecInfo, llvm::Function &F, HierarchicalSplitInfo HI) {
  auto *MDAlloca = llvm::MDNode::get(
      EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), MDKind::LoopState)});

  llvm::SmallPtrSet<llvm::BasicBlock *, 32> SubCfgsBlocks;
  for (auto &SubCfg : SubCfgs)
    SubCfgsBlocks.insert(SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end());
  {
    llvm::SmallVector<llvm::AllocaInst *, 8> WL;
    for (auto &I : *EntryBlock) {
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
          continue; // already arrayified
        if (utils::anyOfUsers<llvm::Instruction>(Alloca, [&SubCfgsBlocks](llvm::Instruction *UI) {
              return !SubCfgsBlocks.contains(UI->getParent());
            }))
          continue;
        if (!isAllocaSubCfgInternal(Alloca, SubCfgs, DT))
          WL.push_back(Alloca);
      }
    }

    for (auto *I : WL) {
      llvm::IRBuilder AllocaBuilder{I};
      llvm::Type *T = I->getAllocatedType();
      if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
        auto ArrSize = ArrSizeC->getLimitedValue();
        if (ArrSize > 1) {
          T = llvm::ArrayType::get(T, ArrSize);
          HIPSYCL_DEBUG_WARNING << "Caution, alloca was array\n";
        }
      }

      auto *Alloca = AllocaBuilder.CreateAlloca(T, ReqdArrayElements, I->getName() + "_alloca");
      Alloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
      Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      for (auto &SubCfg : SubCfgs) {
        auto *GepIp = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();
        auto *ContiguousIdx = SubCfg.getHI().getContiguousIdx();

        llvm::IRBuilder LoadBuilder{GepIp};
        auto *GEP = llvm::cast<llvm::GetElementPtrInst>(LoadBuilder.CreateInBoundsGEP(
            Alloca->getAllocatedType(), Alloca, ContiguousIdx, I->getName() + "_gep"));
        GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

        llvm::replaceDominatedUsesWith(I, GEP, DT, SubCfg.getLoadBB());
      }
      I->eraseFromParent();
    }
  }

  // If alloca is only used in SUBGROUPS that are between the same work group barriers, then
  // the alloca is not arrayified on the work group level.
  // Since, the non arrayfied alloca is still in the entry block of the WORKGROUP function, we do
  // not see the alloca. Thus, we need to look at the arguments of the SUBGROUP level function,
  // and if they are non arrayified alloca, then we arrayify them.
  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    std::vector<std::pair<llvm::AllocaInst *, llvm::Argument *>> WL;
    for (auto [Arg, Alloca] : *HI.ArgsToAloca) {
      // already arrayified
      if (Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
        continue;

      if (!isAllocaSubCfgInternal(Arg, SubCfgs, DT)) {
        WL.emplace_back(Alloca, Arg);
      }
    }

    // One of the problems is that the allocas are still stored in the entry of the work group
    // function. Thus in the current function (sub group level function) we only see the argument.
    for (auto [Alloca, Arg] : WL) {
      llvm::IRBuilder AllocaBuilder{Alloca};
      llvm::Type *T = Alloca->getAllocatedType();
      if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(Alloca->getArraySize())) {
        auto ArrSize = ArrSizeC->getLimitedValue();
        if (ArrSize > 1) {
          T = llvm::ArrayType::get(T, ArrSize);
          HIPSYCL_DEBUG_WARNING << "Caution, alloca was array\n";
        }
      }

      auto *ArrayifiedAlloca =
          AllocaBuilder.CreateAlloca(T, ReqdArrayElements, Alloca->getName() + "_alloca");
      ArrayifiedAlloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
      ArrayifiedAlloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      for (auto &SubCfg : SubCfgs) {
        auto *GepIp = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();
        auto *ContiguousIdx = SubCfg.getHI().SGIdArg;

        llvm::IRBuilder LoadBuilder{GepIp};
        auto *GEP = llvm::cast<llvm::GetElementPtrInst>(LoadBuilder.CreateInBoundsGEP(
            Alloca->getAllocatedType(), Arg, ContiguousIdx, Alloca->getName() + "_gep"));
        GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

        llvm::replaceDominatedUsesWith(Arg, GEP, DT, SubCfg.getLoadBB());
      }
      // I can do this, because alloca is only used in the current subgroup function
      Alloca->replaceAllUsesWith(ArrayifiedAlloca);
      Alloca->eraseFromParent();
    }
  }
}

void moveAllocasToEntry(llvm::Function &F, llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  llvm::SmallVector<llvm::AllocaInst *, 4> AllocaWL;
  for (auto *BB : Blocks)
    for (auto &I : *BB)
      if (auto *AllocaInst = llvm::dyn_cast<llvm::AllocaInst>(&I))
        AllocaWL.push_back(AllocaInst);
  for (auto *I : AllocaWL)
    I->moveBefore(F.getEntryBlock().getTerminator());
}

llvm::DenseMap<llvm::BasicBlock *, size_t>
getBarrierIds(llvm::BasicBlock *Entry, llvm::SmallPtrSetImpl<llvm::BasicBlock *> &ExitingBlocks,
              llvm::ArrayRef<llvm::BasicBlock *> Blocks, const SplitterAnnotationInfo &SAA,
              HierarchicalSplitInfo HI) {
  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;
  // mark exit barrier with the corresponding id:
  for (auto *BB : ExitingBlocks)
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[Entry] = EntryBarrierId;

  // store all other barrier blocks with a unique id:
  size_t BarrierId = 1;
  auto hasOnlyBarrier = HI.Level == HierarchicalLevel::H_CBS_SUBGROUP ? utils::hasOnlySubBarrier
                                                                      : utils::hasOnlyBarrier;

  for (auto *BB : Blocks)
    if (Barriers.find(BB) == Barriers.end() && hasOnlyBarrier(BB, SAA))
      Barriers.insert({BB, BarrierId++});
  return Barriers;
}

namespace {
class CBSIntrinsic {
public:
  void vectorizeAllInstances(
      llvm::Function &F, std::vector<SubCFG> &SubCfgs,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont =
          nullptr) && {
    for (auto *Intrinsic : getIntrinsic(F, getName())) {
      auto *V = vectorizeFunction(F, *Intrinsic, findSubCfg(SubCfgs, Intrinsic), cont);
      Intrinsic->replaceAllUsesWith(V);
      Intrinsic->eraseFromParent();
    }
  }

protected:
  enum class Shape { VARYING, UNIFORM };

private:
  llvm::Value *vectorizeFunction(
      llvm::Function &F, llvm::CallInst &Intrinsic, SubCFG &SubCfg,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont) {
    llvm::IRBuilder Builder{F.getContext()};
    // First argument is always the data argument
    const auto *Op0 = Intrinsic.getOperand(0);

    auto *UniformBlock = SubCfg.getEntry();
    auto *TerminatorInstruction = UniformBlock->getTerminator();
    Builder.SetInsertPoint(TerminatorInstruction);

    std::pair<llvm::Value *, Shape> res{};
    if (auto [Storage, Type] = getOrCreateValue(Builder, SubCfg, Intrinsic, cont);
        Type == Shape::UNIFORM) {
      if (not llvm::dyn_cast<llvm::Constant>(Storage)) {
        Storage = Builder.CreateLoad(Intrinsic.getFunctionType()->getReturnType(), Storage);
      }
      res = vectorizeUniformValue(Storage, Builder, Intrinsic);
    } else {
      auto *VType = llvm::VectorType::get(Op0->getType(), llvm::ElementCount::getFixed(SGSize));
      auto *VLoad = Builder.CreateLoad(VType, Storage, "vec_load");
      res = vectorizeValue(VLoad, Builder, Intrinsic);
    }

    if (res.second == Shape::UNIFORM) {
      return res.first;
    }

    Builder.SetInsertPoint(&Intrinsic);
    return extractElement(Builder, res.first, SubCfg.getHI().SGIdArg, Intrinsic);
  }

  static std::pair<llvm::Value *, Shape> getOrCreateValue(
      llvm::IRBuilder<> &Builder, SubCFG &SubCfg, llvm::CallInst &Intrinsic,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont) {
    if (auto *Constant = llvm::dyn_cast<llvm::Constant>(Intrinsic.getOperand(0))) {
      return {Constant, Shape::UNIFORM};
    }

    auto *Load = llvm::dyn_cast<llvm::LoadInst>(Intrinsic.getOperand(0));

    // Contiguous case
    if (not Load) {
      llvm::SmallVector<llvm::Instruction *, 4> WL;
      llvm::SmallPtrSet<llvm::Instruction *, 16> Result;
      llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;

      auto *FirstInst = llvm::dyn_cast<llvm::Instruction>(Intrinsic.getOperand(0));
      assert(FirstInst);
      WL.emplace_back(FirstInst);
      while (not WL.empty()) {
        auto *Inst = WL.pop_back_val();
        if (LookedAt.contains(Inst)) {
          continue;
        }
        LookedAt.insert(Inst);
        Result.insert(Inst);

        if (Inst == SubCfg.getHI().SGIdArg) {
          continue;
        }

        for (auto &Operand : Inst->operands()) {
          if (!llvm::isa<llvm::Instruction>(Operand)) {
            continue;
          }
          WL.emplace_back(llvm::dyn_cast<llvm::Instruction>(Operand));
        }
      }
      auto OrderedInsts = SubCfg.topoSortInstructions(Result);

      // TODO incomplete sub-groups
      auto *LoopHead =
          Builder.GetInsertBlock()->splitBasicBlock(Builder.GetInsertPoint(), "UniformLoopHead");
      auto *LoopBody = LoopHead->splitBasicBlock(LoopHead->begin(), "LoopBody");
      auto *LoopLatch = LoopBody->splitBasicBlock(LoopBody->begin(), "LoopLatch");
      auto *NewUniformBlock = LoopLatch->splitBasicBlock(LoopLatch->begin(), "NewUniformBlock");

      // Induction variable
      Builder.SetInsertPoint(LoopHead->getFirstNonPHI());
      llvm::AllocaInst *Alloca = Builder.CreateAlloca(llvm::Type::getInt64Ty(Builder.getContext()));
      Builder.CreateStore(llvm::ConstantInt::get(Alloca->getAllocatedType(), 0, false), Alloca);

      // Loop Latch
      Builder.SetInsertPoint(LoopLatch->getFirstNonPHI());
      auto *I = Builder.CreateLoad(Alloca->getAllocatedType(), Alloca);
      auto *I2 = Builder.CreateAdd(I, llvm::ConstantInt::get(I->getType(), 1, false));
      Builder.CreateStore(I2, Alloca);
      auto *Cond = Builder.CreateICmpSLT(I, llvm::ConstantInt::get(I->getType(), SGSize, false));
      Builder.CreateCondBr(Cond, LoopBody, NewUniformBlock);
      LoopLatch->getTerminator()->eraseFromParent();

      // LoopHead Alloca
      Builder.SetInsertPoint(LoopHead->getFirstNonPHI());
      auto *Value = Builder.CreateAlloca(
          OrderedInsts.back()->getType(),
          llvm::ConstantInt::getIntegerValue(
              SubCfg.getHI().SGIdArg->getType(),
              llvm::APInt(SubCfg.getHI().SGIdArg->getType()->getIntegerBitWidth(), SGSize)));

      // Replicate instructions
      llvm::SmallPtrSet<llvm::Instruction *, 16> InstsToRemap;
      llvm::ValueToValueMapTy UniVMap;

      Builder.SetInsertPoint(LoopBody->getFirstNonPHI());
      auto *LoadInductionVar = Builder.CreateLoad(Alloca->getAllocatedType(), Alloca);

      UniVMap[SubCfg.getHI().SGIdArg] = LoadInductionVar;
      UniVMap[SubCfg.getHI().ContiguousIdx] = Builder.CreateAdd(
          LoadInductionVar,
          llvm::dyn_cast<llvm::Instruction>(SubCfg.getHI().ContiguousIdx)->getOperand(1));

      llvm::Instruction *LastInstruction{};
      for (auto *Inst : OrderedInsts) {
        LastInstruction = Inst;
        if (Inst == SubCfg.getHI().SGIdArg || Inst == SubCfg.getHI().ContiguousIdx) {
          continue;
        }
        auto *IClone = Inst->clone();
        Builder.Insert(IClone);
        UniVMap[Inst] = IClone;
        InstsToRemap.insert(IClone);
      }
      LastInstruction = llvm::dyn_cast<llvm::Instruction>(UniVMap[LastInstruction]);
      assert(LastInstruction);
      for (auto *IToRemap : InstsToRemap)
        remapInstruction(IToRemap, UniVMap);

      // Insert LastInstruction into vector
      Builder.CreateStore(LastInstruction,
                          Builder.CreateGEP(Value->getAllocatedType(), Value, {LoadInductionVar}));

      Builder.SetInsertPoint(NewUniformBlock->getTerminator());
      return {Value, Shape::VARYING};
    }
    assert(Load and "Op0 must be load inst");

    // NOT UNIFORM
    if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Load->getPointerOperand())) {
      auto *Storage = GEP->getPointerOperand();

      // IS SUBGROUP LOCAL
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Storage)) {
        auto *Size = llvm::dyn_cast<llvm::ConstantInt>(Alloca->getArraySize());
        assert(Size);
        assert(Size->getSExtValue() == SGSize);
        return {Storage, Shape::VARYING};
      } else {
        assert(llvm::dyn_cast<llvm::Argument>(Storage));
        auto *SgIdx = llvm::dyn_cast<llvm::Instruction>(SubCfg.getHI().ContiguousIdx);
        assert(SgIdx);
        auto *GroupIdx = SgIdx->getOperand(1);
        return {Builder.CreateGEP(Intrinsic.getArgOperand(0)->getType(), Storage, {GroupIdx}),
                Shape::VARYING};
      }
    }
    // UNIFORM (SUB_GROUP_LOCAL)
    else if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Load->getPointerOperand())) {
      llvm::outs() << "UNIFORM\n";
      // IS SUBGROUP LOCAL
      auto *Size = llvm::dyn_cast<llvm::ConstantInt>(Alloca->getArraySize());
      assert(Size);
      // assert(Size->getSExtValue() == SGSize);
      return {Alloca, Shape::UNIFORM};
    } else {
      assert(llvm::dyn_cast<llvm::Argument>(Load->getPointerOperand()));
      return {Load->getPointerOperand(), Shape::UNIFORM};
    }
    llvm::outs() << "ERROR\n";
    std::exit(1);
  }

  static SubCFG &findSubCfg(std::vector<SubCFG> &SubCfgs, llvm::Instruction *I) {
    auto it = std::find_if(SubCfgs.begin(), SubCfgs.end(), [&](const SubCFG &subCfg) {
      return std::any_of(subCfg.getNewBlocks().begin(), subCfg.getNewBlocks().end(),
                         [&](const llvm::BasicBlock *BB) { return BB == I->getParent(); });
    });
    assert(it != SubCfgs.end());
    return *it;
  }

  static llvm::SmallVector<llvm::CallInst *, 8> getIntrinsic(llvm::Function &F,
                                                             std::string_view str) {
    llvm::SmallVector<llvm::CallInst *, 8> Intrinsics{};
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
          if (auto *CalledF = CallInst->getCalledFunction()) {
            if (CalledF->getName().contains(str))
              Intrinsics.emplace_back(CallInst);
          }
        }
      }
    }
    return Intrinsics;
  }

  virtual std::string_view getName() = 0;
  virtual std::pair<llvm::Value *, Shape> vectorizeUniformValue(llvm::Value *Storage,
                                                                llvm::IRBuilder<> &Builder,
                                                                llvm::CallInst &Intrinsic) = 0;
  virtual std::pair<llvm::Value *, Shape>
  vectorizeValue(llvm::LoadInst *VLoad, llvm::IRBuilder<> &Builder, llvm::CallInst &Intrinsic) = 0;
  virtual llvm::Value *extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                                      llvm::Value *SgInductionVariable, llvm::CallInst &Intrinsic) {
    return Builder.CreateExtractElement(Value, SgInductionVariable);
  }
};

class ReduceIntrinsic final : public CBSIntrinsic {
  std::string_view getName() override { return "__cbs_reduce"; }

  std::pair<llvm::Value *, Shape> vectorizeUniformValue(llvm::Value *Storage,
                                                        llvm::IRBuilder<> &Builder,
                                                        llvm::CallInst &Intrinsic) override {
    auto *Idx = llvm::dyn_cast<llvm::ConstantInt>(Intrinsic.getOperand(1));
    assert(Idx and "Op must be constant int");
    const auto v = Idx->getSExtValue();
    // ADD
    if (v == 0) {
      return {Builder.CreateMul(Storage,
                                llvm::ConstantInt::get(
                                    Builder.getContext(),
                                    llvm::APInt(Storage->getType()->getIntegerBitWidth(), SGSize))),
              Shape::UNIFORM};
    }
    // MUL
    if (v == 1) {
      // POW(, 32)
      llvm::Value *result = Storage;
      for (auto i = 1ul; i < SGSize; ++i) {
        result = Builder.CreateMul(result, Storage);
      }
      return {result, Shape::UNIFORM};
    }
    // min
    if (v == 2) {
      return {Storage, Shape::UNIFORM};
    }
    // max
    if (v == 3) {
      return {Storage, Shape::UNIFORM};
    }
    assert(false);
    return {};
  }

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::LoadInst *VLoad, llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override {
    const auto *Idx = llvm::dyn_cast<llvm::ConstantInt>(Intrinsic.getOperand(1));
    assert(Idx and "Op must be constant int");
    const auto v = Idx->getSExtValue();
    if (v == 0) {
      return {Builder.CreateAddReduce(VLoad), Shape::UNIFORM};
    }
    if (v == 1) {
      return {Builder.CreateMulReduce(VLoad), Shape::UNIFORM};
    }
    if (v == 2) {
      return {Builder.CreateIntMinReduce(VLoad), Shape::UNIFORM};
    }
    if (v == 3) {
      return {Builder.CreateIntMaxReduce(VLoad), Shape::UNIFORM};
    }
    assert(false);
    return {};
  }
};

template <bool Left> class Shift final : public CBSIntrinsic {
  std::string_view getName() override { return Left ? "__cbs_shift_left" : "__cbs_shift_right"; }
  std::pair<llvm::Value *, Shape> vectorizeUniformValue(llvm::Value *Storage,
                                                        llvm::IRBuilder<> &Builder,
                                                        llvm::CallInst &Intrinsic) override {
    return {Storage, Shape::UNIFORM};
  }
  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::LoadInst *VLoad, llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override {
    auto *Idx = Intrinsic.getOperand(1);
    assert(Idx and "Op must be constant int");
    if (auto *Op1V = llvm::dyn_cast<llvm::ConstantInt>(Idx)) {
      std::array<int, SGSize> mask{};
      {
        std::iota(mask.begin(), mask.end(), 0);
        for (auto &x : mask) {
          x = x + Op1V->getSExtValue();
          if (x >= SGSize) {
            x = 0;
          }
        }
      }
      return {Builder.CreateShuffleVector(VLoad, mask), Shape::VARYING};
    } else {
      return {VLoad, Shape::UNIFORM};
    }
  }

  llvm::Value *extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                              llvm::Value *SgInductionVariable,
                              llvm::CallInst &Intrinsic) override {
    auto *Op1 = Intrinsic.getOperand(1);
    auto *Idx = Builder.CreateURem(
        Left ? Builder.CreateAdd(SgInductionVariable, Op1)
             : Builder.CreateSub(SgInductionVariable, Op1),
        llvm::ConstantInt::get(
            Builder.getContext(),
            llvm::APInt(SgInductionVariable->getType()->getIntegerBitWidth(), SGSize)));

    return Builder.CreateExtractElement(Value, Idx);
  }
};

class ShuffleIntrinsic final : public CBSIntrinsic {
  std::string_view getName() override { return "__cbs_shuffle"; }

  std::pair<llvm::Value *, Shape> vectorizeUniformValue(llvm::Value *Storage,
                                                        llvm::IRBuilder<> &Builder,
                                                        llvm::CallInst &Intrinsic) override {
    return {Storage, Shape::UNIFORM};
  }

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::LoadInst *VLoad, llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override {
    auto *Idx = Intrinsic.getOperand(1);
    if (auto *Op1V = llvm::dyn_cast<llvm::ConstantInt>(Idx)) {
      return {Builder.CreateExtractElement(VLoad, Op1V), Shape::UNIFORM};
    } else {
      return {VLoad, Shape::VARYING};
    }
  }

  llvm::Value *extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                              llvm::Value *SgInductionVariable,
                              llvm::CallInst &Intrinsic) override {
    auto *Op1 = Intrinsic.getOperand(1);
    return Builder.CreateExtractElement(Value, Op1);
  }
};

class ExtractIntrinsic final : public CBSIntrinsic {
  std::string_view getName() override { return "__cbs_extract"; }

  std::pair<llvm::Value *, Shape> vectorizeUniformValue(llvm::Value *Storage,
                                                        llvm::IRBuilder<> &Builder,
                                                        llvm::CallInst &Intrinsic) override {
    return {Storage, Shape::UNIFORM};
  }

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::LoadInst *VLoad, llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override {
    auto *Idx = Intrinsic.getOperand(1);
    return {Builder.CreateExtractElement(VLoad, Idx), Shape::UNIFORM};
  }
};

} // namespace

void formSubCfgGeneric(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, bool IsSscp,
                       size_t Dim, llvm::ArrayRef<llvm::Value *> LocalSize,
                       llvm::Value *ReqdArrayElements, HierarchicalSplitInfo HI);

void formSubgroupCfgs(SubCFG &Cfg, llvm::Function &F, const SplitterAnnotationInfo &SAA,
                      bool IsSscp, size_t Dim, llvm::ArrayRef<llvm::Value *> LocalSize,
                      HierarchicalSplitInfo HI) {
  auto Blocks = Cfg.getNewBlocks(); // copy
  Blocks.insert(Blocks.begin(), Cfg.getLoadBB());

  llvm::SetVector<llvm::Value *> Inputs, Outputs;
  llvm::CodeExtractorAnalysisCache CEAC{F};
  llvm::CodeExtractor CE{Blocks};
  assert(CE.isEligible());
  auto *NewF = CE.extractCodeRegion(CEAC, Inputs, Outputs);

  // Create a mapping between the old instructions (which now may have become
  // function arguments)
  // InAndOutToArgs: Old Instruction -> Function Argument
  llvm::ValueToValueMapTy InAndOutToArgs;
  llvm::SmallDenseMap<llvm::Argument *, llvm::AllocaInst *, 8> ArgsToAlloca;
  {
    HIPSYCL_DEBUG_INFO << "Inputs:"
                       << "\n";
    int Cnter = 0;
    for (auto I : Inputs) {
      HIPSYCL_DEBUG_INFO << *I << " -> " << *NewF->getArg(Cnter) << "\n";
      InAndOutToArgs[I] = NewF->getArg(Cnter);
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(I)) {
        ArgsToAlloca[NewF->getArg(Cnter)] = Alloca;
      }
      Cnter++;
    }
    HIPSYCL_DEBUG_INFO << "Outputs:"
                       << "\n";
    for (auto O : Outputs) {
      HIPSYCL_DEBUG_INFO << *O << " -> " << *NewF->getArg(Cnter) << "\n";
      InAndOutToArgs[O] = NewF->getArg(Cnter);
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(O)) {
        ArgsToAlloca[NewF->getArg(Cnter)] = Alloca;
      }
      Cnter++;
    }
  }

  // Create barriers at the beginning and end of the cfg
  {
    utils::createSubBarrier(NewF->getEntryBlock().getTerminator(),
                            const_cast<SplitterAnnotationInfo &>(SAA));
    for (auto &BB : *NewF)
      if (BB.getTerminator()->getNumSuccessors() == 0)
        utils::createSubBarrier(BB.getTerminator(), const_cast<SplitterAnnotationInfo &>(SAA));
  }

  llvm::DominatorTree NewDT{*NewF};
  llvm::PostDominatorTree NewPDT{*NewF};
  llvm::LoopInfo NewLI{NewDT};

  // Because we now work on a new function we can not directly access the old induction variables
  // or local sizes. These values are now either function arguments or do not exists at all
  // (because they are not used in the new function).
  // Thus, we now replace them with the fuction arguments or load them from their respective
  // global variables.
  llvm::SmallVector<llvm::Value *, 3> NewIndVars;
  llvm::SmallVector<llvm::Value *, 3> NewLocalSize;
  {
    std::transform(Cfg.getWIIndVars().begin(), Cfg.getWIIndVars().end(),
                   std::back_inserter(NewIndVars),
                   [&InAndOutToArgs, NewF, D = 0](llvm::Value *V) mutable -> llvm::Value * {
                     D++;
                     HIPSYCL_DEBUG_ERROR << "index mapping: " << *V << "\n";

                     // TODO How can the old induction variables be in the new function?
                     if (auto I = llvm::dyn_cast<llvm::Instruction>(V);
                         I && I->getParent()->getParent() == NewF) {
                       HIPSYCL_DEBUG_ERROR << " meself --> " << *I << "\n";
                       return I;
                     }
                     if (const auto It = InAndOutToArgs.find(V); It != InAndOutToArgs.end()) {
                       HIPSYCL_DEBUG_ERROR << " --> " << *It->second << "\n";
                       return It->second;
                     }
                     HIPSYCL_DEBUG_ERROR << "meh --> "
                                         << *mergeGVLoadsInEntry(*NewF, LocalIdGlobalNames[D - 1])
                                         << "\n";

                     return mergeGVLoadsInEntry(*NewF, LocalIdGlobalNames[D - 1]);
                   });

    std::transform(LocalSize.begin(), LocalSize.end(), std::back_inserter(NewLocalSize),
                   [&InAndOutToArgs, NewF, D = 0](llvm::Value *V) mutable -> llvm::Value * {
                     HIPSYCL_DEBUG_ERROR << "LocalSize get from vmap: " << *V << "\n";
                     ++D;
                     if (auto It = InAndOutToArgs.find(V); It != InAndOutToArgs.end()) {
                       HIPSYCL_DEBUG_INFO << " --> VMapped : " << *It->second << "\n";
                       return It->second;
                     }
                     HIPSYCL_DEBUG_INFO << " --> load from global: " << *V << "\n";
                     // TODO why with type & what has it to do with GlobalVarToIdxMap
                     return mergeGVLoadsInEntry(*NewF, LocalSizeGlobalNames[D - 1], V->getType());
                   });
  }

  auto GetFromArgsOrDummyLoad = [&InAndOutToArgs, NewF](llvm::Value *V) -> llvm::Value * {
    HIPSYCL_DEBUG_ERROR << "get from vmap: " << *V << "\n";
    if (auto It = InAndOutToArgs.find(V); It != InAndOutToArgs.end()) {
      HIPSYCL_DEBUG_ERROR << " --> VMapped : " << *It->second << "\n";
      return It->second;
    }
    HIPSYCL_DEBUG_ERROR << " --> load from undef ptr: " << *V << "\n";
    return llvm::IRBuilder{NewF->getEntryBlock().getFirstNonPHI()}.CreateLoad(
        V->getType(), llvm::UndefValue::get(llvm::PointerType::get(V->getType(), 0)));
  };

  llvm::Value *NewIndVar = GetFromArgsOrDummyLoad(HI.ContiguousIdx);
  llvm::Value *SGIdArg = GetFromArgsOrDummyLoad(mergeGVLoadsInEntry(F, SgIdGlobalName));

  // Replace SGIdArg (and all its uses) with load from global variable (SgIdGlobalName)
  // This makes the handling of SgId simpler.
  {
    HIPSYCL_DEBUG_ERROR << "SGIDArg: " << *SGIdArg << "\n";
    for (auto U : SGIdArg->users()) {
      HIPSYCL_DEBUG_ERROR << "SGIDArg user: " << *U << "\n";
    }
    SGIdArg->replaceAllUsesWith(mergeGVLoadsInEntry(*NewF, SgIdGlobalName, SGIdArg->getType()));
    SGIdArg = mergeGVLoadsInEntry(*NewF, SgIdGlobalName);
    HIPSYCL_DEBUG_ERROR << "New SGIDArg: " << *SGIdArg << "\n";
  }

  formSubCfgGeneric(*NewF, NewLI, NewDT, NewPDT, SAA, IsSscp, Dim,
                    {llvm::ConstantInt::get(LocalSize[0]->getType(), SGSize)},
                    llvm::ConstantInt::get(LocalSize[0]->getType(), SGSize),
                    {HierarchicalLevel::H_CBS_SUBGROUP, NewLocalSize, NewIndVars, NewIndVar,
                     SGIdArg, &ArgsToAlloca});

  // The SgIdArg in NewF should not have any users.
  // They should have been replaced with the subgroup induction variable
  assert(SGIdArg->getNumUses() == 0);

  assert(std::distance(NewF->user_begin(), NewF->user_end()) == 1);
  utils::checkedInlineFunction(llvm::cast<llvm::CallBase>(NewF->user_back()), "[SubCFG]");
  NewF->eraseFromParent();
}

void formSubCfgGeneric(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, bool IsSscp,
                       size_t Dim, llvm::ArrayRef<llvm::Value *> LocalSize,
                       llvm::Value *ReqdArrayElements, HierarchicalSplitInfo HI) {
  auto *Entry = &F.getEntryBlock();

  std::vector<llvm::BasicBlock *> Blocks;
  {
    Blocks.reserve(std::distance(F.begin(), F.end()));
    std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });
  }

  // non-entry block Allocas are considered broken, move to entry.
  moveAllocasToEntry(F, Blocks);
  mergeGVLoadsInEntry(F, SgIdGlobalName);

  auto RImpl = getRegion(F, LI, Blocks);
  Region R{*RImpl};
  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, Dim, HI);
  VecInfo.setPinnedShape(*HI.ContiguousIdx, VectorShape::cont());

  llvm::SmallPtrSet<llvm::BasicBlock *, 2> ExitingBlocks;
  R.getEndingBlocks(ExitingBlocks);

  if (ExitingBlocks.empty()) {
    HIPSYCL_DEBUG_ERROR << "[SubCFG] Invalid kernel! No kernel exits!\n";
    llvm_unreachable("[SubCFG] Invalid kernel! No kernel exits!\n");
  }

  auto Barriers = getBarrierIds(Entry, ExitingBlocks, Blocks, SAA, HI);

  llvm::IRBuilder Builder{F.getEntryBlock().getFirstNonPHI()};
  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  auto *LastBarrierIdStorage =
      Builder.CreateAlloca(DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");

  // create subcfgs
  std::vector<SubCFG> SubCFGs;
  for (auto &[Barrier, Id] : Barriers) {
    HIPSYCL_DEBUG_INFO << "Create SubCFG from " << Barrier->getName() << "(" << Barrier
                       << ") id: " << Id << "\n";
    if (Id != ExitBarrierId)
      SubCFGs.emplace_back(Barrier, LastBarrierIdStorage, Barriers, SAA, Dim, HI);
  }

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> BaseInstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> InstContReplicaMap;
  llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> LoadToAlloca{};

  for (auto &Cfg : SubCFGs)
    Cfg.arrayifyMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, SubCFGs,
                                  F.getEntryBlock().getTerminator(), ReqdArrayElements, VecInfo, F);

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> RemappedInstAllocaMap;
  // TODO remapped Instr alloca is used for fixSingleSubCfgValue
  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    Cfg.replicate(F, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap,
                  *ExitingBlocks.begin(), LocalSize, IsSscp, LoadToAlloca);
    purgeLifetime(Cfg);
  }

  llvm::BasicBlock *WhileHeader =
      generateWhileSwitchAround(&F.getEntryBlock(), F.getEntryBlock().getSingleSuccessor(),
                                *ExitingBlocks.begin(), LastBarrierIdStorage, SubCFGs);

  llvm::removeUnreachableBlocks(F);

  DT.recalculate(F);
  arrayifyAllocas(&F.getEntryBlock(), DT, SubCFGs, ReqdArrayElements, VecInfo, F, HI);

  for (auto &Cfg : SubCFGs) {
    Cfg.fixSingleSubCfgValues(DT, RemappedInstAllocaMap, ReqdArrayElements, VecInfo, LoadToAlloca);
  }

  if (HI.Level == HierarchicalLevel::H_CBS_GROUP) {
    for (auto &Cfg : SubCFGs) {
      formSubgroupCfgs(Cfg, F, SAA, IsSscp, Dim, LocalSize, HI);
    }
  }

  /* TODO
   * 1. Incomplete sub-groups (masked load and stores?) or conditional uniform branch
   */
  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    ReduceIntrinsic{}.vectorizeAllInstances(F, SubCFGs, &InstContReplicaMap);
    Shift<true>{}.vectorizeAllInstances(F, SubCFGs);
    Shift<false>{}.vectorizeAllInstances(F, SubCFGs);
    ShuffleIntrinsic{}.vectorizeAllInstances(F, SubCFGs);
    ExtractIntrinsic{}.vectorizeAllInstances(F, SubCFGs);

    assert(!llvm::verifyFunction(F, &llvm::errs()));
  }

  // simplify while loop to get single latch that isn't marked as wi-loop to prevent
  // misunderstandings.
  auto *WhileLoop = utils::updateDtAndLi(LI, DT, WhileHeader, F);
  llvm::simplifyLoop(WhileLoop, &DT, &LI, nullptr, nullptr, nullptr, false);
}

void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                 llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, bool IsSscp) {
  const std::size_t Dim = getRangeDim(F);
  HIPSYCL_DEBUG_INFO << "[SubCFG] Kernel is " << Dim << "-dimensional\n";

  const auto LocalSize = getLocalSizeValues(F, Dim, IsSscp);

  auto *Entry = &F.getEntryBlock();

  llvm::IRBuilder Builder{Entry->getTerminator()};
  llvm::Value *ReqdArrayElements = LocalSize[0];
  for (size_t D = 1; D < LocalSize.size(); ++D)
    ReqdArrayElements = Builder.CreateMul(ReqdArrayElements, LocalSize[D]);

  // get a common (pseudo) index value to be replaced by the actual index later
  Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
  auto *IndVarT = getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
  llvm::Instruction *IndVar =
      Builder.CreateLoad(IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)));

  formSubCfgGeneric(
      F, LI, DT, PDT, SAA, IsSscp, Dim, LocalSize, ReqdArrayElements,
      {utils::hasSubBarriers(F, SAA) ? HierarchicalLevel::H_CBS_GROUP : HierarchicalLevel::CBS,
       {},
       {},
       IndVar,
       nullptr});

  for (auto i = Dim; i < 3; ++i) {
    auto *Load = mergeGVLoadsInEntry(F, LocalIdGlobalNames[i]);
    Load->replaceAllUsesWith(Builder.getIntN(64, 0));
    Load->eraseFromParent();
  }

  {
    Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
    auto* Alloca = Builder.CreateAlloca(
        llvm::IntegerType::getInt8Ty(F.getContext()),
        Builder.getIntN(64, 1024 * 1024));
    auto *Load = mergeGVLoadsInEntry(F, WorkGroupSharedMemory);
    Load->replaceAllUsesWith(Alloca);
    Load->eraseFromParent();
  }
  {
    Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
    auto* Alloca = Builder.CreateAlloca(
        llvm::IntegerType::getInt8Ty(F.getContext()),
        Builder.getIntN(64, 1024 * 1024 / 32));
    auto *Load = mergeGVLoadsInEntry(F, SubGroupSharedMemory);
    Load->replaceAllUsesWith(Alloca);
    Load->eraseFromParent();
  }
  // F.viewCFG();

  // The dummy induction variable can now be removed. It should not have any users.
  {
    for (auto U : IndVar->users()) {
      HIPSYCL_DEBUG_ERROR << "IndVar still in use: " << *U << "\n";
    }
    IndVar->eraseFromParent();
  }
  assert(!llvm::verifyFunction(F, &llvm::errs()) && "Function verification failed");
}

void createLoopsAroundKernel(llvm::Function &F, llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                             llvm::PostDominatorTree &PDT, bool IsSscp) {

  auto *Body = llvm::SplitBlock(&F.getEntryBlock(), &*F.getEntryBlock().getFirstInsertionPt(), &DT,
                                &LI, nullptr, "wibody", true);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG());

  Body = Body->getSingleSuccessor();

  llvm::SmallVector<llvm::BasicBlock *, 4> ExitBBs;
  llvm::BasicBlock *ExitBB = llvm::BasicBlock::Create(F.getContext(), "exit", &F);
  llvm::IRBuilder<> Bld{ExitBB};
  Bld.CreateRetVoid();
  for (auto &BB : F) {
    if (BB.getTerminator()->getNumSuccessors() == 0 &&
        !llvm::isa<llvm::UnreachableInst>(BB.getTerminator()) && &BB != ExitBB) {
      auto *oldTerm = BB.getTerminator();
      Bld.SetInsertPoint(oldTerm);
      Bld.CreateBr(ExitBB);
      oldTerm->eraseFromParent();
    }
  }

  llvm::SmallVector<llvm::BasicBlock *, 8> Blocks{};
  Blocks.reserve(std::distance(F.begin(), F.end()));
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });

  auto *sgIdLoad = mergeGVLoadsInEntry(F, SgIdGlobalName);
  moveAllocasToEntry(F, Blocks);

  const auto Dim = getRangeDim(F);

  // insert dummy induction variable that can be easily identified and replaced later
  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  auto *IndVarT = getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
  llvm::Value *Idx =
      Builder.CreateLoad(IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)));

  auto LocalSize = getLocalSizeValues(F, Dim, IsSscp);

  llvm::ValueToValueMapTy VMap;
  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  auto *LastHeader = Body;

  createLoopsAround(F, ExitBB, LocalSize, 0, VMap, Latches, LastHeader, Idx, IsSscp,
                    {HierarchicalLevel::CBS, {}, {}, Idx, nullptr});

  F.getEntryBlock().getTerminator()->setSuccessor(0, LastHeader);
  llvm::remapInstructionsInBlocks(Blocks, VMap);

  sgIdLoad->eraseFromParent();

  // remove uses of the undefined global id variables
  for (int D = 0; D < Dim; ++D)
    if (auto *Load =
            llvm::cast_or_null<llvm::LoadInst>(getLoadForGlobalVariable(F, LocalIdGlobalNames[D])))
      Load->eraseFromParent();
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG())
}
} // namespace

namespace hipsycl::compiler {
void SubCfgFormationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequiredTransitive<llvm::DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<llvm::PostDominatorTreeWrapperPass>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool SubCfgFormationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F) || getRangeDim(F) == 0)
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  if (utils::hasBarriers(F, SAA) || utils::hasSubBarriers(F, SAA))
    formSubCfgs(F, LI, DT, PDT, SAA, false);
  else
    createLoopsAroundKernel(F, DT, LI, PDT, false);

  return true;
}

char SubCfgFormationPassLegacy::ID = 0;

llvm::PreservedAnalyses SubCfgFormationPass::run(llvm::Function &F,
                                                 llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());

  if (!SAA || !SAA->isKernelFunc(&F) || getRangeDim(F) == 0)
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  if (utils::hasBarriers(F, *SAA) || utils::hasSubBarriers(F, *SAA)) {
    formSubCfgs(F, LI, DT, PDT, *SAA, IsSscp_);
    assert(!llvm::verifyFunction(F, &llvm::outs()));
  } else {
    createLoopsAroundKernel(F, DT, LI, PDT, IsSscp_);
  }

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler