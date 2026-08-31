/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#include "cuda_indexing.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"

namespace {

using RPU::cuda::checkedNBlocks;
using RPU::cuda::checkedInt64Size;
using RPU::cuda::checkedIntSize;
using RPU::cuda::checkedPositiveProduct;
using RPU::cuda::checkedPositiveSize;
using RPU::cuda::checkedSizeProduct;
using RPU::cuda::planIndexChunks;

TEST(CudaIndexing, CheckedProductCrossesInt32BoundaryWithoutOverflow) {
  constexpr int64_t int_max = std::numeric_limits<int32_t>::max();

  EXPECT_EQ(checkedPositiveProduct(1, int_max, "test"), static_cast<size_t>(int_max));
  EXPECT_EQ(checkedPositiveProduct(1, int_max + 1, "test"), static_cast<size_t>(int_max) + 1);
  EXPECT_EQ(checkedPositiveProduct(43008, 50257, "test"), 2161453056ULL);
}

TEST(CudaIndexing, CheckedSizeAcceptsLargePositiveValuesAndRejectsInvalidValues) {
  constexpr int64_t int_max = std::numeric_limits<int32_t>::max();

  EXPECT_EQ(checkedPositiveSize(int_max + 1, "test"), static_cast<size_t>(int_max) + 1);
  EXPECT_THROW(checkedPositiveSize(0, "test"), std::runtime_error);
  EXPECT_THROW(checkedPositiveSize(-1, "test"), std::runtime_error);
}

TEST(CudaIndexing, CheckedProductRejectsInvalidDimensionsAndSizeTOverflow) {
  EXPECT_THROW(checkedPositiveProduct(0, 1, "test"), std::runtime_error);
  EXPECT_THROW(checkedPositiveProduct(1, 0, "test"), std::runtime_error);
  EXPECT_THROW(checkedPositiveProduct(-1, 1, "test"), std::runtime_error);
  EXPECT_THROW(checkedPositiveProduct(1, -1, "test"), std::runtime_error);

  if (std::numeric_limits<size_t>::max() == std::numeric_limits<uint64_t>::max()) {
    EXPECT_THROW(
        checkedPositiveProduct(std::numeric_limits<int64_t>::max(), 3, "test"),
        std::overflow_error);
  }
}

TEST(CudaIndexing, BufferByteSizeIsCheckedBeforeAllocation) {
  constexpr size_t size_max = std::numeric_limits<size_t>::max();

  EXPECT_EQ(checkedSizeProduct(43008, 50257, "buffer"), 2161453056ULL);
  EXPECT_EQ(checkedSizeProduct(size_max, 1, "buffer"), size_max);
  EXPECT_THROW(checkedSizeProduct(size_max, 2, "buffer"), std::overflow_error);
  EXPECT_EQ(
      checkedInt64Size(static_cast<size_t>(std::numeric_limits<int64_t>::max()), "buffer"),
      std::numeric_limits<int64_t>::max());
  if (size_max > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    EXPECT_THROW(
        checkedInt64Size(
            static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1ULL, "buffer"),
        std::overflow_error);
  }
}

TEST(CudaIndexing, LegacyIntCountIsCheckedAtTheApiBoundary) {
  EXPECT_EQ(
      checkedIntSize(std::numeric_limits<int>::max(), "legacy API"),
      std::numeric_limits<int>::max());
  EXPECT_THROW(
      checkedIntSize(static_cast<size_t>(std::numeric_limits<int>::max()) + 1, "legacy API"),
      std::overflow_error);
}

TEST(CudaIndexing, BlockCalculationUsesTheFull64BitLaunchSize) {
  constexpr size_t int_max = static_cast<size_t>(std::numeric_limits<int32_t>::max());

  EXPECT_EQ(checkedNBlocks(int_max, 512), 4194304);
  EXPECT_EQ(checkedNBlocks(int_max + 1, 512), 4194304);
  EXPECT_EQ(checkedNBlocks(int_max + 512, 512), 4194305);
  EXPECT_THROW(checkedNBlocks(0, 512), std::runtime_error);
  EXPECT_THROW(checkedNBlocks(1, 0), std::runtime_error);
  EXPECT_THROW(
      checkedNBlocks(
          static_cast<size_t>(std::numeric_limits<int>::max()) * 2ULL + 1ULL, 1),
      std::overflow_error);
}

TEST(CudaIndexing, ChunkPlanCoversBoundaryExactlyOnce) {
  constexpr size_t int_max = static_cast<size_t>(std::numeric_limits<int32_t>::max());

  for (const size_t total : {int_max, int_max + 1, int_max + 2}) {
    const auto chunks = planIndexChunks(total, int_max);
    size_t expected_offset = 0;
    size_t covered = 0;
    for (const auto &chunk : chunks) {
      EXPECT_EQ(chunk.offset, expected_offset);
      EXPECT_GT(chunk.size, 0U);
      EXPECT_LE(chunk.size, int_max);
      expected_offset += chunk.size;
      covered += chunk.size;
    }
    EXPECT_EQ(covered, total);
    EXPECT_EQ(expected_offset, total);
  }

  EXPECT_EQ(planIndexChunks(int_max, int_max).size(), 1U);
  EXPECT_EQ(planIndexChunks(int_max + 1, int_max).size(), 2U);
  EXPECT_THROW(planIndexChunks(0, int_max), std::runtime_error);
  EXPECT_THROW(planIndexChunks(1, 0), std::runtime_error);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
