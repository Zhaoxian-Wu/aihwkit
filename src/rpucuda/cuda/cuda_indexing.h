/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace RPU {
namespace cuda {

using index_t = int64_t;

struct IndexChunk {
  size_t offset;
  size_t size;
};

inline size_t checkedSizeProduct(size_t first, size_t second, const char *description) {
  if (first > 0 && second > std::numeric_limits<size_t>::max() / first) {
    std::ostringstream message;
    message << description << " size overflows size_t: " << first << " * " << second;
    throw std::overflow_error(message.str());
  }
  return first * second;
}

inline int64_t checkedInt64Size(size_t size, const char *description) {
  if (size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    std::ostringstream message;
    message << description << " exceeds the supported int64_t element count: " << size;
    throw std::overflow_error(message.str());
  }
  return static_cast<int64_t>(size);
}

inline int checkedIntSize(size_t size, const char *description) {
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    std::ostringstream message;
    message << description << " exceeds the supported int element count: " << size;
    throw std::overflow_error(message.str());
  }
  return static_cast<int>(size);
}

inline size_t checkedPositiveSize(index_t value, const char *description) {
  if (value <= 0) {
    std::ostringstream message;
    message << description << " must be positive, got " << value;
    throw std::runtime_error(message.str());
  }
  return static_cast<size_t>(value);
}

inline size_t checkedPositiveProduct(index_t first, index_t second, const char *description) {
  if (first <= 0 || second <= 0) {
    std::ostringstream message;
    message << description << " dimensions must be positive, got " << first << " and " << second;
    throw std::runtime_error(message.str());
  }

  const auto first_unsigned = checkedPositiveSize(first, description);
  const auto second_unsigned = checkedPositiveSize(second, description);
  return checkedSizeProduct(first_unsigned, second_unsigned, description);
}

inline int checkedNBlocks(size_t size, int nthreads) {
  if (size == 0) {
    throw std::runtime_error("CUDA launch size must be positive");
  }
  if (nthreads <= 0) {
    throw std::runtime_error("CUDA threads per block must be positive");
  }

  const size_t blocks = 1 + (size - 1) / static_cast<size_t>(nthreads);
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA grid size exceeds the supported 32-bit block count");
  }
  return static_cast<int>(blocks);
}

inline std::vector<IndexChunk> planIndexChunks(
    size_t total_size, size_t max_chunk_size = static_cast<size_t>(std::numeric_limits<int>::max())) {
  if (total_size == 0) {
    throw std::runtime_error("Chunked CUDA operation size must be positive");
  }
  if (max_chunk_size == 0) {
    throw std::runtime_error("CUDA chunk size must be positive");
  }

  std::vector<IndexChunk> chunks;
  for (size_t offset = 0; offset < total_size;) {
    const size_t remaining = total_size - offset;
    const size_t chunk_size = remaining < max_chunk_size ? remaining : max_chunk_size;
    chunks.push_back({offset, chunk_size});
    offset += chunk_size;
  }
  return chunks;
}

} // namespace cuda
} // namespace RPU
