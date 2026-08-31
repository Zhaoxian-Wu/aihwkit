# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Regression test for non-perfect CUDA MVMs that cross the int32 element boundary."""

from unittest import SkipTest, TestCase

import torch

from .helpers.cuda_large_mbatch_repro import make_tile, run_case
from .helpers.testcases import SKIP_CUDA_TESTS


OUT_SIZE = 50257
ROWS = 43008
IN_SIZE = 1
CHUNK_ROWS = 1024
GIB = 1024**3


def _required_free_memory_bytes() -> int:
    """Return a conservative free-memory requirement for the regression case."""

    element_bytes = torch.tensor([], dtype=torch.float32).element_size()
    output_bytes = ROWS * OUT_SIZE * element_bytes
    chunk_bytes = CHUNK_ROWS * OUT_SIZE * element_bytes
    stats_chunk_bytes = min(output_bytes, 64 * 1024 * 1024 * element_bytes)

    # The returned output and the IO manager's reusable output buffer can each
    # hold the full result. Also reserve chunk/statistics temporaries and allocator headroom.
    return 2 * output_bytes + 4 * chunk_bytes + 2 * stats_chunk_bytes + 3 * GIB


class CudaLargeMBatchTest(TestCase):
    """Exercise the first output size known to overflow the old int32 indexing path."""

    def test_nonideal_forward_crosses_int32_boundary(self) -> None:
        """Compare one large non-perfect MVM with bounded-size reference chunks."""

        if SKIP_CUDA_TESTS or not torch.cuda.is_available():
            raise SkipTest("CUDA support is not available")

        required_bytes = _required_free_memory_bytes()
        candidates = []
        for device_index in range(torch.cuda.device_count()):
            free_bytes, _ = torch.cuda.mem_get_info(device_index)
            candidates.append((free_bytes, device_index))

        free_bytes, device_index = max(candidates)
        if free_bytes < required_bytes:
            raise SkipTest(
                "large-MVM regression requires "
                f"{required_bytes / GIB:.1f} GiB free GPU memory; "
                f"best device has {free_bytes / GIB:.1f} GiB"
            )

        with torch.cuda.device(device_index):
            tile = make_tile(OUT_SIZE, IN_SIZE, perfect=False, random_noise=False)
            result = run_case(
                tile,
                rows=ROWS,
                in_size=IN_SIZE,
                out_size=OUT_SIZE,
                chunk_rows=CHUNK_ROWS,
                transposed=False,
            )

        self.assertGreater(result["elements"], torch.iinfo(torch.int32).max)
        self.assertEqual(result["direct"]["finite_fraction"], 1.0)
        self.assertEqual(result["reference"]["finite_fraction"], 1.0)
        self.assertLessEqual(result["max_abs_error"], 1e-6)
