# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Symmetric native forward and backward bound-management tests."""

from copy import deepcopy
from typing import Literal

import pytest
import torch
from torch.testing import assert_close

from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters import (
    AnalogMVType,
    BoundManagementType,
    IOParameters,
    NoiseManagementType,
)
from aihwkit.simulator.tiles import AnalogTile

from .helpers.testcases import SKIP_CUDA_TESTS

Direction = Literal["forward", "backward"]
DIRECTIONS = ["forward", "backward"]
BACKENDS = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable")),
]
BM = BoundManagementType
NM = NoiseManagementType


def _io(bm: BoundManagementType = BM.ITERATIVE, **kwargs) -> IOParameters:
    return IOParameters(
        bound_management=bm,
        noise_management=NM.NONE,
        inp_res=0.0,
        out_res=0.0,
        out_noise=0.0,
        inp_bound=1.0,
        out_bound=1.0,
        **kwargs,
    )


def _tile(
    weights: torch.Tensor,
    backend: str,
    forward: IOParameters | None = None,
    backward: IOParameters | None = None,
    trans: bool = False,
    bias: bool = False,
) -> AnalogTile:
    config = SingleRPUConfig(
        device=ConstantStepDevice(w_min=-1.0, w_max=1.0, w_min_dtod=0.0, w_max_dtod=0.0),
        forward=deepcopy(forward if forward is not None else _io(BM.NONE)),
        backward=deepcopy(backward if backward is not None else _io(BM.NONE)),
    )
    config.mapping.digital_bias = False
    tile = AnalogTile(
        weights.shape[0], weights.shape[1], config, bias=bias, in_trans=trans, out_trans=trans
    )
    tile.set_weights(weights, torch.zeros(weights.shape[0]) if bias else None)
    return tile.cuda() if backend == "cuda" else tile


def _managed_tile(
    matrix: torch.Tensor,
    io: IOParameters,
    direction: Direction,
    backend: str,
    trans: bool = False,
    bias: bool = False,
) -> AnalogTile:
    """Create a tile whose selected direction computes ``input @ matrix``."""
    weights = matrix.T.contiguous() if direction == "forward" else matrix.contiguous()
    if direction == "forward":
        return _tile(weights, backend, forward=io, trans=trans, bias=bias)
    return _tile(weights, backend, backward=io, trans=trans, bias=bias)


def _run(
    tile: AnalogTile, inputs: torch.Tensor, direction: Direction, trans: bool = False
) -> torch.Tensor:
    values = inputs.T.contiguous() if trans else inputs
    values = values.to(tile.device)
    if direction == "forward":
        result = tile.joint_forward(values)
    else:
        result = tile.backward(values)
    result = result.cpu()
    return result.T.contiguous() if trans else result


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("nm", [NM.NONE, NM.ABS_MAX])
def test_none_matches_io_reference(
    direction: Direction, backend: str, batch: int, trans: bool, nm: NM
) -> None:
    """NONE retains the DAC -> MVM -> ADC -> output-scaling result."""
    matrix = torch.tensor([[0.75, -0.5], [0.25, 0.75], [-0.5, 0.25]])
    inputs = torch.tensor([[0.75, -0.5, 0.25]]).repeat(batch, 1)
    io = _io(BM.NONE)
    io.noise_management = nm
    io.inp_res = 1 / 16
    io.out_res = 1 / 16
    io.out_bound = 0.25
    io.out_scale = 1.5

    scale = (
        inputs.abs().amax(dim=1, keepdim=True)
        if nm == NM.ABS_MAX
        else torch.ones((batch, 1))
    )
    dac_step = 2 * io.inp_bound * io.inp_res
    dac = torch.round(inputs / scale / dac_step)
    dac = (dac * dac_step).clamp(-io.inp_bound, io.inp_bound)
    adc_step = 2 * io.out_bound * io.out_res
    expected = (torch.round((dac @ matrix) / adc_step) * adc_step).clamp(
        -io.out_bound, io.out_bound
    ) * (scale * io.out_scale)

    actual = _run(_managed_tile(matrix, io, direction, backend, trans), inputs, direction, trans)
    assert_close(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("nm", [NM.NONE, NM.ABS_MAX])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_iterative_recovers_clipped_rectangular_mvm(
    direction: Direction, backend: str, batch: int, trans: bool, nm: NM, sign: float
) -> None:
    """Several retries recover ``input @ matrix`` for positive and negative saturation."""
    matrix = torch.tensor([[0.75, 0.5, -0.25]]).repeat(8, 1)
    inputs = sign * torch.ones(batch, 8)
    if nm == NM.ABS_MAX:
        inputs *= torch.arange(1, batch + 1).unsqueeze(1) * 2.0
    elif batch > 1:
        inputs[1] *= 0.03125  # A sample that never clips in a retrying batch.
        inputs[2] = 0.0
    io = _io()
    io.noise_management = nm
    io.out_scale = 1.5
    managed = _managed_tile(matrix, io, direction, backend, trans)
    io.bound_management = BM.NONE
    unmanaged = _managed_tile(matrix, io, direction, backend, trans)
    expected = (inputs @ matrix) * io.out_scale

    assert (_run(unmanaged, inputs, direction, trans) - expected).abs().max() > 1.0
    assert_close(_run(managed, inputs, direction, trans), expected, atol=1e-6, rtol=0)
    # Repeated calls restart BM scales, including after the all-zero shortcut.
    assert_close(_run(managed, inputs * 0, direction, trans), expected * 0, atol=0, rtol=0)
    assert_close(_run(managed, inputs, direction, trans), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("nm", [NM.NONE, NM.ABS_MAX])
def test_unsaturated_does_not_retry(
    direction: Direction, backend: str, batch: int, nm: NM
) -> None:
    """DAC quantization makes an unnecessary halving observable in the output."""
    matrix = torch.full((2, 3), 0.125)
    inputs = torch.tensor([[0.375, 1.0]]).repeat(batch, 1)
    io = _io()
    io.noise_management = nm
    io.inp_res = 1 / 16  # Absolute DAC step 1/8; 3/8 cannot survive a halving.
    managed = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    io.bound_management = BM.NONE
    unmanaged = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(managed, unmanaged, atol=0, rtol=0)
    assert_close(managed, inputs @ matrix, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "forward_bm,backward_bm", [(BM.NONE, BM.ITERATIVE), (BM.ITERATIVE, BM.NONE)]
)
def test_forward_backward_configuration_independent(
    backend: str, forward_bm: BM, backward_bm: BM
) -> None:
    """Forward and backward use their own bounds, scales, NM and BM settings."""
    weights = torch.full((8, 3), 0.75)
    forward = _io(forward_bm)
    forward.out_bound = 0.5
    forward.out_scale = 0.5
    backward = _io(backward_bm)
    backward.noise_management = NM.ABS_MAX
    backward.out_scale = 1.5
    tile = _tile(weights, backend, forward=forward, backward=backward)
    for _ in range(2):
        actual_f = tile.joint_forward(torch.ones(2, 3, device=tile.device)).cpu()
        actual_b = _run(tile, torch.full((2, 8), 2.0), "backward")
        expected_f = 1.125 if forward_bm == BM.ITERATIVE else 0.25
        expected_b = 18.0 if backward_bm == BM.ITERATIVE else 3.0
        assert_close(actual_f, torch.full((2, 8), expected_f), atol=1e-6, rtol=0)
        assert_close(actual_b, torch.full((2, 3), expected_b), atol=1e-6, rtol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("nm", [NM.NONE, NM.ABS_MAX])
@pytest.mark.parametrize("limit", ["factor", "resolution"])
def test_retry_limit_returns_last_clipped_result(
    direction: Direction, backend: str, batch: int, nm: NM, limit: str
) -> None:
    """Stop after the first pass whose factor exceeds the configured limit."""
    matrix = torch.full((16, 3), 0.75)
    inputs = torch.ones(batch, 16)
    io = _io()
    io.noise_management = nm
    if limit == "factor":
        io.max_bm_factor = 2
    else:
        io.inp_res = 1 / 32  # Absolute DAC step = 1/16.
        io.max_bm_res = 1 / 8  # Limit = 2; final pass uses factor 4.
    actual = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(actual, torch.full((batch, 3), 4.0), atol=1e-6, rtol=0)
    assert (actual - inputs @ matrix).abs().min() > 1.0


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("test_negative_bound,expected", [(True, -2.25), (False, -1.0)])
def test_negative_bound_flag_matches_single_and_batch(
    direction: Direction, backend: str, test_negative_bound: bool, expected: float
) -> None:
    """Negative-bound handling has the same meaning for vector and batch kernels."""
    matrix = torch.full((3, 8), -0.75)
    io = _io()
    io.bm_test_negative_bound = test_negative_bound
    tile = _managed_tile(matrix, io, direction, backend)
    vector = torch.ones(1, 3)
    single = _run(tile, vector, direction)
    batch = _run(tile, vector.repeat(4, 1), direction)

    # The unbounded result is y = 3 * 1 * (-0.75) = -2.25. Testing the
    # negative bound recovers it; otherwise the first result remains clipped.
    assert_close(batch, torch.full((4, 8), expected), atol=1e-6, rtol=0)
    assert_close(single, batch[:1], atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "direction,backend",
    [
        pytest.param("forward", "cpu"),
        pytest.param("backward", "cpu"),
        pytest.param(
            "forward",
            "cuda",
            marks=pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable"),
        ),
        pytest.param(
            "backward",
            "cuda",
            marks=pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("batch", [1, 4])
def test_ignored_negative_bound_does_not_hide_positive_clipping(
    direction: Direction, backend: str, batch: int
) -> None:
    """An ignored negative saturation cannot clear an earlier positive failure."""
    matrix = torch.tensor([[0.75, -0.75]]).repeat(8, 1)
    io = _io()
    io.bm_test_negative_bound = False
    inputs = torch.ones(batch, 8)
    actual = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(actual, inputs @ matrix, atol=1e-6, rtol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
def test_nonunit_input_bound_and_analog_bias(
    direction: Direction, backend: str, batch: int
) -> None:
    """Non-unit DAC range and the extra analog-bias column have correct sizes."""
    matrix = torch.full((8, 3), 0.75)
    io = _io()
    io.inp_bound = 2.0
    tile = _managed_tile(matrix, io, direction, backend, bias=True)
    inputs = torch.full((batch, 8), 1.5)
    assert_close(_run(tile, inputs, direction), inputs @ matrix, atol=1e-6, rtol=0)


@pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable")
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("nm", [NM.NONE, NM.ABS_MAX])
def test_cpu_cuda_agree(direction: Direction, batch: int, nm: NM) -> None:
    """The native backends agree after retries and at the factor limit."""
    matrix = torch.tensor([[0.75, -0.5, 0.25]]).repeat(8, 1)
    inputs = torch.ones(batch, 8)
    for limit in (2, 1000):
        io = _io()
        io.noise_management = nm
        io.max_bm_factor = limit
        cpu = _run(_managed_tile(matrix, io, direction, "cpu"), inputs, direction)
        gpu = _run(_managed_tile(matrix, io, direction, "cuda"), inputs, direction)
        assert_close(gpu, cpu, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "direction,backend",
    [
        pytest.param("forward", "cpu"),
        pytest.param("backward", "cpu"),
        pytest.param(
            "forward",
            "cuda",
            marks=pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable"),
        ),
        pytest.param(
            "backward",
            "cuda",
            marks=pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize(
    "mv_type", [AnalogMVType.POS_NEG_SEPARATE, AnalogMVType.POS_NEG_SEPARATE_DIGITAL_SUM]
)
def test_split_mvm_retries(
    direction: Direction, backend: str, mv_type: AnalogMVType
) -> None:
    """Retries rebuild the input after split MVM passes use scratch buffers."""
    io = _io()
    io.mv_type = mv_type
    matrix = torch.full((8, 3), 0.75)
    inputs = torch.ones(2, 8)
    inputs[:, -1] = -1.0
    actual = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(actual, inputs @ matrix, atol=1e-6, rtol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
def test_iterative_worst_case_uses_npsum_on_second_pass(
    direction: Direction, backend: str, batch: int
) -> None:
    """The second pass uses the conservative NPSum scale in both directions."""
    matrix = torch.full((8, 3), 0.5)
    inputs = torch.ones(batch, 8)
    io = _io(BM.ITERATIVE_WORST_CASE)
    io.noise_management = NM.ABS_MAX
    io.max_bm_factor = 1

    # First pass: alpha_absmax = 1, so y_raw = 8 * 0.5 = 4 > out_bound.
    # Second pass: alpha_npsum = max(1, 8 * 0.6 / 1) = 4.8, hence
    # y_raw = 4 / 4.8 < 1. Restoring alpha_npsum returns the exact y = 4.
    actual = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(actual, inputs @ matrix, atol=1e-6, rtol=0)

    # Ordinary iterative BM halves the input on its second pass. Its reduction
    # reaches 2 > max_bm_factor, so it stops with 2 * clip(4 / 2) = 2.
    io.bound_management = BM.ITERATIVE
    iterative = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(iterative, torch.full((batch, 3), 2.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch", [1, 4])
def test_iterative_worst_case_uses_npsum_without_initial_noise_management(
    direction: Direction, backend: str, batch: int
) -> None:
    """A disabled first-pass NM must not suppress the worst-case retry scale."""
    matrix = torch.full((8, 3), 0.5)
    inputs = torch.ones(batch, 8)
    io = _io(BM.ITERATIVE_WORST_CASE)
    io.noise_management = NM.NONE
    io.max_bm_factor = 1

    # The first pass clips 4 to 1. The second pass must use NPSum = 4.8,
    # which avoids clipping and restores the original result of 4. Without
    # that scale, the retry eventually stops at the factor limit with 2.
    actual = _run(_managed_tile(matrix, io, direction, backend), inputs, direction)
    assert_close(actual, inputs @ matrix, atol=1e-6, rtol=0)


@pytest.mark.skipif(SKIP_CUDA_TESTS, reason="CUDA unavailable")
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_cuda_batch_uses_incremental_bm_factor_without_nm(direction: Direction) -> None:
    """Identical vectors do not depend on selecting the single or batch CUDA kernel."""
    matrix = torch.full((8, 1), 0.75)
    io = _io()
    io.inp_res = 1 / 16
    tile = _managed_tile(matrix, io, direction, "cuda")
    vector = torch.full((1, 8), 0.625)

    single = _run(tile, vector, direction)
    batch = _run(tile, vector.repeat(4, 1), direction)

    # DAC step delta = 2 * inp_bound * inp_res = 1 / 8. For BM scale s,
    # y_raw(s) = 8 * 0.75 * Q_delta(0.625 / s). The first two attempts saturate:
    #   s=1: y_raw = 8 * 0.75 * 0.625 = 3.75 > 1,
    #   s=2: y_raw = 8 * 0.75 * 0.375 = 2.25 > 1.
    # The third attempt succeeds and restores its scale in the final output:
    #   s=4: y_raw = 8 * 0.75 * 0.125 = 0.75; y = s * y_raw = 3.0.
    assert_close(single, torch.full((1, 1), 3.0), atol=0, rtol=0)
    assert_close(batch, single.repeat(4, 1), atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("indexed", [False, True])
def test_convolution_backward_publishes_only_final_retry(backend: str, indexed: bool) -> None:
    """Overlapping patches do not accumulate discarded backward BM attempts."""
    config = SingleRPUConfig(
        device=ConstantStepDevice(w_min=-1.0, w_max=1.0, w_min_dtod=0.0, w_max_dtod=0.0),
        forward=_io(BM.NONE),
        backward=_io(),
    )
    layer = AnalogConv2d(1, 8, 2, bias=False, rpu_config=config, use_indexed=indexed)
    weights = torch.full((8, 1, 2, 2), 0.75)
    layer.set_weights(weights)
    if backend == "cuda":
        layer = layer.cuda()
    inputs = torch.zeros(2, 1, 3, 3, device=backend, requires_grad=True)
    layer(inputs).sum().backward()
    expected = torch.tensor([[6.0, 12.0, 6.0], [12.0, 24.0, 12.0], [6.0, 12.0, 6.0]])
    assert_close(inputs.grad.cpu(), expected.expand(2, 1, 3, 3), atol=1e-6, rtol=0)
