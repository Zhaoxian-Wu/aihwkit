"""Native indexed accumulation matches convolution weight derivatives."""

from copy import deepcopy
from math import prod

import pytest
import torch

from aihwkit.nn import (
    AnalogConv1d,
    AnalogConv2d,
    AnalogConv3d,
    AnalogConv1dMapped,
    AnalogConv2dMapped,
    AnalogConv3dMapped,
)
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.parameters import ConvParameters


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("analog_bias", [False, True])
@pytest.mark.parametrize("pre_update", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_indexed_accumulator_matches_torch(dimension, analog_bias, pre_update, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = FloatingPointRPUConfig()
    config.mapping.weight_scaling_omega = 0
    config.mapping.learn_out_scaling = False
    config.mapping.digital_bias = not analog_bias
    cls = [AnalogConv1d, AnalogConv2d, AnalogConv3d][dimension - 1]
    layer = cls(
        2,
        3,
        3,
        padding=1,
        stride=2,
        bias=analog_bias,
        rpu_config=config,
        **({"use_indexed": False} if dimension == 2 else {}),
    ).to(device)
    tile = layer.analog_module
    context = tile.analog_ctx
    context.enable_buffer()
    context.set_digital_grad_accumulation(use_pre_update=pre_update)
    scale = torch.tensor([0.2, 0.7, 1.4], device=device)
    tile.set_mapping_scales(scale)
    tile.mapping_lr_scale = 0.4
    expected = torch.zeros((tile.out_size, tile.in_size + int(tile.analog_bias)), device=device)
    generator = torch.Generator().manual_seed(714)
    for size in [7, 9]:
        x = torch.randn((2, 2) + (size,) * dimension, generator=generator).to(device)
        weight = torch.zeros((3, 2) + (3,) * dimension, requires_grad=True, device=device)
        conv = [torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d][
            dimension - 1
        ]
        with torch.backends.cudnn.flags(allow_tf32=False):
            output = conv(x, weight, padding=1, stride=2)
            error = torch.randn(output.shape, generator=generator).to(device)
            layer._recalculate_indexes(x)
            context.accumulate_digital_gradient(x, error)
            reference_error = error
            if pre_update:
                reference_error = error / (
                    scale.reshape((1, 3) + (1,) * dimension) * tile.mapping_lr_scale
                )
            output.backward(reference_error)
        grad = weight.grad.flatten(1)
        if analog_bias:
            grad = torch.cat((grad, reference_error.flatten(2).sum((0, 2))[:, None]), dim=1)
        expected += grad
        torch.testing.assert_close(context.grad, expected, rtol=2e-5, atol=2e-5)
    assert not context.analog_input and not context.analog_grad_output


def test_indexed_accumulation_requires_convolution_parameters():
    """An arbitrary index matrix must not silently reuse stale convolution metadata."""
    layer = AnalogConv2d(1, 1, 1, bias=False, rpu_config=FloatingPointRPUConfig())
    tile = layer.analog_module
    x = torch.randn(2, 1, 2, 2)
    layer._recalculate_indexes(x)
    assert tile.indexed_conv_parameters is not None
    tile.set_indexed(layer.fold_indices, tile.image_sizes)
    tile.analog_ctx.set_digital_grad_accumulation()
    with pytest.raises(RuntimeError, match="require standard convolution parameters"):
        tile.analog_ctx.accumulate_digital_gradient(x, torch.ones_like(x))
    assert tile.analog_ctx.grad is None


def test_set_indexed_rejects_incompatible_convolution_parameters():
    """Reject metadata with a different kernel size than the installed tile."""
    layer = AnalogConv2d(1, 1, 1, bias=False, rpu_config=FloatingPointRPUConfig())
    with pytest.raises(ValueError, match="parameters do not match"):
        layer.analog_module.set_indexed(
            torch.tensor([2], dtype=torch.int32),
            [1, 1, 1, 1, 1],
            convolution_parameters=ConvParameters((3, 3), (1, 1), (0, 0), (1, 1)),
        )
    assert not layer.analog_module.is_indexed()


def test_convolution_metadata_is_rebuilt_after_loading():
    """Checkpoint loading reconstructs convolution metadata from the layer."""
    layer = AnalogConv2d(
        2,
        3,
        (2, 3),
        padding=(1, 2),
        stride=(2, 1),
        dilation=(2, 1),
        bias=False,
        use_indexed=True,
        rpu_config=FloatingPointRPUConfig(),
    )
    x = torch.randn(2, 2, 7, 9)
    expected = layer(x).detach()
    parameters = layer.analog_module.indexed_conv_parameters
    assert "indexed_conv_parameters" not in layer.analog_module.__getstate__()
    restored = deepcopy(layer)
    restored.analog_module.analog_ctx.set_digital_grad_accumulation()
    output = restored(x)
    torch.testing.assert_close(output, expected)
    assert restored.analog_module.indexed_conv_parameters == parameters
    error = torch.randn_like(output)
    output.backward(error)
    weight = torch.zeros(3, 2, 2, 3, requires_grad=True)
    reference = torch.nn.functional.conv2d(
        x, weight, padding=(1, 2), stride=(2, 1), dilation=(2, 1)
    )
    reference.backward(error)
    torch.testing.assert_close(restored.analog_module.analog_ctx.grad, weight.grad.flatten(1))


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("mode", ["sequential", "deferred", "deferred_reverse"])
@pytest.mark.parametrize("analog_bias", [False, True])
@pytest.mark.parametrize("pre_update", [False, True])
def test_cuda_indexed_autograd_accumulates_without_traces(
    dimension, mode, analog_bias, pre_update
):
    torch.manual_seed(317)
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native indexed reads")
    config = FloatingPointRPUConfig()
    config.mapping.weight_scaling_omega = 0
    config.mapping.learn_out_scaling = False
    config.mapping.digital_bias = not analog_bias
    cls = [AnalogConv1d, AnalogConv2d, AnalogConv3d][dimension - 1]
    layer = cls(
        2,
        3,
        3,
        padding=1,
        stride=2,
        bias=analog_bias,
        rpu_config=config,
        **({"use_indexed": True} if dimension == 2 else {}),
    ).cuda()
    tile = layer.analog_module
    context = tile.analog_ctx
    context.enable_buffer()
    context.set_digital_grad_accumulation(use_pre_update=pre_update)
    scale = torch.tensor([0.2, 0.7, 1.4], device="cuda")
    tile.set_mapping_scales(scale)
    tile.mapping_lr_scale = 0.4
    logical_weight, logical_bias = layer.get_weights()
    logical_weight = logical_weight.reshape((3, 2) + (3,) * dimension).cuda()
    if logical_bias is not None:
        logical_bias = logical_bias.cuda()
    expected = torch.zeros((tile.out_size, tile.in_size + int(tile.analog_bias)), device="cuda")
    # Different geometries accumulate sequentially; deferred jobs share geometry.
    jobs = []
    conv = [torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d][
        dimension - 1
    ]
    sizes = [7, 9] if mode == "sequential" else [7, 7]
    for size in sizes:
        stride = 1 if size == 7 else 2
        layer.stride = (stride,) * dimension
        x = torch.randn((2, 2) + (size,) * dimension, device="cuda", requires_grad=True)
        output = layer(x)
        error = torch.randn_like(output)
        reference_x = x.detach().clone().requires_grad_()
        weight = logical_weight.detach().clone().requires_grad_()
        with torch.backends.cudnn.flags(allow_tf32=False):
            reference = conv(reference_x, weight, logical_bias, padding=1, stride=stride)
            reference.backward(error)
            if pre_update:
                # Apply scaling before reduction, matching the command-unit
                # contract rather than scaling an already rounded gradient.
                command_error = error / (
                    scale.reshape((1, 3) + (1,) * dimension) * tile.mapping_lr_scale
                )
                command_weight = torch.zeros_like(weight, requires_grad=True)
                command_output = conv(x.detach(), command_weight, padding=1, stride=stride)
                command_output.backward(command_error)
        torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-5)
        grad = (command_weight.grad if pre_update else weight.grad).flatten(1)
        if analog_bias:
            bias_error = command_error if pre_update else error
            grad = torch.cat((grad, bias_error.flatten(2).sum((0, 2)).unsqueeze(1)), dim=1)
        expected += grad
        if mode == "sequential":
            with torch.backends.cudnn.flags(allow_tf32=False):
                output.backward(error)
            torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(context.grad, expected, rtol=2e-5, atol=2e-5)
        else:
            jobs.append((output, error, x, reference_x.grad))
    if mode == "deferred_reverse":
        jobs.reverse()
    for output, error, x, expected_input_grad in jobs:
        with torch.backends.cudnn.flags(allow_tf32=False):
            output.backward(error)
        torch.testing.assert_close(x.grad, expected_input_grad, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(context.grad, expected, rtol=2e-5, atol=2e-5)
    assert not context.analog_input and not context.analog_grad_output


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("pre_update", [False, True])
def test_cuda_mapped_indexed_gradients_match_torch(dimension, pre_update):
    """Use local channel counts for tiles split across input and output channels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for native indexed reads")
    torch.manual_seed(721)
    config = FloatingPointRPUConfig()
    config.mapping.weight_scaling_omega = 0
    config.mapping.learn_out_scaling = False
    kernel = [(3,), (2, 3), (2, 3, 2)][dimension - 1]
    kernel_elements = prod(kernel)
    config.mapping.max_input_size = 2 * kernel_elements
    config.mapping.max_output_size = 2
    cls = [AnalogConv1dMapped, AnalogConv2dMapped, AnalogConv3dMapped][dimension - 1]
    layer = cls(5, 5, kernel, padding=1, stride=2, bias=True, rpu_config=config).cuda()
    for tile in layer.analog_tiles():
        tile.analog_ctx.set_digital_grad_accumulation(use_pre_update=pre_update)
        tile.set_mapping_scales(torch.linspace(0.4, 0.9, tile.out_size, device="cuda"))
        tile.mapping_lr_scale = 0.3
    logical_weight, logical_bias = layer.get_weights()
    weight = logical_weight.reshape((5, 5) + kernel).cuda().requires_grad_()
    bias = logical_bias.cuda().requires_grad_()
    conv = [torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d][
        dimension - 1
    ]
    for size in [7, 9]:
        x = torch.randn((2, 5) + (size,) * dimension, device="cuda", requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        with torch.backends.cudnn.flags(allow_tf32=False):
            output = layer(x)
            reference = conv(reference_x, weight, bias, padding=1, stride=2)
            torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-5)
            error = torch.randn_like(output)
            output.backward(error)
            reference.backward(error)
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(layer.bias.grad, bias.grad, rtol=2e-5, atol=2e-5)
        in_start = 0
        for in_channels, tiles in zip(layer.in_sizes, layer.array):
            out_start = 0
            for tile in tiles:
                expected = weight.grad[
                    out_start : out_start + tile.out_size, in_start : in_start + in_channels
                ].flatten(1)
                if pre_update:
                    expected = expected / (tile.mapping_scales.unsqueeze(1) * tile.mapping_lr_scale)
                torch.testing.assert_close(tile.analog_ctx.grad, expected, rtol=2e-5, atol=2e-5)
                assert not tile.analog_ctx.analog_input and not tile.analog_ctx.analog_grad_output
                out_start += tile.out_size
            in_start += in_channels
