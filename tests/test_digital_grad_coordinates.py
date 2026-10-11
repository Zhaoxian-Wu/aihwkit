# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Independent digital oracles for native gradient coordinates and chain rules."""

import pytest
import torch

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")),
]


def make_layer(vector: bool, learned: bool, device: str, analog_bias: bool = False):
    """Build a deterministic tile with deliberately non-unit row scales."""
    config = FloatingPointRPUConfig()
    config.mapping.weight_scaling_omega = 0.0
    config.mapping.weight_scaling_lr_compensation = False
    config.mapping.learn_out_scaling = learned
    config.mapping.out_scaling_columnwise = vector
    config.mapping.digital_bias = not analog_bias
    layer = AnalogLinear(4, 3, bias=analog_bias, rpu_config=config).to(device)
    tile = layer.analog_module
    tile.set_mapping_scales(torch.tensor([0.2, 0.7, 1.4] if vector else [0.3], device=device))
    tile.mapping_lr_scale = 0.4
    if learned:
        with torch.no_grad():
            tile.out_scaling_alpha.copy_(
                torch.tensor([0.8, -1.3, 0.6] if vector else [1.2], device=device)
            )
    return layer


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "vector,learned", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("mode", ["logical_weight", "pre_update"])
def test_native_coordinates_and_unchanged_chain_derivatives(device, vector, learned, mode):
    """Check accumulated weight, input and alpha gradients against digital formulas."""
    torch.manual_seed(1729)
    layer = make_layer(vector, learned, device)
    tile, context = layer.analog_module, layer.analog_module.analog_ctx
    context.enable_buffer()
    context.set_digital_grad_accumulation(use_pre_update=mode == "pre_update")
    mapping = tile.get_mapping_scales().reshape(-1, 1)
    alpha = tile.out_scaling_alpha
    alpha_value = 1 if alpha is None else alpha.detach().reshape(-1, 1)
    weights = tile.get_weights(apply_weight_scaling=False)[0].to(device)
    expected_grad = torch.zeros_like(weights)
    expected_alpha_grad = None if alpha is None else torch.zeros_like(alpha)
    for batch in (2, 1):
        x = torch.randn(batch, 5, 4, device=device, requires_grad=True)
        error = torch.randn(batch, 5, 3, device=device)
        output = layer(x)
        torch.testing.assert_close(output, x @ (weights * mapping * alpha_value).T)
        output.backward(error)
        torch.testing.assert_close(x.grad, error @ (weights * mapping * alpha_value))
        expected_grad += error.reshape(-1, 3).T @ x.detach().reshape(-1, 4)
        factor = alpha_value
        if mode == "pre_update":
            factor = alpha_value / (mapping * tile.mapping_lr_scale)
        torch.testing.assert_close(context.grad, expected_grad * factor)
        assert not context.analog_input and not context.analog_grad_output
        if alpha is not None:
            row_gradient = (error * (x.detach() @ (weights * mapping).T)).sum((0, 1))
            expected_alpha_grad += row_gradient if vector else row_gradient.sum().reshape(1)
            torch.testing.assert_close(alpha.grad, expected_alpha_grad)
    optimizer = AnalogSGD(layer.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    assert context.grad is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("analog_bias", [False, True])
def test_direct_accumulation_is_logical_without_preprocessing(
    device, transposed, analog_bias, monkeypatch
):
    """Test direct calls, bias columns and transposed axes without a backward read."""
    layer = make_layer(True, True, device, analog_bias)
    tile, context = layer.analog_module, layer.analog_module.analog_ctx
    tile.in_trans = tile.out_trans = transposed

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preprocessing must not run in default digital accumulation")

    monkeypatch.setattr(tile, "pre_update", forbidden)
    monkeypatch.setattr(tile, "pre_backward", forbidden)
    x, error = torch.randn(2, 5, 4, device=device), torch.randn(2, 5, 3, device=device)
    expected = error.reshape(-1, 3).T @ x.reshape(-1, 4)
    if analog_bias:
        expected = torch.cat((expected, error.sum((0, 1)).reshape(-1, 1)), dim=1)
    if transposed:
        x, error = x.movedim(-1, 0), error.movedim(-1, 0)
    context.accumulate_digital_gradient(x, error)
    context.accumulate_digital_gradient(x, error)
    torch.testing.assert_close(context.grad, 2 * expected)


@pytest.mark.parametrize("device", DEVICES)
def test_outer_chain_factors_follow_autograd_without_registration(device):
    """Outstanding forward graphs can use different external chain factors."""
    layer = make_layer(True, True, device)
    tile, context = layer.analog_module, layer.analog_module.analog_ctx
    context.set_digital_grad_accumulation()
    first_x = torch.randn(5, 4, device=device, requires_grad=True)
    first_scale = torch.tensor([0.4, 0.8, 1.6], device=device, requires_grad=True)
    first = layer(first_x) * first_scale
    second_x = torch.randn(2, 4, device=device, requires_grad=True)
    second = layer(second_x) * 2
    first_error, second_error = torch.randn_like(first), torch.randn_like(second)
    first.backward(first_error)
    second.backward(second_error)
    alpha = tile.out_scaling_alpha.detach()
    expected = (first_error * first_scale.detach() * alpha).T @ first_x.detach() + (
        second_error * 2 * alpha
    ).T @ second_x.detach()
    torch.testing.assert_close(context.grad, expected)
    weights = tile.get_weights(apply_weight_scaling=False)[0].to(device)
    native_scale = tile.get_mapping_scales() * tile.out_scaling_alpha.detach()
    torch.testing.assert_close(
        first_x.grad, (first_error * first_scale.detach()) @ (weights * native_scale[:, None])
    )
    torch.testing.assert_close(
        first_scale.grad, (first_error * (first.detach() / first_scale.detach())).sum(0)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pre_update", [False, True])
def test_zero_alpha_preserves_weight_and_scale_derivatives(device, pre_update):
    """Zero alpha has a valid W gradient; no inverse-alpha division is needed."""
    layer = make_layer(True, True, device)
    tile, context = layer.analog_module, layer.analog_module.analog_ctx
    context.set_digital_grad_accumulation(use_pre_update=pre_update)
    with torch.no_grad():
        tile.out_scaling_alpha[0] = 0
    x = torch.randn(2, 4, device=device, requires_grad=True)
    error = torch.randn(2, 3, device=device)
    mapped = tile.get_weights(apply_weight_scaling=False)[0].to(device)
    mapped *= tile.get_mapping_scales().reshape(-1, 1)
    layer(x).backward(error)
    expected = (error * tile.out_scaling_alpha.detach()).T @ x.detach()
    if pre_update:
        expected /= tile.get_mapping_scales().reshape(-1, 1) * tile.mapping_lr_scale
    torch.testing.assert_close(context.grad, expected)
    torch.testing.assert_close(
        tile.out_scaling_alpha.grad, (error * (x.detach() @ mapped.T)).sum(0)
    )
    torch.testing.assert_close(x.grad, (error * tile.out_scaling_alpha.detach()) @ mapped)


def test_optimizer_flag_forwarding_and_ordinary_trace_behavior():
    """Native optimizer forwards flags; ordinary analog traces retain alpha's chain."""
    layer = make_layer(True, True, "cpu")
    tile, context = layer.analog_module, layer.analog_module.analog_ctx
    optimizer = AnalogSGD(layer.parameters(), lr=0.01)
    optimizer.set_digital_grad_accumulation(use_pre_update=True)
    assert context.digital_grad_use_pre_update
    optimizer.set_digital_grad_accumulation(False)
    error = torch.randn(2, 3)
    layer(torch.randn(2, 4)).backward(error)
    torch.testing.assert_close(
        context.analog_grad_output[0], error * tile.out_scaling_alpha.detach()
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("lr_scale", [1.0, 0.4])
def test_fixed_scale_early_pre_update_matches_late_logical_sgd(device, vector, lr_scale):
    """Compare multiple SGD steps to an independent digital logical-weight optimizer."""
    early = make_layer(vector, False, device).analog_module
    late = make_layer(vector, False, device).analog_module
    early.mapping_lr_scale = lr_scale
    early.analog_ctx.set_digital_grad_accumulation(use_pre_update=True)
    late.analog_ctx.set_digital_grad_accumulation()
    logical_weight = torch.nn.Parameter(torch.zeros(3, 4, device=device))
    digital = torch.optim.SGD([logical_weight], lr=0.03)
    early_physical = torch.zeros_like(logical_weight)
    mapping = early.get_mapping_scales().reshape(-1, 1)
    for batches in ((1, 2), (3, 1), (2, 2)):
        early.analog_ctx.grad = late.analog_ctx.grad = None
        logical_gradient = torch.zeros_like(logical_weight)
        for batch in batches:
            x, error = torch.randn(batch, 4, device=device), torch.randn(batch, 3, device=device)
            early.analog_ctx.accumulate_digital_gradient(x, error)
            late.analog_ctx.accumulate_digital_gradient(x, error)
            logical_gradient += error.T @ x
        torch.testing.assert_close(late.analog_ctx.grad, logical_gradient)
        torch.testing.assert_close(early.analog_ctx.grad, logical_gradient / (mapping * lr_scale))
        logical_weight.grad = logical_gradient
        digital.step()
        # The write-command coordinate must use the matching backend LR.
        early_physical -= (0.03 * lr_scale) * early.analog_ctx.grad
        torch.testing.assert_close(early_physical * mapping, logical_weight.detach())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pre_update", [False, True])
def test_input_backward_preprocessing_runs_once_independently(device, pre_update, monkeypatch):
    """W-gradient accumulation must not run an additional backward preprocessing."""
    layer = make_layer(True, True, device)
    tile = layer.analog_module
    tile.analog_ctx.set_digital_grad_accumulation(use_pre_update=pre_update)
    original_backward, original_update = tile.pre_backward, tile.pre_update
    calls = {"backward": 0, "update": 0}

    def track_backward(*args, **kwargs):
        calls["backward"] += 1
        return original_backward(*args, **kwargs)

    def track_update(*args, **kwargs):
        calls["update"] += 1
        return original_update(*args, **kwargs)

    monkeypatch.setattr(tile, "pre_backward", track_backward)
    monkeypatch.setattr(tile, "pre_update", track_update)
    x = torch.randn(2, 4, device=device, requires_grad=True)
    layer(x).sum().backward()
    assert calls == {"backward": 1, "update": int(pre_update)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pre_update", [False, True])
def test_disabled_accumulation_keeps_native_sgd_traces_and_updates(device, pre_update):
    """Legacy project markers must not change the disabled native trace path."""
    layer = make_layer(True, True, device)
    tile, context = layer.analog_module, layer.analog_module.analog_ctx
    context.logical_gradient_units = True  # A legacy downstream marker is ignored.
    optimizer = AnalogSGD(layer.parameters(), lr=0.03)
    optimizer.set_digital_grad_accumulation(False, use_pre_update=pre_update)
    mapping = tile.get_mapping_scales().reshape(-1, 1)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        before = tile.get_weights(apply_weight_scaling=False)[0].to(device)
        alpha = tile.out_scaling_alpha.detach().clone()
        x = torch.randn(2, 4, device=device, requires_grad=True)
        error = torch.randn(2, 3, device=device)
        layer(x).backward(error)
        native_error = error * alpha
        torch.testing.assert_close(context.analog_input[0], x.detach())
        torch.testing.assert_close(context.analog_grad_output[0], native_error)
        assert context.grad is None
        torch.testing.assert_close(x.grad, native_error @ (before * mapping))
        optimizer.step()
        after = tile.get_weights(apply_weight_scaling=False)[0].to(device)
        torch.testing.assert_close(after, before - 0.03 * (native_error.T @ x.detach()) / mapping)
