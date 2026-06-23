# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Tests for digital gradient accumulation in analog contexts."""

from torch import Tensor, manual_seed, randn, zeros
from torch.nn import Conv2d

from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice

from .helpers.testcases import AihwkitTestCase


def get_analog_context(layer: AnalogLinear | AnalogConv2d) -> AnalogContext:
    """Return the analog context parameter of an analog layer."""
    for param in layer.parameters():
        if isinstance(param, AnalogContext):
            return param
    raise RuntimeError("No AnalogContext found")


class DigitalGradAccumulationTest(AihwkitTestCase):
    """Tests for opt-in digital gradient accumulation."""

    @staticmethod
    def get_rpu_config() -> SingleRPUConfig:
        """Return a deterministic analog training config."""
        rpu_config = SingleRPUConfig(
            device=ConstantStepDevice(w_max_dtod=0.0, w_min_dtod=0.0, up_down_dtod=0.0)
        )
        rpu_config.forward.is_perfect = True
        rpu_config.backward.is_perfect = True
        return rpu_config

    def test_backward_accumulates_digital_grad_without_retaining_traces(self) -> None:
        """Check that backward accumulates grad and clears x/d traces when enabled."""
        manual_seed(123)
        layer = AnalogLinear(3, 2, bias=False, rpu_config=self.get_rpu_config())
        analog_ctx = get_analog_context(layer)
        optimizer = AnalogSGD(layer.parameters(), lr=0.1)
        optimizer.regroup_param_groups(layer)
        optimizer.set_digital_grad_accumulation(True)
        optimizer.zero_grad(set_to_none=True)

        expected_grad = zeros(2, 3)
        inputs = [
            Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            Tensor([[0.7, 0.8, 0.9]]),
        ]
        grad_outputs = [
            Tensor([[0.2, -0.4], [0.6, 0.8]]),
            Tensor([[-0.5, 0.3]]),
        ]

        for x_input, d_input in zip(inputs, grad_outputs):
            layer(x_input).backward(d_input)
            expected_grad += d_input.reshape(-1, 2).transpose(0, 1).matmul(
                x_input.reshape(-1, 3)
            )

            self.assertEqual(analog_ctx.analog_input, [])
            self.assertEqual(analog_ctx.analog_grad_output, [])
            self.assertIsNotNone(analog_ctx.grad)

        self.assertEqual(analog_ctx.data.shape, expected_grad.shape)
        self.assertTensorAlmostEqual(analog_ctx.grad, expected_grad)

        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(analog_ctx.grad)
        self.assertEqual(analog_ctx.analog_input, [])
        self.assertEqual(analog_ctx.analog_grad_output, [])

    def test_conv2d_backward_accumulates_digital_grad(self) -> None:
        """Check that non-indexed Conv2d accumulates its unfolded weight gradient."""
        manual_seed(123)
        layer = AnalogConv2d(
            2,
            3,
            kernel_size=2,
            padding=1,
            bias=False,
            rpu_config=self.get_rpu_config(),
            use_indexed=False,
        )
        reference_layer = Conv2d(2, 3, kernel_size=2, padding=1, bias=False)
        analog_ctx = get_analog_context(layer)
        optimizer = AnalogSGD(layer.parameters(), lr=0.1)
        optimizer.regroup_param_groups(layer)
        optimizer.set_digital_grad_accumulation(True)
        optimizer.zero_grad(set_to_none=True)

        for batch_size in (2, 1):
            x_input = randn(batch_size, 2, 3, 4)
            d_input = randn(batch_size, 3, 4, 5)

            layer(x_input).backward(d_input)
            reference_layer(x_input).backward(d_input)

            self.assertEqual(analog_ctx.analog_input, [])
            self.assertEqual(analog_ctx.analog_grad_output, [])
