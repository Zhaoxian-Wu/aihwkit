# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Autograd functions for aihwkit."""

from typing import Any, Optional, Tuple

from torch import Tensor, empty_like
from torch.autograd import Function, no_grad
from aihwkit.optim.context import AnalogContext


class AnalogFunction(Function):
    """Function for analog functions."""

    # pylint: disable=arguments-differ, protected-access, abstract-method

    @staticmethod
    @no_grad()
    def forward(
        ctx: Any,
        analog_ctx: AnalogContext,
        analog_tile: Any,
        input_: Tensor,
        shared_weights: Optional[Tensor] = None,
        is_test: bool = False,
    ) -> Tensor:
        """Execute the forward pass in the analog tile.
        Note: Indexed versions can used when analog_ctx.use_indexed is
        set to True.
        """
        # `ctx` is the parameter required by PyTorch to store the context
        # no need to pass it through ```AnalogFunction.apply(...)````.
        # Store in context for using during `backward()`.
        ctx.analog_ctx = analog_ctx
        ctx.analog_tile = analog_tile
        ctx.shared_weights = None
        ctx.saved_analog_tensors = [input_]
        runtime = analog_tile.get_runtime()

        use_indexed = analog_ctx.use_indexed
        if shared_weights is not None:
            ctx.shared_weights = shared_weights
            analog_tile.ensure_shared_weights(shared_weights)
            analog_ctx.use_torch_update = True
        else:
            analog_ctx.use_torch_update = False

        # Invoke the forward pass in the tile instance.
        if use_indexed:
            out = analog_tile.joint_forward_indexed(input_, is_test, ctx)
        else:
            out = analog_tile.joint_forward(input_, is_test, ctx)

        if runtime.offload_input:
            ctx.saved_analog_tensors[0] = ctx.saved_analog_tensors[0].cpu()

        ctx.save_for_backward(*ctx.saved_analog_tensors)
        ctx.saved_analog_tensors = []
        return out

    @staticmethod
    @no_grad()
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[
        Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]
    ]:
        """Execute the backward pass in the analog tile."""
        analog_ctx = ctx.analog_ctx
        analog_tile = ctx.analog_tile
        ctx.saved_analog_tensors = ctx.saved_tensors
        input_ = ctx.saved_analog_tensors[0]
        runtime = analog_tile.get_runtime()

        shared_weights_grad = None
        use_indexed = analog_ctx.use_indexed

        if ctx.shared_weights is not None:
            analog_tile.ensure_shared_weights(ctx.shared_weights)

        # Call the backward function in the tile instance.
        if use_indexed:
            grad_input = analog_tile.backward_indexed(grad_output, ctx)
        else:
            grad_input = analog_tile.backward(grad_output, ctx)

        if analog_ctx.use_torch_update:
            if runtime.offload_input:
                input_ = input_.to(analog_tile.device)

            # Grad computed directly (for inference training)
            shared_weights_grad = empty_like(ctx.shared_weights)
            analog_tile.set_delta_weights(shared_weights_grad)
            if use_indexed:
                analog_tile.update_indexed(input_, grad_output)
            else:
                analog_tile.update(input_, grad_output)
            analog_tile.reset_delta_weights()
        else:
            # Store activation and errors for optimizer (for analog training)
            analog_ctx.analog_input.append(input_)

            if runtime.offload_gradient:
                store_gradients = grad_output.cpu()
            else:
                store_gradients = grad_output
            analog_ctx.analog_grad_output.append(store_gradients)

        ctx.saved_analog_tensors = []
        return None, None, grad_input, shared_weights_grad, None
