# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Parameter context for analog tiles."""

# pylint: disable=attribute-defined-outside-init

from typing import Optional, Type, Union, Any, TYPE_CHECKING

from torch import dtype, Tensor, no_grad
from torch.nn import Parameter
from torch import device as torch_device

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.base import SimulatorTileWrapper


class AnalogContext(Parameter):
    """Context for analog optimizer.

    Note: `data` attribution, inherited from `torch.nn.Parameter`, is a tensor of training parameter
    If `analog_bias` (which is provided by `analog_tile`) is False,
        `data` has the same meaning as `torch.nn.Parameter`
    If `analog_bias` (which is provided by `analog_tile`) is True,
        The last column of `data` is the `bias` term

    Even though it allows us to access the weights directly, always keep in mind that it is used
        only for studying propuses. To simulate the real reading, call the `read_weights` method
        instead, i.e. given `analog_ctx: AnalogContext`,
        estimated_weights, estimated_bias = analog_ctx.analog_tile.read_weights()

    Similarly, even though this feature allows us to update the weights directly,
        always keep in mind that the real RPU devices change their weights only
        by "pulse update" method.

    Therefore, use the following update methods instead of
        writing `data` directly in the analog optimizer:
        ---
        analog_ctx.analog_tile.update(...)
        analog_ctx.analog_tile.update_indexed(...)
        ---
    """

    def __new__(
        cls: Type["AnalogContext"],
        analog_tile: "SimulatorTileWrapper",
        parameter: Optional[Parameter] = None,
    ) -> "AnalogContext":
        # pylint: disable=signature-differs
        if parameter is None:
            return Parameter.__new__(
                cls,
                data=analog_tile.tile.get_weights(),
                requires_grad=True,
            )
        # analog_tile.tile can comes from different classes:
        #   aihwkit.silulator.rpu_base.devices.AnalogTile (C++)
        #   TorchInferenceTile (Python)
        # It stores the "weight" matrix;
        #   If analog_tile.analog_bias is True, it also stores the "bias" matrix

        parameter.__class__ = cls
        return parameter

    def __init__(
        self, analog_tile: "SimulatorTileWrapper", parameter: Optional[Parameter] = None
    ):  # pylint: disable=unused-argument
        super().__init__()
        self.analog_tile = analog_tile
        self.use_torch_update = False
        self.use_indexed = False
        self.analog_input = []  # type: list
        self.analog_grad_output = []  # type: list
        self.reset(analog_tile)

    def set_indexed(self, value: bool = True) -> None:
        """Set the context to forward_indexed."""
        self.use_indexed = value

    def set_data(self, data: Tensor) -> None:
        """Set the data value of the Tensor."""
        with no_grad():
            self.data.copy_(data)

    def get_data(self) -> Tensor:
        """Get the data value of the underlying Tensor."""
        return self.data.detach()

    def reset(self, analog_tile: Optional["SimulatorTileWrapper"] = None) -> None:
        """Reset the gradient trace and optionally sets the tile pointer."""

        if analog_tile is not None:
            self.analog_tile = analog_tile
            self.analog_tile.analog_ctx = self

        self.analog_input = []
        self.analog_grad_output = []

    def has_gradient(self) -> bool:
        """Return whether a gradient trace was stored."""
        return len(self.analog_input) > 0

    def __copy__(self) -> Parameter:
        """Turn off copying of the pointers. Context will be re-created
        when tile is created"""
        return Parameter(self.data)

    def __deepcopy__(self, memo: Any) -> Parameter:
        """Turn off deep copying. Context will be re-created when tile is created"""
        return Parameter(self.data)

    def cuda(self, device: Optional[Union[torch_device, str, int]] = None) -> "AnalogContext":
        """Move the context to a cuda device.

        Args:
             device: the desired device of the tile.

        Returns:
            This context in the specified device.
        """
        if not self.analog_tile.is_cuda:
            self.data = self.analog_tile.tile.get_weights()  # type: Tensor
            self.analog_tile = self.analog_tile.cuda(device)
            self.reset(self.analog_tile)
        return self

    def cpu(self) -> "AnalogContext":
        """Move the context to CPU.

        Note:
            This is a no-op for CPU context.

        Returns:
            self
        """
        self.data = self.data.cpu()
        if self.analog_tile is not None and self.analog_tile.is_cuda:
            self.analog_tile = self.analog_tile.cpu()
            self.reset(self.analog_tile)
        return self

    def to(self, *args: Any, **kwargs: Any) -> "AnalogContext":
        """Move analog tiles of the current context to a device.

        Note:
            Please be aware that moving analog tiles from GPU to CPU is
            currently not supported.

        Caution:
            Other tensor conversions than moving the device to CUDA,
            such as changing the data type are not supported for analog
            tiles and will be simply ignored.

        Returns:
            This module in the specified device.
        """
        # pylint: disable=invalid-name
        self.data = self.data.to(*args, **kwargs)
        device = None
        if "device" in kwargs:
            device = kwargs["device"]
        elif len(args) > 0 and not isinstance(args[0], (Tensor, dtype)):
            device = torch_device(args[0])

        if device is not None:
            device = torch_device(device)
            if device.type == "cuda" and not self.analog_tile.is_cuda:
                self.cuda(device)
            elif device.type == "cpu" and self.analog_tile.is_cuda:
                self.cpu()
        return self

    def __repr__(self) -> str:
        return "AnalogContext of " + self.analog_tile.get_brief_info()
