# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Parameters describing standard convolutions."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ConvParameters:
    """Describe a standard convolution used for indexed digital gradients.

    Each tuple contains one entry per spatial dimension. Indexed convolutions
    use zero padding and a single group. Input and output channel counts are
    determined separately for each tile, including mapped convolution tiles.
    """

    kernel_size: Tuple[int, ...]
    """Spatial kernel dimensions."""

    stride: Tuple[int, ...]
    """Stride along each spatial dimension."""

    padding: Tuple[int, ...]
    """Zero padding on each side of each spatial dimension."""

    dilation: Tuple[int, ...]
    """Spacing between kernel elements along each spatial dimension."""
