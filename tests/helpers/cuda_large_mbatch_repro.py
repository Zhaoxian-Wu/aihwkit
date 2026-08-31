#!/usr/bin/env python3
"""Reproduce and characterize large-batch non-perfect CUDA MVM behavior."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from typing import Dict, Iterable, Tuple

import torch

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType
from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery


class TensorStats:
    """Accumulate tensor statistics without allocating full-size temporaries."""

    def __init__(self) -> None:
        self.count = 0
        self.finite_count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0
        self.total_square = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        """Add one bounded-size tensor chunk."""

        finite = torch.isfinite(tensor)
        chunk_count = tensor.numel()
        chunk_finite = int(finite.sum().item())
        self.count += chunk_count
        self.finite_count += chunk_finite
        if chunk_finite != chunk_count:
            return
        self.minimum = min(self.minimum, tensor.min().item())
        self.maximum = max(self.maximum, tensor.max().item())
        self.total += tensor.sum(dtype=torch.float64).item()
        self.total_square += tensor.square().sum(dtype=torch.float64).item()

    def result(self) -> Dict[str, float]:
        """Return finalized finite/output-distribution statistics."""

        if self.finite_count == 0 or self.finite_count != self.count:
            fraction = self.finite_count / self.count if self.count else 0.0
            return {
                "min": math.nan,
                "max": math.nan,
                "mean": math.nan,
                "std": math.nan,
                "finite_fraction": fraction,
            }
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
            "finite_fraction": 1.0,
        }


def tensor_stats(tensor: torch.Tensor, chunk_elements: int = 64 * 1024 * 1024) -> Dict[str, float]:
    """Return compact statistics with bounded temporary memory."""

    stats = TensorStats()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return {
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "finite_fraction": 0.0,
        }
    for start in range(0, flat.numel(), chunk_elements):
        stats.update(flat[start : start + chunk_elements])
    return stats.result()


def make_tile(
    out_size: int, in_size: int, perfect: bool, random_noise: bool
) -> AnalogTileWithoutPeriphery:
    """Create a deterministic IO-managed tile with no random noise."""

    config = SingleRPUConfig(device=ConstantStepDevice())
    for io in (config.forward, config.backward):
        io.is_perfect = perfect
        io.inp_res = -1.0
        io.out_res = -1.0
        io.inp_noise = 0.01 if random_noise else 0.0
        io.out_noise = 0.01 if random_noise else 0.0
        io.inp_sto_round = False
        io.out_sto_round = False
        io.noise_management = NoiseManagementType.NONE
        io.bound_management = BoundManagementType.NONE

    tile = AnalogTileWithoutPeriphery(out_size, in_size, config, bias=False).cuda()
    weights = torch.linspace(-0.75, 0.75, out_size * in_size, device="cuda").reshape(
        out_size, in_size
    )
    tile.tile.set_weights(weights)
    tile.eval()
    return tile


@torch.no_grad()
def stream_reference(
    tile: AnalogTileWithoutPeriphery,
    inputs: torch.Tensor,
    direct: torch.Tensor,
    chunk_rows: int,
    direction: str,
    transposed: bool,
) -> Tuple[Dict[str, float], float, float, float]:
    """Compare against row chunks without retaining a second full output."""

    reference_stats = TensorStats()
    max_abs_error = 0.0
    total_abs_error = 0.0
    total_elements = 0
    rows = inputs.shape[1] if transposed else inputs.shape[0]
    started = time.perf_counter()
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        input_chunk = inputs[:, start:stop] if transposed else inputs[start:stop]
        if direction == "forward":
            reference = tile.tile.forward(
                input_chunk, False, transposed, transposed, True, False
            )
        else:
            reference = tile.tile.backward(
                input_chunk, False, transposed, transposed, False
            )
        direct_chunk = direct[:, start:stop] if transposed else direct[start:stop]
        reference_stats.update(reference)
        difference = (direct_chunk - reference).abs()
        max_abs_error = max(max_abs_error, difference.max().item())
        total_abs_error += difference.sum(dtype=torch.float64).item()
        total_elements += difference.numel()
        del difference, direct_chunk, reference, input_chunk
    torch.cuda.synchronize()
    return (
        reference_stats.result(),
        max_abs_error,
        total_abs_error / total_elements,
        time.perf_counter() - started,
    )


@torch.no_grad()
def run_case(
    tile: AnalogTileWithoutPeriphery,
    rows: int,
    in_size: int,
    out_size: int,
    chunk_rows: int,
    transposed: bool,
) -> Dict[str, object]:
    """Run one direct and manually chunked forward case."""

    row_index = torch.arange(rows, device="cuda", dtype=torch.float32).unsqueeze(1)
    col_index = torch.arange(in_size, device="cuda", dtype=torch.float32).unsqueeze(0)
    inputs = 0.5 * torch.sin(row_index * 0.001 + col_index * 0.17)
    if transposed:
        inputs = inputs.t().contiguous()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    direct = tile.tile.forward(inputs, False, transposed, transposed, True, False)
    torch.cuda.synchronize()
    direct_seconds = time.perf_counter() - started

    reference_stats, max_abs_error, mean_abs_error, chunked_seconds = stream_reference(
        tile, inputs, direct, chunk_rows, "forward", transposed
    )
    result: Dict[str, object] = {
        "rows": rows,
        "in_size": in_size,
        "out_size": out_size,
        "elements": rows * out_size,
        "transposed": transposed,
        "int32_max": torch.iinfo(torch.int32).max,
        "direct": tensor_stats(direct),
        "reference": reference_stats,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "direct_seconds": direct_seconds,
        "chunked_seconds": chunked_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    del direct, inputs, row_index, col_index
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def run_backward_case(
    tile: AnalogTileWithoutPeriphery,
    rows: int,
    in_size: int,
    out_size: int,
    chunk_rows: int,
    transposed: bool,
) -> Dict[str, object]:
    """Run one direct and manually chunked backward case."""

    inputs = torch.empty((rows, out_size), device="cuda", dtype=torch.float32)
    inputs.uniform_(-0.5, 0.5)
    if transposed:
        inputs = inputs.t().contiguous()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    direct = tile.tile.backward(inputs, False, transposed, transposed, False)
    torch.cuda.synchronize()
    direct_seconds = time.perf_counter() - started

    reference_stats, max_abs_error, mean_abs_error, chunked_seconds = stream_reference(
        tile, inputs, direct, chunk_rows, "backward", transposed
    )
    result: Dict[str, object] = {
        "rows": rows,
        "input_width": out_size,
        "output_width": in_size,
        "input_elements": rows * out_size,
        "transposed": transposed,
        "int32_max": torch.iinfo(torch.int32).max,
        "direct": tensor_stats(direct),
        "reference": reference_stats,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "direct_seconds": direct_seconds,
        "chunked_seconds": chunked_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    del direct, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_rows(value: str) -> Iterable[int]:
    """Parse comma-separated positive row counts."""

    rows = [int(item) for item in value.split(",")]
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be a comma-separated list of positive integers")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=parse_rows, default=parse_rows("41984,43008"))
    parser.add_argument("--in-size", type=int, default=1)
    parser.add_argument("--out-size", type=int, default=50257)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--perfect", action="store_true")
    parser.add_argument("--random-noise", action="store_true")
    parser.add_argument("--transposed", action="store_true")
    parser.add_argument("--direction", choices=("forward", "backward", "both"), default="forward")
    args = parser.parse_args()

    if args.in_size <= 0 or args.out_size <= 0 or args.chunk_rows <= 0:
        parser.error("in-size, out-size, and chunk-rows must be positive")

    torch.manual_seed(20260829)
    torch.cuda.manual_seed_all(20260829)
    tile = make_tile(args.out_size, args.in_size, args.perfect, args.random_noise)
    cases = []
    for rows in args.rows:
        if args.direction in ("forward", "both"):
            cases.append(
                {
                    "direction": "forward",
                    **run_case(
                        tile,
                        rows,
                        args.in_size,
                        args.out_size,
                        args.chunk_rows,
                        args.transposed,
                    ),
                }
            )
        if args.direction in ("backward", "both"):
            cases.append(
                {
                    "direction": "backward",
                    **run_backward_case(
                        tile,
                        rows,
                        args.in_size,
                        args.out_size,
                        args.chunk_rows,
                        args.transposed,
                    ),
                }
            )

    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "perfect": args.perfect,
        "random_noise": args.random_noise,
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
