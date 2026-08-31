#!/usr/bin/env python3
"""Run the downstream nanoGPT script with a temporary large-MVM guard bypass.

This helper is intentionally external to analog-transformer: the downstream
fail-closed guard remains unchanged while validating an AIHWKit candidate.
"""

import os
import runpy
import sys
from typing import Any

import torch
from torch import Tensor
from torch import distributed as dist

import aihwkit
from aihwkit.optim.context import AnalogContext
from MixedPrecisionOptim.LogicalAdamW import LogicalMpAdamW  # pylint: disable=import-error
import model  # pylint: disable=import-error
import scaled_analog_linear  # pylint: disable=import-error


def _install_logit_probe() -> None:
    """Log one vocabulary-vector sample for every downstream forward call."""

    original_forward = model.GPT.forward

    def forward_with_probe(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        logits = output[0]
        if logits is not None and logits.numel() > 0:
            sample = logits.detach()[0, 0].float()
            finite = torch.isfinite(sample)
            finite_count = int(finite.sum())
            print(
                "[aihwkit-large-mbatch-probe] "
                f"rank={os.environ.get('RANK', '0')} "
                f"shape={tuple(logits.shape)} "
                f"finite={finite_count}/{sample.numel()} "
                f"min={sample.min().item():.9g} max={sample.max().item():.9g} "
                f"mean={sample.mean().item():.9g} std={sample.std().item():.9g}",
                flush=True,
            )
        return output

    model.GPT.forward = forward_with_probe


def _flatten_tensors(value: Any, path: str) -> list[tuple[str, Tensor]]:
    """Flatten tensor-bearing optimizer/tile state into named leaves."""

    if isinstance(value, Tensor):
        return [(path, value)]
    if isinstance(value, dict):
        leaves = []
        for key in sorted(value, key=str):
            leaves.extend(_flatten_tensors(value[key], f"{path}.{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_flatten_tensors(item, f"{path}[{index}]"))
        return leaves
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [(path, torch.tensor(float(value), dtype=torch.float64))]
    return []


def _optimizer_rank_difference(optimizer: LogicalMpAdamW) -> tuple[float, str, int]:
    """Return the maximum post-step tensor difference across all DDP ranks."""

    leaves = []
    seen = set()
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, param in enumerate(group["params"]):
            if id(param) in seen:
                continue
            seen.add(id(param))
            prefix = f"group{group_index}.param{param_index}"
            leaves.extend(_flatten_tensors(param, f"{prefix}.value"))
            leaves.extend(_flatten_tensors(optimizer.state[param], f"{prefix}.state"))
            if isinstance(param, AnalogContext):
                tile_state = optimizer._ddp_capture_tile_state(param)
                leaves.extend(_flatten_tensors(tile_state, f"{prefix}.tile"))

    metadata = [(path, tuple(value.shape)) for path, value in leaves]
    all_metadata = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_metadata, metadata)
    if any(item != metadata for item in all_metadata):
        raise RuntimeError("DDP optimizer/tile state metadata differs across ranks")

    device = torch.device("cuda", torch.cuda.current_device())
    maximum = torch.zeros((), dtype=torch.float64, device=device)
    worst_path = ""
    for path, value in leaves:
        local = value.detach().to(device=device, dtype=torch.float64)
        reference = local.clone()
        dist.broadcast(reference, src=0)
        if local.numel():
            difference = (local - reference).abs().max()
            if not torch.isfinite(difference):
                raise RuntimeError(f"non-finite DDP state difference at {path}")
            if difference > maximum:
                maximum.copy_(difference)
                worst_path = path

    local_candidate = (float(maximum), worst_path)
    candidates = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(candidates, local_candidate)
    global_maximum, global_worst_path = max(candidates, key=lambda item: item[0])
    return float(global_maximum), str(global_worst_path), len(leaves)


def _install_ddp_state_probe() -> None:
    """Check optimizer and complete tile state after every DDP update."""

    original_step = LogicalMpAdamW.step
    step_count = 0

    def step_with_probe(self, *args, **kwargs):
        nonlocal step_count
        result = original_step(self, *args, **kwargs)
        if dist.is_initialized():
            maximum, worst_path, leaves = _optimizer_rank_difference(self)
            if dist.get_rank() == 0:
                print(
                    "[aihwkit-large-mbatch-ddp-state] "
                    f"step={step_count} leaves={leaves} "
                    f"max_abs={maximum:.9g} worst={worst_path}",
                    flush=True,
                )
        step_count += 1
        return result

    LogicalMpAdamW.step = step_with_probe


def main() -> None:
    """Bypass only the empirical guard, then execute the requested script."""

    if len(sys.argv) < 2:
        raise SystemExit("usage: run_downstream_without_guard.py SCRIPT [ARGS ...]")

    target = sys.argv[1]
    old_limit = scaled_analog_linear.MAX_NONIDEAL_CUDA_MVM_ROWS
    scaled_analog_linear.MAX_NONIDEAL_CUDA_MVM_ROWS = (1 << 63) - 1
    print(
        "[aihwkit-large-mbatch-probe] temporary guard bypass: "
        f"{old_limit} -> {scaled_analog_linear.MAX_NONIDEAL_CUDA_MVM_ROWS}",
        flush=True,
    )
    print(
        f"[aihwkit-large-mbatch-probe] aihwkit={aihwkit.__file__}",
        flush=True,
    )
    _install_logit_probe()
    if os.environ.get("AIHWKIT_DOWNSTREAM_DDP_STATE_PROBE") == "1":
        _install_ddp_state_probe()
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
