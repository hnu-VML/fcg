#!/usr/bin/env python3
"""
Training and evaluation utilities for FusionNet.

These helpers keep the main training script small by encapsulating
common loops and checkpoint handling.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import math
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def format_metrics(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "n/a"
    return ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in the batch dict onto the target device."""
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def _select_objective_losses(
    loss_dict: Dict[str, torch.Tensor], surgery_keys: Optional[Sequence[str]]
) -> Sequence[torch.Tensor]:
    """
    Build a list of objective tensors for gradient surgery.
    Defaults to all tensor losses except the aggregated 'total'.
    """
    keys = list(surgery_keys) if surgery_keys else [k for k in loss_dict if k != "total"]
    obj_names = []
    objectives = []
    for key in keys:
        val = loss_dict.get(key)
        if torch.is_tensor(val):
            obj_names.append(key)
            objectives.append(val)
    return obj_names,objectives

def _grad_cosine_similarity(gi, gj, eps: float = 1e-12) -> float:
    """Cosine similarity between two per-parameter gradient lists."""
    dot = 0.0
    ni = 0.0
    nj = 0.0
    for a, b in zip(gi, gj):
        dot += float((a * b).sum().item())
        ni += float((a * a).sum().item())
        nj += float((b * b).sum().item())
    denom = math.sqrt(ni) * math.sqrt(nj)
    if denom < eps:
        return 0.0
    return dot / (denom + eps)


def _grad_l2_norm(gi, eps: float = 1e-12) -> float:
    """L2 norm of a per-parameter gradient list."""
    total = 0.0
    for g in gi:
        total += float((g * g).sum().item())
    return math.sqrt(total + eps)


def _all_pairwise_grad_cosines(grad_list, obj_names=None) -> Dict[str, float]:
    """
    Compute cosine similarities for all gradient pairs (i<j).
    Returns dict of scalars.
    """
    out = {}
    n = len(grad_list)
    if n <= 1:
        return out

    names = obj_names if obj_names and len(obj_names) == n else [str(i) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _grad_cosine_similarity(grad_list[i], grad_list[j])
            out[f"{names[i]}__vs__{names[j]}"] = c
    return out


def _all_pairwise_grad_norm_ratios(grad_list, obj_names=None, eps: float = 1e-12) -> Dict[str, float]:
    """Compute pairwise norm ratios (i/j) for all gradient pairs (i<j)."""
    out = {}
    n = len(grad_list)
    if n <= 1:
        return out
    names = obj_names if obj_names and len(obj_names) == n else [str(i) for i in range(n)]
    norms = [_grad_l2_norm(gi, eps=eps) for gi in grad_list]
    for i in range(n):
        for j in range(i + 1, n):
            out[f"{names[i]}__over__{names[j]}"] = norms[i] / (norms[j] + eps)
    return out


def _compute_grads_per_objective(objectives: Sequence[torch.Tensor], params: Sequence[torch.nn.Parameter]):
    """Compute per-objective gradients with autograd.grad so they stay separate."""
    grads = []
    for idx, loss in enumerate(objectives):
        grad = torch.autograd.grad(
            loss,
            params,
            retain_graph=idx < len(objectives) - 1,
            allow_unused=True,
        )
        grads.append([g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(grad, params)])
    return grads


def _pcgrad_combine(grad_list):
    """Project conflicting gradients (PCGrad) and average the results."""
    if not grad_list:
        return []
    if len(grad_list) == 1:
        return [g.clone() for g in grad_list[0]]

    projected = []
    num_obj = len(grad_list)
    for i, gi in enumerate(grad_list):
        gi_proj = [g.clone() for g in gi]
        order = list(range(num_obj))
        random.shuffle(order)
        for j in order:
            if j == i:
                continue
            gj = grad_list[j]
            dot = sum((g1 * g2).sum() for g1, g2 in zip(gi_proj, gj))
            if dot < 0:
                denom = sum((g2 * g2).sum() for g2 in gj) + 1e-12
                gi_proj = [g1 - (dot / denom) * g2 for g1, g2 in zip(gi_proj, gj)]
        projected.append(gi_proj)
    combined = [sum(gs) / float(num_obj) for gs in zip(*projected)]
    return combined


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_steps: Optional[int],
    grad_clip: float,
    grad_surgery: str = "none",
    surgery_keys: Optional[Sequence[str]] = None,
    writer: Optional[SummaryWriter] = None,
    global_step_start: int = 0,
) -> Tuple[Dict[str, float], int]:
    model.train()
    metric_sums: Dict[str, float] = {}
    steps = 0
    global_step = global_step_start
    log_keys = []
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=100, dynamic_ncols=True)
    for batch_idx, batch in enumerate(progress, start=1):
        batch = _move_batch_to_device(batch, device)
        sar = batch["sar"]
        opt = batch["opt"]
        optimizer.zero_grad(set_to_none=True)
        output = model(sar, opt)
        loss_dict = loss_fn(output, batch)
        loss = loss_dict["total"]

        if grad_surgery == "pcgrad":
            params = [p for p in model.parameters() if p.requires_grad]

            # --- [修改] 取回 objective 名称 + tensor ---
            obj_names, objectives = _select_objective_losses(loss_dict, surgery_keys)

            if not objectives:
                loss.backward()
            else:
                grads = _compute_grads_per_objective(objectives, params)

                # --- [新增] 计算所有梯度对的余弦相似度（投影前） ---
                pair_cos = _all_pairwise_grad_cosines(grads, obj_names)
                # --- [新增] 计算两两梯度模长比（投影前） ---
                norm_ratios = _all_pairwise_grad_norm_ratios(grads, obj_names)

                # --- [新增] 写 TensorBoard（像 loss 一样每 step 记录） ---
                # 注意：这里用 global_step+1（因为你后面才 global_step += 1）
                if writer and (pair_cos or norm_ratios):
                    step_for_tb = global_step + 1
                    for name, val in pair_cos.items():
                        writer.add_scalar(f"pcgrad/cos/{name}", val, step_for_tb)
                    # 可选：加两个汇总曲线，快速看冲突程度
                    vals = list(pair_cos.values())
                    if vals:
                        writer.add_scalar("pcgrad/cos_min", min(vals), step_for_tb)
                        writer.add_scalar("pcgrad/cos_mean", sum(vals) / len(vals), step_for_tb)
                    for name, val in norm_ratios.items():
                        writer.add_scalar(f"pcgrad/norm_ratio/{name}", val, step_for_tb)

                combined = _pcgrad_combine(grads)
                for param, grad in zip(params, combined):
                    grad_to_set = grad.detach()
                    if param.grad is None:
                        param.grad = grad_to_set.clone()
                    else:
                        param.grad.copy_(grad_to_set)
        else:
            loss.backward()


        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if device.type == "cuda" and torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated(device=device) / 1024 ** 2
        else:
            mem_mb = 0.0

        detached_losses = {
            key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for key, value in loss_dict.items()
        }
        if not log_keys and "total" in detached_losses:
            log_keys.append("total")
        for key in detached_losses:
            if key not in log_keys:
                log_keys.append(key)
        for key, value in detached_losses.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value
        steps += 1
        global_step += 1
        should_log = (
            log_interval is None
            or log_interval <= 0
            or batch_idx % log_interval == 0
            or batch_idx == 1
        )
        if should_log:
            avg_metrics = {key: metric_sums[key] / steps for key in metric_sums}
            parts = []
            keys_to_show = log_keys or list(avg_metrics.keys())
            for key in keys_to_show:
                if key in avg_metrics:
                    parts.append(f"{key}={avg_metrics[key]:.4f}")
            parts.append(f"mem={mem_mb:.1f}MB")
            progress.set_postfix_str(" | ".join(parts))
            if writer:
                for key, value in detached_losses.items():
                    writer.add_scalar(f"loss/train_{key}", value, global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
        if max_steps and steps >= max_steps:
            LOGGER.info("Reached max_steps=%d for epoch %d.", max_steps, epoch)
            break

    count = max(steps, 1)
    avg_metrics = {key: metric_sums[key] / count for key in metric_sums}
    progress.close()
    return avg_metrics, global_step


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    loss_fn: nn.Module,
    device: torch.device,
    save_image_names: Optional[Set[str]] = None,
    save_dir: Optional[Path] = None,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    eval_tag: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None
    model.eval()
    metric_sums: Dict[str, float] = {}
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", ncols=100, dynamic_ncols=True):
            batch = _move_batch_to_device(batch, device)
            sar = batch["sar"]
            opt = batch["opt"]
            output = model(sar, opt)
            loss_dict = loss_fn(output, batch)
            for key, value in loss_dict.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + (
                    float(value.detach().item()) if torch.is_tensor(value) else float(value)
                )
            steps += 1
            if save_image_names and save_dir:
                batch_names = batch.get("name") or []
                matched_indices = [idx for idx, name in enumerate(batch_names) if name in save_image_names]
                if matched_indices:
                    for idx in matched_indices:
                        sample_name = batch_names[idx]
                        sample_dir = save_dir / sample_name
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        sar_img = sar[idx].detach().cpu()
                        opt_img = opt[idx].detach().cpu()
                        out_img = output[idx].detach().cpu()
                        save_image(sar_img, sample_dir / "sar.png")
                        save_image(opt_img, sample_dir / "opt.png")
                        save_image(out_img, sample_dir / "output.png")
                        if writer is not None:
                            tag_prefix = eval_tag or "eval"
                            writer.add_image(f"{tag_prefix}/{sample_name}/sar", sar_img, global_step or steps)
                            writer.add_image(f"{tag_prefix}/{sample_name}/opt", opt_img, global_step or steps)
                            writer.add_image(f"{tag_prefix}/{sample_name}/output", out_img, global_step or steps)
    count = max(steps, 1)
    return {key: metric_sums[key] / count for key in metric_sums}


def save_checkpoint(state: dict, checkpoint_dir: Path, name: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / name
    torch.save(state, path)
    LOGGER.info("Saved checkpoint to %s", path)
    return path


def _capture_rng_state() -> Dict[str, object]:
    """Snapshot RNG states so training can be resumed deterministically."""
    state: Dict[str, object] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    try:
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state: Optional[Dict[str, object]]) -> None:
    """Restore RNG states if present in the checkpoint."""
    if not state:
        return
    try:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "python" in state:
            random.setstate(state["python"])
    except Exception as exc:
        LOGGER.warning("Failed to restore RNG state fully: %s", exc)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler,
    *,
    load_rng_state: bool = True,
) -> Tuple[int, float, int, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if load_rng_state:
        _restore_rng_state(checkpoint.get("rng_state"))
    start_epoch = checkpoint.get("epoch", 0)
    best_val = checkpoint.get("best_val", float("inf"))
    global_step = checkpoint.get("global_step", 0)
    LOGGER.info("Resumed from %s at epoch %d (global_step=%d).", path, start_epoch, global_step)
    return start_epoch, best_val, global_step, checkpoint.get("config", {})


__all__ = [
    "format_metrics",
    "train_one_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "_capture_rng_state",
]
