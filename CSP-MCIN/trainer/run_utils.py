from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def current_timestamp() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def prepare_run_dirs(
    base_dir: Path, run_name: Optional[str], resume_run: Optional[Path]
) -> Tuple[Path, Path, Path, Path]:
    if resume_run:
        run_dir = resume_run.expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        name = run_name or current_timestamp()
        run_dir = (base_dir / name).resolve()
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = run_dir / "ckpt"
    images_dir = run_dir / "images"
    tb_dir = run_dir / "tb"
    for folder in (ckpt_dir, images_dir, tb_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, images_dir, tb_dir


def find_latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "ckpt"
    search_dir = ckpt_dir if ckpt_dir.exists() else run_dir
    candidates = list(search_dir.glob("*.pt")) + list(search_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {search_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def config_snapshot(args, source_config: Path) -> Dict[str, Any]:
    return {
        "data": {
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "image_size": args.image_size,
            "inchannel": list(args.inchannel),
            "outchannel": args.outchannel,
        },
        "dataloader": {
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory),
        },
        "model": {
            "model_class": args.model_class,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "dropout": args.dropout,
            "final_embedding_dim": args.final_embedding_dim,
            "depth": args.depth,
            "stage1_checkpoint": str(args.stage1_checkpoint) if args.stage1_checkpoint else None,
        },
        "optimization": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "type": args.scheduler,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "min_lr": args.min_lr,
        },
        "training": {
            "device": args.device,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "grad_clip": args.grad_clip,
            "grad_surgery": args.grad_surgery,
            "surgery_keys": list(args.surgery_keys) if args.surgery_keys else None,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "save_best": args.save_best,
            "stage2": args.stage2,
        },
        "loss": {
            "stage1_loss": args.stage1_loss,
            "stage2_loss": args.stage2_loss,
            "stage1_params": args.stage1_params,
            "stage2_params": args.stage2_params,
        },
        "logging": {
            "runs_dir": str(args.runs_dir),
            "run_dir": str(args.run_dir),
            "run_name": args.run_name,
            "tensorboard": not args.no_tensorboard,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "config_source": str(source_config),
            "eval_save_names": list(args.eval_save_names) if args.eval_save_names else [],
        },
    }


def persist_run_config(run_dir: Path, snapshot: Dict[str, Any]) -> Path:
    target = run_dir / "config.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)
    return target


__all__ = [
    "config_snapshot",
    "current_timestamp",
    "find_latest_checkpoint",
    "persist_run_config",
    "prepare_run_dirs",
]
