 #!/usr/bin/env python3
"""
Training script for FusionNet.

The script provides a configurable pipeline that:
  * builds paired SAR/optical datasets from directories or a metadata CSV
  * splits the data into train/validation folds
  * trains FusionNet with checkpointing and TensorBoard logging
"""

from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from data import create_train_and_val_dataloaders
import lossfunction as loss_module
import model as model_module
from training_engine import (
    _capture_rng_state,
    evaluate,
    format_metrics,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from trainer.config import flatten_config, load_config_file
from trainer.run_utils import (
    config_snapshot,
    find_latest_checkpoint,
    persist_run_config,
    prepare_run_dirs,
)

import warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FusionNet for SAR/Optical fusion.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Base directory to store runs (overrides config logging.runs_dir).")
    parser.add_argument("--resume", type=Path, default=None, help="Path to an existing run_* directory to resume.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run folder name; defaults to run_YYYYmmdd_HHMMSS.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard even if enabled in config.")
    return parser.parse_args()
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> Optimizer:
    return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer: Optimizer, args: argparse.Namespace):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return None


def _merge_args_with_checkpoint_config(args: argparse.Namespace, checkpoint_config: Dict[str, object]) -> argparse.Namespace:
    """
    For resume, prefer checkpoint settings to keep shapes/datasets consistent.
    Critical CLI/run-location knobs stay as provided.
    """
    if not checkpoint_config:
        return args
    keep_cli_keys = {
        "epochs",
        "runs_dir",
        "run_dir",
        "run_name",
        "resume_from",
        "ckpt_dir",
        "images_dir",
        "tb_dir",
        "config_path",
        "log_file",
        "no_tensorboard",
    }
    for key, ckpt_value in checkpoint_config.items():
        if key in keep_cli_keys or not hasattr(args, key):
            continue
        current = getattr(args, key)
        if current != ckpt_value:
            LOGGER.warning(
                "Overriding arg %s=%r with checkpoint value %r to match resumed run.",
                key,
                current,
                ckpt_value,
            )
            setattr(args, key, ckpt_value)
    return args


def _get_class_by_name(module, name: str, category: str):
    """
    Resolve a class from a module by name (case-insensitive), raising a clear
    error if it does not exist.
    """
    candidates = {attr.lower(): attr for attr in dir(module) if not attr.startswith("_")}
    lookup_key = name.lower()
    if lookup_key not in candidates:
        raise ValueError(f"Unknown {category} class '{name}' in {module.__name__}.")
    attr_name = candidates[lookup_key]
    cls = getattr(module, attr_name)
    if not isinstance(cls, type):
        raise ValueError(f"{category} '{name}' in {module.__name__} is not a class.")
    return cls


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model_cls = _get_class_by_name(model_module, args.model_class, "model")
    model = model_cls(
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        final_embedding_dim=args.final_embedding_dim,
        depth=args.depth,
        inchannel=args.inchannel,
        outchannel=args.outchannel,
    )
    return model.to(device)


def _build_loss(args: argparse.Namespace, device: torch.device) -> nn.Module:
    key = args.stage2_loss if args.stage2 else args.stage1_loss
    loss_cls = _get_class_by_name(loss_module, key, "loss")
    params = args.stage2_params if args.stage2 else args.stage1_params
    params = params or {}
    if not isinstance(params, dict):
        raise ValueError(f"Loss params for {key} must be a dict, got {type(params)}")
    return loss_cls(**params).to(device)


def main(cli_args: argparse.Namespace) -> None:
    config_path = cli_args.config.expanduser().resolve()
    raw_config = load_config_file(config_path)
    flat_config = flatten_config(raw_config, base_dir=config_path.parent)
    if cli_args.runs_dir is not None:
        flat_config["runs_dir"] = cli_args.runs_dir
    if cli_args.run_name is not None:
        flat_config["run_name"] = cli_args.run_name
    flat_config["no_tensorboard"] = cli_args.no_tensorboard or flat_config.get("no_tensorboard", False)
    resume_run = cli_args.resume or flat_config.get("resume_from")
    flat_config["resume_from"] = resume_run

    args = argparse.Namespace(**flat_config)
    args.config_path = config_path

    if args.train_dir is None or args.val_dir is None:
        raise ValueError("Config must provide data.train_dir and data.val_dir.")
    args.train_dir = Path(args.train_dir).expanduser().resolve()
    args.val_dir = Path(args.val_dir).expanduser().resolve()
    args.runs_dir = Path(args.runs_dir).expanduser().resolve()
    args.stage1_checkpoint = Path(args.stage1_checkpoint).expanduser().resolve() if args.stage1_checkpoint else None
    args.resume_from = Path(args.resume_from).expanduser().resolve() if args.resume_from else None
    args.stage1_params = dict(args.stage1_params or {})
    args.stage2_params = dict(args.stage2_params or {})
    if not args.train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {args.train_dir}")
    if not args.val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {args.val_dir}")
    args.inchannel = tuple(args.inchannel)
    args.eval_save_names = list(args.eval_save_names or [])
    args.pin_memory = bool(args.pin_memory)

    run_dir, ckpt_dir, images_dir, tb_dir = prepare_run_dirs(args.runs_dir, args.run_name, args.resume_from)
    args.run_dir = run_dir
    args.ckpt_dir = ckpt_dir
    args.images_dir = images_dir
    args.tb_dir = tb_dir
    args.run_name = run_dir.name

    configure_logging()
    writer: Optional[SummaryWriter] = None

    LOGGER.info("Using config file: %s", args.config_path)
    LOGGER.info("Run directory: %s", args.run_dir)
    checkpoint_config: Dict[str, object] = {}
    resume_checkpoint_path: Optional[Path] = None
    if args.resume_from:
        resume_checkpoint_path = find_latest_checkpoint(args.run_dir)
        checkpoint_blob = torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint_blob.get("config") or {}
        args = _merge_args_with_checkpoint_config(args, checkpoint_config)
        args.inchannel = tuple(args.inchannel)
        args.eval_save_names = list(args.eval_save_names or [])
        for path_key in ("train_dir", "val_dir", "stage1_checkpoint"):
            val = getattr(args, path_key, None)
            if val is not None and not isinstance(val, Path):
                setattr(args, path_key, Path(val).expanduser().resolve())
        LOGGER.info("Resuming from %s", resume_checkpoint_path)
    seed_everything(args.seed)

    snapshot = config_snapshot(args, args.config_path)
    persist_run_config(args.run_dir, snapshot)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    LOGGER.info("Using device: %s", device)

    train_loader, val_loader = create_train_and_val_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        inchannel=args.inchannel,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = _build_model(args, device)
    if args.stage1_checkpoint and not resume_checkpoint_path:
        checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        LOGGER.info(
            "Loaded stage1 pretrained model from %s, matched layers: %d/%d",
            args.stage1_checkpoint,
            len(pretrained_dict),
            len(model_dict),
        )
    loss_fn = _build_loss(args, device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Total params: %s, trainable: %s", f"{total:,}", f"{trainable:,}")

    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)
    LOGGER.info(
        "Grad surgery: %s (objectives=%s)",
        args.grad_surgery,
        args.surgery_keys if args.surgery_keys else "auto",
    )

    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    best_checkpoint_path: Optional[Path] = None
    if resume_checkpoint_path:
        start_epoch, best_val, global_step, _ = load_checkpoint(resume_checkpoint_path, model, optimizer, scheduler)
        if args.epochs <= start_epoch:
            new_epochs = start_epoch + 1
            LOGGER.warning(
                "epochs (%d) <= checkpoint epoch (%d); extending to %d.",
                args.epochs,
                start_epoch,
                new_epochs,
            )
            args.epochs = new_epochs

    if not args.no_tensorboard:
        purge_step = global_step if global_step > 0 else None
        writer = SummaryWriter(log_dir=args.tb_dir, purge_step=purge_step)
    eval_save_names: Set[str] = set(args.eval_save_names or [])

    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_steps=args.max_steps,
            grad_clip=args.grad_clip,
            grad_surgery=args.grad_surgery,
            surgery_keys=args.surgery_keys,
            writer=writer,
            global_step_start=global_step,
        )
        LOGGER.info("Epoch %d train losses: %s", epoch, format_metrics(train_metrics))

        val_metrics = None
        current_val = None
        if val_loader is not None and (epoch + 1) % args.eval_interval == 0:
            eval_dir: Optional[Path] = None
            # Keep TensorBoard tags stable so images/PR curves stack as steps on a single card.
            eval_tag: Optional[str] = "eval" if eval_save_names else None
            if eval_save_names:
                eval_dir = args.images_dir / f"eval_samples_epoch{epoch+1}"
                eval_dir.mkdir(parents=True, exist_ok=True)
            val_metrics = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                save_image_names=eval_save_names,
                save_dir=eval_dir,
                writer=writer,
                global_step=epoch + 1,
                eval_tag=eval_tag,
            )
            LOGGER.info("Epoch %d val losses: %s", epoch, format_metrics(val_metrics or {}))
            if writer and val_metrics:
                for key, value in val_metrics.items():
                    writer.add_scalar(f"loss/val_{key}", value, epoch + 1)
            current_val = (val_metrics or {}).get("total")

            if args.save_best and current_val is not None and current_val < best_val:
                best_val = current_val
                best_name = "best_epoch.pt"
                if best_checkpoint_path and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                best_checkpoint_path = save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "best_val": best_val,
                        "global_step": global_step,
                        "rng_state": _capture_rng_state(),
                        "config": vars(args),
                    },
                    args.ckpt_dir,
                    best_name,
                )

            if (epoch + 1) % args.checkpoint_interval == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "best_val": best_val,
                        "global_step": global_step,
                        "rng_state": _capture_rng_state(),
                        "config": vars(args),
                    },
                    args.ckpt_dir,
                    f"epoch_{epoch+1:04d}.pt",
                )

        if scheduler:
            scheduler.step()

    if writer:
        writer.close()
    LOGGER.info("Training complete.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
