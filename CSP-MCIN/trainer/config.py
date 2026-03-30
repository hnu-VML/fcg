from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_MARKERS = {"default", "defualt"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "train_dir": None,
        "val_dir": None,
        "image_size": 224,
        "inchannel": [1, 1],
        "outchannel": 1,
    },
    "dataloader": {
        "train_batch_size": 4,
        "val_batch_size": 4,
        "num_workers": 4,
        "pin_memory": False,
    },
    "model": {
        "model_class": "FusionNet_2gate",
        "num_heads": 1,
        "mlp_ratio": 1.0,
        "dropout": 0.1,
        "final_embedding_dim": 1024,
        "depth": 18,
        "stage1_checkpoint": None,
    },
    "optimization": {
        "lr": 1e-4,
        "weight_decay": 1e-4,
    },
    "scheduler": {
        "type": "none",
        "step_size": 10,
        "gamma": 0.5,
        "min_lr": 1e-6,
    },
    "training": {
        "device": "cuda",
        "epochs": 20,
        "max_steps": None,
        "seed": 42,
        "grad_clip": 1.0,
        "grad_surgery": "none",
        "surgery_keys": None,
        "log_interval": 20,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "save_best": False,
        "stage2": False,
    },
    "loss": {
        "stage1_loss": "pix_semantic_joint_loss_shallow_feature",
        "stage2_loss": "pix_semantic_joint_loss_shallow_feature",
        "stage1_params": {},
        "stage2_params": {},
    },
    "logging": {
        "runs_dir": "runs",
        "run_name": None,
        "tensorboard": True,
        "resume_from": None,
        "eval_save_names": [],
    },
}


def is_default_marker(value: Any) -> bool:
    return isinstance(value, str) and value.lower() in DEFAULT_MARKERS


def apply_defaults(user_cfg: Any, default_cfg: Any) -> Any:
    """Recursively fill missing or default-marked values with defaults."""
    if is_default_marker(user_cfg):
        return copy.deepcopy(default_cfg)
    if isinstance(default_cfg, dict):
        user_cfg = user_cfg or {}
        merged = {}
        for key, default_value in default_cfg.items():
            merged[key] = apply_defaults(user_cfg.get(key), default_value)
        for key, value in user_cfg.items():
            if key not in merged:
                merged[key] = value
        return merged
    if user_cfg is None:
        return copy.deepcopy(default_cfg)
    return user_cfg


def resolve_path(value: Optional[str], base_dir: Path) -> Optional[Path]:
    if value is None or is_default_marker(value):
        return None
    path_value = Path(value)
    if not path_value.is_absolute():
        path_value = (base_dir / path_value).expanduser()
    return path_value.resolve()


def load_config_file(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return apply_defaults(data, DEFAULT_CONFIG)


def flatten_config(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Map nested YAML config to the flat namespace consumed by training."""
    data_cfg = config.get("data", {})
    loader_cfg = config.get("dataloader", {})
    model_cfg = config.get("model", {})
    opt_cfg = config.get("optimization", {})
    sched_cfg = config.get("scheduler", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    logging_cfg = config.get("logging", {})

    train_dir = resolve_path(data_cfg.get("train_dir"), base_dir)
    val_dir = resolve_path(data_cfg.get("val_dir"), base_dir)
    stage1_ckpt = resolve_path(model_cfg.get("stage1_checkpoint"), base_dir)

    runs_dir_value = logging_cfg.get("runs_dir")
    if runs_dir_value is None or is_default_marker(runs_dir_value) or runs_dir_value == "runs":
        runs_dir = Path("runs")
    else:
        runs_dir = resolve_path(runs_dir_value, base_dir) or Path("runs")
    resume_run = resolve_path(logging_cfg.get("resume_from"), base_dir)
    eval_save_names = logging_cfg.get("eval_save_names")

    flat: Dict[str, Any] = {
        "train_dir": train_dir,
        "val_dir": val_dir,
        "image_size": data_cfg.get("image_size"),
        "train_batch_size": loader_cfg.get("train_batch_size"),
        "val_batch_size": loader_cfg.get("val_batch_size"),
        "num_workers": loader_cfg.get("num_workers"),
        "pin_memory": bool(loader_cfg.get("pin_memory")),
        "epochs": train_cfg.get("epochs"),
        "device": train_cfg.get("device"),
        "lr": opt_cfg.get("lr"),
        "weight_decay": opt_cfg.get("weight_decay"),
        "grad_clip": train_cfg.get("grad_clip"),
        "grad_surgery": train_cfg.get("grad_surgery"),
        "surgery_keys": train_cfg.get("surgery_keys"),
        "log_interval": train_cfg.get("log_interval"),
        "eval_interval": train_cfg.get("eval_interval"),
        "checkpoint_interval": train_cfg.get("checkpoint_interval"),
        "save_best": train_cfg.get("save_best"),
        "no_tensorboard": not bool(logging_cfg.get("tensorboard", True)),
        "max_steps": train_cfg.get("max_steps"),
        "seed": train_cfg.get("seed"),
        "num_heads": model_cfg.get("num_heads"),
        "mlp_ratio": model_cfg.get("mlp_ratio"),
        "dropout": model_cfg.get("dropout"),
        "final_embedding_dim": model_cfg.get("final_embedding_dim"),
        "model_class": model_cfg.get("model_class"),
        "scheduler": sched_cfg.get("type"),
        "step_size": sched_cfg.get("step_size"),
        "gamma": sched_cfg.get("gamma"),
        "min_lr": sched_cfg.get("min_lr"),
        "depth": model_cfg.get("depth"),
        "inchannel": tuple(data_cfg.get("inchannel", (1, 1))),
        "outchannel": data_cfg.get("outchannel"),
        "stage1_checkpoint": stage1_ckpt,
        "stage2": bool(train_cfg.get("stage2")),
        "stage1_loss": loss_cfg.get("stage1_loss"),
        "stage2_loss": loss_cfg.get("stage2_loss"),
        "stage1_params": loss_cfg.get("stage1_params"),
        "stage2_params": loss_cfg.get("stage2_params"),
        "runs_dir": runs_dir,
        "run_name": logging_cfg.get("run_name"),
        "resume_from": resume_run,
        "eval_save_names": eval_save_names,
    }
    return flat


__all__ = [
    "DEFAULT_CONFIG",
    "apply_defaults",
    "flatten_config",
    "is_default_marker",
    "load_config_file",
    "resolve_path",
]
