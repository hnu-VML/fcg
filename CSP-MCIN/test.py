#!/usr/bin/env python3
"""
Inference script for FusionNet.

The script reuses the dataset utilities from ``data.py`` to iterate over paired
SAR/optical images, loads a checkpoint, and writes the fused outputs to
``<checkpoint>.pt_testresult`` next to the checkpoint file. Whenever possible it
pulls hyper-parameters from the checkpoint, so you only need to override values
that should differ at test time.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

import matplotlib

from data import PairedImageDataset
import model as model_module
LOGGER = logging.getLogger("test")

try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tifffile = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FusionNet inference on paired SAR/optical images.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory that contains sar/ and opt/ folders.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pt checkpoint.")
    parser.add_argument("--image-size", type=int, default=None, help="Square resolution to resize images to.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers.")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Force-enable pin_memory.")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Force-disable pin_memory.")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on.")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save selected intermediate feature maps for each sample (FusionNet_2gate only).",
    )
    parser.add_argument(
        "--intermediate-names",
        type=str,
        default="auto",
        help=(
            "Comma-separated intermediate feature names to save when --save-intermediates is enabled. "
            "Use 'auto' to save all returned features."
        ),
    )
    parser.add_argument(
        "--save-rgb",
        action="store_true",
        help="Save RGB fused images by combining fused Y with optical CbCr (enables YCbCr->RGB conversion).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for fused images (defaults to <checkpoint>.pt_testresult).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_loader(data_dir: Path, runtime_cfg: Dict[str, Any]) -> DataLoader:
    dataset = PairedImageDataset(
        sar_dir=data_dir / "sar",
        opt_dir=data_dir / "opt",
        image_size=runtime_cfg["image_size"],
        inchannel=runtime_cfg["inchannel"],
    )
    LOGGER.info("Loaded %d samples from %s", len(dataset), data_dir)
    return DataLoader(
        dataset,
        batch_size=runtime_cfg["batch_size"],
        shuffle=False,
        num_workers=runtime_cfg["num_workers"],
        pin_memory=runtime_cfg["pin_memory"],
        drop_last=False,
    )


def resolve_model_kwargs(
    runtime_cfg: Dict[str, Any],
) -> Dict[str, object]:
    return {
        "num_heads": runtime_cfg["num_heads"],
        "mlp_ratio": runtime_cfg["mlp_ratio"],
        "dropout": runtime_cfg["dropout"],
        "final_embedding_dim": runtime_cfg["final_embedding_dim"],
        "depth": runtime_cfg["depth"],
        "inchannel": runtime_cfg["inchannel"],
        "outchannel": runtime_cfg["outchannel"],
    }


def _get_class_by_name(module, name: str, category: str):
    """Resolve a class from a module by name (case-insensitive)."""
    candidates = {attr.lower(): attr for attr in dir(module) if not attr.startswith("_")}
    lookup_key = name.lower()
    if lookup_key not in candidates:
        raise ValueError(f"Unknown {category} class '{name}' in {module.__name__}.")
    attr_name = candidates[lookup_key]
    cls = getattr(module, attr_name)
    if not isinstance(cls, type):
        raise ValueError(f"{category} '{name}' in {module.__name__} is not a class.")
    return cls


def load_checkpoint(path: Path) -> Tuple[Path, Dict[str, Any]]:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_path, checkpoint


def load_model(
    checkpoint: Dict[str, Any],
    model_kwargs: Dict[str, object],
    model_class: str,
    device: torch.device,
):
    model_cls = _get_class_by_name(model_module, model_class, "model")
    model = model_cls(**model_kwargs)
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise KeyError("Checkpoint is missing the 'model' state_dict.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _parse_intermediate_names(raw_names: str) -> Tuple[str, ...]:
    raw = raw_names.strip()
    if raw.lower() in {"auto", "all", "*"}:
        return ()
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    if not names:
        raise ValueError("No valid feature name found in --intermediate-names.")
    return names


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert YCbCr tensor in [0,1] to RGB for fused image export."""
    y, cb, cr = ycbcr[:, 0:1, ...], ycbcr[:, 1:2, ...], ycbcr[:, 2:3, ...]
    cb_shift = cb - 0.5
    cr_shift = cr - 0.5
    r = y + 1.402 * cr_shift
    g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
    b = y + 1.772 * cb_shift
    return torch.cat([r, g, b], dim=1)


def _feature_to_visual_map(feature: torch.Tensor) -> torch.Tensor:
    # Reduce CxHxW features to 1xHxW for visualization.
    if feature.ndim == 2:
        return feature.unsqueeze(0)
    if feature.ndim != 3:
        raise ValueError(f"Expected 2D or 3D feature tensor, got shape={tuple(feature.shape)}")
    if feature.shape[0] == 1:
        return feature
    return feature.mean(dim=0, keepdim=True)


def _visual_map_to_heatmap(vis_map: torch.Tensor, cmap_name: str = "magma") -> torch.Tensor:
    if vis_map.ndim == 3 and vis_map.shape[0] == 1:
        vis_map = vis_map[0]
    if vis_map.ndim != 2:
        raise ValueError(f"Expected 2D visualization map, got shape={tuple(vis_map.shape)}")
    data = vis_map.detach().cpu().float().numpy()
    min_val = float(data.min())
    max_val = float(data.max())
    if max_val - min_val < 1e-6:
        norm = np.zeros_like(data, dtype=np.float32)
    else:
        norm = (data - min_val) / (max_val - min_val)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = cmap(norm)[..., :3].astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _save_feature_tiff(feature: torch.Tensor, path: Path) -> None:
    array = feature.detach().cpu().float().numpy()
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected feature tensor [C,H,W] or [H,W], got shape={tuple(array.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if tifffile is not None:
        tifffile.imwrite(str(path), array.astype(np.float32))
        return
    # Fallback: save as multipage TIFF (one channel per page).
    from PIL import Image

    pages = [Image.fromarray(array[c].astype(np.float32), mode="F") for c in range(array.shape[0])]
    if not pages:
        return
    pages[0].save(str(path), save_all=True, append_images=pages[1:])


def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_intermediates: bool = False,
    intermediate_names: Tuple[str, ...] = (),
    save_rgb: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving fused outputs to %s", output_dir)
    feature_root = output_dir / "intermediates"
    if save_intermediates:
        feature_root.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving intermediate feature maps to %s", feature_root)
    resolved_names: Optional[Tuple[str, ...]] = intermediate_names or None
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=100, dynamic_ncols=True):
            sar = batch["sar"].to(device, non_blocking=True)
            opt = batch["opt"].to(device, non_blocking=True)
            opt_ycbcr = batch["opt_ycbcr"].to(device, non_blocking=True) if save_rgb else None
            names = batch["name"]

            feature_dict = None
            if save_intermediates:
                try:
                    result = model(sar, opt, return_features=True)
                except TypeError as exc:
                    raise RuntimeError(
                        "--save-intermediates requires model forward(..., return_features=True). "
                        "Current model class may not support this argument."
                    ) from exc
                if not (isinstance(result, tuple) and len(result) == 2):
                    raise RuntimeError("Model did not return (output, feature_dict) while --save-intermediates is enabled.")
                output, feature_dict = result
                if resolved_names is None:
                    resolved_names = tuple(feature_dict.keys())
                    LOGGER.info("Auto intermediate names: %s", resolved_names)
                missing = [name for name in resolved_names if name not in feature_dict]
                if missing:
                    raise KeyError(
                        f"Missing requested intermediate features: {missing}. "
                        f"Available: {list(feature_dict.keys())}"
                    )
            else:
                output = model(sar, opt)
            output_to_save = output
            if save_rgb:
                if output.shape[1] == 1:
                    if opt_ycbcr is None:
                        raise RuntimeError("opt_ycbcr not found in batch while --save-rgb is enabled.")
                    fused_ycbcr = torch.cat([output, opt_ycbcr[:, 1:2, ...], opt_ycbcr[:, 2:3, ...]], dim=1)
                    output_to_save = _ycbcr_to_rgb(fused_ycbcr)
                elif output.shape[1] == 3:
                    output_to_save = output
                else:
                    raise ValueError(f"Unsupported output channel count for RGB save: {output.shape[1]}")
            for idx, name in enumerate(names):
                save_path = output_dir / f"{name}.png"
                save_image(output_to_save[idx].detach().cpu(), save_path)
                if feature_dict is not None:
                    sample_dir = feature_root / name
                    vis_dir = sample_dir / "visualization"
                    ori_dir = sample_dir / "original"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    ori_dir.mkdir(parents=True, exist_ok=True)
                    for feature_name in resolved_names or ():
                        feature = feature_dict[feature_name][idx].detach().cpu()
                        vis_map = _feature_to_visual_map(feature)
                        heatmap = _visual_map_to_heatmap(vis_map)
                        save_image(heatmap, vis_dir / f"{feature_name}.png")
                        _save_feature_tiff(feature, ori_dir / f"{feature_name}.tiff")


def build_runtime_config(args: argparse.Namespace, checkpoint_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = checkpoint_config or {}

    def pick(cli_value, *config_keys, default):
        if cli_value is not None:
            return cli_value
        for key in config_keys:
            if key in config and config[key] is not None:
                return config[key]
        return default

    runtime = {
        "image_size": pick(args.image_size, "image_size", default=224),
        "batch_size": pick(args.batch_size, "val_batch_size", "train_batch_size", default=4),
        "num_workers": pick(args.num_workers, "num_workers", default=4),
        "pin_memory": pick(args.pin_memory, "pin_memory", default=False),
        "num_heads": pick(None, "num_heads", default=1),
        "mlp_ratio": pick(None, "mlp_ratio", default=1.0),
        "dropout": pick(None, "dropout", default=0.1),
        "final_embedding_dim": pick(None, "final_embedding_dim", default=1024),
        "depth": pick(None, "depth", default=18),
        "inchannel": pick(None, "inchannel", default=(1, 1)),
        "outchannel": pick(None, "outchannel", default=1),
        "device": pick(args.device, "device", default=None),
        "model_class": pick(None, "model_class", default="FusionNet_2gate"),
    }

    inchannel = runtime["inchannel"]
    if isinstance(inchannel, list):
        inchannel = tuple(inchannel)
    elif isinstance(inchannel, tuple):
        inchannel = tuple(inchannel)
    else:
        inchannel = tuple(int(c) for c in inchannel)
    runtime["inchannel"] = inchannel
    return runtime


def main() -> None:
    args = parse_args()
    configure_logging()
    checkpoint_path, checkpoint = load_checkpoint(args.checkpoint)
    runtime_cfg = build_runtime_config(args, checkpoint.get("config"))
    device_str = runtime_cfg["device"]
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Effective runtime config: %s", runtime_cfg)

    loader = build_loader(args.data_dir, runtime_cfg)
    model_kwargs = resolve_model_kwargs(runtime_cfg)
    model = load_model(checkpoint, model_kwargs, runtime_cfg["model_class"], device)
    LOGGER.info("Model class: %s | kwargs: %s", runtime_cfg["model_class"], model_kwargs)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else checkpoint_path.with_name(
        f"{checkpoint_path.name}_testresult"
    )
    intermediate_names: Tuple[str, ...] = ()
    if args.save_intermediates:
        intermediate_names = _parse_intermediate_names(args.intermediate_names)
    run_inference(
        model,
        loader,
        device,
        output_dir,
        save_intermediates=args.save_intermediates,
        intermediate_names=intermediate_names,
        save_rgb=args.save_rgb,
    )
    LOGGER.info("Inference complete.")


if __name__ == "__main__":
    main()
