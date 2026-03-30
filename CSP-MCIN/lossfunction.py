from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from modules.RemoteClipResNet50 import remoteclip_resnet50
from modules.SARClipResNet50 import sarclip_resnet50
from utils import traditional_metrics


def build_clip_model_pair(pretrained=True, trainable=False, normalize=True) -> Dict[str, nn.Module]:
    """Build the RemoteCLIP/SARCLIP backbones once and reuse them across losses."""
    return {
        "remoteclip_model": remoteclip_resnet50(pretrained=pretrained, trainable=trainable, normalize=normalize),
        "sarclip_model": sarclip_resnet50(pretrained=pretrained, trainable=trainable, normalize=normalize),
    }


class _ClipLossBase(nn.Module):
    """Shared helpers for CLIP-based losses."""

    def __init__(
        self,
        clip_models: Dict[str, nn.Module] | None = None,
        remoteclip_model: nn.Module | None = None,
        sarclip_model: nn.Module | None = None,
    ):
        super().__init__()
        self.remoteclip_model, self.sarclip_model = self._init_clip_models(
            clip_models=clip_models,
            remoteclip_model=remoteclip_model,
            sarclip_model=sarclip_model,
        )

    @staticmethod
    def _init_clip_models(
        clip_models: Dict[str, nn.Module] | None,
        remoteclip_model: nn.Module | None,
        sarclip_model: nn.Module | None,
    ):
        if clip_models is not None:
            if remoteclip_model is not None or sarclip_model is not None:
                raise ValueError("Provide either clip_models or individual models, not both.")
            if not hasattr(clip_models, "get"):
                raise TypeError("clip_models must be a mapping containing the CLIP backbones.")
            remoteclip_model = clip_models.get("remoteclip_model") or clip_models.get("remote")
            sarclip_model = clip_models.get("sarclip_model") or clip_models.get("sar")
            if remoteclip_model is None or sarclip_model is None:
                raise ValueError("clip_models must contain 'remoteclip_model' and 'sarclip_model'.")
        elif remoteclip_model is None and sarclip_model is None:
            models = build_clip_model_pair(pretrained=True, trainable=False, normalize=True)
            remoteclip_model = models["remoteclip_model"]
            sarclip_model = models["sarclip_model"]
        else:
            if remoteclip_model is None or sarclip_model is None:
                raise ValueError("Both remoteclip_model and sarclip_model must be provided together.")
        return remoteclip_model, sarclip_model

    @staticmethod
    def soft_clamp01_softplus(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
        return x + (F.softplus(beta * (0.0 - x)) - F.softplus(beta * (x - 1.0))) / beta

    @staticmethod
    def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
        y, cb, cr = ycbcr[:, 0:1, ...], ycbcr[:, 1:2, ...], ycbcr[:, 2:3, ...]
        cb_shift = cb - 0.5
        cr_shift = cr - 0.5
        r = y + 1.402 * cr_shift
        g = y - 0.344136 * cb_shift - 0.714136 * cr_shift
        b = y + 1.772 * cb_shift
        return torch.cat([r, g, b], dim=1)

    def _prepare_output_variants(self, output: torch.Tensor, opt_ycbcr: torch.Tensor):
        output = self.soft_clamp01_softplus(output, beta=20.0)
        if output.shape[1] == 1:
            cb = opt_ycbcr[:, 1:2, ...]
            cr = opt_ycbcr[:, 2:3, ...]
            out_ycbcr = torch.cat([output, cb, cr], dim=1)
            output_optlike = self.soft_clamp01_softplus(self._ycbcr_to_rgb(out_ycbcr), beta=20.0)
            output_sarlike = output.repeat(1, 3, 1, 1)
        else:
            output_optlike = output
            output_sarlike = output[:, 0:1, ...].repeat(1, 3, 1, 1)
        return output, output_optlike, output_sarlike


class clip_infonce_loss(_ClipLossBase):
    def __init__(
        self,
        sar_weight=0.2,
        opt_weight=0.8,
        temperature=0.07,
        clip_models: Dict[str, nn.Module] | None = None,
        remoteclip_model: nn.Module | None = None,
        sarclip_model: nn.Module | None = None,
    ):
        super().__init__(
            clip_models=clip_models,
            remoteclip_model=remoteclip_model,
            sarclip_model=sarclip_model,
        )
        self.sar_weight = sar_weight
        self.opt_weight = opt_weight
        self.tau = temperature

    @staticmethod
    def _infonce(anchor: torch.Tensor, positive: torch.Tensor, tau: float) -> torch.Tensor:
        logits = (anchor @ positive.t()) / tau
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return F.cross_entropy(logits, labels)

    def forward(self, output: torch.Tensor, batch):
        output, output_optlike, output_sarlike = self._prepare_output_variants(output, batch["opt_ycbcr"])
        sar_3ch = batch["sar_3ch"]
        opt_rgb = batch["opt_rgb"]

        out_o = self.remoteclip_model(output_optlike, return_features=False)
        out_s = self.sarclip_model(output_sarlike, return_features=False)
        emb_o = self.remoteclip_model(opt_rgb, return_features=False)
        emb_s = self.sarclip_model(sar_3ch, return_features=False)

        loss_opt = self._infonce(out_o, emb_o, self.tau)
        loss_sar = self._infonce(out_s, emb_s, self.tau)
        total = self.opt_weight * loss_opt + self.sar_weight * loss_sar

        with torch.no_grad():
            pos_cos_opt = (out_o * emb_o).sum(dim=1).mean()
            pos_cos_sar = (out_s * emb_s).sum(dim=1).mean()
            logits_opt = out_o @ emb_o.t()
            logits_sar = out_s @ emb_s.t()
            top1_opt = (logits_opt.argmax(dim=1) == torch.arange(out_o.size(0), device=out_o.device)).float().mean()
            top1_sar = (logits_sar.argmax(dim=1) == torch.arange(out_s.size(0), device=out_s.device)).float().mean()

        return {
            "total": total,
            "infonce_opt": self.opt_weight * loss_opt,
            "infonce_sar": self.sar_weight * loss_sar,
            "poscos_opt": pos_cos_opt,
            "poscos_sar": pos_cos_sar,
            "top1_opt": top1_opt,
            "top1_sar": top1_sar,
        }


class clip_shallow_feature_loss(_ClipLossBase):
    def __init__(
        self,
        layer_weights: Dict[str, float] | None = None,
        sar_weight=0.5,
        opt_weight=0.5,
        clip_models: Dict[str, nn.Module] | None = None,
        remoteclip_model: nn.Module | None = None,
        sarclip_model: nn.Module | None = None,
    ):
        super().__init__(
            clip_models=clip_models,
            remoteclip_model=remoteclip_model,
            sarclip_model=sarclip_model,
        )
        self.layer_weights = self._sanitize_layer_weights(layer_weights)
        self.sar_weight = sar_weight
        self.opt_weight = opt_weight

    @staticmethod
    def _available_layers() -> tuple:
        return ("stem", "layer1", "layer2", "layer3", "layer4")

    def _sanitize_layer_weights(self, layer_weights: Dict[str, float] | None) -> Dict[str, float]:
        if layer_weights is None:
            layer_weights = {"layer1": 1.0}
        if isinstance(layer_weights, (list, tuple)):
            if all(isinstance(item, str) for item in layer_weights):
                layer_weights = {item: 1.0 for item in layer_weights}
            else:
                layer_weights = {name: weight for name, weight in layer_weights}
        if not isinstance(layer_weights, dict):
            raise TypeError("layer_weights must be a dict or a list of (layer, weight) pairs.")
        if not layer_weights:
            raise ValueError("layer_weights must be non-empty.")

        valid_layers = set(self._available_layers())
        sanitized: Dict[str, float] = {}
        for name, weight in layer_weights.items():
            if name not in valid_layers:
                raise ValueError(f"Unknown layer '{name}'. Available: {sorted(valid_layers)}")
            sanitized[name] = float(weight)
        return sanitized

    def _weighted_layer_mse(self, out_feats: Dict[str, torch.Tensor], tgt_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for layer, weight in self.layer_weights.items():
            out_feat = out_feats.get(layer)
            tgt_feat = tgt_feats.get(layer)
            if out_feat is None or tgt_feat is None:
                raise KeyError(f"Missing feature map for layer '{layer}'.")
            total = total + weight * F.mse_loss(out_feat, tgt_feat)
        return total

    def forward(self, output: torch.Tensor, batch):
        output, output_optlike, output_sarlike = self._prepare_output_variants(output, batch["opt_ycbcr"])
        sar_3ch = batch["sar_3ch"]
        opt_rgb = batch["opt_rgb"]

        _, out_opt_feats = self.remoteclip_model(output_optlike, return_features=True)
        _, out_sar_feats = self.sarclip_model(output_sarlike, return_features=True)

        with torch.no_grad():
            _, opt_feats = self.remoteclip_model(opt_rgb, return_features=True)
            _, sar_feats = self.sarclip_model(sar_3ch, return_features=True)

        loss_opt = self._weighted_layer_mse(out_opt_feats, opt_feats)
        loss_sar = self._weighted_layer_mse(out_sar_feats, sar_feats)
        total = self.opt_weight * loss_opt + self.sar_weight * loss_sar
        return {
            "total": total,
            "shallow_opt": self.opt_weight * loss_opt,
            "shallow_sar": self.sar_weight * loss_sar,
        }


class pix_signif_loss(nn.Module):
    def __init__(self, scd_weight=0.5, cc_weight=0.5, sar_weight=0.5, opt_weight=0.5):
        super().__init__()
        self.scd_weight = scd_weight
        self.cc_weight = cc_weight
        self.sar_weight = sar_weight
        self.opt_weight = opt_weight

    @staticmethod
    def soft_clamp01_softplus(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
        return x + (F.softplus(beta * (0.0 - x)) - F.softplus(beta * (x - 1.0))) / beta

    def forward(self, output: torch.Tensor, batch) -> torch.Tensor:
        output = self.soft_clamp01_softplus(output, 20)
        sar_scd_loss = 2 - traditional_metrics.SCD(batch["opt"], batch["sar"], output, split=True)["SCD2"].mean()
        sar_cc_loss = 1 - traditional_metrics.CC(batch["opt"], batch["sar"], output)["CC2"].mean()
        opt_scd_loss = 2 - traditional_metrics.SCD(batch["opt"], batch["sar"], output, split=True)["SCD1"].mean()
        opt_cc_loss = 1 - traditional_metrics.CC(batch["opt"], batch["sar"], output)["CC1"].mean()

        sar_pix_signif_loss = self.scd_weight * sar_scd_loss + self.cc_weight * sar_cc_loss
        opt_pix_signif_loss = self.scd_weight * opt_scd_loss + self.cc_weight * opt_cc_loss
        total_loss = self.sar_weight * sar_pix_signif_loss + self.opt_weight * opt_pix_signif_loss
        return {
            "total": total_loss,
            "sar_pix_signif_loss": self.sar_weight * sar_pix_signif_loss,
            "opt_pix_signif_loss": self.opt_weight * opt_pix_signif_loss,
        }


class pix_semantic_joint_loss_shallow_feature(nn.Module):
    """Joint pixel-significance and CLIP semantic loss used by the WOS config."""

    def __init__(
        self,
        pix_weight=0.7,
        semantic_weight=0.3,
        sar_weight=0.3,
        opt_weight=0.7,
        scd_weight=0.6,
        cc_weight=0.4,
        temperature=0.07,
        semantic_shallow_weight=0.5,
        semantic_infonce_weight=0.5,
        layer_weights: Dict[str, float] | None = None,
        clip_models: Dict[str, nn.Module] | None = None,
        remoteclip_model: nn.Module | None = None,
        sarclip_model: nn.Module | None = None,
    ):
        super().__init__()
        models = clip_models
        if models is None and (remoteclip_model is not None or sarclip_model is not None):
            if remoteclip_model is None or sarclip_model is None:
                raise ValueError("Both remoteclip_model and sarclip_model must be provided together.")
            models = {"remoteclip_model": remoteclip_model, "sarclip_model": sarclip_model}
        if models is None:
            models = build_clip_model_pair(pretrained=True, trainable=False, normalize=True)

        self.clip_infonce_loss = clip_infonce_loss(
            sar_weight=sar_weight,
            opt_weight=opt_weight,
            temperature=temperature,
            clip_models=models,
        )
        self.clip_shallow_feature_loss = clip_shallow_feature_loss(
            layer_weights=layer_weights,
            sar_weight=sar_weight,
            opt_weight=opt_weight,
            clip_models=models,
        )
        self.pix_signif_loss = pix_signif_loss(
            scd_weight=scd_weight,
            cc_weight=cc_weight,
            sar_weight=sar_weight,
            opt_weight=opt_weight,
        )
        self.pix_weight = pix_weight
        self.semantic_weight = semantic_weight
        self.semantic_shallow_weight = semantic_shallow_weight
        self.semantic_infonce_weight = semantic_infonce_weight

    def forward(self, output: torch.Tensor, batch) -> torch.Tensor:
        clip_infonce_dict = self.clip_infonce_loss(output, batch)
        clip_shallow_dict = self.clip_shallow_feature_loss(output, batch)
        pix_signif_dict = self.pix_signif_loss(output, batch)

        sar_pix_signif_loss = pix_signif_dict["sar_pix_signif_loss"]
        opt_pix_signif_loss = pix_signif_dict["opt_pix_signif_loss"]
        sar_semantic_shallow_loss = clip_shallow_dict["shallow_sar"]
        opt_semantic_shallow_loss = clip_shallow_dict["shallow_opt"]
        sar_semantic_infonce_loss = clip_infonce_dict["infonce_sar"]
        opt_semantic_infonce_loss = clip_infonce_dict["infonce_opt"]

        sar_semantic_loss = (
            self.semantic_shallow_weight * sar_semantic_shallow_loss
            + self.semantic_infonce_weight * sar_semantic_infonce_loss
        )
        opt_semantic_loss = (
            self.semantic_shallow_weight * opt_semantic_shallow_loss
            + self.semantic_infonce_weight * opt_semantic_infonce_loss
        )

        pix_total_loss = sar_pix_signif_loss + opt_pix_signif_loss
        semantic_total_loss = sar_semantic_loss + opt_semantic_loss
        sar_total_loss = self.pix_weight * sar_pix_signif_loss + self.semantic_weight * sar_semantic_loss
        opt_total_loss = self.pix_weight * opt_pix_signif_loss + self.semantic_weight * opt_semantic_loss
        total_loss = sar_total_loss + opt_total_loss
        return {
            "total": total_loss,
            "sar_total_loss": sar_total_loss,
            "opt_total_loss": opt_total_loss,
            "pix_total_loss": pix_total_loss,
            "semantic_total_loss": semantic_total_loss,
            "sar_pix_signif_loss": sar_pix_signif_loss,
            "opt_pix_signif_loss": opt_pix_signif_loss,
            "sar_semantic_loss": sar_semantic_loss,
            "opt_semantic_loss": opt_semantic_loss,
        }
