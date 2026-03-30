import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def _corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    x = x - x.mean(dim=(2, 3), keepdim=True)
    y = y - y.mean(dim=(2, 3), keepdim=True)
    num = (x * y).sum(dim=(2, 3))
    denom = torch.sqrt(x.square().sum(dim=(2, 3)) * y.square().sum(dim=(2, 3)))
    return num / (denom + eps)


def CC(
    a: torch.Tensor,
    b: torch.Tensor,
    fused: torch.Tensor,
    eps: float = 1e-10,
    reduce: bool = True,
):
    corr_af = _corr(a, fused, eps=eps)
    corr_bf = _corr(b, fused, eps=eps)
    cc = 0.5 * (corr_af + corr_bf)
    if not reduce:
        return {"CC": cc, "CC1": corr_af, "CC2": corr_bf}
    return {"CC": cc.mean(dim=1), "CC1": corr_af.mean(dim=1), "CC2": corr_bf.mean(dim=1)}


def SCD(
    a: torch.Tensor,
    b: torch.Tensor,
    fused: torch.Tensor,
    eps: float = 1e-10,
    reduce: bool = True,
    split: bool = False,
):
    corr_fb_a = _corr(fused - b, a, eps=eps)
    corr_fa_b = _corr(fused - a, b, eps=eps)
    scd = corr_fb_a + corr_fa_b
    if not split:
        return scd.mean(dim=1) if reduce else scd
    if not reduce:
        return {"SCD": scd, "SCD1": corr_fb_a, "SCD2": corr_fa_b}
    return {"SCD": scd.mean(dim=1), "SCD1": corr_fb_a.mean(dim=1), "SCD2": corr_fa_b.mean(dim=1)}


def SSIM(a: torch.Tensor, b: torch.Tensor, fused: torch.Tensor, reduce: bool = True):
    data_range = 1.0
    batch, channels, height, width = a.shape

    fused_flat = fused.reshape(batch * channels, 1, height, width)
    a_flat = a.reshape(batch * channels, 1, height, width)
    b_flat = b.reshape(batch * channels, 1, height, width)

    ssim_af = ssim(fused_flat, a_flat, data_range=data_range, size_average=False).view(batch, channels)
    ssim_bf = ssim(fused_flat, b_flat, data_range=data_range, size_average=False).view(batch, channels)
    ssim_val = 0.5 * (ssim_af + ssim_bf)

    if not reduce:
        return {"SSIM": ssim_val, "SSIM1": ssim_af, "SSIM2": ssim_bf}
    return {"SSIM": ssim_val.mean(dim=1), "SSIM1": ssim_af.mean(dim=1), "SSIM2": ssim_bf.mean(dim=1)}


def MSE(a: torch.Tensor, b: torch.Tensor, fused: torch.Tensor, reduce: bool = True):
    mse_af = F.mse_loss(fused, a, reduction="none")
    mse_bf = F.mse_loss(fused, b, reduction="none")
    mse = 0.5 * (mse_af + mse_bf).mean(dim=(2, 3))
    if not reduce:
        return {"MSE": mse, "MSE1": mse_af.mean(dim=(2, 3)), "MSE2": mse_bf.mean(dim=(2, 3))}
    return {"MSE": mse.mean(dim=1), "MSE1": mse_af.mean(dim=(2, 3)).mean(dim=1), "MSE2": mse_bf.mean(dim=(2, 3)).mean(dim=1)}
