import logging
import torch, open_clip
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def _default_ckpt_path(model_name: str) -> Path:
    root = Path(__file__).resolve().parent
    return root / f"RemoteCLIP-{model_name}.pt"

def build_remoteclip_image_encoder(model_name: str = "RN50", pretrained: bool = True,
                                   checkpoint_path: str | None = None):
    model = open_clip.create_model(model_name)
    if pretrained:
        ckpt_path = Path(checkpoint_path) if checkpoint_path else _default_ckpt_path(model_name)
        ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        model.load_state_dict(ckpt)
        logging.info(
            "open_clip.create_model may warn about missing pretrained weights; safe to ignore, weights loaded from %s",
            ckpt_path,
        )
    # model = model.to(device).eval()
    image_encoder = model.visual#.eval()
    return image_encoder

class RemoteClipResNet50(torch.nn.Module):
    def __init__(self, image_encoder: torch.nn.Module, normalize: bool = True):
        super().__init__()
        self.image_encoder = image_encoder
        self.normalize = normalize

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        # x = F.adaptive_avg_pool2d(x, (224, 224))
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std       
        intermediates = None
        if return_features:
            if not hasattr(self.image_encoder, "forward_intermediates"):
                raise AttributeError("image_encoder does not expose forward_intermediates for feature extraction.")
            outputs = self.image_encoder.forward_intermediates(
                x,
                indices=None,
                stop_early=False,
                normalize_intermediates=False,
                intermediates_only=False,
                output_fmt='NCHW',
                output_extra_tokens=False,
            )
            features = outputs["image_features"]
            names = ["stem", "layer1", "layer2", "layer3", "layer4"]
            intermediates = {name: feat for name, feat in zip(names, outputs.get("image_intermediates", []))}
        else:
            features = self.image_encoder(x)

        if self.normalize:
            features = torch.nn.functional.normalize(features, dim=-1)

        if return_features:
            return features, intermediates
        return features
    
def remoteclip_resnet50(pretrained: bool = True, checkpoint_path: str | None = None,
                        normalize: bool = True,
                        trainable: bool = False):
    '''返回已经归一化的RemoteCLIP ResNet50模型,输入要求3通道'''
    image_encoder = build_remoteclip_image_encoder(
        model_name="RN50",
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )  
    model = RemoteClipResNet50(image_encoder, normalize=normalize)
    for p in model.parameters():
        p.requires_grad_(trainable)  
    if not trainable:
        model.eval()
    
    return model


if __name__ == "__main__":
    model = remoteclip_resnet50(pretrained=True,trainable=True, normalize=True)
    example = torch.randn(1, 3, 224, 224)
    print(next(model.parameters()).device)
    print(example.device)
    with torch.no_grad():
        feats, feat_maps = model(example, return_features=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, trainable: {trainable:,}")
    for name, fmap in feat_maps.items():
        if fmap is not None:
            print(f"{name} feature map shape: {tuple(fmap.shape)}")
    print(f"Output feature shape: {tuple(feats.shape)}")
