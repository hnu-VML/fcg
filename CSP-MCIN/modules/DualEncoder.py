import torch.nn as nn
import torchvision.models as models


class DualEncoderResNet(nn.Module):
    """Dual-branch ResNet-18 encoder used by the public WOS configuration."""

    def __init__(self, depth: int = 18, final_embedding_dim: int | None = None, inchannel: tuple = (1, 3)):
        super().__init__()
        if depth != 18:
            raise ValueError("The public release only keeps the ResNet-18 encoder used by configs/WOS.yaml.")

        self.depth = depth
        self.final_embedding_dim = final_embedding_dim
        self.sar_branch = models.resnet18(weights=None)
        self.opt_branch = models.resnet18(weights=None)

        if inchannel[0] == 1:
            self._convert_branch_stem_to_single_channel(self.sar_branch)
        if inchannel[1] == 1:
            self._convert_branch_stem_to_single_channel(self.opt_branch)

    @staticmethod
    def _convert_branch_stem_to_single_channel(branch: nn.Module) -> None:
        conv1 = branch.conv1
        branch.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False,
        )

    def forward(self, sar, opt, return_features=True):
        sar_final_emb, sar_feats = self._forward_branch(self.sar_branch, sar, return_features)
        opt_final_emb, opt_feats = self._forward_branch(self.opt_branch, opt, return_features)
        if return_features:
            return sar_final_emb, opt_final_emb, sar_feats, opt_feats
        return sar_final_emb, opt_final_emb

    @staticmethod
    def _forward_branch(branch: nn.Module, x, return_features: bool):
        feats = {}

        x = branch.conv1(x)
        x = branch.bn1(x)
        x = branch.relu(x)
        if return_features:
            feats["conv1"] = x

        x = branch.maxpool(x)
        x = branch.layer1(x)
        if return_features:
            feats["layer1"] = x

        x = branch.layer2(x)
        if return_features:
            feats["layer2"] = x

        x = branch.layer3(x)
        if return_features:
            feats["layer3"] = x

        x = branch.layer4(x)
        if return_features:
            feats["layer4"] = x

        return x, feats if return_features else None
