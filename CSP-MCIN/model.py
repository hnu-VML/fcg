from modules.DualEncoder import DualEncoderResNet
from modules.DualTransformerBlock import DualTransformerBlock
from modules.Gatefusion import gatefusion_2, gatefusion_3
from torch import nn
import torch


class FusionNet_2gate(nn.Module):
    """Fusion network kept for the public WOS configuration."""

    def __init__(
        self,
        num_heads=1,
        mlp_ratio=1.0,
        dropout=0.1,
        final_embedding_dim=1024,
        depth=18,
        inchannel=(1, 3),
        outchannel=3,
    ):
        super().__init__()
        if depth != 18:
            raise ValueError("The public release only keeps the ResNet-18 FusionNet_2gate variant.")

        transformer_block_cfg = [64, 128, 256, 512]
        upsample_cfg = [(64, outchannel, 2), (128, 64, 2), (256, 64, 2), (512, 128, 2), (512, 256, 2)]
        self.dual_encoder = DualEncoderResNet(final_embedding_dim=final_embedding_dim, depth=depth, inchannel=inchannel)
        self.transformer_blocks = nn.ModuleDict(
            {
                f"transformer_block_{idx}": DualTransformerBlock(inc, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for idx, inc in enumerate(transformer_block_cfg, start=1)
            }
        )
        self.upsamples = nn.ModuleDict(
            {
                f"upsample_{idx}": nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                    nn.Conv2d(inc, outc, kernel_size=5, padding=2, padding_mode="reflect"),
                )
                for idx, (inc, outc, scale) in enumerate(upsample_cfg, start=1)
            }
        )
        self.gatefusion_2_blocks = nn.ModuleDict(
            {f"gatefusion_2_block_{idx}": gatefusion_2(c=inc) for idx, inc in enumerate(transformer_block_cfg, start=1)}
        )
        self.gatefusion_3_block_1 = gatefusion_3(c=upsample_cfg[1][1])
        self.gatefusion_3_block_2 = gatefusion_3(c=upsample_cfg[0][1])

    def forward(self, sar, opt, return_features: bool = False):
        _, _, sar_feats, opt_feats = self.dual_encoder(sar, opt, return_features=True)
        sar_feat1, opt_feat1 = self.transformer_blocks["transformer_block_1"](sar_feats["layer1"], opt_feats["layer1"])
        sar_feat2, opt_feat2 = self.transformer_blocks["transformer_block_2"](sar_feats["layer2"], opt_feats["layer2"])
        sar_feat3, opt_feat3 = self.transformer_blocks["transformer_block_3"](sar_feats["layer3"], opt_feats["layer3"])
        sar_feat4, opt_feat4 = self.transformer_blocks["transformer_block_4"](sar_feats["layer4"], opt_feats["layer4"])

        fusionfeat1 = self.gatefusion_2_blocks["gatefusion_2_block_1"](sar_feat1, opt_feat1)
        fusionfeat2 = self.gatefusion_2_blocks["gatefusion_2_block_2"](sar_feat2, opt_feat2)
        fusionfeat3 = self.gatefusion_2_blocks["gatefusion_2_block_3"](sar_feat3, opt_feat3)
        fusionfeat4 = self.gatefusion_2_blocks["gatefusion_2_block_4"](sar_feat4, opt_feat4)

        mainflow = torch.concat([self.upsamples["upsample_5"](fusionfeat4), fusionfeat3], dim=1)
        mainflow = torch.concat([self.upsamples["upsample_4"](mainflow), fusionfeat2], dim=1)
        mainflow = torch.concat([self.upsamples["upsample_3"](mainflow), fusionfeat1], dim=1)
        earlyfusion = self.upsamples["upsample_2"](mainflow)
        gatefused1 = self.gatefusion_3_block_1(sar_feats["conv1"], opt_feats["conv1"], earlyfusion)
        gatefused1 = self.upsamples["upsample_1"](gatefused1)
        gatefused2 = self.gatefusion_3_block_2(sar, opt, gatefused1)

        if return_features:
            intermediates = {
                "sar_feat_ori": sar_feats["layer1"],
                "opt_feat_ori": opt_feats["layer1"],
                "sar_feat_after_tf": sar_feat1,
                "opt_feat_after_tf": opt_feat1,
                "feat_after_gfu": fusionfeat1,
            }
            return gatefused2, intermediates
        return gatefused2
