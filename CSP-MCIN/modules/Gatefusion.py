import torch.nn as nn
import torch
class gatefusion_3(nn.Module):
    def __init__(self, c): 
        super().__init__()
        hidden = max(1, c // 2)
        self.ln = nn.LayerNorm(c)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(c, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, c, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def _ln(self, x):
        # NCHW -> NHWC -> LN -> NCHW
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)

    def forward(self, sar, opt, earlyfusion):
        gates = torch.stack([
            self.conv2(self.gelu(self.conv1(self._ln(sar)))),
            self.conv2(self.gelu(self.conv1(self._ln(opt)))),
            self.conv2(self.gelu(self.conv1(self._ln(earlyfusion))))
        ], dim=1)              # (B,3,C,H,W)
        gates = self.softmax(gates)  # 模态间归一化
        fused = gates[:,0]*sar + gates[:,1]*opt + gates[:,2]*earlyfusion
        return fused
    
class gatefusion_2(nn.Module):
    def __init__(self, c): 
        super().__init__()
        hidden = max(1, c // 2)
        self.ln = nn.LayerNorm(c)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(c, hidden, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden, c, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def _ln(self, x):
        # NCHW -> NHWC -> LN -> NCHW
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)

    def forward(self, sar, opt):
        gates = torch.stack([
            self.conv2(self.gelu(self.conv1(self._ln(sar)))),
            self.conv2(self.gelu(self.conv1(self._ln(opt))))
        ], dim=1)              # (B,2,C,H,W)
        gates = self.softmax(gates)  # 模态间归一化
        fused = gates[:,0]*sar + gates[:,1]*opt
        return fused
