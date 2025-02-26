import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange
from typing import Optional

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        # [b, h, w, c]
        x = self.norm(x)  # norm_layer默认最后一层为channel
        return self.fn(x, *args, **kwargs)

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,  # 总维度
            dim_head,  # 每一个头部的维度
            heads,  # 头部数量
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)  # 改变通道数量
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))  # 每一个头部注意力图的尺度系数不同，同一个头部注意力图的尺度系数相同
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        # print(x_in.shape)
        x = x_in.reshape(b,h*w,c)  # [b, hw, c]
        q_inp = self.to_q(x)  # [b, hw, heads*c]
        k_inp = self.to_k(x)  # [b, hw, heads*c]
        v_inp = self.to_v(x)  # [b, hw, heads*c]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))  # [b, heads, hw, c]
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)  # [b, heads, c, hw]
        k = k.transpose(-2, -1)  # [b, heads, c, hw]
        v = v.transpose(-2, -1)  # [b, heads, c, hw]
        q = F.normalize(q, dim=-1, p=2)  # [b, heads, c, hw]
        k = F.normalize(k, dim=-1, p=2)  # [b, heads, c, hw]
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q  [b, heads, c, c]
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # [b, hw, heads*c]
        out_c = self.proj(x).view(b, h, w, c)  # [b, h, w, c]
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2)) # [b, h, w, c] -> [b, c, h, w](卷积认为第二个通道为channel维度)
        return out.permute(0, 2, 3, 1) # [b, c, h, w] -> [b, h, w, c]

class MSAB(nn.Module):  # 不改变输入大小与通道数量
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            downsample=True
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
        if downsample:
            self.downsample = nn.Conv2d(dim//2, dim, 4, 2, 1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # patch merging layer
        if self.downsample is not None:
            x = self.downsample(x)  # downsample is PatchMerging layer

        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x  # [b, h, w, c]
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)  # [b, c, h, w]

        return out

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        # 定义相对位置偏置表格，里面的参数需要是通过学习得出的
        # table每一列的参数数量为(2m-1)*(2m-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # 返回tensor(0,1,2,...,windowsize[0]-1)
        coords_w = torch.arange(self.window_size[1])
        # meshgrid用于生成网格，返回两个张量：每个网格的行标与列标，使用stack()将两个张量拼接为一个张量
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw] 绝对位置索引
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw] 广播机制求相对位置索引
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 行列索引相加
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]  sum(-1)的axis = -1，为最后一个轴
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性映射
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # （输入通道数，输出通道数）
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        # print(x.shape)
        B_, N, C = x.shape

        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None: # SW-MSA
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else: # W-MSA
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # 通过线性层，对多个头部的输出进行一个融合
        x = self.proj_drop(x)
        return x  # attention模块的输出

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):

        B, H, W, C = x.shape  # x: [B, H, W, C]

        x = x.view(B, H * W, C)  # reshape

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)  # reshape

        # pad feature maps to multiples(倍数) of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # 0, 0表示倒数第一个维度；pad_l, pad_r表示倒数第二个维度；pad_t, pad_b表示倒数第三个维度
        _, Hp, Wp, _ = x.shape  # 获得pad后图的高与宽

        # cyclic shift
        if self.shift_size > 0:  # SW-MSA
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))  # 从上往下移、从左往右移
        else:
            shifted_x = x  # MSA
            attn_mask = None  # 未移动window里不需要掩盖矩阵，每一个窗口单独计算attention

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # shape操作  [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)  # reshape

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.view(B, H, W, C)

        return x  # [B, H, W, C]

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=True, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 整数除法，计算窗口的移动行数与列数

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,  ## SW
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # self.first_conv2D = Conv3x3(self.dim, self.dim, 3, 1)
        # self.prelu2D = nn.PReLU()
        # self.second_conv2D = Conv3x3(self.dim, self.dim, 3, 1)
        #
        # self.one_conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, bias=False)

        # patch merging layer
        if downsample:
            self.downsample = nn.Conv2d(dim//2, dim, 4, 2, 1, bias=False)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size  # H向上取整
        Wp = int(np.ceil(W / self.window_size)) * self.window_size  # W向上取整
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        # 切片
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        # 遍历所有切片，索引赋值
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  #
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]  划分窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]  view的作用类似reshape，-1在这里的意思是让电脑帮我们计算
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]  同一区域的元素为0，不同区域的非0
        # [nW, Mh*Mw, Mh*Mw] 广播机制
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)  # downsample is PatchMerging layer

        B, C, H, W = x.shape  # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # 经过block的输出
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        # x = x.transpose(1, 2).view(B, self.dim, H, W)  # [B, C, H, W]
        # # x = self.second_conv2D(self.prelu2D(self.first_conv2D(x)))
        # # x = self.first_conv2D(x)
        # x = self.one_conv(x)
        # x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        # x = x + y

        return x  # ([B, H, W, C])

class MST(nn.Module):
    def __init__(self, n_feat=31, window_size=7, depths=[2, 2, 2], swin_num_heads=[2, 4, 8], mlp_ratio=4., qkv_bias=True,
                 dim=62, stage=3, num_blocks=[2, 2, 2], drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(MST, self).__init__()
        self.stage = stage
        self.shift_size = window_size // 2
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Encoder(heads [1,2,4])
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim  # dim表示自注意力机制的输入维数
        for i in range(stage):  # 0，1, 2
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=n_feat, heads=dim_stage // n_feat),  # dim_head为一个头部的维数；heads为头部数
                BasicLayer(dim=dim_stage,
                           depth=depths[i],
                           num_heads=swin_num_heads[i],
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias,
                           drop=drop_rate,
                           attn_drop=attn_drop_rate,
                           drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                           norm_layer=norm_layer,
                           use_checkpoint=use_checkpoint),
                nn.Conv2d(dim_stage*2, dim_stage, 1, 1, bias=False)]))
            if i < stage-1:
                dim_stage *= 2

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage-1):  # 0, 1
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),  # 1*1卷积
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 2 - i], dim_head=n_feat,
                    heads=(dim_stage // 2) // n_feat, downsample=False),
                BasicLayer(dim=dim_stage // 2,
                           depth=depths[(self.stage - 2 - i)],
                           num_heads=swin_num_heads[(self.stage - 2 - i)],
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias,
                           drop=drop_rate,
                           attn_drop=attn_drop_rate,
                           drop_path=dpr[sum(depths[:(self.stage - 2 - i)]):sum(
                               depths[:(self.stage - 2 - i) + 1])],
                           norm_layer=norm_layer,
                           downsample=False,
                           use_checkpoint=use_checkpoint)
            ]))
            dim_stage //= 2

        self.up = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y): # x:MSI y:HSI
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Encoder
        fea_encoder = []
        for i, (MSAB, Swin, Fution_encode) in enumerate(self.encoder_layers):
            y = MSAB(y)  # [B, C, H, W]
            x = Swin(x)  # [B, C, H, W]
            # print(y.shape,x.shape)
            y = Fution_encode(torch.cat([y,x], dim=1))
            if i < self.stage-1:
                fea_encoder.append(y)

        # Decoder
        for i, (FeaUpSample, Fution_decode, Fution, MSAB, Swin) in enumerate(self.decoder_layers):
            y = FeaUpSample(y)  # 转置卷积2倍上采样
            y = Fution_decode(torch.cat([y, fea_encoder[self.stage-2-i]], dim=1))  # 1*1卷积层使得通道数减半
            y = Fution(torch.cat([MSAB(y),Swin(y)],dim=1)) # 通道注意力与空间注意力特征融合
            # print(x.shape)

        # final upsample
        out = self.up(y)
        # print(out.shape)

        return out

class SSTF_Unet(nn.Module):
    def __init__(self, in_channels_MSI=3, out_channels=31, n_feat=31):
        super(SSTF_Unet, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=16, mode='bicubic', align_corners=True)
        self.conv_in_MSI = nn.Conv2d(in_channels_MSI, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)

        self.body = MST(n_feat=n_feat, dim=62, stage=3, num_blocks=[1,1,1],window_size=7, depths=[2, 2, 2],
                        swin_num_heads=[2, 4, 8], mlp_ratio=4., qkv_bias=True)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x, y): # x:MSI y:HSI
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        y = self.up_sample(y)

        x = self.conv_in_MSI(x)   # 扩展维度

        h = self.body(x, y)

        h = self.conv_out(h)
        h = y + h
        return h
