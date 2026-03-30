import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, query, context):
        q = self._split_heads(self.query_proj(query))
        k = self._split_heads(self.key_proj(context))
        v = self._split_heads(self.value_proj(context))

        scale = self.head_dim ** 0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = self._combine_heads(context)
        return self.output_proj(context)
class Self_TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, input):
        # Cross-Attention
        query = input
        input = self.norm1(input)
        query = query + self.attn(input, input) #
        # Feed-Forward Network
        query = query + self.mlp(self.norm2(query))
        return query
class Cross_TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        # Cross-Attention
        query = query + self.attn(self.norm1(query), self.norm2(context)) #
        # Feed-Forward Network
        query = query + self.mlp(self.norm3(query))
        return query
class DualTransformerBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, mlp_ratio=1.0, dropout=0.1):
        super().__init__()
        self.cross_transformer_sar = Cross_TransformerBlock(feature_dim, num_heads, mlp_ratio, dropout)
        self.cross_transformer_opt = Cross_TransformerBlock(feature_dim, num_heads, mlp_ratio, dropout)
        self.self_transformer_opt = Self_TransformerBlock(feature_dim, num_heads, mlp_ratio, dropout)
        self.self_transformer_sar = Self_TransformerBlock(feature_dim, num_heads, mlp_ratio, dropout)
    def forward(self, sar_feature, opt_feature):
        b, c, h, w = sar_feature.shape
        sar_feature = sar_feature.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)
        opt_feature = opt_feature.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)

        sar_feature = self.self_transformer_sar(sar_feature)
        opt_feature = self.self_transformer_opt(opt_feature)

        sar_out = self.cross_transformer_sar(sar_feature, opt_feature)
        opt_out = self.cross_transformer_opt(opt_feature, sar_feature)

        sar_out = sar_out.transpose(1, 2).reshape(b, c, h, w) # (B, H*W, C) -> (B, C, H, W)
        opt_out = opt_out.transpose(1, 2).reshape(b, c, h, w)  # (B, H*W, C) -> (B, C, H, W)
        return sar_out, opt_out
    
