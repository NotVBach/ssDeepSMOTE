# https://github.com/facebookresearch/mae
# https://github.com/google-research/vision_transformer
# https://github.com/huggingface/pytorch-image-models

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1-D sine-cosine embedding of the given positions (MAE recipe)
def sincos_1d(dim, positions):
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / 10000 ** omega
    angles = np.einsum("m,d->md", positions.reshape(-1).astype(np.float64), omega)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


# Fixed 2-D sine-cosine positional embedding for a grid_size x grid_size token grid, shape (grid_size ** 2, dim)
def sincos_2d(dim, grid_size):
    coords = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(coords, coords)
    emb = np.concatenate([sincos_1d(dim // 2, grid_h), sincos_1d(dim // 2, grid_w)], axis=1)
    return torch.from_numpy(emb).float()


# Stochastic depth: drop the residual branch of whole samples with probability p while training
def drop_path(x, p, training):
    if p <= 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class Attention(nn.Module):
    # Multi-head self-attention with a fused qkv projection
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    # Attend over the tokens with PyTorch's fused scaled-dot-product kernel
    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    # Pre-norm transformer block (attention + GELU MLP) with stochastic depth
    def __init__(self, dim, num_heads, mlp_ratio, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_rate = drop_rate

    # Apply the attention and MLP residual branches
    def forward(self, x):
        x = x + drop_path(self.attn(self.norm1(x)), self.drop_rate, self.training)
        return x + drop_path(self.mlp(self.norm2(x)), self.drop_rate, self.training)


class ViTEncoder(nn.Module):
    # ViT encoder f: image (3, S, S) -> latent R^latent_dim, with a learned mask token for masked-patch pretraining
    def __init__(self, image_size=128, patch_size=16, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, latent_dim=512, drop_path=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid ** 2
        self.latent_dim = latent_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.register_buffer("pos_embed", sincos_2d(embed_dim, self.grid).unsqueeze(0), persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        rates = torch.linspace(0.0, float(drop_path), depth).tolist()
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio, rate) for rate in rates)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, latent_dim)
        self.reset_parameters()

    # MAE-style initialisation: xavier for linear layers and the patch projection, small normal for the mask token
    def reset_parameters(self):
        weight = self.patch_embed.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # Encode images to latents; `mask` (B, N) bool replaces those patch tokens with the mask token
    def forward(self, x, mask=None):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        if mask is not None:
            weight = mask.unsqueeze(-1).to(tokens.dtype)
            tokens = tokens * (1.0 - weight) + self.mask_token.to(tokens.dtype) * weight
        tokens = tokens + self.pos_embed.to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens.mean(dim=1)))
