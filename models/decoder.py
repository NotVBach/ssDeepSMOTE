# https://github.com/openai/guided-diffusion
# https://github.com/dd1github/DeepSMOTE

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AdaGroupNorm(nn.Module):
    # GroupNorm whose per-channel scale and shift come from the condition vector (zero-initialised, so it starts as plain GroupNorm)
    def __init__(self, channels, cond_dim, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).to(x.dtype).chunk(2, dim=1)
        return self.norm(x).to(x.dtype) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = AdaGroupNorm(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_channels, cond_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, cond):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return h + self.skip(x)


class ConditionalDecoder(nn.Module):
    # Decoder g(z, y, size): latent + class embedding + crop size -> image in [-1, 1]
    def __init__(self, latent_dim=512, num_classes=8, image_size=128, channels=(512, 512, 256, 256, 128, 64), cond_dim=256, class_dim=128, size_dim=64, grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)
        num_up = int(math.log2(image_size // 4))
        if len(channels) != num_up + 1:
            raise ValueError(f"decoder_channels needs {num_up + 1} entries for image_size {image_size}")
        self.null_class = num_classes
        self.channels = list(channels)
        in_dim = latent_dim + class_dim + size_dim
        self.class_embed = nn.Embedding(num_classes + 1, class_dim)
        self.size_mlp = nn.Sequential(nn.Linear(2, size_dim), nn.SiLU(), nn.Linear(size_dim, size_dim))
        self.cond_mlp = nn.Sequential(nn.Linear(in_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.fc = nn.Linear(in_dim, self.channels[0] * 16)
        self.blocks = nn.ModuleList(UpBlock(self.channels[i], self.channels[i + 1], cond_dim) for i in range(num_up))
        self.out_norm = nn.GroupNorm(min(32, self.channels[-1]), self.channels[-1])
        self.out_conv = nn.Conv2d(self.channels[-1], 3, 3, padding=1)

    @torch.no_grad()
    def init_class_embeddings_from_null(self):
        null = self.class_embed.weight[self.null_class].clone()
        self.class_embed.weight[: self.null_class].copy_(null.expand(self.null_class, -1))

    # Decode latents z with labels
    def forward(self, z, labels, sizes):
        if labels is None:
            labels = torch.full((z.shape[0],), self.null_class, dtype=torch.long, device=z.device)
        h = torch.cat([z.float(), self.class_embed(labels).float(), self.size_mlp(sizes.float()).float()], dim=1)
        cond = self.cond_mlp(h)
        x = self.fc(h).view(-1, self.channels[0], 4, 4)
        use_checkpoint = self.grad_checkpoint and torch.is_grad_enabled()
        for block in self.blocks:
            x = checkpoint(block, x, cond, use_reentrant=False) if use_checkpoint else block(x, cond)
        return torch.tanh(self.out_conv(F.silu(self.out_norm(x).to(x.dtype))))
