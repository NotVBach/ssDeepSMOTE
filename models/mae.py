# https://github.com/facebookresearch/mae

import torch
import torch.nn as nn


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.6, visible_loss_weight=0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = float(mask_ratio)
        self.visible_loss_weight = float(visible_loss_weight)
    def random_mask(self, batch, device):
        num_patches = self.encoder.num_patches
        num_masked = int(round(self.mask_ratio * num_patches))
        ranks = torch.rand(batch, num_patches, device=device).argsort(dim=1).argsort(dim=1)
        return ranks < num_masked

    def patchify(self, x):
        p = self.encoder.patch_size
        batch, channels, height, width = x.shape
        x = x.reshape(batch, channels, height // p, p, width // p, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(batch, (height // p) * (width // p), p * p * channels)

    def mask_to_pixels(self, mask):
        grid, p = self.encoder.grid, self.encoder.patch_size
        pixels = mask.float().view(-1, 1, grid, grid)
        return pixels.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)

    def forward(self, x, sizes):
        mask = self.random_mask(x.shape[0], x.device)
        recon = self.decoder(self.encoder(x, mask), None, sizes)
        err = (self.patchify(recon.float()) - self.patchify(x.float())).abs().mean(dim=-1)
        masked = mask.float()
        loss = (err * masked).sum() / masked.sum().clamp_min(1.0)
        if self.visible_loss_weight > 0:
            visible = 1.0 - masked
            loss = loss + self.visible_loss_weight * (err * visible).sum() / visible.sum().clamp_min(1.0)
        return loss, recon, mask
