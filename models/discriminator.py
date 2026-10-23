# https://github.com/NVlabs/stylegan2-ada-pytorch
# https://github.com/pfnet-research/sngan_projection
# https://github.com/LMescheder/GAN_stability

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# StyleGAN minibatch standard deviation: append one feature map holding the mean per-group stddev
def minibatch_stddev(x, group_size):
    batch, channels, height, width = x.shape
    group = min(int(group_size), batch)
    while batch % group:
        group -= 1
    y = x.float().view(group, -1, channels, height, width)
    y = (y - y.mean(dim=0)).pow(2).mean(dim=0).add(1e-8).sqrt()
    y = y.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1).repeat(group, 1, height, width)
    return torch.cat([x, y.to(x.dtype)], dim=1)


class DownBlock(nn.Module):
    # Residual block that halves the resolution: two 3x3 convs with LeakyReLU, average pooling and a 1x1 skip
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    # Downsample by two, scaling the residual sum by 1/sqrt(2)
    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.avg_pool2d(F.leaky_relu(self.conv2(h), 0.2), 2)
        return (h + self.skip(F.avg_pool2d(x, 2))) / math.sqrt(2.0)


class ProjectionDiscriminator(nn.Module):
    def __init__(self, num_classes, image_size=128, channels=(64, 128, 256, 512, 512, 512), mbstd_group=4, proj_dim=128):
        super().__init__()
        num_down = int(math.log2(image_size // 4))
        if len(channels) != num_down + 1:
            raise ValueError(f"disc channels needs {num_down + 1} entries for image_size {image_size}")
        last = channels[-1]
        self.mbstd_group = mbstd_group
        self.from_rgb = nn.Conv2d(3, channels[0], 1)
        self.blocks = nn.Sequential(*[DownBlock(channels[i], channels[i + 1]) for i in range(num_down)])
        self.conv = nn.Conv2d(last + 1, last, 3, padding=1)
        self.fc = nn.Linear(last * 16, last)
        self.out = nn.Linear(last, 1)
        self.proj = nn.Linear(last, proj_dim)
        self.embed = nn.Embedding(num_classes, proj_dim)

    def forward(self, x, labels):
        h = self.blocks(F.leaky_relu(self.from_rgb(x), 0.2))
        h = F.leaky_relu(self.conv(minibatch_stddev(h, self.mbstd_group)), 0.2)
        phi = F.leaky_relu(self.fc(h.flatten(1)), 0.2)
        return self.out(phi).squeeze(1) + (self.proj(phi) * self.embed(labels).to(phi.dtype)).sum(dim=1)


# R1 penalty gamma / 2 * E||grad_x D(x, y)||^2
def r1_penalty(disc, x, labels, gamma):
    x = x.detach().float().requires_grad_(True)
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = disc(x, labels)
    (grad,) = torch.autograd.grad(logits.sum(), x, create_graph=True)
    return 0.5 * float(gamma) * grad.pow(2).flatten(1).sum(dim=1).mean()
