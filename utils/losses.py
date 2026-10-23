import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_ops import to_unit

MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


class LPIPSLoss(nn.Module):
    # Frozen LPIPS network (VGG features) used as a perceptual distance
    def __init__(self, net="vgg"):
        super().__init__()
        self.model = lpips.LPIPS(net=net, verbose=False)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.eval()

    # Keep the LPIPS network in eval mode whatever mode is requested
    def train(self, mode=True):
        return super().train(False)

    # Mean LPIPS distance between two image batches in [-1, 1]
    def forward(self, x, y):
        return self.model(x, y).float().mean()


# Mean absolute error in fp32
def l1_loss(x, y):
    return (x.float() - y.float()).abs().mean()


# Normalised 1-D Gaussian window
def gaussian_window(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    window = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return window / window.sum()


# Separable depthwise Gaussian filter with valid padding
def gaussian_blur(x, window):
    channels = x.shape[1]
    kernel_x = window.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = window.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    return F.conv2d(F.conv2d(x, kernel_x, groups=channels), kernel_y, groups=channels)


# Per-image mean SSIM and contrast-structure terms for images in [0, 1]
def ssim_terms(x, y, window):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = gaussian_blur(x, window), gaussian_blur(y, window)
    sigma_xx = gaussian_blur(x * x, window) - mu_x ** 2
    sigma_yy = gaussian_blur(y * y, window) - mu_y ** 2
    sigma_xy = gaussian_blur(x * y, window) - mu_x * mu_y
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    ssim = ((2.0 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)) * cs
    return ssim.flatten(1).mean(1), cs.flatten(1).mean(1)


# Multi-scale SSIM over levels scales (standard weights renormalised) for [-1, 1] inputs, computed in fp32
def ms_ssim(x, y, levels=3, window_size=11, sigma=1.5):
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = to_unit(x.float()), to_unit(y.float())
        weights = torch.tensor(MS_SSIM_WEIGHTS[:levels], device=x.device)
        weights = weights / weights.sum()
        window = gaussian_window(window_size, sigma, x.device)
        values = []
        for level in range(levels):
            ssim, cs = ssim_terms(x, y, window)
            if level == levels - 1:
                values.append(ssim)
            else:
                values.append(cs)
                x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
        stacked = torch.stack(values, dim=0).clamp_min(1e-6)
        return torch.prod(stacked ** weights.view(-1, 1), dim=0).mean()


# Hinge discriminator loss: 0.5 E[relu(1 - D(real))] + 0.5 E[relu(1 + D(fake))]
def d_hinge_loss(real_logits, fake_logits):
    return 0.5 * F.relu(1.0 - real_logits.float()).mean() + 0.5 * F.relu(1.0 + fake_logits.float()).mean()


# Generator adversarial loss -E[D(fake)]
def g_adv_loss(fake_logits):
    return -fake_logits.float().mean()


# Cycle loss mean squared gap between the re-encoded and target latents, divided by the target RMS
def cycle_loss(z_reencoded, z_target):
    z_target = z_target.detach().float()
    return (z_reencoded.float() - z_target).pow(2).mean() / z_target.pow(2).mean().sqrt().clamp_min(1e-6)


# Classification loss frozen-classifier cross-entropy plus squared distance between the l2-normalised feature and class centroid
def class_loss(logits, feats, labels, centroids):
    ce = F.cross_entropy(logits.float(), labels)
    feat = F.normalize(feats.float(), dim=1)
    centre = F.normalize(centroids[labels].float(), dim=1)
    return ce + (feat - centre).pow(2).sum(dim=1).mean()
