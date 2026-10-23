import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image


# Map images from [-1, 1] to [0, 1]
def to_unit(x):
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)


# Pixel box (left, top, right, bottom) that a width x height crop occupies after centred square padding and resizing to size x size
def content_box(width, height, size):
    side = max(width, height, 1e-6)
    box_w = int(min(size, max(1, round(width * size / side))))
    box_h = int(min(size, max(1, round(height * size / side))))
    left = (size - box_w) // 2
    top = (size - box_h) // 2
    return left, top, left + box_w, top + box_h


# Variance of the 3x3 Laplacian of the grayscale image on 0-255 scale inside each crop content box
def laplacian_variance(x, sizes_px, margin=2):
    sizes_px = sizes_px.detach().float().cpu()
    with torch.autocast(device_type=x.device.type, enabled=False):
        img = to_unit(x.detach().float()) * 255.0
        gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device).view(1, 1, 3, 3)
        lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kernel)[:, 0]
    size = x.shape[-1]
    values = []
    for i in range(x.shape[0]):
        left, top, right, bottom = content_box(float(sizes_px[i, 0]), float(sizes_px[i, 1]), size)
        left, top, right, bottom = left + margin, top + margin, right - margin, bottom - margin
        region = lap[i, top:bottom, left:right] if right - left > 1 and bottom - top > 1 else lap[i]
        values.append(region.var())
    return torch.stack(values)


# Cut the content box out of one decoded [-1, 1] image and resize it back to its true pixel size
def unpad_to_pil(x, width, height):
    size = x.shape[-1]
    array = to_unit(x.detach().float()).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    left, top, right, bottom = content_box(width, height, size)
    crop = Image.fromarray(array[top:bottom, left:right])
    return crop.resize((max(1, int(round(width))), max(1, int(round(height)))), Image.Resampling.BICUBIC)


# Save a batch of [-1, 1] images as one PNG grid
def save_grid(x, path, nrow):
    save_image(to_unit(x.detach().float().cpu()), path, nrow=nrow, padding=2)
