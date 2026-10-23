import csv
import functools
import math
import os
import random

import numpy as np
import torch


# Seed Python, NumPy and PyTorch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Return the CUDA device or the CPU
def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def autocast(cfg, device):
    enabled = device.type == "cuda" and cfg.get("amp") == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


# Turn gradient tracking on or off for every parameter of a module
def requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad_(flag)


def param_groups(module, weight_decay, lr=None):
    embedding_ids = {id(m.weight) for m in module.modules() if isinstance(m, torch.nn.Embedding)}
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or id(param) in embedding_ids or name.endswith("mask_token"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if lr is not None:
        for group in groups:
            group["lr"] = lr
    return [group for group in groups if group["params"]]


# Learning-rate multiplier linear warmup, then cosine decay
def warmup_cosine(step, total_steps, warmup_steps, min_ratio):
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# LambdaLR scheduler that follows the warmup cosine multiplier
def build_scheduler(optimizer, total_steps, warmup_steps, min_ratio=0.0):
    factor = functools.partial(warmup_cosine, total_steps=total_steps, warmup_steps=warmup_steps, min_ratio=min_ratio)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# Save a checkpoint atomically (write a temporary file, then rename)
def save_checkpoint(state, path):
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


# Load a checkpoint onto the CPU
def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# Number of optimisation steps per epoch, capped by `max_steps_per_epoch` when it is set
def steps_per_epoch(num_batches, max_steps):
    return max(1, min(num_batches, int(max_steps))) if max_steps else max(1, num_batches)


# Run `fn(x)` over a loader without gradients; returns its concatenated outputs, then labels and sizes, on the CPU
@torch.no_grad()
def predict(fn, loader, device, cfg):
    chunks = []
    for x, labels, sizes, _ in loader:
        with autocast(cfg, device):
            out = fn(x.to(device, non_blocking=True))
        out = out if isinstance(out, tuple) else (out,)
        chunks.append([o.float().cpu() for o in out] + [labels, sizes])
    return [torch.cat(parts) for parts in zip(*chunks)]


class MeanTracker:
    # Start with nothing accumulated
    def __init__(self):
        self.sums = {}
        self.counts = {}

    # Accumulate a dict of scalars, each weighted by `n`
    def update(self, values, n=1):
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
            self.counts[key] = self.counts.get(key, 0) + n

    # Running mean of every tracked value
    def means(self):
        return {key: self.sums[key] / max(1, self.counts[key]) for key in self.sums}


class CSVLogger:
    # Remember the CSV path; `reset` deletes an old file, otherwise new rows are appended under its header
    def __init__(self, path, reset=False):
        self.path = path
        if reset and os.path.exists(path):
            os.remove(path)
        self.fields = None
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as handle:
                self.fields = next(csv.reader(handle), None)

    # Append one row, writing the header the first time
    def log(self, row):
        new_file = self.fields is None
        if new_file:
            self.fields = list(row.keys())
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
