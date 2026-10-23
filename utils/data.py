import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

MANIFEST_COLUMNS = ["path", "label", "class_name", "width", "height", "source_image", "ann_id", "synthetic"]


# Path of the crop manifest CSV of one split
def manifest_path(cfg, split):
    return os.path.join(cfg["paths"]["output_root"], "data", f"{split}.csv")


# Read a manifest
def load_manifest(path, limit_per_class=None, seed=0):
    df = pd.read_csv(path)
    if limit_per_class:
        df = df.sample(frac=1.0, random_state=seed).groupby("label").head(int(limit_per_class))
        df = df.sort_values(["label", "path"])
    return df.reset_index(drop=True)


# Pick per_class random rows of every class
def sample_per_class(df, per_class, seed=0):
    picked = df.sample(frac=1.0, random_state=seed).groupby("label").head(per_class)
    return picked.sort_values(["label", "path"]).reset_index(drop=True)


# Pad a crop to a centred square with a constant colour, then resize it to size x size
def pad_and_resize(img, size, fill):
    width, height = img.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), tuple(int(v) for v in fill))
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.Resampling.BICUBIC)


class CropDataset(Dataset):
    # Keep manifest paths, labels and (w, h) / size_norm sizes
    def __init__(self, df, image_size, pad_fill, size_norm, augment=False, cache=False, num_threads=8):
        self.paths = df["path"].tolist()
        self.labels = torch.as_tensor(df["label"].to_numpy(), dtype=torch.long)
        self.sizes = torch.as_tensor(df[["width", "height"]].to_numpy(dtype=np.float32) / float(size_norm))
        self.image_size = int(image_size)
        self.pad_fill = pad_fill
        self.augment = augment
        self.cache = None
        if cache and self.paths:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                images = list(tqdm(pool.map(self.load, range(len(self.paths))), total=len(self.paths), desc="caching crops", leave=False))
            self.cache = torch.stack(images)

    # Number of crops
    def __len__(self):
        return len(self.paths)

    # Read one crop from disk as a padded, resized uint8 tensor (3, S, S)
    def load(self, index):
        with Image.open(self.paths[index]) as img:
            img = pad_and_resize(img.convert("RGB"), self.image_size, self.pad_fill)
        return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()

    # Return (image in [-1, 1], label, size, index); augmentation flips and rotates, swapping (w, h) on odd quarter turns
    def __getitem__(self, index):
        image = self.cache[index] if self.cache is not None else self.load(index)
        x = image.float().div(127.5).sub(1.0)
        size = self.sizes[index]
        if self.augment:
            if torch.rand(()) < 0.5:
                x = x.flip(2)
            if torch.rand(()) < 0.5:
                x = x.flip(1)
            turns = int(torch.randint(0, 4, ()))
            if turns:
                x = torch.rot90(x, turns, dims=(1, 2))
            if turns % 2:
                size = size.flip(0)
        return x, self.labels[index], size, index


# Build a CropDataset from the shared `data` config
def build_dataset(df, cfg, augment=False, cache=None):
    data = cfg["data"]
    use_cache = data["cache_in_memory"] if cache is None else cache
    return CropDataset(df, data["image_size"], data["pad_fill"], data["size_norm"], augment=augment, cache=use_cache, num_threads=max(1, int(data["num_workers"])))


# Build a DataLoader with the configured number of workers (or `workers`) and pinned memory
def make_loader(dataset, cfg, batch_size=None, shuffle=False, drop_last=False, batch_sampler=None, workers=None):
    num_workers = int(cfg["data"]["num_workers"]) if workers is None else workers
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


class ClassBalancedBatchSampler(Sampler):
    # Index samples by class; every batch holds `samples_per_class` crops from each of `classes_per_batch` classes
    def __init__(self, labels, classes_per_batch, samples_per_class, num_batches, seed=0):
        labels = np.asarray(labels)
        self.class_indices = [np.flatnonzero(labels == value) for value in np.unique(labels)]
        self.classes_per_batch = min(int(classes_per_batch), len(self.class_indices))
        self.samples_per_class = int(samples_per_class)
        self.num_batches = int(num_batches)
        self.rng = np.random.default_rng(seed)

    # Number of batches per epoch
    def __len__(self):
        return self.num_batches

    # Yield lists of dataset indices, drawing classes without and samples with replacement only when a class is too small
    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.choice(len(self.class_indices), self.classes_per_batch, replace=False)
            batch = []
            for position in chosen:
                pool = self.class_indices[position]
                batch.extend(self.rng.choice(pool, self.samples_per_class, replace=len(pool) < self.samples_per_class).tolist())
            yield batch
