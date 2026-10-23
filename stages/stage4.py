import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.factory import build_decoder, build_encoder, load_frozen_classifier
from utils.config import build_parser, load_config, save_config, stage_dir, stage_file
from utils.data import MANIFEST_COLUMNS, build_dataset, load_manifest, make_loader, manifest_path
from utils.image_ops import laplacian_variance, save_grid, unpad_to_pil
from utils.latent import ClassGaussian, cosine_knn, interpolate, max_cosine, sample_lambda
from utils.train_utils import autocast, get_device, load_checkpoint, requires_grad, set_seed

CHECKS = ["confidence", "mahalanobis", "duplicate", "blur"]


# Load the encoder and decoder, frozen and in eval mode
def load_generator(cfg, device):
    state = load_checkpoint(stage_file(cfg, "stage3", "generator.pt"))
    encoder, decoder = build_encoder(cfg), build_decoder(cfg)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    for module in (encoder, decoder):
        module.to(device).eval()
        requires_grad(module, False)
    return encoder, decoder


# Encode every real training crop and measure its Laplacian variance at the working resolution
@torch.no_grad()
def encode_real(encoder, dataset, cfg, device):
    loader = make_loader(dataset, cfg, batch_size=int(cfg["data"]["eval_batch_size"]))
    size_norm = float(cfg["data"]["size_norm"])
    latents, blur = [], []
    for x, _, sizes, _ in tqdm(loader, desc="encoding real crops"):
        x = x.to(device, non_blocking=True)
        with autocast(cfg, device):
            latents.append(encoder(x).float())
        blur.append(laplacian_variance(x, sizes * size_norm).cpu())
    return torch.cat(latents), torch.cat(blur)


# real latents, sizes, paths, cosine kNN table, Ledoit-Wolf Gaussian and the Mahalanobis / blur thresholds
def calibrate(cls, latents, blur, labels, sizes, paths, gcfg):
    member = labels == cls
    z = latents[member.to(latents.device)]
    z_numpy = z.cpu().numpy()
    gaussian = ClassGaussian(z_numpy)
    return {
        "z": z,
        "sizes": sizes[member.to(sizes.device)],
        "paths": [path for path, keep in zip(paths, member.tolist()) if keep],
        "knn": cosine_knn(z, int(gcfg["k"])),
        "gaussian": gaussian,
        "maha_threshold": float(np.percentile(gaussian.distance(z_numpy), float(gcfg["maha_percentile"]))),
        "blur_threshold": float(np.percentile(blur[member].numpy(), float(gcfg["blur_percentile"]))),
    }


@torch.no_grad()
def decode_and_verify(z_syn, sizes_syn, cls, ref, encoder, decoder, clf, cfg, device):
    gcfg = cfg["generate"]
    labels = torch.full((z_syn.shape[0],), cls, dtype=torch.long, device=device)
    with autocast(cfg, device):
        x_syn = decoder(z_syn, labels, sizes_syn).float()
        probs = torch.softmax(clf(x_syn).float(), dim=1)
        z_re = encoder(x_syn).float()
    confidence = probs[:, cls]
    maha = torch.from_numpy(ref["gaussian"].distance(z_re.cpu().numpy())).float().to(device)
    dup = max_cosine(z_re, ref["z"])
    blur = laplacian_variance(x_syn, sizes_syn * float(cfg["data"]["size_norm"])).to(device)
    checks = {
        "confidence": (probs.argmax(dim=1) == cls) & (confidence >= float(gcfg["conf_threshold"])),
        "mahalanobis": maha <= ref["maha_threshold"],
        "duplicate": dup <= float(gcfg["dup_threshold"]),
        "blur": blur >= ref["blur_threshold"],
    }
    scores = {"confidence": confidence, "mahalanobis": maha, "dup_cosine": dup, "lapvar": blur}
    return x_syn, checks, scores


# Generate verified synthetic crops of one class until pass or the attempt budget runs out; returns rows, counters and preview images
def generate_class(cls, need, ref, models, cfg, device, out_dir, rng):
    encoder, decoder, clf = models
    gcfg = cfg["generate"]
    name = cfg["data"]["classes"][cls]
    size_norm = float(cfg["data"]["size_norm"])
    class_dir = os.path.join(out_dir, "synthetic", name)
    os.makedirs(class_dir, exist_ok=True)
    num_real, k = ref["z"].shape[0], ref["knn"].shape[1]
    budget = int(np.ceil(need * float(gcfg["max_attempts_factor"])))
    anchors = np.concatenate([rng.permutation(num_real) for _ in range(budget // num_real + 2)])
    counters = {"attempts": 0, "passed": 0, **{check: 0 for check in CHECKS}}
    rows, preview = [], []
    progress = tqdm(total=need, desc=f"generating {name}")
    while counters["passed"] < need and counters["attempts"] < budget:
        start = counters["attempts"]
        n = min(int(gcfg["batch_size"]), budget - start)
        anchor = torch.as_tensor(anchors[start:start + n], device=device)
        neighbour = ref["knn"][anchor, torch.randint(0, k, (n,), device=device)]
        lam = sample_lambda(n, gcfg["lambda_min"], gcfg["lambda_max"], device)
        z_syn = interpolate(ref["z"][anchor], ref["z"][neighbour], lam)
        sizes_syn = interpolate(ref["sizes"][anchor], ref["sizes"][neighbour], lam)
        x_syn, checks, scores = decode_and_verify(z_syn, sizes_syn, cls, ref, encoder, decoder, clf, cfg, device)
        passed = checks["confidence"] & checks["mahalanobis"] & checks["duplicate"] & checks["blur"]
        counters["attempts"] += n
        for check in CHECKS:
            counters[check] += int(checks[check].sum())
        keep = torch.nonzero(passed, as_tuple=True)[0][: need - counters["passed"]].tolist()
        pixel_sizes = (sizes_syn * size_norm).cpu().tolist()
        anchor_list, neighbour_list, lam_list = anchor.tolist(), neighbour.tolist(), lam.squeeze(1).tolist()
        score_lists = {key: value.cpu().tolist() for key, value in scores.items()}
        for i in keep:
            width, height = pixel_sizes[i]
            path = os.path.join(class_dir, f"syn_{name}_{counters['passed']:05d}.png").replace(os.sep, "/")
            unpad_to_pil(x_syn[i], width, height).save(path)
            rows.append({
                "path": path,
                "label": cls,
                "class_name": name,
                "width": max(1, int(round(width))),
                "height": max(1, int(round(height))),
                "source_image": "",
                "ann_id": -1,
                "synthetic": 1,
                "anchor": ref["paths"][anchor_list[i]],
                "neighbour": ref["paths"][neighbour_list[i]],
                "lambda": lam_list[i],
                **{key: values[i] for key, values in score_lists.items()},
            })
            if len(preview) < int(gcfg["preview"]):
                preview.append(x_syn[i].cpu())
            counters["passed"] += 1
        progress.update(len(keep))
    progress.close()
    return rows, counters, preview


# Save a two-row grid per class. real crops on top, accepted synthetic crops below
def save_preview(dataset, cls, preview, path, count):
    real_index = torch.nonzero(dataset.labels == cls, as_tuple=True)[0][:count].tolist()
    real = [dataset[i][0] for i in real_index]
    if real or preview:
        save_grid(torch.stack(real + preview), path, nrow=max(1, count))


# SMOTE, latent space, decode, verify every image, and write the balanced training manifest
def main():
    args = build_parser("synthetic generation").parse_args()
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))
    device = get_device()
    out_dir = stage_dir(cfg, "stage4")
    save_config(cfg, out_dir)
    shutil.rmtree(os.path.join(out_dir, "synthetic"), ignore_errors=True)
    os.makedirs(os.path.join(out_dir, "grids"), exist_ok=True)
    gcfg = cfg["generate"]
    classes = cfg["data"]["classes"]

    df = load_manifest(manifest_path(cfg, "train"), cfg["data"]["limit_per_class"], int(cfg["seed"]))
    dataset = build_dataset(df, cfg, augment=False)
    encoder, decoder = load_generator(cfg, device)
    clf = load_frozen_classifier(cfg, stage_file(cfg, "stage2", "classifier.pt"), device)
    latents, blur = encode_real(encoder, dataset, cfg, device)
    labels = dataset.labels
    sizes = dataset.sizes.to(device)
    paths = df["path"].tolist()
    counts = np.bincount(labels.numpy(), minlength=len(classes))
    target = int(counts.max()) if str(gcfg["target"]).lower() == "max" else int(gcfg["target"])
    rng = np.random.default_rng(int(cfg["seed"]))
    print(f"balance target: {target} crops per class")

    synthetic_rows, report = [], []
    for cls, name in enumerate(classes):
        need = max(0, target - int(counts[cls]))
        entry = {"class": name, "real": int(counts[cls]), "target": target, "need": need, "generated": 0, "attempts": 0}
        if need == 0 or counts[cls] < 2:
            report.append(entry)
            print(f"{name}: nothing to generate ({counts[cls]} real crops)")
            continue
        ref = calibrate(cls, latents, blur, labels, sizes, paths, gcfg)
        rows, counters, preview = generate_class(cls, need, ref, (encoder, decoder, clf), cfg, device, out_dir, rng)
        synthetic_rows.extend(rows)
        attempts = max(1, counters["attempts"])
        entry.update({
            "generated": counters["passed"],
            "attempts": counters["attempts"],
            "pass_rate": counters["passed"] / attempts,
            **{f"{check}_pass_rate": counters[check] / attempts for check in CHECKS},
            "maha_threshold": ref["maha_threshold"],
            "blur_threshold": ref["blur_threshold"],
        })
        report.append(entry)
        save_preview(dataset, cls, preview, os.path.join(out_dir, "grids", f"{name}.png"), int(gcfg["preview"]))
        print(f"{name}: generated {counters['passed']}/{need} from {counters['attempts']} attempts")

    synthetic = pd.DataFrame(synthetic_rows)
    synthetic.to_csv(os.path.join(out_dir, "synthetic.csv"), index=False)
    pd.DataFrame(report).to_csv(os.path.join(out_dir, "verification_report.csv"), index=False)
    balanced = pd.concat([df[MANIFEST_COLUMNS], synthetic.reindex(columns=MANIFEST_COLUMNS)], ignore_index=True) if len(synthetic) else df[MANIFEST_COLUMNS]
    balanced.to_csv(os.path.join(out_dir, "balanced_train.csv"), index=False)
    final = balanced["label"].value_counts().reindex(range(len(classes)), fill_value=0).astype(int).tolist()
    print("balanced training set: " + ", ".join(f"{name} {count}" for name, count in zip(classes, final)))


if __name__ == "__main__":
    main()
