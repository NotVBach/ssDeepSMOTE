import itertools
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.classifier import EncoderClassifier
from models.factory import build_encoder
from utils.config import build_parser, load_config, save_config, stage_dir, stage_file
from utils.data import build_dataset, load_manifest, make_loader, manifest_path
from utils.metrics import classification_report
from utils.train_utils import (
    CSVLogger,
    MeanTracker,
    autocast,
    build_scheduler,
    get_device,
    load_checkpoint,
    param_groups,
    predict,
    save_checkpoint,
    set_seed,
    steps_per_epoch,
)


# Classify every crop of a loader and return the metrics report
def evaluate(model, loader, cfg, device):
    model.eval()
    logits, labels, _ = predict(model, loader, device, cfg)
    return classification_report(labels.numpy(), logits.argmax(dim=1).numpy(), cfg["data"]["classes"])


# Per-class mean latent of the fine-tuned encoder over un-augmented training crops
def compute_centroids(encoder, loader, num_classes, cfg, device):
    encoder.eval()
    latents, labels, _ = predict(encoder, loader, device, cfg)
    centroids = torch.zeros(num_classes, latents.shape[1])
    counts = torch.zeros(num_classes, dtype=torch.long)
    for cls in range(num_classes):
        member = labels == cls
        counts[cls] = int(member.sum())
        if counts[cls] > 0:
            centroids[cls] = latents[member].mean(dim=0)
    return centroids, counts


# Weights and metadata of the classifier as stored in classifier.pt
def classifier_state(model, cfg, epoch, report):
    return {
        "encoder": model.encoder.state_dict(),
        "head": model.head.state_dict(),
        "dropout": float(cfg["classifier"]["dropout"]),
        "classes": cfg["data"]["classes"],
        "epoch": epoch,
        "val": report,
    }


# fine-tune the stage 1 encoder with a FC head, keep the best val macro-F1, then compute class centroids
def main():
    args = build_parser("supervised fine-tuning").parse_args()
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))
    device = get_device()
    out_dir = stage_dir(cfg, "stage2")
    save_config(cfg, out_dir)
    tcfg = cfg["train"]
    classes = cfg["data"]["classes"]
    limit = cfg["data"]["limit_per_class"]
    eval_batch = int(cfg["data"]["eval_batch_size"])

    train_df = load_manifest(manifest_path(cfg, "train"), limit, int(cfg["seed"]))
    train_set = build_dataset(train_df, cfg, augment=True)
    train_loader = make_loader(train_set, cfg, batch_size=min(int(tcfg["batch_size"]), len(train_set)), shuffle=True, drop_last=True)
    val_loader = make_loader(build_dataset(load_manifest(manifest_path(cfg, "val"), limit, int(cfg["seed"])), cfg), cfg, batch_size=eval_batch)

    encoder = build_encoder(cfg)
    encoder.load_state_dict(load_checkpoint(stage_file(cfg, "stage1", "mae.pt"))["encoder"])
    model = EncoderClassifier(encoder, len(classes), cfg["classifier"]["dropout"]).to(device)
    weight_decay = float(tcfg["weight_decay"])
    groups = param_groups(model.encoder, weight_decay, lr=float(tcfg["encoder_lr"])) + param_groups(model.head, weight_decay, lr=float(tcfg["head_lr"]))
    optimizer = torch.optim.AdamW(groups, betas=tuple(tcfg["betas"]))
    steps = steps_per_epoch(len(train_loader), tcfg.get("max_steps_per_epoch"))
    epochs = int(tcfg["epochs"])
    scheduler = build_scheduler(optimizer, epochs * steps, int(tcfg["warmup_epochs"]) * steps)

    start_epoch, best_f1, best_epoch = 0, -1.0, 0
    last_path = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "classifier.pt")
    if os.path.exists(last_path) and not args.fresh:
        state = load_checkpoint(last_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_f1, best_epoch = int(state["epoch"]), float(state["best_f1"]), int(state["best_epoch"])
        print(f"resumed from {last_path} at epoch {start_epoch}")
    logger = CSVLogger(os.path.join(out_dir, "history.csv"), reset=args.fresh)

    for epoch in range(start_epoch, epochs):
        model.train()
        tracker = MeanTracker()
        progress = tqdm(itertools.islice(train_loader, steps), total=steps, desc=f"stage2 {epoch + 1}/{epochs}")
        for x, labels, _, _ in progress:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(cfg, device):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), labels, label_smoothing=float(tcfg["label_smoothing"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            optimizer.step()
            scheduler.step()
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
            values = torch.stack([loss.detach(), accuracy]).tolist()
            tracker.update({"train_loss": values[0], "train_acc": values[1]}, x.shape[0])
            progress.set_postfix(loss=f"{values[0]:.4f}")
        report = evaluate(model, val_loader, cfg, device)
        if report["macro_f1"] > best_f1:
            best_f1, best_epoch = report["macro_f1"], epoch + 1
            save_checkpoint(classifier_state(model, cfg, epoch + 1, report), best_path)
        row = {
            "epoch": epoch + 1,
            **tracker.means(),
            "val_acc": report["accuracy"],
            "val_balanced_acc": report["balanced_accuracy"],
            "val_macro_f1": report["macro_f1"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[-1]["lr"],
        }
        save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_f1": best_f1, "best_epoch": best_epoch}, last_path)
        logger.log(row)
        print(row)

    best = load_checkpoint(best_path)
    model.encoder.load_state_dict(best["encoder"])
    model.head.load_state_dict(best["head"])
    train_set.augment = False
    centroids, counts = compute_centroids(model.encoder, make_loader(train_set, cfg, batch_size=eval_batch), len(classes), cfg, device)
    save_checkpoint({"centroids": centroids, "counts": counts, "classes": classes}, os.path.join(out_dir, "centroids.pt"))
    test_loader = make_loader(build_dataset(load_manifest(manifest_path(cfg, "test"), limit, int(cfg["seed"])), cfg, cache=False), cfg, batch_size=eval_batch)
    summary = {"best_epoch": int(best["epoch"]), "val": best["val"], "test": evaluate(model, test_loader, cfg, device), "centroid_counts": counts.tolist()}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"best epoch {summary['best_epoch']}: val macro-F1 {summary['val']['macro_f1']:.4f}, test macro-F1 {summary['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
