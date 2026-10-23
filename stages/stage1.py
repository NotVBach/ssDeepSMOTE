import itertools
import os

import torch
from tqdm import tqdm

from models.factory import build_decoder, build_encoder
from models.mae import MaskedAutoencoder
from utils.config import build_parser, load_config, save_config, stage_dir
from utils.data import build_dataset, load_manifest, make_loader, manifest_path, sample_per_class
from utils.image_ops import save_grid
from utils.train_utils import (
    CSVLogger,
    MeanTracker,
    autocast,
    build_scheduler,
    get_device,
    load_checkpoint,
    param_groups,
    save_checkpoint,
    set_seed,
    steps_per_epoch,
)


# Save a grid with rows: originals, masked inputs, reconstructions, reconstructions with the visible patches pasted back
@torch.no_grad()
def save_samples(model, batch, path, cfg, device):
    model.eval()
    x, _, sizes, _ = batch
    x, sizes = x.to(device), sizes.to(device)
    with autocast(cfg, device):
        _, recon, mask = model(x, sizes)
    recon = recon.float()
    pixel_mask = model.mask_to_pixels(mask)
    masked = x * (1.0 - pixel_mask)
    pasted = masked + recon * pixel_mask
    save_grid(torch.cat([x, masked, recon, pasted]), path, nrow=x.shape[0])
    model.train()


# pretrain the ViT encoder and the decoder as a masked autoencoder on unlabeled training crops
def main():
    args = build_parser("MAE pretraining").parse_args()
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))
    device = get_device()
    out_dir = stage_dir(cfg, "stage1")
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    save_config(cfg, out_dir)
    tcfg = cfg["train"]

    df = load_manifest(manifest_path(cfg, "train"), cfg["data"]["limit_per_class"], int(cfg["seed"]))
    dataset = build_dataset(df, cfg, augment=True)
    loader = make_loader(dataset, cfg, batch_size=min(int(tcfg["batch_size"]), len(dataset)), shuffle=True, drop_last=True)
    fixed_set = build_dataset(sample_per_class(df, 2, int(cfg["seed"])), cfg, cache=False)
    fixed = next(iter(make_loader(fixed_set, cfg, batch_size=len(fixed_set), workers=0)))

    model = MaskedAutoencoder(build_encoder(cfg), build_decoder(cfg), cfg["mae"]["mask_ratio"], cfg["mae"]["visible_loss_weight"]).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, float(tcfg["weight_decay"])), lr=float(tcfg["lr"]), betas=tuple(tcfg["betas"]))
    steps = steps_per_epoch(len(loader), tcfg.get("max_steps_per_epoch"))
    epochs = int(tcfg["epochs"])
    scheduler = build_scheduler(optimizer, epochs * steps, int(tcfg["warmup_epochs"]) * steps)

    start_epoch = 0
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path) and not args.fresh:
        state = load_checkpoint(last_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        print(f"resumed from {last_path} at epoch {start_epoch}")
    logger = CSVLogger(os.path.join(out_dir, "history.csv"), reset=args.fresh)

    for epoch in range(start_epoch, epochs):
        model.train()
        tracker = MeanTracker()
        progress = tqdm(itertools.islice(loader, steps), total=steps, desc=f"stage1 {epoch + 1}/{epochs}")
        for x, _, sizes, _ in progress:
            x = x.to(device, non_blocking=True)
            sizes = sizes.to(device, non_blocking=True)
            with autocast(cfg, device):
                loss, _, _ = model(x, sizes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            optimizer.step()
            scheduler.step()
            value = loss.item()
            tracker.update({"loss": value}, x.shape[0])
            progress.set_postfix(loss=f"{value:.4f}")
        row = {"epoch": epoch + 1, **tracker.means(), "lr": optimizer.param_groups[0]["lr"]}
        save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1}, last_path)
        save_checkpoint({"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(), "model_cfg": cfg["model"], "epoch": epoch + 1}, os.path.join(out_dir, "mae.pt"))
        logger.log(row)
        print(row)
        if (epoch + 1) % int(tcfg["sample_every"]) == 0 or epoch + 1 == epochs:
            save_samples(model, fixed, os.path.join(out_dir, "samples", f"epoch_{epoch + 1:03d}.png"), cfg, device)


if __name__ == "__main__":
    main()
