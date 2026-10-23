import os

import torch
from tqdm import tqdm

from models.discriminator import r1_penalty
from models.factory import build_decoder, build_discriminator, build_encoder, load_frozen_classifier
from utils.config import build_parser, load_config, save_config, stage_dir, stage_file
from utils.data import ClassBalancedBatchSampler, build_dataset, load_manifest, make_loader, manifest_path, sample_per_class
from utils.image_ops import save_grid
from utils.latent import interpolate, sample_lambda, same_class_derangement
from utils.losses import LPIPSLoss, class_loss, cycle_loss, d_hinge_loss, g_adv_loss, l1_loss, ms_ssim
from utils.train_utils import (
    CSVLogger,
    MeanTracker,
    autocast,
    get_device,
    load_checkpoint,
    param_groups,
    requires_grad,
    save_checkpoint,
    set_seed,
    steps_per_epoch,
)

LOG_KEYS = ["total", "recon", "l1", "lpips", "msssim", "perm", "cycle", "cls", "adv", "d_loss", "d_real", "d_fake", "r1"]


class Stage3Trainer:
    # Hold the trainable encoder/decoder, the discriminator, the frozen classifier with its centroids, and both optimizers
    def __init__(self, cfg, device, encoder, decoder, disc, clf, centroids):
        tcfg = cfg["train"]
        self.cfg = cfg
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.disc = disc
        self.clf = clf
        self.centroids = centroids
        self.loss_cfg = {key: float(value) for key, value in cfg["loss"].items()}
        self.disc_cfg = cfg["disc"]
        self.lpips = LPIPSLoss().to(device)
        weight_decay = float(tcfg["weight_decay"])
        groups = param_groups(encoder, weight_decay, lr=float(tcfg["encoder_lr"])) + param_groups(decoder, weight_decay, lr=float(tcfg["decoder_lr"]))
        self.opt_g = torch.optim.AdamW(groups, betas=tuple(tcfg["betas"]))
        self.opt_d = torch.optim.AdamW(disc.parameters(), lr=float(tcfg["disc_lr"]), betas=tuple(tcfg["disc_betas"]), weight_decay=0.0)
        self.grad_clip = tcfg.get("grad_clip")
        self.step = 0

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()
        self.disc.train()

    # Interpolate every latent and size
    def mix(self, z, sizes, labels):
        partner = same_class_derangement(labels)
        lam = sample_lambda(labels.numel(), self.loss_cfg["lambda_min"], self.loss_cfg["lambda_max"], z.device)
        return interpolate(z, z[partner], lam), interpolate(sizes, sizes[partner], lam)
    
    def generator_loss(self, x, labels, sizes, use_adv):
        lc = self.loss_cfg
        z = self.encoder(x)
        x_hat = self.decoder(z, labels, sizes)
        l1 = l1_loss(x_hat, x)
        perceptual = self.lpips(x_hat, x)
        msssim = ms_ssim(x_hat, x)
        recon = lc["w_l1"] * l1 + lc["w_lpips"] * perceptual + lc["w_msssim"] * (1.0 - msssim)

        perm = same_class_derangement(labels)
        perm_loss = l1_loss(self.decoder(z[perm], labels, sizes[perm]), x)

        z_mix, sizes_mix = self.mix(z, sizes, labels)
        x_mix = self.decoder(z_mix, labels, sizes_mix)
        cyc = cycle_loss(self.encoder(x_mix), z_mix)
        feats, logits = self.clf.features_and_logits(x_mix)
        cls = class_loss(logits, feats, labels, self.centroids)
        adv = g_adv_loss(self.disc(x_mix, labels)) if use_adv else torch.zeros((), device=x.device)

        total = lc["w_recon"] * recon + lc["w_perm"] * perm_loss + lc["w_cycle"] * cyc + lc["w_cls"] * cls + lc["w_adv"] * adv
        logs = {"total": total, "recon": recon, "l1": l1, "lpips": perceptual, "msssim": msssim, "perm": perm_loss, "cycle": cyc, "cls": cls, "adv": adv}
        return total, logs, x_mix
    
    def discriminator_step(self, x, labels, x_mix):
        dc = self.disc_cfg
        requires_grad(self.disc, True)
        with autocast(self.cfg, self.device):
            real = self.disc(x, labels).float()
            fake = self.disc(x_mix, labels).float()
        loss = d_hinge_loss(real, fake)
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        logs = {"d_loss": loss.detach(), "d_real": real.mean().detach(), "d_fake": fake.mean().detach()}
        interval = int(dc["r1_interval"])
        if interval > 0 and self.step % interval == 0:
            pick = torch.randperm(x.shape[0], device=x.device)[: int(dc["r1_batch"])]
            r1 = r1_penalty(self.disc, x[pick], labels[pick], float(dc["r1_gamma"]))
            (r1 * interval).backward()
            logs["r1"] = r1.detach()
        self.opt_d.step()
        return logs

    # One encoder/decoder update followed by one discriminator update, returns the logged scalars
    def train_step(self, x, labels, sizes, use_adv):
        requires_grad(self.disc, False)
        with autocast(self.cfg, self.device):
            total, logs, x_mix = self.generator_loss(x, labels, sizes, use_adv)
        self.opt_g.zero_grad(set_to_none=True)
        total.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), float(self.grad_clip))
        self.opt_g.step()
        if use_adv:
            logs.update(self.discriminator_step(x, labels, x_mix.detach()))
        self.step += 1
        keys = list(logs.keys())
        values = torch.stack([logs[key].detach().float() for key in keys]).tolist()
        return dict(zip(keys, values))

    # Full training state for resuming
    def state(self, epoch):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "disc": self.disc.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "epoch": epoch,
            "step": self.step,
        }

    def load_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.disc.load_state_dict(state["disc"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.step = int(state["step"])
        return int(state["epoch"])


# Save a grid with rows: real crops, reconstructions, their same-class partners, and lambda = 0.5 interpolants
@torch.no_grad()
def save_samples(trainer, batch, path):
    device = trainer.device
    x, labels, sizes, _ = batch
    x, labels, sizes = x.to(device), labels.to(device), sizes.to(device)
    trainer.encoder.eval()
    trainer.decoder.eval()
    with autocast(trainer.cfg, device):
        z = trainer.encoder(x)
        x_hat = trainer.decoder(z, labels, sizes)
        partner = same_class_derangement(labels)
        half = torch.full((labels.numel(), 1), 0.5, device=device)
        x_mix = trainer.decoder(interpolate(z, z[partner], half), labels, interpolate(sizes, sizes[partner], half))
    save_grid(torch.cat([x, x_hat.float(), x[partner], x_mix.float()]), path, nrow=x.shape[0])
    trainer.train_mode()


# train the encoder and class-conditional decoder (against the discriminator) to decode same-class interpolants
def main():
    args = build_parser("encoder/decoder training").parse_args()
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))
    device = get_device()
    out_dir = stage_dir(cfg, "stage3")
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    save_config(cfg, out_dir)
    tcfg = cfg["train"]

    df = load_manifest(manifest_path(cfg, "train"), cfg["data"]["limit_per_class"], int(cfg["seed"]))
    dataset = build_dataset(df, cfg, augment=bool(tcfg["augment"]))
    classes_per_batch, samples_per_class = int(tcfg["classes_per_batch"]), int(tcfg["samples_per_class"])
    steps = steps_per_epoch(len(dataset) // (classes_per_batch * samples_per_class), tcfg.get("max_steps_per_epoch"))
    sampler = ClassBalancedBatchSampler(df["label"].to_numpy(), classes_per_batch, samples_per_class, steps, int(cfg["seed"]))
    loader = make_loader(dataset, cfg, batch_sampler=sampler)
    fixed_set = build_dataset(sample_per_class(df, 2, int(cfg["seed"])), cfg, cache=False)
    fixed = next(iter(make_loader(fixed_set, cfg, batch_size=len(fixed_set), workers=0)))

    encoder = build_encoder(cfg).to(device)
    decoder = build_decoder(cfg).to(device)
    disc = build_discriminator(cfg).to(device)
    clf = load_frozen_classifier(cfg, stage_file(cfg, "stage2", "classifier.pt"), device)
    centroids = load_checkpoint(stage_file(cfg, "stage2", "centroids.pt"))["centroids"].to(device)

    last_path = os.path.join(out_dir, "last.pt")
    resume = os.path.exists(last_path) and not args.fresh
    if not resume:
        mae = load_checkpoint(stage_file(cfg, "stage1", "mae.pt"))
        encoder.load_state_dict(mae["encoder"])
        decoder.load_state_dict(mae["decoder"])
        decoder.init_class_embeddings_from_null()
    trainer = Stage3Trainer(cfg, device, encoder, decoder, disc, clf, centroids)
    start_epoch = 0
    if resume:
        start_epoch = trainer.load_state(load_checkpoint(last_path))
        print(f"resumed from {last_path} at epoch {start_epoch}")
    logger = CSVLogger(os.path.join(out_dir, "history.csv"), reset=args.fresh)

    epochs = int(tcfg["epochs"])
    adv_start = int(cfg["loss"]["adv_start_epoch"])
    trainer.train_mode()
    for epoch in range(start_epoch, epochs):
        tracker = MeanTracker()
        progress = tqdm(loader, total=steps, desc=f"stage3 {epoch + 1}/{epochs}")
        for x, labels, sizes, _ in progress:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sizes = sizes.to(device, non_blocking=True)
            values = trainer.train_step(x, labels, sizes, epoch >= adv_start)
            tracker.update(values, x.shape[0])
            progress.set_postfix(total=f"{values['total']:.3f}", recon=f"{values['recon']:.3f}")
        means = tracker.means()
        row = {"epoch": epoch + 1, **{key: means.get(key, float("nan")) for key in LOG_KEYS}}
        save_checkpoint(trainer.state(epoch + 1), last_path)
        save_checkpoint({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "model_cfg": cfg["model"], "epoch": epoch + 1}, os.path.join(out_dir, "generator.pt"))
        logger.log(row)
        print(row)
        if (epoch + 1) % int(tcfg["sample_every"]) == 0 or epoch + 1 == epochs:
            save_samples(trainer, fixed, os.path.join(out_dir, "samples", f"epoch_{epoch + 1:03d}.png"))
    save_checkpoint({"disc": disc.state_dict()}, os.path.join(out_dir, "disc.pt"))


if __name__ == "__main__":
    main()
