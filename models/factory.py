from models.classifier import EncoderClassifier
from models.decoder import ConditionalDecoder
from models.discriminator import ProjectionDiscriminator
from models.vit_encoder import ViTEncoder
from utils.train_utils import load_checkpoint, requires_grad


# ViT encoder built from the shared `model` config
def build_encoder(cfg):
    m = cfg["model"]
    return ViTEncoder(
        image_size=int(cfg["data"]["image_size"]),
        patch_size=int(m["patch_size"]),
        embed_dim=int(m["embed_dim"]),
        depth=int(m["depth"]),
        num_heads=int(m["num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        latent_dim=int(m["latent_dim"]),
        drop_path=float(m.get("drop_path", 0.0)),
    )


# Class-conditional decoder built from the shared `model` config
def build_decoder(cfg):
    m = cfg["model"]
    return ConditionalDecoder(
        latent_dim=int(m["latent_dim"]),
        num_classes=len(cfg["data"]["classes"]),
        image_size=int(cfg["data"]["image_size"]),
        channels=[int(c) for c in m["decoder_channels"]],
        cond_dim=int(m["cond_dim"]),
        class_dim=int(m["class_dim"]),
        size_dim=int(m["size_dim"]),
        grad_checkpoint=bool(m.get("decoder_checkpoint", False)),
    )


def build_discriminator(cfg):
    d = cfg["disc"]
    return ProjectionDiscriminator(
        num_classes=len(cfg["data"]["classes"]),
        image_size=int(cfg["data"]["image_size"]),
        channels=[int(c) for c in d["channels"]],
        mbstd_group=int(d["mbstd_group"]),
        proj_dim=int(d["proj_dim"]),
    )


# Rebuild the stage-2 classifier from its checkpoint
def load_frozen_classifier(cfg, path, device):
    state = load_checkpoint(path)
    model = EncoderClassifier(build_encoder(cfg), len(cfg["data"]["classes"]), state.get("dropout", 0.0))
    model.encoder.load_state_dict(state["encoder"])
    model.head.load_state_dict(state["head"])
    model.to(device).eval()
    requires_grad(model, False)
    return model
