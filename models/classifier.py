# https://github.com/facebookresearch/mae

import torch.nn as nn


class EncoderClassifier(nn.Module):
    # Encoder followed by the SiLU -> Dropout -> Linear classification head
    def __init__(self, encoder, num_classes, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.SiLU(), nn.Dropout(float(dropout)), nn.Linear(encoder.latent_dim, num_classes))

    # Class logits
    def forward(self, x):
        return self.head(self.encoder(x))

    # Latent features and class logits from a single encoder pass
    def features_and_logits(self, x):
        z = self.encoder(x)
        return z, self.head(z)
