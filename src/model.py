import torch
import torch.nn as nn
from typing import Tuple

"""
model.py

Defines the EEGSeizureDetector Transformer classifier.
"""


class EEGSeizureDetector(nn.Module):
    """Transformer-based EEG seizure detector.

    Inputs:
        x: Tensor of shape (batch_size, n_channels, seq_len).

    Outputs:
        logits: Tensor of shape (batch_size, 1) (before sigmoid).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        classifier_dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_dropout = nn.Dropout(classifier_dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, n_channels, seq_len).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        b, c, t = x.shape
        x = x.view(b * c, t, 1)
        x = self.input_proj(x)  # (b*c, t, d_model)

        encoding = self.encoder(x)  # (b*c, t, d_model)
        pooled = encoding.mean(dim=1)  # (b*c, d_model)

        pooled = pooled.view(b, c, -1).mean(dim=1)  # (b, d_model)

        pooled = self.cls_dropout(pooled)
        logits = self.cls_head(pooled)  # (b, 1)
        return logits
