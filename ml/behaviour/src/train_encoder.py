"""Milestone 4: temporal convolutional (TCN) autoencoder — PyTorch.

Architecture per docs/PRODUCT_SPEC.md section 8.2, adapted from the Keras spec
to PyTorch (Conv1d expects [batch, channels, length], not [batch, length,
channels] — inputs are permuted accordingly; Keras' padding="causal" has no
direct PyTorch equivalent so it is implemented explicitly as a left-only pad
of (kernel_size-1)*dilation before each conv):

Encoder: Input [7, 24] -> Conv1d(24->32, k=3, dilation=1, causal, relu)
    -> Conv1d(32->16, k=3, dilation=2, causal, relu) -> GlobalAveragePool(time)
    -> Dense(16, relu) -> Embedding Dense(8, linear)
Decoder (training only): Dense(8 -> 7*12) -> reshape [7, 12]

Missingness-awareness: the 12 missing-indicator channels are part of the
24-channel input itself (channels 12:24), so the encoder can learn to weight
observed vs. imputed (zero-filled) values — this is the "masking /
missing-value-aware input" the spec names, implemented as mask-as-extra-
channel rather than a separate Keras Masking layer (which has no direct
PyTorch equivalent for per-channel masking in a conv stack).

Loss: masked MSE — reconstruction error is computed only at (day, feature)
positions where missing_mask == 0 (a real observed value), matching section
8.2's "loss = error calculated only where original feature values exist."
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
N_FEATURES = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CausalConv1d(nn.Module):
    """Conv1d with left-only ("causal") padding, matching Keras padding="causal"."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_channels: int = 24, embedding_size: int = 8):
        super().__init__()
        self.conv1 = CausalConv1d(input_channels, 32, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(32, 16, kernel_size=3, dilation=2)
        self.dense = nn.Linear(16, 16)
        self.embedding = nn.Linear(16, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time=7, channels=24] -> [batch, channels, time] for Conv1d
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=2)  # GlobalAveragePooling1D over the time axis
        x = F.relu(self.dense(x))
        return self.embedding(x)  # linear, no activation


class TCNAutoencoder(nn.Module):
    """Encoder + training-only decoder head. Only TCNEncoder is exported to LiteRT."""

    def __init__(self, input_channels: int = 24, embedding_size: int = 8, window_days: int = 7):
        super().__init__()
        self.encoder = TCNEncoder(input_channels, embedding_size)
        self.window_days = window_days
        self.decoder_dense = nn.Linear(embedding_size, window_days * N_FEATURES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder_dense(embedding).view(-1, self.window_days, N_FEATURES)
        return embedding, reconstruction


def masked_mse_loss(reconstruction: torch.Tensor, target_features: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error computed only where missing_mask == 0 (value was
    really observed). If a batch has zero observed values (all missing),
    returns 0 to avoid a divide-by-zero — such a batch contributes no signal."""
    observed = 1.0 - missing_mask
    squared_error = (reconstruction - target_features) ** 2 * observed
    denom = observed.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=reconstruction.device, requires_grad=True)
    return squared_error.sum() / denom


def train_encoder(
    train_windows: np.ndarray,
    val_windows: np.ndarray | None = None,
    config: dict | None = None,
    seed: int = 42,
) -> TCNAutoencoder:
    """Train one TCN autoencoder on the given windows ([N, 7, 24], already
    scaled/normalized by the caller). Returns the trained model (best
    validation-loss checkpoint if val_windows is given, else final epoch)."""
    set_seed(seed)
    cfg = (config or {}).get("training", {})
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 150)
    lr = cfg.get("learning_rate", 0.001)
    early_stopping_patience = cfg.get("early_stopping_patience", 15)
    reduce_lr_patience = cfg.get("reduce_lr_patience", 7)
    min_lr = cfg.get("minimum_learning_rate", 0.00001)

    model = TCNAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=reduce_lr_patience, min_lr=min_lr
    )

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    train_features = train_tensor[:, :, :N_FEATURES]
    train_mask = train_tensor[:, :, N_FEATURES:]

    has_val = val_windows is not None and len(val_windows) > 0
    if has_val:
        val_tensor = torch.tensor(val_windows, dtype=torch.float32)
        val_features = val_tensor[:, :, :N_FEATURES]
        val_mask = val_tensor[:, :, N_FEATURES:]

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n = train_tensor.shape[0]

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_x = train_tensor[idx]
            batch_target = train_features[idx]
            batch_mask = train_mask[idx]

            optimizer.zero_grad()
            _, reconstruction = model(batch_x)
            loss = masked_mse_loss(reconstruction, batch_target, batch_mask)
            loss.backward()
            # Without this, training on the real (StandardScaler-normalized but
            # still heterogeneous) feature set diverges: embedding magnitudes
            # were observed reaching ~1e8 after full training, confirmed via
            # the fp32-vs-Keras numeric verification in export_litert.py, not
            # a hypothetical concern.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        if has_val:
            model.eval()
            with torch.no_grad():
                _, val_reconstruction = model(val_tensor)
                val_loss = masked_mse_loss(val_reconstruction, val_features, val_mask).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        else:
            scheduler.step(epoch_loss)

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
    return model


def get_embeddings(model: TCNAutoencoder, windows: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        embeddings = model.encoder(x)
    return embeddings.numpy()
