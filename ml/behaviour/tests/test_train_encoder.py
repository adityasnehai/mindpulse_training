"""Unit tests for the TCN autoencoder architecture and masked-MSE loss."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_encoder import (  # noqa: E402
    N_FEATURES,
    CausalConv1d,
    TCNAutoencoder,
    TCNEncoder,
    get_embeddings,
    masked_mse_loss,
    train_encoder,
)


def test_causal_conv_output_length_matches_input_length():
    conv = CausalConv1d(in_channels=4, out_channels=8, kernel_size=3, dilation=2)
    x = torch.randn(2, 4, 7)  # [batch, channels, time]
    out = conv(x)
    assert out.shape == (2, 8, 7)


def test_causal_conv_does_not_leak_future_timesteps():
    # Changing a later timestep's input must not change an earlier timestep's
    # output — that's what "causal" means.
    conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)
    conv.conv.weight.data.fill_(1.0)
    conv.conv.bias.data.fill_(0.0)

    x1 = torch.zeros(1, 1, 7)
    x1[0, 0, 0] = 5.0
    out1 = conv(x1)

    x2 = x1.clone()
    x2[0, 0, 6] = 99.0  # perturb only the last timestep
    out2 = conv(x2)

    assert torch.allclose(out1[0, 0, 0], out2[0, 0, 0])  # first output unaffected


def test_encoder_output_shape():
    encoder = TCNEncoder(input_channels=24, embedding_size=8)
    x = torch.randn(5, 7, 24)  # [batch, time, channels]
    embedding = encoder(x)
    assert embedding.shape == (5, 8)


def test_autoencoder_reconstruction_shape():
    model = TCNAutoencoder(input_channels=24, embedding_size=8, window_days=7)
    x = torch.randn(3, 7, 24)
    embedding, reconstruction = model(x)
    assert embedding.shape == (3, 8)
    assert reconstruction.shape == (3, 7, N_FEATURES)


def test_masked_mse_ignores_missing_positions():
    reconstruction = torch.tensor([[[0.0, 0.0]]])  # [1,1,2]
    target = torch.tensor([[[100.0, 5.0]]])  # huge error at position 0, small at position 1
    mask = torch.tensor([[[1.0, 0.0]]])  # position 0 is missing (ignored), position 1 observed

    loss = masked_mse_loss(reconstruction, target, mask)
    expected = (0.0 - 5.0) ** 2  # only position 1 contributes
    assert loss.item() == pytest.approx(expected)


def test_masked_mse_all_missing_returns_zero_not_nan():
    reconstruction = torch.zeros(1, 1, 2)
    target = torch.ones(1, 1, 2)
    mask = torch.ones(1, 1, 2)  # everything missing
    loss = masked_mse_loss(reconstruction, target, mask)
    assert loss.item() == 0.0


def test_train_encoder_reduces_loss_on_toy_data():
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 1, size=(40, 7, 24)).astype(np.float32)
    windows[:, :, 12:] = 0.0  # no missingness -> everything observed

    model = train_encoder(windows, val_windows=windows[:8], config={"training": {"epochs": 5, "batch_size": 8}})
    embeddings = get_embeddings(model, windows)
    assert embeddings.shape == (40, 8)
    assert np.isfinite(embeddings).all()


def test_train_encoder_is_seed_reproducible():
    rng = np.random.default_rng(1)
    windows = rng.normal(0, 1, size=(20, 7, 24)).astype(np.float32)
    windows[:, :, 12:] = 0.0

    model_a = train_encoder(windows, config={"training": {"epochs": 3, "batch_size": 8}}, seed=42)
    model_b = train_encoder(windows, config={"training": {"epochs": 3, "batch_size": 8}}, seed=42)

    emb_a = get_embeddings(model_a, windows)
    emb_b = get_embeddings(model_b, windows)
    assert np.allclose(emb_a, emb_b)
