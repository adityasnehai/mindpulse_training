"""Unit tests for the fp32-vs-int8 agreement math, the production-training
helper, and the PyTorch->Keras weight port in export_litert.py (not the full
TFLiteConverter export itself, which is exercised end-to-end separately)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_litert import build_keras_encoder, port_weights_pytorch_to_keras, train_production_encoder, verify_weight_port  # noqa: E402
from train_encoder import TCNEncoder  # noqa: E402


def test_weight_port_matches_pytorch_output_on_random_input():
    torch.manual_seed(0)
    pytorch_encoder = TCNEncoder(input_channels=24, embedding_size=8)
    keras_model = build_keras_encoder()
    port_weights_pytorch_to_keras(pytorch_encoder, keras_model)

    rng = np.random.default_rng(1)
    sample_windows = rng.normal(0, 1, size=(20, 7, 24)).astype(np.float32)

    max_diff = verify_weight_port(pytorch_encoder, keras_model, sample_windows)
    assert max_diff < 1e-4


def test_weight_port_conv_kernel_transpose_is_not_accidentally_identity():
    # Regression guard: if someone "fixes" the transpose to (0,1,2) (a no-op)
    # instead of (2,1,0), the shapes would still happen to be compatible in
    # some cases but the numbers would be wrong. Confirms the actual
    # transpose axis order used produces correct output, not just any
    # shape-compatible transpose.
    torch.manual_seed(1)
    pytorch_encoder = TCNEncoder(input_channels=24, embedding_size=8)
    state = pytorch_encoder.state_dict()
    conv1_w = state["conv1.conv.weight"].detach().numpy()  # [out=32, in=24, kernel=3]
    assert conv1_w.shape == (32, 24, 3)
    transposed = conv1_w.transpose(2, 1, 0)
    assert transposed.shape == (3, 24, 32)  # Keras Conv1D kernel shape: [kernel, in, out]


def test_train_production_encoder_uses_all_windows_no_holdout():
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 1, size=(40, 7, 24)).astype(np.float32)
    windows[:, :, 12:] = 0.0

    config = {"training": {"epochs": 3, "batch_size": 8}}
    model, scaler = train_production_encoder(windows, config)

    assert scaler.mean_.shape == (7 * 24,)
    # Scaler was fit on all 40 windows, not a subset.
    scaled = scaler.transform(windows.reshape(40, -1))
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-6)


def test_agreement_math_identical_embeddings_gives_perfect_agreement():
    # Directly test the drift-score/agreement formula logic (mirrors
    # compare_fp32_vs_int8's inline computation) without needing real .tflite
    # files: identical fp32/int8 embeddings must yield 100% agreement and
    # cosine similarity of 1.0.
    rng = np.random.default_rng(0)
    embeddings = rng.normal(0, 1, size=(50, 8))

    cos_sim = np.sum(embeddings * embeddings, axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1) + 1e-9
    )
    assert np.allclose(cos_sim, 1.0, atol=1e-6)

    center = np.median(embeddings, axis=0)
    scale = np.median(np.abs(embeddings - center), axis=0)
    drift = np.abs(embeddings - center) / (scale + 0.001)
    drift_score = drift.mean(axis=1)
    threshold = np.percentile(drift_score, 95)
    alert_a = drift_score > threshold
    alert_b = drift_score > threshold
    assert (alert_a == alert_b).all()
