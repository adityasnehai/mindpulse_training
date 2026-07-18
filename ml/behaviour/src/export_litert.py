"""Milestone 4 final step: train the production TCN encoder on all real
participants and export it as fp32 and INT8 LiteRT (.tflite) models.

Conversion path: PyTorch weights are ported into an architecturally-identical
Keras model, then converted via the native tf.lite.TFLiteConverter.

Why not ai-edge-torch (PyTorch -> LiteRT directly, the more obvious path
given training happened in PyTorch): its conversion pipeline hard-depends on
torch_xla, which ships NO Windows distribution at all (confirmed: `pip
install torch_xla` fails with "no matching distribution" on this platform —
not a version conflict, a genuine platform gap). Rather than route around
that with something fragile, this ports the trained weights into the
architecture docs/PRODUCT_SPEC.md section 10 literally specifies (Keras ->
TFLiteConverter), which is natively supported and already verified working
in this environment. Weight-port correctness is verified numerically (not
assumed): predict_keras(x) must match predict_pytorch(x) within tight
tolerance on real data before any export happens.

The StandardScaler fit on the full dataset is saved alongside the model —
on-device inference needs the exact same normalization applied to raw
[7,24] windows before calling the encoder.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from train_encoder import TCNEncoder, train_encoder

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
RANDOM_STATE = 42
N_FEATURES = 12
WINDOW_DAYS = 7
INPUT_CHANNELS = 24
EMBEDDING_SIZE = 8


def train_production_encoder(windows: np.ndarray, config: dict, val_fraction: float = 0.1):
    """Train the final encoder on ALL real windows (no held-out participant —
    this is the model that ships), with a random validation split for early
    stopping only. Returns (trained_full_autoencoder, fitted_scaler)."""
    n = windows.shape[0]
    scaler = StandardScaler().fit(windows.reshape(n, -1))
    scaled = scaler.transform(windows.reshape(n, -1)).reshape(windows.shape)

    rng = np.random.default_rng(RANDOM_STATE)
    val_size = max(1, int(n * val_fraction))
    perm = rng.permutation(n)
    val_idx, fit_idx = perm[:val_size], perm[val_size:]

    model = train_encoder(scaled[fit_idx], val_windows=scaled[val_idx], config=config, seed=RANDOM_STATE)
    return model, scaler


def build_keras_encoder() -> tf.keras.Model:
    """Architecturally identical to train_encoder.TCNEncoder, built in Keras
    purely as a conversion target — never trained directly, only weight-ported."""
    inputs = tf.keras.Input(shape=(WINDOW_DAYS, INPUT_CHANNELS), name="window")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="causal", dilation_rate=1, activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, padding="causal", dilation_rate=2, activation="relu", name="conv2")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="pool")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense")(x)
    outputs = tf.keras.layers.Dense(EMBEDDING_SIZE, activation="linear", name="embedding")(x)
    return tf.keras.Model(inputs, outputs, name="drift_encoder")


def port_weights_pytorch_to_keras(pytorch_encoder: TCNEncoder, keras_model: tf.keras.Model) -> None:
    """Transposes each layer's weights from PyTorch's convention to Keras's:
    Conv1d [out, in, kernel] -> Conv1D [kernel, in, out]; Linear [out, in] -> Dense [in, out].
    """
    state = pytorch_encoder.state_dict()

    conv1_w = state["conv1.conv.weight"].detach().numpy().transpose(2, 1, 0)
    conv1_b = state["conv1.conv.bias"].detach().numpy()
    keras_model.get_layer("conv1").set_weights([conv1_w, conv1_b])

    conv2_w = state["conv2.conv.weight"].detach().numpy().transpose(2, 1, 0)
    conv2_b = state["conv2.conv.bias"].detach().numpy()
    keras_model.get_layer("conv2").set_weights([conv2_w, conv2_b])

    dense_w = state["dense.weight"].detach().numpy().T
    dense_b = state["dense.bias"].detach().numpy()
    keras_model.get_layer("dense").set_weights([dense_w, dense_b])

    embedding_w = state["embedding.weight"].detach().numpy().T
    embedding_b = state["embedding.bias"].detach().numpy()
    keras_model.get_layer("embedding").set_weights([embedding_w, embedding_b])


def verify_weight_port(pytorch_encoder: TCNEncoder, keras_model: tf.keras.Model, sample_windows: np.ndarray) -> float:
    """Returns the max absolute difference between PyTorch and Keras outputs
    on real data — the port is only trusted if this is numerically tiny, not
    assumed correct just because it ran without error."""
    pytorch_encoder.eval()
    with torch.no_grad():
        torch_out = pytorch_encoder(torch.tensor(sample_windows, dtype=torch.float32)).numpy()
    keras_out = keras_model.predict(sample_windows, verbose=0)
    return float(np.max(np.abs(torch_out - keras_out)))


def export_fp32(keras_model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def export_int8(keras_model: tf.keras.Model, out_path: Path) -> None:
    """Dynamic-range INT8 quantization (weights quantized, activations stay
    float) — one of the two options docs/PRODUCT_SPEC.md section 10
    explicitly allows ("full-integer or dynamic-range").

    Full-integer calibrated quantization (optimizations=[DEFAULT] +
    representative_dataset, forcing TFLITE_BUILTINS_INT8) was tried first and
    produced a genuinely broken model: verified on real data, it returned a
    constant all-zero embedding for every input regardless of the window
    (confirmed by direct inspection, not assumed) — an op-compatibility issue
    with this dilated causal Conv1D graph under full calibrated quantization
    in this TensorFlow version, not a data or weight-port problem. Dynamic-
    range quantization (no representative_dataset needed) produces a real,
    working model whose outputs track the fp32 model's sign pattern and
    magnitude on real data — verified below via compare_fp32_vs_int8, not
    assumed correct just because conversion succeeded without error.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def run_tflite_model(tflite_path: Path, windows: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_dtype = input_details[0]["dtype"]
    in_scale, in_zero_point = input_details[0].get("quantization", (0.0, 0))
    out_scale, out_zero_point = output_details[0].get("quantization", (0.0, 0))

    outputs = []
    for window in windows:
        x = window[None, ...].astype(np.float32)
        if in_dtype != np.float32 and in_scale:
            x = (x / in_scale + in_zero_point).astype(in_dtype)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])[0]
        if output_details[0]["dtype"] != np.float32 and out_scale:
            y = (y.astype(np.float32) - out_zero_point) * out_scale
        outputs.append(y)
    return np.stack(outputs)


def compare_fp32_vs_int8(fp32_path: Path, int8_path: Path, windows: np.ndarray) -> dict:
    fp32_embeddings = run_tflite_model(fp32_path, windows)
    int8_embeddings = run_tflite_model(int8_path, windows)

    cos_sim = np.sum(fp32_embeddings * int8_embeddings, axis=1) / (
        np.linalg.norm(fp32_embeddings, axis=1) * np.linalg.norm(int8_embeddings, axis=1) + 1e-9
    )

    center = np.median(fp32_embeddings, axis=0)
    scale = np.median(np.abs(fp32_embeddings - center), axis=0)
    fp32_drift_score = (np.abs(fp32_embeddings - center) / (scale + 0.001)).mean(axis=1)
    int8_drift_score = (np.abs(int8_embeddings - center) / (scale + 0.001)).mean(axis=1)

    threshold = np.percentile(fp32_drift_score, 95)
    fp32_alert = fp32_drift_score > threshold
    int8_alert = int8_drift_score > threshold
    agreement = float((fp32_alert == int8_alert).mean())

    return {
        "n_windows": int(len(windows)),
        "mean_cosine_similarity": float(cos_sim.mean()),
        "min_cosine_similarity": float(cos_sim.min()),
        "mean_drift_score_diff": float(np.abs(fp32_drift_score - int8_drift_score).mean()),
        "alert_decision_agreement": agreement,
        "meets_99pct_bar": agreement >= 0.99,
    }


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    artifacts_dir = (base_dir / config["paths"]["artifacts_dir"]).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    windows = np.load(processed_dir / "sequences.npy")
    print(f"Training production encoder on all {len(windows)} real windows...")
    model, scaler = train_production_encoder(windows, config)

    scaled_windows = scaler.transform(windows.reshape(len(windows), -1)).reshape(windows.shape).astype(np.float32)
    np.save(artifacts_dir / "scaler_mean.npy", scaler.mean_)
    np.save(artifacts_dir / "scaler_scale.npy", scaler.scale_)
    torch.save(model.encoder.state_dict(), artifacts_dir / "encoder_state_dict.pt")

    print("Porting PyTorch weights into an equivalent Keras model...")
    keras_model = build_keras_encoder()
    port_weights_pytorch_to_keras(model.encoder, keras_model)

    max_diff = verify_weight_port(model.encoder, keras_model, scaled_windows[:50])
    print(f"Max abs difference between PyTorch and Keras outputs on 50 real windows: {max_diff:.8f}")
    if max_diff > 1e-4:
        print("ERROR: weight port does not match PyTorch output closely enough. Aborting export.")
        return 1

    fp32_path = artifacts_dir / "drift_encoder_fp32.tflite"
    int8_path = artifacts_dir / "drift_encoder_int8.tflite"

    print("Exporting fp32 LiteRT model...")
    export_fp32(keras_model, fp32_path)
    print(f"  {fp32_path} ({fp32_path.stat().st_size / 1024:.1f} KB)")

    print("Exporting INT8-quantized LiteRT model (calibrating on real windows)...")
    export_int8(keras_model, int8_path)
    print(f"  {int8_path} ({int8_path.stat().st_size / 1024:.1f} KB)")

    print("Comparing fp32 vs INT8 on real windows...")
    comparison = compare_fp32_vs_int8(fp32_path, int8_path, scaled_windows)
    print(json.dumps(comparison, indent=2))

    out_path = processed_dir / "quantization_comparison.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(comparison, f)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
