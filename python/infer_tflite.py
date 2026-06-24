#!/usr/bin/env python3
"""Voicute Wake Word — TFLite inference (32-bit Raspberry Pi / ARM)

Usage:
    # 1. Convert ONNX → TFLite first:
    python convert_to_tflite.py --model-dir ../models/ --output-dir ../models/

    # 2. Run inference:
    pip install tflite-runtime numpy sounddevice
    python infer_tflite.py --model-dir ../models/
    python infer_tflite.py --model-dir ../models/ --threshold 0.6

Why TFLite?
    ONNX Runtime has no prebuilt 32-bit ARM (armv7l) wheel.
    TFLite Runtime has first-class 32-bit ARM support via official wheels.
    Same ONNX models → onnx2tf → TFLite → runs on Pi 3B / Zero / Zero 2W.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import sounddevice as sd

# ── TFLite Runtime import (tflite-runtime first, fallback to TF) ──

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_LIB = "tflite-runtime"
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_LIB = "tensorflow"
    except ImportError:
        print("ERROR: Install tflite-runtime or tensorflow:")
        print("  pip install tflite-runtime   # recommended for Pi 32-bit")
        print("  pip install tensorflow        # fallback (larger)")
        sys.exit(1)

SAMPLE_RATE = 16000
MEL_HOP_SAMPLES = 160
EMB_WINDOW = 76
EMB_STEP = 8
N_MELS = 32


# ── Model loading ──

def load_tflite(path: str):
    """Load a TFLite model and allocate tensors. Accepts .tflite or .onnx path."""
    if path.endswith(".onnx"):
        alt = path.rsplit(".onnx", 1)[0] + ".tflite"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"TFLite model not found: {alt}\n"
                f"  Run: python convert_to_tflite.py --model-dir <dir>")
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def get_input_shape(interpreter):
    """Return the shape tuple of the first input tensor."""
    return tuple(interpreter.get_input_details()[0]["shape"])


def load_models(model_dir: str):
    """Load TFLite models and config from model_dir."""
    mel_path = os.path.join(model_dir, "melspectrogram.onnx")
    emb_path = os.path.join(model_dir, "embedding_model.onnx")
    info_path = os.path.join(model_dir, "model_info.json")

    # Auto-detect .tflite or fall back to .onnx for conversion error
    for label, path in [("melspectrogram", mel_path), ("embedding", emb_path)]:
        tfl_path = path.rsplit(".onnx", 1)[0] + ".tflite"
        if not os.path.exists(tfl_path):
            raise FileNotFoundError(
                f"TFLite model not found: {tfl_path}\n"
                f"  Run: pip install onnx2tf && python linux/convert_to_tflite.py "
                f"--model-dir {os.path.abspath(model_dir)}/")

    mel_interp = load_tflite(mel_path)
    emb_interp = load_tflite(emb_path)

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    models = []
    max_ef = 0
    multi_class = False
    n_classes = 0
    wake_word_names = []

    if info.get("model_type") == "multi":
        multi_class = True
        n_classes = info["n_classes"]
        wake_word_names = info["wake_words"]
        ef = info["emb_frames"]
        max_ef = ef
        mc_path = os.path.join(model_dir, info["model_file"])
        mc_interp = load_tflite(mc_path)
        models.append({
            "name": "multi",
            "interpreter": mc_interp,
            "emb_frames": ef,
        })

    elif info.get("multi_model") and info.get("models"):
        for m in info["models"]:
            cls_path = os.path.join(model_dir, m["model_file"])
            interp = load_tflite(cls_path)
            ef = m["emb_frames"]
            models.append({
                "name": m["wake_word"],
                "interpreter": interp,
                "emb_frames": ef,
            })
            max_ef = max(max_ef, ef)
    else:
        cls_path = os.path.join(model_dir, info["model_file"])
        interp = load_tflite(cls_path)
        ef = info.get("emb_frames", 16)
        models.append({
            "name": info["wake_word"],
            "interpreter": interp,
            "emb_frames": ef,
        })
        max_ef = ef

    # Calculate audio buffer size from mel model's expected input length
    mel_shape = get_input_shape(mel_interp)
    mel_audio_len = mel_shape[1]  # [1, N]

    print(f"TFLite models loaded ({TFLITE_LIB}), maxEmbFrames={max_ef}")
    for m in models:
        print(f"  {m['name']}: {m['emb_frames']} frames")
    return (mel_interp, emb_interp, models, max_ef,
            multi_class, n_classes, wake_word_names, mel_audio_len)


# ── Slow mel spectrogram (NumPy fallback — NO librosa dependency) ──

def _slow_mel(audio: np.ndarray, n_mels: int, hop: int, win: int) -> np.ndarray:
    """NumPy mel spectrogram — works on any platform, zero dependencies."""
    n_fft = 2 ** int(np.ceil(np.log2(win)))
    n_frames = (len(audio) - win) // hop + 1
    if n_frames < 1:
        return np.zeros((1, 32), dtype=np.float32)
    mel = np.zeros((n_frames, n_mels), dtype=np.float32)
    hann = np.hanning(win).astype(np.float32)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    mel_freqs = 2595.0 * np.log10(1.0 + fft_freqs / 700.0)
    mel_bins = np.linspace(
        2595.0 * np.log10(1.0 + 0 / 700.0),
        2595.0 * np.log10(1.0 + (SAMPLE_RATE / 2) / 700.0),
        n_mels + 2)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + win].astype(np.float32)
        frame = frame * hann
        mag = np.abs(np.fft.rfft(frame, n=n_fft))
        for j in range(n_mels):
            lower = mel_bins[j]
            center = mel_bins[j + 1]
            upper = mel_bins[j + 2]
            weights = np.maximum(0, np.minimum(
                (mel_freqs - lower) / (center - lower + 1e-10),
                (upper - mel_freqs) / (upper - center + 1e-10)))
            mel[i, j] = np.dot(mag, weights) / (center - lower + 1e-10)
    mel = np.log(np.maximum(mel, 1e-6))
    return mel


# ── Inference ──

def detect(audio: np.ndarray, mel_interp, emb_interp, models,
           max_ef, multi_class, n_classes, wake_word_names,
           mel_audio_len):
    """Run TFLite wake word detection on float32 audio at 16kHz."""
    # Pad or trim audio to match mel model's expected input length
    audio = audio.astype(np.float32).reshape(-1)
    if len(audio) < mel_audio_len:
        audio = np.pad(audio, (0, mel_audio_len - len(audio)))
    audio = audio[:mel_audio_len]

    # ── Mel spectrogram ──

    mel_in_details = mel_interp.get_input_details()[0]
    mel_out_details = mel_interp.get_output_details()[0]
    mel_interp.set_tensor(mel_in_details["index"], audio.reshape(1, -1))
    mel_interp.invoke()
    mel_out = mel_interp.get_tensor(mel_out_details["index"])

    # If TFLite mel model produces correct output, skip NumPy fallback
    frames = mel_out.shape[2] if mel_out.ndim >= 3 else mel_out.shape[1]
    if frames < EMB_WINDOW:
        return None

    # TFLite mel model might output different shape — normalize to [frames, 32]
    if mel_out.ndim >= 3:
        mel2d = mel_out[0, 0]  # [frames, 32]
    else:
        mel2d = mel_out[0]     # [frames, 32]

    # Apply the same transform as training pipeline: x/10 + 2
    # (only if not already baked into the TFLite model)
    mel2d = mel2d / 10.0 + 2.0

    # ── Embedding batch ──

    start_frame = max(0, frames - EMB_WINDOW - (max_ef - 1) * EMB_STEP)
    emb_shape = get_input_shape(emb_interp)  # [N, 76, 32, 1]
    emb_batch_size = emb_shape[0]  # how many windows the model expects
    if emb_batch_size < max_ef:
        print(f"WARNING: TFLite embedding expects {emb_batch_size} windows, "
              f"need {max_ef} — reconvert with correct batch size")

    emb_batch = np.zeros((max_ef, EMB_WINDOW, 32, 1), dtype=np.float32)
    for w in range(max_ef):
        off = min(start_frame + w * EMB_STEP, frames - EMB_WINDOW)
        for f in range(EMB_WINDOW):
            for m in range(32):
                emb_batch[w, f, m, 0] = mel2d[off + f, m]

    # Pad or slice to match model expected batch size
    if max_ef < emb_batch_size:
        # Pad with repeats of last frame
        padded = np.zeros((emb_batch_size, EMB_WINDOW, 32, 1), dtype=np.float32)
        padded[:max_ef] = emb_batch
        emb_batch = padded

    emb_in_details = emb_interp.get_input_details()[0]
    emb_out_details = emb_interp.get_output_details()[0]
    emb_interp.set_tensor(emb_in_details["index"], emb_batch)
    emb_interp.invoke()
    embeddings = emb_interp.get_tensor(emb_out_details["index"])
    # Expected output: [batch, 1, 1, 96] or similar

    # Flatten to [batch, 96]
    emb_flat = embeddings.reshape(embeddings.shape[0], -1)[:max_ef]

    # ── Multi-class model ──

    if multi_class:
        flat_dim = max_ef * 96
        multi_in = emb_flat.reshape(1, flat_dim)

        mc_in_details = models[0]["interpreter"].get_input_details()[0]
        mc_out_details = models[0]["interpreter"].get_output_details()[0]
        models[0]["interpreter"].set_tensor(mc_in_details["index"], multi_in.astype(np.float32))
        models[0]["interpreter"].invoke()
        probs = models[0]["interpreter"].get_tensor(mc_out_details["index"])[0]

        K = n_classes - 1  # exclude background
        best_idx = int(np.argmax(probs[:K]))
        return {
            "wakeWord": wake_word_names[best_idx],
            "probability": float(probs[best_idx]),
            "background": float(probs[K]),
        }

    # ── Binary classifiers ──

    K = len(models)
    sigmoid_probs = np.zeros(K, dtype=np.float32)

    for i, model in enumerate(models):
        slice_start = max_ef - model["emb_frames"]
        wake_in = emb_flat[slice_start:slice_start + model["emb_frames"]]
        wake_in = wake_in.reshape(1, model["emb_frames"], 96)

        cls_in_details = model["interpreter"].get_input_details()[0]
        cls_out_details = model["interpreter"].get_output_details()[0]
        model["interpreter"].set_tensor(
            cls_in_details["index"], wake_in.astype(np.float32))
        model["interpreter"].invoke()
        out = model["interpreter"].get_tensor(cls_out_details["index"])
        sigmoid_probs[i] = float(out.flatten()[0])

    # Sigmoid → Softmax with background class
    logits = np.zeros(K + 1, dtype=np.float32)
    for i in range(K):
        p = np.clip(sigmoid_probs[i], 1e-6, 1 - 1e-6)
        logits[i] = np.log(p / (1 - p))
    logits[K] = 0  # background

    logits -= logits.max()
    softmax = np.exp(logits)
    softmax /= softmax.sum()

    best_idx = int(np.argmax(softmax[:K]))
    return {
        "wakeWord": models[best_idx]["name"],
        "probability": float(softmax[best_idx]),
        "background": float(softmax[K]),
    }


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Voicute Wake Word — TFLite (32-bit Pi)")
    parser.add_argument("--model-dir", default="../models/",
                        help="Model directory with .tflite files + model_info.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (softmax probability)")
    parser.add_argument("--cooldown", type=float, default=3.5,
                        help="Cooldown between detections (seconds)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio input devices")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    (mel_interp, emb_interp, models, max_ef,
     multi_class, n_classes, wake_word_names,
     mel_audio_len) = load_models(args.model_dir)

    audio_buffer = deque(maxlen=SAMPLE_RATE * 4)
    last_detect = 0
    detections = 0

    def callback(indata, frames, time_info, status):
        nonlocal last_detect, detections
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        audio_buffer.extend(indata[:, 0])
        if len(audio_buffer) < mel_audio_len:
            return

        audio = np.array(list(audio_buffer)[-mel_audio_len:], dtype=np.float32)
        result = detect(
            audio, mel_interp, emb_interp, models, max_ef,
            multi_class, n_classes, wake_word_names, mel_audio_len)

        if result and result["probability"] > args.threshold:
            now = time.time()
            if now - last_detect > args.cooldown:
                last_detect = now
                detections += 1
                print(f"\n>>> [{detections}] {result['wakeWord']} "
                      f"({result['probability']:.2%})  bg={result['background']:.2%}")

    words_display = " | ".join(wake_word_names) if wake_word_names else \
        " | ".join(m["name"] for m in models)
    print(f"Listening for: {words_display}")
    print(f"  threshold={args.threshold} cooldown={args.cooldown}s "
          f"mel_len={mel_audio_len}")
    print("  (Ctrl+C to stop)\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            callback=callback, dtype="float32",
                            blocksize=4096):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\nStopped. Detections: {detections}")
    except Exception as e:
        print(f"Error: {e}")
        print("  Hint: check microphones with --list-devices")


if __name__ == "__main__":
    main()
