#!/usr/bin/env python3
"""Voicute Wake Word — Linux inference (ONNX Runtime)

Usage:
    pip install onnxruntime numpy sounddevice
    python infer.py --model-dir ../models/
    python infer.py --model-dir ../models/ --threshold 0.6

Place your trained .onnx model(s) + model_info.json in the model directory,
alongside melspectrogram.onnx and embedding_model.onnx.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import sounddevice as sd

SAMPLE_RATE = 16000
MEL_HOP_SAMPLES = 160
EMB_WINDOW = 76
EMB_STEP = 8

# ── Model loading ──

def load_models(model_dir: str):
    mel_path = os.path.join(model_dir, "melspectrogram.onnx")
    emb_path = os.path.join(model_dir, "embedding_model.onnx")
    info_path = os.path.join(model_dir, "model_info.json")

    mel_session = ort.InferenceSession(mel_path, providers=["CPUExecutionProvider"])
    emb_session = ort.InferenceSession(emb_path, providers=["CPUExecutionProvider"])

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    models = []
    max_ef = 0

    if info.get("multi_model") and info.get("models"):
        for m in info["models"]:
            session = ort.InferenceSession(
                os.path.join(model_dir, m["model_file"]),
                providers=["CPUExecutionProvider"])
            ef = m["emb_frames"]
            models.append({"name": m["wake_word"], "session": session, "emb_frames": ef})
            max_ef = max(max_ef, ef)
    else:
        session = ort.InferenceSession(
            os.path.join(model_dir, info["model_file"]),
            providers=["CPUExecutionProvider"])
        ef = info.get("emb_frames", 16)
        models.append({"name": info["wake_word"], "session": session, "emb_frames": ef})
        max_ef = ef

    print(f"Loaded {len(models)} model(s), maxEmbFrames={max_ef}")
    return mel_session, emb_session, models, max_ef


# ── Inference ──

def detect(audio: np.ndarray, mel_sess, emb_sess, models, max_ef):
    audio = audio.astype(np.float32)
    if len(audio) < 32000:
        audio = np.pad(audio, (0, 32000 - len(audio)))
    audio = audio[:32000]

    # Mel spectrogram
    mel_in = audio.reshape(1, -1)
    mel_out = mel_sess.run(None, {"input": mel_in})[0]
    frames = mel_out.shape[2]
    if frames < EMB_WINDOW:
        return None

    mel2d = mel_out[0, 0] / 10.0 + 2.0

    # Embedding batch
    start_frame = max(0, frames - EMB_WINDOW - (max_ef - 1) * EMB_STEP)
    emb_batch = np.zeros((max_ef, EMB_WINDOW, 32, 1), dtype=np.float32)
    for w in range(max_ef):
        off = min(start_frame + w * EMB_STEP, frames - EMB_WINDOW)
        for f in range(EMB_WINDOW):
            for m in range(32):
                emb_batch[w, f, m, 0] = mel2d[off + f, m]

    emb_out = emb_sess.run(None, {"input_1": emb_batch})[0]
    embeddings = emb_out  # [max_ef, 1, 1, 96]

    # Run classifiers
    K = len(models)
    sigmoid_probs = np.zeros(K, dtype=np.float32)
    for i, model in enumerate(models):
        slice_start = max_ef - model["emb_frames"]
        wake_in = np.zeros((1, model["emb_frames"], 96), dtype=np.float32)
        for w in range(model["emb_frames"]):
            for d in range(96):
                wake_in[0, w, d] = embeddings[slice_start + w, 0, 0, d]
        out = model["session"].run(None, {"input": wake_in})[0]
        sigmoid_probs[i] = float(out[0, 0, 0])

    # Sigmoid → Softmax
    logits = np.zeros(K + 1, dtype=np.float32)
    for i in range(K):
        p = np.clip(sigmoid_probs[i], 1e-6, 1 - 1e-6)
        logits[i] = np.log(p / (1 - p))
    logits[K] = 0  # background

    logits -= logits.max()
    softmax = np.exp(logits)
    softmax /= softmax.sum()

    best_idx = np.argmax(softmax[:K])
    return {
        "wakeWord": models[best_idx]["name"],
        "probability": float(softmax[best_idx]),
        "background": float(softmax[K]),
    }


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Voicute Wake Word Inference")
    parser.add_argument("--model-dir", default="../models/", help="Path to model directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--cooldown", type=float, default=3.5, help="Cooldown in seconds")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    mel_sess, emb_sess, models, max_ef = load_models(args.model_dir)

    audio_buffer = deque(maxlen=SAMPLE_RATE * 4)
    last_detect = 0

    def callback(indata, frames, time_info, status):
        nonlocal last_detect
        audio_buffer.extend(indata[:, 0])
        if len(audio_buffer) < 32000:
            return

        audio = np.array(audio_buffer)[-32000:].astype(np.float32)
        result = detect(audio, mel_sess, emb_sess, models, max_ef)

        if result and result["probability"] > args.threshold:
            now = time.time()
            if now - last_detect > args.cooldown:
                last_detect = now
                print(f"\n>>> {result['wakeWord']} ({result['probability']:.2%})")

    print(f"Listening... threshold={args.threshold} cooldown={args.cooldown}s")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback,
                        dtype="float32", blocksize=4096):
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
