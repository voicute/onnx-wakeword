#!/usr/bin/env python3
"""Convert ONNX wake word models to TFLite for 32-bit Raspberry Pi / ARM.

Usage:
    pip install onnx2tf
    python convert_to_tflite.py --model-dir ../models/
    python convert_to_tflite.py --model-dir ../models/ --output-dir ../models/
"""

import argparse
import json
import os
import sys
import subprocess


SAMPLE_RATE = 16000
EMB_WINDOW = 76
EMB_STEP = 8
MEL_HOP_SAMPLES = 160
MEL_WIN_SAMPLES = int(SAMPLE_RATE * 0.025)


def calc_audio_samples(emb_frames: int) -> int:
    """Calculate audio samples needed for a given emb_frames value."""
    mel_frames = EMB_WINDOW + (emb_frames - 1) * EMB_STEP
    return mel_frames * MEL_HOP_SAMPLES + MEL_WIN_SAMPLES


def run_onnx2tf(input_path: str, output_dir: str, fixed_shape: dict = None,
                keep_shape: bool = False):
    """Convert a single ONNX model to TFLite using onnx2tf CLI."""
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", input_path,
        "-o", output_dir,
        "-osd",  # keep same output names (important for multi-model compat)
    ]
    if not keep_shape:
        cmd.append("--non_verbose")
    if fixed_shape:
        # Fix dynamic axes to concrete values
        for name, shape in fixed_shape.items():
            cmd.extend(["-k", name])
            cmd.append(",".join(str(s) for s in shape))

    print(f"  converting: {os.path.basename(input_path)}")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-500:] if result.stderr else 'unknown'}")
        return False

    # onnx2tf creates output in a subfolder; find the .tflite file
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".tflite"):
                tflite_path = os.path.join(root, f)
                # Copy to flat output directory
                target = os.path.join(output_dir, f)
                if tflite_path != target:
                    with open(tflite_path, "rb") as src:
                        with open(target, "wb") as dst:
                            dst.write(src.read())
                    print(f"    -> {f}")
                return True

    print(f"    WARNING: no .tflite found in {output_dir}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX wake word models to TFLite (for 32-bit Pi)")
    parser.add_argument("--model-dir", default="../models/",
                        help="Directory containing ONNX models + model_info.json")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as model-dir)")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir or model_dir)

    info_path = os.path.join(model_dir, "model_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Collect required models
    models_to_convert = []
    mel_frames_map = {}  # model -> emb_frames

    if info.get("model_type") == "multi":
        ef = info["emb_frames"]
        models_to_convert.append(info["model_file"])
        mel_frames_map[info["model_file"]] = ef
    elif info.get("multi_model") and info.get("models"):
        for m in info["models"]:
            models_to_convert.append(m["model_file"])
            mel_frames_map[m["model_file"]] = m["emb_frames"]
    else:
        models_to_convert.append(info["model_file"])
        mel_frames_map[info["model_file"]] = info.get("emb_frames", 16)

    # Determine max emb_frames for shared mel/embedding
    max_ef = max(mel_frames_map.values()) if mel_frames_map else 16
    audio_samples = calc_audio_samples(max_ef)
    emb_batch = max_ef

    print(f"Config: maxEmbFrames={max_ef}, audioSamples={audio_samples}, "
          f"embBatch={emb_batch}")
    print(f"Output directory: {output_dir}\n")

    # 1. Mel spectrogram
    mel_onnx = os.path.join(model_dir, "melspectrogram.onnx")
    mel_out = os.path.join(output_dir, "melspectrogram_tflite")
    if os.path.exists(mel_onnx):
        print("[1/3] Mel spectrogram — fixing input to [1, {}]".format(audio_samples))
        # Fix batch=1 and time dimension
        os.makedirs(mel_out, exist_ok=True)
        ok = run_onnx2tf(mel_onnx, mel_out,
                         fixed_shape={"input": [1, audio_samples]})
        if ok:
            print("  melspectrogram: OK\n")
        else:
            print("  melspectrogram: FAILED (run with --verbose for details)\n")
    else:
        print(f"  SKIP: {mel_onnx} not found\n")

    # 2. Embedding
    emb_onnx = os.path.join(model_dir, "embedding_model.onnx")
    emb_out = os.path.join(output_dir, "embedding_model_tflite")
    if os.path.exists(emb_onnx):
        print(f"[2/3] Embedding — fixing batch to [{emb_batch}, 76, 32, 1]")
        os.makedirs(emb_out, exist_ok=True)
        ok = run_onnx2tf(emb_onnx, emb_out,
                         fixed_shape={"input_1": [emb_batch, 76, 32, 1]})
        if ok:
            print("  embedding: OK\n")
        else:
            print("  embedding: FAILED (run with --verbose for details)\n")
    else:
        print(f"  SKIP: {emb_onnx} not found\n")

    # 3. Classifiers
    print("[3/3] Classifiers")
    for model_file in models_to_convert:
        onnx_path = os.path.join(model_dir, model_file)
        ef = mel_frames_map[model_file]
        cls_out = os.path.join(output_dir, model_file.rsplit(".", 1)[0] + "_tflite")
        if os.path.exists(onnx_path):
            print(f"  {model_file} — fixing to [1, {ef}, 96]")
            os.makedirs(cls_out, exist_ok=True)
            ok = run_onnx2tf(onnx_path, cls_out,
                             fixed_shape={"input": [1, ef, 96]})
            if ok:
                print(f"  {model_file}: OK\n")
            else:
                print(f"  {model_file}: FAILED\n")
        else:
            print(f"  SKIP: {onnx_path} not found\n")

    print("Done. TFLite models are in:", output_dir)
    print("\nNext steps (on Raspberry Pi):")
    print("  pip install tflite-runtime numpy sounddevice")
    print("  sudo apt install libportaudio2")
    print(f"  python infer_tflite.py --model-dir {output_dir}/")


if __name__ == "__main__":
    main()
