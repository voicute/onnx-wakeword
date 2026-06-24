"""Test L2 detection layer on AISHELL — replicates @handhin's evaluation.

Usage:
  python test_l2_fa.py --model ../models/manbo.onnx --mel ../models/melspectrogram.onnx
  python test_l2_fa.py --model ../models/manbo.onnx --mel ../models/melspectrogram.onnx --max-files 500
"""

import argparse
import os
import sys
import time
import wave
import numpy as np
import onnxruntime as ort

# Add detection_logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_logic import DetectionLogic

SAMPLE_RATE = 16000
MEL_HOP = 160          # 10ms at 16kHz
MEL_WIN = 400          # 25ms at 16kHz
N_MELS = 32
MEL_TIME = 98          # ~1.0s window
AUDIO_WIN = (MEL_TIME - 1) * 160 + 512 + 160  # = 16192, matches training AUDIO_TARGET_98
HOP_SAMPLES = int(0.04 * SAMPLE_RATE)  # 40ms stride (matches 4-frame hop in Android)


def load_wav(path):
    """Load WAV, return float32 audio at 16kHz mono."""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
    # Resample if needed
    if sr != SAMPLE_RATE:
        import math
        ratio = SAMPLE_RATE / sr
        new_len = int(len(data) * ratio)
        data = np.interp(np.linspace(0, len(data) - 1, new_len), np.arange(len(data)), data).astype(np.int16)
    return data.astype(np.float32)


def run_test(model_path, mel_path, wav_files, thr=0.5, cons_frames=2, max_files=None,
             l1=True, l2=True, l3=False, l4=False, l5=False):
    """Run sliding-window inference on WAV files, return FA count and total hours."""
    # Load ONNX models
    mel_sess = ort.InferenceSession(mel_path, providers=["CPUExecutionProvider"])
    model_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    dl = DetectionLogic(thr=thr, cons_frames=cons_frames)
    dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = l1, l2, l3, l4, l5

    total_fa = 0
    total_duration_s = 0
    total_windows = 0
    skipped = 0

    files = wav_files[:max_files] if max_files else wav_files
    t0 = time.time()

    for fi, fpath in enumerate(files):
        try:
            audio = load_wav(fpath)
        except Exception:
            skipped += 1
            continue

        duration_s = len(audio) / SAMPLE_RATE
        total_duration_s += duration_s

        # Sliding window
        n_windows = max(1, (len(audio) - AUDIO_WIN) // HOP_SAMPLES + 1)
        if n_windows <= 0:
            skipped += 1
            continue

        dl.reset()

        for w in range(n_windows):
            start = w * HOP_SAMPLES
            chunk = audio[start:start + AUDIO_WIN]
            if len(chunk) < AUDIO_WIN:
                # Pad last window
                chunk = np.pad(chunk, (0, AUDIO_WIN - len(chunk)))

            # Mel spectrogram via ONNX
            mel_in = chunk.reshape(1, -1).astype(np.float32)
            mel_out = mel_sess.run(None, {"input": mel_in})[0]
            # mel_out shape: [1, 1, frames, 32] — take last 98 frames
            frames = mel_out.shape[2]
            if frames < 1:
                continue

            # Build TCN input: last MEL_TIME frames, shape [1, 98, 32]
            mel_data = mel_out[0, 0]  # [frames, 32], typically 99 frames
            frames = mel_data.shape[0]
            mel_data = mel_data / 10.0 + 2.0  # standard preprocessing

            mel_start = max(0, frames - MEL_TIME)
            tcn_input = np.zeros((1, MEL_TIME, N_MELS), dtype=np.float32)
            for f in range(MEL_TIME):
                src_f = mel_start + f
                if src_f < frames:
                    tcn_input[0, f, :] = mel_data[src_f, :]

            # Model inference
            model_out = model_sess.run(None, {"input": tcn_input})[0]
            prob = float(model_out[0]) if model_out.ndim <= 2 else float(model_out[0, 0])

            # RMS from audio chunk
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            now_ms = int(start / SAMPLE_RATE * 1000)
            word = "曼波" if prob > thr else ""
            dl.record(prob, word, rms, now_ms)
            result = dl.evaluate(word, prob, rms, now_ms)

            total_windows += 1

            if result:  # non-empty string = actual trigger
                total_fa += 1

        # Progress
        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            dur = total_duration_s / 3600
            fa_h = total_fa / dur if dur > 0 else 0
            print(f"  [{fi+1}/{len(files)}] {dur:.1f}h  FA={total_fa}  FA/h={fa_h:.1f}  "
                  f"bg={dl.bg:.4f}  windows={total_windows}  elapsed={elapsed:.0f}s")

    total_hours = total_duration_s / 3600
    fa_per_hour = total_fa / total_hours if total_hours > 0 else 0
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Files: {len(files)-skipped}  |  Skipped: {skipped}")
    print(f"Duration: {total_hours:.2f}h  |  Windows: {total_windows}")
    print(f"FA: {total_fa}  |  FA/h: {fa_per_hour:.1f}")
    print(f"Config: thr={thr} cons={cons_frames}  "
          f"L1={'on' if l1 else 'off'} L2={'on' if l2 else 'off'} "
          f"L3={'on' if l3 else 'off'} L4={'on' if l4 else 'off'} L5={'on' if l5 else 'off'}")
    print(f"Final bg={dl.bg:.4f}  |  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return fa_per_hour, total_hours


def main():
    parser = argparse.ArgumentParser(description="Test L2 detection on AISHELL")
    parser.add_argument("--model", default="../models/manbo.onnx", help="Wake word ONNX model")
    parser.add_argument("--mel", default="../models/melspectrogram.onnx", help="Melspectrogram ONNX model")
    parser.add_argument("--thr", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--cons", type=int, default=2, help="Consecutive frames (L1)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit test files")
    parser.add_argument("--aishell-dir", required=True,
                        help="AISHELL-1 wav directory (e.g. path/to/data_aishell/wav/test)")
    args = parser.parse_args()

    if not args.aishell_dir:
        print("ERROR: --aishell-dir is required (e.g. path/to/data_aishell/wav/test)")
        sys.exit(1)
    aishell_root = args.aishell_dir

    if not os.path.isdir(aishell_root):
        print(f"ERROR: Directory not found: {aishell_root}")
        sys.exit(1)

    # Collect WAVs
    wav_files = []
    for root, dirs, files in os.walk(aishell_root):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(root, f))
    wav_files.sort()
    print(f"AISHELL dir: {aishell_root}")
    print(f"WAV files: {len(wav_files)}")
    print(f"Model: {args.model}")
    print(f"Mel: {args.mel}")

    # Resolve model paths relative to script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(script_dir, args.model)
    mel_path = args.mel if os.path.isabs(args.mel) else os.path.join(script_dir, args.mel)

    # ── Test 1: Raw (no detection layers) ──
    print(f"\n{'='*60}")
    print("TEST 1: Raw (L1=off, L2=off)")
    print(f"{'='*60}")
    raw_fa, hours = run_test(model_path, mel_path, wav_files, thr=args.thr,
                              cons_frames=args.cons, max_files=args.max_files,
                              l1=False, l2=False, l3=False, l4=False, l5=False)

    # ── Test 2: L2 only ──
    print(f"\n{'='*60}")
    print("TEST 2: L2 only (L1=off, L2=on)")
    print(f"{'='*60}")
    l2_fa, _ = run_test(model_path, mel_path, wav_files, thr=args.thr,
                         cons_frames=args.cons, max_files=args.max_files,
                         l1=False, l2=True, l3=False, l4=False, l5=False)

    # ── Test 3: L1 only ──
    print(f"\n{'='*60}")
    print("TEST 3: L1 only (L1=on, L2=off)")
    print(f"{'='*60}")
    l1_fa, _ = run_test(model_path, mel_path, wav_files, thr=args.thr,
                         cons_frames=args.cons, max_files=args.max_files,
                         l1=True, l2=False, l3=False, l4=False, l5=False)

    # ── Test 4: L1+L2 ──
    print(f"\n{'='*60}")
    print("TEST 4: L1+L2 (L1=on, L2=on)")
    print(f"{'='*60}")
    l1l2_fa, _ = run_test(model_path, mel_path, wav_files, thr=args.thr,
                           cons_frames=args.cons, max_files=args.max_files,
                           l1=True, l2=True, l3=False, l4=False, l5=False)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw:      FA/h={raw_fa:.1f}")
    print(f"  L2 only:  FA/h={l2_fa:.1f}  (Δ = {raw_fa - l2_fa:.1f})")
    print(f"  L1 only:  FA/h={l1_fa:.1f}")
    print(f"  L1+L2:    FA/h={l1l2_fa:.1f}  (Δ from L1 only = {l1_fa - l1l2_fa:.1f})")
    print(f"\n  L2 contribution:")
    if raw_fa > 0:
        print(f"    L2-only reduction:  {raw_fa - l2_fa:.1f} ({100*(raw_fa-l2_fa)/raw_fa:.0f}%)")
    if l1_fa > 0:
        print(f"    L2-on-L1 reduction:  {l1_fa - l1l2_fa:.1f} ({100*(l1_fa-l1l2_fa)/l1_fa:.0f}%)")


if __name__ == "__main__":
    main()
