"""Batch AISHELL FA/h benchmark — computes prob once, applies all detection configs.

Usage:
  python bench_fa.py --models manbo --hours 2
"""
import argparse, os, sys, time, wave, random
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_logic import DetectionLogic

SAMPLE_RATE = 16000
MEL_TIME, N_MELS = 98, 32
AUDIO_WIN = (MEL_TIME - 1) * 160 + 512 + 160  # 16192
HOP = int(0.04 * SAMPLE_RATE)


def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate(); nch = wf.getnchannels(); n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype=np.int16)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        nl = int(len(data) * ratio)
        data = np.interp(np.linspace(0, len(data)-1, nl), np.arange(len(data)), data).astype(np.int16)
    return data.astype(np.float32)


def sweep_file(fpath, mel_sess, model_sessions, thr=0.5, cons=2):
    """Run inference on one file, return probs and RMS per window."""
    try:
        audio = load_wav(fpath)
    except Exception:
        return None

    n_win = max(1, (len(audio) - AUDIO_WIN) // HOP + 1)
    all_model_probs = {name: [] for name in model_sessions}
    all_rms = []

    for w in range(n_win):
        start = w * HOP
        chunk = audio[start:start + AUDIO_WIN]
        if len(chunk) < AUDIO_WIN:
            chunk = np.pad(chunk, (0, AUDIO_WIN - len(chunk)))

        mel_out = mel_sess.run(None, {"input": chunk.reshape(1, -1).astype(np.float32)})[0]
        frames = mel_out.shape[2]
        if frames < 1:
            continue
        mel_data = mel_out[0, 0] / 10.0 + 2.0
        mel_start = max(0, frames - MEL_TIME)
        tcn_in = np.zeros((1, MEL_TIME, N_MELS), dtype=np.float32)
        for f in range(MEL_TIME):
            src_f = mel_start + f
            if src_f < frames:
                tcn_in[0, f, :] = mel_data[src_f, :]

        for name, sess in model_sessions.items():
            prob = float(sess.run(None, {"input": tcn_in})[0][0, 0])
            all_model_probs[name].append(prob)

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        all_rms.append(rms)

    return all_model_probs, all_rms, len(audio) / SAMPLE_RATE


def evaluate_sweep(model_name, probs, rms, total_dur, thr=0.5, cons=2):
    """Run all detection configs on pre-computed probs/rms. Probs/rms is a list of per-file lists for proper reset."""
    configs = [
        ("Raw",      False, False, False, False, False),
        ("L2 only",  False, True,  False, False, False),
        ("L1 only",  True,  False, False, False, False),
        ("L1+L2",    True,  True,  False, False, False),
        ("L1+L2+L3", True,  True,  True,  False, False),
        ("L1+L2+L5", True,  True,  False, False, True),
        ("All L1-L5",True,  True,  True,  True,  True),
    ]

    results = {}
    for cfg_name, l1, l2, l3, l4, l5 in configs:
        dl = DetectionLogic(thr=thr, cons_frames=cons)
        dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = l1, l2, l3, l4, l5

        fa = 0
        for file_probs, file_rms in zip(probs, rms):  # per-file lists
            dl.reset()  # fresh state for each file
            for idx, (prob, rms_val) in enumerate(zip(file_probs, file_rms)):
                now_ms = idx * 40
                word = model_name if prob > thr else ""
                dl.record(prob, word, rms_val, now_ms)
                result = dl.evaluate(word, prob, rms_val, now_ms)
                if result:
                    fa += 1

        hours = total_dur / 3600
        fa_h = fa / hours if hours > 0 else 0
        results[cfg_name] = fa_h

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="manbo,manbo_voice,nihaodiannao",
                        help="Model keys: manbo, manbo_voice, nihaodiannao, kaishibofang, gugugaga, laifu")
    parser.add_argument("--hours", type=float, default=2.0, help="Target test hours")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--cons", type=int, default=2)
    parser.add_argument("--aishell-dir", required=True,
                        help="AISHELL-1 wav directory (e.g. path/to/data_aishell/wav/test)")
    args = parser.parse_args()

    # Model registry
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    MODEL_REGISTRY = {
        "manbo":           os.path.join(MODEL_DIR, "manbo.onnx"),
        "manbo_voice":     os.path.join(MODEL_DIR, "manbo_voice_model.onnx"),
        "nihaodiannao":    os.path.join(MODEL_DIR, "nihaodiannao.onnx"),
        "kaishibofang":    os.path.join(MODEL_DIR, "kaishibofang.onnx"),
        "gugugaga":        os.path.join(MODEL_DIR, "gugugaga.onnx"),
        "laifu":           os.path.join(MODEL_DIR, "laifu.onnx"),
    }

    model_keys = [k.strip() for k in args.models.split(",")]
    model_paths = {}
    for k in model_keys:
        if k in MODEL_REGISTRY:
            model_paths[k] = MODEL_REGISTRY[k]
        else:
            print(f"Unknown model: {k}")
    if not model_paths:
        print("No valid models specified")
        sys.exit(1)

    # Collect WAVs
    wavs = []
    for root, dirs, files in os.walk(args.aishell_dir):
        for f in files:
            if f.endswith('.wav'):
                wavs.append(os.path.join(root, f))
    wavs.sort()
    random.seed(42)
    random.shuffle(wavs)

    # Estimate files needed for target hours (AISHELL avg ~4s/file)
    est_files = int(args.hours * 3600 / 4)
    wavs = wavs[:min(est_files * 2, len(wavs))]  # take 2x to be safe
    print(f"AISHELL dir: {args.aishell_dir}")
    print(f"Files: {len(wavs)}  |  Target: ~{args.hours}h")
    print(f"Models: {list(model_paths.keys())}")
    print()

    # Load ONNX sessions
    mel_path = os.path.join(MODEL_DIR, "melspectrogram.onnx")
    mel_sess = ort.InferenceSession(mel_path, providers=["CPUExecutionProvider"])
    model_sessions = {name: ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                      for name, path in model_paths.items()}

    # Pre-compute probs for all files (store per-file for proper reset)
    t0 = time.time()
    all_data = {name: {"probs": [], "rms": [], "dur": 0} for name in model_sessions}
    files_done = 0
    total_dur_h = 0

    for fi, fpath in enumerate(wavs):
        result = sweep_file(fpath, mel_sess, model_sessions, thr=args.thr, cons=args.cons)
        if result is None:
            continue
        model_probs, rms_vals, dur = result
        files_done += 1
        total_dur_h += dur / 3600

        for name in model_sessions:
            all_data[name]["probs"].append(model_probs[name])   # list of per-file lists
            all_data[name]["rms"].append(rms_vals)              # list of per-file lists
            all_data[name]["dur"] += dur

        if total_dur_h >= args.hours:
            break

        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(wavs)}] collected {total_dur_h:.1f}h, {elapsed:.0f}s")

    infer_time = time.time() - t0
    print(f"\nInference done: {files_done} files, {total_dur_h:.2f}h in {infer_time:.0f}s")

    # Run detection sweeps
    print()
    for name in model_sessions:
        d = all_data[name]
        print(f"{'='*60}")
        print(f"MODEL: {name}  ({d['dur']/3600:.2f}h)")
        print(f"{'='*60}")
        results = evaluate_sweep(name, d["probs"], d["rms"], d["dur"],
                                 thr=args.thr, cons=args.cons)
        for cfg, fa_h in results.items():
            print(f"  {cfg:<12} FA/h={fa_h:>8.1f}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Config':<12}"
    for name in model_sessions:
        header += f" {name:>14}"
    print(header)
    print("-" * (12 + 15 * len(model_sessions)))

    all_results = {}
    for name in model_sessions:
        all_results[name] = evaluate_sweep(name, all_data[name]["probs"],
                                           all_data[name]["rms"], all_data[name]["dur"],
                                           thr=args.thr, cons=args.cons)

    for cfg in ["Raw", "L2 only", "L1 only", "L1+L2", "L1+L2+L3", "L1+L2+L5", "All L1-L5"]:
        line = f"{cfg:<12}"
        for name in model_sessions:
            line += f" {all_results[name].get(cfg, 0):>14.1f}"
        print(line)

    print(f"\nTest duration: {total_dur_h:.2f}h  |  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
