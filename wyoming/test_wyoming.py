#!/usr/bin/env python3
"""
Test script for wyoming_voicute.py — feed audio and check detection.

Usage:
    # Mode 1: Send a WAV file
    python test_wyoming.py --wav recording.wav

    # Mode 2: Live microphone
    python test_wyoming.py --mic

    # Default: localhost:10400
    python test_wyoming.py --wav test.wav --host 127.0.0.1 --port 10400
"""

import argparse
import json
import socket
import struct
import sys
import time
import wave


def send_frame(sock, ftype: int, payload: bytes):
    """Send a Wyoming frame."""
    sock.sendall(struct.pack(">BH", ftype, len(payload)) + payload)


def recv_frame(sock, timeout: float = 0.1):
    """Receive a Wyoming frame. Returns (ftype, payload) or None."""
    sock.settimeout(timeout)
    try:
        header = sock.recv(3)
        if len(header) < 3:
            return None
        ftype, plen = struct.unpack(">BH", header)
        payload = b""
        while len(payload) < plen:
            chunk = sock.recv(plen - len(payload))
            if not chunk:
                break
            payload += chunk
        return ftype, payload
    except socket.timeout:
        return None


def describe(sock):
    """Send describe request, print response."""
    send_frame(sock, 0, json.dumps({"type": "describe"}).encode())
    time.sleep(0.2)
    result = recv_frame(sock, timeout=1.0)
    if result:
        _, payload = result
        info = json.loads(payload.decode())
        data = info.get("data", {})
        print(f"Service: {data.get('name', '?')} — {data.get('description', '?')}")
        print(f"Wake words: {data.get('wakeWords', [])}")
    else:
        print("No response from service — is it running?")


def stream_wav(sock, wav_path: str, chunk_samples: int = 512):
    """Stream a WAV file as Wyoming audio frames, listen for detections."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        wav_frames = wf.getnframes()
        print(f"WAV: {sr}Hz, {ch}ch, {wav_frames} frames "
              f"({wav_frames/sr:.1f}s)")

        assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

        # Read entire WAV into memory
        raw_wav = wf.readframes(wav_frames)
        if ch == 2:
            import numpy as np
            arr = np.frombuffer(raw_wav, dtype=np.int16).reshape(-1, 2)
            raw_wav = arr.mean(axis=1).astype(np.int16).tobytes()

        # Loop short WAVs so L1 consecutive-frames check has enough windows
        loops = max(1, int(4.0 / (wav_frames / sr)) + 1)  # at least 4s total
        if loops > 1:
            print(f"Looping WAV {loops}× ({loops * wav_frames / sr:.1f}s total)")

        send_frame(sock, 0, json.dumps({
            "type": "set-state",
            "data": {"state": "on"}
        }).encode())
        time.sleep(0.1)

        print("Streaming audio...")

        pos = 0
        data = raw_wav * loops
        while pos < len(data):
            raw = data[pos:pos + chunk_samples * 2]  # 2 bytes/sample
            pos += chunk_samples * 2
            if not raw:
                break

            # Convert stereo → mono if needed
            if ch == 2:
                import numpy as np
                arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
                mono = arr.mean(axis=1).astype(np.int16)
                raw = mono.tobytes()

            send_frame(sock, 1, raw)

            # Check for detection events
            while True:
                result = recv_frame(sock, timeout=0.01)
                if result is None:
                    break
                ftype, payload = result
                if ftype == 0:
                    evt = json.loads(payload.decode())
                    if evt.get("type") == "detection":
                        d = evt["data"]
                        print(f"\n  ✅ DETECTED: {d['name']} "
                              f"({d['probability']*100:.1f}%)\n")
                    elif evt.get("type") == "info":
                        pass  # already printed
                    else:
                        print(f"  ← {evt.get('type')}")

            sys.stdout.write(".")
            sys.stdout.flush()

    print("\nDone.")


def stream_mic(sock, chunk_samples: int = 512, gain: float = 5.0, device: int = None):
    """Stream live microphone as Wyoming audio frames."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. pip install sounddevice")
        return

    import numpy as np

    send_frame(sock, 0, json.dumps({
        "type": "set-state",
        "data": {"state": "on"}
    }).encode())
    time.sleep(0.1)

    print("🎤 Listening... (speak now, Ctrl+C to stop)")
    print("   Level: ", end="", flush=True)

    cb_count = [0]

    def callback(indata, frames, t, status):
        cb_count[0] += 1
        audio = np.asarray(indata[:, 0], dtype=np.float64) * gain
        rms = float(np.sqrt(np.mean(audio ** 2)))
        raw = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        send_frame(sock, 1, raw)

        # Show level meter every ~10 callbacks
        if cb_count[0] % 10 == 0:
            bars = min(20, int(rms * 60))
            sys.stdout.write(f"\r   Level: {'█' * bars}{' ' * (20 - bars)} {rms:.3f}  ")
            sys.stdout.flush()

        # Check for detections
        while True:
            result = recv_frame(sock, timeout=0.001)
            if result is None:
                break
            ftype, payload = result
            if ftype == 0:
                evt = json.loads(payload.decode())
                if evt.get("type") == "detection":
                    d = evt["data"]
                    print(f"\n  ✅ DETECTED: {d['name']} "
                          f"({d['probability']*100:.1f}%)\n")
                    print("   Level: ", end="", flush=True)

    with sd.InputStream(samplerate=16000, channels=1, device=device,
                        blocksize=chunk_samples, callback=callback):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def main():
    p = argparse.ArgumentParser(description="Test Voicute Wyoming service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10400)
    p.add_argument("--wav", help="Path to 16kHz mono WAV file")
    p.add_argument("--mic", action="store_true", help="Use live microphone")
    p.add_argument("--gain", type=float, default=5.0, help="Mic gain multiplier (default: 5.0)")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--device", type=int, default=None, help="Input device index")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        sys.exit(0)

    if not args.wav and not args.mic:
        p.error("Need --wav or --mic")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connected to {args.host}:{args.port}")

    describe(sock)

    if args.wav:
        stream_wav(sock, args.wav)
    elif args.mic:
        stream_mic(sock, gain=args.gain, device=args.device)

    sock.close()


if __name__ == "__main__":
    main()
