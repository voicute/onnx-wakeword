#!/usr/bin/env python3
"""
Test script for wyoming_voicute.py — speaks the real Wyoming protocol
(official `wyoming` package, same as Home Assistant).

Usage:
    # Mode 1: Send a WAV file
    python test_wyoming.py --wav recording.wav

    # Mode 2: Live microphone
    python test_wyoming.py --mic

    # Default: localhost:10400
    python test_wyoming.py --wav test.wav --host 127.0.0.1 --port 10400
"""

import argparse
import asyncio
import contextlib
import sys
import wave

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event
from wyoming.info import Info
from wyoming.wake import Detection

CHUNK_SAMPLES = 512  # 32ms — same block size HA satellites use


async def describe(client: AsyncTcpClient):
    """Send describe request, print the advertised wake models."""
    await client.write_event(Event(type="describe"))
    while True:
        event = await asyncio.wait_for(client.read_event(), timeout=5.0)
        if event is None:
            print("Connection closed without info — is the service running?")
            return False
        if event.type == "info":
            info = Info.from_event(event)
            for prog in info.wake or []:
                print(f"Service: {prog.name} — {prog.description}")
                for model in prog.models:
                    print(f"  wake word: {model.name}  languages={model.languages}")
            return True
        print(f"← {event.type}")


async def stream_wav(client: AsyncTcpClient, wav_path: str):
    detections: list[str] = []
    stop_reader = asyncio.Event()

    async def reader():
        # Background task owns the socket reads — polling read_event() with
        # wait_for() can swallow an event that arrives mid-cancellation.
        while not stop_reader.is_set():
            event = await client.read_event()
            if event is None:
                return
            if event.type == "detection":
                d = Detection.from_event(event)
                detections.append(d.name)
                print(f"\n  ✅ DETECTED: {d.name}\n")

    reader_task = asyncio.create_task(reader())

    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        wav_frames = wf.getnframes()
        print(f"WAV: {sr}Hz, {ch}ch, {wav_frames} frames ({wav_frames/sr:.1f}s)")
        assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

        raw_wav = wf.readframes(wav_frames)
        if ch == 2:
            import numpy as np
            arr = np.frombuffer(raw_wav, dtype=np.int16).reshape(-1, 2)
            raw_wav = arr.mean(axis=1).astype(np.int16).tobytes()

        # Loop short WAVs so L1 consecutive-frames check has enough windows
        loops = max(1, int(4.0 / (wav_frames / sr)) + 1)  # at least 4s total
        if loops > 1:
            print(f"Looping WAV {loops}× ({loops * wav_frames / sr:.1f}s total)")

        await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
        print("Streaming audio...")

        data = raw_wav * loops
        pos = 0
        while pos < len(data):
            raw = data[pos:pos + CHUNK_SAMPLES * 2]
            pos += CHUNK_SAMPLES * 2
            if not raw:
                break

            if ch == 2:  # mono-ize each chunk for even framing
                import numpy as np
                arr = np.frombuffer(raw, dtype=np.int16)
                if arr.size % 2:
                    arr = arr[:-1]
                arr = arr.reshape(-1, 2)
                raw = arr.mean(axis=1).astype(np.int16).tobytes()

            await client.write_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=raw).event()
            )
            await asyncio.sleep(0)  # yield so the reader drains between chunks

            sys.stdout.write(".")
            sys.stdout.flush()

        await client.write_event(AudioStop().event())

    # AudioStop → the server replies NotDetected if nothing woke. Give the
    # reader up to 1.5s to catch in-flight events (detections arrive with a
    # processing lag). If nothing more comes (detection already sent), the
    # wait times out — that is the normal exit, not an error.
    stop_reader.set()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(reader_task, timeout=1.5)
    reader_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await reader_task

    print(f"\nDone. {len(detections)} detection(s).")


async def stream_mic(client: AsyncTcpClient, chunk_samples: int = 512,
                     gain: float = 5.0, device: int = None):
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. pip install sounddevice")
        return

    import numpy as np

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
    print("🎤 Listening... (speak now, Ctrl+C to stop)")
    print("   Level: ", end="", flush=True)

    cb_count = [0]

    def callback(indata, frames, t, status):
        cb_count[0] += 1
        audio = np.asarray(indata[:, 0], dtype=np.float64) * gain
        rms = float(np.sqrt(np.mean(audio ** 2)))
        raw = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        # PortAudio callbacks run on a non-loop thread — plain
        # queue.put_nowait() isn't thread-safe here and the sender task
        # may never wake up. Must hop into the loop explicitly.
        loop.call_soon_threadsafe(queue.put_nowait, raw)

        if cb_count[0] % 10 == 0:
            bars = min(20, int(rms * 60))
            sys.stdout.write(f"\r   Level: {'█' * bars}{' ' * (20 - bars)} {rms:.3f}  ")
            sys.stdout.flush()

    async def sender():
        while True:
            raw = await queue.get()
            await client.write_event(
                AudioChunk(rate=16000, width=2, channels=1, audio=raw).event()
            )

    async def reader():
        while True:
            event = await client.read_event()
            if event is None:
                return
            if event.type == "detection":
                d = Detection.from_event(event)
                print(f"\n  ✅ DETECTED: {d.name}\n")
                print("   Level: ", end="", flush=True)

    send_task = asyncio.create_task(sender())
    read_task = asyncio.create_task(reader())

    with sd.InputStream(samplerate=16000, channels=1, device=device,
                        blocksize=chunk_samples, callback=callback):
        try:
            await asyncio.gather(send_task, read_task)
        except asyncio.CancelledError:
            pass


async def main_async():
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
        return

    if not args.wav and not args.mic:
        p.error("Need --wav or --mic")

    async with AsyncTcpClient(args.host, args.port) as client:
        print(f"Connected to {args.host}:{args.port}")
        if not await describe(client):
            return
        if args.wav:
            await stream_wav(client, args.wav)
        elif args.mic:
            await stream_mic(client, gain=args.gain, device=args.device)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
