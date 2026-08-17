#!/usr/bin/env python3
"""
Wyoming wake word service for Voicute Causal DS-TCN models.

Protocol: https://github.com/rhasspy/wyoming
Compatible with Home Assistant Wyoming integration.

Usage:
    python wyoming_voicute.py \
        --model-info models/model_info.json \
        --mel models/melspectrogram.onnx \
        --preload "Hey Friday" \
        --threshold 0.4

Or with custom URI and layers:
    python wyoming_voicute.py \
        --uri tcp://0.0.0.0:10400 \
        --model-info models/model_info.json \
        --mel models/melspectrogram.onnx \
        --threshold 0.4 --L1 1 --L3 1 --L5 0
"""

import argparse
import asyncio
import json
import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Add parent dir so we can import from python/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.wakeword_engine import WakeWordEngine, SAMPLE_RATE

_LOGGER = logging.getLogger("voicute-wyoming")

# ── Wyoming protocol frame format ──────────────────────────────────────
# Each frame: uint8 type (0=event, 1=audio) + uint16 payload_len (BE)

STRIDE_SAMPLES = SAMPLE_RATE // 20  # Process every 50ms (800 samples), matches web/wakeword.js


class VoicuteWyomingService:
    """Wyoming wake word service wrapping a Voicute WakeWordEngine."""

    def __init__(self, engine: WakeWordEngine, preload_word: str = None):
        self.engine = engine
        self.preload_word = preload_word
        self._enabled = True
        self._audio = bytearray()
        self._last_process = 0.0

    async def start(self, host: str, port: int):
        self._server = await asyncio.start_server(self._handle, host, port)
        _LOGGER.info(f"Listening on tcp://{host}:{port}")
        async with self._server:
            await self._server.serve_forever()

    async def _handle(self, reader, writer):
        addr = writer.get_extra_info("peername")
        _LOGGER.info(f"Connected: {addr}")
        self._audio.clear()

        try:
            while True:
                header = await reader.readexactly(3)
                ftype, plen = struct.unpack(">BH", header)
                payload = await reader.readexactly(plen) if plen else b""

                if ftype == 0:   # Event (JSON)
                    await self._on_event(payload, writer)
                elif ftype == 1:  # Audio (PCM16 LE)
                    self._on_audio(payload, writer)
        except asyncio.IncompleteReadError:
            _LOGGER.debug(f"Disconnected: {addr}")
        except Exception as e:
            _LOGGER.error(f"Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _on_event(self, payload: bytes, writer):
        try:
            evt = json.loads(payload.decode())
        except Exception:
            return
        t = evt.get("type", "")

        if t == "describe":
            words = ([self.preload_word] if self.preload_word
                     else self.engine.keywords or [m["name"] for m in self.engine.models])
            self._send(writer, json.dumps({
                "type": "info",
                "data": {
                    "wake": True,
                    "asr": False, "handle": False, "intent": False,
                    "wakeWords": words,
                    "version": "1.1",
                    "name": "voicute",
                    "description": f"Voicute KWS — {len(words)} keyword(s)",
                    "attribution": {"name": "Voicute",
                                    "url": "https://github.com/voicute/onnx-wakeword"},
                }
            }).encode())
        elif t == "set-state":
            st = evt.get("data", {}).get("state", "on")
            self._enabled = (st == "on")
            _LOGGER.info(f"State → {st}")
        elif t == "not-set-state":
            # HA sends this; same format as set-state
            st = evt.get("data", {}).get("state", "on")
            self._enabled = (st == "on")

    def _on_audio(self, pcm: bytes, writer):
        """Sliding-window audio processing."""
        if not self._enabled:
            return

        self._audio.extend(pcm)
        needed_bytes = self.engine.audio_samples_needed * 2
        stride_bytes = STRIDE_SAMPLES * 2  # 100ms slide = 3200 bytes

        if len(self._audio) < needed_bytes:
            return  # Not enough audio yet

        # Process one window per call (natural rate from audio chunks)
        chunk = bytes(self._audio[:needed_bytes])
        del self._audio[:stride_bytes]

        audio_i16 = np.frombuffer(chunk, dtype=np.int16)
        # Raw int16 range — matches Android (floatAudio[i]=(float)audio[i]),
        # Web (input[i]*32767), and python pyaudio path. The mel model expects
        # int16 PCM, NOT float [-1,1]; dividing by 32768 here made audio 32768×
        # too quiet (= "gain insufficient" / "needs word-by-word speech").
        audio = audio_i16.astype(np.float32)
        result = self.engine.predict(audio)

        if result is None:
            return

        # Feed the L5 energy-jump filter with this window's RMS. Kept in int16
        # units to match the pyaudio path, so L5's thresholds stay calibrated.
        rms = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
        self.engine.l5_rms = rms
        self.engine.rms_hist[self.engine.l5_ri] = rms
        self.engine.rms_t_hist[self.engine.l5_ri] = time.time() * 1000
        self.engine.l5_ri = (self.engine.l5_ri + 1) % 128

        word, prob = result["word"], result["prob"]
        detected = self.engine.detect(word, prob, result["cons_frames"])
        if detected:
            _LOGGER.info(f"✅ {detected}  ({prob:.1%})")
            self._send(writer, json.dumps({
                "type": "detection",
                "data": {
                    "name": detected,
                    "probability": round(float(prob), 4),
                    "timestamp": int(time.time() * 1000),
                }
            }).encode())

    @staticmethod
    def _send(writer, payload: bytes):
        writer.write(struct.pack(">BH", 0, len(payload)) + payload)


def main():
    p = argparse.ArgumentParser(description="Voicute Wyoming Wake Word Service")
    p.add_argument("--uri", default="tcp://0.0.0.0:10400")
    p.add_argument("--model-info", required=True, help="model_info.json path")
    p.add_argument("--mel", required=True, help="melspectrogram.onnx path")
    p.add_argument("--preload", nargs="*", default=None, help="wake words to advertise")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--cooldown", type=int, default=1500, help="cooldown ms")
    p.add_argument("--L1", type=int, default=1)
    p.add_argument("--L3", type=int, default=1)
    p.add_argument("--L5", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse URI → host:port
    uri = args.uri.replace("tcp://", "")
    host, port_s = uri.rsplit(":", 1) if ":" in uri else (uri, "10400")

    # Load engine
    engine = WakeWordEngine()
    engine.load(args.model_info, args.mel)
    engine.threshold = args.threshold
    engine.cooldown_ms = args.cooldown

    for name, val in [("L1", args.L1), ("L2", 0), ("L3", args.L3),
                       ("L4", 0), ("L5", args.L5)]:
        setattr(engine, name, bool(val))

    _LOGGER.info(f"Model: {args.model_info}")
    _LOGGER.info(f"Keywords: {[m['name'] for m in engine.models]}")
    _LOGGER.info(f"Threshold={engine.threshold:.2f}  "
                 f"Cooldown={engine.cooldown_ms}ms  "
                 f"L1={engine.L1} L3={engine.L3} L5={engine.L5}")
    _LOGGER.info(f"Window: {engine.audio_samples_needed} samples "
                 f"({engine.audio_samples_needed/SAMPLE_RATE:.1f}s)")

    preload = args.preload[0] if args.preload and len(args.preload) == 1 else None
    svc = VoicuteWyomingService(engine, preload_word=preload)

    try:
        asyncio.run(svc.start(host, int(port_s)))
    except KeyboardInterrupt:
        _LOGGER.info("Stopped.")


if __name__ == "__main__":
    main()
