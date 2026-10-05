#!/usr/bin/env python3
"""
Wyoming wake word service for Voicute Causal DS-TCN models.

Protocol: https://github.com/rhasspy/wyoming — implemented with the official
`wyoming` package (JSON-line event headers), so Home Assistant's Wyoming
integration can connect. Earlier versions used a hand-rolled binary framing
that only our own test client understood; HA could never connect (issue #5).

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
import sys
import time
from pathlib import Path

import numpy as np
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart
from wyoming.event import Event
from wyoming.info import Attribution, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

# Add parent dir so we can import from python/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.wakeword_engine import WakeWordEngine, SAMPLE_RATE

_LOGGER = logging.getLogger("voicute-wyoming")

STRIDE_SAMPLES = SAMPLE_RATE // 20  # Process every 50ms (800 samples), matches web/wakeword.js

SERVICE_NAME = "voicute"
SERVICE_VERSION = "1.1.1"
SERVICE_URL = "https://github.com/voicute/onnx-wakeword"

# --debug: append received PCM here for offline analysis (temp dir)
DUMP_PATH = str(Path(__file__).resolve().parent / "debug_dump.pcm")


def _model_language(model_file: str, phrase: str = "") -> str:
    """Guess a BCP-47 language from the model path ('en/hey.onnx' → 'en')."""
    parts = Path(model_file.replace("\\", "/")).parts
    if parts and len(parts[0]) == 2 and parts[0].isalpha():
        return parts[0].lower()
    # No usable path (ZIP bundle): fall back to a CJK check on the phrase.
    return "zh" if any("一" <= c <= "鿿" for c in phrase) else "en"


def build_info(words: list[tuple[str, str]]) -> Info:
    """Wyoming Info event from (wake word, model_file) pairs."""
    attribution = Attribution(name="Voicute", url=SERVICE_URL)
    models = [
        WakeModel(
            name=name,
            description=f"Wake phrase: {name}",
            languages=[_model_language(model_file, name)],
            attribution=attribution,
            installed=True,
            version=SERVICE_VERSION,
            phrase=name,
        )
        for name, model_file in words
    ]
    return Info(
        wake=[
            WakeProgram(
                name=SERVICE_NAME,
                description=f"Voicute KWS — {len(models)} keyword(s)",
                attribution=attribution,
                installed=True,
                version=SERVICE_VERSION,
                models=models,
            )
        ]
    )


class VoicuteEventHandler(AsyncEventHandler):
    """One Wyoming session. Each HA connection gets its own audio buffer."""

    def __init__(self, service: "VoicuteWyomingService", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service = service
        self._converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self._audio = bytearray()
        self._detect_names: set[str] | None = None  # None = all words
        self._detected = False  # woke during the current audio stream?

    async def handle_event(self, event: Event) -> bool:
        etype = event.type

        if etype == "describe":
            info = build_info(self.service.word_list())
            await self.write_event(info.event())

        elif etype == "detect":
            names = Detect.from_event(event).names
            self._detect_names = set(names) if names else None

        elif etype == "audio-start":
            chunk = AudioStart.from_event(event)
            self._converter = AudioChunkConverter(
                rate=chunk.rate, width=chunk.width, channels=chunk.channels
            )
            self._audio.clear()
            self._detected = False
            self._enabled_reset()

        elif etype == "audio-chunk":
            chunk = AudioChunk.from_event(event)
            pcm = self._converter.convert(chunk)

            # --debug: dump received PCM for offline analysis
            if _LOGGER.isEnabledFor(logging.DEBUG):
                with open(DUMP_PATH, "ab") as f:
                    f.write(bytes(pcm.audio))

            detection = self.service.process_audio(
                bytes(pcm.audio), self._audio, self._detect_names
            )
            if detection is not None:
                self._detected = True
                await self.write_event(detection)

        elif etype == "audio-stop":
            # Same as wyoming-openwakeword: only report when nothing woke
            # during this stream — a Detection followed by NotDetected would
            # confuse the client.
            if not self._detected:
                await self.write_event(NotDetected().event())

        return True

    def _enabled_reset(self) -> None:
        self.service.engine_reset_state()


class VoicuteWyomingService:
    """Wyoming wake word service wrapping a Voicute WakeWordEngine."""

    def __init__(self, engine: WakeWordEngine, model_info_path: str,
                 preload_word: str = None):
        self.engine = engine
        self.preload_word = preload_word
        self._enabled = True
        self._name_to_file = self._load_name_to_file(model_info_path)
        self._last_prob_log = 0.0  # --debug: throttle per-second prob report

    @staticmethod
    def _load_name_to_file(model_info_path: str) -> dict[str, str]:
        """wake word → model_file from the raw model_info.json.

        The engine drops the file paths at load time; the Info event needs
        them back for per-model language detection. ZIP bundles (which the
        engine also accepts) fall back to empty paths.
        """
        try:
            with open(model_info_path, "r", encoding="utf-8") as f:
                header = f.read(2)
            if header == "PK":
                return {}  # ZIP bundle — no plain paths available
            with open(model_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except OSError:
            return {}

        out: dict[str, str] = {}
        if info.get("model_type") == "multi_keyword":
            model_file = info.get("model_file", "model.onnx")
            for kw in info.get("keywords", []):
                name = kw if isinstance(kw, str) else kw.get("name", "")
                out[name] = model_file
        elif info.get("multi_model"):
            for m in info.get("models", []):
                out[m.get("wake_word", "")] = m.get("model_file", "")
        else:
            out[info.get("wake_word", "")] = info.get("model_file", "model.onnx")
        return out

    def word_list(self) -> list[tuple[str, str]]:
        """(wake word, model_file) pairs for the Info event."""
        if self.preload_word:
            return [(self.preload_word, self._name_to_file.get(self.preload_word, ""))]
        return [(m["name"], self._name_to_file.get(m["name"], ""))
                for m in self.engine.models]

    def engine_reset_state(self) -> None:
        """Reset per-stream detection state at AudioStart."""
        self._enabled = True

    def process_audio(self, pcm: bytes, audio_buf: bytearray,
                      detect_names: set[str] | None = None):
        """Sliding-window detection over 16kHz mono PCM16.

        `audio_buf` is the per-connection backlog; `detect_names` (when not
        None) restricts which wake words may fire. Returns a Detection
        event when a word wakes, else None.
        """
        if not self._enabled:
            return None

        self._audio_extend(audio_buf, pcm)
        needed_bytes = self.engine.audio_samples_needed * 2
        stride_bytes = STRIDE_SAMPLES * 2  # 50ms slide = 1600 bytes

        if len(audio_buf) < needed_bytes:
            return None  # Not enough audio yet

        # Process one window per chunk (natural rate from audio chunks)
        chunk = bytes(audio_buf[:needed_bytes])
        del audio_buf[:stride_bytes]

        audio_i16 = np.frombuffer(chunk, dtype=np.int16)
        # Raw int16 range — matches Android (floatAudio[i]=(float)audio[i]),
        # Web (input[i]*32767), and python pyaudio path. The mel model expects
        # int16 PCM, NOT float [-1,1]; dividing by 32768 here made audio 32768×
        # too quiet (= "gain insufficient" / "needs word-by-word speech").
        audio = audio_i16.astype(np.float32)
        result = self.engine.predict(audio)
        if result is None:
            return None

        # Feed the L5 energy-jump filter with this window's RMS. Kept in int16
        # units to match the pyaudio path, so L5's thresholds stay calibrated.
        rms = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
        self.engine.l5_rms = rms
        self.engine.rms_hist[self.engine.l5_ri] = rms
        self.engine.rms_t_hist[self.engine.l5_ri] = time.time() * 1000
        self.engine.l5_ri = (self.engine.l5_ri + 1) % 128

        word, prob = result["word"], result["prob"]

        # --debug: once per second, report the live probability so a silent
        # mic/threshold problem is visible without a detection.
        now = time.time()
        if _LOGGER.isEnabledFor(logging.DEBUG) and now - self._last_prob_log > 1.0:
            self._last_prob_log = now
            _LOGGER.debug("prob=%.3f word=%s rms=%.0f", prob, word, rms)

        if detect_names is not None and word not in detect_names:
            return None
        detected = self.engine.detect(word, prob, result["cons_frames"])
        if detected:
            _LOGGER.info(f"✅ {detected}  ({prob:.1%})")
            return Detection(name=detected, timestamp=int(time.time() * 1000)).event()
        return None

    def _audio_extend(self, audio_buf: bytearray, pcm: bytes) -> None:
        # Cap the backlog at ~4× (window + stride) — beyond that the client
        # is streaming far faster than we consume and old audio is stale.
        max_bytes = (self.engine.audio_samples_needed + STRIDE_SAMPLES) * 2 * 4
        audio_buf.extend(pcm)
        if len(audio_buf) > max_bytes:
            del audio_buf[: len(audio_buf) - max_bytes]


async def run(host: str, port: int, service: VoicuteWyomingService) -> None:
    def handler_factory(reader, writer):
        return VoicuteEventHandler(service, reader, writer)

    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    _LOGGER.info(f"Listening on tcp://{host}:{port}")
    await server.run(handler_factory)


def main():
    p = argparse.ArgumentParser(description="Voicute Wyoming Wake Word Service")
    p.add_argument("--uri", default="tcp://0.0.0.0:10400")
    p.add_argument("--model-info", required=True, help="model_info.json path")
    p.add_argument("--mel", required=True, help="melspectrogram.onnx path")
    p.add_argument("--preload", nargs="*", default=None, help="wake words to advertise")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--cooldown", type=int, default=1500, help="cooldown ms")
    p.add_argument("--L1", type=int, default=0)
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
    svc = VoicuteWyomingService(engine, args.model_info, preload_word=preload)

    try:
        asyncio.run(run(host, int(port_s), svc))
    except KeyboardInterrupt:
        _LOGGER.info("Stopped.")


if __name__ == "__main__":
    main()
