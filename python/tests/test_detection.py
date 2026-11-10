"""Detection pipeline (L1-L5) unit tests — pure logic, no audio, no model.

Semantics matched to the production port (Android DetectionLogic):
- ``hi = prob > thr and word`` — any frame failing this resets the
  consecutive counter immediately.
- L1 gap tolerance (MAX_GAP) applies to brief prob dips where the model
  still reports a word, not to empty-word frames.
- L2 needs ``record()`` history: the first candidate after silence has a
  peak≈background and is suppressed; history turns it around.
- L5 uses ``record()`` RMS history, pends on energy jump, and confirms
  after a quiet post-speech tail (~700 ms).
"""
from onnx_wakeword import DetectionLogic
from onnx_wakeword.engine import WakeWordEngine

WORD = 'Hey Jarvis'


def make_logic(l1=True, l2=False, l3=False, l4=False, l5=False, cons=3, thr=0.5):
    dl = DetectionLogic(thr=thr, cons_frames=cons)
    dl.l1, dl.l2, dl.l3, dl.l4, dl.l5 = l1, l2, l3, l4, l5
    return dl


class TestThresholdGate:
    def test_below_threshold_returns_none(self):
        dl = make_logic(l1=False)
        assert dl.evaluate(WORD, 0.49, 100, 1000) is None
        assert dl.evaluate(WORD, 0.51, 100, 1001) == WORD

    def test_gap_frames_do_not_immediately_reset(self):
        # Android parity: a sub-threshold frame does not reset the counter;
        # it survives up to MAX_GAP=2 dip frames before resetting.
        dl = make_logic(cons=3)
        dl.evaluate(WORD, 0.9, 100, 1000)        # cons=1
        dl.evaluate(WORD, 0.9, 100, 1010)        # cons=2
        dl.evaluate('', 0.1, 100, 1020)          # gap=1, counter survives
        assert dl.cons == 2
        dl.evaluate('', 0.1, 100, 1030)          # gap=2
        dl.evaluate('', 0.1, 100, 1040)          # gap=3 > MAX_GAP → reset
        assert dl.cons == 0


class TestL1ConsecutiveFrames:
    def test_requires_cons_frames(self):
        dl = make_logic(cons=3)
        assert dl.evaluate(WORD, 0.9, 100, 1000) is None   # cons=1
        assert dl.evaluate(WORD, 0.9, 100, 1010) is None   # cons=2
        assert dl.evaluate(WORD, 0.9, 100, 1020) == WORD   # cons=3 → trigger

    def test_gap_then_recount(self):
        # One dip frame within tolerance: counter continues, not restarted.
        dl = make_logic(cons=3)
        dl.evaluate(WORD, 0.9, 100, 1000)                  # cons=1
        dl.evaluate('', 0.1, 100, 1010)                    # gap=1, survives
        dl.evaluate(WORD, 0.9, 100, 1020)                  # cons=2
        assert dl.evaluate(WORD, 0.9, 100, 1030) == WORD   # cons=3


class TestEngineGapTolerance:
    """engine.detect() delegates to DetectionLogic and keeps counting
    through brief prob dips (word still reported) up to MAX_GAP=2 frames."""

    @staticmethod
    def _engine():
        eng = WakeWordEngine()
        eng.set_L1(True)   # gap tolerance is an L1 feature
        eng.set_threshold(0.5)
        return eng

    def test_dip_within_gap_keeps_count(self):
        eng = self._engine()
        assert eng.detect(WORD, 0.9, 3, now=1000) is None   # cons=1
        assert eng.detect(WORD, 0.9, 3, now=1010) is None   # cons=2
        assert eng.detect(WORD, 0.4, 3, now=1020) is None   # gap=1 ≤ MAX_GAP
        assert eng.detect(WORD, 0.9, 3, now=1030) == WORD   # cons=3 → trigger

    def test_gap_beyond_tolerance_resets(self):
        eng = self._engine()
        eng.detect(WORD, 0.9, 3, now=1000)
        eng.detect(WORD, 0.9, 3, now=1010)
        eng.detect(WORD, 0.4, 3, now=1020)                  # gap=1
        eng.detect(WORD, 0.4, 3, now=1030)                  # gap=2
        eng.detect(WORD, 0.4, 3, now=1040)                  # gap=3 → reset
        # Counter restarted from 0: needs 3 fresh consecutive frames.
        assert eng.detect(WORD, 0.9, 3, now=1050) is None   # cons=1
        assert eng.detect(WORD, 0.9, 3, now=1060) is None   # cons=2
        assert eng.detect(WORD, 0.9, 3, now=1070) == WORD   # cons=3


class TestL3Cooldown:
    def test_cooldown_blocks_retrigger(self):
        dl = make_logic(l1=False, l3=True)
        t0 = 60000  # start well past the epoch-relative initial cooldown window
        assert dl.evaluate(WORD, 0.9, 100, t0) == WORD
        assert dl.evaluate(WORD, 0.9, 100, t0 + 500) is None    # within 1500 ms CD
        assert dl.evaluate(WORD, 0.9, 100, t0 + 1601) == WORD   # past cooldown


class TestL4Burst:
    def test_three_bursts_block(self):
        dl = make_logic(l1=False, l4=True)
        assert dl.evaluate(WORD, 0.95, 100, 1000) == WORD   # burst 1
        assert dl.evaluate(WORD, 0.95, 100, 2000) == WORD   # burst 2
        assert dl.evaluate(WORD, 0.95, 100, 3000) is None   # burst 3 → blocked
        assert dl.evaluate(WORD, 0.95, 100, 4000) is None   # still blocked (5 s)
        assert dl.evaluate(WORD, 0.95, 100, 9000) == WORD   # unblocked


class TestL2PeakBackground:
    def _bg_history(self, dl, prob, n, t0=1000):
        for i in range(n):
            dl.record(prob, '', 100, t0 + i * 10)

    def test_first_candidate_after_quiet_is_suppressed(self):
        dl = make_logic(l1=False, l2=True)
        self._bg_history(dl, 0.002, 20)                    # peak≈bg ≈ noise floor
        # peak (0.002) <= 3×bg (≈0.0033) → suppressed even above threshold
        assert dl.evaluate(WORD, 0.55, 100, 1300) is None

    def test_model_fluctuation_suppressed(self):
        dl = make_logic(l1=False, l2=True)
        self._bg_history(dl, 0.6, 1000)                    # bg EMA rises ≈0.6
        # peak 0.6 <= 3×bg ≈1.79 → suppressed
        assert dl.evaluate(WORD, 0.55, 100, 12000) is None

    def test_clear_peak_passes(self):
        dl = make_logic(l1=False, l2=True)
        self._bg_history(dl, 0.002, 20)
        dl.record(0.9, WORD, 100, 1210)                    # a real peak in history
        # peak 0.9 ≫ 3×bg ≈0.005 → passes
        assert dl.evaluate(WORD, 0.9, 100, 1300) == WORD


class TestL5EnergyJump:
    def _feed_history(self, dl, rms, t0=1000, n=6):
        for i in range(n):
            dl.record(0.0, '', rms, t0 + i * 100)

    def test_no_energy_jump_suppressed(self):
        dl = make_logic(l1=False, l5=True)
        dl.l5_delta = 1200
        self._feed_history(dl, rms=200)                    # pre window: 200 ms floor
        # curRms 300 < preMin(200) + 1200 → suppressed
        assert dl.evaluate(WORD, 0.9, 300, 3000) is None

    def test_energy_jump_pends_then_confirms(self):
        dl = make_logic(l1=False, l5=True)
        dl.l5_delta = 1200
        self._feed_history(dl, rms=200)
        # big jump → pending post-speech confirmation, no immediate trigger
        assert dl.evaluate(WORD, 0.9, 5000, 3000) is None
        # after POST_DELAY (700 ms), quiet tail (rms < preMin × 2.5) confirms
        for i in range(5):
            dl.record(0.0, '', 150, 3500 + i * 50)
        assert dl.evaluate('', 0.1, 150, 3800) == WORD

    def test_noisy_tail_never_confirms(self):
        dl = make_logic(l1=False, l5=True)
        dl.l5_delta = 1200
        self._feed_history(dl, rms=200)
        assert dl.evaluate(WORD, 0.9, 5000, 3000) is None
        for i in range(5):
            dl.record(0.0, '', 2000, 3500 + i * 50)        # loud tail — keep waiting
        assert dl.evaluate('', 0.1, 2000, 3800) is None


class TestEngineDetect:
    def test_engine_l1_gate(self):
        eng = WakeWordEngine()
        eng.set_L1(True)
        eng.threshold = 0.5
        assert eng.detect(WORD, 0.9, 3, now=1000) is None
        assert eng.detect(WORD, 0.9, 3, now=1010) is None
        assert eng.detect(WORD, 0.9, 3, now=1020) == WORD

    def test_engine_defaults_all_layers_off(self):
        # Default contract: model carries the false-trigger rejection;
        # out of the box the engine is a plain threshold gate.
        eng = WakeWordEngine()
        assert eng._dl.l1 is False and eng._dl.l2 is False and eng._dl.l3 is False
        assert eng._dl.l4 is False and eng._dl.l5 is False

    def test_engine_default_single_frame_triggers(self):
        # With L1 off (default), one frame above threshold triggers.
        eng = WakeWordEngine()
        eng.set_threshold(0.5)
        assert eng.detect(WORD, 0.9, 3, now=1000) == WORD
        assert eng.detect(WORD, 0.49, 3, now=1010) is None     # below threshold

    def test_engine_cooldown_toggle(self):
        eng = WakeWordEngine()
        eng.set_threshold(0.5)
        eng.set_L3(True)
        assert eng.detect(WORD, 0.9, 3, now=1000) == WORD      # first trigger
        assert eng.detect(WORD, 0.9, 3, now=2000) is None      # within cooldown
        eng.set_L3(False)                                      # runtime off
        assert eng.detect(WORD, 0.9, 3, now=2010) == WORD      # gate disabled

    def test_engine_threshold_clamp(self):
        eng = WakeWordEngine()
        eng.set_threshold(5.0)
        assert eng.threshold <= 0.95
        eng.set_threshold(-1)
        assert eng.threshold >= 0.3
