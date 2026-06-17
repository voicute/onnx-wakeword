package com.voicute.wakeword;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.NoiseSuppressor;
import android.util.Log;

/**
 * Audio capture with ring buffer and RMS computation.
 *
 * Runs on its own thread. Writes captured PCM into a lock-free ring buffer.
 * Inference thread reads from the ring buffer at its own pace.
 */
public class AudioCapture {

    private static final String TAG = "Voicute.Audio";

    public interface Listener {
        void onMicError(String message);
    }

    // Audio params
    static final int SAMPLE_RATE = 16000;

    // Ring buffer: ~3 seconds of 16-bit mono audio
    private final int ringSize = SAMPLE_RATE * 3;
    private final short[] ringBuffer;
    private volatile int ringPos;

    private AudioRecord recorder;
    private NoiseSuppressor noiseSuppressor;
    private Thread captureThread;
    private volatile boolean running;
    private volatile boolean alive;
    private Listener listener;

    // Stats
    private volatile float latestRms;
    private volatile long lastAudioActivityMs;

    public AudioCapture() {
        ringBuffer = new short[ringSize];
    }

    public void setListener(Listener l) { this.listener = l; }

    /** Ring buffer write position (for inference thread to snapshot). */
    public int getRingPos() { return ringPos; }

    /**
     * Read the most recent `samplesNeeded` samples from the ring buffer.
     * Reads BACKWARDS from the current write position — always gets the
     * freshest audio without needing locks or atomics.
     */
    public short[] readChunk(int pos, int samplesNeeded) {
        short[] chunk = new short[samplesNeeded];
        for (int i = 0; i < samplesNeeded; i++) {
            int idx = (pos - samplesNeeded + i + ringSize) % ringSize;
            chunk[i] = ringBuffer[idx];
        }
        return chunk;
    }

    /** RMS of the most recently written chunk. */
    public float getLatestRms() { return latestRms; }

    /** Timestamp of last audio activity (RMS > 200). */
    public long getLastAudioActivityMs() { return lastAudioActivityMs; }

    /** Whether capture thread is alive and recording. */
    public boolean isAlive() { return alive; }

    /** Start capture thread. */
    public void start() {
        running = true;
        ringPos = 0;
        captureThread = new Thread(this::captureLoop, "AudioCapture");
        captureThread.start();
    }

    /** Stop capture thread and release resources. */
    public void stop() {
        running = false;
        if (captureThread != null) {
            captureThread.interrupt();
            try { captureThread.join(1000); } catch (InterruptedException ignored) {}
            captureThread = null;
        }
        releaseRecorder();
    }

    /** Restart: stop old recorder and thread, then start fresh. */
    public void restart() {
        releaseRecorder();
        alive = false;
        if (captureThread != null && captureThread.isAlive()) {
            captureThread.interrupt();
            try { captureThread.join(2000); } catch (InterruptedException ignored) {}
        }
        captureThread = new Thread(this::captureLoop, "AudioCapture");
        captureThread.start();
    }

    private void releaseRecorder() {
        if (noiseSuppressor != null) {
            try { noiseSuppressor.setEnabled(false); } catch (Exception ignored) {}
            try { noiseSuppressor.release(); } catch (Exception ignored) {}
            noiseSuppressor = null;
        }
        if (recorder != null) {
            try { recorder.stop(); } catch (Exception ignored) {}
            try { recorder.release(); } catch (Exception ignored) {}
            recorder = null;
        }
    }

    // --- Capture loop ---

    private void captureLoop() {
        try {
            alive = false;
            int bufferSize = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
            bufferSize = Math.max(bufferSize, WakeWordEngine.MEL_HOP_SAMPLES * 16);

            recorder = new AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize);

            if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                if (listener != null) listener.onMicError("麦克风初始化失败");
                running = false;
                return;
            }

            if (NoiseSuppressor.isAvailable()) {
                noiseSuppressor = NoiseSuppressor.create(recorder.getAudioSessionId());
                if (noiseSuppressor != null) noiseSuppressor.setEnabled(true);
            }

            recorder.startRecording();
            alive = true;

            short[] readChunk = new short[WakeWordEngine.MEL_HOP_SAMPLES * 8];
            long lastCaptureLog = 0;
            long recordStartMs = System.currentTimeMillis();
            float maxRmsSinceStart = 0;
            int zeroReadCount = 0;

            while (running && !Thread.interrupted()) {
                // Read raw PCM from microphone
                int read = recorder.read(readChunk, 0, readChunk.length);
                // Zero reads = mic is being held by another app (e.g. voice assistant)
                if (read <= 0) {
                    zeroReadCount++;
                    if (zeroReadCount > 10 && listener != null) {
                        listener.onMicError("麦克风被占用，请关闭其他使用麦克风的应用");
                    }
                    continue;
                }
                zeroReadCount = 0;

                // Write captured PCM into the lock-free ring buffer.
                // The inference thread reads from this buffer at its own pace.
                for (int i = 0; i < read; i++) {
                    ringBuffer[ringPos] = readChunk[i];
                    ringPos = (ringPos + 1) % ringSize;
                }

                // Track RMS for silence detection and activity monitoring
                float rms = rmsOf(readChunk, read);
                latestRms = rms;
                if (rms > maxRmsSinceStart) maxRmsSinceStart = rms;
                if (rms > 200) {
                    lastAudioActivityMs = System.currentTimeMillis();
                }

                // After 3 seconds, if we've never seen loud audio, the mic is
                // probably claimed by another app (common on some OEM ROMs)
                if ((System.currentTimeMillis() - recordStartMs) > 3000
                        && maxRmsSinceStart < 50 && listener != null) {
                    listener.onMicError("麦克风被占用，请关闭其他使用麦克风的应用");
                }

                long now = System.currentTimeMillis();
                if (now - lastCaptureLog > 5000) {
                    lastCaptureLog = now;
                    Log.d(TAG, "Capture alive | ringPos=" + ringPos
                            + " rms=" + String.format(java.util.Locale.US, "%.1f", rms));
                }
            }

            recorder.stop();
            recorder.release();
            recorder = null;
            alive = false;
        } catch (Exception e) {
            Log.e(TAG, "Capture thread crashed", e);
            alive = false;
            if (recorder != null) {
                try { recorder.stop(); } catch (Exception ignored) {}
                try { recorder.release(); } catch (Exception ignored) {}
                recorder = null;
            }
        }
    }

    private float rmsOf(short[] buf, int len) {
        double sum = 0;
        for (int i = 0; i < len; i++) sum += (double) buf[i] * buf[i];
        return (float) Math.sqrt(sum / len);
    }

    /** Save last N samples from ring buffer as a WAV file (for false-trigger analysis). */
    public void saveWav(int pos, int samples, String filePath) {
        short[] pcm = readChunk(pos, samples);
        try {
            java.io.DataOutputStream dos = new java.io.DataOutputStream(
                    new java.io.FileOutputStream(filePath));
            int dataLen = pcm.length * 2;
            // WAV header
            dos.writeBytes("RIFF");
            dos.writeInt(Integer.reverseBytes(36 + dataLen));
            dos.writeBytes("WAVE");
            dos.writeBytes("fmt ");
            dos.writeInt(Integer.reverseBytes(16));
            dos.writeShort(Short.reverseBytes((short) 1));      // PCM
            dos.writeShort(Short.reverseBytes((short) 1));      // mono
            dos.writeInt(Integer.reverseBytes(SAMPLE_RATE));
            dos.writeInt(Integer.reverseBytes(SAMPLE_RATE * 2)); // byte rate
            dos.writeShort(Short.reverseBytes((short) 2));      // block align
            dos.writeShort(Short.reverseBytes((short) 16));     // bits/sample
            dos.writeBytes("data");
            dos.writeInt(Integer.reverseBytes(dataLen));
            // PCM data
            for (short s : pcm) dos.writeShort(Short.reverseBytes(s));
            dos.close();
        } catch (java.io.IOException e) {
            Log.e(TAG, "saveWav failed: " + e.getMessage());
        }
    }
}
