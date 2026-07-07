package com.voicute.wakeword;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.util.Log;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements AudioCapture.Listener {

    private static final String TAG = "Voicute";
    private static final int PERM_REQ = 1001;
    private static final long INFERENCE_INTERVAL_MS = 30;            // ~33 fps inference
    private static final long CAPTURE_RESTART_IDLE_MS = 1_800_000;   // restart capture after 30min idle
    private static final long MIN_RESTART_GAP_MS = 600_000;          // don't restart more than every 10min
    private static final float RMS_GATE = 30f;                       // skip inference when near-silent (saves CPU)

    // --- Core components ---
    private WakeWordEngine engine;
    private AudioCapture audioCapture;
    private DetectionLogic detection;

    // --- Threading ---
    private Thread inferThread;
    private Handler mainHandler;
    private PowerManager.WakeLock wakeLock;

    // --- State ---
    private volatile boolean running;
    private volatile float threshold = 0.50f;
    private volatile long lastInferMs;
    private long lastCaptureRestartMs;
    private long lastUiUpdate;
    private long lastDebugLog;
    private float latestProb;
    private String latestWord = "";

    // --- UI ---
    private TextView statusText, scoreText, detectText, counterText;
    private TextView thresholdText;
    private SeekBar thresholdSeekBar;
    private CheckBox l5CheckBox;
    private CheckBox saveAudioCheckBox;
    private SeekBar l5RatioSeekBar;
    private TextView l5RatioText;
    private Button toggleButton;
    private LinearLayout logContainer;
    private ScrollView logScrollView;

    // --- Logging ---
    private SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);
    private SimpleDateFormat fileTimeFmt = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US);
    private File detectionLogFile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        scoreText = findViewById(R.id.scoreText);
        detectText = findViewById(R.id.detectText);
        counterText = findViewById(R.id.counterText);
        thresholdText = findViewById(R.id.thresholdText);
        thresholdSeekBar = findViewById(R.id.thresholdSeekBar);
        l5CheckBox = findViewById(R.id.l5CheckBox);
        saveAudioCheckBox = findViewById(R.id.saveAudioCheckBox);
        l5RatioSeekBar = findViewById(R.id.l5RatioSeekBar);
        l5RatioText = findViewById(R.id.l5RatioText);
        toggleButton = findViewById(R.id.toggleButton);
        logContainer = findViewById(R.id.logContainer);
        logScrollView = findViewById(R.id.logScrollView);

        mainHandler = new Handler(Looper.getMainLooper());

        // Detection log file
        detectionLogFile = new File(getExternalFilesDir(null), "detection_log.txt");
        try {
            if (!detectionLogFile.exists()) detectionLogFile.createNewFile();
            FileWriter fw = new FileWriter(detectionLogFile, true);
            fw.write("=== Session: " + fileTimeFmt.format(new Date()) + " ===\n");
            fw.close();
        } catch (IOException e) { Log.e(TAG, "Log file error", e); }

        // Threshold slider
        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                threshold = p / 100.0f;
                thresholdText.setText(String.format(Locale.US, "%d%%", p));
            }
            @Override public void onStartTrackingTouch(SeekBar s) {}
            @Override public void onStopTrackingTouch(SeekBar s) {}
        });
        threshold = thresholdSeekBar.getProgress() / 100.0f;

        // L5 toggle
        l5CheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (detection != null) detection.l5Enabled = isChecked;
        });

        saveAudioCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            saveAudio = isChecked;
        });

        // L5 delta slider: 100 ~ 3100 (default 1200)
        l5RatioSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean fromUser) {
                int delta = 100 + p * 100;  // 0-30 → 100-3100
                l5RatioText.setText(String.format(Locale.US, "+%d", delta));
                if (detection != null) detection.l5Delta = delta;
            }
            @Override public void onStartTrackingTouch(SeekBar s) {}
            @Override public void onStopTrackingTouch(SeekBar s) {}
        });
        // Init slider to default 1200 (progress=11)
        l5RatioSeekBar.setProgress(11);

        toggleButton.setOnClickListener(v -> {
            if (running) stopListening(); else startListening();
        });

        findViewById(R.id.clearButton).setOnClickListener(v -> {
            logContainer.removeAllViews();
            detection.reset();
            counterText.setText("");
            detectText.setText("");
        });

        // Load models
        new Thread(() -> {
            engine = new WakeWordEngine(MainActivity.this);
            audioCapture = new AudioCapture();
            audioCapture.setListener(MainActivity.this);
            detection = new DetectionLogic();
            runOnUiThread(() -> {
                if (engine.isLoaded()) {
                    statusText.setText("已加载 " + engine.getModelCount() + " 个模型");
                    TextView sub = findViewById(R.id.wakeWordSubtitle);
                    if (sub != null) sub.setText("唤醒词: " + engine.getWakeWordDisplay());
                    toggleButton.setEnabled(true);
                } else {
                    String err = engine.getErrorMessage();
                    statusText.setText(err != null ? "错误: " + err : "模型加载失败");
                }
            });
        }).start();
        toggleButton.setEnabled(false);
    }

    @Override
    public void onMicError(String msg) {
        mainHandler.post(() -> statusText.setText(msg));
    }

    // ===================================================================
    // Start / Stop
    // ===================================================================

    private void startListening() {
        if (!hasPermission()) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, PERM_REQ);
            return;
        }
        // Guard: don't start if already running
        if (inferThread != null && inferThread.isAlive()) {
            Log.w(TAG, "Inference thread already running — ignoring duplicate start");
            return;
        }

        running = true;
        detection.reset();
        toggleButton.setText("停止");
        statusText.setText("正在监听 \"" + engine.getWakeWordDisplay() + "\"...");

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (pm != null) {
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Voicute:WakeWord");
            wakeLock.acquire();
        }

        audioCapture.start();
        inferThread = new Thread(this::inferenceLoop, "Inference");
        inferThread.start();
    }

    private void stopListening() {
        running = false;
        toggleButton.setText("开始");
        statusText.setText("已停止");

        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
            wakeLock = null;
        }

        if (inferThread != null) {
            inferThread.interrupt();
            try { inferThread.join(2000); } catch (InterruptedException ignored) {}
            inferThread = null;
        }
        if (audioCapture != null) audioCapture.stop();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopListening();
        if (engine != null) {
            new Thread(() -> engine.close()).start();
        }
    }

    private boolean hasPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] p, int[] results) {
        super.onRequestPermissionsResult(requestCode, p, results);
        if (requestCode == PERM_REQ && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            startListening();
        }
    }

    // ===================================================================
    // Inference loop
    // ===================================================================

    /**
     * Main inference loop. Runs at ~33fps on a dedicated thread.
     *
     * Pipeline: Audio ring buffer → RMS gate → ONNX inference →
     *           DetectionLogic evaluation → UI update.
     *
     * The audio capture thread writes PCM into a lock-free ring buffer.
     * This thread reads from it, decoupled so inference speed doesn't
     * affect audio capture and vice versa.
     */
    private void inferenceLoop() {
        // Warm-up delay: wait for ring buffer to fill with audio
        try { Thread.sleep(500); } catch (InterruptedException e) { return; }

        long lastHeartbeat = 0;

        while (running && !Thread.interrupted()) {
            long t0 = System.currentTimeMillis();

            // Watchdog: if capture thread crashed or was killed by OS,
            // restart it (with minimum gap to avoid restart storms)
            if (!audioCapture.isAlive() && running) {
                Log.w(TAG, "Capture dead — restarting");
                long now = System.currentTimeMillis();
                if (now - lastCaptureRestartMs >= MIN_RESTART_GAP_MS) {
                    lastCaptureRestartMs = now;
                    audioCapture.restart();
                    try { Thread.sleep(800); } catch (InterruptedException e) { break; }
                    if (!audioCapture.isAlive()) {
                        mainHandler.post(() -> statusText.setText("麦克风错误 - 请重启应用"));
                        break;
                    }
                }
            }

            // Idle restart to prevent hardware sleep
            long idleMs = audioCapture.getLastAudioActivityMs();
            if (running && audioCapture.isAlive() && idleMs > 0
                    && (t0 - idleMs) > CAPTURE_RESTART_IDLE_MS
                    && (t0 - lastCaptureRestartMs) >= MIN_RESTART_GAP_MS) {
                Log.w(TAG, "Audio idle 30min — restarting capture");
                lastCaptureRestartMs = t0;
                audioCapture.restart();
            }

            // Read latest audio chunk from the ring buffer (lock-free)
            int pos = audioCapture.getRingPos();
            short[] audioChunk = audioCapture.readChunk(pos, engine.getAudioSamplesNeeded());

            // Compute RMS (root-mean-square) of the audio chunk.
            // RMS ≈ perceived loudness: silent room ~20, normal speech ~500-2000.
            float sumSq = 0;
            for (short s : audioChunk) sumSq += (float) s * (float) s;
            float chunkRms = (float) Math.sqrt(sumSq / audioChunk.length);

            // Skip inference when quiet — saves CPU/battery, prevents noise from
            // feeding the model (which would output random low probs and pollute bg)
            WakeWordEngine.DetectionResult result = null;
            if (chunkRms >= RMS_GATE) {
                result = engine.process(audioChunk);
            }

            long t1 = System.currentTimeMillis();
            lastInferMs = t1 - t0;

            String word = "";
            float prob = 0;
            float bgProb = 0;
            int modelConsFrames = 5;
            if (result != null) {
                word = result.wakeWord != null ? result.wakeWord : "";
                prob = result.probability;
                bgProb = result.backgroundMean;
                modelConsFrames = result.recommendedConsFrames;
            }

            latestProb = prob;
            latestWord = word;

            // DS-CNN per-frame debug (helps diagnose recognition issues)
            if (engine.isDscnnMode() && prob > 0.05f) {
                Log.d(TAG, String.format(Locale.US,
                        "DS-CNN: word=%s prob=%.3f bg=%.3f", word, prob, bgProb));
            }

            // Feed DetectionLogic (pass RMS for L5 pre-speech check)
            detection.record(prob, word, chunkRms, t1);
            String triggered = detection.evaluate(word, prob, chunkRms, threshold,
                    modelConsFrames, 0, t1);

            if (triggered != null) {
                final String tw = triggered;
                final float tp = detection.lastTrigProb;
                final float tb = bgProb;
                mainHandler.post(() -> showDetection(tw, tp, tb, chunkRms,
                        detection.baseCons));
            }

            // Periodic debug (1s)
            if (t1 - lastDebugLog > 1000) {
                lastDebugLog = t1;
                Log.d(TAG, String.format(Locale.US,
                        "word=%s prob=%.2f bg=%.4f peak=%.2f rms=%.0f preRms=%.0f cons=%d/%d L%d",
                        latestWord.isEmpty() ? "-" : latestWord,
                        latestProb, detection.dbgBg, detection.dbgPeak,
                        chunkRms, detection.dbgPreRms,
                        detection.consFrames, detection.baseCons, detection.dbgFail));
            }

            // Heartbeat
            if (t1 - lastHeartbeat > 30000) {
                lastHeartbeat = t1;
                Log.i(TAG, "Inference alive | captureAlive=" + audioCapture.isAlive()
                        + " ringPos=" + pos);
            }

            // UI update
            if ((t1 - lastUiUpdate) > 400) {
                lastUiUpdate = t1;
                mainHandler.post(() -> {
                    scoreText.setText(String.format(Locale.US,
                            "word:%s prob:%.1f%% rms:%.0f L5:%s %dms",
                            latestWord, latestProb * 100, chunkRms,
                            detection.l5Enabled ? ("+" + detection.l5Delta) : "off",
                            lastInferMs));
                });
            }

            long elapsed = System.currentTimeMillis() - t0;
            long sleepMs = INFERENCE_INTERVAL_MS - elapsed;
            if (sleepMs > 0) {
                try { Thread.sleep(sleepMs); } catch (InterruptedException e) { break; }
            }
        }
    }

    // ===================================================================
    // UI
    // ===================================================================

    private boolean saveAudio = false;  // toggle for false-trigger analysis

    private void showDetection(String word, float prob, float bg, float rms, int cons) {
        detectText.setText(String.format(Locale.US, "%s 检测到! (%.0f%%)", word, prob * 100));

        // Save last 2s audio for false-trigger analysis (async)
        if (saveAudio && audioCapture != null) {
            final int pos = audioCapture.getRingPos();
            final AudioCapture cap = audioCapture;
            final String w = word;
            final java.io.File outDir = new java.io.File(getExternalFilesDir(null), "wake_audio");
            outDir.mkdirs();
            new Thread(() -> {
                String ts = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
                        .format(new java.util.Date());
                String path = outDir + "/wake_" + w + "_" + ts + ".wav";
                cap.saveWav(pos, 32000, path);
                Log.i(TAG, "Audio saved: " + path);
            }).start();
        }
        detectText.setAlpha(1f);
        counterText.setText(String.format(Locale.US, "已检测到 %d 次", detection.count));
        addLogEntry(word, prob);

        // Persistent log
        try {
            FileWriter fw = new FileWriter(detectionLogFile, true);
            BufferedWriter bw = new BufferedWriter(fw);
            float maxVal = 0;
            bw.write(String.format(Locale.US,
                    "#%d  %s  word=%s  prob=%.1f%%  bg=%.1f%%  rms=%.0f  cons=%d\n",
                    detection.count, fileTimeFmt.format(new Date()),
                    word, prob * 100, bg * 100, rms, cons));
            bw.close();
        } catch (IOException e) { Log.e(TAG, "Log write error", e); }

        mainHandler.postDelayed(() -> {
            detectText.animate().alpha(0f).setDuration(400).start();
        }, 1200);
    }

    private void addLogEntry(String word, float prob) {
        String now = timeFmt.format(new Date());
        int count = detection.count;

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(8, 6, 8, 6);
        row.setBackgroundColor(count % 2 == 0 ? 0xFF1a1a2e : 0xFF16213e);

        TextView idxView = new TextView(this);
        idxView.setText(String.format(Locale.US, "#%d", count));
        idxView.setTextSize(11);
        idxView.setTextColor(0xFF888888);
        idxView.setWidth(60);
        row.addView(idxView);

        TextView msgView = new TextView(this);
        msgView.setText(word);
        msgView.setTextSize(12);
        msgView.setTextColor(0xFFe94560);
        msgView.setTypeface(null, Typeface.BOLD);
        msgView.setLayoutParams(new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        row.addView(msgView);

        TextView timeView = new TextView(this);
        timeView.setText(now);
        timeView.setTextSize(10);
        timeView.setTextColor(0xFF999999);
        timeView.setWidth(100);
        timeView.setGravity(Gravity.END);
        row.addView(timeView);

        TextView probView = new TextView(this);
        probView.setText(String.format(Locale.US, "%.0f%%", prob * 100));
        probView.setTextSize(11);
        probView.setTextColor(prob > 0.8f ? 0xFF16c79a : 0xFFffd93d);
        probView.setTypeface(null, Typeface.BOLD);
        probView.setWidth(60);
        probView.setGravity(Gravity.END);
        row.addView(probView);

        logContainer.addView(row, 0);
        if (logContainer.getChildCount() > 50) {
            logContainer.removeViewAt(logContainer.getChildCount() - 1);
        }
        logScrollView.post(() -> logScrollView.fullScroll(ScrollView.FOCUS_UP));
    }
}
