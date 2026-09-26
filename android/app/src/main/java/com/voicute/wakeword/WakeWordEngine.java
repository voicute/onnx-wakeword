package com.voicute.wakeword;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Wake word inference engine.
 *
 * Supports two modes:
 *   1. Multi-model (legacy): one ONNX per keyword, each outputs [B,1] sigmoid
 *   2. Multi-keyword (new):  single ONNX outputs [B,N] sigmoid array — one forward pass
 *
 * Pipeline: audio → melspectrogram.onnx → classifier → sigmoid.
 */
public class WakeWordEngine {

    private static final String TAG = "WakeWordEngine";
    private static final String MEL_MODEL = "melspectrogram.onnx";

    // Audio parameters (constant)
    static final int SAMPLE_RATE = 16000;
    static final float MEL_HOP_SEC = 0.010f;
    static final float MEL_WIN_SEC = 0.025f;
    static final int MEL_HOP_SAMPLES = (int) (SAMPLE_RATE * MEL_HOP_SEC);
    static final int N_MELS = 34;      // legacy DS-CNN width; runtime width resolved from model input shape
    private int inputMels = N_MELS;    // resolved in bindModelInputShape() at load time
    static final int RAW_MELS = 32;     // mel spectrogram output channels (unchanged)

    /** Detection result with specific wake word name. */
    public static class DetectionResult {
        public final String wakeWord;
        public final float probability;
        /** Mean probability across ALL models — represents background noise level. */
        public final float backgroundMean;
        /** Per-model recommended consecutive frames (from model_info.json). */
        public final int recommendedConsFrames;

        public DetectionResult(String wakeWord, float probability, float backgroundMean,
                               int recommendedConsFrames) {
            this.wakeWord = wakeWord;
            this.probability = probability;
            this.backgroundMean = backgroundMean;
            this.recommendedConsFrames = recommendedConsFrames;
        }
    }

    private static class ModelSlot {
        final String wakeWord;
        final String modelFile;
        final int consFrames;
        OrtSession session;

        ModelSlot(String wakeWord, String modelFile, int consFrames) {
            this.wakeWord = wakeWord;
            this.modelFile = modelFile;
            this.consFrames = consFrames;
        }
    }

    // === New: single multi-keyword model ===
    private boolean isMultiKeyword;
    private String[] keywords;                // index → keyword name
    private OrtSession multiKwSession;        // single session for multi-keyword model
    private int multiKwConsFrames = 2;

    // === Legacy: multi-model ===
    private final List<ModelSlot> models = new ArrayList<>();

    private String[] wakeWordNames;
    private int melFramesNeeded;
    private int audioSamplesNeeded;
    private int dscnnMelTime = 98;

    public int getMelFramesNeeded() { return melFramesNeeded; }
    public int getAudioSamplesNeeded() { return audioSamplesNeeded; }

    /** Pipe-separated display string of all wake words. */
    public String getWakeWordDisplay() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < wakeWordNames.length; i++) {
            if (i > 0) sb.append(" | ");
            sb.append(wakeWordNames[i]);
        }
        return sb.toString();
    }

    /** Number of wake word models loaded. */
    public int getModelCount() { return isMultiKeyword ? keywords.length : models.size(); }
    public boolean isDscnnMode() { return true; }

    private final OrtEnvironment env;
    private OrtSession melSession;

    private boolean loaded;
    private String errorMessage = null;
    private int debugLogCount = 0;
    private static final int DEBUG_LOG_MAX = 50;
    private long engineStartTime = System.currentTimeMillis();
    private static final long STARTUP_SKIP_MS = 3000;  // skip first 3s to avoid cold-start FP

    public WakeWordEngine(Context context) {
        env = OrtEnvironment.getEnvironment();
        try {
            AssetManager am = context.getAssets();
            byte[] infoBytes;
            try (InputStream is = am.open("model_info.json")) {
                infoBytes = new byte[is.available()];
                int offset = 0;
                while (offset < infoBytes.length) {
                    int read = is.read(infoBytes, offset, infoBytes.length - offset);
                    if (read < 0) break;
                    offset += read;
                }
            }
            JSONObject info = new JSONObject(new String(infoBytes, "UTF-8"));

            dscnnMelTime = info.optInt("mel_time", 98);
            Log.i(TAG, "mel_time=" + dscnnMelTime);

            // ── New: single multi-keyword model ──
            if (info.has("model_type") && "multi_keyword".equals(info.getString("model_type"))) {
                isMultiKeyword = true;
                JSONArray kwArray = info.getJSONArray("keywords");
                keywords = new String[kwArray.length()];
                for (int i = 0; i < kwArray.length(); i++) {
                    keywords[i] = kwArray.getString(i);
                }
                multiKwConsFrames = info.optInt("cons_frames", 2);
                String modelFile = info.getString("model_file");

                wakeWordNames = keywords;
                multiKwSession = loadModel(context, modelFile);
                Log.i(TAG, "Multi-keyword mode: " + keywords.length + " keywords in 1 model ("
                        + modelFile + ")");

            // ── Legacy: multi-model ──
            } else if (info.optBoolean("multi_model", false) && info.has("models")) {
                JSONArray modelArray = info.getJSONArray("models");
                for (int i = 0; i < modelArray.length(); i++) {
                    JSONObject m = modelArray.getJSONObject(i);
                    String word = m.getString("wake_word");
                    String file = m.getString("model_file");
                    int cf = m.optInt("cons_frames", 5);
                    models.add(new ModelSlot(word, file, cf));
                    Log.i(TAG, "Registered: " + word + " file=" + file + " cons_frames=" + cf);
                }
                wakeWordNames = new String[models.size()];
                for (int i = 0; i < models.size(); i++) {
                    wakeWordNames[i] = models.get(i).wakeWord;
                }

            // ── Legacy: single model ──
            } else {
                String word = info.getString("wake_word");
                String file = info.getString("model_file");
                int cf = info.optInt("cons_frames", 5);
                models.add(new ModelSlot(word, file, cf));
                wakeWordNames = new String[]{word};
                Log.i(TAG, "Single model: " + word + " cons_frames=" + cf);
            }

            melFramesNeeded = dscnnMelTime;
            audioSamplesNeeded = melFramesNeeded * MEL_HOP_SAMPLES + (int) (SAMPLE_RATE * MEL_WIN_SEC);
            Log.i(TAG, "melFramesNeeded=" + melFramesNeeded
                    + " audioSamplesNeeded=" + audioSamplesNeeded);

            // Load mel model + classifier(s)
            melSession = loadModel(context, MEL_MODEL);
            if (!isMultiKeyword) {
                for (ModelSlot m : models) {
                    m.session = loadModel(context, m.modelFile);
                    bindModelInputShape(m.session, m.modelFile);
                }
            } else if (multiKwSession != null) {
                // model_file was read inside the multi_keyword branch; re-derive
                // the name for the error message only.
                bindModelInputShape(multiKwSession, "multi_keyword model");
            }

            loaded = true;
            Log.i(TAG, "All models loaded (" + (isMultiKeyword ? "multi-kw" : "multi-model") + ")");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load models — check assets/ for model_info.json + .onnx files", e);
            errorMessage = e.getMessage();
            loaded = false;
        }
    }

    /**
     * Bind runtime input dims from the classifier ONNX at load time:
     * mel_time must match model_info.json (fail-fast, ORT would otherwise only
     * fail on the first inference), and the last dim is the real mel width
     * (32 = current-gen tcn; 34 = legacy DS-CNN with hidden padding).
     */
    private void bindModelInputShape(OrtSession session, String modelFile) throws OrtException {
        ai.onnxruntime.TensorInfo tinfo = (ai.onnxruntime.TensorInfo)
                session.getInputInfo().values().iterator().next().getInfo();
        long[] shape = tinfo.getShape();
        boolean timeMatch = false;
        for (long d : shape) {
            if (d == dscnnMelTime) { timeMatch = true; break; }
        }
        if (!timeMatch) {
            throw new OrtException("mel_time mismatch: model_info.json mel_time="
                    + dscnnMelTime + " but " + modelFile + " input shape="
                    + java.util.Arrays.toString(shape));
        }
        long last = shape[shape.length - 1];
        int width = (last > 0) ? (int) last : RAW_MELS;
        if (inputMels != N_MELS && inputMels != width) {
            throw new OrtException("model bundle mixes different input widths: "
                    + inputMels + " vs " + width + " (" + modelFile + ")");
        }
        inputMels = width;
    }

    private OrtSession loadModel(Context context, String filename) throws IOException, OrtException {
        AssetManager am = context.getAssets();
        byte[] modelBytes;
        try (InputStream is = am.open(filename)) {
            modelBytes = new byte[is.available()];
            int offset = 0;
            while (offset < modelBytes.length) {
                int read = is.read(modelBytes, offset, modelBytes.length - offset);
                if (read < 0) break;
                offset += read;
            }
        }
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        return env.createSession(modelBytes, opts);
    }

    public boolean isLoaded() { return loaded; }
    public String getErrorMessage() { return errorMessage; }

    /**
     * Run inference on raw 16-bit PCM audio.
     *
     * @param audio 16-bit mono PCM, 16 kHz
     * @return best matching DetectionResult or null on error
     */
    public DetectionResult process(short[] audio) {
        if (!loaded) return null;

        try {
            // 1. Convert to float
            float[] floatAudio = new float[audio.length];
            for (int i = 0; i < audio.length; i++) {
                floatAudio[i] = (float) audio[i];
            }

            // 2. Mel spectrogram
            OnnxTensor melIn = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(floatAudio), new long[]{1, audio.length});
            OrtSession.Result melOut = melSession.run(
                    Collections.singletonMap("input", melIn));
            float[][][][] mel = (float[][][][]) melOut.get(0).getValue();
            melIn.close();
            melOut.close();

            int frames = mel[0][0].length;
            if (frames < dscnnMelTime / 4) return null;  // need at least some frames

            // 3. Apply transform: x/10 + 2
            float[][] mel2d = new float[frames][RAW_MELS];
            for (int f = 0; f < frames; f++) {
                for (int m = 0; m < RAW_MELS; m++) {
                    mel2d[f][m] = mel[0][0][f][m] / 10.0f + 2.0f;
                }
            }

            // 4. Prepare classifier input with hidden zero-channel padding
            // Skip first 3s to avoid cold-start false triggers
            if (System.currentTimeMillis() - engineStartTime < STARTUP_SKIP_MS) return null;

            int melStart = Math.max(0, frames - dscnnMelTime);
            float[][][] dscnnInput = new float[1][dscnnMelTime][inputMels];
            for (int f = 0; f < dscnnMelTime; f++) {
                int srcF = melStart + f;
                if (srcF >= 0 && srcF < frames) {
                    // Copy first RAW_MELS channels, last HIDDEN_PAD stay zero
                    System.arraycopy(mel2d[srcF], 0, dscnnInput[0][f], 0, RAW_MELS);
                }
            }

            // Flatten to 1D (includes hidden padding)
            float[] flatInput = new float[dscnnMelTime * inputMels];
            for (int f = 0; f < dscnnMelTime; f++) {
                System.arraycopy(dscnnInput[0][f], 0, flatInput, f * inputMels, inputMels);
            }

            // 5. Run classifier(s)
            float bestSigmoid = 0;
            String bestWord = null;
            int bestConsFrames = 2;

            if (isMultiKeyword) {
                // ── New: single multi-keyword model → [1, N] output ──
                OnnxTensor kwIn = OnnxTensor.createTensor(env,
                        java.nio.FloatBuffer.wrap(flatInput),
                        new long[]{1, dscnnMelTime, inputMels});
                OrtSession.Result kwOut = multiKwSession.run(
                        Collections.singletonMap("input", kwIn));
                float[][] scored = (float[][]) kwOut.get(0).getValue();  // [1, N]
                kwIn.close();
                kwOut.close();

                for (int i = 0; i < keywords.length; i++) {
                    if (scored[0][i] > bestSigmoid) {
                        bestSigmoid = scored[0][i];
                        bestWord = keywords[i];
                    }
                }
                bestConsFrames = multiKwConsFrames;

                // Debug: log top-3 predictions
                if (debugLogCount < 200) {
                    debugLogCount++;
                    // Find top 3
                    int[] topIdx = new int[]{-1, -1, -1};
                    float[] topVal = new float[]{-1, -1, -1};
                    for (int i = 0; i < keywords.length; i++) {
                        float v = scored[0][i];
                        if (v > topVal[0]) { topVal[2] = topVal[1]; topIdx[2] = topIdx[1];
                            topVal[1] = topVal[0]; topIdx[1] = topIdx[0];
                            topVal[0] = v; topIdx[0] = i; }
                        else if (v > topVal[1]) { topVal[2] = topVal[1]; topIdx[2] = topIdx[1];
                            topVal[1] = v; topIdx[1] = i; }
                        else if (v > topVal[2]) { topVal[2] = v; topIdx[2] = i; }
                    }
                    float melMean = 0;
                    for (int f = 0; f < dscnnMelTime; f++)
                        for (int m = 0; m < inputMels; m++)
                            melMean += dscnnInput[0][f][m];
                    melMean /= (dscnnMelTime * inputMels);
                    Log.d(TAG, String.format(Locale.US,
                            "[Multi-KW] top1=%s(%.3f) top2=%s(%.3f) top3=%s(%.3f) melMean=%.1f",
                            keywords[topIdx[0]], topVal[0],
                            topIdx[1] >= 0 ? keywords[topIdx[1]] : "-", topVal[1],
                            topIdx[2] >= 0 ? keywords[topIdx[2]] : "-", topVal[2],
                            melMean));
                }

            } else {
                // ── Legacy: loop over multiple single-keyword models ──
                for (ModelSlot model : models) {
                    OnnxTensor dscnnIn = OnnxTensor.createTensor(env,
                            java.nio.FloatBuffer.wrap(flatInput),
                            new long[]{1, dscnnMelTime, inputMels});
                    OrtSession.Result dscnnOut = model.session.run(
                            Collections.singletonMap("input", dscnnIn));
                    Object rawOut = dscnnOut.get(0).getValue();
                    float sigmoid;
                    if (rawOut instanceof float[][]) {
                        float[][] score = (float[][]) rawOut;
                        sigmoid = score[0][0];
                    } else if (rawOut instanceof float[]) {
                        float[] score = (float[]) rawOut;
                        sigmoid = score[0];
                    } else {
                        Log.e(TAG, "DS-CNN unexpected output type: " + rawOut.getClass().getName());
                        sigmoid = 0f;
                    }
                    if (sigmoid > bestSigmoid) {
                        bestSigmoid = sigmoid;
                        bestWord = model.wakeWord;
                        bestConsFrames = model.consFrames;
                    }
                    dscnnIn.close();
                    dscnnOut.close();
                }
            }

            // Log first 20 inferences (legacy debug)
            if (!isMultiKeyword && debugLogCount < 20) {
                debugLogCount++;
                float melSum = 0, melMin = Float.MAX_VALUE, melMax = -Float.MAX_VALUE;
                for (int f = 0; f < dscnnMelTime; f++) {
                    for (int m = 0; m < inputMels; m++) {
                        float v = dscnnInput[0][f][m];
                        melSum += v;
                        if (v < melMin) melMin = v;
                        if (v > melMax) melMax = v;
                    }
                }
                float m = melSum / (dscnnMelTime * inputMels);
                Log.d(TAG, String.format(Locale.US,
                        "[DS-CNN] %d models sig=%.4f word=%s melMean=%.2f melMin=%.2f melMax=%.2f",
                        models.size(), bestSigmoid, bestWord != null ? bestWord : "-",
                        m, melMin, melMax));
            }

            float bgProb = 1.0f - bestSigmoid;
            String detected = bestSigmoid > 0.5f ? bestWord : null;
            return new DetectionResult(detected, bestSigmoid, bgProb, bestConsFrames);

        } catch (OrtException e) {
            Log.e(TAG, "Inference error", e);
            return null;
        }
    }

    public void close() {
        try {
            if (melSession != null) melSession.close();
            if (multiKwSession != null) multiKwSession.close();
            for (ModelSlot m : models) {
                if (m.session != null) m.session.close();
            }
            if (env != null) env.close();
        } catch (OrtException e) {
            Log.e(TAG, "Error closing sessions", e);
        }
    }
}
