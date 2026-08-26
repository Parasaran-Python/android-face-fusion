package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/** FaceFusion-compatible BiSeNet semantic face parser used for natural paste-back masks. */
public final class FaceParser {
    private static final String TAG = "FaceParser";
    private static final int INPUT_SIZE = 512;

    private final Context context;
    private final OrtEnvironment env;
    private OrtSession session;
    private String inputName;

    public FaceParser(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize() throws Exception {
        ModelDownloader downloader = new ModelDownloader(context);
        File model = downloader.getModelFile(ModelDownloader.PARSER_MODEL);
        session = OrtSessionHelper.createSession(env, model.getAbsolutePath(), TAG);
        inputName = session.getInputNames().iterator().next();
        Log.i(TAG, "BiSeNet face parser initialized");
    }

    public float[] createMask(Bitmap alignedFace) {
        if (session == null || alignedFace == null) return null;
        int outputWidth = alignedFace.getWidth();
        int outputHeight = alignedFace.getHeight();
        Bitmap resized = alignedFace;
        if (outputWidth != INPUT_SIZE || outputHeight != INPUT_SIZE) {
            resized = Bitmap.createScaledBitmap(alignedFace, INPUT_SIZE, INPUT_SIZE, true);
        }
        try {
            float[] input = bitmapToInput(resized);
            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(input), new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
                 OrtSession.Result outputs = session.run(Collections.singletonMap(inputName, tensor))) {
                Object value = outputs.get(0).getValue();
                if (!(value instanceof float[][][][])) {
                    Log.w(TAG, "Unexpected parser output type: " + value.getClass().getName());
                    return null;
                }
                float[][][][] logits = (float[][][][]) value;
                if (logits.length < 1 || logits[0].length < 14) return null;
                int classes = logits[0].length;
                int height = logits[0][0].length;
                int width = logits[0][0][0].length;
                float[] mask = new float[width * height];

                // Use the full probability distribution instead of an argmax class label. Pixels
                // near skin/hair/eye/lip boundaries become naturally fractional rather than hard cuts.
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        float maxLogit = Float.NEGATIVE_INFINITY;
                        for (int c = 0; c < classes; c++) {
                            maxLogit = Math.max(maxLogit, logits[0][c][y][x]);
                        }
                        double probabilitySum = 0.0;
                        double weightedBlend = 0.0;
                        for (int c = 0; c < classes; c++) {
                            double probability = Math.exp(Math.max(-20.0, logits[0][c][y][x] - maxLogit));
                            probabilitySum += probability;
                            weightedBlend += probability * regionBlendWeight(c);
                        }
                        mask[y * width + x] = probabilitySum > 0.0
                            ? clamp01((float) (weightedBlend / probabilitySum)) : 0.0f;
                    }
                }
                if (width == outputWidth && height == outputHeight) return mask;
                return resizeMaskBilinear(mask, width, height, outputWidth, outputHeight);
            }
        } catch (Exception e) {
            Log.w(TAG, "Semantic face parsing failed; using geometric blend mask", e);
            return null;
        } finally {
            if (resized != alignedFace && !resized.isRecycled()) resized.recycle();
        }
    }

    /**
     * Base semantic identity weights. Brows now favour the generated face so the target brow
     * is actually overwritten instead of surviving underneath as a second visible eyebrow.
     * Eyes and mouth cavity remain target-led because gaze, blinking and teeth are expression cues.
     */
    private float regionBlendWeight(int label) {
        switch (label) {
            case 1:  // skin
            case 10: // nose
            case 12: // upper lip
            case 13: // lower lip
                return 1.0f;
            case 2:  // left brow
            case 3:  // right brow
                return 0.72f;
            case 4:  // left eye
            case 5:  // right eye
                return 0.20f;
            case 11: // mouth cavity / teeth
                return 0.12f;
            case 6:  // glasses: retain mostly target while allowing a tiny transition
                return 0.04f;
            default:
                // Preserve target ears, neck, hair, hat, clothing and background.
                return 0.0f;
        }
    }

    private float[] bitmapToInput(Bitmap bitmap) {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        int plane = pixels.length;
        float[] output = new float[plane * 3];
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        for (int i = 0; i < plane; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            output[i] = (r - mean[0]) / std[0];
            output[plane + i] = (g - mean[1]) / std[1];
            output[plane * 2 + i] = (b - mean[2]) / std[2];
        }
        return output;
    }

    private float[] resizeMaskBilinear(float[] source, int sourceWidth, int sourceHeight, int width, int height) {
        float[] result = new float[width * height];
        float xScale = width > 1 ? (float) (sourceWidth - 1) / (width - 1) : 0.0f;
        float yScale = height > 1 ? (float) (sourceHeight - 1) / (height - 1) : 0.0f;
        for (int y = 0; y < height; y++) {
            float sy = y * yScale;
            int y0 = (int) sy;
            int y1 = Math.min(sourceHeight - 1, y0 + 1);
            float fy = sy - y0;
            for (int x = 0; x < width; x++) {
                float sx = x * xScale;
                int x0 = (int) sx;
                int x1 = Math.min(sourceWidth - 1, x0 + 1);
                float fx = sx - x0;
                float top = source[y0 * sourceWidth + x0] * (1.0f - fx)
                    + source[y0 * sourceWidth + x1] * fx;
                float bottom = source[y1 * sourceWidth + x0] * (1.0f - fx)
                    + source[y1 * sourceWidth + x1] * fx;
                result[y * width + x] = clamp01(top * (1.0f - fy) + bottom * fy);
            }
        }
        return result;
    }

    private float clamp01(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }

    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (Exception e) {
                Log.w(TAG, "Error closing parser", e);
            }
            session = null;
        }
    }
}
