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
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int bestClass = 0;
                        float bestScore = Float.NEGATIVE_INFINITY;
                        for (int c = 0; c < classes; c++) {
                            float score = logits[0][c][y][x];
                            if (score > bestScore) {
                                bestScore = score;
                                bestClass = c;
                            }
                        }
                        mask[y * width + x] = regionBlendWeight(bestClass);
                    }
                }
                if (width == outputWidth && height == outputHeight) return mask;
                return resizeMaskNearest(mask, width, height, outputWidth, outputHeight);
            }
        } catch (Exception e) {
            Log.w(TAG, "Semantic face parsing failed; using geometric blend mask", e);
            return null;
        } finally {
            if (resized != alignedFace && !resized.isRecycled()) resized.recycle();
        }
    }

    /**
     * Graduated semantic blending instead of binary cut-outs. Skin/brows/nose/lips carry
     * identity strongly, while the target's gaze and mouth interior remain dominant.
     */
    private float regionBlendWeight(int label) {
        switch (label) {
            case 1:  // skin
            case 2:  // left brow
            case 3:  // right brow
            case 10: // nose
            case 12: // upper lip
            case 13: // lower lip
                return 1.0f;
            case 4:  // left eye
            case 5:  // right eye
                return 0.14f;
            case 11: // mouth cavity / teeth
                return 0.08f;
            default:
                // Preserve target glasses, ears, neck, hair, hat, clothing and background.
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

    private float[] resizeMaskNearest(float[] source, int sourceWidth, int sourceHeight, int width, int height) {
        float[] result = new float[width * height];
        for (int y = 0; y < height; y++) {
            int sy = Math.min(sourceHeight - 1, y * sourceHeight / height);
            for (int x = 0; x < width; x++) {
                int sx = Math.min(sourceWidth - 1, x * sourceWidth / width);
                result[y * width + x] = source[sy * sourceWidth + sx];
            }
        }
        return result;
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
