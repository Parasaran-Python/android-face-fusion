package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/** HyperSwap 1a 256 implementation matching FaceFusion's model contract. */
public class FaceSwapper {
    private static final String TAG = "FaceSwapper";
    private static final int HYPERSWAP_SIZE = 256;
    public static final int QUALITY_SIZE = 512;

    private final Context context;
    private final OrtEnvironment env;
    private OrtSession session;
    private String imageInputName;
    private String embeddingInputName;

    public FaceSwapper(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    Context getAppContext() {
        return context;
    }

    public void initialize() throws Exception {
        ModelDownloader downloader = new ModelDownloader(context);
        Log.i(TAG, "Loading HyperSwap 1a 256...");
        File model = downloader.getModelFile(ModelDownloader.HYPERSWAP_MODEL);

        try {
            session = OrtSessionHelper.createSession(env, model.getAbsolutePath(), TAG);
            resolveInputNames();
            validateHyperSwapInputs();
            Log.i(TAG, "HyperSwap 1a 256 initialized successfully: " + model.getAbsolutePath());
        } catch (Exception e) {
            closeSessionOnly();
            throw new Exception("HyperSwap 1a 256 failed to initialize: " + e.getMessage(), e);
        }
    }

    public int getInputSize() {
        return HYPERSWAP_SIZE;
    }

    public boolean isUsingHyperSwap() {
        return session != null;
    }

    private void resolveInputNames() throws Exception {
        imageInputName = null;
        embeddingInputName = null;

        if (session.getInputInfo().containsKey("target")) imageInputName = "target";
        if (session.getInputInfo().containsKey("source")) embeddingInputName = "source";

        if (imageInputName == null || embeddingInputName == null) {
            for (Map.Entry<String, NodeInfo> entry : session.getInputInfo().entrySet()) {
                TensorInfo info = (TensorInfo) entry.getValue().getInfo();
                long[] shape = info.getShape();
                if (shape.length == 4 && imageInputName == null) imageInputName = entry.getKey();
                if (shape.length == 2 && embeddingInputName == null) embeddingInputName = entry.getKey();
            }
        }

        if (imageInputName == null || embeddingInputName == null) {
            throw new Exception("Could not resolve HyperSwap source/target inputs");
        }
        Log.i(TAG, "HyperSwap inputs: source=" + embeddingInputName + ", target=" + imageInputName);
    }

    private void validateHyperSwapInputs() throws Exception {
        NodeInfo imageNode = session.getInputInfo().get(imageInputName);
        NodeInfo embeddingNode = session.getInputInfo().get(embeddingInputName);
        if (imageNode == null || embeddingNode == null) throw new Exception("HyperSwap inputs missing");

        long[] imageShape = ((TensorInfo) imageNode.getInfo()).getShape();
        long[] embeddingShape = ((TensorInfo) embeddingNode.getInfo()).getShape();
        if (imageShape.length != 4 || imageShape[1] != 3
            || (imageShape[2] > 0 && imageShape[2] != HYPERSWAP_SIZE)
            || (imageShape[3] > 0 && imageShape[3] != HYPERSWAP_SIZE)) {
            throw new Exception("Unexpected HyperSwap target tensor shape");
        }
        if (embeddingShape.length != 2 || (embeddingShape[1] > 0 && embeddingShape[1] != 512)) {
            throw new Exception("Unexpected HyperSwap source tensor shape");
        }
    }

    public Bitmap swapFace(Bitmap targetFace, float[] sourceEmbedding, Bitmap targetImage) throws OrtException {
        if (session == null) throw new IllegalStateException("HyperSwap is not initialized");
        if (sourceEmbedding == null || sourceEmbedding.length != 512) {
            throw new IllegalArgumentException("Expected a 512-dimensional ArcFace embedding");
        }

        Bitmap resizedTarget = targetFace;
        if (targetFace.getWidth() != HYPERSWAP_SIZE || targetFace.getHeight() != HYPERSWAP_SIZE) {
            resizedTarget = Bitmap.createScaledBitmap(targetFace, HYPERSWAP_SIZE, HYPERSWAP_SIZE, true);
        }

        float[] targetData = bitmapToHyperSwapArray(resizedTarget);
        float[] sourceData = l2Normalize(sourceEmbedding);

        try (OnnxTensor targetTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(targetData), new long[]{1, 3, HYPERSWAP_SIZE, HYPERSWAP_SIZE});
             OnnxTensor sourceTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(sourceData), new long[]{1, 512})) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(embeddingInputName, sourceTensor);
            inputs.put(imageInputName, targetTensor);

            try (OrtSession.Result results = session.run(inputs)) {
                Object value = results.get(0).getValue();
                if (!(value instanceof float[][][][])) {
                    throw new OrtException("Unexpected HyperSwap output type: " + value.getClass().getName());
                }
                return hyperSwapOutputToBitmap(((float[][][][]) value)[0]);
            }
        } finally {
            if (resizedTarget != targetFace && !resizedTarget.isRecycled()) resizedTarget.recycle();
        }
    }

    /**
     * FaceFusion pixel boost at 512px. This is not a simple quadrant split: it uses the
     * same interleaved reshape/transpose layout as FaceFusion's implode/explode helpers.
     */
    public Bitmap swapFace512(Bitmap targetFace512, float[] sourceEmbedding) throws OrtException {
        Bitmap input = targetFace512;
        if (targetFace512.getWidth() != QUALITY_SIZE || targetFace512.getHeight() != QUALITY_SIZE) {
            input = Bitmap.createScaledBitmap(targetFace512, QUALITY_SIZE, QUALITY_SIZE, true);
        }
        Bitmap[] tiles = implodePixelBoost(input);
        Bitmap[] swappedTiles = new Bitmap[4];
        try {
            for (int i = 0; i < 4; i++) {
                swappedTiles[i] = swapFace(tiles[i], sourceEmbedding, input);
            }
            return explodePixelBoost(swappedTiles);
        } finally {
            for (Bitmap tile : tiles) {
                if (tile != null && !tile.isRecycled()) tile.recycle();
            }
            for (Bitmap tile : swappedTiles) {
                if (tile != null && !tile.isRecycled()) tile.recycle();
            }
            if (input != targetFace512 && !input.isRecycled()) input.recycle();
        }
    }

    private Bitmap[] implodePixelBoost(Bitmap input) {
        int[] source = new int[QUALITY_SIZE * QUALITY_SIZE];
        input.getPixels(source, 0, QUALITY_SIZE, 0, 0, QUALITY_SIZE, QUALITY_SIZE);
        Bitmap[] result = new Bitmap[4];
        for (int offsetY = 0; offsetY < 2; offsetY++) {
            for (int offsetX = 0; offsetX < 2; offsetX++) {
                int[] pixels = new int[HYPERSWAP_SIZE * HYPERSWAP_SIZE];
                for (int y = 0; y < HYPERSWAP_SIZE; y++) {
                    int sourceY = y * 2 + offsetY;
                    for (int x = 0; x < HYPERSWAP_SIZE; x++) {
                        int sourceX = x * 2 + offsetX;
                        pixels[y * HYPERSWAP_SIZE + x] = source[sourceY * QUALITY_SIZE + sourceX];
                    }
                }
                int index = offsetY * 2 + offsetX;
                Bitmap tile = Bitmap.createBitmap(HYPERSWAP_SIZE, HYPERSWAP_SIZE, Bitmap.Config.ARGB_8888);
                tile.setPixels(pixels, 0, HYPERSWAP_SIZE, 0, 0, HYPERSWAP_SIZE, HYPERSWAP_SIZE);
                result[index] = tile;
            }
        }
        return result;
    }

    private Bitmap explodePixelBoost(Bitmap[] tiles) {
        int[][] tilePixels = new int[4][];
        for (int i = 0; i < 4; i++) {
            tilePixels[i] = new int[HYPERSWAP_SIZE * HYPERSWAP_SIZE];
            tiles[i].getPixels(tilePixels[i], 0, HYPERSWAP_SIZE, 0, 0,
                HYPERSWAP_SIZE, HYPERSWAP_SIZE);
        }
        int[] output = new int[QUALITY_SIZE * QUALITY_SIZE];
        for (int offsetY = 0; offsetY < 2; offsetY++) {
            for (int offsetX = 0; offsetX < 2; offsetX++) {
                int index = offsetY * 2 + offsetX;
                int[] pixels = tilePixels[index];
                for (int y = 0; y < HYPERSWAP_SIZE; y++) {
                    int outputY = y * 2 + offsetY;
                    for (int x = 0; x < HYPERSWAP_SIZE; x++) {
                        int outputX = x * 2 + offsetX;
                        output[outputY * QUALITY_SIZE + outputX] = pixels[y * HYPERSWAP_SIZE + x];
                    }
                }
            }
        }
        Bitmap result = Bitmap.createBitmap(QUALITY_SIZE, QUALITY_SIZE, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, QUALITY_SIZE, 0, 0, QUALITY_SIZE, QUALITY_SIZE);
        return result;
    }

    private float[] bitmapToHyperSwapArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] output = new float[3 * pixels.length];

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            output[i] = ((((pixel >> 16) & 0xFF) / 255.0f) - 0.5f) / 0.5f;
            output[pixels.length + i] = ((((pixel >> 8) & 0xFF) / 255.0f) - 0.5f) / 0.5f;
            output[2 * pixels.length + i] = (((pixel & 0xFF) / 255.0f) - 0.5f) / 0.5f;
        }
        return output;
    }

    private Bitmap hyperSwapOutputToBitmap(float[][][] data) throws OrtException {
        if (data == null || data.length < 3 || data[0].length == 0 || data[0][0].length == 0) {
            throw new OrtException("HyperSwap returned an empty output tensor");
        }

        int height = data[0].length;
        int width = data[0][0].length;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float value = data[c][y][x];
                    if (!Float.isFinite(value)) {
                        bitmap.recycle();
                        throw new OrtException("HyperSwap returned NaN/Infinity output");
                    }
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }
        }

        boolean normalizedOutput = min < 0.0f || max <= 1.5f;
        Log.i(TAG, "HyperSwap output range: min=" + min + ", max=" + max
            + ", normalized=" + normalizedOutput);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float rf = data[0][y][x];
                float gf = data[1][y][x];
                float bf = data[2][y][x];

                if (normalizedOutput) {
                    rf = (rf * 0.5f + 0.5f) * 255.0f;
                    gf = (gf * 0.5f + 0.5f) * 255.0f;
                    bf = (bf * 0.5f + 0.5f) * 255.0f;
                }

                int r = clampToByte(rf);
                int g = clampToByte(gf);
                int b = clampToByte(bf);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private float[] l2Normalize(float[] embedding) {
        float normSq = 0.0f;
        for (float value : embedding) normSq += value * value;
        float norm = (float) Math.sqrt(normSq);
        if (norm <= 1e-12f || !Float.isFinite(norm)) {
            throw new IllegalArgumentException("Invalid ArcFace source embedding");
        }
        float[] normalized = new float[embedding.length];
        for (int i = 0; i < embedding.length; i++) normalized[i] = embedding[i] / norm;
        return normalized;
    }

    private int clampToByte(float value) {
        return Math.max(0, Math.min(255, Math.round(value)));
    }

    private void closeSessionOnly() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.w(TAG, "Error closing HyperSwap session", e);
            }
            session = null;
        }
    }

    public void close() {
        closeSessionOnly();
    }
}
