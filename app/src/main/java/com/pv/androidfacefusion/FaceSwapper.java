package com.pv.androidfacefusion;

import android.content.Context;
import android.content.SharedPreferences;
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

/** Selectable high-quality face swapper supporting HyperSwap 1a/1c and native SimSwap 512. */
public class FaceSwapper {
    private static final String TAG = "FaceSwapper";
    private static final String PREFS = "face_fusion_quality";
    private static final String PREF_MODEL = "swapper_model";

    private static final int HYPERSWAP_SIZE = 256;
    private static final int SIMSWAP_SIZE = 512;
    public static final int QUALITY_SIZE = 768;
    private static final int PIXEL_BOOST_TOTAL = QUALITY_SIZE / HYPERSWAP_SIZE;

    public enum ModelChoice {
        HYPERSWAP_1A("hyperswap_1a", "HyperSwap 1a", "Stable 256 model • 768 pixel boost"),
        HYPERSWAP_1C("hyperswap_1c", "HyperSwap 1c", "Newer 256 model • 768 pixel boost"),
        SIMSWAP_512("simswap_512", "SimSwap 512", "Native 512 model • non-commercial model license");

        public final String id;
        public final String displayName;
        public final String description;

        ModelChoice(String id, String displayName, String description) {
            this.id = id;
            this.displayName = displayName;
            this.description = description;
        }

        static ModelChoice fromId(String id) {
            for (ModelChoice choice : values()) {
                if (choice.id.equals(id)) return choice;
            }
            return HYPERSWAP_1A;
        }
    }

    private final Context context;
    private final OrtEnvironment env;
    private OrtSession session;
    private OrtSession embeddingConverterSession;
    private String imageInputName;
    private String embeddingInputName;
    private String converterInputName;
    private ModelChoice activeModel;

    public FaceSwapper(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    Context getAppContext() {
        return context;
    }

    public static ModelChoice getSelectedModel(Context context) {
        SharedPreferences prefs = context.getApplicationContext()
            .getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        return ModelChoice.fromId(prefs.getString(PREF_MODEL, ModelChoice.HYPERSWAP_1A.id));
    }

    public static void setSelectedModel(Context context, ModelChoice choice) {
        context.getApplicationContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(PREF_MODEL, choice.id).apply();
    }

    public void initialize() throws Exception {
        ensureModelLoaded(getSelectedModel(context));
    }

    public int getInputSize() {
        return activeModel == ModelChoice.SIMSWAP_512 ? SIMSWAP_SIZE : HYPERSWAP_SIZE;
    }

    public String getActiveModelName() {
        return activeModel != null ? activeModel.displayName : getSelectedModel(context).displayName;
    }

    public boolean isUsingHyperSwap() {
        return session != null && activeModel != ModelChoice.SIMSWAP_512;
    }

    private synchronized void ensureSelectedModelLoaded() throws Exception {
        ensureModelLoaded(getSelectedModel(context));
    }

    private synchronized void ensureModelLoaded(ModelChoice choice) throws Exception {
        if (session != null && activeModel == choice) return;

        ModelDownloader downloader = new ModelDownloader(context);
        File swapperFile;
        File converterFile = null;
        if (choice == ModelChoice.HYPERSWAP_1C) {
            swapperFile = downloader.getModelFile(ModelDownloader.HYPERSWAP_1C_MODEL);
        } else if (choice == ModelChoice.SIMSWAP_512) {
            swapperFile = downloader.getModelFile(ModelDownloader.SIMSWAP_512_MODEL);
            converterFile = downloader.getModelFile(ModelDownloader.CROSSFACE_SIMSWAP_MODEL);
        } else {
            swapperFile = downloader.getModelFile(ModelDownloader.HYPERSWAP_MODEL);
        }

        OrtSession newSession = null;
        OrtSession newConverter = null;
        try {
            Log.i(TAG, "Loading " + choice.displayName + "...");
            newSession = OrtSessionHelper.createSession(env, swapperFile.getAbsolutePath(), TAG);
            String[] resolvedInputs = resolveInputNames(newSession);
            validateInputs(newSession, resolvedInputs[0], resolvedInputs[1], choice);

            String resolvedConverterInput = null;
            if (choice == ModelChoice.SIMSWAP_512) {
                newConverter = OrtSessionHelper.createSession(env, converterFile.getAbsolutePath(), TAG + "Converter");
                resolvedConverterInput = newConverter.getInputNames().contains("input")
                    ? "input" : newConverter.getInputNames().iterator().next();
            }

            closeSessions();
            session = newSession;
            embeddingConverterSession = newConverter;
            imageInputName = resolvedInputs[0];
            embeddingInputName = resolvedInputs[1];
            converterInputName = resolvedConverterInput;
            activeModel = choice;
            newSession = null;
            newConverter = null;
            Log.i(TAG, choice.displayName + " initialized successfully");
        } catch (Exception e) {
            if (newSession != null) {
                try { newSession.close(); } catch (Exception ignored) {}
            }
            if (newConverter != null) {
                try { newConverter.close(); } catch (Exception ignored) {}
            }
            throw new Exception(choice.displayName + " failed to initialize: " + e.getMessage(), e);
        }
    }

    private String[] resolveInputNames(OrtSession targetSession) throws Exception {
        String image = targetSession.getInputInfo().containsKey("target") ? "target" : null;
        String embedding = targetSession.getInputInfo().containsKey("source") ? "source" : null;

        if (image == null || embedding == null) {
            for (Map.Entry<String, NodeInfo> entry : targetSession.getInputInfo().entrySet()) {
                TensorInfo info = (TensorInfo) entry.getValue().getInfo();
                long[] shape = info.getShape();
                if (shape.length == 4 && image == null) image = entry.getKey();
                if (shape.length == 2 && embedding == null) embedding = entry.getKey();
            }
        }
        if (image == null || embedding == null) {
            throw new Exception("Could not resolve source/target model inputs");
        }
        return new String[]{image, embedding};
    }

    private void validateInputs(OrtSession targetSession, String imageName, String embeddingName,
                                ModelChoice choice) throws Exception {
        NodeInfo imageNode = targetSession.getInputInfo().get(imageName);
        NodeInfo embeddingNode = targetSession.getInputInfo().get(embeddingName);
        if (imageNode == null || embeddingNode == null) throw new Exception("Swapper inputs missing");

        int expectedSize = choice == ModelChoice.SIMSWAP_512 ? SIMSWAP_SIZE : HYPERSWAP_SIZE;
        long[] imageShape = ((TensorInfo) imageNode.getInfo()).getShape();
        long[] embeddingShape = ((TensorInfo) embeddingNode.getInfo()).getShape();
        if (imageShape.length != 4 || imageShape[1] != 3
                || (imageShape[2] > 0 && imageShape[2] != expectedSize)
                || (imageShape[3] > 0 && imageShape[3] != expectedSize)) {
            throw new Exception("Unexpected " + choice.displayName + " target tensor shape");
        }
        if (embeddingShape.length != 2 || (embeddingShape[1] > 0 && embeddingShape[1] != 512)) {
            throw new Exception("Unexpected " + choice.displayName + " source tensor shape");
        }
    }

    public Bitmap swapFace(Bitmap targetFace, float[] sourceEmbedding, Bitmap targetImage) throws OrtException {
        try {
            ensureSelectedModelLoaded();
        } catch (Exception e) {
            throw new OrtException("Could not load selected swapper: " + e.getMessage());
        }
        int nativeSize = getInputSize();
        Bitmap resizedTarget = targetFace;
        if (targetFace.getWidth() != nativeSize || targetFace.getHeight() != nativeSize) {
            resizedTarget = Bitmap.createScaledBitmap(targetFace, nativeSize, nativeSize, true);
        }
        try {
            return runNativeSwap(resizedTarget, sourceEmbedding);
        } finally {
            if (resizedTarget != targetFace && !resizedTarget.isRecycled()) resizedTarget.recycle();
        }
    }

    /**
     * High-quality path. HyperSwap runs FaceFusion's exact 768 interleaved pixel boost.
     * SimSwap sees one native 512 face and is reprojected back into the same 768 HyperSwap
     * canonical space so the existing semantic/pose compositor remains unchanged.
     */
    public Bitmap swapFaceQuality(Bitmap targetFace, float[] sourceEmbedding) throws OrtException {
        try {
            ensureSelectedModelLoaded();
        } catch (Exception e) {
            throw new OrtException("Could not load selected swapper: " + e.getMessage());
        }

        Bitmap input = targetFace;
        if (targetFace.getWidth() != QUALITY_SIZE || targetFace.getHeight() != QUALITY_SIZE) {
            input = Bitmap.createScaledBitmap(targetFace, QUALITY_SIZE, QUALITY_SIZE, true);
        }
        try {
            if (activeModel == ModelChoice.SIMSWAP_512) {
                Bitmap simInput = CanonicalWarp.hyperSwapToSimSwap(input, SIMSWAP_SIZE);
                Bitmap simOutput = null;
                try {
                    simOutput = runNativeSwap(simInput, sourceEmbedding);
                    return CanonicalWarp.simSwapToHyperSwap(simOutput, QUALITY_SIZE);
                } finally {
                    if (!simInput.isRecycled()) simInput.recycle();
                    if (simOutput != null && !simOutput.isRecycled()) simOutput.recycle();
                }
            }

            Bitmap[] tiles = implodePixelBoost(input);
            Bitmap[] swappedTiles = new Bitmap[tiles.length];
            try {
                for (int i = 0; i < tiles.length; i++) {
                    swappedTiles[i] = runNativeSwap(tiles[i], sourceEmbedding);
                }
                return explodePixelBoost(swappedTiles);
            } finally {
                for (Bitmap tile : tiles) {
                    if (tile != null && !tile.isRecycled()) tile.recycle();
                }
                for (Bitmap tile : swappedTiles) {
                    if (tile != null && !tile.isRecycled()) tile.recycle();
                }
            }
        } finally {
            if (input != targetFace && !input.isRecycled()) input.recycle();
        }
    }

    private Bitmap runNativeSwap(Bitmap targetFace, float[] sourceEmbedding) throws OrtException {
        if (session == null || activeModel == null) throw new IllegalStateException("Face swapper is not initialized");
        if (sourceEmbedding == null || sourceEmbedding.length != 512) {
            throw new IllegalArgumentException("Expected a 512-dimensional ArcFace embedding");
        }

        int nativeSize = getInputSize();
        float[] targetData = bitmapToModelArray(targetFace, activeModel);
        float[] sourceData = prepareEmbedding(sourceEmbedding, activeModel);

        try (OnnxTensor targetTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(targetData), new long[]{1, 3, nativeSize, nativeSize});
             OnnxTensor sourceTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(sourceData), new long[]{1, 512})) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(embeddingInputName, sourceTensor);
            inputs.put(imageInputName, targetTensor);

            try (OrtSession.Result results = session.run(inputs)) {
                Object value = results.get(0).getValue();
                if (!(value instanceof float[][][][])) {
                    throw new OrtException("Unexpected " + activeModel.displayName + " output type: "
                        + value.getClass().getName());
                }
                return modelOutputToBitmap(((float[][][][]) value)[0], activeModel);
            }
        }
    }

    private float[] prepareEmbedding(float[] sourceEmbedding, ModelChoice choice) throws OrtException {
        if (choice != ModelChoice.SIMSWAP_512) return l2Normalize(sourceEmbedding);
        if (embeddingConverterSession == null || converterInputName == null) {
            throw new OrtException("SimSwap embedding converter is not initialized");
        }

        try (OnnxTensor input = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(sourceEmbedding), new long[]{1, 512});
             OrtSession.Result outputs = embeddingConverterSession.run(
                java.util.Collections.singletonMap(converterInputName, input))) {
            Object value = outputs.get(0).getValue();
            float[] converted = readEmbedding(value);
            if (converted == null || converted.length != 512) {
                throw new OrtException("Unexpected SimSwap embedding converter output");
            }
            return l2Normalize(converted);
        }
    }

    private float[] readEmbedding(Object value) {
        if (value instanceof float[][]) {
            float[][] data = (float[][]) value;
            if (data.length > 0 && data[0].length >= 512) {
                float[] result = new float[512];
                System.arraycopy(data[0], 0, result, 0, 512);
                return result;
            }
        }
        if (value instanceof float[]) {
            float[] data = (float[]) value;
            if (data.length >= 512) {
                float[] result = new float[512];
                System.arraycopy(data, 0, result, 0, 512);
                return result;
            }
        }
        return null;
    }

    private Bitmap[] implodePixelBoost(Bitmap input) {
        int[] source = new int[QUALITY_SIZE * QUALITY_SIZE];
        input.getPixels(source, 0, QUALITY_SIZE, 0, 0, QUALITY_SIZE, QUALITY_SIZE);
        Bitmap[] result = new Bitmap[PIXEL_BOOST_TOTAL * PIXEL_BOOST_TOTAL];
        for (int offsetY = 0; offsetY < PIXEL_BOOST_TOTAL; offsetY++) {
            for (int offsetX = 0; offsetX < PIXEL_BOOST_TOTAL; offsetX++) {
                int[] pixels = new int[HYPERSWAP_SIZE * HYPERSWAP_SIZE];
                for (int y = 0; y < HYPERSWAP_SIZE; y++) {
                    int sourceY = y * PIXEL_BOOST_TOTAL + offsetY;
                    for (int x = 0; x < HYPERSWAP_SIZE; x++) {
                        int sourceX = x * PIXEL_BOOST_TOTAL + offsetX;
                        pixels[y * HYPERSWAP_SIZE + x] = source[sourceY * QUALITY_SIZE + sourceX];
                    }
                }
                int index = offsetY * PIXEL_BOOST_TOTAL + offsetX;
                Bitmap tile = Bitmap.createBitmap(HYPERSWAP_SIZE, HYPERSWAP_SIZE, Bitmap.Config.ARGB_8888);
                tile.setPixels(pixels, 0, HYPERSWAP_SIZE, 0, 0, HYPERSWAP_SIZE, HYPERSWAP_SIZE);
                result[index] = tile;
            }
        }
        return result;
    }

    private Bitmap explodePixelBoost(Bitmap[] tiles) {
        int tileCount = PIXEL_BOOST_TOTAL * PIXEL_BOOST_TOTAL;
        if (tiles == null || tiles.length != tileCount) {
            throw new IllegalArgumentException("Expected " + tileCount + " HyperSwap pixel-boost tiles");
        }
        int[][] tilePixels = new int[tileCount][];
        for (int i = 0; i < tileCount; i++) {
            tilePixels[i] = new int[HYPERSWAP_SIZE * HYPERSWAP_SIZE];
            tiles[i].getPixels(tilePixels[i], 0, HYPERSWAP_SIZE, 0, 0,
                HYPERSWAP_SIZE, HYPERSWAP_SIZE);
        }
        int[] output = new int[QUALITY_SIZE * QUALITY_SIZE];
        for (int offsetY = 0; offsetY < PIXEL_BOOST_TOTAL; offsetY++) {
            for (int offsetX = 0; offsetX < PIXEL_BOOST_TOTAL; offsetX++) {
                int index = offsetY * PIXEL_BOOST_TOTAL + offsetX;
                int[] pixels = tilePixels[index];
                for (int y = 0; y < HYPERSWAP_SIZE; y++) {
                    int outputY = y * PIXEL_BOOST_TOTAL + offsetY;
                    for (int x = 0; x < HYPERSWAP_SIZE; x++) {
                        int outputX = x * PIXEL_BOOST_TOTAL + offsetX;
                        output[outputY * QUALITY_SIZE + outputX] = pixels[y * HYPERSWAP_SIZE + x];
                    }
                }
            }
        }
        Bitmap result = Bitmap.createBitmap(QUALITY_SIZE, QUALITY_SIZE, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, QUALITY_SIZE, 0, 0, QUALITY_SIZE, QUALITY_SIZE);
        return result;
    }

    private float[] bitmapToModelArray(Bitmap bitmap, ModelChoice choice) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] output = new float[3 * pixels.length];

        float mean = choice == ModelChoice.SIMSWAP_512 ? 0.0f : 0.5f;
        float std = choice == ModelChoice.SIMSWAP_512 ? 1.0f : 0.5f;
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            output[i] = ((((pixel >> 16) & 0xFF) / 255.0f) - mean) / std;
            output[pixels.length + i] = ((((pixel >> 8) & 0xFF) / 255.0f) - mean) / std;
            output[2 * pixels.length + i] = (((pixel & 0xFF) / 255.0f) - mean) / std;
        }
        return output;
    }

    private Bitmap modelOutputToBitmap(float[][][] data, ModelChoice choice) throws OrtException {
        if (data == null || data.length < 3 || data[0].length == 0 || data[0][0].length == 0) {
            throw new OrtException(choice.displayName + " returned an empty output tensor");
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
                        throw new OrtException(choice.displayName + " returned NaN/Infinity output");
                    }
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }
        }

        boolean looksByteScaled = max > 2.0f;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float rf = data[0][y][x];
                float gf = data[1][y][x];
                float bf = data[2][y][x];
                if (!looksByteScaled) {
                    if (choice == ModelChoice.SIMSWAP_512) {
                        rf *= 255.0f;
                        gf *= 255.0f;
                        bf *= 255.0f;
                    } else {
                        rf = (rf * 0.5f + 0.5f) * 255.0f;
                        gf = (gf * 0.5f + 0.5f) * 255.0f;
                        bf = (bf * 0.5f + 0.5f) * 255.0f;
                    }
                }
                int r = clampToByte(rf);
                int g = clampToByte(gf);
                int b = clampToByte(bf);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        Log.i(TAG, choice.displayName + " output range: min=" + min + ", max=" + max);
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

    private void closeSessions() {
        if (session != null) {
            try { session.close(); } catch (Exception e) { Log.w(TAG, "Error closing swapper session", e); }
            session = null;
        }
        if (embeddingConverterSession != null) {
            try { embeddingConverterSession.close(); } catch (Exception e) { Log.w(TAG, "Error closing converter session", e); }
            embeddingConverterSession = null;
        }
        activeModel = null;
        imageInputName = null;
        embeddingInputName = null;
        converterInputName = null;
    }

    public void close() {
        closeSessions();
    }
}
