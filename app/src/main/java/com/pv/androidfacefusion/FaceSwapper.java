package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * Face swapper with HyperSwap 1b 256 as the primary model and the legacy
 * INSwapper 128 pipeline retained as an automatic compatibility fallback.
 */
public class FaceSwapper {
    private static final String TAG = "FaceSwapper";
    private static final int HYPERSWAP_SIZE = 256;
    private static final int INSWAPPER_SIZE = 128;

    private enum Backend {
        HYPERSWAP,
        INSWAPPER
    }

    private final Context context;
    private final OrtEnvironment env;
    private OrtSession session;
    private Backend backend;
    private int inputSize = HYPERSWAP_SIZE;
    private String imageInputName;
    private String embeddingInputName;
    private float[][] emap;

    public FaceSwapper(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize() throws Exception {
        ModelDownloader downloader = new ModelDownloader(context);

        try {
            Log.i(TAG, "Loading HyperSwap 1b 256...");
            File model = downloader.getModelFile(ModelDownloader.HYPERSWAP_MODEL);
            session = OrtSessionHelper.createSession(env, model.getAbsolutePath(), TAG);
            backend = Backend.HYPERSWAP;
            inputSize = HYPERSWAP_SIZE;
            resolveInputNames();
            validateHyperSwapInputs();
            Log.i(TAG, "HyperSwap 1b 256 initialized successfully");
            return;
        } catch (Exception hyperSwapError) {
            Log.e(TAG, "HyperSwap initialization failed; falling back to INSwapper 128", hyperSwapError);
            closeSessionOnly();
        }

        try {
            File model = downloader.getModelFile(ModelDownloader.INSWAPPER_MODEL);
            session = OrtSessionHelper.createSession(env, model.getAbsolutePath(), TAG);
            backend = Backend.INSWAPPER;
            inputSize = INSWAPPER_SIZE;
            resolveInputNames();
            emap = loadEmapFromAssets();
            Log.w(TAG, "Using legacy INSwapper 128 fallback");
        } catch (Exception fallbackError) {
            closeSessionOnly();
            throw new Exception("HyperSwap and INSwapper fallback both failed: " + fallbackError.getMessage(), fallbackError);
        }
    }

    public int getInputSize() {
        return inputSize;
    }

    public boolean isUsingHyperSwap() {
        return backend == Backend.HYPERSWAP;
    }

    private void resolveInputNames() throws Exception {
        imageInputName = null;
        embeddingInputName = null;

        for (Map.Entry<String, NodeInfo> entry : session.getInputInfo().entrySet()) {
            TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
            long[] shape = tensorInfo.getShape();
            if (shape.length == 4) {
                imageInputName = entry.getKey();
            } else if (shape.length == 2) {
                embeddingInputName = entry.getKey();
            }
        }

        if (imageInputName == null || embeddingInputName == null) {
            throw new Exception("Could not resolve swapper image/embedding inputs from ONNX tensor shapes");
        }
        Log.d(TAG, "Resolved inputs: embedding=" + embeddingInputName + ", image=" + imageInputName);
    }

    private void validateHyperSwapInputs() throws Exception {
        NodeInfo imageNode = session.getInputInfo().get(imageInputName);
        NodeInfo embeddingNode = session.getInputInfo().get(embeddingInputName);
        if (imageNode == null || embeddingNode == null) {
            throw new Exception("HyperSwap inputs not found");
        }

        long[] imageShape = ((TensorInfo) imageNode.getInfo()).getShape();
        long[] embeddingShape = ((TensorInfo) embeddingNode.getInfo()).getShape();
        if (imageShape.length != 4 || imageShape[1] != 3) {
            throw new Exception("Unexpected HyperSwap image input shape");
        }
        if (imageShape[2] > 0 && imageShape[2] != HYPERSWAP_SIZE) {
            throw new Exception("Expected HyperSwap height 256 but model reports " + imageShape[2]);
        }
        if (imageShape[3] > 0 && imageShape[3] != HYPERSWAP_SIZE) {
            throw new Exception("Expected HyperSwap width 256 but model reports " + imageShape[3]);
        }
        if (embeddingShape.length != 2 || (embeddingShape[1] > 0 && embeddingShape[1] != 512)) {
            throw new Exception("Unexpected HyperSwap embedding input shape");
        }
    }

    public Bitmap swapFace(Bitmap targetFace, float[] sourceEmbedding, Bitmap targetImage) throws OrtException {
        if (session == null || backend == null) {
            throw new IllegalStateException("Face swapper not initialized");
        }
        if (sourceEmbedding == null || sourceEmbedding.length != 512) {
            throw new IllegalArgumentException("Expected a 512-dimensional source embedding");
        }

        Bitmap resizedTarget = targetFace;
        if (targetFace.getWidth() != inputSize || targetFace.getHeight() != inputSize) {
            resizedTarget = Bitmap.createScaledBitmap(targetFace, inputSize, inputSize, true);
        }

        float[] imageData = backend == Backend.HYPERSWAP
            ? bitmapToHyperSwapArray(resizedTarget)
            : bitmapToInSwapperArray(resizedTarget);

        float[] identity = backend == Backend.HYPERSWAP
            ? l2Normalize(sourceEmbedding)
            : applyEmapTransformation(sourceEmbedding);

        long[] imageShape = {1, 3, inputSize, inputSize};
        long[] embeddingShape = {1, identity.length};

        try (OnnxTensor imageTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), imageShape);
             OnnxTensor embeddingTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(identity), embeddingShape)) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(imageInputName, imageTensor);
            inputs.put(embeddingInputName, embeddingTensor);

            try (OrtSession.Result results = session.run(inputs)) {
                float[][][][] output = (float[][][][]) results.get(0).getValue();
                return backend == Backend.HYPERSWAP
                    ? hyperSwapOutputToBitmap(output[0])
                    : inSwapperOutputToBitmap(output[0]);
            }
        } finally {
            if (resizedTarget != targetFace && !resizedTarget.isRecycled()) {
                resizedTarget.recycle();
            }
        }
    }

    private float[] bitmapToHyperSwapArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] output = new float[3 * pixels.length];

        // FaceFusion HyperSwap: RGB in [0,1], then mean=.5/std=.5 => [-1,1].
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            output[i] = (r - 0.5f) / 0.5f;
            output[pixels.length + i] = (g - 0.5f) / 0.5f;
            output[2 * pixels.length + i] = (b - 0.5f) / 0.5f;
        }
        return output;
    }

    private float[] bitmapToInSwapperArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] output = new float[3 * pixels.length];

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            output[i] = ((pixel >> 16) & 0xFF) / 255.0f;
            output[pixels.length + i] = ((pixel >> 8) & 0xFF) / 255.0f;
            output[2 * pixels.length + i] = (pixel & 0xFF) / 255.0f;
        }
        return output;
    }

    private Bitmap hyperSwapOutputToBitmap(float[][][] data) {
        int height = data[0].length;
        int width = data[0][0].length;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        // HyperSwap output is tanh-like; reverse mean/std normalization.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = clampToByte((data[0][y][x] * 0.5f + 0.5f) * 255.0f);
                int g = clampToByte((data[1][y][x] * 0.5f + 0.5f) * 255.0f);
                int b = clampToByte((data[2][y][x] * 0.5f + 0.5f) * 255.0f);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private Bitmap inSwapperOutputToBitmap(float[][][] data) {
        int height = data[0].length;
        int width = data[0][0].length;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = clampToByte(data[0][y][x] * 255.0f);
                int g = clampToByte(data[1][y][x] * 255.0f);
                int b = clampToByte(data[2][y][x] * 255.0f);
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private float[] l2Normalize(float[] embedding) {
        float norm = 0.0f;
        for (float value : embedding) norm += value * value;
        norm = (float) Math.sqrt(norm);
        float[] normalized = new float[embedding.length];
        if (norm <= 0.0f) {
            System.arraycopy(embedding, 0, normalized, 0, embedding.length);
            return normalized;
        }
        for (int i = 0; i < embedding.length; i++) normalized[i] = embedding[i] / norm;
        return normalized;
    }

    private float[] applyEmapTransformation(float[] embedding) {
        if (emap == null || emap.length != embedding.length) {
            return l2Normalize(embedding);
        }
        float[] result = new float[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            float sum = 0.0f;
            for (int j = 0; j < embedding.length; j++) {
                sum += embedding[j] * emap[j][i];
            }
            result[i] = sum;
        }
        return l2Normalize(result);
    }

    private float[][] loadEmapFromAssets() throws IOException {
        try (InputStream input = context.getAssets().open("emap.bin")) {
            byte[] header = new byte[8];
            if (input.read(header) != 8) throw new IOException("Failed to read EMAP header");
            ByteBuffer headerBuffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            int rows = headerBuffer.getInt();
            int cols = headerBuffer.getInt();
            if (rows != 512 || cols != 512) {
                throw new IOException("Invalid EMAP dimensions: " + rows + "x" + cols);
            }

            byte[] bytes = new byte[rows * cols * 4];
            int total = 0;
            while (total < bytes.length) {
                int read = input.read(bytes, total, bytes.length - total);
                if (read < 0) throw new IOException("Unexpected EOF reading EMAP");
                total += read;
            }

            ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            float[][] matrix = new float[rows][cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) matrix[r][c] = buffer.getFloat();
            }
            return matrix;
        }
    }

    private int clampToByte(float value) {
        return Math.max(0, Math.min(255, Math.round(value)));
    }

    private void closeSessionOnly() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.w(TAG, "Error closing ONNX session", e);
            }
            session = null;
        }
    }

    public void close() {
        closeSessionOnly();
    }
}
