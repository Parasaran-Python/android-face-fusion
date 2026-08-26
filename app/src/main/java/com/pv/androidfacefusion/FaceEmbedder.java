package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Face embedder to extract face features using the FaceFusion ArcFace model.
 */
public class FaceEmbedder {
    private static final String TAG = "FaceEmbedder";
    private static final int INPUT_SIZE = 112;

    private final OrtEnvironment env;
    private final Context context;
    private OrtSession session;

    public FaceEmbedder(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize() throws Exception {
        try {
            Log.d(TAG, "Loading FaceFusion ArcFace embedding model...");
            ModelDownloader downloader = new ModelDownloader(context);
            File modelFile = downloader.getModelFile(ModelDownloader.REC_MODEL);

            Log.d(TAG, "Model file ready, size: " + modelFile.length() + " bytes");
            session = OrtSessionHelper.createSession(env, modelFile.getAbsolutePath(), TAG);
            Log.d(TAG, "FaceFusion ArcFace embedding model initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error loading face embedding model", e);
            throw new Exception("Failed to load face embedding model: " + e.getMessage(), e);
        }
    }

    public float[] getEmbedding(Bitmap faceBitmap) throws OrtException {
        if (session == null) {
            throw new IllegalStateException("Model not initialized");
        }

        Bitmap resizedFace = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true);
        try {
            float[] inputData = bitmapToFloatArray(resizedFace);
            long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};

            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
                String inputName = session.getInputNames().iterator().next();
                try (OrtSession.Result results = session.run(
                    java.util.Collections.singletonMap(inputName, inputTensor))) {
                    float[][] embedding = (float[][]) results.get(0).getValue();
                    return l2Normalize(embedding[0]);
                }
            }
        } finally {
            if (resizedFace != faceBitmap && !resizedFace.isRecycled()) {
                resizedFace.recycle();
            }
        }
    }

    private float[] l2Normalize(float[] embedding) {
        float norm = 0.0f;
        for (float value : embedding) {
            norm += value * value;
        }
        norm = (float) Math.sqrt(norm);

        if (norm > 0) {
            float[] normalized = new float[embedding.length];
            for (int i = 0; i < embedding.length; i++) {
                normalized[i] = embedding[i] / norm;
            }
            return normalized;
        }

        return embedding;
    }

    private float[] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] output = new float[3 * width * height];

        // FaceFusion ArcFace pipeline: RGB / 127.5 - 1, then CHW.
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            output[i] = r / 127.5f - 1.0f;
            output[pixels.length + i] = g / 127.5f - 1.0f;
            output[2 * pixels.length + i] = b / 127.5f - 1.0f;
        }

        return output;
    }

    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.e(TAG, "Error closing session", e);
            }
            session = null;
        }
    }
}
