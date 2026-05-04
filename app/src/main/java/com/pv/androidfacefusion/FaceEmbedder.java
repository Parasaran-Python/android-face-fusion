package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;

import java.util.EnumSet;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.providers.NNAPIFlags;


/**
 * Face embedder to extract face features using ONNX model
 */
public class FaceEmbedder {
    private static final String TAG = "FaceEmbedder";
    private static final int INPUT_SIZE = 112;
    
    private OrtEnvironment env;
    private OrtSession session;
    private Context context;

    public FaceEmbedder(Context context) {
        this.context = context;
        this.env = OrtEnvironment.getEnvironment();
    }
    
    public void initialize() throws Exception {
        try {
            Log.d(TAG, "Loading ArcFace embedding model...");
    
            // however you currently obtain the model file...
            ModelDownloader downloader = new ModelDownloader(context);
            File modelFile = downloader.getModelFile("w600k_r50.onnx");
    
            env = OrtEnvironment.getEnvironment();
    
            // NNAPI session options (same as FaceDetector)
            SessionOptions opts = createNnapiSessionOptions();
            session = env.createSession(modelFile.getAbsolutePath(), opts);
    
            Log.i(TAG, "ArcFace session created with NNAPI options");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize FaceEmbedder", e);
            throw e;
        }
    }

    public float[] getEmbedding(Bitmap faceBitmap) throws OrtException {
        if (session == null) {
            throw new IllegalStateException("Model not initialized");
        }

        // Preprocess image - align and resize
        Bitmap resizedFace = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true);
        float[] inputData = bitmapToFloatArray(resizedFace);

        // Create input tensor
        long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        // Get the actual input name from the model
        String inputName = session.getInputNames().iterator().next();
        
        // Run inference
        OrtSession.Result results = session.run(
            java.util.Collections.singletonMap(inputName, inputTensor)
        );

        // Extract embedding
        float[][] embedding = (float[][]) results.get(0).getValue();
        
        inputTensor.close();
        results.close();

        // Apply L2 normalization (CRITICAL for face recognition)
        float[] normalizedEmbedding = l2Normalize(embedding[0]);
        
        return normalizedEmbedding;
    }
    
    /**
     * L2 normalization of the embedding vector
     * This is CRITICAL for proper face similarity comparison
     */
    private float[] l2Normalize(float[] embedding) {
        float norm = 0.0f;
        
        // Calculate L2 norm
        for (float value : embedding) {
            norm += value * value;
        }
        norm = (float) Math.sqrt(norm);
        
        // Normalize to unit length
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
        
        // Convert to CHW format with normalization
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            output[i] = (r - 127.5f) / 127.5f;
            output[pixels.length + i] = (g - 127.5f) / 127.5f;
            output[2 * pixels.length + i] = (b - 127.5f) / 127.5f;
        }

        return output;
    }

    private static OrtSession.SessionOptions createNnapiSessionOptions() throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    
        // FP16 relaxation is a standard NNAPI perf option
        opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
    
        opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        return opts;
    }
    
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.e(TAG, "Error closing session", e);
            }
        }
    }
}
