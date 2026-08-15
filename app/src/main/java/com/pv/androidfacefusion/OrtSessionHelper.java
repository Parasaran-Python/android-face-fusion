package com.pv.androidfacefusion;

import android.util.Log;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Helper utility to create optimized ONNX Runtime sessions with NNAPI (NPU/GPU/DSP)
 * hardware acceleration and automatic fallback to multi-threaded CPU execution.
 */
public class OrtSessionHelper {
    private static final String TAG = "OrtSessionHelper";

    /**
     * Creates an ONNX session with hardware acceleration (NNAPI) if available,
     * falling back seamlessly to multi-threaded CPU if NNAPI initialization fails.
     *
     * @param env The OrtEnvironment instance
     * @param modelPath The absolute path to the ONNX model file
     * @param tag Logging tag for tracking session initialization
     * @return Initialized OrtSession
     * @throws OrtException if session creation fails entirely
     */
    public static OrtSession createSession(OrtEnvironment env, String modelPath, String tag) throws OrtException {
        return createSession(env, modelPath, tag, true);
    }

    /**
     * Creates an ONNX session with optional hardware acceleration (NNAPI) and CPU fallback.
     *
     * @param env The OrtEnvironment instance
     * @param modelPath The absolute path to the ONNX model file
     * @param tag Logging tag for tracking session initialization
     * @param tryNnapi Whether to attempt NNAPI hardware acceleration
     * @return Initialized OrtSession
     * @throws OrtException if session creation fails entirely
     */
    public static OrtSession createSession(OrtEnvironment env, String modelPath, String tag, boolean tryNnapi) throws OrtException {
        int optimalThreads = getOptimalThreadCount();

        if (tryNnapi) {
            try {
                OrtSession.SessionOptions nnapiOptions = new OrtSession.SessionOptions();
                nnapiOptions.setIntraOpNumThreads(optimalThreads);
                nnapiOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                nnapiOptions.addNnapi();
                
                OrtSession session = env.createSession(modelPath, nnapiOptions);
                Log.i(tag, "Successfully initialized model with NNAPI hardware acceleration (NPU/GPU): " + modelPath);
                return session;
            } catch (Exception e) {
                Log.w(tag, "NNAPI acceleration unavailable or failed for " + modelPath + ", falling back to CPU: " + e.getMessage());
            }
        }

        // Fallback or explicit CPU configuration
        OrtSession.SessionOptions cpuOptions = new OrtSession.SessionOptions();
        cpuOptions.setIntraOpNumThreads(optimalThreads);
        cpuOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        
        OrtSession session = env.createSession(modelPath, cpuOptions);
        Log.i(tag, "Successfully initialized model on CPU with " + optimalThreads + " threads: " + modelPath);
        return session;
    }

    /**
     * Determines optimal CPU thread count based on device cores.
     * Uses min(4, availableProcessors) to avoid thermal throttling and leave cores for UI.
     */
    public static int getOptimalThreadCount() {
        int cores = Runtime.getRuntime().availableProcessors();
        return Math.max(1, Math.min(4, cores));
    }
}
