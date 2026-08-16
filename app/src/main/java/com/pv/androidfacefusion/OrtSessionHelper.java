package com.pv.androidfacefusion;

import android.util.Log;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Helper utility to create optimized ONNX Runtime sessions configured with
 * multi-threaded CPU execution and maximum graph optimizations.
 */
public class OrtSessionHelper {
    private static final String TAG = "OrtSessionHelper";

    /**
     * Creates an optimized ONNX session configured for multi-threaded CPU execution.
     *
     * @param env The OrtEnvironment instance
     * @param modelPath The absolute path to the ONNX model file
     * @param tag Logging tag for tracking session initialization
     * @return Initialized OrtSession
     * @throws OrtException if session creation fails
     */
    public static OrtSession createSession(OrtEnvironment env, String modelPath, String tag) throws OrtException {
        int optimalThreads = getOptimalThreadCount();

        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(optimalThreads);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            
            OrtSession session = env.createSession(modelPath, options);
            Log.i(tag, "Successfully initialized model on CPU with " + optimalThreads + " threads (ALL_OPT): " + modelPath);
            return session;
        }
    }

    /**
     * Determines optimal CPU thread count based on device cores.
     * Uses min(4, availableProcessors) to avoid thermal throttling, prevent efficiency core
     * bottlenecks on big.LITTLE architectures, and leave headroom for UI rendering.
     */
    public static int getOptimalThreadCount() {
        int cores = Runtime.getRuntime().availableProcessors();
        return Math.max(1, Math.min(4, cores));
    }
}
