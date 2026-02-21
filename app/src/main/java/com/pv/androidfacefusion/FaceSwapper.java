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

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * Face swapper using ONNX model for face swapping
 * Matches Python INSwapper implementation
 */
public class FaceSwapper {
    private static final String TAG = "FaceSwapper";
    private static final int INPUT_SIZE = 128;
    
    private OrtEnvironment env;
    private OrtSession session;
    private Context context;
    private float[][] emap;  // Embedding transformation matrix
    private String imageInputName;     // Resolved model input name for image
    private String embeddingInputName; // Resolved model input name for embedding

    public FaceSwapper(Context context) {
        this.context = context;
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize() throws Exception {
        try {
            // Download model if needed and load from file path directly
            Log.d(TAG, "Loading face swapping model...");
            ModelDownloader downloader = new ModelDownloader(context);
            File modelFile = downloader.getModelFile("inswapper_128.onnx");
            
            Log.d(TAG, "Model file ready, size: " + modelFile.length() + " bytes");
            
            // Load directly from file path to avoid OOM with large files
            // This is critical for the 555 MB inswapper model
            session = env.createSession(modelFile.getAbsolutePath());

            // Resolve input names by tensor shape to avoid Set ordering issues
            // Python: input_names[0] = image (4D), input_names[1] = embedding (2D)
            resolveInputNames();

            // Extract emap from model (last initializer)
            // This is critical for proper embedding transformation
            extractEmap(modelFile);
            
            Log.d(TAG, "Face swapping model initialized successfully with emap");
        } catch (Exception e) {
            Log.e(TAG, "Error loading face swapping model", e);
            throw new Exception("Failed to load face swapping model: " + e.getMessage());
        }
    }
    
    /**
     * Resolve input names by tensor shape to ensure correct ordering.
     * Java Set doesn't guarantee iteration order, so we match by shape:
     * - Image input: 4D tensor [1, 3, 128, 128]
     * - Embedding input: 2D tensor [1, 512]
     */
    private void resolveInputNames() throws Exception {
        java.util.Map<String, NodeInfo> inputInfo = session.getInputInfo();
        for (java.util.Map.Entry<String, NodeInfo> entry : inputInfo.entrySet()) {
            TensorInfo tensorInfo = (TensorInfo) entry.getValue().getInfo();
            long[] shape = tensorInfo.getShape();
            if (shape.length == 4) {
                imageInputName = entry.getKey();
            } else if (shape.length == 2) {
                embeddingInputName = entry.getKey();
            }
        }
        Log.d(TAG, "Resolved input names: image=" + imageInputName + ", embedding=" + embeddingInputName);

        if (imageInputName == null || embeddingInputName == null) {
            throw new Exception("Could not resolve model input names by shape");
        }
    }

    /**
     * Extract the emap transformation matrix from the ONNX model
     * This matches the Python code: self.emap = numpy_helper.to_array(graph.initializer[-1])
     * 
     * The EMAP is a learned 512x512 transformation matrix that is CRITICAL for proper face swapping.
     * It must be extracted from the ONNX model using the extract_emap.py script and placed in assets/
     */
    private void extractEmap(File modelFile) {
        try {
            // Try to load pre-extracted EMAP from assets
            Log.d(TAG, "Loading EMAP from assets...");
            emap = loadEmapFromAssets();
            
            if (emap != null) {
                Log.d(TAG, "✅ EMAP loaded successfully from assets: " + emap.length + "x" + emap[0].length);
                
                // Verify it's not an identity matrix
                boolean isIdentity = true;
                for (int i = 0; i < Math.min(10, emap.length) && isIdentity; i++) {
                    for (int j = 0; j < Math.min(10, emap[0].length) && isIdentity; j++) {
                        float expected = (i == j) ? 1.0f : 0.0f;
                        if (Math.abs(emap[i][j] - expected) > 0.01f) {
                            isIdentity = false;
                        }
                    }
                }
                
                if (isIdentity) {
                    Log.w(TAG, "⚠️  WARNING: EMAP appears to be identity matrix - this will reduce quality!");
                } else {
                    Log.d(TAG, "✅ EMAP is a proper learned transformation matrix");
                }
                return;
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load EMAP from assets", e);
        }
        
        // Fallback: Use identity matrix (WILL REDUCE QUALITY!)
        Log.w(TAG, "⚠️  CRITICAL: Using identity matrix for EMAP - quality will be reduced!");
        Log.w(TAG, "To fix: Run extract_emap.py and copy emap.bin to app/src/main/assets/");
        
        int emapSize = 512;
        emap = new float[emapSize][emapSize];
        
        // Initialize as identity matrix
        for (int i = 0; i < emapSize; i++) {
            for (int j = 0; j < emapSize; j++) {
                emap[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    /**
     * Load EMAP from assets/emap.bin
     * File format: 8 bytes header (2 ints: rows, cols) + float32 matrix data
     */
    private float[][] loadEmapFromAssets() throws IOException {
        try (InputStream fis = context.getAssets().open("emap.bin")) {
            // Read header (2 ints: rows, cols)
            byte[] header = new byte[8];
            if (fis.read(header) != 8) {
                throw new IOException("Failed to read EMAP header");
            }
            
            ByteBuffer headerBuffer = ByteBuffer.wrap(header);
            headerBuffer.order(ByteOrder.LITTLE_ENDIAN);
            int rows = headerBuffer.getInt();
            int cols = headerBuffer.getInt();
            
            Log.d(TAG, "EMAP dimensions from file: " + rows + "x" + cols);
            
            if (rows != 512 || cols != 512) {
                throw new IOException("Invalid EMAP dimensions: " + rows + "x" + cols + " (expected 512x512)");
            }
            
            // Read matrix data
            int dataSize = rows * cols * 4; // 4 bytes per float
            byte[] data = new byte[dataSize];
            int totalRead = 0;
            while (totalRead < dataSize) {
                int read = fis.read(data, totalRead, dataSize - totalRead);
                if (read == -1) {
                    throw new IOException("Unexpected end of file reading EMAP data");
                }
                totalRead += read;
            }
            
            // Convert to float array
            ByteBuffer buffer = ByteBuffer.wrap(data);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            
            float[][] matrix = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] = buffer.getFloat();
                }
            }
            
            return matrix;
        }
    }

    public Bitmap swapFace(Bitmap targetFace, float[] sourceEmbedding, Bitmap targetImage) throws OrtException {
        if (session == null) {
            throw new IllegalStateException("Model not initialized");
        }

        // Preprocess target face
        Bitmap resizedTarget = Bitmap.createScaledBitmap(targetFace, INPUT_SIZE, INPUT_SIZE, true);
        float[] targetData = bitmapToFloatArray(resizedTarget);

        // Transform source embedding with emap (matching Python code)
        // Python: latent = np.dot(latent, self.emap)
        //         latent /= np.linalg.norm(latent)
        float[] transformedEmbedding = applyEmapTransformation(sourceEmbedding);
        
        // Create input tensors
        long[] imageShape = {1, 3, INPUT_SIZE, INPUT_SIZE};
        long[] embeddingShape = {1, transformedEmbedding.length};
        
        OnnxTensor targetTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(targetData), imageShape);
        OnnxTensor embeddingTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(transformedEmbedding), embeddingShape);

        // Use resolved input names (determined by shape during initialization)
        Log.d(TAG, "Model input names: image=" + imageInputName + ", embedding=" + embeddingInputName);

        // Run inference
        java.util.Map<String, OnnxTensor> inputs = new java.util.HashMap<>();
        inputs.put(imageInputName, targetTensor);
        inputs.put(embeddingInputName, embeddingTensor);
        
        OrtSession.Result results = session.run(inputs);

        // Get swapped face
        float[][][][] outputData = (float[][][][]) results.get(0).getValue();
        Bitmap swappedFace = floatArrayToBitmap(outputData[0]);
        
        targetTensor.close();
        embeddingTensor.close();
        results.close();

        return swappedFace;
    }
    
    /**
     * Apply emap transformation to embedding
     * Matches Python: latent = np.dot(latent, self.emap); latent /= np.linalg.norm(latent)
     */
    private float[] applyEmapTransformation(float[] embedding) {
        if (emap == null || emap.length != embedding.length) {
            Log.w(TAG, "Emap not properly initialized, using original embedding");
            return embedding;
        }
        
        // Matrix multiplication: result = embedding * emap
        float[] result = new float[embedding.length];
        for (int i = 0; i < embedding.length; i++) {
            float sum = 0.0f;
            for (int j = 0; j < embedding.length; j++) {
                sum += embedding[j] * emap[j][i];
            }
            result[i] = sum;
        }
        
        // L2 normalize the result
        float norm = 0.0f;
        for (float value : result) {
            norm += value * value;
        }
        norm = (float) Math.sqrt(norm);
        
        if (norm > 0) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= norm;
            }
        }
        
        return result;
    }

    private float[] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] output = new float[3 * width * height];
        
        // CRITICAL FIX: INSwapper uses input_std = 255.0, NOT 127.5
        // Python code: self.input_std = 255.0 (line 20 in inswapper.py)
        // Normalization: (pixel - 0.0) / 255.0 = pixel / 255.0
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            output[i] = r / 255.0f;
            output[pixels.length + i] = g / 255.0f;
            output[2 * pixels.length + i] = b / 255.0f;
        }

        return output;
    }

    private Bitmap floatArrayToBitmap(float[][][] data) {
        int height = data[0].length;
        int width = data[0][0].length;
        
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];

        // CRITICAL FIX: Denormalize with 255.0, matching Python code
        // Python: bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) Math.max(0, Math.min(255, data[0][y][x] * 255.0f));
                int g = (int) Math.max(0, Math.min(255, data[1][y][x] * 255.0f));
                int b = (int) Math.max(0, Math.min(255, data[2][y][x] * 255.0f));
                
                pixels[y * width + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
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
