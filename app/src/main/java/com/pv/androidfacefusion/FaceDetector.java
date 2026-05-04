package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.providers.NNAPIFlags;



/**
 * Face detector using ONNX model (similar to InsightFace detection)
 */
public class FaceDetector {
    private static final String TAG = "FaceDetector";
    private static final int INPUT_SIZE = 640;
    
    private OrtEnvironment env;
    private OrtSession session;
    private Context context;

    public static class Face {
        public RectF bbox;
        public float[] landmarks; // 5 landmarks (x, y) pairs = 10 values
        public float score;

        public Face(RectF bbox, float[] landmarks, float score) {
            this.bbox = bbox;
            this.landmarks = landmarks;
            this.score = score;
        }
    }

    public FaceDetector(Context context) {
        this.context = context;
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize() throws Exception {
        try {
            Log.d(TAG, "Loading face detection model...");
            ModelDownloader downloader = new ModelDownloader(context);
            File modelFile = downloader.getModelFile("det_10g.onnx");
    
            env = OrtEnvironment.getEnvironment();
    
            SessionOptions opts = createNnapiSessionOptions();
    
            Log.i(TAG, "Creating SCRFD session with NNAPI");
    
            session = env.createSession(modelFile.getAbsolutePath(), opts);
    
            Log.i(TAG, "SCRFD session created");
    
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize FaceDetector", e);
            throw e;
        }
    }

    private float[] imageScale = new float[2];  // Store scale factors for coordinate mapping
    private float[] imagePadding = new float[2]; // Store padding offsets
    
    public List<Face> detectFaces(Bitmap bitmap) throws OrtException {
        if (session == null) {
            throw new IllegalStateException("Model not initialized");
        }

        Log.d(TAG, "=== FACE DETECTION START ===");
        Log.d(TAG, "Input bitmap: " + bitmap.getWidth() + "x" + bitmap.getHeight() + ", config=" + bitmap.getConfig());

        // Preprocess image with aspect ratio preservation
        Bitmap resizedBitmap = resizeWithPadding(bitmap);
        float[] inputData = bitmapToFloatArray(resizedBitmap);

        // Create input tensor
        long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        // Get the actual input name from the model
        String inputName = session.getInputNames().iterator().next();
        
        // Run inference
        OrtSession.Result results = session.run(
            java.util.Collections.singletonMap(inputName, inputTensor)
        );

        // Parse results with default threshold (0.5)
        List<Face> faces = parseDetections(results, bitmap.getWidth(), bitmap.getHeight());
        
        // If no faces found, try again with lower threshold
        if (faces.isEmpty()) {
            Log.w(TAG, "No faces found with threshold 0.5, trying with lower threshold 0.3");
            faces = parseDetectionsWithThreshold(results, bitmap.getWidth(), bitmap.getHeight(), 0.3f);
        }
        
        inputTensor.close();
        results.close();

        Log.d(TAG, "Final detection result: " + faces.size() + " face(s) found");
        return faces;
    }
    
    private Bitmap resizeWithPadding(Bitmap source) {
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();

        // Calculate scale to fit image into INPUT_SIZE x INPUT_SIZE while maintaining aspect ratio
        // Matches Python: det_scale = float(new_height) / img.shape[0]
        float scale = Math.min((float) INPUT_SIZE / sourceWidth, (float) INPUT_SIZE / sourceHeight);

        int newWidth = Math.round(sourceWidth * scale);
        int newHeight = Math.round(sourceHeight * scale);

        // Store scale and padding for coordinate mapping
        // Python places image at top-left (0,0) with no padding offset
        imageScale[0] = scale;
        imageScale[1] = scale;
        imagePadding[0] = 0;
        imagePadding[1] = 0;

        Log.d(TAG, "Resize: source=" + sourceWidth + "x" + sourceHeight +
            ", scale=" + scale + ", newSize=" + newWidth + "x" + newHeight);

        // Create padded image with black background (matching Python: np.zeros)
        Bitmap padded = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(padded);
        canvas.drawColor(0xFF000000);

        // Place image at top-left (matching Python: det_img[:new_height, :new_width, :] = resized_img)
        Bitmap resized = Bitmap.createScaledBitmap(source, newWidth, newHeight, true);
        canvas.drawBitmap(resized, 0, 0, null);

        resized.recycle();

        return padded;
    }

    private float[] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] output = new float[3 * width * height];
        
        // Convert to CHW format with normalization matching Python RetinaFace
        // Python uses: blob = cv2.dnn.blobFromImage(img, 1.0/128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
        // This means: (pixel - 127.5) / 128.0
        float mean = 127.5f;
        float std = 128.0f;

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            // Note: swapRB=True in Python means BGR, but Android uses RGB, so we keep RGB order
            output[i] = (r - mean) / std;
            output[pixels.length + i] = (g - mean) / std;
            output[2 * pixels.length + i] = (b - mean) / std;
        }

        // DEBUG: Check value ranges and first pixel
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (float val : output) {
            min = Math.min(min, val);
            max = Math.max(max, val);
        }
        
        int firstPixel = pixels[0];
        int firstR = (firstPixel >> 16) & 0xFF;
        int firstG = (firstPixel >> 8) & 0xFF;
        int firstB = firstPixel & 0xFF;
        
        Log.d(TAG, "Preprocessing debug:");
        Log.d(TAG, "  First pixel RGB: R=" + firstR + ", G=" + firstG + ", B=" + firstB);
        Log.d(TAG, "  First normalized: R=" + output[0] + ", G=" + output[pixels.length] + ", B=" + output[2*pixels.length]);
        Log.d(TAG, "  Output range: min=" + String.format("%.3f", min) + ", max=" + String.format("%.3f", max) + " (expected: ~-1.0 to ~1.0)");

        return output;
    }

    private List<Face> parseDetections(OrtSession.Result results, int originalWidth, int originalHeight) {
        return parseDetectionsWithThreshold(results, originalWidth, originalHeight, 0.5f);
    }
    
    private List<Face> parseDetectionsWithThreshold(OrtSession.Result results, int originalWidth, int originalHeight, float threshold) {
        List<Face> faces = new ArrayList<>();
        
        try {
            // Log output information for debugging (only first time)
            if (threshold >= 0.5f) {
                Log.d(TAG, "Number of outputs: " + results.size());
                
                for (int i = 0; i < results.size(); i++) {
                    try {
                        Object value = results.get(i).getValue();
                        Log.d(TAG, "Output " + i + " type: " + value.getClass().getName());
                        
                        if (value instanceof float[][]) {
                            float[][] data = (float[][]) value;
                            Log.d(TAG, "Output " + i + " shape: [" + data.length + "][" + 
                                (data.length > 0 ? data[0].length : 0) + "]");
                        } else if (value instanceof float[][][]) {
                            float[][][] data = (float[][][]) value;
                            Log.d(TAG, "Output " + i + " shape: [" + data.length + "][" + 
                                (data.length > 0 ? data[0].length : 0) + "][" +
                                (data.length > 0 && data[0].length > 0 ? data[0][0].length : 0) + "]");
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error inspecting output " + i, e);
                    }
                }
            }
            
            // SCRFD model output formats:
            // 9 outputs: 3 FPN levels (stride 8,16,32), 2 anchors/pos, WITH keypoints
            //   0-2: Scores [N][1], 3-5: Boxes [N][4], 6-8: Landmarks [N][10]
            // 10 outputs: 5 FPN levels (stride 8,16,32,64,128), 1 anchor/pos, NO keypoints
            //   0-4: Scores [N][1], 5-9: Boxes [N][4]
            // 6 outputs: 3 FPN levels, 2 anchors/pos, NO keypoints
            //   0-2: Scores [N][1], 3-5: Boxes [N][4]

            if (results.size() == 9) {
                Log.d(TAG, "Parsing SCRFD format (9 outputs, 3 scales with kps) threshold=" + threshold);
                faces = parseSCRFDOutputsWithThreshold(results, originalWidth, originalHeight, threshold);
            } else if (results.size() == 10) {
                Log.d(TAG, "Parsing SCRFD format (10 outputs, 5 scales no kps) threshold=" + threshold);
                faces = parseSCRFD10Outputs(results, originalWidth, originalHeight, threshold);
            } else if (results.size() == 6) {
                Log.d(TAG, "Parsing SCRFD format (6 outputs, 3 scales no kps) threshold=" + threshold);
                faces = parseSCRFD6Outputs(results, originalWidth, originalHeight, threshold);
            } else {
                Log.w(TAG, "Unexpected number of outputs: " + results.size());
            }
            
            Log.d(TAG, "Total faces detected: " + faces.size());
            
        } catch (Exception e) {
            Log.e(TAG, "Error parsing detections", e);
            e.printStackTrace();
        }

        return faces;
    }
    
    private List<Face> parseSCRFDOutputsWithThreshold(OrtSession.Result results, int originalWidth, int originalHeight, float threshold) {
        List<Face> allFaces = new ArrayList<>();

        try {
            // Process 3 scales (stride 8, 16, 32)
            int[] strides = {8, 16, 32};

            // Track statistics for debugging
            int totalCandidates = 0;
            float maxScore = 0.0f;

            for (int scaleIdx = 0; scaleIdx < 3; scaleIdx++) {
                float[][] scores = (float[][]) results.get(scaleIdx).getValue();
                float[][] boxes = (float[][]) results.get(scaleIdx + 3).getValue();
                float[][] landmarks = (float[][]) results.get(scaleIdx + 6).getValue();

                int stride = strides[scaleIdx];
                int totalAnchors = scores.length;

                int featH = INPUT_SIZE / stride;
                int featW = INPUT_SIZE / stride;
                int gridPositions = featH * featW;

                // Determine num_anchors per position from output shape
                // Python: self._num_anchors = 2 for 9-output SCRFD models
                int numAnchorsPerPos = totalAnchors / gridPositions;
                if (numAnchorsPerPos < 1) numAnchorsPerPos = 1;

                Log.d(TAG, "Processing scale " + scaleIdx + ", stride=" + stride +
                    ", totalAnchors=" + totalAnchors + ", grid=" + featW + "x" + featH +
                    ", anchorsPerPos=" + numAnchorsPerPos);

                // Find max score at this scale for debugging
                for (int i = 0; i < totalAnchors; i++) {
                    maxScore = Math.max(maxScore, scores[i][0]);
                    if (scores[i][0] > 0.1f) {  // Count candidates above 0.1
                        totalCandidates++;
                    }
                }

                int faceCount = 0;

                for (int i = 0; i < totalAnchors; i++) {
                    float score = scores[i][0];

                    // Match Python: pos_inds = np.where(scores>=threshold)[0]
                    if (score >= threshold) {
                        // Calculate anchor center accounting for num_anchors per position
                        // Python: anchor_centers duplicated for each anchor at same position
                        int gridIndex = i / numAnchorsPerPos;
                        int anchorY = gridIndex / featW;
                        int anchorX = gridIndex % featW;
                        // Python: anchor_centers = (anchor_centers * stride) — NO +0.5 offset
                        float anchorCenterX = anchorX * stride;
                        float anchorCenterY = anchorY * stride;
                        
                        // SCRFD uses distance format: [left, top, right, bottom] distances from anchor
                        float leftDist = boxes[i][0] * stride;
                        float topDist = boxes[i][1] * stride;
                        float rightDist = boxes[i][2] * stride;
                        float bottomDist = boxes[i][3] * stride;
                        
                        // Calculate bounding box in INPUT_SIZE (640x640) coordinates first
                        float x1_model = anchorCenterX - leftDist;
                        float y1_model = anchorCenterY - topDist;
                        float x2_model = anchorCenterX + rightDist;
                        float y2_model = anchorCenterY + bottomDist;
                        
                        // Log before scaling for debugging
                        if (faceCount < 3) {  // Log first few detections only
                            Log.d(TAG, "Before scaling - anchor: (" + anchorCenterX + "," + anchorCenterY + 
                                "), box in 640x640: [" + x1_model + "," + y1_model + "," + x2_model + "," + y2_model + "]" +
                                ", distances: [" + leftDist + "," + topDist + "," + rightDist + "," + bottomDist + "]" +
                                ", raw box values: [" + boxes[i][0] + "," + boxes[i][1] + "," + boxes[i][2] + "," + boxes[i][3] + "]");
                        }
                        
                        // Remove padding first
                        x1_model = x1_model - imagePadding[0];
                        y1_model = y1_model - imagePadding[1];
                        x2_model = x2_model - imagePadding[0];
                        y2_model = y2_model - imagePadding[1];
                        
                        // Scale back to original image coordinates
                        float x1 = x1_model / imageScale[0];
                        float y1 = y1_model / imageScale[1];
                        float x2 = x2_model / imageScale[0];
                        float y2 = y2_model / imageScale[1];
                        
                        // Log before clamping
                        if (faceCount < 3) {
                            Log.d(TAG, "After removing padding and scaling - box in original: [" + x1 + "," + y1 + "," + x2 + "," + y2 + "]" +
                                ", scale=" + imageScale[0] + ", padding=" + imagePadding[0] + "," + imagePadding[1] + 
                                ", originalSize=" + originalWidth + "x" + originalHeight);
                        }
                        
                        // Clamp to image bounds
                        x1 = Math.max(0, Math.min(x1, originalWidth));
                        y1 = Math.max(0, Math.min(y1, originalHeight));
                        x2 = Math.max(0, Math.min(x2, originalWidth));
                        y2 = Math.max(0, Math.min(y2, originalHeight));
                        
                        // Validate bounding box
                        float boxWidth = x2 - x1;
                        float boxHeight = y2 - y1;
                        
                        // Filter out invalid or too small boxes
                        if (boxWidth < 20 || boxHeight < 20) {
                            continue; // Box too small
                        }
                        
                        // Check aspect ratio (faces should be roughly square-ish, 0.5 to 2.0 ratio)
                        float aspectRatio = boxWidth / boxHeight;
                        if (aspectRatio < 0.3f || aspectRatio > 3.0f) {
                            Log.d(TAG, "Rejecting invalid aspect ratio: bbox=" + new RectF(x1, y1, x2, y2) + 
                                ", score=" + score + ", aspectRatio=" + aspectRatio);
                            continue;
                        }
                        
                        // Filter out boxes that touch multiple edges (corner detections)
                        boolean leftEdge = (x1 == 0);
                        boolean rightEdge = (x2 == originalWidth);
                        boolean topEdge = (y1 == 0);
                        boolean bottomEdge = (y2 == originalHeight);
                        
                        int touchedEdges = (leftEdge ? 1 : 0) + (rightEdge ? 1 : 0) + 
                                          (topEdge ? 1 : 0) + (bottomEdge ? 1 : 0);
                        
                        // Reject corner detections (touching 2+ edges) unless box is reasonably sized
                        if (touchedEdges >= 2) {
                            // Allow if the box takes up a substantial part of the image (>10% area)
                            float boxArea = boxWidth * boxHeight;
                            float imageArea = originalWidth * originalHeight;
                            float areaRatio = boxArea / imageArea;
                            
                            if (areaRatio < 0.1f) {
                                Log.d(TAG, "Rejecting edge detection: bbox=" + new RectF(x1, y1, x2, y2) + 
                                    ", score=" + score + ", touchedEdges=" + touchedEdges + ", areaRatio=" + areaRatio);
                                continue;
                            }
                        }
                        
                        RectF bbox = new RectF(x1, y1, x2, y2);
                        
                        // Process landmarks (also distance format from anchor)
                        float[] facelandmarks = new float[10];
                        for (int j = 0; j < 5; j++) {
                            float lmX_model = (anchorCenterX + landmarks[i][j * 2] * stride) - imagePadding[0];
                            float lmY_model = (anchorCenterY + landmarks[i][j * 2 + 1] * stride) - imagePadding[1];
                            
                            float lmX = lmX_model / imageScale[0];
                            float lmY = lmY_model / imageScale[1];
                            
                            facelandmarks[j * 2] = Math.max(0, Math.min(lmX, originalWidth));
                            facelandmarks[j * 2 + 1] = Math.max(0, Math.min(lmY, originalHeight));
                        }
                        
                        Log.d(TAG, "Detected face at scale " + scaleIdx + ": bbox=" + bbox + ", score=" + score);
                        
                        allFaces.add(new Face(bbox, facelandmarks, score));
                        faceCount++;
                    }
                }
                
                Log.d(TAG, "Scale " + scaleIdx + " found " + faceCount + " faces");
            }
            
            // Log detection statistics
            Log.d(TAG, "Detection stats: max_score=" + maxScore + ", candidates_above_0.1=" + totalCandidates + 
                      ", threshold=" + threshold + ", total_detections=" + allFaces.size());
            
            // If no faces found, log helpful debug info
            if (allFaces.isEmpty()) {
                Log.w(TAG, "NO FACES DETECTED! Check:");
                Log.w(TAG, "  1. Max score found: " + maxScore + " (should be > " + threshold + ")");
                Log.w(TAG, "  2. Candidates > 0.1: " + totalCandidates);
                Log.w(TAG, "  3. Image quality: Is face clear and well-lit?");
                Log.w(TAG, "  4. Face size: Is face large enough in image?");
                Log.w(TAG, "  5. Image orientation: Is image rotated?");
            }
            
            // Apply NMS to remove duplicate detections
            if (!allFaces.isEmpty()) {
                allFaces = applyNMS(allFaces, 0.4f);
                Log.d(TAG, "After NMS: " + allFaces.size() + " faces");
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Error parsing SCRFD outputs", e);
            e.printStackTrace();
        }
        
        return allFaces;
    }
    
    /**
     * Parse 10-output SCRFD model (5 FPN levels, 1 anchor/pos, NO keypoints).
     * Strides: [8, 16, 32, 64, 128], fmc=5, num_anchors=1
     * Outputs 0-4: Scores, Outputs 5-9: Boxes
     * Keypoints are derived from bounding boxes as approximate positions.
     */
    private List<Face> parseSCRFD10Outputs(OrtSession.Result results, int originalWidth, int originalHeight, float threshold) {
        return parseSCRFDNoKps(results, originalWidth, originalHeight, threshold,
                new int[]{8, 16, 32, 64, 128}, 5);
    }

    /**
     * Parse 6-output SCRFD model (3 FPN levels, 2 anchors/pos, NO keypoints).
     * Strides: [8, 16, 32], fmc=3, num_anchors=2
     * Outputs 0-2: Scores, Outputs 3-5: Boxes
     */
    private List<Face> parseSCRFD6Outputs(OrtSession.Result results, int originalWidth, int originalHeight, float threshold) {
        return parseSCRFDNoKps(results, originalWidth, originalHeight, threshold,
                new int[]{8, 16, 32}, 3);
    }

    /**
     * Generic parser for SCRFD models WITHOUT keypoint outputs.
     * Derives approximate 5-point landmarks from bounding box.
     */
    private List<Face> parseSCRFDNoKps(OrtSession.Result results, int originalWidth, int originalHeight,
                                        float threshold, int[] strides, int fmc) {
        List<Face> allFaces = new ArrayList<>();
        try {
            for (int scaleIdx = 0; scaleIdx < strides.length; scaleIdx++) {
                float[][] scores = (float[][]) results.get(scaleIdx).getValue();
                float[][] boxes = (float[][]) results.get(scaleIdx + fmc).getValue();
                int stride = strides[scaleIdx];
                int totalAnchors = scores.length;
                int featH = INPUT_SIZE / stride;
                int featW = INPUT_SIZE / stride;
                int gridPositions = featH * featW;
                int numAnchorsPerPos = Math.max(1, totalAnchors / gridPositions);

                for (int i = 0; i < totalAnchors; i++) {
                    if (scores[i][0] < threshold) continue;

                    int gridIndex = i / numAnchorsPerPos;
                    int anchorY = gridIndex / featW;
                    int anchorX = gridIndex % featW;
                    float anchorCenterX = anchorX * stride;
                    float anchorCenterY = anchorY * stride;

                    float x1_model = anchorCenterX - boxes[i][0] * stride;
                    float y1_model = anchorCenterY - boxes[i][1] * stride;
                    float x2_model = anchorCenterX + boxes[i][2] * stride;
                    float y2_model = anchorCenterY + boxes[i][3] * stride;

                    float x1 = (x1_model - imagePadding[0]) / imageScale[0];
                    float y1 = (y1_model - imagePadding[1]) / imageScale[1];
                    float x2 = (x2_model - imagePadding[0]) / imageScale[0];
                    float y2 = (y2_model - imagePadding[1]) / imageScale[1];

                    x1 = Math.max(0, Math.min(x1, originalWidth));
                    y1 = Math.max(0, Math.min(y1, originalHeight));
                    x2 = Math.max(0, Math.min(x2, originalWidth));
                    y2 = Math.max(0, Math.min(y2, originalHeight));

                    float boxWidth = x2 - x1;
                    float boxHeight = y2 - y1;
                    if (boxWidth < 20 || boxHeight < 20) continue;
                    float aspectRatio = boxWidth / boxHeight;
                    if (aspectRatio < 0.3f || aspectRatio > 3.0f) continue;

                    // Derive approximate 5-point landmarks from bounding box
                    float[] landmarks = deriveLandmarksFromBbox(x1, y1, x2, y2);

                    allFaces.add(new Face(new RectF(x1, y1, x2, y2), landmarks, scores[i][0]));
                }
            }

            if (!allFaces.isEmpty()) {
                allFaces = applyNMS(allFaces, 0.4f);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error parsing SCRFD outputs (no kps)", e);
        }
        return allFaces;
    }

    /**
     * Derive approximate 5 facial landmarks from bounding box.
     * Uses standard facial proportions for frontal faces.
     * Order: left_eye, right_eye, nose, left_mouth, right_mouth
     */
    private float[] deriveLandmarksFromBbox(float x1, float y1, float x2, float y2) {
        float w = x2 - x1;
        float h = y2 - y1;
        return new float[]{
            x1 + 0.30f * w, y1 + 0.33f * h,  // left eye
            x1 + 0.70f * w, y1 + 0.33f * h,  // right eye
            x1 + 0.50f * w, y1 + 0.55f * h,  // nose
            x1 + 0.35f * w, y1 + 0.72f * h,  // left mouth corner
            x1 + 0.65f * w, y1 + 0.72f * h   // right mouth corner
        };
    }

    private List<Face> applyNMS(List<Face> faces, float iouThreshold) {
        // Sort by score descending
        faces.sort((a, b) -> Float.compare(b.score, a.score));
        
        List<Face> result = new ArrayList<>();
        boolean[] suppressed = new boolean[faces.size()];
        
        for (int i = 0; i < faces.size(); i++) {
            if (suppressed[i]) continue;
            
            result.add(faces.get(i));
            RectF boxA = faces.get(i).bbox;
            
            for (int j = i + 1; j < faces.size(); j++) {
                if (suppressed[j]) continue;
                
                RectF boxB = faces.get(j).bbox;
                float iou = calculateIoU(boxA, boxB);
                
                if (iou > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        
        return result;
    }
    
    private float calculateIoU(RectF boxA, RectF boxB) {
        float intersectLeft = Math.max(boxA.left, boxB.left);
        float intersectTop = Math.max(boxA.top, boxB.top);
        float intersectRight = Math.min(boxA.right, boxB.right);
        float intersectBottom = Math.min(boxA.bottom, boxB.bottom);
        
        float intersectWidth = Math.max(0, intersectRight - intersectLeft);
        float intersectHeight = Math.max(0, intersectBottom - intersectTop);
        float intersectArea = intersectWidth * intersectHeight;
        
        float boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);
        float unionArea = boxAArea + boxBArea - intersectArea;
        
        return intersectArea / unionArea;
    }

    private static OrtSession.SessionOptions createNnapiSessionOptions() throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    
        // NNAPI flags on Android are an EnumSet, not bit flags
        opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
    
        opts.setOptimizationLevel(
                OrtSession.SessionOptions.OptLevel.ALL_OPT
        );
    
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
