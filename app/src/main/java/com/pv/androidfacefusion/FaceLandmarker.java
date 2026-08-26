package com.pv.androidfacefusion;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/** FaceFusion-compatible 2DFAN4 landmark refinement for more stable pose alignment. */
public final class FaceLandmarker {
    private static final String TAG = "FaceLandmarker";
    private static final int INPUT_SIZE = 256;
    private static final float MIN_LANDMARK_SCORE = 0.5f;

    public static final class Result {
        public final float[] landmarks68;
        public final float[] landmarks5;
        public final float score;

        Result(float[] landmarks68, float[] landmarks5, float score) {
            this.landmarks68 = landmarks68;
            this.landmarks5 = landmarks5;
            this.score = score;
        }
    }

    private final Context context;
    private final OrtEnvironment env;
    private OrtSession session;
    private String inputName;

    public FaceLandmarker(Context context) {
        this.context = context.getApplicationContext();
        this.env = OrtEnvironment.getEnvironment();
    }

    public void initialize() throws Exception {
        ModelDownloader downloader = new ModelDownloader(context);
        File model = downloader.getModelFile(ModelDownloader.LANDMARKER_MODEL);
        session = OrtSessionHelper.createSession(env, model.getAbsolutePath(), TAG);
        inputName = session.getInputNames().iterator().next();
        Log.i(TAG, "2DFAN4 landmark refinement initialized");
    }

    public Result refine(Bitmap image, FaceDetector.Face face) {
        if (session == null || image == null || face == null || face.bbox == null) return null;

        float faceWidth = Math.max(1.0f, face.bbox.width());
        float faceHeight = Math.max(1.0f, face.bbox.height());
        float maxDimension = Math.max(faceWidth, faceHeight);

        // FaceFusion 2DFAN uses scale=195/max(face width,height), then centers the bbox in 256.
        float cropSize = maxDimension * INPUT_SIZE / 195.0f;
        float centerX = face.bbox.centerX();
        float centerY = face.bbox.centerY();
        float cropLeft = centerX - cropSize * 0.5f;
        float cropTop = centerY - cropSize * 0.5f;

        Bitmap crop = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(crop);
        Matrix imageToCrop = new Matrix();
        imageToCrop.setRectToRect(
            new RectF(cropLeft, cropTop, cropLeft + cropSize, cropTop + cropSize),
            new RectF(0, 0, INPUT_SIZE, INPUT_SIZE),
            Matrix.ScaleToFit.FILL);
        canvas.drawBitmap(image, imageToCrop, new Paint(Paint.FILTER_BITMAP_FLAG));

        int faceAngle = estimateCardinalFaceAngle(face.landmarks);
        Bitmap inferenceCrop = crop;
        Matrix cropToInference = new Matrix();
        if (faceAngle != 0) {
            inferenceCrop = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
            Canvas rotatedCanvas = new Canvas(inferenceCrop);
            cropToInference.setRotate(faceAngle, INPUT_SIZE * 0.5f, INPUT_SIZE * 0.5f);
            rotatedCanvas.drawBitmap(crop, cropToInference, new Paint(Paint.FILTER_BITMAP_FLAG));
        }

        try {
            float[] input = bitmapToInput(inferenceCrop);
            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(input), new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
                 OrtSession.Result outputs = session.run(Collections.singletonMap(inputName, tensor))) {
                float[] local68 = readLandmarkOutput(outputs.get(0).getValue());
                if (local68 == null) return null;

                float score = outputs.size() > 1 ? readHeatmapScore(outputs.get(1).getValue()) : 0.0f;
                if (score < MIN_LANDMARK_SCORE) {
                    Log.d(TAG, "Ignoring low-confidence refined landmarks: score=" + score);
                    return null;
                }

                Matrix inferenceToCrop = new Matrix();
                if (faceAngle != 0 && !cropToInference.invert(inferenceToCrop)) {
                    Log.w(TAG, "Could not invert landmarker pose transform");
                    return null;
                }

                float[] image68 = new float[136];
                float[] point = new float[2];
                for (int i = 0; i < 68; i++) {
                    point[0] = local68[i * 2] * 4.0f;
                    point[1] = local68[i * 2 + 1] * 4.0f;
                    if (faceAngle != 0) inferenceToCrop.mapPoints(point);
                    image68[i * 2] = cropLeft + point[0] * cropSize / INPUT_SIZE;
                    image68[i * 2 + 1] = cropTop + point[1] * cropSize / INPUT_SIZE;
                }

                float[] refined5 = convert68To5(image68);
                if (!isValid(refined5, image.getWidth(), image.getHeight())) return null;
                Log.d(TAG, "2DFAN refinement accepted: score=" + score + ", poseAngle=" + faceAngle);
                return new Result(image68, refined5, score);
            }
        } catch (Exception e) {
            Log.w(TAG, "Landmark refinement failed; using detector landmarks", e);
            return null;
        } finally {
            if (inferenceCrop != crop && !inferenceCrop.isRecycled()) inferenceCrop.recycle();
            if (!crop.isRecycled()) crop.recycle();
        }
    }

    /**
     * FaceFusion's landmarker is rotation-aware. SCRFD already gives us reliable eye points,
     * so use their eye-line orientation to select the same cardinal 0/90/180/270 family.
     * Normal and slight-side portraits stay at 0; rotated photos are normalized for 2DFAN.
     */
    private int estimateCardinalFaceAngle(float[] landmarks5) {
        if (landmarks5 == null || landmarks5.length < 4) return 0;
        float dx = landmarks5[2] - landmarks5[0];
        float dy = landmarks5[3] - landmarks5[1];
        if (!Float.isFinite(dx) || !Float.isFinite(dy) || (Math.abs(dx) + Math.abs(dy)) < 1e-4f) return 0;

        double theta = Math.toDegrees(Math.atan2(dy, dx));
        theta = (theta % 360.0 + 360.0) % 360.0;
        int[] angles = {0, 90, 180, 270};
        int best = 0;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int angle : angles) {
            double distance = Math.abs(theta - angle);
            distance = Math.min(distance, 360.0 - distance);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = angle;
            }
        }
        return best;
    }

    private float[] bitmapToInput(Bitmap bitmap) {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        int plane = pixels.length;
        float[] output = new float[plane * 3];
        for (int i = 0; i < plane; i++) {
            int pixel = pixels[i];
            // 2DFAN consumes the OpenCV/BGR crop used by FaceFusion.
            output[i] = (pixel & 0xFF) / 255.0f;
            output[plane + i] = ((pixel >> 8) & 0xFF) / 255.0f;
            output[plane * 2 + i] = ((pixel >> 16) & 0xFF) / 255.0f;
        }
        return output;
    }

    private float[] readLandmarkOutput(Object value) {
        float[] result = new float[136];
        if (value instanceof float[][][]) {
            float[][][] data = (float[][][]) value;
            if (data.length < 1 || data[0].length < 68) return null;
            for (int i = 0; i < 68; i++) {
                if (data[0][i].length < 2) return null;
                result[i * 2] = data[0][i][0];
                result[i * 2 + 1] = data[0][i][1];
            }
            return result;
        }
        if (value instanceof float[][]) {
            float[][] data = (float[][]) value;
            if (data.length >= 68 && data[0].length >= 2) {
                for (int i = 0; i < 68; i++) {
                    result[i * 2] = data[i][0];
                    result[i * 2 + 1] = data[i][1];
                }
                return result;
            }
            if (data.length == 1 && data[0].length >= 136) {
                int stride = data[0].length >= 204 ? 3 : 2;
                for (int i = 0; i < 68; i++) {
                    result[i * 2] = data[0][i * stride];
                    result[i * 2 + 1] = data[0][i * stride + 1];
                }
                return result;
            }
        }
        Log.w(TAG, "Unexpected 2DFAN4 landmark output type: " + value.getClass().getName());
        return null;
    }

    private float readHeatmapScore(Object value) {
        if (!(value instanceof float[][][][])) return 0.0f;
        float[][][][] heatmap = (float[][][][]) value;
        if (heatmap.length < 1 || heatmap[0].length < 68) return 0.0f;
        double sum = 0.0;
        for (int i = 0; i < 68; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int y = 0; y < heatmap[0][i].length; y++) {
                for (int x = 0; x < heatmap[0][i][y].length; x++) {
                    max = Math.max(max, heatmap[0][i][y][x]);
                }
            }
            if (!Float.isFinite(max)) return 0.0f;
            sum += max;
        }
        float rawMean = (float) (sum / 68.0);
        return Math.max(0.0f, Math.min(1.0f, rawMean / 0.9f));
    }

    private float[] convert68To5(float[] points) {
        float[] five = new float[10];
        averageRange(points, 36, 42, five, 0);
        averageRange(points, 42, 48, five, 2);
        copyPoint(points, 30, five, 4);
        copyPoint(points, 48, five, 6);
        copyPoint(points, 54, five, 8);
        return five;
    }

    private void averageRange(float[] source, int start, int end, float[] dest, int destOffset) {
        float x = 0.0f;
        float y = 0.0f;
        for (int i = start; i < end; i++) {
            x += source[i * 2];
            y += source[i * 2 + 1];
        }
        float count = end - start;
        dest[destOffset] = x / count;
        dest[destOffset + 1] = y / count;
    }

    private void copyPoint(float[] source, int index, float[] dest, int destOffset) {
        dest[destOffset] = source[index * 2];
        dest[destOffset + 1] = source[index * 2 + 1];
    }

    private boolean isValid(float[] landmarks, int width, int height) {
        for (int i = 0; i < landmarks.length; i += 2) {
            float x = landmarks[i];
            float y = landmarks[i + 1];
            if (!Float.isFinite(x) || !Float.isFinite(y)) return false;
            if (x < -width * 0.25f || x > width * 1.25f || y < -height * 0.25f || y > height * 1.25f) return false;
        }
        return true;
    }

    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (Exception e) {
                Log.w(TAG, "Error closing landmarker", e);
            }
            session = null;
        }
    }
}
