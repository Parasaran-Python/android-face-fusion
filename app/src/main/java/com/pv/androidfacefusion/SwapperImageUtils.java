package com.pv.androidfacefusion;

import android.graphics.Bitmap;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * HyperSwap 256 alignment and paste-back using the same geometry, affine
 * estimation, interpolation, border handling and soft oval mask as ReActor.
 */
public final class SwapperImageUtils {
    private static volatile boolean openCvReady;

    private SwapperImageUtils() {}

    private static final float[][] HYPERSWAP_256_NORMALIZED = {
        {84.87f / 256.0f, 105.94f / 256.0f},
        {171.13f / 256.0f, 105.94f / 256.0f},
        {128.00f / 256.0f, 146.66f / 256.0f},
        {96.95f / 256.0f, 188.64f / 256.0f},
        {159.05f / 256.0f, 188.64f / 256.0f}
    };

    public static Bitmap alignFace(Bitmap image, float[] landmarks, int targetSize) {
        if (landmarks == null || landmarks.length < 10) {
            return Bitmap.createScaledBitmap(image, targetSize, targetSize, true);
        }

        ensureOpenCv();
        Mat affine = estimateAffineTransform(unpackLandmarks(landmarks), scaledTemplate(targetSize));
        Mat source = new Mat();
        Mat aligned = new Mat();
        try {
            Utils.bitmapToMat(image, source);
            Imgproc.warpAffine(
                source,
                aligned,
                affine,
                new Size(targetSize, targetSize),
                Imgproc.INTER_CUBIC,
                Core.BORDER_REFLECT,
                new Scalar(0, 0, 0, 255));

            Bitmap result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(aligned, result);
            return result;
        } finally {
            source.release();
            aligned.release();
            affine.release();
        }
    }

    public static Bitmap blendFace(Bitmap targetImage, Bitmap swappedFace, float[] landmarks, int faceSize) {
        if (landmarks == null || landmarks.length < 10) {
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }

        ensureOpenCv();
        Mat affine = estimateAffineTransform(unpackLandmarks(landmarks), scaledTemplate(faceSize));

        int width = targetImage.getWidth();
        int height = targetImage.getHeight();

        Mat inverse = new Mat();
        Mat swappedMat = new Mat();
        Mat warpedFaceMat = new Mat();
        Mat cropMask = Mat.zeros(faceSize, faceSize, CvType.CV_32FC1);
        Mat warpedMask = new Mat();

        try {
            Imgproc.invertAffineTransform(affine, inverse);
            Utils.bitmapToMat(swappedFace, swappedMat);

            Imgproc.warpAffine(
                swappedMat,
                warpedFaceMat,
                inverse,
                new Size(width, height),
                Imgproc.INTER_LANCZOS4,
                Core.BORDER_CONSTANT,
                new Scalar(127.5, 127.5, 127.5, 255));

            Imgproc.ellipse(
                cropMask,
                new Point(faceSize / 2.0, faceSize / 2.0),
                new Size(faceSize * 0.35, faceSize * 0.40),
                0.0,
                0.0,
                360.0,
                new Scalar(1.0),
                -1);
            Imgproc.GaussianBlur(cropMask, cropMask, new Size(15, 15), 0.0);

            Imgproc.warpAffine(
                cropMask,
                warpedMask,
                inverse,
                new Size(width, height),
                Imgproc.INTER_CUBIC,
                Core.BORDER_CONSTANT,
                new Scalar(0.0));
            Imgproc.GaussianBlur(warpedMask, warpedMask, new Size(3, 3), 0.0);

            Core.max(warpedMask, new Scalar(0.0), warpedMask);
            Core.min(warpedMask, new Scalar(1.0), warpedMask);

            Bitmap warpedFace = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            try {
                Utils.matToBitmap(warpedFaceMat, warpedFace);
                return blendWithFloatMask(targetImage, warpedFace, warpedMask);
            } finally {
                warpedFace.recycle();
            }
        } finally {
            affine.release();
            inverse.release();
            swappedMat.release();
            warpedFaceMat.release();
            cropMask.release();
            warpedMask.release();
        }
    }

    private static Bitmap blendWithFloatMask(Bitmap target, Bitmap face, Mat mask) {
        int width = target.getWidth();
        int height = target.getHeight();
        int count = width * height;

        int[] targetPixels = new int[count];
        int[] facePixels = new int[count];
        int[] resultPixels = new int[count];
        float[] maskValues = new float[count];

        target.getPixels(targetPixels, 0, width, 0, 0, width, height);
        face.getPixels(facePixels, 0, width, 0, 0, width, height);
        mask.get(0, 0, maskValues);

        for (int i = 0; i < count; i++) {
            float alpha = Math.max(0.0f, Math.min(1.0f, maskValues[i]));
            if (alpha <= 0.0001f) {
                resultPixels[i] = targetPixels[i];
                continue;
            }

            int targetPixel = targetPixels[i];
            int facePixel = facePixels[i];

            int tr = (targetPixel >> 16) & 0xFF;
            int tg = (targetPixel >> 8) & 0xFF;
            int tb = targetPixel & 0xFF;
            int fr = (facePixel >> 16) & 0xFF;
            int fg = (facePixel >> 8) & 0xFF;
            int fb = facePixel & 0xFF;

            int r = clamp(Math.round(tr * (1.0f - alpha) + fr * alpha));
            int g = clamp(Math.round(tg * (1.0f - alpha) + fg * alpha));
            int b = clamp(Math.round(tb * (1.0f - alpha) + fb * alpha));
            resultPixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(resultPixels, 0, width, 0, 0, width, height);
        return result;
    }

    private static void ensureOpenCv() {
        if (openCvReady) return;
        synchronized (SwapperImageUtils.class) {
            if (openCvReady) return;
            if (!OpenCVLoader.initLocal()) {
                throw new IllegalStateException("OpenCV failed to initialize for HyperSwap");
            }
            openCvReady = true;
        }
    }

    private static Mat estimateAffineTransform(float[][] src, float[][] dst) {
        Point[] srcPoints = new Point[src.length];
        Point[] dstPoints = new Point[dst.length];
        for (int i = 0; i < src.length; i++) {
            srcPoints[i] = new Point(src[i][0], src[i][1]);
            dstPoints[i] = new Point(dst[i][0], dst[i][1]);
        }

        MatOfPoint2f srcMat = new MatOfPoint2f(srcPoints);
        MatOfPoint2f dstMat = new MatOfPoint2f(dstPoints);
        try {
            Mat affine = Calib3d.estimateAffinePartial2D(srcMat, dstMat);
            if (affine == null || affine.empty() || affine.rows() != 2 || affine.cols() != 3) {
                if (affine != null) affine.release();
                throw new IllegalArgumentException("Could not estimate HyperSwap affine transform");
            }
            return affine;
        } finally {
            srcMat.release();
            dstMat.release();
        }
    }

    private static float[][] unpackLandmarks(float[] landmarks) {
        float[][] points = new float[5][2];
        for (int i = 0; i < 5; i++) {
            points[i][0] = landmarks[i * 2];
            points[i][1] = landmarks[i * 2 + 1];
        }
        return points;
    }

    private static float[][] scaledTemplate(int size) {
        float[][] result = new float[5][2];
        for (int i = 0; i < 5; i++) {
            result[i][0] = HYPERSWAP_256_NORMALIZED[i][0] * size;
            result[i][1] = HYPERSWAP_256_NORMALIZED[i][1] * size;
        }
        return result;
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
