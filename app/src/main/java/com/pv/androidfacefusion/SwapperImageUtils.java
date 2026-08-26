package com.pv.androidfacefusion;

import android.graphics.Bitmap;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/** OpenCV alignment and paste-back helpers for the FaceFusion/HyperSwap path. */
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

    private static final float[][] ARCFACE_112_V2_NORMALIZED = {
        {0.34191607f, 0.46157411f},
        {0.65653393f, 0.45983393f},
        {0.50022500f, 0.64050536f},
        {0.37097589f, 0.82469196f},
        {0.63151696f, 0.82325089f}
    };

    public static Bitmap alignArcFace112(Bitmap image, float[] landmarks) {
        if (landmarks == null || landmarks.length < 10) {
            return Bitmap.createScaledBitmap(image, 112, 112, true);
        }
        ensureOpenCv();
        Mat affine = estimateSimilarityTransform(
            unpackLandmarks(landmarks), scaledTemplate(ARCFACE_112_V2_NORMALIZED, 112));
        Mat source = new Mat();
        Mat aligned = new Mat();
        try {
            Utils.bitmapToMat(image, source);
            Imgproc.warpAffine(source, aligned, affine, new Size(112, 112), Imgproc.INTER_AREA,
                Core.BORDER_REPLICATE, new Scalar(0, 0, 0, 255));
            Bitmap result = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(aligned, result);
            return result;
        } finally {
            source.release();
            aligned.release();
            affine.release();
        }
    }

    public static Bitmap alignFace(Bitmap image, float[] landmarks, int targetSize) {
        if (landmarks == null || landmarks.length < 10) {
            return Bitmap.createScaledBitmap(image, targetSize, targetSize, true);
        }
        ensureOpenCv();
        Mat affine = estimateSimilarityTransform(
            unpackLandmarks(landmarks), scaledTemplate(HYPERSWAP_256_NORMALIZED, targetSize));
        Mat source = new Mat();
        Mat aligned = new Mat();
        try {
            Utils.bitmapToMat(image, source);
            Imgproc.warpAffine(source, aligned, affine, new Size(targetSize, targetSize), Imgproc.INTER_CUBIC,
                Core.BORDER_REFLECT, new Scalar(0, 0, 0, 255));
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
        Mat affine = estimateSimilarityTransform(
            unpackLandmarks(landmarks), scaledTemplate(HYPERSWAP_256_NORMALIZED, faceSize));
        int width = targetImage.getWidth();
        int height = targetImage.getHeight();
        Mat inverse = invertAffineTransform(affine);
        Mat swappedMat = new Mat();
        Mat warpedFaceMat = new Mat();
        Mat cropMask = Mat.zeros(faceSize, faceSize, CvType.CV_32FC1);
        Mat warpedMask = new Mat();
        try {
            Utils.bitmapToMat(swappedFace, swappedMat);
            Imgproc.warpAffine(swappedMat, warpedFaceMat, inverse, new Size(width, height),
                Imgproc.INTER_LANCZOS4, Core.BORDER_CONSTANT, new Scalar(127.5, 127.5, 127.5, 255));

            Imgproc.ellipse(cropMask, new Point(faceSize / 2.0, faceSize / 2.0),
                new Size(faceSize * 0.35, faceSize * 0.40), 0.0, 0.0, 360.0,
                new Scalar(1.0), -1);
            Imgproc.GaussianBlur(cropMask, cropMask, new Size(15, 15), 0.0);
            Imgproc.warpAffine(cropMask, warpedMask, inverse, new Size(width, height),
                Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(0.0));
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
                throw new IllegalStateException("OpenCV failed to initialize for face processing");
            }
            openCvReady = true;
        }
    }

    /** Least-squares 2D similarity transform, source -> destination. */
    private static Mat estimateSimilarityTransform(float[][] src, float[][] dst) {
        int n = src.length;
        double srcCx = 0.0, srcCy = 0.0, dstCx = 0.0, dstCy = 0.0;
        for (int i = 0; i < n; i++) {
            srcCx += src[i][0];
            srcCy += src[i][1];
            dstCx += dst[i][0];
            dstCy += dst[i][1];
        }
        srcCx /= n;
        srcCy /= n;
        dstCx /= n;
        dstCy /= n;

        double srcNorm = 0.0;
        double a = 0.0;
        double b = 0.0;
        for (int i = 0; i < n; i++) {
            double sx = src[i][0] - srcCx;
            double sy = src[i][1] - srcCy;
            double dx = dst[i][0] - dstCx;
            double dy = dst[i][1] - dstCy;
            srcNorm += sx * sx + sy * sy;
            a += sx * dx + sy * dy;
            b += sx * dy - sy * dx;
        }
        if (srcNorm < 1e-10) {
            throw new IllegalArgumentException("Invalid face landmarks for alignment");
        }

        double m00 = a / srcNorm;
        double m01 = -b / srcNorm;
        double m10 = b / srcNorm;
        double m11 = a / srcNorm;
        double tx = dstCx - (m00 * srcCx + m01 * srcCy);
        double ty = dstCy - (m10 * srcCx + m11 * srcCy);

        Mat affine = new Mat(2, 3, CvType.CV_64FC1);
        affine.put(0, 0, new double[]{m00, m01, tx, m10, m11, ty});
        return affine;
    }

    /** Inverts a 2x3 affine matrix without relying on unavailable Android OpenCV APIs. */
    private static Mat invertAffineTransform(Mat affine) {
        double[] values = new double[6];
        affine.get(0, 0, values);
        double a = values[0];
        double b = values[1];
        double c = values[2];
        double d = values[3];
        double e = values[4];
        double f = values[5];
        double det = a * e - b * d;
        if (Math.abs(det) < 1e-12) {
            throw new IllegalArgumentException("Face affine transform is not invertible");
        }

        double ia = e / det;
        double ib = -b / det;
        double id = -d / det;
        double ie = a / det;
        double ic = (b * f - e * c) / det;
        double iff = (d * c - a * f) / det;

        Mat inverse = new Mat(2, 3, CvType.CV_64FC1);
        inverse.put(0, 0, new double[]{ia, ib, ic, id, ie, iff});
        return inverse;
    }

    private static float[][] unpackLandmarks(float[] landmarks) {
        float[][] points = new float[5][2];
        for (int i = 0; i < 5; i++) {
            points[i][0] = landmarks[i * 2];
            points[i][1] = landmarks[i * 2 + 1];
        }
        return points;
    }

    private static float[][] scaledTemplate(float[][] normalizedTemplate, int size) {
        float[][] result = new float[5][2];
        for (int i = 0; i < 5; i++) {
            result[i][0] = normalizedTemplate[i][0] * size;
            result[i][1] = normalizedTemplate[i][1] * size;
        }
        return result;
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
