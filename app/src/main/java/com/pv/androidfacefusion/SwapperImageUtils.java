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

/**
 * HyperSwap 256 alignment and paste-back using the same geometry, interpolation,
 * border handling and soft oval mask as the proven ReActor CPU implementation.
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
        double[] affineValues = estimateSimilarityTransform(
            unpackLandmarks(landmarks), scaledTemplate(targetSize));

        Mat source = new Mat();
        Mat aligned = new Mat();
        Mat affine = affineMat(affineValues);
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
        double[] affineValues = estimateSimilarityTransform(
            unpackLandmarks(landmarks), scaledTemplate(faceSize));

        int width = targetImage.getWidth();
        int height = targetImage.getHeight();

        Mat affine = affineMat(affineValues);
        Mat inverse = new Mat();
        Mat swappedMat = new Mat();
        Mat warpedFaceMat = new Mat();
        Mat cropMask = Mat.zeros(faceSize, faceSize, CvType.CV_32FC1);
        Mat warpedMask = new Mat();

        try {
            Imgproc.invertAffineTransform(affine, inverse);
            Utils.bitmapToMat(swappedFace, swappedMat);

            // ReActor paste_back: high quality inverse face warp with a neutral gray border.
            Imgproc.warpAffine(
                swappedMat,
                warpedFaceMat,
                inverse,
                new Size(width, height),
                Imgproc.INTER_LANCZOS4,
                Core.BORDER_CONSTANT,
                new Scalar(127.5, 127.5, 127.5, 255));

            // ReActor mask: centered oval, axes 35% x 40%, Gaussian blur 15.
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

    private static Mat affineMat(double[] values) {
        Mat affine = new Mat(2, 3, CvType.CV_64FC1);
        affine.put(0, 0, values);
        return affine;
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

    /** Closed-form least-squares 2D similarity transform, source -> destination. */
    private static double[] estimateSimilarityTransform(float[][] src, float[][] dst) {
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
            throw new IllegalArgumentException("Invalid face landmarks for HyperSwap alignment");
        }

        double m00 = a / srcNorm;
        double m01 = -b / srcNorm;
        double m10 = b / srcNorm;
        double m11 = a / srcNorm;
        double tx = dstCx - (m00 * srcCx + m01 * srcCy);
        double ty = dstCy - (m10 * srcCx + m11 * srcCy);

        return new double[]{m00, m01, tx, m10, m11, ty};
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
