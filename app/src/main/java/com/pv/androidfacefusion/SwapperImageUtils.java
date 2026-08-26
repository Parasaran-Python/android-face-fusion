package com.pv.androidfacefusion;

import android.graphics.Bitmap;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
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
        return blendFace(targetImage, null, swappedFace, landmarks, faceSize, null);
    }

    /**
     * Natural high-resolution paste-back. The semantic mask preserves hair/background while
     * colour matching adapts the swapped skin to the target lighting. Only the affected ROI
     * is warped, so full-resolution target images do not require full-frame temporary masks.
     */
    public static Bitmap blendFace(Bitmap targetImage, Bitmap alignedTarget, Bitmap swappedFace,
                                   float[] landmarks, int faceSize, float[] semanticMask) {
        if (landmarks == null || landmarks.length < 10) {
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }
        ensureOpenCv();

        Mat affine = estimateSimilarityTransform(
            unpackLandmarks(landmarks), scaledTemplate(HYPERSWAP_256_NORMALIZED, faceSize));
        Mat inverse = invertAffineTransform(affine);
        int[] roi = calculatePasteRoi(inverse, faceSize, targetImage.getWidth(), targetImage.getHeight());
        int roiWidth = roi[2] - roi[0];
        int roiHeight = roi[3] - roi[1];
        if (roiWidth <= 0 || roiHeight <= 0) {
            affine.release();
            inverse.release();
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }

        float[] cropMaskValues = createNaturalCropMask(faceSize, semanticMask);
        Bitmap correctedFace = alignedTarget != null
            ? matchColorAndLighting(alignedTarget, swappedFace, cropMaskValues)
            : swappedFace;

        Mat cropMask = new Mat(faceSize, faceSize, CvType.CV_32FC1);
        Mat swappedMat = new Mat();
        Mat warpedFaceMat = new Mat();
        Mat warpedMask = new Mat();
        Mat roiMatrix = offsetAffine(inverse, roi[0], roi[1]);
        try {
            cropMask.put(0, 0, cropMaskValues);
            Imgproc.GaussianBlur(cropMask, cropMask, new Size(15, 15), 0.0);
            Core.max(cropMask, new Scalar(0.0), cropMask);
            Core.min(cropMask, new Scalar(1.0), cropMask);

            Utils.bitmapToMat(correctedFace, swappedMat);
            Imgproc.warpAffine(swappedMat, warpedFaceMat, roiMatrix, new Size(roiWidth, roiHeight),
                Imgproc.INTER_LANCZOS4, Core.BORDER_CONSTANT, new Scalar(127.5, 127.5, 127.5, 255));
            Imgproc.warpAffine(cropMask, warpedMask, roiMatrix, new Size(roiWidth, roiHeight),
                Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(0.0));
            Imgproc.GaussianBlur(warpedMask, warpedMask, new Size(3, 3), 0.0);
            Core.max(warpedMask, new Scalar(0.0), warpedMask);
            Core.min(warpedMask, new Scalar(1.0), warpedMask);

            Bitmap warpedFace = Bitmap.createBitmap(roiWidth, roiHeight, Bitmap.Config.ARGB_8888);
            try {
                Utils.matToBitmap(warpedFaceMat, warpedFace);
                return blendRoi(targetImage, warpedFace, warpedMask, roi[0], roi[1]);
            } finally {
                warpedFace.recycle();
            }
        } finally {
            affine.release();
            inverse.release();
            roiMatrix.release();
            cropMask.release();
            swappedMat.release();
            warpedFaceMat.release();
            warpedMask.release();
            if (correctedFace != swappedFace && !correctedFace.isRecycled()) correctedFace.recycle();
        }
    }

    private static float[] createNaturalCropMask(int size, float[] semanticMask) {
        float[] mask = new float[size * size];
        boolean hasSemantic = semanticMask != null && semanticMask.length == mask.length;
        double cx = (size - 1) * 0.5;
        double cy = (size - 1) * 0.5;
        double rx = size * 0.39;
        double ry = size * 0.44;
        for (int y = 0; y < size; y++) {
            double dy = (y - cy) / ry;
            for (int x = 0; x < size; x++) {
                double dx = (x - cx) / rx;
                float oval = dx * dx + dy * dy <= 1.0 ? 1.0f : 0.0f;
                int index = y * size + x;
                mask[index] = hasSemantic ? clamp01(semanticMask[index]) * oval : oval;
            }
        }
        return mask;
    }

    /** Conservative masked channel transfer: adapts broad skin tone/lighting while preserving detail. */
    private static Bitmap matchColorAndLighting(Bitmap target, Bitmap swapped, float[] mask) {
        int width = swapped.getWidth();
        int height = swapped.getHeight();
        Bitmap targetSized = target;
        if (target.getWidth() != width || target.getHeight() != height) {
            targetSized = Bitmap.createScaledBitmap(target, width, height, true);
        }
        int count = width * height;
        int[] targetPixels = new int[count];
        int[] swappedPixels = new int[count];
        targetSized.getPixels(targetPixels, 0, width, 0, 0, width, height);
        swapped.getPixels(swappedPixels, 0, width, 0, 0, width, height);

        double[] targetMean = new double[3];
        double[] swapMean = new double[3];
        double weightSum = 0.0;
        for (int i = 0; i < count; i++) {
            float weight = mask != null && mask.length == count ? mask[i] : 1.0f;
            if (weight < 0.2f) continue;
            int tp = targetPixels[i];
            int sp = swappedPixels[i];
            targetMean[0] += ((tp >> 16) & 0xFF) * weight;
            targetMean[1] += ((tp >> 8) & 0xFF) * weight;
            targetMean[2] += (tp & 0xFF) * weight;
            swapMean[0] += ((sp >> 16) & 0xFF) * weight;
            swapMean[1] += ((sp >> 8) & 0xFF) * weight;
            swapMean[2] += (sp & 0xFF) * weight;
            weightSum += weight;
        }
        if (weightSum < 32.0) {
            if (targetSized != target) targetSized.recycle();
            return swapped.copy(Bitmap.Config.ARGB_8888, false);
        }
        for (int c = 0; c < 3; c++) {
            targetMean[c] /= weightSum;
            swapMean[c] /= weightSum;
        }

        double[] targetVar = new double[3];
        double[] swapVar = new double[3];
        for (int i = 0; i < count; i++) {
            float weight = mask != null && mask.length == count ? mask[i] : 1.0f;
            if (weight < 0.2f) continue;
            int tp = targetPixels[i];
            int sp = swappedPixels[i];
            double[] tv = {((tp >> 16) & 0xFF), ((tp >> 8) & 0xFF), (tp & 0xFF)};
            double[] sv = {((sp >> 16) & 0xFF), ((sp >> 8) & 0xFF), (sp & 0xFF)};
            for (int c = 0; c < 3; c++) {
                double td = tv[c] - targetMean[c];
                double sd = sv[c] - swapMean[c];
                targetVar[c] += td * td * weight;
                swapVar[c] += sd * sd * weight;
            }
        }

        double[] scale = new double[3];
        for (int c = 0; c < 3; c++) {
            double targetStd = Math.sqrt(targetVar[c] / weightSum);
            double swapStd = Math.sqrt(swapVar[c] / weightSum);
            double raw = swapStd > 1.0 ? targetStd / swapStd : 1.0;
            scale[c] = Math.max(0.85, Math.min(1.18, raw));
        }

        final double strength = 0.58;
        int[] output = new int[count];
        for (int i = 0; i < count; i++) {
            int sp = swappedPixels[i];
            int sr = (sp >> 16) & 0xFF;
            int sg = (sp >> 8) & 0xFF;
            int sb = sp & 0xFF;
            int[] source = {sr, sg, sb};
            int[] corrected = new int[3];
            for (int c = 0; c < 3; c++) {
                double mapped = (source[c] - swapMean[c]) * scale[c] + targetMean[c];
                corrected[c] = clamp((int) Math.round(source[c] * (1.0 - strength) + mapped * strength));
            }
            output[i] = 0xFF000000 | (corrected[0] << 16) | (corrected[1] << 8) | corrected[2];
        }
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        if (targetSized != target) targetSized.recycle();
        return result;
    }

    private static Bitmap blendRoi(Bitmap target, Bitmap face, Mat mask, int x, int y) {
        int width = face.getWidth();
        int height = face.getHeight();
        int count = width * height;
        int[] targetPixels = new int[count];
        int[] facePixels = new int[count];
        float[] maskValues = new float[count];
        target.getPixels(targetPixels, 0, width, x, y, width, height);
        face.getPixels(facePixels, 0, width, 0, 0, width, height);
        mask.get(0, 0, maskValues);
        for (int i = 0; i < count; i++) {
            float alpha = clamp01(maskValues[i]);
            if (alpha <= 0.0001f) continue;
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
            targetPixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        Bitmap result = target.copy(Bitmap.Config.ARGB_8888, true);
        result.setPixels(targetPixels, 0, width, x, y, width, height);
        return result;
    }

    private static int[] calculatePasteRoi(Mat inverse, int faceSize, int targetWidth, int targetHeight) {
        double[] m = new double[6];
        inverse.get(0, 0, m);
        double[][] corners = {{0, 0}, {faceSize, 0}, {faceSize, faceSize}, {0, faceSize}};
        double minX = Double.POSITIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        for (double[] corner : corners) {
            double tx = m[0] * corner[0] + m[1] * corner[1] + m[2];
            double ty = m[3] * corner[0] + m[4] * corner[1] + m[5];
            minX = Math.min(minX, tx);
            minY = Math.min(minY, ty);
            maxX = Math.max(maxX, tx);
            maxY = Math.max(maxY, ty);
        }
        int x1 = Math.max(0, (int) Math.floor(minX) - 2);
        int y1 = Math.max(0, (int) Math.floor(minY) - 2);
        int x2 = Math.min(targetWidth, (int) Math.ceil(maxX) + 2);
        int y2 = Math.min(targetHeight, (int) Math.ceil(maxY) + 2);
        return new int[]{x1, y1, x2, y2};
    }

    private static Mat offsetAffine(Mat affine, int xOffset, int yOffset) {
        double[] m = new double[6];
        affine.get(0, 0, m);
        m[2] -= xOffset;
        m[5] -= yOffset;
        Mat result = new Mat(2, 3, CvType.CV_64FC1);
        result.put(0, 0, m);
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

    private static float clamp01(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
