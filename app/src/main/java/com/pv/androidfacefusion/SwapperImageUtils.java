package com.pv.androidfacefusion;

import android.graphics.Bitmap;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

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
        return blendFace(targetImage, null, swappedFace, landmarks, null, faceSize, null);
    }

    public static Bitmap blendFace(Bitmap targetImage, Bitmap alignedTarget, Bitmap swappedFace,
                                   float[] landmarks, int faceSize, float[] semanticMask) {
        return blendFace(targetImage, alignedTarget, swappedFace, landmarks, null, faceSize, semanticMask);
    }

    /** Natural high-resolution paste-back with semantic, expression and pose-aware blending. */
    public static Bitmap blendFace(Bitmap targetImage, Bitmap alignedTarget, Bitmap swappedFace,
                                   float[] landmarks5, float[] landmarks68, int faceSize, float[] semanticMask) {
        if (landmarks5 == null || landmarks5.length < 10) {
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }
        ensureOpenCv();

        Mat affine = estimateSimilarityTransform(
            unpackLandmarks(landmarks5), scaledTemplate(HYPERSWAP_256_NORMALIZED, faceSize));
        Mat inverse = invertAffineTransform(affine);
        int[] roi = calculatePasteRoi(inverse, faceSize, targetImage.getWidth(), targetImage.getHeight());
        int roiWidth = roi[2] - roi[0];
        int roiHeight = roi[3] - roi[1];
        if (roiWidth <= 0 || roiHeight <= 0) {
            affine.release();
            inverse.release();
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }

        float[] cropMaskValues = createNaturalCropMask(faceSize, semanticMask, affine, landmarks68);
        Bitmap correctedFace = alignedTarget != null
            ? matchColorLightingAndTexture(alignedTarget, swappedFace, cropMaskValues)
            : swappedFace;

        Mat cropMask = new Mat(faceSize, faceSize, CvType.CV_32FC1);
        Mat swappedMat = new Mat();
        Mat warpedFaceMat = new Mat();
        Mat warpedMask = new Mat();
        Mat roiMatrix = offsetAffine(inverse, roi[0], roi[1]);
        try {
            cropMask.put(0, 0, cropMaskValues);
            Imgproc.GaussianBlur(cropMask, cropMask, new Size(0, 0), Math.max(3.0, faceSize * 0.014));
            Core.max(cropMask, new Scalar(0.0), cropMask);
            Core.min(cropMask, new Scalar(1.0), cropMask);

            Utils.bitmapToMat(correctedFace, swappedMat);
            Imgproc.warpAffine(swappedMat, warpedFaceMat, roiMatrix, new Size(roiWidth, roiHeight),
                Imgproc.INTER_LANCZOS4, Core.BORDER_CONSTANT, new Scalar(127.5, 127.5, 127.5, 255));
            Imgproc.warpAffine(cropMask, warpedMask, roiMatrix, new Size(roiWidth, roiHeight),
                Imgproc.INTER_CUBIC, Core.BORDER_CONSTANT, new Scalar(0.0));
            Imgproc.GaussianBlur(warpedMask, warpedMask, new Size(0, 0),
                Math.max(1.4, Math.min(3.6, Math.max(roiWidth, roiHeight) * 0.005)));
            Core.max(warpedMask, new Scalar(0.0), warpedMask);
            Core.min(warpedMask, new Scalar(1.0), warpedMask);

            Bitmap warpedFace = Bitmap.createBitmap(roiWidth, roiHeight, Bitmap.Config.ARGB_8888);
            try {
                Utils.matToBitmap(warpedFaceMat, warpedFace);
                return blendRoiEdgeAware(targetImage, warpedFace, warpedMask, roi[0], roi[1]);
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

    private static float[] createNaturalCropMask(int size, float[] semanticMask,
                                                  Mat affine, float[] landmarks68) {
        float[] mask = new float[size * size];
        boolean hasSemantic = semanticMask != null && semanticMask.length == mask.length;
        if (hasSemantic) {
            for (int i = 0; i < mask.length; i++) mask[i] = clamp01(semanticMask[i]);
        } else {
            double cx = (size - 1) * 0.5;
            double cy = (size - 1) * 0.5;
            double rx = size * 0.39;
            double ry = size * 0.44;
            for (int y = 0; y < size; y++) {
                double dy = (y - cy) / ry;
                for (int x = 0; x < size; x++) {
                    double dx = (x - cx) / rx;
                    mask[y * size + x] = dx * dx + dy * dy <= 1.0 ? 1.0f : 0.0f;
                }
            }
        }

        if (landmarks68 != null && landmarks68.length >= 136) {
            float[] aligned68 = transformLandmarks68(affine, landmarks68);
            float[] poseMask = createPoseContourMask(size, aligned68);
            if (poseMask != null) {
                for (int i = 0; i < mask.length; i++) {
                    float poseInfluence = 0.38f + 0.62f * clamp01(poseMask[i]);
                    mask[i] *= poseInfluence;
                }
            }
            applyPoseAsymmetry(mask, size, aligned68);
            applyExpressionAwareProtection(mask, size, aligned68);
        }
        return mask;
    }

    /** Build a broad soft silhouette. BiSeNet remains the hard hair/background protection. */
    private static float[] createPoseContourMask(int size, float[] aligned68) {
        List<Point> polygon = new ArrayList<>();
        for (int i = 0; i <= 16; i++) {
            polygon.add(new Point(aligned68[i * 2], aligned68[i * 2 + 1]));
        }
        double foreheadLift = size * 0.135;
        for (int i = 26; i >= 17; i--) {
            polygon.add(new Point(aligned68[i * 2], aligned68[i * 2 + 1] - foreheadLift));
        }
        if (polygon.size() < 3) return null;

        Mat contourMask = Mat.zeros(size, size, CvType.CV_32FC1);
        MatOfPoint contour = new MatOfPoint();
        try {
            contour.fromList(polygon);
            Imgproc.fillConvexPoly(contourMask, contour, new Scalar(1.0));
            Imgproc.GaussianBlur(contourMask, contourMask, new Size(0, 0), Math.max(3.0, size * 0.020));
            Core.max(contourMask, new Scalar(0.0), contourMask);
            Core.min(contourMask, new Scalar(1.0), contourMask);
            float[] result = new float[size * size];
            contourMask.get(0, 0, result);
            return result;
        } finally {
            contour.release();
            contourMask.release();
        }
    }

    /** Slightly protect the far cheek/temple on turned faces without creating a hard boundary. */
    private static void applyPoseAsymmetry(float[] mask, int size, float[] p) {
        float leftEye = (p[36 * 2] + p[39 * 2]) * 0.5f;
        float rightEye = (p[42 * 2] + p[45 * 2]) * 0.5f;
        float eyeMid = (leftEye + rightEye) * 0.5f;
        float eyeSpan = Math.max(1.0f, Math.abs(rightEye - leftEye));
        float noseX = p[30 * 2];
        float yaw = Math.max(-1.0f, Math.min(1.0f, (noseX - eyeMid) / (eyeSpan * 0.52f)));
        if (Math.abs(yaw) < 0.10f) return;

        boolean farSideLeft = yaw > 0.0f;
        float strength = Math.min(0.18f, Math.abs(yaw) * 0.16f);
        float center = size * 0.5f;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                boolean onFarSide = farSideLeft ? x < center : x > center;
                if (!onFarSide) continue;
                float distance = Math.abs(x - center) / center;
                if (distance < 0.35f) continue;
                float factor = 1.0f - strength * ((distance - 0.35f) / 0.65f);
                int i = y * size + x;
                mask[i] *= Math.max(0.80f, factor);
            }
        }
    }

    /** Preserve the target's gaze and mouth mechanics according to actual expression openness. */
    private static void applyExpressionAwareProtection(float[] mask, int size, float[] p) {
        float leftEyeOpen = opennessRatio(p, 36, 39, 37, 41, 38, 40);
        float rightEyeOpen = opennessRatio(p, 42, 45, 43, 47, 44, 46);
        applySoftFeatureProtection(mask, size, featureCenter(p, 36, 41),
            featureWidth(p, 36, 39) * 0.72f, size * 0.030f,
            eyeProtectionWeight(leftEyeOpen));
        applySoftFeatureProtection(mask, size, featureCenter(p, 42, 47),
            featureWidth(p, 42, 45) * 0.72f, size * 0.030f,
            eyeProtectionWeight(rightEyeOpen));

        float mouthWidth = Math.max(1.0f, distance(p, 48, 54));
        float mouthOpen = distance(p, 62, 66) / mouthWidth;
        float[] mouthCenter = featureCenter(p, 48, 67);
        float mouthWeight = mouthOpen > 0.16f ? 0.10f : (mouthOpen > 0.08f ? 0.18f : 0.28f);
        applySoftFeatureProtection(mask, size, mouthCenter,
            mouthWidth * 0.60f, Math.max(size * 0.035f, mouthWidth * 0.22f), mouthWeight);
    }

    private static float eyeProtectionWeight(float openness) {
        if (openness > 0.32f) return 0.12f;
        if (openness > 0.20f) return 0.20f;
        return 0.32f;
    }

    private static float opennessRatio(float[] p, int left, int right,
                                       int upperA, int lowerA, int upperB, int lowerB) {
        float width = Math.max(1.0f, distance(p, left, right));
        float vertical = (distance(p, upperA, lowerA) + distance(p, upperB, lowerB)) * 0.5f;
        return vertical / width;
    }

    private static void applySoftFeatureProtection(float[] mask, int size, float[] center,
                                                   float radiusX, float radiusY, float minimumWeight) {
        radiusX = Math.max(4.0f, radiusX);
        radiusY = Math.max(4.0f, radiusY);
        int x0 = Math.max(0, (int) Math.floor(center[0] - radiusX * 1.6f));
        int x1 = Math.min(size - 1, (int) Math.ceil(center[0] + radiusX * 1.6f));
        int y0 = Math.max(0, (int) Math.floor(center[1] - radiusY * 1.6f));
        int y1 = Math.min(size - 1, (int) Math.ceil(center[1] + radiusY * 1.6f));
        for (int y = y0; y <= y1; y++) {
            float dy = (y - center[1]) / radiusY;
            for (int x = x0; x <= x1; x++) {
                float dx = (x - center[0]) / radiusX;
                float d2 = dx * dx + dy * dy;
                if (d2 >= 2.56f) continue;
                float core = clamp01(1.0f - (float) Math.sqrt(d2) / 1.6f);
                float localLimit = 1.0f - core * (1.0f - minimumWeight);
                int i = y * size + x;
                mask[i] = Math.min(mask[i], localLimit);
            }
        }
    }

    private static float[] transformLandmarks68(Mat affine, float[] landmarks68) {
        double[] a = new double[6];
        affine.get(0, 0, a);
        float[] result = new float[136];
        for (int i = 0; i < 68; i++) {
            double x = landmarks68[i * 2];
            double y = landmarks68[i * 2 + 1];
            result[i * 2] = (float) (a[0] * x + a[1] * y + a[2]);
            result[i * 2 + 1] = (float) (a[3] * x + a[4] * y + a[5]);
        }
        return result;
    }

    private static float[] featureCenter(float[] p, int start, int end) {
        float x = 0.0f, y = 0.0f;
        int count = end - start + 1;
        for (int i = start; i <= end; i++) {
            x += p[i * 2];
            y += p[i * 2 + 1];
        }
        return new float[]{x / count, y / count};
    }

    private static float featureWidth(float[] p, int left, int right) {
        return Math.max(1.0f, distance(p, left, right));
    }

    private static float distance(float[] p, int a, int b) {
        float dx = p[a * 2] - p[b * 2];
        float dy = p[a * 2 + 1] - p[b * 2 + 1];
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Match broad colour, then transfer target low-frequency luminance and a conservative amount
     * of target high-frequency texture. This keeps identity from HyperSwap while restoring the
     * target photo's lighting, pores and fine camera texture.
     */
    private static Bitmap matchColorLightingAndTexture(Bitmap target, Bitmap swapped, float[] mask) {
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
            if (weight < 0.18f) continue;
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
        float[] targetLuma = new float[count];
        float[] swapLuma = new float[count];
        for (int i = 0; i < count; i++) {
            int tp = targetPixels[i];
            int sp = swappedPixels[i];
            int tr = (tp >> 16) & 0xFF, tg = (tp >> 8) & 0xFF, tb = tp & 0xFF;
            int sr = (sp >> 16) & 0xFF, sg = (sp >> 8) & 0xFF, sb = sp & 0xFF;
            targetLuma[i] = 0.299f * tr + 0.587f * tg + 0.114f * tb;
            swapLuma[i] = 0.299f * sr + 0.587f * sg + 0.114f * sb;
            float weight = mask != null && mask.length == count ? mask[i] : 1.0f;
            if (weight < 0.18f) continue;
            double[] tv = {tr, tg, tb};
            double[] sv = {sr, sg, sb};
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
            scale[c] = Math.max(0.88, Math.min(1.15, raw));
        }

        int lowRadius = Math.max(5, Math.min(22, Math.round(Math.min(width, height) * 0.025f)));
        float[] targetLow = boxBlur(targetLuma, width, height, lowRadius);
        float[] swapLow = boxBlur(swapLuma, width, height, lowRadius);

        final double colourStrength = 0.48;
        final float lightingStrength = 0.34f;
        final float textureStrength = 0.17f;
        int[] output = new int[count];
        for (int i = 0; i < count; i++) {
            int sp = swappedPixels[i];
            int sr = (sp >> 16) & 0xFF, sg = (sp >> 8) & 0xFF, sb = sp & 0xFF;
            int[] source = {sr, sg, sb};
            float naturalWeight = mask != null && mask.length == count ? clamp01(mask[i] * 1.25f) : 1.0f;
            float lightingDelta = (targetLow[i] - swapLow[i]) * lightingStrength * naturalWeight;
            float targetDetail = (targetLuma[i] - targetLow[i]) * textureStrength * naturalWeight;
            int[] corrected = new int[3];
            for (int c = 0; c < 3; c++) {
                double mapped = (source[c] - swapMean[c]) * scale[c] + targetMean[c];
                double colour = source[c] * (1.0 - colourStrength) + mapped * colourStrength;
                corrected[c] = clamp((int) Math.round(colour + lightingDelta + targetDetail));
            }
            output[i] = 0xFF000000 | (corrected[0] << 16) | (corrected[1] << 8) | corrected[2];
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        if (targetSized != target) targetSized.recycle();
        return result;
    }

    /** Fast separable box blur used only for low-frequency luminance transfer. */
    private static float[] boxBlur(float[] source, int width, int height, int radius) {
        if (radius <= 0) return source.clone();
        float[] horizontal = new float[source.length];
        float[] result = new float[source.length];
        float[] prefix = new float[Math.max(width, height) + 1];

        for (int y = 0; y < height; y++) {
            prefix[0] = 0.0f;
            int row = y * width;
            for (int x = 0; x < width; x++) prefix[x + 1] = prefix[x] + source[row + x];
            for (int x = 0; x < width; x++) {
                int left = Math.max(0, x - radius);
                int right = Math.min(width - 1, x + radius);
                horizontal[row + x] = (prefix[right + 1] - prefix[left]) / (right - left + 1);
            }
        }

        for (int x = 0; x < width; x++) {
            prefix[0] = 0.0f;
            for (int y = 0; y < height; y++) prefix[y + 1] = prefix[y] + horizontal[y * width + x];
            for (int y = 0; y < height; y++) {
                int top = Math.max(0, y - radius);
                int bottom = Math.min(height - 1, y + radius);
                result[y * width + x] = (prefix[bottom + 1] - prefix[top]) / (bottom - top + 1);
            }
        }
        return result;
    }

    /** Preserve strong target edges slightly so jaw shadows, eyelids and facial folds remain coherent. */
    private static Bitmap blendRoiEdgeAware(Bitmap target, Bitmap face, Mat mask, int x, int y) {
        int width = face.getWidth();
        int height = face.getHeight();
        int count = width * height;
        int[] targetPixels = new int[count];
        int[] facePixels = new int[count];
        float[] maskValues = new float[count];
        float[] luma = new float[count];
        target.getPixels(targetPixels, 0, width, x, y, width, height);
        face.getPixels(facePixels, 0, width, 0, 0, width, height);
        mask.get(0, 0, maskValues);
        for (int i = 0; i < count; i++) {
            int p = targetPixels[i];
            luma[i] = 0.299f * ((p >> 16) & 0xFF) + 0.587f * ((p >> 8) & 0xFF) + 0.114f * (p & 0xFF);
        }

        for (int py = 0; py < height; py++) {
            for (int px = 0; px < width; px++) {
                int i = py * width + px;
                float alpha = clamp01(maskValues[i]);
                if (alpha <= 0.0001f) continue;

                if (px > 0 && px + 1 < width && py > 0 && py + 1 < height) {
                    float gx = Math.abs(luma[i + 1] - luma[i - 1]);
                    float gy = Math.abs(luma[i + width] - luma[i - width]);
                    float edge = clamp01((gx + gy) / 110.0f);
                    alpha *= 1.0f - 0.12f * edge;
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
                targetPixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }

        Bitmap result = target.isMutable() ? target : target.copy(Bitmap.Config.ARGB_8888, true);
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
            if (!OpenCVLoader.initLocal()) throw new IllegalStateException("OpenCV failed to initialize for face processing");
            openCvReady = true;
        }
    }

    /** Least-squares 2D similarity transform, source -> destination. */
    private static Mat estimateSimilarityTransform(float[][] src, float[][] dst) {
        int n = src.length;
        double srcCx = 0.0, srcCy = 0.0, dstCx = 0.0, dstCy = 0.0;
        for (int i = 0; i < n; i++) {
            srcCx += src[i][0]; srcCy += src[i][1]; dstCx += dst[i][0]; dstCy += dst[i][1];
        }
        srcCx /= n; srcCy /= n; dstCx /= n; dstCy /= n;

        double srcNorm = 0.0, a = 0.0, b = 0.0;
        for (int i = 0; i < n; i++) {
            double sx = src[i][0] - srcCx;
            double sy = src[i][1] - srcCy;
            double dx = dst[i][0] - dstCx;
            double dy = dst[i][1] - dstCy;
            srcNorm += sx * sx + sy * sy;
            a += sx * dx + sy * dy;
            b += sx * dy - sy * dx;
        }
        if (srcNorm < 1e-10) throw new IllegalArgumentException("Invalid face landmarks for alignment");

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

    /** Inverts a 2x3 affine matrix without unavailable Android OpenCV APIs. */
    private static Mat invertAffineTransform(Mat affine) {
        double[] values = new double[6];
        affine.get(0, 0, values);
        double a = values[0], b = values[1], c = values[2], d = values[3], e = values[4], f = values[5];
        double det = a * e - b * d;
        if (Math.abs(det) < 1e-12) throw new IllegalArgumentException("Face affine transform is not invertible");
        Mat inverse = new Mat(2, 3, CvType.CV_64FC1);
        inverse.put(0, 0, new double[]{
            e / det, -b / det, (b * f - e * c) / det,
            -d / det, a / det, (d * c - a * f) / det
        });
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
