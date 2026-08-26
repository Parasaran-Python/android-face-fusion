package com.pv.androidfacefusion;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;

/**
 * Alignment and paste-back helpers for 256px face swappers such as HyperSwap.
 * Uses the 256px landmark geometry used by a known working HyperSwap CPU implementation.
 */
public final class SwapperImageUtils {
    private SwapperImageUtils() {}

    // HyperSwap 256 reference points from the working ReActor implementation.
    // Stored normalized so alignment/paste-back stay consistent if size changes.
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

        float[][] src = unpackLandmarks(landmarks);
        float[][] dst = scaledTemplate(targetSize);
        Matrix transform = estimateSimilarityTransform(src, dst);

        Bitmap aligned = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(aligned);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG | Paint.DITHER_FLAG);
        canvas.drawBitmap(image, transform, paint);
        return aligned;
    }

    public static Bitmap blendFace(Bitmap targetImage, Bitmap swappedFace, float[] landmarks, int faceSize) {
        if (landmarks == null || landmarks.length < 10) {
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }

        float[][] src = unpackLandmarks(landmarks);
        float[][] dst = scaledTemplate(faceSize);
        Matrix transform = estimateSimilarityTransform(src, dst);
        Matrix inverse = new Matrix();
        if (!transform.invert(inverse)) {
            return targetImage.copy(Bitmap.Config.ARGB_8888, true);
        }

        int width = targetImage.getWidth();
        int height = targetImage.getHeight();
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG | Paint.DITHER_FLAG);

        Bitmap warpedFace = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        new Canvas(warpedFace).drawBitmap(swappedFace, inverse, paint);

        Bitmap cropMask = createFeatherMask(faceSize);
        Bitmap warpedMask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        new Canvas(warpedMask).drawBitmap(cropMask, inverse, paint);

        int pixelCount = width * height;
        int[] targetPixels = new int[pixelCount];
        int[] facePixels = new int[pixelCount];
        int[] maskPixels = new int[pixelCount];
        int[] resultPixels = new int[pixelCount];

        targetImage.getPixels(targetPixels, 0, width, 0, 0, width, height);
        warpedFace.getPixels(facePixels, 0, width, 0, 0, width, height);
        warpedMask.getPixels(maskPixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixelCount; i++) {
            float alpha = ((maskPixels[i] >>> 24) & 0xFF) / 255.0f;
            if (alpha <= 0.001f) {
                resultPixels[i] = targetPixels[i];
                continue;
            }

            int tr = (targetPixels[i] >> 16) & 0xFF;
            int tg = (targetPixels[i] >> 8) & 0xFF;
            int tb = targetPixels[i] & 0xFF;
            int fr = (facePixels[i] >> 16) & 0xFF;
            int fg = (facePixels[i] >> 8) & 0xFF;
            int fb = facePixels[i] & 0xFF;

            int r = clamp(Math.round(fr * alpha + tr * (1.0f - alpha)));
            int g = clamp(Math.round(fg * alpha + tg * (1.0f - alpha)));
            int b = clamp(Math.round(fb * alpha + tb * (1.0f - alpha)));
            resultPixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(resultPixels, 0, width, 0, 0, width, height);

        cropMask.recycle();
        warpedMask.recycle();
        warpedFace.recycle();
        return result;
    }

    private static Bitmap createFeatherMask(int size) {
        Bitmap mask = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[size * size];
        float hardInset = size * 0.08f;
        float feather = Math.max(8.0f, size * 0.12f);

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                float edge = Math.min(Math.min(x, size - 1 - x), Math.min(y, size - 1 - y));
                float alpha = (edge - hardInset) / feather;
                alpha = Math.max(0.0f, Math.min(1.0f, alpha));
                alpha = alpha * alpha * (3.0f - 2.0f * alpha);
                int a = clamp(Math.round(alpha * 255.0f));
                pixels[y * size + x] = (a << 24) | 0x00FFFFFF;
            }
        }
        mask.setPixels(pixels, 0, size, 0, 0, size, size);
        return mask;
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

    private static Matrix estimateSimilarityTransform(float[][] src, float[][] dst) {
        int n = src.length;
        float srcCx = 0f, srcCy = 0f, dstCx = 0f, dstCy = 0f;
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

        float srcNorm = 0f;
        float a = 0f;
        float b = 0f;
        for (int i = 0; i < n; i++) {
            float sx = src[i][0] - srcCx;
            float sy = src[i][1] - srcCy;
            float dx = dst[i][0] - dstCx;
            float dy = dst[i][1] - dstCy;
            srcNorm += sx * sx + sy * sy;
            a += sx * dx + sy * dy;
            b += sx * dy - sy * dx;
        }

        if (srcNorm < 1e-10f) {
            return new Matrix();
        }

        float m00 = a / srcNorm;
        float m01 = -b / srcNorm;
        float m10 = b / srcNorm;
        float m11 = a / srcNorm;
        float tx = dstCx - (m00 * srcCx + m01 * srcCy);
        float ty = dstCy - (m10 * srcCx + m11 * srcCy);

        Matrix matrix = new Matrix();
        matrix.setValues(new float[]{m00, m01, tx, m10, m11, ty, 0f, 0f, 1f});
        return matrix;
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
