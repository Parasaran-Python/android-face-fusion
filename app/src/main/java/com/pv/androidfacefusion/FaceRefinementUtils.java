package com.pv.androidfacefusion;

import android.graphics.Bitmap;

/** Conservative post-swap local deformation for brows, eyes and mouth on expressive targets. */
public final class FaceRefinementUtils {
    private static final int[] CONTROL_POINTS = {
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 30, 31, 33, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67
    };

    private FaceRefinementUtils() {}

    /**
     * Nudges generated expressive geometry toward the target's real 68-point structure while
     * leaving the rest of the swapped identity untouched. Maximum deformation is deliberately
     * capped so a noisy landmark result cannot bend the whole face.
     */
    public static Bitmap correctExpression(Bitmap swapped, float[] generated68, float[] target68) {
        if (swapped == null || generated68 == null || target68 == null
                || generated68.length < 136 || target68.length < 136) {
            return swapped;
        }

        int width = swapped.getWidth();
        int height = swapped.getHeight();
        float scale = Math.min(width, height) / 768.0f;
        float maxControlShift = 26.0f * scale;
        float maxCombinedShift = 20.0f * scale;

        float[] dx = new float[CONTROL_POINTS.length];
        float[] dy = new float[CONTROL_POINTS.length];
        boolean[] valid = new boolean[CONTROL_POINTS.length];
        int validCount = 0;
        double totalShiftSq = 0.0;
        float minX = width, minY = height, maxX = 0, maxY = 0;

        for (int i = 0; i < CONTROL_POINTS.length; i++) {
            int point = CONTROL_POINTS[i];
            float gx = generated68[point * 2];
            float gy = generated68[point * 2 + 1];
            float tx = target68[point * 2];
            float ty = target68[point * 2 + 1];
            if (!finite(gx) || !finite(gy) || !finite(tx) || !finite(ty)) continue;
            if (gx < -width * 0.1f || gx > width * 1.1f || gy < -height * 0.1f || gy > height * 1.1f
                    || tx < -width * 0.1f || tx > width * 1.1f || ty < -height * 0.1f || ty > height * 1.1f) {
                continue;
            }

            float ddx = tx - gx;
            float ddy = ty - gy;
            float magnitude = (float) Math.sqrt(ddx * ddx + ddy * ddy);
            if (magnitude > maxControlShift && magnitude > 1e-4f) {
                float factor = maxControlShift / magnitude;
                ddx *= factor;
                ddy *= factor;
                magnitude = maxControlShift;
            }
            dx[i] = ddx;
            dy[i] = ddy;
            valid[i] = true;
            validCount++;
            totalShiftSq += magnitude * magnitude;

            float radius = radiusForPoint(point, scale);
            minX = Math.min(minX, tx - radius);
            maxX = Math.max(maxX, tx + radius);
            minY = Math.min(minY, ty - radius);
            maxY = Math.max(maxY, ty + radius);
        }

        if (validCount < 8) return swapped;
        float rmsShift = (float) Math.sqrt(totalShiftSq / validCount);
        if (rmsShift < 1.15f * scale) return swapped;

        int x0 = Math.max(0, (int) Math.floor(minX));
        int x1 = Math.min(width - 1, (int) Math.ceil(maxX));
        int y0 = Math.max(0, (int) Math.floor(minY));
        int y1 = Math.min(height - 1, (int) Math.ceil(maxY));
        if (x1 <= x0 || y1 <= y0) return swapped;

        int[] source = new int[width * height];
        int[] output = new int[source.length];
        swapped.getPixels(source, 0, width, 0, 0, width, height);
        System.arraycopy(source, 0, output, 0, source.length);

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                float sumW = 0.0f;
                float shiftX = 0.0f;
                float shiftY = 0.0f;

                for (int i = 0; i < CONTROL_POINTS.length; i++) {
                    if (!valid[i]) continue;
                    int point = CONTROL_POINTS[i];
                    float tx = target68[point * 2];
                    float ty = target68[point * 2 + 1];
                    float radius = radiusForPoint(point, scale);
                    float rx = x - tx;
                    float ry = y - ty;
                    float d2 = rx * rx + ry * ry;
                    float r2 = radius * radius;
                    if (d2 >= r2) continue;
                    float w = 1.0f - d2 / r2;
                    w *= w;
                    sumW += w;
                    shiftX += dx[i] * w;
                    shiftY += dy[i] * w;
                }

                if (sumW <= 1e-5f) continue;
                shiftX = (shiftX / sumW) * 0.72f;
                shiftY = (shiftY / sumW) * 0.72f;
                float magnitude = (float) Math.sqrt(shiftX * shiftX + shiftY * shiftY);
                if (magnitude > maxCombinedShift && magnitude > 1e-4f) {
                    float factor = maxCombinedShift / magnitude;
                    shiftX *= factor;
                    shiftY *= factor;
                }

                float sourceX = x - shiftX;
                float sourceY = y - shiftY;
                output[y * width + x] = bilinearSample(source, width, height, sourceX, sourceY);
            }
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        return result;
    }

    private static float radiusForPoint(int point, float scale) {
        if (point >= 48) return 58.0f * scale; // mouth and lips
        if (point >= 36) return 44.0f * scale; // eyes
        if (point >= 17 && point <= 26) return 50.0f * scale; // brows
        return 38.0f * scale; // nose bridge / cheek expression support
    }

    private static int bilinearSample(int[] pixels, int width, int height, float x, float y) {
        x = Math.max(0.0f, Math.min(width - 1.001f, x));
        y = Math.max(0.0f, Math.min(height - 1.001f, y));
        int x0 = (int) Math.floor(x);
        int y0 = (int) Math.floor(y);
        int x1 = Math.min(width - 1, x0 + 1);
        int y1 = Math.min(height - 1, y0 + 1);
        float fx = x - x0;
        float fy = y - y0;

        int p00 = pixels[y0 * width + x0];
        int p10 = pixels[y0 * width + x1];
        int p01 = pixels[y1 * width + x0];
        int p11 = pixels[y1 * width + x1];

        int r = interpolateChannel((p00 >> 16) & 0xFF, (p10 >> 16) & 0xFF,
            (p01 >> 16) & 0xFF, (p11 >> 16) & 0xFF, fx, fy);
        int g = interpolateChannel((p00 >> 8) & 0xFF, (p10 >> 8) & 0xFF,
            (p01 >> 8) & 0xFF, (p11 >> 8) & 0xFF, fx, fy);
        int b = interpolateChannel(p00 & 0xFF, p10 & 0xFF, p01 & 0xFF, p11 & 0xFF, fx, fy);
        return 0xFF000000 | (r << 16) | (g << 8) | b;
    }

    private static int interpolateChannel(int a, int b, int c, int d, float fx, float fy) {
        float top = a + (b - a) * fx;
        float bottom = c + (d - c) * fx;
        return Math.max(0, Math.min(255, Math.round(top + (bottom - top) * fy)));
    }

    private static boolean finite(float value) {
        return !Float.isNaN(value) && !Float.isInfinite(value);
    }
}
