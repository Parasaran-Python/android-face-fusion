package com.pv.androidfacefusion;

import android.graphics.Bitmap;

/** Small image/mask adjustments that separate source identity strength from naturalness processing. */
public final class IdentityStrengthUtils {
    private IdentityStrengthUtils() {}

    /**
     * Raises face-region authority without destroying the deliberately low eye/mouth weights.
     * alpha' = 1 - (1-alpha)^k gives smooth monotonic strengthening and keeps zero/one exact.
     */
    public static float[] strengthenMask(float[] source, IdentityStrengthSettings.Level level) {
        if (source == null || level == null || level == IdentityStrengthSettings.Level.NATURAL) return source;
        float[] result = new float[source.length];
        float k = level.maskExponent;
        for (int i = 0; i < source.length; i++) {
            float a = Math.max(0.0f, Math.min(1.0f, source[i]));
            result[i] = 1.0f - (float) Math.pow(1.0f - a, k);
        }
        return result;
    }

    /**
     * The compositor uses the aligned target as its colour/lighting/texture reference. Blend that
     * reference toward the generated face as identity strength rises, so naturalisation cannot
     * quietly turn a decisive swap back into a gentle target/source merge.
     */
    public static Bitmap createNaturalisationReference(Bitmap alignedTarget, Bitmap swapped,
                                                        IdentityStrengthSettings.Level level) {
        if (alignedTarget == null || swapped == null || level == null
                || level == IdentityStrengthSettings.Level.NATURAL) {
            return alignedTarget;
        }
        int width = swapped.getWidth();
        int height = swapped.getHeight();
        Bitmap target = alignedTarget;
        if (alignedTarget.getWidth() != width || alignedTarget.getHeight() != height) {
            target = Bitmap.createScaledBitmap(alignedTarget, width, height, true);
        }
        int count = width * height;
        int[] targetPixels = new int[count];
        int[] swapPixels = new int[count];
        int[] output = new int[count];
        target.getPixels(targetPixels, 0, width, 0, 0, width, height);
        swapped.getPixels(swapPixels, 0, width, 0, 0, width, height);

        float tw = level.targetReferenceWeight;
        float sw = 1.0f - tw;
        for (int i = 0; i < count; i++) {
            int tp = targetPixels[i];
            int sp = swapPixels[i];
            int r = Math.round(((tp >> 16) & 0xFF) * tw + ((sp >> 16) & 0xFF) * sw);
            int g = Math.round(((tp >> 8) & 0xFF) * tw + ((sp >> 8) & 0xFF) * sw);
            int b = Math.round((tp & 0xFF) * tw + (sp & 0xFF) * sw);
            output[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        result.setPixels(output, 0, width, 0, 0, width, height);
        if (target != alignedTarget && !target.isRecycled()) target.recycle();
        return result;
    }
}
