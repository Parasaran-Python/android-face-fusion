package com.pv.androidfacefusion;

/** Lightweight landmark transform helpers using the same HyperSwap canonical template as paste-back. */
public final class LandmarkWarp {
    private static final float[][] HYPERSWAP_ARCFACE_128 = {
        {0.36167656f, 0.40387734f},
        {0.63696719f, 0.40235469f},
        {0.50019687f, 0.56044219f},
        {0.38710391f, 0.72160547f},
        {0.61507734f, 0.72034453f}
    };

    private LandmarkWarp() {}

    public static float[] toHyperSwapCanonical(float[] landmarks5, float[] landmarks68, int size) {
        if (landmarks5 == null || landmarks5.length < 10 || landmarks68 == null || landmarks68.length < 136) {
            return null;
        }
        float[][] src = new float[5][2];
        float[][] dst = new float[5][2];
        for (int i = 0; i < 5; i++) {
            src[i][0] = landmarks5[i * 2];
            src[i][1] = landmarks5[i * 2 + 1];
            dst[i][0] = HYPERSWAP_ARCFACE_128[i][0] * size;
            dst[i][1] = HYPERSWAP_ARCFACE_128[i][1] * size;
        }
        double[] m = estimateSimilarity(src, dst);
        float[] result = new float[136];
        for (int i = 0; i < 68; i++) {
            double x = landmarks68[i * 2];
            double y = landmarks68[i * 2 + 1];
            result[i * 2] = (float) (m[0] * x + m[1] * y + m[2]);
            result[i * 2 + 1] = (float) (m[3] * x + m[4] * y + m[5]);
        }
        return result;
    }

    private static double[] estimateSimilarity(float[][] src, float[][] dst) {
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
        if (srcNorm < 1e-10) throw new IllegalArgumentException("Invalid face landmarks");

        double m00 = a / srcNorm;
        double m01 = -b / srcNorm;
        double m10 = b / srcNorm;
        double m11 = a / srcNorm;
        double tx = dstCx - (m00 * srcCx + m01 * srcCy);
        double ty = dstCy - (m10 * srcCx + m11 * srcCy);
        return new double[]{m00, m01, tx, m10, m11, ty};
    }
}
