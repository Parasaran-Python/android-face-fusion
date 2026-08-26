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

/** Reprojects already aligned face crops between FaceFusion canonical templates. */
public final class CanonicalWarp {
    private static volatile boolean openCvReady;

    private static final float[][] HYPERSWAP_ARCFACE_128 = {
        {0.36167656f, 0.40387734f},
        {0.63696719f, 0.40235469f},
        {0.50019687f, 0.56044219f},
        {0.38710391f, 0.72160547f},
        {0.61507734f, 0.72034453f}
    };

    private static final float[][] SIMSWAP_ARCFACE_112_V1 = {
        {0.35473214f, 0.45658929f},
        {0.64526786f, 0.45658929f},
        {0.50000000f, 0.61154464f},
        {0.37913393f, 0.77687500f},
        {0.62086607f, 0.77687500f}
    };

    private CanonicalWarp() {}

    public static Bitmap hyperSwapToSimSwap(Bitmap hyperSwapAligned, int simSwapSize) {
        return warpBetweenTemplates(hyperSwapAligned, HYPERSWAP_ARCFACE_128,
            SIMSWAP_ARCFACE_112_V1, simSwapSize);
    }

    public static Bitmap simSwapToHyperSwap(Bitmap simSwapAligned, int hyperSwapSize) {
        return warpBetweenTemplates(simSwapAligned, SIMSWAP_ARCFACE_112_V1,
            HYPERSWAP_ARCFACE_128, hyperSwapSize);
    }

    private static Bitmap warpBetweenTemplates(Bitmap sourceBitmap, float[][] sourceTemplate,
                                                float[][] targetTemplate, int outputSize) {
        ensureOpenCv();
        float[][] sourcePoints = scaledTemplate(sourceTemplate, sourceBitmap.getWidth(), sourceBitmap.getHeight());
        float[][] targetPoints = scaledTemplate(targetTemplate, outputSize, outputSize);
        Mat affine = estimateSimilarityTransform(sourcePoints, targetPoints);
        Mat source = new Mat();
        Mat output = new Mat();
        try {
            Utils.bitmapToMat(sourceBitmap, source);
            Imgproc.warpAffine(source, output, affine, new Size(outputSize, outputSize),
                Imgproc.INTER_CUBIC, Core.BORDER_REFLECT, new Scalar(0, 0, 0, 255));
            Bitmap result = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(output, result);
            return result;
        } finally {
            source.release();
            output.release();
            affine.release();
        }
    }

    private static float[][] scaledTemplate(float[][] template, int width, int height) {
        float[][] result = new float[template.length][2];
        for (int i = 0; i < template.length; i++) {
            result[i][0] = template[i][0] * width;
            result[i][1] = template[i][1] * height;
        }
        return result;
    }

    /** Direct least-squares similarity transform; avoids unsupported Android OpenCV calib3d APIs. */
    private static Mat estimateSimilarityTransform(float[][] src, float[][] dst) {
        int n = Math.min(src.length, dst.length);
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
        if (srcNorm < 1e-10) throw new IllegalArgumentException("Invalid canonical face template");

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

    private static void ensureOpenCv() {
        if (openCvReady) return;
        synchronized (CanonicalWarp.class) {
            if (openCvReady) return;
            if (!OpenCVLoader.initLocal()) {
                throw new IllegalStateException("OpenCV failed to initialize for canonical face warp");
            }
            openCvReady = true;
        }
    }
}
