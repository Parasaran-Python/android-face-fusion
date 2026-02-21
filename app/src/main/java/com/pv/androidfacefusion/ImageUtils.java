package com.pv.androidfacefusion;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;

/**
 * Utility class for image processing operations
 * Implements proper InsightFace-style alignment
 */
public class ImageUtils {

    // ArcFace 112x112 reference landmarks (normalized coordinates)
    private static final float[][] ARCFACE_SRC = {
        {38.2946f, 51.6963f},  // left eye
        {73.5318f, 51.5014f},  // right eye
        {56.0252f, 71.7366f},  // nose
        {41.5493f, 92.3655f},  // left mouth
        {70.7299f, 92.2041f}   // right mouth
    };
    
    // Standard 128x128 reference landmarks for inswapper
    // Python: ratio = 128/128 = 1.0, diff_x = 8.0 → arcface_dst unchanged, X += 8.0
    private static final float[][] FFHQ_SRC_128 = {
        {38.2946f + 8.0f, 51.6963f},  // left eye
        {73.5318f + 8.0f, 51.5014f},  // right eye
        {56.0252f + 8.0f, 71.7366f},  // nose
        {41.5493f + 8.0f, 92.3655f},  // left mouth
        {70.7299f + 8.0f, 92.2041f}   // right mouth
    };

    /**
     * Align face based on landmarks using similarity transformation
     * This matches InsightFace's alignment methodology
     */
    public static Bitmap alignFace(Bitmap image, float[] landmarks, int targetSize) {
        if (landmarks == null || landmarks.length < 10) {
            return Bitmap.createScaledBitmap(image, targetSize, targetSize, true);
        }

        // Select appropriate reference landmarks based on target size
        float[][] refLandmarks = (targetSize == 112) ? ARCFACE_SRC : FFHQ_SRC_128;
        
        // Convert flat array to 2D points (5 landmarks x 2 coordinates)
        float[][] srcPoints = new float[5][2];
        for (int i = 0; i < 5; i++) {
            srcPoints[i][0] = landmarks[i * 2];     // x
            srcPoints[i][1] = landmarks[i * 2 + 1]; // y
        }
        
        // Estimate similarity transform (rotation, scale, translation)
        Matrix transformMatrix = estimateSimilarityTransform(srcPoints, refLandmarks);
        
        // Apply transformation
        Bitmap aligned = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(aligned);
        canvas.drawBitmap(image, transformMatrix, new Paint(Paint.FILTER_BITMAP_FLAG));
        
        return aligned;
    }
    
    /**
     * Estimate similarity transformation matrix from source to destination landmarks.
     * Uses the closed-form Procrustes solution for 2D similarity transforms,
     * mathematically equivalent to skimage.transform.SimilarityTransform.estimate().
     *
     * For centered point sets, the optimal similarity transform (rotation + uniform scale)
     * that minimizes sum of squared errors has matrix elements:
     *   M = [[a/σ², -b/σ²], [b/σ², a/σ²]] + translation
     * where:
     *   a = Σ(src_x·dst_x + src_y·dst_y)  (correlation)
     *   b = Σ(src_x·dst_y - src_y·dst_x)  (cross-correlation)
     *   σ² = Σ(src_x² + src_y²)            (source variance)
     */
    private static Matrix estimateSimilarityTransform(float[][] src, float[][] dst) {
        int n = src.length;

        // Compute centroids
        float srcCenterX = 0, srcCenterY = 0;
        float dstCenterX = 0, dstCenterY = 0;

        for (int i = 0; i < n; i++) {
            srcCenterX += src[i][0];
            srcCenterY += src[i][1];
            dstCenterX += dst[i][0];
            dstCenterY += dst[i][1];
        }

        srcCenterX /= n;
        srcCenterY /= n;
        dstCenterX /= n;
        dstCenterY /= n;

        // Center the points
        float[][] srcCentered = new float[n][2];
        float[][] dstCentered = new float[n][2];

        for (int i = 0; i < n; i++) {
            srcCentered[i][0] = src[i][0] - srcCenterX;
            srcCentered[i][1] = src[i][1] - srcCenterY;
            dstCentered[i][0] = dst[i][0] - dstCenterX;
            dstCentered[i][1] = dst[i][1] - dstCenterY;
        }

        // Compute source variance: σ² = Σ(x² + y²)
        float srcNorm = 0;
        for (int i = 0; i < n; i++) {
            srcNorm += srcCentered[i][0] * srcCentered[i][0] + srcCentered[i][1] * srcCentered[i][1];
        }

        // Avoid division by zero
        if (srcNorm < 1e-10f) {
            android.util.Log.e("ImageUtils", "Invalid landmarks: zero source variance");
            return new Matrix(); // Return identity matrix
        }

        // Compute cross-correlation terms
        // a = Σ(src_x·dst_x + src_y·dst_y), b = Σ(src_x·dst_y - src_y·dst_x)
        float a = 0, b = 0;
        for (int i = 0; i < n; i++) {
            a += srcCentered[i][0] * dstCentered[i][0] + srcCentered[i][1] * dstCentered[i][1];
            b += srcCentered[i][0] * dstCentered[i][1] - srcCentered[i][1] * dstCentered[i][0];
        }

        // Procrustes solution: matrix elements = correlation / source_variance
        // This is the optimal least-squares similarity transform
        float a_mat = a / srcNorm;
        float b_mat = -b / srcNorm;
        float d_mat = b / srcNorm;
        float e_mat = a / srcNorm;

        // Translation: t = dst_center - M * src_center
        float c_mat = dstCenterX - (a_mat * srcCenterX + b_mat * srcCenterY);
        float f_mat = dstCenterY - (d_mat * srcCenterX + e_mat * srcCenterY);

        // Check for NaN or Infinity
        if (Float.isNaN(a_mat) || Float.isInfinite(a_mat) ||
            Float.isNaN(b_mat) || Float.isInfinite(b_mat) ||
            Float.isNaN(c_mat) || Float.isInfinite(c_mat) ||
            Float.isNaN(d_mat) || Float.isInfinite(d_mat) ||
            Float.isNaN(e_mat) || Float.isInfinite(e_mat) ||
            Float.isNaN(f_mat) || Float.isInfinite(f_mat)) {
            android.util.Log.e("ImageUtils", "Invalid matrix values: NaN or Infinity");
            return new Matrix(); // Return identity matrix
        }

        Matrix matrix = new Matrix();
        float[] values = {a_mat, b_mat, c_mat, d_mat, e_mat, f_mat, 0, 0, 1};
        matrix.setValues(values);

        return matrix;
    }

    /**
     * Crop face from image using bounding box
     */
    public static Bitmap cropFace(Bitmap image, RectF bbox) {
        int x = Math.max(0, (int) bbox.left);
        int y = Math.max(0, (int) bbox.top);
        int width = Math.min((int) bbox.width(), image.getWidth() - x);
        int height = Math.min((int) bbox.height(), image.getHeight() - y);

        if (width <= 0 || height <= 0) {
            return null;
        }

        return Bitmap.createBitmap(image, x, y, width, height);
    }

    /**
     * Blend faces using transformation matrix with advanced masking
     * Matches Python INSwapper paste-back logic with differential blending
     */
    public static Bitmap blendFaces(Bitmap targetImage, Bitmap swappedFace, float[] landmarks, int faceSize) {
        try {
            // Convert landmarks to 2D array
            float[][] srcPoints = new float[5][2];
            for (int i = 0; i < 5; i++) {
                srcPoints[i][0] = landmarks[i * 2];
                srcPoints[i][1] = landmarks[i * 2 + 1];
            }

            // Use appropriate reference based on face size
            float[][] refLandmarks = (faceSize == 112) ? ARCFACE_SRC : FFHQ_SRC_128;

            // Get transformation matrix
            Matrix transformMatrix = estimateSimilarityTransform(srcPoints, refLandmarks);

            // Invert to paste back
            Matrix inverseMatrix = new Matrix();
            if (transformMatrix.invert(inverseMatrix)) {
                // Advanced blending matching Python implementation
                return advancedBlend(targetImage, swappedFace, inverseMatrix, faceSize);
            } else {
                android.util.Log.e("ImageUtils", "Failed to invert transformation matrix, using simple blend");
                return simpleBlend(targetImage, swappedFace, landmarks);
            }
        } catch (Exception e) {
            android.util.Log.e("ImageUtils", "Error in blendFaces", e);
            return simpleBlend(targetImage, swappedFace, landmarks);
        }
    }
    
    /**
     * Advanced blending with masking, erosion, and Gaussian blur.
     * Matches Python INSwapper's paste-back logic:
     *   1. Warp swapped face back to original coordinates
     *   2. Create white mask, warp to original coordinates
     *   3. Threshold mask at 20 to get binary mask
     *   4. Compute mask_size from mask area
     *   5. Erode mask with kernel max(mask_size//10, 10)
     *   6. Gaussian blur mask with kernel max(mask_size//20, 5)*2+1
     *   7. Blend: result = mask * face + (1 - mask) * target
     */
    private static Bitmap advancedBlend(Bitmap targetImage, Bitmap swappedFace, Matrix inverseMatrix, int faceSize) {
        int targetWidth = targetImage.getWidth();
        int targetHeight = targetImage.getHeight();

        // Create white mask for the face region
        Bitmap maskBitmap = Bitmap.createBitmap(faceSize, faceSize, Bitmap.Config.ARGB_8888);
        Canvas maskCanvas = new Canvas(maskBitmap);
        maskCanvas.drawColor(0xFFFFFFFF); // White mask

        // Warp the swapped face back to original position
        Bitmap warpedFace = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
        Canvas warpCanvas = new Canvas(warpedFace);
        Paint paint = new Paint();
        paint.setAntiAlias(true);
        paint.setFilterBitmap(true);
        paint.setDither(true);
        warpCanvas.drawBitmap(swappedFace, inverseMatrix, paint);

        // Warp the mask
        Bitmap warpedMask = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
        Canvas warpMaskCanvas = new Canvas(warpedMask);
        warpMaskCanvas.drawBitmap(maskBitmap, inverseMatrix, paint);

        // Apply blending with the mask
        Bitmap blended = applyMaskBlending(targetImage, warpedFace, warpedMask);

        // Clean up intermediate bitmaps
        maskBitmap.recycle();
        warpedFace.recycle();
        warpedMask.recycle();

        return blended;
    }

    /**
     * Apply mask-based blending matching Python INSwapper paste-back.
     * Uses dynamic kernel sizes based on mask area, matching:
     *   k_erode = max(mask_size//10, 10)
     *   k_blur  = max(mask_size//20, 5)*2+1
     */
    private static Bitmap applyMaskBlending(Bitmap target, Bitmap face, Bitmap mask) {
        int width = target.getWidth();
        int height = target.getHeight();

        int[] targetPixels = new int[width * height];
        int[] facePixels = new int[width * height];
        int[] maskPixels = new int[width * height];

        target.getPixels(targetPixels, 0, width, 0, 0, width, height);
        face.getPixels(facePixels, 0, width, 0, 0, width, height);
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height);

        // Extract mask as float array and threshold at 20 (matching Python: img_white[img_white>20] = 255)
        float[] maskFloat = new float[width * height];
        int minMaskX = width, maxMaskX = 0, minMaskY = height, maxMaskY = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int val = (maskPixels[idx] >> 16) & 0xFF;
                if (val > 20) {
                    maskFloat[idx] = 255.0f;
                    minMaskX = Math.min(minMaskX, x);
                    maxMaskX = Math.max(maxMaskX, x);
                    minMaskY = Math.min(minMaskY, y);
                    maxMaskY = Math.max(maxMaskY, y);
                } else {
                    maskFloat[idx] = 0.0f;
                }
            }
        }

        // Compute mask_size from bounding box of white region (matching Python)
        int maskH = maxMaskY - minMaskY;
        int maskW = maxMaskX - minMaskX;
        int maskSize = (int) Math.sqrt(maskH * maskW);

        if (maskSize < 10) {
            // Mask too small, return target unchanged
            return target.copy(Bitmap.Config.ARGB_8888, true);
        }

        // Dynamic erosion kernel: k = max(mask_size//10, 10) (matching Python)
        int erodeK = Math.max(maskSize / 10, 10);
        maskFloat = erodeMaskFloat(maskFloat, width, height, erodeK,
                                   minMaskX, minMaskY, maxMaskX, maxMaskY);

        // Dynamic Gaussian blur kernel: k = max(mask_size//20, 5), blur_size = 2*k+1
        int blurK = Math.max(maskSize / 20, 5);
        int blurSize = 2 * blurK + 1;
        // OpenCV sigma formula when sigma=0: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        float sigma = 0.3f * ((blurSize - 1) * 0.5f - 1) + 0.8f;
        maskFloat = gaussianBlurFloat(maskFloat, width, height, blurSize, sigma,
                                      minMaskX, minMaskY, maxMaskX, maxMaskY, erodeK + blurSize);

        // Normalize mask to [0, 1]
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] resultPixels = new int[width * height];

        for (int i = 0; i < width * height; i++) {
            float maskAlpha = maskFloat[i] / 255.0f;

            if (maskAlpha < 0.01f) {
                resultPixels[i] = targetPixels[i];
                continue;
            }

            int targetR = (targetPixels[i] >> 16) & 0xFF;
            int targetG = (targetPixels[i] >> 8) & 0xFF;
            int targetB = targetPixels[i] & 0xFF;

            int faceR = (facePixels[i] >> 16) & 0xFF;
            int faceG = (facePixels[i] >> 8) & 0xFF;
            int faceB = facePixels[i] & 0xFF;

            // Blend: result = mask * face + (1 - mask) * target
            int resultR = Math.min(255, Math.max(0, (int) (maskAlpha * faceR + (1 - maskAlpha) * targetR)));
            int resultG = Math.min(255, Math.max(0, (int) (maskAlpha * faceG + (1 - maskAlpha) * targetG)));
            int resultB = Math.min(255, Math.max(0, (int) (maskAlpha * faceB + (1 - maskAlpha) * targetB)));

            resultPixels[i] = 0xFF000000 | (resultR << 16) | (resultG << 8) | resultB;
        }

        result.setPixels(resultPixels, 0, width, 0, 0, width, height);
        return result;
    }

    /**
     * Erode mask using float array. Only processes the mask bounding box region for efficiency.
     * Matches Python: cv2.erode(img_mask, np.ones((k,k)), iterations=1)
     */
    private static float[] erodeMaskFloat(float[] mask, int width, int height, int kernelSize,
                                          int minX, int minY, int maxX, int maxY) {
        float[] result = mask.clone();
        int halfK = kernelSize / 2;

        // Expand processing region by kernel size
        int startY = Math.max(0, minY - halfK);
        int endY = Math.min(height, maxY + halfK + 1);
        int startX = Math.max(0, minX - halfK);
        int endX = Math.min(width, maxX + halfK + 1);

        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                float minVal = 255.0f;
                for (int ky = -halfK; ky <= halfK; ky++) {
                    int ny = y + ky;
                    if (ny < 0 || ny >= height) { minVal = 0; continue; }
                    for (int kx = -halfK; kx <= halfK; kx++) {
                        int nx = x + kx;
                        if (nx < 0 || nx >= width) { minVal = 0; continue; }
                        minVal = Math.min(minVal, mask[ny * width + nx]);
                    }
                }
                result[y * width + x] = minVal;
            }
        }
        return result;
    }

    /**
     * Gaussian blur on float mask. Only processes the mask region + margin for efficiency.
     * Matches Python: cv2.GaussianBlur(img_mask, blur_size, 0)
     */
    private static float[] gaussianBlurFloat(float[] mask, int width, int height,
                                             int kernelSize, float sigma,
                                             int minX, int minY, int maxX, int maxY, int margin) {
        float[] kernel1D = createGaussianKernel1D(kernelSize, sigma);
        int halfK = kernelSize / 2;

        // Expand processing region
        int startY = Math.max(0, minY - margin);
        int endY = Math.min(height, maxY + margin + 1);
        int startX = Math.max(0, minX - margin);
        int endX = Math.min(width, maxX + margin + 1);

        // Separable Gaussian blur: horizontal pass then vertical pass
        float[] temp = mask.clone();

        // Horizontal pass
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                float sum = 0, wsum = 0;
                for (int kx = -halfK; kx <= halfK; kx++) {
                    int nx = x + kx;
                    if (nx >= 0 && nx < width) {
                        float w = kernel1D[kx + halfK];
                        sum += mask[y * width + nx] * w;
                        wsum += w;
                    }
                }
                temp[y * width + x] = sum / wsum;
            }
        }

        // Vertical pass
        float[] result = temp.clone();
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                float sum = 0, wsum = 0;
                for (int ky = -halfK; ky <= halfK; ky++) {
                    int ny = y + ky;
                    if (ny >= 0 && ny < height) {
                        float w = kernel1D[ky + halfK];
                        sum += temp[ny * width + x] * w;
                        wsum += w;
                    }
                }
                result[y * width + x] = sum / wsum;
            }
        }
        return result;
    }

    /**
     * Create 1D Gaussian kernel (for separable convolution - much faster than 2D)
     */
    private static float[] createGaussianKernel1D(int size, float sigma) {
        float[] kernel = new float[size];
        int halfSize = size / 2;
        float sum = 0;
        for (int x = -halfSize; x <= halfSize; x++) {
            float value = (float) Math.exp(-(x * x) / (2 * sigma * sigma));
            kernel[x + halfSize] = value;
            sum += value;
        }
        for (int i = 0; i < size; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }
    
    /**
     * Fallback simple blending method
     */
    private static Bitmap simpleBlend(Bitmap targetImage, Bitmap swappedFace, float[] landmarks) {
        Bitmap result = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        
        // Calculate approximate position from landmarks
        float centerX = (landmarks[0] + landmarks[2]) / 2;  // Average of left and right eye
        float centerY = (landmarks[1] + landmarks[3]) / 2;
        
        // Simple scale and position
        float scale = Math.abs(landmarks[2] - landmarks[0]) / 40f; // Rough estimate
        int scaledSize = (int) (swappedFace.getWidth() * scale);
        
        if (scaledSize > 0 && scaledSize < targetImage.getWidth() && scaledSize < targetImage.getHeight()) {
            Bitmap scaled = Bitmap.createScaledBitmap(swappedFace, scaledSize, scaledSize, true);
            Canvas canvas = new Canvas(result);
            Paint paint = new Paint();
            paint.setAntiAlias(true);
            paint.setFilterBitmap(true);
            
            float left = centerX - scaledSize / 2f;
            float top = centerY - scaledSize / 2f;
            canvas.drawBitmap(scaled, left, top, paint);
            scaled.recycle();
        }
        
        return result;
    }

    /**
     * Resize image while maintaining aspect ratio
     */
    public static Bitmap resizeImage(Bitmap image, int maxSize) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        float ratio = Math.min(
            (float) maxSize / width,
            (float) maxSize / height
        );
        
        int newWidth = Math.round(width * ratio);
        int newHeight = Math.round(height * ratio);
        
        return Bitmap.createScaledBitmap(image, newWidth, newHeight, true);
    }
}
