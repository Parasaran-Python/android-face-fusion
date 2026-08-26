package com.pv.androidfacefusion;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.List;
import java.util.Map;
import java.util.Set;

/** Coordinates detection, refined alignment, ArcFace identity extraction and HyperSwap swapping. */
public class FaceFusionProcessor {
    private static final String TAG = "FaceFusionProcessor";

    private final FaceDetector faceDetector;
    private final FaceEmbedder faceEmbedder;
    private final FaceSwapper faceSwapper;
    private final FaceLandmarker faceLandmarker;
    private final FaceParser faceParser;

    public FaceFusionProcessor(FaceDetector detector, FaceEmbedder embedder, FaceSwapper swapper) {
        this.faceDetector = detector;
        this.faceEmbedder = embedder;
        this.faceSwapper = swapper;

        FaceLandmarker landmarker = null;
        try {
            landmarker = new FaceLandmarker(swapper.getAppContext());
            landmarker.initialize();
        } catch (Exception e) {
            Log.w(TAG, "68-point landmark refinement unavailable; detector landmarks remain active", e);
            if (landmarker != null) landmarker.close();
            landmarker = null;
        }
        this.faceLandmarker = landmarker;

        FaceParser parser = null;
        try {
            parser = new FaceParser(swapper.getAppContext());
            parser.initialize();
        } catch (Exception e) {
            Log.w(TAG, "Semantic face parser unavailable; geometric blend fallback remains active", e);
            if (parser != null) parser.close();
            parser = null;
        }
        this.faceParser = parser;
    }

    public FaceFusionProcessor(FaceDetector detector, FaceEmbedder embedder, FaceSwapper swapper,
                               FaceLandmarker landmarker, FaceParser parser) {
        this.faceDetector = detector;
        this.faceEmbedder = embedder;
        this.faceSwapper = swapper;
        this.faceLandmarker = landmarker;
        this.faceParser = parser;
    }

    public Bitmap processFaceFusion(Bitmap sourceImage, Bitmap targetImage) throws Exception {
        return processFaceFusion(sourceImage, targetImage, 0);
    }

    public Bitmap processFaceFusion(Bitmap sourceImage, Bitmap targetImage, Set<Integer> selectedFaceIndices) throws Exception {
        if (selectedFaceIndices == null || selectedFaceIndices.isEmpty()) {
            return processFaceFusionMultiple(sourceImage, targetImage);
        }

        List<FaceDetector.Face> sourceFaces = faceDetector.detectFaces(sourceImage);
        if (sourceFaces.isEmpty()) {
            throw new Exception("No face detected in source image. Please use an image with a clear face.");
        }
        float[] sourceEmbedding = getSourceEmbedding(sourceImage, sourceFaces.get(0));

        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) {
            throw new Exception("No face detected in target image.");
        }

        Bitmap result = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        for (int i = 0; i < targetFaces.size(); i++) {
            if (!selectedFaceIndices.contains(i)) continue;
            Bitmap next = swapOne(result, targetFaces.get(i), sourceEmbedding);
            if (result != targetImage && !result.isRecycled()) result.recycle();
            result = next;
        }
        return result;
    }

    public Bitmap processFaceFusionWithMapping(Bitmap targetImage, Map<Integer, float[]> targetIndexToEmbeddingMap) throws Exception {
        if (targetIndexToEmbeddingMap == null || targetIndexToEmbeddingMap.isEmpty()) {
            throw new Exception("No target face mappings selected.");
        }

        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) {
            throw new Exception("No face detected in target image.");
        }

        Bitmap result = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        for (int i = 0; i < targetFaces.size(); i++) {
            float[] embedding = targetIndexToEmbeddingMap.get(i);
            if (embedding == null) continue;
            Bitmap next = swapOne(result, targetFaces.get(i), embedding);
            if (result != targetImage && !result.isRecycled()) result.recycle();
            result = next;
        }
        return result;
    }

    public Bitmap processFaceFusion(Bitmap sourceImage, Bitmap targetImage, int targetFaceIndex) throws Exception {
        if (targetFaceIndex == -1) {
            return processFaceFusionMultiple(sourceImage, targetImage);
        }

        Log.d(TAG, "Starting " + FaceSwapper.QUALITY_SIZE + " quality face fusion; native HyperSwap tile="
            + faceSwapper.getInputSize() + ", HyperSwap=" + faceSwapper.isUsingHyperSwap());

        List<FaceDetector.Face> sourceFaces = faceDetector.detectFaces(sourceImage);
        if (sourceFaces.isEmpty()) {
            throw new Exception("No face detected in source image. Please use an image with a clear face.");
        }

        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) {
            String suggestion = "";
            if (targetImage.getWidth() < 300 || targetImage.getHeight() < 300) {
                suggestion = " The target image is quite small (" + targetImage.getWidth() + "x"
                    + targetImage.getHeight() + "). Try using a larger image with a clearer face.";
            }
            throw new Exception("No face detected in target image." + suggestion);
        }

        if (targetFaceIndex < 0 || targetFaceIndex >= targetFaces.size()) targetFaceIndex = 0;

        float[] sourceEmbedding = getSourceEmbedding(sourceImage, sourceFaces.get(0));
        return swapOne(targetImage, targetFaces.get(targetFaceIndex), sourceEmbedding);
    }

    public List<FaceDetector.Face> detectTargetFaces(Bitmap targetImage) throws Exception {
        return faceDetector.detectFaces(targetImage);
    }

    public Bitmap processFaceFusionMultiple(Bitmap sourceImage, Bitmap targetImage) throws Exception {
        List<FaceDetector.Face> sourceFaces = faceDetector.detectFaces(sourceImage);
        if (sourceFaces.isEmpty()) throw new Exception("No face detected in source image");
        float[] sourceEmbedding = getSourceEmbedding(sourceImage, sourceFaces.get(0));

        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) throw new Exception("No face detected in target image");

        Bitmap result = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        for (FaceDetector.Face targetFace : targetFaces) {
            Bitmap next = swapOne(result, targetFace, sourceEmbedding);
            if (result != targetImage && !result.isRecycled()) result.recycle();
            result = next;
        }
        return result;
    }

    private float[] getSourceEmbedding(Bitmap sourceImage, FaceDetector.Face sourceFace) throws Exception {
        float[] sourceLandmarks = getRefinedLandmarks5(sourceImage, sourceFace);
        Bitmap alignedSource = SwapperImageUtils.alignArcFace112(sourceImage, sourceLandmarks);
        try {
            return faceEmbedder.getEmbedding(alignedSource);
        } finally {
            if (!alignedSource.isRecycled()) alignedSource.recycle();
        }
    }

    private Bitmap swapOne(Bitmap targetImage, FaceDetector.Face targetFace, float[] sourceEmbedding) throws Exception {
        float[] targetLandmarks = getRefinedLandmarks5(targetImage, targetFace);
        int qualitySize = FaceSwapper.QUALITY_SIZE;
        Bitmap alignedTarget = SwapperImageUtils.alignFace(targetImage, targetLandmarks, qualitySize);
        float[] semanticMask = faceParser != null ? faceParser.createMask(alignedTarget) : null;
        Bitmap swappedFace = null;
        try {
            swappedFace = faceSwapper.swapFaceQuality(alignedTarget, sourceEmbedding);
            return SwapperImageUtils.blendFace(
                targetImage, alignedTarget, swappedFace, targetLandmarks, qualitySize, semanticMask);
        } catch (Exception qualityError) {
            throw new Exception(qualitySize + " quality face swap failed: " + qualityError.getMessage(), qualityError);
        } finally {
            if (!alignedTarget.isRecycled()) alignedTarget.recycle();
            if (swappedFace != null && !swappedFace.isRecycled()) swappedFace.recycle();
        }
    }

    private float[] getRefinedLandmarks5(Bitmap image, FaceDetector.Face face) {
        if (faceLandmarker != null) {
            FaceLandmarker.Result refined = faceLandmarker.refine(image, face);
            if (refined != null && refined.landmarks5 != null && refined.landmarks5.length >= 10) {
                return refined.landmarks5;
            }
        }
        return face.landmarks;
    }
}
