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
        if (sourceFaces.isEmpty()) throw new Exception("No face detected in source image. Please use an image with a clear face.");
        float[] sourceEmbedding = getSourceEmbedding(sourceImage, sourceFaces.get(0));

        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) throw new Exception("No face detected in target image.");

        Bitmap result = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        for (int i = 0; i < targetFaces.size(); i++) {
            if (!selectedFaceIndices.contains(i)) continue;
            Bitmap previous = result;
            Bitmap next = swapOne(previous, targetFaces.get(i), sourceEmbedding);
            if (next != previous && previous != targetImage && !previous.isRecycled()) previous.recycle();
            result = next;
        }
        return result;
    }

    public Bitmap processFaceFusionWithMapping(Bitmap targetImage, Map<Integer, float[]> targetIndexToEmbeddingMap) throws Exception {
        if (targetIndexToEmbeddingMap == null || targetIndexToEmbeddingMap.isEmpty()) throw new Exception("No target face mappings selected.");
        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) throw new Exception("No face detected in target image.");

        Bitmap result = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        for (int i = 0; i < targetFaces.size(); i++) {
            float[] embedding = targetIndexToEmbeddingMap.get(i);
            if (embedding == null) continue;
            Bitmap previous = result;
            Bitmap next = swapOne(previous, targetFaces.get(i), embedding);
            if (next != previous && previous != targetImage && !previous.isRecycled()) previous.recycle();
            result = next;
        }
        return result;
    }

    public Bitmap processFaceFusion(Bitmap sourceImage, Bitmap targetImage, int targetFaceIndex) throws Exception {
        if (targetFaceIndex == -1) return processFaceFusionMultiple(sourceImage, targetImage);

        Log.d(TAG, "Starting " + FaceSwapper.QUALITY_SIZE + " quality face fusion; native HyperSwap tile="
            + faceSwapper.getInputSize() + ", HyperSwap=" + faceSwapper.isUsingHyperSwap());

        List<FaceDetector.Face> sourceFaces = faceDetector.detectFaces(sourceImage);
        if (sourceFaces.isEmpty()) throw new Exception("No face detected in source image. Please use an image with a clear face.");
        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) throw new Exception("No face detected in target image.");
        if (targetFaceIndex < 0 || targetFaceIndex >= targetFaces.size()) targetFaceIndex = 0;

        float[] sourceEmbedding = getSourceEmbedding(sourceImage, sourceFaces.get(0));

        // Always process against a dedicated mutable working bitmap so the caller's original
        // target can never be modified in place by ROI compositing.
        Bitmap workingTarget = targetImage.copy(Bitmap.Config.ARGB_8888, true);
        try {
            Bitmap result = swapOne(workingTarget, targetFaces.get(targetFaceIndex), sourceEmbedding);
            if (result != workingTarget && !workingTarget.isRecycled()) workingTarget.recycle();
            return result;
        } catch (Exception e) {
            if (!workingTarget.isRecycled()) workingTarget.recycle();
            throw e;
        }
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
            Bitmap previous = result;
            Bitmap next = swapOne(previous, targetFace, sourceEmbedding);
            if (next != previous && previous != targetImage && !previous.isRecycled()) previous.recycle();
            result = next;
        }
        return result;
    }

    private float[] getSourceEmbedding(Bitmap sourceImage, FaceDetector.Face sourceFace) throws Exception {
        FaceLandmarker.Result refined = refine(sourceImage, sourceFace);
        float[] sourceLandmarks = selectStableLandmarks5(sourceFace, refined);
        Bitmap alignedSource = SwapperImageUtils.alignArcFace112(sourceImage, sourceLandmarks);
        try {
            return faceEmbedder.getEmbedding(alignedSource);
        } finally {
            if (!alignedSource.isRecycled()) alignedSource.recycle();
        }
    }

    private Bitmap swapOne(Bitmap targetImage, FaceDetector.Face targetFace, float[] sourceEmbedding) throws Exception {
        FaceLandmarker.Result refined = refine(targetImage, targetFace);
        float[] targetLandmarks = selectStableLandmarks5(targetFace, refined);
        float[] targetLandmarks68 = isRefinedGeometryReliable(targetFace, refined)
            ? refined.landmarks68 : null;
        int qualitySize = FaceSwapper.QUALITY_SIZE;
        Bitmap alignedTarget = SwapperImageUtils.alignFace(targetImage, targetLandmarks, qualitySize);
        float[] semanticMask = faceParser != null ? faceParser.createMask(alignedTarget) : null;
        Bitmap swappedFace = null;
        try {
            swappedFace = faceSwapper.swapFaceQuality(alignedTarget, sourceEmbedding);
            return SwapperImageUtils.blendFace(targetImage, alignedTarget, swappedFace,
                targetLandmarks, targetLandmarks68, qualitySize, semanticMask);
        } catch (Exception qualityError) {
            throw new Exception(qualitySize + " quality face swap failed: " + qualityError.getMessage(), qualityError);
        } finally {
            if (!alignedTarget.isRecycled()) alignedTarget.recycle();
            if (swappedFace != null && !swappedFace.isRecycled()) swappedFace.recycle();
        }
    }

    private FaceLandmarker.Result refine(Bitmap image, FaceDetector.Face face) {
        return faceLandmarker != null ? faceLandmarker.refine(image, face) : null;
    }

    /**
     * Only allow 68-point geometry to control pose/expression masks when the same refined
     * result is credible against SCRFD. This prevents an unstable 2DFAN result from being
     * rejected for alignment but still distorting the mask around eyes, mouth or jaw.
     */
    private boolean isRefinedGeometryReliable(FaceDetector.Face face, FaceLandmarker.Result refined) {
        if (refined == null || refined.landmarks68 == null || refined.landmarks68.length < 136
                || refined.landmarks5 == null || refined.landmarks5.length < 10) {
            return false;
        }

        float score = Math.max(0.0f, Math.min(1.0f, refined.score));
        if (score < 0.55f) return false;
        if (face.landmarks == null || face.landmarks.length < 10) return score >= 0.75f;

        float eyeDx = face.landmarks[2] - face.landmarks[0];
        float eyeDy = face.landmarks[3] - face.landmarks[1];
        float eyeSpan = Math.max(1.0f, (float) Math.sqrt(eyeDx * eyeDx + eyeDy * eyeDy));
        float displacementSq = 0.0f;
        float maxDisplacement = 0.0f;
        for (int i = 0; i < 5; i++) {
            float dx = refined.landmarks5[i * 2] - face.landmarks[i * 2];
            float dy = refined.landmarks5[i * 2 + 1] - face.landmarks[i * 2 + 1];
            float distance = (float) Math.sqrt(dx * dx + dy * dy) / eyeSpan;
            displacementSq += distance * distance;
            maxDisplacement = Math.max(maxDisplacement, distance);
        }
        float rmsDisplacement = (float) Math.sqrt(displacementSq / 5.0f);
        boolean reliable = maxDisplacement <= 0.45f && rmsDisplacement <= 0.28f;
        if (!reliable) {
            Log.d(TAG, "2DFAN 68-point mask geometry rejected: score=" + score
                + ", rmsShift=" + rmsDisplacement + ", maxShift=" + maxDisplacement);
        }
        return reliable;
    }

    /**
     * Confidence plus displacement-aware geometry fusion. Confidence alone is not enough:
     * refined landmarks that move implausibly far from the stable detector geometry are
     * restrained even when the landmarker reports a strong score.
     */
    private float[] selectStableLandmarks5(FaceDetector.Face face, FaceLandmarker.Result refined) {
        if (refined == null || refined.landmarks5 == null || refined.landmarks5.length < 10) {
            return face.landmarks;
        }
        if (face.landmarks == null || face.landmarks.length < 10) return refined.landmarks5;

        float score = Math.max(0.0f, Math.min(1.0f, refined.score));
        if (score < 0.55f) {
            Log.d(TAG, "2DFAN confidence too low for alignment; using SCRFD anchors: score=" + score);
            return face.landmarks;
        }

        float eyeDx = face.landmarks[2] - face.landmarks[0];
        float eyeDy = face.landmarks[3] - face.landmarks[1];
        float eyeSpan = Math.max(1.0f, (float) Math.sqrt(eyeDx * eyeDx + eyeDy * eyeDy));
        float displacementSq = 0.0f;
        float maxDisplacement = 0.0f;
        for (int i = 0; i < 5; i++) {
            float dx = refined.landmarks5[i * 2] - face.landmarks[i * 2];
            float dy = refined.landmarks5[i * 2 + 1] - face.landmarks[i * 2 + 1];
            float distance = (float) Math.sqrt(dx * dx + dy * dy) / eyeSpan;
            displacementSq += distance * distance;
            maxDisplacement = Math.max(maxDisplacement, distance);
        }
        float rmsDisplacement = (float) Math.sqrt(displacementSq / 5.0f);

        float confidence = Math.min(1.0f, (score - 0.55f) / 0.30f);
        float refinedWeight = 0.35f + 0.57f * confidence;

        // A single point moving >45% of the detector inter-eye distance, or an overall RMS
        // shift >28%, is a warning that the refined geometry is likely unstable for alignment.
        float displacementPenalty = 1.0f;
        if (maxDisplacement > 0.45f || rmsDisplacement > 0.28f) {
            float severity = Math.max(maxDisplacement / 0.45f, rmsDisplacement / 0.28f);
            displacementPenalty = Math.max(0.20f, 1.0f / severity);
            refinedWeight *= displacementPenalty;
        }
        refinedWeight = Math.max(0.12f, Math.min(0.92f, refinedWeight));
        float detectorWeight = 1.0f - refinedWeight;

        float[] fused = new float[10];
        for (int i = 0; i < 10; i++) {
            fused[i] = refined.landmarks5[i] * refinedWeight + face.landmarks[i] * detectorWeight;
        }
        Log.d(TAG, "Fused SCRFD/2DFAN geometry: score=" + score
            + ", rmsShift=" + rmsDisplacement + ", maxShift=" + maxDisplacement
            + ", penalty=" + displacementPenalty + ", refinedWeight=" + refinedWeight);
        return fused;
    }
}
