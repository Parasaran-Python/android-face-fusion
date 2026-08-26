package com.pv.androidfacefusion;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import java.util.List;
import java.util.Map;
import java.util.Set;

/** Coordinates detection, refined alignment, ArcFace identity extraction and selectable face swapping. */
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
        if (targetIndexToEmbeddingMap == null || targetIndexToEmbeddingMap.isEmpty()) {
            throw new Exception("No target face mappings selected.");
        }
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

        Log.d(TAG, "Starting " + FaceSwapper.QUALITY_SIZE + " quality face fusion with "
            + faceSwapper.getActiveModelName() + "; native input=" + faceSwapper.getInputSize());

        List<FaceDetector.Face> sourceFaces = faceDetector.detectFaces(sourceImage);
        if (sourceFaces.isEmpty()) throw new Exception("No face detected in source image. Please use an image with a clear face.");
        List<FaceDetector.Face> targetFaces = faceDetector.detectFaces(targetImage);
        if (targetFaces.isEmpty()) throw new Exception("No face detected in target image.");
        if (targetFaceIndex < 0 || targetFaceIndex >= targetFaces.size()) targetFaceIndex = 0;

        float[] sourceEmbedding = getSourceEmbedding(sourceImage, sourceFaces.get(0));

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
        float[] sourceLandmarks = selectPreferredLandmarks5(sourceFace, refined);
        Bitmap alignedSource = SwapperImageUtils.alignArcFace112(sourceImage, sourceLandmarks);
        try {
            return faceEmbedder.getEmbedding(alignedSource);
        } finally {
            if (!alignedSource.isRecycled()) alignedSource.recycle();
        }
    }

    private Bitmap swapOne(Bitmap targetImage, FaceDetector.Face targetFace, float[] sourceEmbedding) throws Exception {
        FaceLandmarker.Result refined = refine(targetImage, targetFace);
        float[] targetLandmarks = selectPreferredLandmarks5(targetFace, refined);
        float[] targetLandmarks68 = refined != null ? refined.landmarks68 : null;
        int qualitySize = FaceSwapper.QUALITY_SIZE;
        Bitmap alignedTarget = SwapperImageUtils.alignFace(targetImage, targetLandmarks, qualitySize);
        float[] semanticMask = faceParser != null ? faceParser.createMask(alignedTarget) : null;
        IdentityStrengthSettings.Level identityStrength = IdentityStrengthSettings.get(faceSwapper.getAppContext());
        float[] identityMask = IdentityStrengthUtils.strengthenMask(semanticMask, identityStrength);
        Bitmap swappedFace = null;
        Bitmap naturalisationReference = null;
        try {
            swappedFace = faceSwapper.swapFaceQuality(alignedTarget, sourceEmbedding);
            swappedFace = refineGeneratedExpression(swappedFace, targetLandmarks, targetLandmarks68, qualitySize);
            naturalisationReference = IdentityStrengthUtils.createNaturalisationReference(
                alignedTarget, swappedFace, identityStrength);
            Log.d(TAG, "Compositing " + faceSwapper.getActiveModelName() + " at "
                + identityStrength.displayName + " identity strength");
            return SwapperImageUtils.blendFace(targetImage, naturalisationReference, swappedFace,
                targetLandmarks, targetLandmarks68, qualitySize, identityMask);
        } catch (Exception qualityError) {
            throw new Exception(faceSwapper.getActiveModelName() + " quality face swap failed: "
                + qualityError.getMessage(), qualityError);
        } finally {
            if (naturalisationReference != null && naturalisationReference != alignedTarget
                    && naturalisationReference != swappedFace && !naturalisationReference.isRecycled()) {
                naturalisationReference.recycle();
            }
            if (!alignedTarget.isRecycled()) alignedTarget.recycle();
            if (swappedFace != null && !swappedFace.isRecycled()) swappedFace.recycle();
        }
    }

    /**
     * Additive refinement only: the selected swapper result remains authoritative. If generated
     * landmarks cannot be measured safely, keep that model's output unchanged rather than
     * substituting an older swapper or a different identity path.
     */
    private Bitmap refineGeneratedExpression(Bitmap swappedFace, float[] targetLandmarks5,
                                             float[] targetLandmarks68, int qualitySize) {
        if (faceLandmarker == null || targetLandmarks68 == null || targetLandmarks68.length < 136) {
            return swappedFace;
        }

        try {
            float[] targetAligned68 = LandmarkWarp.toHyperSwapCanonical(
                targetLandmarks5, targetLandmarks68, qualitySize);
            if (targetAligned68 == null) return swappedFace;

            List<FaceDetector.Face> generatedFaces = faceDetector.detectFaces(swappedFace);
            FaceDetector.Face generatedFace = largestFace(generatedFaces);
            if (generatedFace == null) return swappedFace;

            FaceLandmarker.Result generatedRefined = faceLandmarker.refine(swappedFace, generatedFace);
            if (generatedRefined == null || generatedRefined.landmarks68 == null
                    || generatedRefined.landmarks68.length < 136) {
                return swappedFace;
            }

            Bitmap corrected = FaceRefinementUtils.correctExpression(
                swappedFace, generatedRefined.landmarks68, targetAligned68);
            if (corrected != swappedFace) {
                if (!swappedFace.isRecycled()) swappedFace.recycle();
                Log.d(TAG, "Applied local 68-point expression correction for brows/eyes/mouth");
                return corrected;
            }
        } catch (Exception e) {
            Log.w(TAG, "Expression refinement skipped; preserving selected model output", e);
        }
        return swappedFace;
    }

    private FaceDetector.Face largestFace(List<FaceDetector.Face> faces) {
        if (faces == null || faces.isEmpty()) return null;
        FaceDetector.Face best = null;
        float bestArea = -1.0f;
        for (FaceDetector.Face face : faces) {
            RectF bbox = face != null ? face.bbox : null;
            if (bbox == null) continue;
            float area = Math.max(0.0f, bbox.width()) * Math.max(0.0f, bbox.height());
            if (area > bestArea) {
                bestArea = area;
                best = face;
            }
        }
        return best;
    }

    private FaceLandmarker.Result refine(Bitmap image, FaceDetector.Face face) {
        return faceLandmarker != null ? faceLandmarker.refine(image, face) : null;
    }

    /** Trust 2DFAN after its own confidence/coordinate validation; SCRFD is only the real fallback. */
    private float[] selectPreferredLandmarks5(FaceDetector.Face face, FaceLandmarker.Result refined) {
        if (refined != null && refined.landmarks5 != null && refined.landmarks5.length >= 10) {
            return refined.landmarks5;
        }
        return face.landmarks;
    }
}
