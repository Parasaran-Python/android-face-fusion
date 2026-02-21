# Architecture: Python vs Android Mapping

Technical reference for maintaining parity between the Python FaceFusion codebase and this Android port.

## Model Normalization

| Model | Python code | Formula | Android location |
|---|---|---|---|
| SCRFD (det_10g) | `input_mean=127.5, input_std=128.0` | `(px - 127.5) / 128.0` | `FaceDetector.bitmapToFloatArray()` |
| ArcFace (w600k_r50) | `input_mean=127.5, input_std=127.5` | `(px - 127.5) / 127.5` | `FaceEmbedder.bitmapToFloatArray()` |
| INSwapper (inswapper_128) | `input_mean=0.0, input_std=255.0` | `px / 255.0` | `FaceSwapper.bitmapToFloatArray()` |

All models use **NCHW** layout (batch, channels, height, width) with **RGB** channel order.

## Face Detection (SCRFD)

**Python**: `models/retinaface.py`
**Android**: `FaceDetector.java`

Output format depends on model variant:
- **9 outputs** (3 FPN levels x 3): scores, bboxes, keypoints — strides `[8, 16, 32]`, 2 anchors
- **10 outputs** (5 FPN levels x 2): scores, bboxes — strides `[8, 16, 32, 64, 128]`, 1 anchor
- **6 outputs** (3 FPN levels x 2): scores, bboxes — strides `[8, 16, 32]`, 1 anchor

Box format: `distance2bbox` — `[left, top, right, bottom]` distances from anchor center.
Landmark format: `distance2kps` — `[dx1, dy1, dx2, dy2, ..., dx5, dy5]` offsets from anchor center.

NMS threshold: 0.4. Detection threshold: 0.5.

## Face Alignment (Similarity Transform)

**Python**: `utils/face_align.py` → `skimage.transform.SimilarityTransform().estimate()`
**Android**: `ImageUtils.estimateSimilarityTransform()`

Uses Procrustes analysis (Umeyama algorithm) to compute a 2x3 affine matrix from 5 detected landmarks to reference templates.

Reference landmarks for **112x112** (ArcFace):
```
[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
[41.5493, 92.3655], [70.7299, 92.2041]
```

Reference landmarks for **128x128** (INSwapper): same points scaled by `128/112` and shifted by `+8.0` in X.

## EMAP Transformation

**Python**: `inswapper.py` line 18: `self.emap = numpy_helper.to_array(graph.initializer[-1])`
**Android**: `FaceSwapper.extractEmap()` loads from `assets/emap.bin`

Transform: `latent = dot(embedding, emap)` then L2-normalize.

Binary format of `emap.bin`: 8-byte header (2 little-endian int32: rows=512, cols=512) + 512x512 float32 matrix.

## Face Blending

**Python**: `inswapper.py` paste-back logic
**Android**: `ImageUtils.blendFaces()`

Steps:
1. Create white mask on aligned face region (128x128)
2. Inverse-warp mask back to original image coordinates
3. Threshold mask at `> 20` (set to 255)
4. **Erode** mask — kernel size: `max(maskSize / 10, 10)` where maskSize = `sqrt(mask_area)`
5. **Gaussian blur** mask — kernel size: `2 * max(maskSize / 20, 5) + 1`, sigma from OpenCV formula
6. Normalize mask to `[0, 1]` float and alpha-blend: `result = swapped * alpha + original * (1 - alpha)`

## Key Fixes Applied (vs initial Android port)

1. **Similarity transform**: Replaced incorrect `sqrt(dstNorm/srcNorm)` scale with Procrustes solution (`a/srcNorm`, `-b/srcNorm`)
2. **Blending kernels**: Changed from fixed 3x3 erosion / 5x5 blur to dynamic sizes matching Python
3. **Detection outputs**: Added support for 10-output and 6-output SCRFD variants (was only handling 9)
4. **Memory leak**: Removed unused bitmap allocation in `blendFaces()`
