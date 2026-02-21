# Android Face Fusion

Android app for face swapping using ONNX models on-device. Port of the Python [FaceFusion](https://github.com/facefusion/facefusion) pipeline.

## Quick Start

1. Open in Android Studio, sync Gradle, build & run
2. On first launch, models auto-download from HuggingFace (~739 MB)
3. Select source image (face to use) and target image (face to replace)
4. Tap **Swap Faces**

## Models

All models download automatically on first launch from `huggingface.co/leonelhs/insightface`. Manual placement in `app/src/main/assets/` also works.

| Model | Purpose | Input | Output | Size |
|---|---|---|---|---|
| `det_10g.onnx` | Face detection (SCRFD) | 640x640 RGB | Boxes + 5 landmarks | ~16 MB |
| `w600k_r50.onnx` | Face embedding (ArcFace) | 112x112 aligned face | 512-dim vector | ~166 MB |
| `inswapper_128.onnx` | Face swapping (INSwapper) | 128x128 face + 512-dim embedding | 128x128 swapped face | ~553 MB |

### EMAP

The INSwapper model requires a 512x512 transformation matrix (EMAP) extracted from its last graph initializer. Pre-extracted as `app/src/main/assets/emap.bin`. To re-extract:

```bash
python extract_emap.py
cp emap.bin app/src/main/assets/
```

## Pipeline

1. **Detect** faces in both images (SCRFD, 640x640, norm: `(px-127.5)/128`)
2. **Align** source face to 112x112 using 5-point landmarks + similarity transform
3. **Embed** aligned source face into 512-dim vector (ArcFace, norm: `(px-127.5)/127.5`)
4. **Transform** embedding: `latent = dot(embedding, emap); latent /= norm(latent)`
5. **Align** target face to 128x128 for swapping
6. **Swap** face (INSwapper, norm: `px/255.0`)
7. **Blend** swapped face back using inverse transform + mask erosion + Gaussian blur

## Project Structure

```
app/src/main/java/com/pv/androidfacefusion/
├── MainActivity.java          UI and image loading
├── FaceFusionProcessor.java   Pipeline orchestrator
├── FaceDetector.java          SCRFD face detection
├── FaceEmbedder.java          ArcFace face embedding
├── FaceSwapper.java           INSwapper face swapping
├── ImageUtils.java            Alignment, transforms, blending
├── ModelDownloader.java       HuggingFace model downloader
└── Face.java                  Face data (bbox, landmarks, embedding)
```

## Requirements

- Android Studio, Android SDK API 26+
- Device with ~1 GB free RAM, ~800 MB storage
- Internet on first launch (model download)

## Troubleshooting

| Problem | Fix |
|---|---|
| "Failed to load models" | Check internet connection and free storage (800+ MB) |
| "No face detected" | Use clear, frontal face images with good lighting |
| Crash during processing | Reduce image size; ensure 1+ GB free RAM |
| Poor swap quality | Verify `emap.bin` exists in assets (not identity matrix) |

## Dependencies

- [ONNX Runtime Android](https://onnxruntime.ai/) - Model inference
- [Glide](https://github.com/bumptech/glide) - URL image loading
- [Material Components](https://material.io/develop/android) - UI

## Credits

- [FaceFusion](https://github.com/facefusion/facefusion) - Original Python implementation
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis models
