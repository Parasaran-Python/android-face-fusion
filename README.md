# Android Face Fusion

[![Android CI](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/ci.yml)
[![CodeQL Analysis](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/codeql.yml/badge.svg)](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/codeql.yml)

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

## License

This project's source code is licensed under the [MIT License](LICENSE).

**Important:** The AI models (SCRFD, ArcFace, INSwapper) downloaded at runtime are provided by [InsightFace](https://github.com/deepinsight/insightface) and are subject to their own **non-commercial research license**. You must comply with InsightFace's licensing terms when using these models. This project does not redistribute the models — they are downloaded directly from HuggingFace by the end user.

## Disclaimer

This project is for **educational and research purposes only**. Face swapping technology can be misused. By using this software, you agree to:

- Only use it on images where you have consent from all individuals depicted
- Not use it to create deceptive, harmful, or non-consensual content
- Comply with all applicable laws and regulations in your jurisdiction
- Take full responsibility for how you use the output

The authors are not responsible for any misuse of this software.

## Credits

- [FaceFusion](https://github.com/facefusion/facefusion) - Original Python implementation
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis models
