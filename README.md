# Android Face Fusion

[![Android CI](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/ci.yml)
[![CodeQL Analysis](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/codeql.yml/badge.svg)](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/codeql.yml)
[![Gitleaks Security Scan](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/gitleaks.yml)
[![Dependency Review](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/Parasaran-Python/android-face-fusion/actions/workflows/dependency-review.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Android API](https://img.shields.io/badge/API-26%2B-brightgreen.svg)](https://developer.android.com)
[![16 KB Page Alignment](https://img.shields.io/badge/16%20KB%20Page%20Size-Compliant-blue.svg)](ARCHITECTURE.md)

Android app for high-fidelity on-device face swapping using ONNX deep learning models. Direct native Android port of the Python [FaceFusion](https://github.com/facefusion/facefusion) pipeline.

---

## Quick Start

1. Open in Android Studio (Jellyfish / Koala or newer), sync Gradle, and run.
2. On first launch, required models auto-download from HuggingFace (~739 MB).
3. Select a **Source** image (face to apply) and a **Target** image (face to replace).
4. Tap **Swap Faces**.

---

## Models

All models download automatically on first launch from `huggingface.co/leonelhs/insightface`. Manual placement in `app/src/main/assets/` is also supported.

| Model | Purpose | Input Dimensions | Output Dimensions | Approximate Size |
|---|---|---|---|---|
| `det_10g.onnx` | Face detection (SCRFD) | 640x640 RGB | Bounding boxes + 5-point landmarks | ~16 MB |
| `w600k_r50.onnx` | Face embedding (ArcFace) | 112x112 aligned face | 512-dimensional vector | ~166 MB |
| `inswapper_128.onnx` | Face swapping (INSwapper) | 128x128 face + 512-dim embedding | 128x128 swapped face | ~553 MB |

### EMAP Transformation Matrix

The INSwapper model requires a 512x512 transformation matrix (EMAP) extracted from its last graph initializer. It is pre-extracted in `app/src/main/assets/emap.bin`. To re-extract:

```bash
python extract_emap.py
cp emap.bin app/src/main/assets/
```

---

## Pipeline Architecture

1. **Detect**: Detect faces in source and target images via SCRFD (`640x640`, normalized: `(px - 127.5) / 128`).
2. **Align**: Align source face to `112x112` using 5-point facial landmarks and similarity transformation.
3. **Embed**: Generate 512-dimensional normalized face embedding via ArcFace (`(px - 127.5) / 127.5`).
4. **Transform**: Transform embedding into latent space: `latent = dot(embedding, emap); latent /= norm(latent)`.
5. **Align Target**: Crop and align target face to `128x128`.
6. **Swap**: Run INSwapper (`px / 255.0`) with latent embedding to produce swapped face.
7. **Blend**: Blend swapped face back using inverse similarity transform, convex mask erosion, and Gaussian blurring.

---

## Continuous Integration & Automated Builds

This project features a comprehensive CI/CD pipeline powered by GitHub Actions:

- **Android CI ([`ci.yml`](.github/workflows/ci.yml))**:
  - Validates Gradle wrapper cryptographic integrity.
  - Verifies Python script syntax and compilation.
  - Runs Android Lint analysis (`lintDebug`).
  - Executes unit test suites (`testDebugUnitTest`).
  - Compiles both **Debug & Release APKs** and **Android App Bundles (AAB)**.
  - Formally verifies **16 KB page-size compliance** on all packaged 64-bit native libraries (`arm64-v8a`, `x86_64`) per Android 15 requirements.
- **Automated Release ([`release.yml`](.github/workflows/release.yml))**:
  - Triggers on git tags (`v*.*.*`) or manual dispatch.
  - Assembles Release APK and App Bundle (`.aab`).
  - Generates SHA-256 checksums (`SHA256SUMS.txt`).
  - Automatically drafts and publishes GitHub Releases with attached distribution binaries.
- **CodeQL Security Scanning ([`codeql.yml`](.github/workflows/codeql.yml))**:
  - Static application security testing (SAST) for Java/Kotlin and Python codebases.
- **Gitleaks Secret Scanning ([`gitleaks.yml`](.github/workflows/gitleaks.yml))**:
  - Continuous scanning to prevent credentials, secrets, or keystores from entering git history.
- **Dependency Review ([`dependency-review.yml`](.github/workflows/dependency-review.yml))**:
  - Validates pull request dependencies for known CVEs and license compliance.
- **PR Title Linter ([`pr-lint.yml`](.github/workflows/pr-lint.yml))**:
  - Enforces Conventional Commits standards on pull requests.

---

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

---

## Requirements

- Android Studio, Android SDK API 26+
- Android NDK `28.2.13676358` (for 16 KB page-aligned libc++ sysroots)
- Device with ~1 GB free RAM, ~800 MB storage
- Active internet connection on first launch for model download

---

## Troubleshooting

| Problem | Potential Resolution |
|---|---|
| "Failed to load models" | Check internet connection and verify free storage (800+ MB). |
| "No face detected" | Use clear, frontal face images with good lighting. |
| Crash during processing | Reduce image dimensions; ensure 1+ GB free RAM. |
| Poor swap quality | Verify `emap.bin` exists in assets (not an identity matrix). |

---

## Contributing & Community

We welcome contributions! Please see:
- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)

---

## License

The source code of **Android Face Fusion** is licensed under the [MIT License](LICENSE).

### Model Weights Licensing Notice
The deep learning models (SCRFD, ArcFace, and INSwapper) downloaded at runtime are created by the [InsightFace](https://github.com/deepinsight/insightface) project and are strictly governed by their own **non-commercial research use** license. This repository contains only original open-source application source code and does not redistribute proprietary model binaries.

---

## Responsible Use & Disclaimer

This software is developed strictly for **educational, academic, and research purposes**. Face manipulation and deep learning synthesis technologies carry significant ethical responsibilities:

- You must obtain explicit consent from all individuals depicted prior to processing.
- Do not use this software to generate deceptive, defamatory, non-consensual, or unlawful content.
- Comply fully with all local and international laws governing biometric data and synthetic media.

The maintainers and contributors assume no liability for misuse of this software.

---

## Credits & Acknowledgments

This project builds upon exceptional open-source research and engineering across the deep learning and computer vision communities:

- **[FaceFusion](https://github.com/facefusion/facefusion)**: Henry Ruhs and the FaceFusion team for the reference Python architecture, pipeline orchestration, and image processing pipeline.
- **[InsightFace (DeepInsight)](https://github.com/deepinsight/insightface)**: Jiankang Deng, Jia Guo, and the DeepInsight research team for developing the state-of-the-art SCRFD, ArcFace, and INSwapper architectures.
  - *SCRFD*: Guo et al., *"Sample and Computation Redistribution for Efficient Face Detection"*, 2021.
  - *ArcFace*: Deng et al., *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"*, CVPR 2019.
- **[ONNX Runtime](https://onnxruntime.ai/)**: Microsoft for the high-performance cross-platform on-device machine learning runtime.
- **[OpenCV](https://opencv.org/)**: The OpenCV development team for foundational computer vision transformation and image manipulation routines.
- **[Glide](https://github.com/bumptech/glide)**: Bumptech for efficient Android image loading and bitmap caching.
- **[LeonelHS](https://huggingface.co/leonelhs/insightface)**: For hosting and maintaining accessible HuggingFace checkpoints of the ONNX models.
- **[Material Components for Android](https://github.com/material-components/material-components-android)**: Google Material Design team for Android modern UI components.

