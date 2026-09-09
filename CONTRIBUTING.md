# Contributing to Android Face Fusion

Thank you for your interest in contributing to **Android Face Fusion**! We welcome contributions from the community to help make on-device neural face swapping faster, more efficient, and accessible.

Please take a moment to read this guide before contributing.

---

## Code of Conduct & Ethical Use

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

### Responsible AI & Ethics
Face swapping technology must be used responsibly. Contributors must agree to:
- Respect privacy, human dignity, and individual consent.
- Never contribute features designed to deceive, harass, impersonate without consent, or bypass safety mechanisms.
- Adhere to the non-commercial research licensing terms of the underlying deep learning models ([InsightFace](https://github.com/deepinsight/insightface)).

---

## How to Contribute

### 1. Reporting Bugs
- Check existing [GitHub Issues](https://github.com/Parasaran-Python/android-face-fusion/issues) before submitting a new one.
- Use the **Bug Report** template to provide detailed steps to reproduce, Android OS version, device model, and relevant logcat output.

### 2. Suggesting Features & Enhancements
- Open a feature request via the **Feature Request** issue template.
- Describe the motivation, desired user experience, and any technical considerations (such as NAPI/GPU acceleration or memory impact).

### 3. Submitting Pull Requests

1. **Fork and Branch**: Fork the repository and create a branch from `master`:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```
2. **Commit Conventions**: We use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat(scope): ...` for new features
   - `fix(scope): ...` for bug fixes
   - `perf(scope): ...` for performance improvements
   - `docs: ...` for documentation changes
   - `ci: ...` for CI/CD workflow updates
   - `test: ...` for tests
   - `chore: ...` for general maintenance
3. **Run Checks Locally**:
   Ensure lint, unit tests, and builds pass cleanly:
   ```bash
   # Run lint and unit tests
   ./gradlew check

   # Build APKs and App Bundles
   ./gradlew assembleDebug assembleRelease bundleDebug bundleRelease
   ```
4. **16 KB Page Alignment**:
   All 64-bit native libraries (`arm64-v8a`, `x86_64`) must adhere to Android 15+ 16 KB page-size alignment.
5. **Open Pull Request**:
   Fill in the provided PR template and await review.

---

## Development Environment

- **JDK**: Java 21 (Temurin recommended)
- **Android SDK**: Compile SDK 37, Min SDK 26, Target SDK 36
- **Android NDK**: Version `28.2.13676358`
- **Build System**: Gradle 9.7+ with Android Gradle Plugin 9.4+
- **Python**: Python 3.10+ (for `extract_emap.py` utility)

---

## License

By contributing to Android Face Fusion, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
