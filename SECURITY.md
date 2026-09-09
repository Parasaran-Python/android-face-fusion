# Security Policy

## Supported Versions

Security updates and patches are actively maintained for the latest release on the `master` branch.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a vulnerability or security concern, please report it responsibly:

1. **Do NOT open a public GitHub issue** for sensitive security vulnerabilities or secret leaks.
2. Please use the [GitHub Private Vulnerability Reporting](https://github.com/Parasaran-Python/android-face-fusion/security/advisories/new) feature to submit security disclosures.
3. Include detailed reproduction steps, proof-of-concept code, and the affected system/device versions.

You will receive an acknowledgement within 48-72 hours, along with regular updates regarding remediation.

---

## Automated Security Protections

This repository employs automated security testing:
- **CodeQL**: Weekly and per-PR static code analysis for security vulnerabilities.
- **Gitleaks**: Automated secret scanning to prevent accidental exposure of keys or credentials.
- **Dependency Review**: Automated scanning for known vulnerable dependencies in pull requests.
- **Gradle Wrapper Verification**: Cryptographic validation of Gradle wrapper binaries.
