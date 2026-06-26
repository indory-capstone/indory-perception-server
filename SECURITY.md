# Security Policy

## Supported Versions

Until the first public release, only the current `main` branch is supported.
After release, update this section with the supported tag or branch policy.

## Reporting a Vulnerability

Do not open a public issue for suspected vulnerabilities, credential exposure,
private camera data, private map data, model artifact leakage, or service access
control problems.

After this repository is published on GitHub, enable private vulnerability
reporting or GitHub Security Advisories and use that channel for reports. Until
then, report issues through the project owner's private maintainer channel.

## Data and Artifact Handling

This repository should not contain secrets, private network details, camera
captures, map databases, rosbag files, benchmark image datasets, model weights,
checkpoints, GGUF files, PaddleOCR exports, or VLM caches.

Before public release, run a secret scan such as:

```bash
trufflehog filesystem .
```

Also review [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md) before
publishing code or dataset references.
