# Changelog

All notable changes are recorded here. This project follows Semantic Versioning
once the public API reaches `1.0.0`.

## Unreleased

- Added routed workbench pages for datasets, training, runs, model management,
  and inference, with frontend regression tests.
- Added model-assisted annotation review and registered, hash-verified training
  snapshots linked to explicit annotation versions.
- Added model registration, approval, atomic activation, activation history,
  and rollback, with database migrations for the new metadata.
- Added optional ONNX export packages with integrity manifests, numerical
  comparison, and CPU benchmarks; target-hardware acceptance remains pending.
- Added local/network security modes, Bearer authentication, request limits,
  and redacted audit events.
- Extended ignore rules for local databases, frontend coverage, and build
  caches while keeping the export source package visible to Git.

## 0.1.0 - 2026-08-29

- Established DefectDock as an independent repository.
- Migrated dataset ingestion, CVAT sync, evaluation, run metadata, API, and
  camera-stream utilities from the internal prototype.
- Replaced the previous training/inference dependency with a built-in
  PyTorch/TorchVision Faster R-CNN adapter.
- Added reproducible configuration snapshots, checkpoint metadata, license
  boundary checks, CI, container files, and migration documentation.
- Added verified Geti, AWS DDA and Anomalib Studio reference scopes with a
  per-file Apache-2.0 provenance and NOTICE adoption policy.
- Added explicit workspace settings, packaged read-only examples, a single-GPU
  background queue, cooperative cancellation, restart recovery, and immutable
  hashed training snapshots.
- Added run provenance manifests for code, dependencies, datasets, hardware and
  pretrained weights; corrected FDR metric naming.
- Added a cross-platform `uv.lock` and a CI lockfile consistency gate.
- Verified the wheel outside the source tree, a healthy lightweight Docker
  container, and a real CUDA smoke training run on an RTX 2060.
- Added packaged Alembic migrations with legacy-database backup, failure
  recovery, restore commands, and explicit annotation-version foreign keys.
- Added Python and pnpm CycloneDX SBOM, vulnerability, license-policy, evidence
  hashing, artifact retention, immutable CI action pins, and Dependabot gates.
- Added a pinned, lockfile-driven Python 3.13/Trixie lightweight image; removed
  runtime build tools and unnecessary GUI libraries, and separated complete
  container findings from the fixable high/critical release gate.
- Added a locked CUDA 13 GPU image variant, Compose override, runtime probe,
  self-hosted GPU evidence workflow, and verified one-epoch RTX 2060 training.
- Embedded validated Git revisions in OCI labels and container training
  manifests, and separated dependency installation from project installation
  to keep subsequent GPU rebuilds fast.
