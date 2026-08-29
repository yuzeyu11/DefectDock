# Changelog

All notable changes are recorded here. This project follows Semantic Versioning
once the public API reaches `1.0.0`.

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
- Verified the wheel outside the source tree, a healthy lightweight Docker
  container, and a real CUDA smoke training run on an RTX 2060.
