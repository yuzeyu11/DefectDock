# Migration from the prototype

DefectDock is a new repository, not a renamed copy of the demonstration project.
Only first-party domain logic with a clear commercial boundary was migrated.

## Retained and renamed

- Image upload validation, hashing and duplicate detection
- Dataset and run domain models
- SQLite metadata stores
- CVAT connection, task creation and versioned annotation synchronization
- Dataset checking, statistics, Pascal VOC conversion and GC10-DET import
- Industrial detection-rate evaluation, threshold optimization, miss analysis
  and acceptance reports
- FastAPI dataset/run endpoints
- Camera, local-video and RTSP robustness utilities
- Relevant unit and API regression tests

These modules were moved to the `defectdock` namespace and had repository-root,
runtime-directory and configuration names corrected for a `src/` layout.

## Rewritten

- Configuration schema: one explicit object-detection task and TorchVision engine
- Training engine: native TorchVision Faster R-CNN adapter and DefectDock checkpoint
- Inference: framework-owned service returning stable detection dictionaries
- CLI: removed prototype-only commands and exposed reproducible lifecycle actions
- API root and health response: no dependency on a local demo HTML file
- Model recommendations: conservative presets with no accuracy or latency promise
- Frontend: React + TypeScript workbench shell instead of the demonstration page
- Packaging, container, CI, README, licensing and security documentation

## Deliberately excluded

- The original `.git` history and local virtual environments
- Datasets, customer material, model weights, output runs and local credentials
- The bundled CVAT source tree; CVAT remains an optional external integration
- Prototype HTML pages, screenshots and scripts tied to one demonstration scene
- Ultralytics package declarations, imports, training/inference adapters, CLI calls,
  default weight names and generated framework caches
- Claims in old reports or plans that are not enforced by executable checks

## Compatibility note

The normalized five-column detection label layout is retained for dataset
interoperability and CVAT exchange. Retaining a text representation does not
retain a training framework, source-code dependency or model license. Function
names that contain `yolo` are currently limited to data-format conversion and
CVAT's official export-format name; they must not be used as evidence of a
runtime engine dependency.
