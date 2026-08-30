# Architecture

## Design goals

DefectDock separates product workflow from model-framework details. Dataset and
run metadata, API contracts, industrial evaluation, and deployment state must
remain usable when a training adapter is replaced.

## Components

| Layer | Responsibility | Current implementation |
|---|---|---|
| Workbench | Operator-facing dataset, annotation, training, artifact activation and inference UI | React + TypeScript operational slice |
| API | Stable HTTP boundary and input validation | FastAPI |
| Application services | Dataset ingestion, CVAT synchronization, config validation and run queries | Python services |
| Metadata | Dataset versions, images, run state and metrics | SQLite for the first deliverable |
| Training adapter | Framework-specific model construction and training | TorchVision Faster R-CNN |
| Evaluation | Detection rate, false positives, misses and threshold selection | Framework-independent Python |
| Inference | Checkpoint loading and normalized detection responses | TorchVision adapter |
| Edge input | Image, local video, USB camera and RTSP | Pillow/OpenCV |

## Data and model flow

1. Images are uploaded or imported into an ignored local data root.
2. Metadata and content hashes are written to SQLite.
3. Annotations are synchronized from CVAT or imported from supported formats.
4. Dataset checks run before the immutable dataset version is frozen.
5. A strict YAML run configuration is validated and hashed.
6. The engine adapter trains from the frozen version and writes structured events.
7. Evaluation emits industrial metrics and a threshold scan.
8. A DefectDock checkpoint stores architecture, classes, input size, framework,
   license identifier, config hash and state dictionary.
9. Deployment activation writes only a local pointer; the model artifact remains
   outside Git.

## Target production evolution

SQLite and in-process training are appropriate for the first workstation
deliverable. Multi-user deployment should replace them with PostgreSQL, object
storage and a durable job queue. The API and engine boundaries are intentionally
kept independent so those changes do not alter dataset or evaluation semantics.

The frontend must consume only versioned API contracts. It must never read
training directories or the SQLite file directly.
