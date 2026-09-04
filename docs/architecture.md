# Architecture

## Design goals

DefectDock separates product workflow from model-framework details. Dataset and
run metadata, API contracts, industrial evaluation, and deployment state must
remain usable when a training adapter is replaced.

## Components

| Layer                | Responsibility                                                                      | Current implementation               |
| -------------------- | ----------------------------------------------------------------------------------- | ------------------------------------ |
| Workbench            | Operator-facing dataset, annotation, training, artifact activation and inference UI | React + TypeScript operational slice |
| API                  | Stable HTTP boundary and input validation                                           | FastAPI                              |
| Application services | Dataset ingestion, CVAT synchronization, config validation and run queries          | Python services                      |
| Metadata             | Dataset, annotation, snapshot, run, model and activation history                    | SQLite for the first deliverable     |
| Training adapter     | Framework-specific model construction and training                                  | TorchVision Faster R-CNN             |
| Evaluation           | Detection rate, false positives, misses and threshold selection                     | Framework-independent Python         |
| Inference            | Checkpoint loading and normalized detection responses                               | TorchVision adapter                  |
| Edge input           | Image, local video, USB camera and RTSP                                             | Pillow/OpenCV                        |

## Data and model flow

1. Images are uploaded or imported into an ignored local data root.
2. Metadata and content hashes are written to SQLite.
3. Annotations are synchronized from CVAT, imported, or proposed by an active model.
4. Model proposals remain candidates until an operator approves them; only approved,
   complete annotations can freeze a dataset.
5. A strict YAML run configuration is validated and hashed.
6. The engine adapter trains from the frozen version and writes structured events.
7. Evaluation emits industrial metrics and a threshold scan.
8. A DefectDock checkpoint stores architecture, classes, input size, framework,
   license identifier, config hash and state dictionary.
9. A successful run is registered with artifact hashes and provenance. Activation
   verifies the artifact, atomically updates the local pointer, and records an
   actor-attributed event; rollback uses the same guarded path.
10. Optional ONNX packages are published only after structural and numerical checks,
    with source hashes, runtime versions, I/O contract, and benchmark results in a manifest.

## Security boundary

The API has two explicit operating modes:

- `local` is the default and the CLI refuses to bind it to a non-loopback address;
- `network` requires a Bearer Token of at least 32 bytes for every endpoint except
  service discovery, health, and API documentation.

Both modes apply a configurable whole-request size limit. Mutating requests,
authentication failures, and rejected oversized requests append body-free,
credential-free JSON records to `.defectdock/audit.jsonl`, correlated by an
`X-Request-ID`. The shared token is a workstation deployment baseline, not a
replacement for TLS termination, user identities, role authorization, or a
central audit store.

## Target production evolution

SQLite and in-process training are appropriate for the first workstation
deliverable. Multi-user deployment should replace them with PostgreSQL, object
storage, a durable job queue, and identity-aware authorization. The API and engine boundaries are intentionally
kept independent so those changes do not alter dataset or evaluation semantics.

The frontend must consume only versioned API contracts. It must never read
training directories or the SQLite file directly.
