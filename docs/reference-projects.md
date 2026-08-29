# Reference projects and clean-room adoption policy

Verified on 2026-08-29. This file records product and architecture references;
it does not declare these projects as DefectDock dependencies and does not mean
their code has been copied into this repository.

## Eligibility matrix

| Project | Verified repository and branch | Repository license | Reference status | Important exclusions |
|---|---|---|---|---|
| Geti | [`open-edge-platform/geti`, `develop`](https://github.com/open-edge-platform/geti) | [Apache-2.0](https://github.com/open-edge-platform/geti/blob/develop/LICENSE) | Approved for architecture study and per-file Apache-2.0 reuse | Optional Ultralytics integrations, related models/weights, trademarks and third-party assets are excluded |
| AWS Defect Detection Application (DDA) | [`awslabs/DefectDetectionApplication`, `main`](https://github.com/awslabs/DefectDetectionApplication) | [Apache-2.0](https://github.com/awslabs/DefectDetectionApplication/blob/main/LICENSE) | Approved for architecture study and per-file Apache-2.0 reuse | AWS Marketplace algorithms, paid services, cloud credentials and separately licensed container/model artifacts are not inherited from the repository license |
| Anomalib / Anomalib Studio source | [`open-edge-platform/anomalib`, `main`](https://github.com/open-edge-platform/anomalib) | [Apache-2.0](https://github.com/open-edge-platform/anomalib/blob/main/LICENSE) | Approved for anomaly-workflow study and per-file Apache-2.0 reuse | Every model/weight keeps its own license; Windows application distribution must be reviewed separately |

The root license is only the first gate. Every file, submodule, vendored asset,
model weight, dataset and transitive dependency must pass its own review before
entering DefectDock.

## What DefectDock should learn

### Geti

Use as the primary reference for the local end-to-end product experience:

- Project → dataset revision → training job → model version traceability
- A train-predict-review-annotate feedback loop with a human in control
- Task-aware project creation for classification, detection and segmentation
- Framework-neutral model metadata and export/deployment contracts
- Dataset/model comparison views and explicit training parameter snapshots
- Local container and workstation deployment as first-class installation modes

Do not copy or enable Geti's optional Ultralytics path. DefectDock should keep its
own engine protocol and permit only separately reviewed adapters.

### AWS DDA

Use as the primary reference for the portal-to-edge delivery boundary:

- Separate control plane, edge agent and inference runtime
- Device identity, grouping, desired deployment and observed deployment state
- Architecture-specific artifacts for x86-64 and ARM64 targets
- Store-and-forward inference result uploads for intermittent networks
- Deployment progress, failure reason, retry and rollback visibility
- Per-site/per-line configuration without rebuilding the model container

DefectDock remains cloud-neutral. Greengrass, SageMaker, Cognito, S3 and other AWS
services belong behind optional adapters rather than the core domain model.

### Anomalib Studio

Use as the primary reference for making anomaly detection a first-class task:

- Normal/anomalous dataset onboarding and validation
- Image-level classification plus pixel-level anomaly maps and masks
- Training, evaluation, inference and export as explicit background jobs
- Threshold selection, score distributions, heatmaps and inspection review
- Model capability metadata instead of hard-coded UI choices
- CPU, CUDA and OpenVINO execution profiles with measured compatibility

Anomalib may later be added as an optional engine dependency after its exact
version, dependencies, selected models and downloadable weights pass the release
license gate. Referencing the Studio UI does not authorize copying trademarks,
screenshots or non-Apache packaged assets.

## Adoption workflow

Before adapting any source file or design asset:

1. Open a reference-intake record containing upstream URL, branch/tag, commit,
   file path, license header, purpose and reviewer.
2. Confirm that the exact file is Apache-2.0 and that no separate directory or
   asset license overrides it.
3. Prefer reimplementation from public behavior and interfaces. Copy source only
   when doing so materially reduces risk or maintenance cost.
4. For copied or modified Apache files, retain applicable copyright and license
   notices, mark material changes and merge relevant upstream `NOTICE` content
   into DefectDock's distribution notices.
5. Review every imported package, model, weight and dataset independently.
6. Add tests that describe DefectDock behavior rather than upstream internals.
7. Record the resulting dependency or provenance entry in the release SBOM.

## Recommended implementation order

1. **P0 — provenance controls:** add a machine-readable reference-intake manifest
   and CI validation before any upstream source is copied.
2. **P1 — lifecycle model:** introduce project, dataset revision, model version and
   deployment entities inspired by Geti, using DefectDock-owned implementations.
3. **P1 — anomaly task:** add an optional Anomalib engine adapter and anomaly
   evaluation contracts without coupling the core API to Anomalib objects.
4. **P1 — asynchronous jobs:** make training, evaluation and export durable jobs
   with progress, cancellation, retry and structured error state.
5. **P2 — edge delivery:** implement a cloud-neutral edge agent, artifact manifest,
   device registry and store-and-forward results inspired by AWS DDA.
6. **P2 — commercial operations:** add RBAC, audit events, approval gates, model
   promotion, staged rollout and rollback.
