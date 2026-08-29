# Licensing and dependency boundary

This document is an engineering control, not legal advice.

## Repository policy

DefectDock's original source is currently all-rights-reserved under the root
`LICENSE`. Do not publish the repository under an open-source license until the
copyright owner has chosen a distribution model and reviewed all migrated code,
datasets, model weights, fonts, images and third-party packages.

## Built-in model path

The built-in training and inference adapter uses PyTorch and TorchVision. Their
official repositories publish permissive license terms:

- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [TorchVision BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE)

MMDetection is not a runtime dependency today. Its adapter interface may be
evaluated later; the official project is published under Apache-2.0:

- [MMDetection repository and license](https://github.com/open-mmlab/mmdetection)

## Approved architecture references

The following upstream repositories may be studied and may contribute individual
files only after per-file review:

- [Geti](https://github.com/open-edge-platform/geti) — Apache-2.0 at repository
  level. Its optional Ultralytics path and associated models remain excluded.
- [AWS Defect Detection Application](https://github.com/awslabs/DefectDetectionApplication)
  — Apache-2.0 at repository level. AWS Marketplace algorithms and service terms
  are separate from the source license.
- [Anomalib and Anomalib Studio source](https://github.com/open-edge-platform/anomalib)
  — Apache-2.0 at repository level. Model weights and Windows packaged-app terms
  require separate review.

The detailed scope and adoption procedure are maintained in
[`reference-projects.md`](reference-projects.md). Apache-2.0 code can be used in
a proprietary product only while its applicable copyright, license, change and
NOTICE obligations are preserved. A repository-level license never overrides a
different license attached to a subcomponent, model, dataset or binary asset.

## Explicit exclusion

Ultralytics publishes its open-source package under AGPL-3.0 and separately
offers an enterprise license. DefectDock does not include or depend on that
package, its source code, its training/inference adapter or its pretrained
weights:

- [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

The exclusion applies to source, lock files, build images, downloadable weights,
example commands and dynamically loaded plugins. A future integration can enter
the product only after a documented license decision and a separate distribution
boundary approved by the owner.

## Release gate

Before every externally distributed build:

1. Freeze Python and Node dependency lock files.
2. Run unit tests and `python scripts/check_license_boundary.py`.
3. Run dependency vulnerability and license inventory tools.
4. Generate an SBOM for the exact container or installer.
5. Review all model weights and datasets as separate licensed assets.
6. Archive the configuration, SBOM, notices and scan results with the release.
7. Obtain legal review for the final commercial distribution combination.

Passing the automated check proves only that the known forbidden runtime package
is absent from scanned manifests/imports. It does not prove overall license
compliance.
