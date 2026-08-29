# Third-party notices

DefectDock source code is distributed under the repository's proprietary
license. External packages keep their own licenses. The deployment owner must
produce a locked dependency inventory and software bill of materials (SBOM)
for every release.

Key runtime boundaries:

| Component | Role | License | Included in repository |
|---|---|---|---|
| PyTorch | tensor runtime and optimization | BSD-style | No; installed as a package |
| TorchVision | built-in detection models and transforms | BSD-3-Clause | No; installed as a package |
| FastAPI | REST API framework | MIT | No; installed as a package |
| OpenCV | camera/video integration | Apache-2.0 | No; installed as a package |
| CVAT SDK | optional annotation integration | MIT | No; optional package |
| Alembic / SQLAlchemy | SQLite schema migration | MIT | No; installed as packages |
| React / React DOM | workbench runtime | MIT | No; installed as packages |

The built-in training and inference path does not depend on the Ultralytics
Python package, its source code, or its pretrained weights. The normalized
`class cx cy width height` annotation layout remains supported as an
interchange format; a data layout is not a runtime dependency.

Canonical license sources are linked from [docs/licensing.md](docs/licensing.md).
The complete version-specific machine-readable inventory is generated per
release as described in [docs/release-evidence.md](docs/release-evidence.md).

Geti, AWS DDA and Anomalib Studio are currently design references, not bundled
dependencies. They therefore are not listed as distributed third-party
components here. If Apache-2.0 source is later copied or modified, applicable
copyright, license and NOTICE content must be added to the distributed notices
at that time; see [docs/reference-projects.md](docs/reference-projects.md).
