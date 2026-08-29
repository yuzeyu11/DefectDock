# Release evidence and dependency policy

DefectDock archives machine-readable dependency evidence for both shipped
surfaces. The workflow is defined in
`.github/workflows/release-evidence.yml` and runs for release tags, manual
dispatches, and pull requests that change dependency inputs.

## Archived evidence

The Python job resolves the locked core, training, and CVAT runtime extras
while excluding development-only tools. It archives:

- `requirements-release.txt`: exact release dependency inventory with hashes;
- `sbom.cdx.json`: CycloneDX 1.5 SBOM generated from `uv.lock`;
- `licenses.json`: license metadata produced by `licensecheck`;
- `vulnerabilities.json`: `pip-audit` results for the hashed inventory;
- `SHA256SUMS`: hashes for the evidence files.

The workbench job archives the corresponding production dependency evidence
from `pnpm-lock.yaml`: a CycloneDX 1.6 SBOM, `pnpm licenses` report, `pnpm
audit` report, and evidence hashes. CI artifacts are retained for 90 days;
release management must copy them beside long-lived release binaries and
container digests before distribution.

The container job rebuilds the lightweight image, records its image ID, and
generates a filesystem/package CycloneDX SBOM with Anchore Syft. Anchore Grype
archives one complete vulnerability report and a second report restricted to
findings with an available fix. A fixable high or critical finding fails the
workflow; unfixed upstream findings remain visible and require a documented
release risk decision. The action revisions are pinned exactly like every
other external CI action.

## Enforced policy

- GitHub Actions are pinned to immutable commit SHAs.
- Python and pnpm lock files are mandatory inputs.
- `scripts/check_license_boundary.py` rejects an Ultralytics runtime import or
  declared Python dependency.
- `scripts/check_dependency_licenses.py` rejects the excluded package and any
  AGPL license found by the ecosystem scanners.
- High or critical frontend advisories and any known Python advisory fail the
  release-evidence workflow after the raw report has been uploaded.
- Dependabot opens weekly updates for Python, pnpm, Docker, and GitHub Actions.

Scanner output is evidence, not legal advice. Empty, unknown, custom, model
weight, CUDA/NVIDIA, base-image, or customer-data terms require manual review.
The final review must cover the exact Docker image digest and model/data
combination, not only the source repository.

## Local commands

The workflow commands can be reproduced locally with uv 0.11.13, Python 3.12,
Node 22, and pnpm 11.19.0. Generated evidence belongs under `build/evidence/`
and is intentionally excluded from Git; attach it to the release instead.
