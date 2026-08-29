# Security policy

## Supported versions

Only the latest maintained release branch receives security fixes. Version
`0.1.x` is an alpha engineering baseline and must not be exposed directly to the
public internet.

## Reporting

Report suspected vulnerabilities privately to the repository owner or the
organization's designated security channel. Do not open a public issue containing
credentials, customer images, dataset paths, network addresses, model artifacts
or exploit details.

Include the affected version, reproduction steps, impact, relevant logs with
secrets removed, and a safe contact method. The owner should acknowledge the
report, assess severity, prepare a fix, and coordinate disclosure.

## Deployment baseline

- Terminate TLS at a trusted reverse proxy.
- Add authentication and role-based authorization before multi-user deployment.
- Keep datasets and checkpoints outside the container image and source tree.
- Run API and training workers with least privilege and separate service accounts.
- Restrict CVAT and camera credentials to secret storage.
- Enforce upload size/type limits and scan archives before extraction.
- Record audit events for dataset freeze, training, model approval and deployment.
- Pin dependencies, scan the release image and archive an SBOM.
