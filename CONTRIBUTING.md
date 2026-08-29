# Contributing

DefectDock is currently developed as a private commercial project. Contributions
require owner authorization and do not change the repository license.

## Development setup

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
ruff check src tests scripts
python scripts/check_license_boundary.py
```

For the workbench:

```bash
cd apps/web
pnpm install
pnpm run lint
pnpm run build
```

## Change requirements

- Keep framework-specific code behind `src/defectdock/engines` or
  `src/defectdock/inference`.
- Add a regression test for every bug fix and a contract test for API changes.
- Never commit datasets, customer images, credentials, model weights or runs.
- Update `CHANGELOG.md` for operator-visible changes.
- Document a license decision before adding a training framework, model weight,
  dataset, font, image asset or copied source file.
- Do not bypass the license-boundary check.

Commits should be small, explain intent, and use an imperative subject. Pull
requests must include verification evidence, migration notes when schemas change,
and a rollback plan for deployment changes.
