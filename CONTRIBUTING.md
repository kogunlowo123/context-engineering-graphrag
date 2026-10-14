# Contributing

## Setup

```bash
uv sync
make check
```

## Requirements for every change

- `make check` passes: ruff, black, mypy `--strict`, bandit, contract drift,
  tests with at least 90% branch coverage.
- New behaviour has tests. Security controls need adversarial cases in
  `tests/security` named after the threat they verify (`test_t<n>_...`).
- Retrieval or ranking changes include an offline evaluation run
  (`POST /v1/collections/{c}/evaluations/retrieval`) on a golden set and the
  before/after MRR and nDCG in the pull request.
- Public API changes are recorded in `CHANGELOG.md`; regenerate contracts with
  `uv run python -m context_engineering_graphrag.contracts`.
- Architecturally significant decisions get an ADR in `docs/architecture/adr/`.
- No credentials, customer data or regulated data in code, tests or fixtures.
  Use synthetic data only.

## Commit messages

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
`chore:`). Breaking changes use `!` and a `BREAKING CHANGE:` footer.

## Review

Changes to `src/context_engineering_graphrag/security`, `ingestion/`,
`policies/`, `configs/` or `deployments/` need approval from a code owner
(see `CODEOWNERS`).
