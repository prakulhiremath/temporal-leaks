# Contributing

1. Fork the repo
2. `git checkout -b feat/your-feature`
3. `pip install -e ".[dev]"`
4. Make changes, add tests
5. `ruff check . && mypy temporal_leaks/ && pytest`
6. Open a PR

All PRs must pass CI. Please keep commits atomic.
