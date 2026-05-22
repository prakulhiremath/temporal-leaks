# Changelog

## [0.1.0] — 2025-05-23

### Added
- `TemporalAudit` class with sklearn-style `.check()` API
- Three perturbation modes: `noise`, `sign_flip`, `nullify`
- Deterministic perturbation via `random_seed` parameter
- Per-column effect size: `mean_absolute_delta`, `max_delta`, `pct_rows_changed`
- Severity classification: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- Standalone HTML report via `AuditReport.to_html()`
- `@temporal_audit` decorator API
- Primitive feature provenance hints
- Pandas and Polars input support
- GitHub Actions CI across Python 3.9–3.12
- Ruff + Mypy strict configuration
