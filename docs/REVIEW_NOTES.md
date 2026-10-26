# UAMIS review and repair notes

Review date: 2026-10-25.

## Fixes applied

1. **RSS parsing resilience:** added a standard-library XML fallback for RSS/Atom when `feedparser` is not installed. Entries without a parseable publication timestamp are still discarded rather than assigned a guessed time. `feedparser` remains in requirements for normal deployments.
2. **Post-news timing consistency:** historical post-event samples and live post-event predictions now use the same anchor rule: the later of the actual's first observed time rounded down to a five-minute boundary, or the configured post-release delay. This avoids training on an earlier reaction window than the live system would have available.
3. **Temporary missing-data recovery:** a prediction initially logged as `INSUFFICIENT DATA` can be replaced if later runs obtain valid data, as long as its outcome is still pending. Existing non-placeholder forecasts are not overwritten.
4. **Weekend coverage:** the hourly calendar/news tick now runs every day, not only Monday-Friday, which is relevant to crypto and weekend headlines. GitHub scheduled runs remain best-effort.
5. **Documentation:** clarified the difference between machine learning, historical event-study/data mining, keyword rules and optional LLM summarisation; updated limitations and test count.

## Verification performed

- Python compilation: passed (`compileall`).
- Automated suite: **69 passed** (`pytest -q`) in this environment. Tests are unit and mocked-integration tests; they do not establish live provider uptime or real-world predictive accuracy.
- Offline demo: ran successfully; it correctly reported synthetic-data results as meaningless for real performance and kept unvalidated forecasts at `STAY AWAY` / `EXPERIMENTAL` / `INSUFFICIENT DATA`.
- Configuration: loaded successfully; all 28 supplied Forex pairs passed the configuration gate.
- Secret scan: no real Telegram token or API key is included in the project archive.

## Important limitations still present

- **No system can be made a perfect predictor.** The model can be wrong and is not an autonomous trading system.
- The economic calendar source is an unofficial community feed that covers the current week; actual/forecast values can be missing.
- Market data sources (Yahoo Finance, Binance, Forex Factory feed, FRED and Telegram) were not live-tested from this environment.
- News classification is keyword-based and coarse. There is no trained news-sentiment model, and the optional LLM is disabled by default and only summarizes existing text.
- Historical sample counts are event-derived and may be correlated across assets/horizons. A large sample count does not guarantee statistical independence.
- Fresh installations will have little historical event data. Model training may correctly remain `NOT_TRAINED`, `EXPERIMENTAL` or `INSUFFICIENT HISTORICAL DATA` for a long time.
- Free intraday history is limited and proxies are used for gold, silver and indices.
- The default schedule is hourly; GitHub can delay or skip runs, so short-horizon forecasts after a news release may be missed.

## Safe first-run procedure

1. Upload the **contents** of this ZIP to the root of a new GitHub repository.
2. Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` only as GitHub Actions Secrets; never commit them.
3. Enable read/write workflow permissions because the project persists SQLite/model state to `uamis-state`.
4. Run `python -m uamis collect --backfill`, then `python -m uamis tick --dry-run` locally (or run the equivalent Actions workflows).
5. Check `python -m uamis status`, inspect provider failures, and use `python -m uamis telegram-test` only after the secrets are configured.
6. Do not treat outputs as validated until enough real observations have accumulated and the model's out-of-sample evaluation gates pass.
