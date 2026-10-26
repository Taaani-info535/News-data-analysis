# Upgrade notes: Fundamental Market Direction Recommendation Engine

Base: your uploaded `UAMIS_fixed.zip` (69 tests passing, including the review fixes listed in `REVIEW_NOTES.md`). Those fixes were kept; nothing was rebuilt from scratch.

## Added
- `uamis/recommend/` (models, evidence, crypto, engine, schedule, report, cycle, validation, po3), `uamis/symbols.py`, `uamis/adapters/coingecko.py`.
- Config: `recommend:` section; `class:` tags on non-forex assets; BNB/SOL/XRP (Binance) alongside BTC/ETH.
- Storage: `recommendations` table. CLI: `recommend`, `po3-check`; `status` validates symbols; `evaluate` also scores recommendations.
- Tests: `test_recommend_core.py`, `test_recommend_flow.py` (see README for the count).
- Docs: `RECOMMENDATION_ENGINE.md`, this file, updates to README/ARCHITECTURE/DATA_SOURCES/LIMITATIONS/TROUBLESHOOTING/EVALUATION, regenerated synthetic `sample_output.md`.
- `requirements.txt`: `tzdata` (timezone database on Windows).

## Changed behaviour (deliberate, reversible)
- **One message per cycle:** the older per-event PRE-NEWS/POST-NEWS probability messages are no longer sent by default (`recommend.legacy_event_reports: false`); the probability ledger is still written and feeds the recommendation as evidence. Set the flag to `true` for both. Heads-up and breaking-news alerts are unchanged.
- `collect_prices` rejects malformed symbols before any request and accepts an optional `resolutions` filter; `alerts.dispatch` accepts an optional per-call cooldown; `eventstudy.scenario_stats` accepts `exclude_event_id`. All backward compatible.
- Six existing integration tests now enable `legacy_event_reports` and match the exact legacy message headers ("Stage: PRE-NEWS"), because the new report also contains the words PRE-NEWS/POST-NEWS. Their assertions are otherwise unchanged.

## Defects found and fixed while building (kept as regression tests)
- Scan reports re-sent the same picture minutes after an event report -> every sent report is now the baseline for "meaningful change".
- A revision detector fed by mismatched units produced bogus "revisions" -> plausibility bound added.
- Surprise z-scores from unit-mismatched history exploded (e.g. 300000 sigma) -> >8 sigma falls back to sign-only and says so.
- Default conflict gate (0.45) could never trigger under a 25-point score band -> default 0.30.
- Symbol validator initially rejected Yahoo's real `DX-Y.NYB` -> pattern fixed.

## Not done / not verifiable here
- No live API or Telegram calls (sandbox blocks the hosts); no GitHub Actions run; no backtest of the recommendation logic (no deep free history); no market-implied rate expectations, positioning or crypto-specific news.
- Your specification text ended mid-sentence in section 16 ("Which features hav..."); the remaining documentation request is answered by the "Verified vs not verified" lists in `docs/RECOMMENDATION_ENGINE.md` and the README.
