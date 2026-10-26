# Architecture

```
GitHub Actions (cron) -> python -m uamis <job> -> SQLite state (branch uamis-state) -> Telegram
```

**Adapters** (`uamis/adapters`) fetch raw data: calendar (Forex Factory community JSON), prices (Yahoo chart endpoint, Binance public klines),
news (official RSS), FRED yields (optional), LLM (optional, off by default). A failing adapter is logged and skipped; fallback never invents data.

**Validation** (`validation.py`): candles with NaN/non-positive prices, high<low, future timestamps or duplicates are dropped. Gaps are left as gaps.

**Storage** (`storage.py`): one SQLite file. Tables: `candles`, `events`, `news`, `predictions` (the ledger), `alerts` (dedupe), `kv`.
Upserts never overwrite a known actual with a blank. Prediction ids are deterministic (event+asset+stage+horizon) so re-runs cannot duplicate them.

**Point-in-time rule** (`prices.py`): a candle opened at T with resolution R is known only at T+R. `close_at(t)` never returns a candle that closes after t.
Missing/too-old data returns `None` and the sample is skipped - nothing is filled or interpolated. A unit test tampers with all future prices and checks features are unchanged.

**Two stages**
- `pre`: features at `t0 - 60 min` (config); outcome window `t0 -> t0+h`.
- `post`: features at `t0 + 15 min` (training) / when the actual was first seen (live), including surprise and the initial reaction; outcome window `anchor -> anchor+h`.

**Labels**: log return r over the window. `thr = max(k * sigma_h, cost)`, `sigma_h` = hourly volatility over the 24 h before the feature time x sqrt(h/1h), `k=0.5`, cost = per-asset assumed bps.
UP if r>thr, DOWN if r<-thr, else SIDEWAYS. `thr` is frozen into each ledger row.

**Event study** (`eventstudy.py`): same category, same asset/horizon; nearest neighbours on prior trend, volatility percentile, forecast-vs-previous (+ surprise and initial reaction for post).
Only events whose outcome completed before the target's feature time. Reports counts, frequencies, Wilson 95% intervals; below `min_samples` -> `INSUFFICIENT HISTORICAL DATA`.

**ML** (`ml.py`): candidates LogisticRegression, RandomForest, HistGradientBoosting (all scikit-learn). Chronological split; walk-forward expanding folds with purging
(training rows' outcome must end before the evaluation block). Winner chosen on out-of-fold Brier; per-class sigmoid calibration fit on out-of-fold predictions; final check on an untouched last 20 %.
`VALIDATED` only if test Brier beats the prior baseline by `min_brier_improvement` and calibration error <= `max_ece`. Otherwise `EXPERIMENTAL`. Too little data -> not trained.

**Risk filter** (`risk.py`): independent of the model. Hard STAY AWAY triggers: missing/stale data, insufficient history, unvalidated model, low top probability, ML-vs-history conflict,
extreme volatility percentile, live deterioration vs baseline, late analysis. Thresholds are config defaults, not universal truths.

**Ledger and evaluation** (`evaluate.py`): forecasts are scored only after `anchor + horizon + grace`. Metrics: directional accuracy with Wilson CI, precision/recall, Brier and skill vs baseline,
top-class calibration bins, gross vs cost-adjusted mean return, breakdowns by asset/category/horizon/status/stage/volatility, with n and distinct-event counts.

**Recommendation engine** (`uamis/recommend/`): `evidence.py` (macro surprises with decay/revision detection, tone, yields, risk sentiment, price z-scores, currency-strength meter),
`crypto.py` (breadth, CoinGecko market cap/dominance), `engine.py` (pure `combine`/`decide` + `Engine` that assembles evidence per instrument and horizon), `schedule.py` (pre/post stages with catch-up),
`report.py` (one grouped message), `cycle.py` (orchestration, dedupe, ledger logging, JSON export), `validation.py` (live scoring of the preferences themselves), `po3.py` (read-only alignment label).
It reads only stored data (candles, events, news, prediction ledger). `symbols.py` validates provider symbols before any request. Table `recommendations` stores LONG/SHORT preferences for later scoring.

**Alerts** (`alerts.py`): plain text, <=3,500 chars/chunk. Key = event+stage+content digest. Same digest -> never resent; changed content inside the cooldown -> suppressed; recorded only after successful delivery.
