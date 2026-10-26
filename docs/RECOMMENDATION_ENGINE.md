# Fundamental Market Direction Recommendation Engine

A **directional filter** for your own, separate PO3 signals. For each instrument and horizon it answers one question:
*does the fundamental environment favour price going up, going down, or is there no clear/safe preference?*
It never trades, never connects to a broker or trading account, and never generates, edits or replaces PO3 signals.

## Outputs

| Label | Meaning for you |
|---|---|
| **LONG PREFERENCE** | Fundamentals + confirming evidence lean toward price rising. Give more weight to a PO3 LONG; it does not make a SHORT invalid. |
| **SHORT PREFERENCE** | Mirror image. |
| **NEUTRAL / WAIT** | No clear preference: weak/conflicting/thin evidence, market closed, a release is pending, or no fundamental driver. |
| **STAY AWAY** | Conditions are unsuitable: data missing or stale, extreme volatility, or an imminent release / post-release spike. |

Every recommendation carries: instrument, direction, score (-100..+100), confidence (LOW / MODERATE / HIGH), supporting evidence, contradicting evidence,
risk factors, data timestamp, freshness (RECENT / DELAYED / STALE / CLOSED / UNAVAILABLE), relevant event, horizon, stage and status.
Machine-readable copy: `state/reports/recommendations_latest.json` (on the `uamis-state` branch).

## How the score is calculated (explainable, no black box)

For instrument *I* and horizon *h* the engine builds a list of **evidence items**. Each is a number in [-1, +1] (positive = bullish for the instrument's price) with a nominal weight (config `recommend.weights`, different per horizon):

| Evidence | Kind | Source | Notes |
|---|---|---|---|
| `macro` | fundamental | released calendar surprises (actual vs forecast) | per currency, recency-decayed (half-life 24 h), weighted by category and importance. Direction is **indicator-specific** (higher unemployment is dovish; higher CPI is hawkish). Surprise size is z-scored against that indicator's own history when >= 8 past surprises exist, otherwise sign-only at half strength. A previous-value *revision* is detected from your own stored history. |
| `tone` | fundamental | official central-bank headlines (keywords) | coarse; usually unavailable because most titles are neutral |
| `rates` | fundamental | US 10Y/2Y change over 5 daily observations (FRED, optional key) | daily and delayed; USD pairs / gold / indices / crypto only |
| `risk` | fundamental | equity-index futures 24 h trend x per-currency risk-beta | AUD/NZD risk-on; JPY/CHF safe-haven; crypto risk-on |
| `cryptomkt` | fundamental | breadth across liquid coins, total market-cap change, BTC dominance (CoinGecko, optional) | crypto only |
| `model` | model | the existing ML / history forecast in the ledger | counts fully only if that model is VALIDATED; otherwise down-weighted x0.25; excluded if live performance deteriorated; absent when no forecast exists |
| `reaction` | model | how the pair moved after comparable past surprises (same category, same surprise direction) | needs >= 20 comparable events, otherwise reported as INSUFFICIENT HISTORICAL DATA |
| `price` | price | volatility-normalised pair trend | confirms/contradicts only |
| `strength` | price | currency-strength meter solved across all pairs | separates the **common currency** component from **pair-specific** divergence |

```
score = 100 * shrink * sum(weight_i * value_i) / sum(weight_i over AVAILABLE evidence)
shrink = min(1, coverage / full_coverage)        # coverage = available weight / applicable weight
```
Evidence that is *not applicable* (e.g. US yields for EURGBP) is dropped. Evidence that *is* applicable but missing shrinks the score toward 0, so one strong driver cannot masquerade as broad confirmation.

### Decision rules (in order; any can override the number)
1. No market data -> **STAY AWAY** (UNAVAILABLE). Weekly market closure -> **WAIT** (MARKET CLOSED). Stale data (> `candles.max_staleness_seconds`) -> **STAY AWAY**.
2. High-impact release affecting the instrument within `blackout_pre_minutes` (default 15) or within `blackout_post_minutes` (default 10) after it -> **STAY AWAY** for short horizons (15m/1h).
3. Release happened but the actual is not available -> **WAIT**, status PENDING, for every horizon.
4. High-importance release pending (default window 90 min) -> **WAIT** for short horizons (the outcome is two-sided). 4h/24h may still show a **PRE-NEWS BIAS** with a stricter score (x1.5) and capped confidence. Disable with `pre_news_two_sided_wait: false`.
5. Hourly volatility above the `risk.vol_pct_high` percentile -> **STAY AWAY**.
6. Coverage < `min_coverage` -> **WAIT**. |score| < `direction_score` (25) -> **WAIT**.
7. Non-price evidence must reach `fundamental_min` and agree in sign: **price momentum alone never creates a preference**.
8. Opposing evidence mass above `max_conflict` (30 %) -> **WAIT** (CONFLICTING).
Otherwise LONG/SHORT PREFERENCE.

**Confidence:** LOW by default; MODERATE needs |score| >= threshold, >= 65 % agreement and >= 50 % coverage; HIGH additionally needs |score| >= 50, >= 80 % agreement, >= 70 % coverage, **live validation of the engine for that horizon** and RECENT data. Unvalidated logic is capped at MODERATE; delayed data -> LOW; PRE-NEWS BIAS is capped at MODERATE (LOW for short horizons).

### Why the thresholds are not "calibrated" yet
The defaults are explainable starting points, not fitted values. The engine logs every LONG/SHORT preference it sends and scores it after its horizon with the same volatility-aware sideways threshold the prediction ledger uses
(`python -m uamis evaluate`). A horizon only becomes **VALIDATED** when it has >= 100 scored preferences, the Wilson lower bound of the hit rate on decisive (UP/DOWN) outcomes exceeds 50 %, and the mean return net of assumed costs is positive.
Until then every report says **UNVALIDATED**. No backtest of the recommendation logic is provided (the free feed has no deep history); track record starts accruing from the first live reports.

## Horizons
15m (needs 5-minute candles), **1h (primary, for your PO3 filter)**, 4h, 24h are scored separately with different weights; conflicts are shown (`Horizon conflicts: EURUSD (1h S vs 24h L)`). A bullish 24h outlook never implies every 1h LONG is acceptable.

## Currency groups
Each pair is shown once, grouped by its strongest shared currency (priority USD, EUR, GBP, JPY, AUD, NZD, CAD, CHF). The common driver (e.g. USD) is explained once in the currency outlook;
pair lines are then individual, and identical all-WAIT groups are compressed to a single line. Pairs sharing a currency are **not** assumed to move together: EURUSD and GBPUSD get separate scores, and a USD release does not drive crosses without a USD leg (EURGBP etc.).

## Crypto
BTC, ETH, BNB, SOL, XRP (Binance USDT pairs; edit `config.yaml`) are scored with USD-macro (inverse, textbook), US yields, risk sentiment, **broad crypto-market** evidence (breadth, total market cap, dominance) and the coin's own trend. The report shows the broad market bias separately and flags coins whose own trend diverges from it.
Crypto-specific news (listings, hacks, upgrades) and per-coin fundamentals are **not covered** (no reliable free source).

## Scheduling
`python -m uamis tick` (scheduled every 15 minutes by `.github/workflows/tick.yml`) runs the cycle. This cadence gives the configured event stages a chance to run separately; GitHub Actions schedules remain best-effort and can be delayed or skipped. Per high-impact event, stages are `pre_90`, `pre_60`, `pre_15` before and `post_15`, `post_60` after (config `pre_checks_minutes`, `post_checks_minutes`).
GitHub cron is best-effort, so a stage is *due* once its time has passed and its window is open; if several are due at once only the **latest** is reported and earlier ones are marked superseded.
With the default hourly cron the 15-minute stages are often skipped; for finer timing on a public repo change the cron in `.github/workflows/tick.yml` to `*/15`. One consolidated message is sent per cycle (a pre- and a post-news event due together share one message).
Stages are marked processed only after a successful send, so a failed delivery is retried and a restart never repeats a report. Between events a **scan** report is sent only when a 1h/4h direction class or a currency outlook changed (WAIT and STAY AWAY count as the same class), at most once per `scan.min_interval_minutes`.

## Relationship with your PO3 strategy
```bash
python -m uamis po3-check EURUSD LONG
```
Prints DIRECTIONALLY ALIGNED / CONFLICTING / NO CLEAR FUNDAMENTAL PREFERENCE / STAY AWAY using the latest stored 1h preference (refuses if it is older than 3 h). Your own tooling can read `recommendations_latest.json` instead. A conflict never invalidates your PO3 signal.
UAMIS does not read your PO3 signals, place orders, or touch any account (a test scans the source for order/broker code and for unexpected network hosts).

## Configuration (`config/config.yaml` -> `recommend:`)
| Key | Default | Meaning |
|---|---|---|
| `enabled` | true | master switch |
| `legacy_event_reports` | false | also send the older per-event probability messages (ledger logging is unaffected) |
| `display_timezone` | UTC | IANA name (e.g. `Asia/Karachi`); reports always show UTC plus local time |
| `pre_checks_minutes` / `post_checks_minutes` | [90,60,15] / [15,60] | stages |
| `blackout_pre_minutes` / `blackout_post_minutes` | 15 / 10 | spike windows |
| `pre_news_two_sided_wait` | true | WAIT for short horizons before a high-importance release |
| `alert_min_importance` | 3 | events that trigger stage reports |
| `lookback_hours`, `halflife_hours` | 96, 24 | memory of released surprises |
| `thresholds.*` | see file | score band, strong score, fundamental minimum, coverage, conflict |
| `weights.*` | see file | evidence weights per group and horizon (each sums to 1.0; tested) |
| `risk_beta.*` | see file | coarse risk-on/off sensitivities |
| `validation.*` | 100 / 0.5 | live-validation gates |
| `scan.*` | true / 60 | change-triggered scan reports |
| `crypto.coingecko` | true | optional global market data |

## Credentials / what needs what
| Feature | Needs |
|---|---|
| Everything core | nothing (free public endpoints) |
| Telegram delivery | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (GitHub Secrets) |
| US yields evidence | `FRED_API_KEY` (free; optional; otherwise that evidence is "unavailable") |
| Crypto market-cap / dominance | none (CoinGecko public endpoint, rate-limited; optional) |
| LLM text polish | optional, off, not used by this engine |

## Verified vs not verified
**Verified by automated tests (mocked data, 100+ new assertions):** scoring and every override, bullish/bearish/neutral/conflicting cases, stale/missing/closed handling, pre/post stage logic with catch-up, duplicate and restart safety, failed-send retry, grouping and the all-28-pairs report, crypto context, 429/rate-limit handling (mocked), timezone/DST conversion, symbol validation, live-validation gates, no order/broker code.
**NOT verified:** any live provider (Yahoo, Binance, Forex Factory feed, RSS, FRED, CoinGecko), real Telegram delivery, GitHub Actions timing on your repository, and the predictive value of the recommendations (no live track record yet).
