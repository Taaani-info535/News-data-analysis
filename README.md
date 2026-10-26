# Universal AI Market Intelligence System (UAMIS)

Free-first **intelligence and prediction** system for economic news and market reactions. It collects scheduled events and official headlines,
stores market candles, mines history for comparable events, trains small calibrated ML models, labels every forecast honestly, tracks its own
live accuracy, and sends concise Telegram alerts. **It never places trades and never claims to predict every move.**

> **Read this first - what to expect.** On day one the system has almost no history, so nearly every forecast will say
> `INSUFFICIENT HISTORICAL DATA` or `EXPERIMENTAL` and the decision will be **STAY AWAY**. That is the system working correctly. A forecast only
> becomes `VALIDATED` after a model beats a naive baseline on a held-out, time-ordered test set with acceptable calibration (see `docs/EVALUATION.md`).
> Building enough history takes weeks to months of running (details in `docs/LIMITATIONS.md`).

## Your 28 Forex pairs

`config/forex_pairs.txt` contains the 28 pairs you supplied, verbatim and in your order (EURUSD ... EURNZD). They are loaded from that file, never hardcoded.
If the file ever has anything other than exactly 28 valid, unique 6-letter pairs (`EUR/USD` is also accepted), Forex analysis switches off and every report says so; the system never guesses pairs.
**Every main report lists all 28 pairs** (`PAIR | 15M | 1H | 4H | 24H | D`). Pairs unrelated to an event's currency are shown as "not affected", never omitted.
Broker suffixes (`EURUSD.m`) are not supported: use the plain 6-letter name. Prices come from Yahoo symbols `<PAIR>=X` (unverified live).

## Market Direction Recommendation Engine (filter for your PO3 signals)

Besides probability forecasts, UAMIS now produces **LONG PREFERENCE / SHORT PREFERENCE / NEUTRAL / WAIT / STAY AWAY** for the 28 pairs, gold/silver/indices/DXY and the configured liquid crypto (BTC, ETH, BNB, SOL, XRP) at 15m, **1h (primary)**, 4h and 24h.
It is a *fundamental directional filter* for your separate PO3 strategy: it never trades, never touches an account, and never replaces or invalidates a PO3 signal.

- **One consolidated Telegram message per cycle**: currency outlook (common drivers explained once), all 28 pairs grouped by currency (identical WAIT groups compressed), other markets, broad-crypto vs coin-specific bias, timeframe summary with horizon conflicts, overall preference and risk.
- **Pre-news / post-news**: `PRE-NEWS BIAS (not a confirmed prediction)` at configurable checks (default 90/60/15 min before) and `POST-NEWS` (default 15/60 min after). Before a high-importance release, short horizons default to WAIT/STAY AWAY because the outcome is two-sided. If the actual is not available the report says **pending**.
- **Explainable scoring** (-100..+100) from macro surprises, central-bank tone, US yields, risk sentiment, crypto-market context, the existing ML/event-study outputs (only when they exist and with their validation status), and price confirmation. Price can confirm but never create a preference. Stale/missing data, closed markets, extreme volatility, conflicts and imminent events override the number.
- **Honest status**: every report says **UNVALIDATED** until the engine's own live outcomes (>= 100 scored preferences per horizon, credible hit rate, positive net return) justify more. No backtest is claimed.
- **PO3 relationship**: `python -m uamis po3-check EURUSD LONG` -> ALIGNED / CONFLICTING / NO CLEAR PREFERENCE (read-only; also `state/reports/recommendations_latest.json`).
- Full explanation, formula, configuration, credentials and the verified/unverified list: **`docs/RECOMMENDATION_ENGINE.md`**. Example (synthetic data): `docs/sample_output.md`.
- The older per-event probability messages are now folded into this single report by default (`recommend.legacy_event_reports: false`); the underlying ledger logging is unchanged. Set it to `true` to receive both.

## What AI means in this project

- **Machine learning:** scikit-learn Logistic Regression, Random Forest and HistGradientBoosting candidates; expanding-window walk-forward validation, a separate chronological holdout, probability calibration, baseline comparison and risk gates. A model is not called `VALIDATED` unless it clears the configured out-of-sample gates.
- **Data mining:** historical event-study comparisons find similar past events and calculate observed UP/DOWN/SIDEWAYS frequencies and confidence intervals, with outcome windows completed before the forecast time.
- **Rules engine:** economic/news headlines are classified with keyword rules; theoretical macro impacts are explanatory rules, not learned predictions.
- **Recommendation scoring:** an explainable weighted-evidence model (not a trained model). It *uses* the ML and event-study outputs above as evidence only when they exist (and down-weights them when not validated); until history accumulates those inputs are simply absent.
- **Generative AI / LLM:** optional summarisation only, disabled by default, and never required for predictions. Without an LLM key this is still a machine-learning + rules + event-study system; it does not use a free autonomous AI model by default.
- **Important limitation:** this is not a perfect predictor. Free data is incomplete, the economic calendar covers the current week, headline classification is coarse, and live data providers have not been verified from the build environment.

## What is in the box

| Area | Where |
|---|---|
| Config (thresholds, assets, schedules) | `config/config.yaml`, `config/forex_pairs.txt` |
| Data adapters (calendar, prices, news, FRED yields, optional LLM) | `uamis/adapters/` |
| Validation, classification, features/labels | `uamis/validation.py`, `classify.py`, `features.py`, `prices.py` |
| Historical event study, ML, calibration | `uamis/eventstudy.py`, `uamis/ml.py` |
| Prediction, risk filter (STAY AWAY), ledger/evaluation | `uamis/predict.py`, `risk.py`, `evaluate.py`, `storage.py` |
| Recommendation engine (scoring, schedule, report, PO3 check, live validation) | `uamis/recommend/`, `uamis/symbols.py` |
| Telegram + dedupe/cooldown | `uamis/alerts.py` |
| Jobs / CLI | `uamis/pipeline.py`, `uamis/cli.py` |
| GitHub Actions | `.github/workflows/*.yml`, `scripts/state_*.sh` |
| Docs | `docs/` (recommendation engine, architecture, data sources, evaluation, troubleshooting, limitations & costs, sample output, upgrade notes) |

## Setup (about 20 minutes, no cost)

### 1. Create the repository
1. Create a **new GitHub repository** (public gives unlimited free Actions minutes; see "Public vs private" below).
2. Unzip this project and upload **the contents** (the folder that contains `README.md`, `uamis/`, `config/`) to the repository root, then commit to `main`.
3. (Your 28 pairs are already in `config/forex_pairs.txt`; change it only if your scanner list changes.)

### 2. Create the Telegram bot
1. In Telegram open **@BotFather** -> `/newbot` -> choose a name and username -> copy the **token** (looks like `123456:ABC...`).
2. Open a chat with your new bot and press **Start** (a bot cannot message you until you do).
3. Get your chat id: open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser after sending the bot any message; find `"chat":{"id":123456789,...}`. For a group, add the bot to the group, send a message, and use the (negative) group id.

### 3. Add GitHub Secrets
Repository -> **Settings -> Secrets and variables -> Actions -> New repository secret**:

| Secret | Required | Value |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | token from BotFather |
| `TELEGRAM_CHAT_ID` | yes | your chat id |
| `FRED_API_KEY` | optional | free key from fred.stlouisfed.org (adds Treasury-yield context lines) |

Never put tokens in files, issues, or logs. The code reads them only from environment variables and never prints them.

### 4. Allow workflows to write the state branch
Repository -> **Settings -> Actions -> General -> Workflow permissions -> Read and write permissions** -> Save.

### 5. First run
1. **Actions** tab -> enable workflows if prompted.
2. Run **"UAMIS collect prices"** manually (`Run workflow`) with **backfill = true**. This downloads the maximum free history and stores it.
3. Run **"UAMIS events tick (alerts)"** once. It loads the calendar, creates the state branch `uamis-state`, and sends alerts for qualifying events.
4. Verify Telegram delivery: run locally `python -m uamis telegram-test` (see below), or temporarily set a calendar event within the window.

From then on the schedules run by themselves:

| Workflow | Schedule (UTC) | Purpose |
|---|---|---|
| collect | hourly at :07 | store new candles |
| tick | every 15 minutes, every day | refresh calendar+news; heads-up (24h); runs the recommendation cycle (PRE-NEWS BIAS / POST-NEWS stages, change-triggered scans) and logs the probability ledger. The 15-minute cadence is needed to catch the configured 90/60/15-minute pre-news and 15/60-minute post-news stages more reliably. Weekend runs also matter for crypto and breaking headlines. GitHub scheduled runs are best-effort and may be delayed or skipped. |
| evaluate | daily 00:20 (+weekly summary Sunday) | score finished forecasts, accuracy report |
| train | Sunday 03:30 | retrain, walk-forward validate, calibrate, re-label models |
| tests | on push | unit/integration tests |

GitHub scheduled runs are **best-effort**: they can start minutes late or be skipped under load. The code uses catch-up windows and idempotent
state, so a late or repeated run does not duplicate alerts, and a missed run is picked up next time.

### 6. Optional: run locally
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # fill in; then export the variables (e.g. `set -a; source .env; set +a`)
python -m pytest -q                       # run the tests
python -m uamis demo                      # offline end-to-end check on SYNTHETIC data (~1-2 min)
python -m uamis status                    # forex gate, row counts, model status
python -m uamis collect --backfill        # live download (needs internet)
python -m uamis tick --dry-run            # prints alerts instead of sending
python -m uamis telegram-test             # sends one test message
python -m uamis train && python -m uamis evaluate
python -m uamis import-events my_history.csv    # optional historical events (name,currency,datetime_utc,importance,forecast,previous,actual)
python -m uamis recommend [--send]              # print (and optionally send) the consolidated direction report now
python -m uamis po3-check EURUSD LONG           # compare your PO3 signal side with the latest UAMIS preference (read-only)
python -m uamis events                          # list stored events with their ids
python -m uamis set-actual <event_id> "3.2%"    # manual actual if the free feed lacks it
```

## Reading an alert
The main message is the **MARKET DIRECTION REPORT** (see `docs/RECOMMENDATION_ENGINE.md`); codes per horizon: L = LONG PREFERENCE, S = SHORT PREFERENCE, W = WAIT, X = STAY AWAY, - = unavailable.
For the probability ledger (still logged, optionally also sent):
`Probability` is shown only as a ratio of three outcomes (UP/DOWN/SIDEWAYS) and is flagged `NOT calibrated-validated` unless the model passed its
held-out test. `Decision` is from an **independent risk filter**: `STAY AWAY`, `WAIT FOR RELEASE`, or `CONDITIONS ACCEPTABLE (not a trade signal)`.
A direction is a forecast, not a recommendation. See `docs/sample_output.md` for the format (synthetic data).

## Public vs private repository
- Public: unlimited free Actions minutes, but anyone can read your **state branch** (predictions, stored candles/events - no secrets). Fine for a hobby project.
- Private: your data is hidden, but free Actions minutes are limited (about 2,000/month). The hourly tick job uses roughly 1-2 min/run (about 30-45 hours/month), which fits; a 15-minute cadence would not.
- The state branch is force-pushed as a single commit each time so repository history does not grow; the DB itself still grows slowly (see limitations).

## Tests (what was actually run)
`python -m pytest -q` -> **151 tests (69 pre-existing + 82 new), all passing** in the build environment (Python 3.12, also re-run in a fresh virtual environment from `requirements.txt`). They are **unit and mocked-integration tests**.
**No live API call was made during development** (the build sandbox blocks the data hosts), so live behaviour of Yahoo, Binance, the Forex Factory feed,
RSS feeds, FRED, CoinGecko and Telegram is **unverified**, as is real GitHub Actions timing and the predictive value of the recommendations. Run `collect`, `tick --dry-run` and `telegram-test` yourself and use `docs/TROUBLESHOOTING.md` if a source fails.

## Honest summary
Free sources, shallow intraday history, noisy markets, tiny samples. The system is built to say "I don't know" loudly and to measure itself, not to impress.
Full list of limits and costs: `docs/LIMITATIONS.md`.
