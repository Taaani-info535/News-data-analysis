# Sample output (SYNTHETIC DEMO)

> Produced by `python -m uamis demo`. The 28 pair **names** are yours (from `config/forex_pairs.txt`) but every price and event is a **random number** generated locally (the demo also restricts crypto/metals/indices to four instruments to keep it fast). Nothing here describes real-market performance or real news. It shows the report format, the consolidated message, and that unproven calls are held at WAIT / STAY AWAY.

```text
*** SYNTHETIC DEMO DATA - random numbers, NOT market data. Results mean nothing about real performance. ***
Training on synthetic history (a minute or so)...
  post_14400: NOT_TRAINED samples=134 INSUFFICIENT HISTORICAL DATA: 134 labelled samples (need >= 190)
  post_3600: NOT_TRAINED samples=134 INSUFFICIENT HISTORICAL DATA: 134 labelled samples (need >= 190)
  post_86400: NOT_TRAINED samples=104 INSUFFICIENT HISTORICAL DATA: 104 labelled samples (need >= 190)
  post_900: NOT_TRAINED samples=134 INSUFFICIENT HISTORICAL DATA: 134 labelled samples (need >= 190)
  pre_14400: EXPERIMENTAL samples=1053 did not beat prior baseline by required margin and/or calibration error too high on held-out test
  pre_3600: NOT_TRAINED samples=134 INSUFFICIENT HISTORICAL DATA: 134 labelled samples (need >= 190)
  pre_86400: EXPERIMENTAL samples=811 did not beat prior baseline by required margin and/or calibration error too high on held-out test
  pre_900: NOT_TRAINED samples=134 INSUFFICIENT HISTORICAL DATA: 134 labelled samples (need >= 190)

===== tick (dry-run): heads-up alert + ONE consolidated market-direction report =====

--- [DRY RUN] heads_up 6bcf5a4c59a07079 ---
AI MARKET INTELLIGENCE
UPCOMING EVENT: USD CPI y/y
Release: 2027-01-15 10:31 UTC (in 0h 30m)
Importance: ***  Forecast: 3.1%  Previous: 3.0%
Possibly affected: AUDUSD, BTCUSD, DXY, EURUSD, GBPUSD, NAS100, NZDUSD, USDCAD, USDCHF, USDJPY, XAUUSD
A pre-event analysis follows closer to the release. No forecast of the result is made.

--- [DRY RUN] rec_event rec:9314c20fa189 ---
UAMIS | MARKET DIRECTION REPORT
Analysis: PRE-NEWS BIAS (not a confirmed prediction) + POST-NEWS
Event: USD Non-Farm Employment Change — 2027-01-15 09:41 UTC
  Forecast 180K | Previous 170K | Actual 230K
Event: USD CPI y/y — 2027-01-15 10:31 UTC
  Forecast 3.1% | Previous 3.0% | Actual: not released
Updated: 2027-01-15 10:01 UTC | Data: 32 RECENT
Status: UNVALIDATED — default thresholds, not yet proven by live outcomes. Use as a cautious filter only.

CURRENCY OUTLOOK (macro surprises, tone, USD yields)
USD: BULLISH (+35, MODERATE) — USD Non-Farm Employment Change: actual 230K vs forecast 180K -> hawkish/supports USD (z-score implausible (314028 sigma,
Neutral / no recent data: EUR, GBP, JPY, AUD, NZD, CAD, CHF
USD-bullish expressions (1h): none | USD-bearish: none
SCENARIOS (textbook theory, not a forecast): if USD CPI y/y comes ABOVE forecast -> up-leaning: USDJPY, USDCHF, USDCAD; down-leaning: EURUSD, GBPUSD, AUDUSD, NZDUSD. BELOW forecast -> reverse. Actual reaction may differ.

FOREX — 28 pairs  (code per horizon: L=LONG PREF, S=SHORT PREF, W=WAIT, X=STAY AWAY, -=unavailable)
USD PAIRS
  EURUSD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  GBPUSD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  USDJPY — WAIT (1h) [event risk] | 15m W · 4h X · 24h X
  USDCHF — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  USDCAD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  AUDUSD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  NZDUSD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
EUR CROSSES
  EURJPY — WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
  EURGBP — WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
  EURCHF — STAY AWAY (1h) [extreme volatility] | 15m X · 4h X · 24h X
  EURCAD — WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
  EURAUD — WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
  EURNZD — WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
GBP CROSSES: GBPJPY, GBPCAD, GBPCHF, GBPAUD, GBPNZD — all WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
JPY CROSSES: AUDJPY, NZDJPY, CADJPY, CHFJPY — all WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
AUD CROSSES: AUDCAD, AUDNZD, AUDCHF — all WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
NZD CROSSES: NZDCAD, NZDCHF — all WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W
CAD CROSSES
  CADCHF — WAIT (1h) [insufficient evidence] | 15m W · 4h W · 24h W

OTHER MARKETS
  XAUUSD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  DXY — WAIT (1h) [event risk] | 15m W · 4h W · 24h W
  NAS100 — WAIT (1h) [event risk] | 15m W · 4h W · 24h W

CRYPTO OUTLOOK
Broad crypto market: UNAVAILABLE — crypto market breadth/market-cap unavailable
  BTCUSD — WAIT (1h) [event risk] | 15m W · 4h W · 24h W

TIMEFRAME SUMMARY
15m: L0 S0 W31 X1 | USD mixed/none (0 vs 0)
1h: L0 S0 W31 X1 | USD mixed/none (0 vs 0)
4h: L0 S0 W30 X2 | USD mixed/none (0 vs 0)
24h: L0 S0 W30 X2 | USD mixed/none (0 vs 0)

OVERALL PREFERENCE (1h, count of 0 LONG vs 0 SHORT preferences; offsetting pair constructions can cancel - read the USD line): WAIT
RISK: MODERATE — USD CPI y/y in 30 min (two-sided outcome); extreme volatility in 1 instrument(s)

Directional filter for your own PO3 signals only. UAMIS does not trade, and a preference is not a guaranteed signal. Not financial advice.

tick result: {'calendar': {'skipped': True}, 'heads_up': 1, 'pre': 0, 'post': 0, 'skipped_dupes': 0, 'recommend': {'enabled': True, 'sent': False, 'mode': 'event', 'stored': 0, 'result': 'dry_run', 'stages': ['post_15', 'pre_60'], 'n_directional': 0}}

===== evaluation =====
ACCURACY SUMMARY
No evaluated predictions yet (pending: 88). Nothing to report - no accuracy is claimed.
*** SYNTHETIC DEMO DATA - random numbers, NOT market data. Results mean nothing about real performance. ***
```
