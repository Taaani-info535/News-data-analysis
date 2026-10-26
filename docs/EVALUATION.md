# Training, validation and evaluation

## Commands
```bash
python -m uamis train          # builds samples, walk-forward validates, calibrates, writes state/models/*.joblib + *.json + training_summary.json
python -m uamis evaluate       # scores finished live forecasts, writes state/reports/metrics.json, prints summary
python -m uamis status         # shows status of every stage/horizon model
```

## Model status meaning
| Status | Meaning | Decision effect |
|---|---|---|
| NOT_TRAINED | fewer than `min_train_samples + min_test_samples` labelled samples after time split | history study only; usually INSUFFICIENT |
| EXPERIMENTAL | trained but did **not** beat the prior baseline by `min_brier_improvement` and/or calibration error > `max_ece` on the untouched test block | probabilities shown with a warning; **STAY AWAY** |
| VALIDATED | passed both gates out-of-sample | may reach `CONDITIONS ACCEPTABLE (not a trade signal)` if no risk trigger fires |

Passing validation on a few dozen events is weak evidence; the live ledger (`evaluate`) is the real test. The risk filter also switches to STAY AWAY if live Brier
of validated forecasts becomes worse than baseline (`risk.deterioration_*`).

## Recommendation engine validation
`python -m uamis evaluate` also scores the LONG/SHORT preferences that were sent (table `recommendations`) after their horizon, with the volatility-aware sideways threshold frozen at creation, and writes `state/reports/recommendation_metrics.json`.
Metrics: hit rate on decisive (UP/DOWN) outcomes with Wilson 95 % CI, sideways share, mean gross/net return per preference, breakdowns by horizon, instrument and score bucket. A horizon becomes VALIDATED only with >= 100 scored preferences,
hit-rate lower bound > 50 % and positive net mean return (`recommend.validation`). These are not probabilities, so Brier/calibration apply to the ML ledger, not to the score.

## Self-improvement claim
The only "learning" is the scheduled retrain + validation above. Models are replaced every run; nothing adapts between runs, and a retrained model can be worse (it will then be labelled EXPERIMENTAL).

## Metrics reported
Directional accuracy with Wilson 95% CI and coverage, precision/recall for UP/DOWN, multi-class Brier and skill vs baseline, top-class calibration bins, mean gross and cost-adjusted return per directional call,
breakdowns by asset/category/horizon/status/stage/volatility, plus `n` and distinct-event counts. Forecasts around one event are correlated: the effective sample is closer to the event count. Pending forecasts are never scored.

## Important design caveats
- 15m and 1h horizons require 5-minute candles at the release time; free 5m history is only ~60 days at first, so these models stay `NOT_TRAINED` for a long time.
- Post-news training anchors at release+15 min; live runs anchor at the first sighting of the actual (a mismatch that grows if GitHub runs are delayed).
- Event history is built from the weekly feed as it runs: months before there are enough events per category/asset (`study.min_samples = 20`).
