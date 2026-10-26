# Honest limitations and cost estimate

## What this is not
No guarantee of profit, no "every move" prediction, no order execution, no investment advice. Probabilities are model outputs on small samples.

## Known limitations
1. **Cold start:** at first there is almost no event history (the free calendar covers only the current week). Expect weeks/months of INSUFFICIENT/EXPERIMENTAL output. `import-events` lets you add your own history.
2. **Shallow intraday data:** Yahoo 5m ~60 days; 1m not used. 15m/1h models need 5m data around events and will train last.
3. **Proxies:** gold/silver/indices use futures symbols; crypto uses USDT pairs; spot spreads/costs are assumptions (`cost_bps`).
4. **Unofficial feeds:** calendar and Yahoo endpoints can break or change terms with no notice; the system degrades to "data missing -> STAY AWAY", never to invented values.
5. **Actuals may be missing** from the free feed; post-news stage then reports "surprise not computable" or uses a manually set actual.
6. **Rules-based news classification** is keyword-based and coarse; unscheduled news gets alerts but no forecast.
7. **Overlapping/correlated samples:** forecasts across assets and horizons for one event are not independent; reported n overstates evidence.
8. **Schedule jitter and post-news timing:** GitHub scheduled runs can start late or be skipped. Post-news training and live forecasts now use the same anchor rule: the later of the first observed actual (rounded down to a 5-minute boundary) and the configured minimum post-release delay. With an hourly tick, short-horizon post forecasts may still be skipped if their outcome window has already elapsed; a 15-minute cron improves timing but uses substantially more Actions minutes.
9. **Calibration with few samples is noisy;** the gates reduce but do not remove false VALIDATED labels. Treat VALIDATED as "not obviously bad out-of-sample".
10. **Storage:** a SQLite file in a git branch. Single-commit force-push avoids history bloat, but the file is re-uploaded each change (tens of MB after a year with many assets). Public repo = public data. No secrets are stored.
11. **Treasury yields** are daily context only. DXY depends on Yahoo's `DX-Y.NYB`.
12. **Not verified live:** live API integrations and Telegram delivery were not exercised in the build sandbox.

## Recommendation-engine limitations
1. **Unvalidated by design:** weights/thresholds are explainable defaults. No backtest exists; the live track record starts with the first reports and the engine is labelled UNVALIDATED until >= 100 scored preferences per horizon justify more.
2. **No market-implied central-bank expectations, positioning or order-flow data** (no reliable free source). "Central-bank expectations" enter only through released rate-decision surprises and coarse official-headline tone.
3. **Tone is keyword-based** and usually unavailable; **crypto-specific news and per-coin fundamentals are not covered**.
4. **Currency strength and price trend are technical evidence** and are only allowed to confirm, never to originate, a preference.
5. **Model/history evidence is inert until history accumulates** (needs trained models or >= 20 comparable past events per category/asset/horizon). When present it genuinely changes the score; when absent it is reported as absent.
6. **Closed-market detection is approximate** (Fri 22:00 to Sun 21:00 UTC); exchange holidays and daylight-saving shifts of futures sessions are not modelled. Provider-side quote delays cannot be detected from candles ("RECENT" means recently stored, not necessarily real-time).
7. **Stage timing depends on GitHub cron:** with the default hourly schedule the 15-minute pre/post checks often fall in a gap and the latest due stage is reported instead. Use a 15-minute cron on a public repo for finer timing; neither behaviour has been observed on a real GitHub run.
8. **Correlated recommendations:** many pairs share the same currency driver, so n in the track record overstates independent evidence.
9. UAMIS does not read your PO3 signals automatically; use `po3-check` or the exported JSON.

## Cost estimate
| Item | Cost |
|---|---|
| GitHub Actions on a public repo | $0 |
| GitHub private repo | $0 within ~2,000 free min/month with the default hourly tick (a 15 min cadence exceeds it) |
| Data (Yahoo, Binance, FF feed, Fed/BLS/ECB RSS, FRED key) | $0 |
| Telegram Bot API | $0 |
| LLM (optional, off) | $0 (not needed) |
| VPS / paid AI / subscriptions | none required |
