# Free data sources (research notes) - verify before relying on them

Checked while designing (terms and limits change; re-check). **None of these were called live from the build environment.**

| Need | Source used | Cost | Key facts / limits | Risks |
|---|---|---|---|---|
| Scheduled events with forecast/previous | Forex Factory community JSON `nfs.faireconomy.media/ff_calendar_thisweek.json` | free, no key | Current week only; includes forecast/previous; `actual` may or may not be populated | **Unofficial**, terms unclear, can change/disappear; check the site's terms; do not hammer it (job fetches at most every 15 min). Use `set-actual` or `import-events` when absent |
| Official release texts | Federal Reserve, BLS, ECB RSS (config `news.feeds`) | free | Official primary sources; headlines only | Feed URLs can move; verify in a browser |
| FX, gold, silver, indices, DXY candles | Yahoo Finance chart endpoint (unofficial) | free | Intraday history is shallow: 5m ~60 days, 1h ~730 days (as widely documented) | Personal-use terms; not redistributable; futures proxies for gold/silver/indices (`GC=F`, `SI=F`, `NQ=F`, `YM=F`, `ES=F`); occasional gaps/outages. Do not republish the stored candles |
| BTC, ETH candles | Binance public market data (`data-api.binance.vision`) | free, no key | 1000 candles/request, weight-limited | USDT pair as USD proxy; some regions restricted; configurable `binance_base_url` |
| Treasury yields | FRED API (`DGS10`, `DGS2`) | free key | Daily only | Context lines only; not an ML feature (daily data is not point-in-time safe intraday) |
| Crypto market-wide context | CoinGecko public `/global` endpoint | free, no key | total market-cap 24h change and BTC dominance (one call per cycle; dominance 24h change comes from our own snapshots) | rate-limited; optional; failure => that evidence is reported unavailable |
| Crypto candles for extra coins | Binance public klines (BNB, SOL, XRP added) | free | same as BTC/ETH | same restrictions |
| Provider symbol validation | local regex checks (`uamis/symbols.py`) | - | malformed symbols are never requested | shape check only; does not prove a symbol exists at the provider |
| News classification | Rules (keywords) | free | Fully offline | Coarse; misses nuance |
| Optional LLM text polish | any OpenAI-compatible endpoint with a free tier | free tier varies | Off by default; may only rephrase | Never required |

**Attribution/terms:** Keep Yahoo/Binance/FF data for your own analysis; do not publish raw datasets. Credit FRED if you publish yields-derived text.
**Not covered for free:** true spot XAU/XAG, tick data, 1-minute history beyond ~7 days, deep historical consensus forecasts (no free source found), real-time news wires.
**Missing capability workaround:** the system stores what it collects, so history grows over time; you may also import your own historical events CSV.
