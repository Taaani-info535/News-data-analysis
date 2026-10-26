# Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Alerts say `FOREX DISABLED` | `config/forex_pairs.txt` was edited and now has != 28 pairs, duplicates or invalid names. Restore your exact 28 pairs. |
| Nothing arrives in Telegram | Secrets missing/misnamed; you never pressed Start on the bot; wrong chat id. Run `python -m uamis telegram-test`. Logs show "Telegram rejected message (HTTP 401/400)" without the token. |
| Workflow fails at "Save state" | Settings -> Actions -> General -> Workflow permissions must be **Read and write**. |
| Everything is INSUFFICIENT / STAY AWAY | Expected early. Check `python -m uamis status` for row counts; run collect with backfill; wait for events to accumulate. |
| `market data missing` / `stale` | Market closed (weekend), symbol blocked, or the collect job failing. Check collect workflow logs: failures are listed per `asset@resolution`. |
| Calendar returns nothing | Feed URL changed or blocked; logs show `calendar refresh failed`. Edit `calendar.feed_urls` or import events with `import-events`. |
| Actual never appears | Free feed may omit actuals. Use `python -m uamis set-actual <event_id> "<value>"` (find the id with `python -m uamis events`). |
| Binance HTTP 451/403 | Region restriction; change `binance_base_url` in config or switch the asset's `source` to `yahoo` with symbol `BTC-USD`. |
| Duplicate-looking alerts | Different stage (heads-up vs pre vs post) or changed content after the cooldown (`alerts.cooldown_minutes`). |
| Scheduled run late/skipped | GitHub cron is best-effort; catch-up logic handles it. For urgent events add a manual `Run workflow`. |
| Two runs at once | Shared concurrency group queues them; if both ran, ids/alert keys are idempotent. |
| Reports are mostly WAIT / STAY AWAY | Expected early: no validated history, a release is pending (two-sided), data is stale or the market is closed. The `[status]` tag on each line says which. |
| `invalid symbol` in collect output | A configured symbol failed the shape check (e.g. `EUR USD=X`, lower-case Binance symbol). Fix `config/config.yaml` / `forex_pairs.txt`. |
| Post-news says PENDING | The free feed has not published the actual. Set it with `python -m uamis set-actual <event_id> "<value>"` (id from `python -m uamis events`); the real POST-NEWS report follows. |
| Crypto market line says unavailable | CoinGecko unreachable/rate-limited or fewer than 2 coins have fresh data. Other evidence still works. |
| `Not/AZone` or timezone note in report | `recommend.display_timezone` is not a valid IANA name; reports fall back to UTC. |
| `po3-check` says UNAVAILABLE | No report has been generated yet, the instrument is not configured, or the latest recommendation is older than 3 hours. |
| Corrupt/huge state | Delete branch `uamis-state` to reset (history and ledger are lost); or lower `candles.retain_days`. |
| Tests fail locally | Python >= 3.10; `pip install -r requirements.txt -r requirements-dev.txt`; run from repo root. |
