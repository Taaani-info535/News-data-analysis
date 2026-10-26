"""Price series access with strict point-in-time semantics.
A candle with open time T and resolution R is only known at T+R (its close time). `close_at(t)` never returns a candle
whose close time is after t, which is the core look-ahead guard."""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TOL_MIN = 600


class PriceSeries:
    def __init__(self, res: int, rows):
        self.res = int(res)
        arr = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], dtype=float).reshape(-1, 5)
        self.ts = arr[:, 0].astype(np.int64)
        self.o, self.h, self.l, self.c = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        self.close_ts = self.ts + self.res
        self._vol_roll: Optional[np.ndarray] = None
        self._vol_roll_key = None

    def __len__(self):
        return len(self.ts)

    def idx_at(self, t: int) -> int:
        """Index of last candle fully closed at or before t (-1 if none)."""
        return int(np.searchsorted(self.close_ts, t, side="right")) - 1

    def close_at(self, t: int, max_age: Optional[int] = None) -> Optional[Tuple[float, int]]:
        i = self.idx_at(t)
        if i < 0:
            return None
        age = t - int(self.close_ts[i])
        if max_age is not None and age > max_age:
            return None
        return float(self.c[i]), int(self.close_ts[i])

    def hourly_returns(self) -> np.ndarray:
        """log returns between consecutive candles; NaN where candles are not adjacent (gaps/weekends)."""
        r = np.full(len(self.c), np.nan)
        if len(self.c) > 1:
            adj = np.diff(self.ts) == self.res
            lr = np.log(self.c[1:] / self.c[:-1])
            r[1:] = np.where(adj, lr, np.nan)
        return r

    def vol_roll(self, window: int, min_n: int) -> np.ndarray:
        if self._vol_roll is None or self._vol_roll_key != (window, min_n):
            s = pd.Series(self.hourly_returns()).rolling(window, min_periods=min_n).std()
            self._vol_roll, self._vol_roll_key = s.to_numpy(), (window, min_n)
        return self._vol_roll


class MarketData:
    """Multi-resolution view for one asset."""

    def __init__(self, series: Dict[int, PriceSeries]):
        self.series = {r: s for r, s in series.items() if len(s) > 0}

    @classmethod
    def from_store(cls, store, asset: str, resolutions=(300, 3600)):
        return cls({r: PriceSeries(r, store.load_candles(asset, r)) for r in resolutions})

    def price_at(self, t: int, max_res: int) -> Optional[Tuple[float, int, int]]:
        """(price, close_ts, res_used) using the finest series with res <= max_res whose latest closed candle is fresh
        enough (tolerance 2*res, min 10 min). None => data missing; callers must skip, never fill."""
        for res in sorted(self.series):
            if res > max_res:
                continue
            hit = self.series[res].close_at(t, max_age=max(TOL_MIN, 2 * res))
            if hit:
                return hit[0], hit[1], res
        return None

    def last_close_ts(self, t: int) -> Optional[Tuple[int, int]]:
        """(latest closed candle close time <= t, res) over finest available series; used for staleness checks."""
        best = None
        for res, s in self.series.items():
            i = s.idx_at(t)
            if i >= 0:
                c = int(s.close_ts[i])
                if best is None or c > best[0]:
                    best = (c, res)
        return best

    def hourly(self) -> Optional[PriceSeries]:
        return self.series.get(3600)
