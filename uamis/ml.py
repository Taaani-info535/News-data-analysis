"""Walk-forward training, selection, calibration and honest validation.
Time order is respected: training rows must have completed outcome windows before the evaluation block starts (purge)."""
from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import Sample, feature_columns
from .models import LABELS

log = logging.getLogger("uamis.ml")
L2I = {l: i for i, l in enumerate(LABELS)}


def candidates(rs: int):
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, class_weight="balanced_subsample",
                                                        random_state=rs, n_jobs=1),
        "grad_boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10,
                                                                random_state=rs),
    }


def to_xy(samples: List[Sample], cols: List[str]):
    X = np.array([[s.features.get(c, 0.0) for c in cols] for s in samples], dtype=float)
    y = np.array([L2I[s.label] for s in samples], dtype=int)
    return X, y


def full_proba(model, X) -> np.ndarray:
    """Probability matrix over all 3 labels in LABELS order (0 for classes unseen in training)."""
    p = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((len(X), len(LABELS)))
    for j, c in enumerate(classes):
        out[:, int(c)] = p[:, j]
    return out


def prior_from(y: np.ndarray) -> np.ndarray:
    c = np.bincount(y, minlength=len(LABELS)).astype(float) + 1.0   # Laplace
    return c / c.sum()


def brier(P: np.ndarray, y: np.ndarray) -> float:
    Y = np.eye(len(LABELS))[y]
    return float(((P - Y) ** 2).sum(axis=1).mean())


def logloss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())


def ece_ovr(P: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    """Mean over classes of one-vs-rest expected calibration error."""
    Y = np.eye(len(LABELS))[y]
    tot = 0.0
    for k in range(len(LABELS)):
        p, t = P[:, k], Y[:, k]
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        e = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                e += m.mean() * abs(p[m].mean() - t[m].mean())
        tot += e
    return tot / len(LABELS)


class Calibrator:
    """Per-class one-vs-rest sigmoid (Platt) on logit(p), fit on out-of-fold predictions; renormalised."""

    def __init__(self):
        self.models: Dict[int, Optional[LogisticRegression]] = {}

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, P: np.ndarray, y: np.ndarray):
        for k in range(len(LABELS)):
            t = (y == k).astype(int)
            if t.sum() < 5 or (1 - t).sum() < 5:
                self.models[k] = None
                continue
            self.models[k] = LogisticRegression(C=1e3, max_iter=200).fit(self._logit(P[:, k]), t)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        out = P.copy()
        for k, m in self.models.items():
            if m is not None:
                out[:, k] = m.predict_proba(self._logit(P[:, k]))[:, 1]
        s = out.sum(axis=1, keepdims=True)
        return out / np.where(s == 0, 1, s)


def walk_forward_oof(samples: List[Sample], cols: List[str], make, n_folds: int, min_train: int = 30):
    """Expanding-window OOF predictions with purging by outcome end time. Returns (P, y, idx_used, prior_P)."""
    times = sorted({s.anchor_ts for s in samples})
    if len(times) < n_folds + 2:
        return None
    start = int(len(times) * 0.4)
    bounds = [times[min(len(times) - 1, start + int((len(times) - start) * k / n_folds))] for k in range(n_folds + 1)]
    Ps, ys, ps = [], [], []
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1] if k < n_folds - 1 else float("inf")
        tr = [s for s in samples if s.anchor_ts < lo and s.end_ts <= lo]
        va = [s for s in samples if lo <= s.anchor_ts < hi]
        if len(tr) < min_train or not va:
            continue
        Xtr, ytr = to_xy(tr, cols)
        if len(set(ytr)) < 2:
            continue
        Xva, yva = to_xy(va, cols)
        m = make().fit(Xtr, ytr)
        Ps.append(full_proba(m, Xva)); ys.append(yva); ps.append(np.tile(prior_from(ytr), (len(va), 1)))
    if not Ps:
        return None
    return np.vstack(Ps), np.concatenate(ys), np.vstack(ps)


def train(samples: List[Sample], stage: str, horizon_s: int, asset_names: List[str], cfg, model_version: str, out_dir: Optional[str] = None) -> Dict:
    """Train/select/calibrate/test. Returns a report dict; saves artifact to out_dir when training succeeded."""
    mlc = cfg.raw["ml"]
    cols = feature_columns(stage, asset_names)
    rep: Dict = {"stage": stage, "horizon_s": horizon_s, "n_samples": len(samples), "status": "NOT_TRAINED", "reason": "", "model_version": model_version}
    if len(samples) < int(mlc["min_train_samples"]) + int(mlc["min_test_samples"]):
        rep["reason"] = f"INSUFFICIENT HISTORICAL DATA: {len(samples)} labelled samples (need >= {int(mlc['min_train_samples']) + int(mlc['min_test_samples'])})"
        return rep
    samples = sorted(samples, key=lambda s: s.anchor_ts)
    times = sorted({s.anchor_ts for s in samples})
    test_start = times[int(len(times) * (1 - float(mlc["test_fraction"])))]
    trainval = [s for s in samples if s.anchor_ts < test_start and s.end_ts <= test_start]
    test = [s for s in samples if s.anchor_ts >= test_start]
    rep.update(n_trainval=len(trainval), n_test=len(test), test_start=int(test_start))
    if len(trainval) < int(mlc["min_train_samples"]) or len(test) < int(mlc["min_test_samples"]):
        rep["reason"] = f"INSUFFICIENT HISTORICAL DATA after time split (trainval={len(trainval)}, test={len(test)})"
        return rep
    rs = int(mlc.get("random_state", 7))
    scores = {}
    oof_store = {}
    for name, make in candidates(rs).items():
        r = walk_forward_oof(trainval, cols, make, int(mlc["n_folds"]))
        if r is None:
            continue
        P, y, PR = r
        scores[name] = {"oof_brier": brier(P, y), "oof_baseline_brier": brier(PR, y), "oof_logloss": logloss(P, y), "n_oof": int(len(y))}
        oof_store[name] = (P, y)
    if not scores:
        rep["reason"] = "walk-forward validation could not run (too few time blocks / classes)"
        return rep
    best = min(scores, key=lambda n: scores[n]["oof_brier"])
    P_oof, y_oof = oof_store[best]
    cal = Calibrator().fit(P_oof, y_oof)
    Xtv, ytv = to_xy(trainval, cols)
    make = candidates(rs)[best]
    model = make().fit(Xtv, ytv)
    Xte, yte = to_xy(test, cols)
    P_raw = full_proba(model, Xte)
    P_cal = cal.transform(P_raw)
    prior = prior_from(ytv)
    P_base = np.tile(prior, (len(yte), 1))
    te = {"brier": brier(P_cal, yte), "brier_uncalibrated": brier(P_raw, yte), "baseline_brier": brier(P_base, yte),
          "logloss": logloss(P_cal, yte), "ece": ece_ovr(P_cal, yte),
          "accuracy": float((P_cal.argmax(1) == yte).mean()), "baseline_accuracy": float((P_base.argmax(1) == yte).mean()), "n": int(len(yte))}
    te["brier_skill"] = 1 - te["brier"] / te["baseline_brier"]
    validated = (te["baseline_brier"] - te["brier"] >= float(mlc["min_brier_improvement"])) and te["ece"] <= float(mlc["max_ece"])
    # global feature importance (permutation on train+val, 3 repeats)
    try:
        pi = permutation_importance(model, Xtv, ytv, n_repeats=3, random_state=rs, scoring="neg_log_loss")
        imp = {c: float(v) for c, v in zip(cols, pi.importances_mean)}
    except Exception:
        imp = {}
    # deployment refit on all history (calibrator and verdict come from the out-of-sample test above)
    Xall, yall = to_xy(samples, cols)
    deploy = make().fit(Xall, yall)
    rep.update(status="VALIDATED" if validated else "EXPERIMENTAL", chosen_model=best, candidates=scores, test=te,
               reason="" if validated else "did not beat prior baseline by required margin and/or calibration error too high on held-out test",
               importance_top=dict(sorted(imp.items(), key=lambda kv: -kv[1])[:8]), columns_n=len(cols))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump({"model": deploy, "calibrator": cal, "cols": cols, "prior": prior.tolist(), "importance": imp, "report": rep,
                     "version": model_version}, os.path.join(out_dir, f"{stage}_{horizon_s}.joblib"))
        with open(os.path.join(out_dir, f"{stage}_{horizon_s}.json"), "w") as f:
            json.dump(rep, f, indent=2, default=str)
    return rep


def load_artifact(model_dir: str, stage: str, horizon_s: int) -> Optional[dict]:
    p = os.path.join(model_dir, f"{stage}_{horizon_s}.joblib")
    if not os.path.exists(p):
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        log.warning("could not load model %s: %s", p, type(e).__name__)
        return None


def predict_proba(art: dict, feats: Dict[str, float]) -> np.ndarray:
    X = np.array([[feats.get(c, 0.0) for c in art["cols"]]], dtype=float)
    return art["calibrator"].transform(full_proba(art["model"], X))[0]
