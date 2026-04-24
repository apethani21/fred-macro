"""M4: Recession prediction via logistic regression (Estrella-Mishkin style).

Uses FRED-available predictors (yield curve inversions, HY spreads, VIX, lending
standards) to estimate the probability of NBER recession onset within the next 4
quarters. Follows the binary classification approach from the methods library.

All series loaded from local parquet — no live API calls here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.analytics.data import load_series


# Predictors used in the model. Each tuple: (series_id, human_label, lag_months)
# Lag is how many months prior the predictor is measured relative to outcome.
# 4-quarter-ahead outcome means we're predicting recession onset 12 months out.
_PREDICTORS: list[tuple[str, str, int]] = [
    ("T10Y3M",      "3M10Y inversion",      1),
    ("T10Y2Y",      "2S10S slope",          1),
    ("BAMLH0A0HYM2","HY OAS",               1),
    ("VIXCLS",      "VIX",                  1),
]

# NBER recession indicator
_RECESSION_SERIES = "USREC"

# Horizon: predict recession onset within this many quarters
_HORIZON_QUARTERS = 4


@dataclass
class RecessionLogitResult:
    current_probability: float          # model-implied recession prob today
    six_month_ago_probability: float    # for comparison
    auroc: float                        # in-sample AUROC (full history)
    coefficients: dict[str, float]      # standardised logit coefficients
    current_predictors: dict[str, float]# current values of each predictor
    n_obs: int
    n_recessions: int
    history: pd.Series                  # full predicted probability time series
    feature_names: list[str]
    fit_ok: bool = True
    error: str = ""


def recession_prediction_logit(
    horizon_quarters: int = _HORIZON_QUARTERS,
) -> RecessionLogitResult:
    """Fit a logistic model predicting NBER recession onset.

    Returns a RecessionLogitResult with current probability, AUROC, and
    coefficient signs/magnitudes.
    """
    # Load NBER recession flag at monthly frequency
    usrec = load_series(_RECESSION_SERIES)
    if usrec is None or usrec.empty:
        return RecessionLogitResult(
            current_probability=float("nan"),
            six_month_ago_probability=float("nan"),
            auroc=float("nan"),
            coefficients={},
            current_predictors={},
            n_obs=0,
            n_recessions=0,
            history=pd.Series(dtype=float),
            feature_names=[],
            fit_ok=False,
            error="USREC not available",
        )

    # Resample to monthly (USREC is already monthly; daily series get monthly mean)
    usrec_m = usrec.resample("MS").last().dropna()

    # Build outcome: was there a recession start in the next `horizon_quarters` quarters?
    # A "recession start" = USREC transitions from 0 → 1.
    horizon_months = horizon_quarters * 3
    recession_start = (usrec_m.diff() > 0).astype(int)  # 1 at month of recession onset
    # Rolling forward max: if ANY of the next horizon_months months has an onset, outcome=1
    outcome = (
        recession_start
        .iloc[::-1]
        .rolling(horizon_months, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-horizon_months)  # align to the prediction date, not the event date
    )

    # Load predictors — skip any series not tracked locally or with < 60 monthly obs
    MIN_PREDICTOR_OBS = 60
    feature_series: dict[str, pd.Series] = {}
    for sid, label, lag in _PREDICTORS:
        try:
            s = load_series(sid)
        except (KeyError, Exception):
            continue
        if s is None or s.empty:
            continue
        s_m = s.resample("MS").mean().dropna()
        if len(s_m) < MIN_PREDICTOR_OBS:
            continue  # insufficient history — would collapse alignment window
        if lag > 0:
            s_m = s_m.shift(lag)
        feature_series[label] = s_m

    if not feature_series:
        return RecessionLogitResult(
            current_probability=float("nan"),
            six_month_ago_probability=float("nan"),
            auroc=float("nan"),
            coefficients={},
            current_predictors={},
            n_obs=0,
            n_recessions=0,
            history=pd.Series(dtype=float),
            feature_names=[],
            fit_ok=False,
            error="No predictors available",
        )

    # Align on common index
    feature_df = pd.DataFrame(feature_series)
    aligned = pd.concat([feature_df, outcome.rename("outcome"), usrec_m.rename("usrec")], axis=1).dropna()
    # Drop the last horizon_months rows where outcome is unknown (future)
    aligned = aligned.iloc[:-horizon_months] if len(aligned) > horizon_months else aligned

    if len(aligned) < 60:
        return RecessionLogitResult(
            current_probability=float("nan"),
            six_month_ago_probability=float("nan"),
            auroc=float("nan"),
            coefficients={},
            current_predictors={},
            n_obs=len(aligned),
            n_recessions=0,
            history=pd.Series(dtype=float),
            feature_names=list(feature_series),
            fit_ok=False,
            error=f"Too few observations: {len(aligned)}",
        )

    feature_names = list(feature_series.keys())
    X = aligned[feature_names].values
    y = aligned["outcome"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_sc, y)

    y_pred_prob = clf.predict_proba(X_sc)[:, 1]
    try:
        auroc = float(roc_auc_score(y, y_pred_prob))
    except ValueError:
        auroc = float("nan")

    history = pd.Series(y_pred_prob, index=aligned.index)

    # Current reading: use the latest available predictor values (outside the training set)
    current_features = _latest_predictor_values(feature_series, scaler, feature_names)
    if current_features is not None:
        current_prob = float(clf.predict_proba(current_features)[0, 1])
    else:
        current_prob = float(history.iloc[-1]) if len(history) else float("nan")

    # 6-month-ago reading (use in-sample if the index goes far enough)
    six_months_ago = aligned.index[-1] - pd.DateOffset(months=6)
    prior_idx = history.index.get_indexer([six_months_ago], method="nearest")[0]
    prior_prob = float(history.iloc[prior_idx]) if prior_idx >= 0 else float("nan")

    coefficients = dict(zip(feature_names, map(float, clf.coef_[0])))
    current_predictor_vals = {
        label: float(feature_series[label].dropna().iloc[-1])
        for label in feature_names
        if not feature_series[label].dropna().empty
    }

    return RecessionLogitResult(
        current_probability=current_prob,
        six_month_ago_probability=prior_prob,
        auroc=auroc,
        coefficients=coefficients,
        current_predictors=current_predictor_vals,
        n_obs=len(aligned),
        n_recessions=int(y.sum()),
        history=history,
        feature_names=feature_names,
        fit_ok=True,
    )


def _latest_predictor_values(
    feature_series: dict[str, pd.Series],
    scaler: "StandardScaler",
    feature_names: list[str],
) -> np.ndarray | None:
    """Build a scaled feature vector from the latest available observations."""
    vals = []
    for name in feature_names:
        s = feature_series.get(name)
        if s is None or s.dropna().empty:
            return None
        vals.append(float(s.dropna().iloc[-1]))
    raw = np.array(vals).reshape(1, -1)
    return scaler.transform(raw)
