"""Research baselines for Bao2 reservoir storage prediction.

This module implements a strict, reproducibility-first baselines experiment for the
ongoing thesis.  It is a *research reconstruction* (NOT a June-reproduction)
that consumes the documented recovery snapshot and produces an independent
artifact set under ``data/research/recovery_baselines_v2``.

Specification (binding, do not loosen without consulting the lead engineer):

1.  Stations are fixed: ``C0D580, C0D550, 72D080, C1D410, C1D420``.
2.  Cross-station averaging rules (per calendar day, no partial averaging):
      * Rainfall (``pp01``): all five stations must be valid *that* calendar day;
        the per-day valid-station count is emitted alongside the averaged value.
      * The five remaining variables (``tx01, tx02, rh01, wd01, ps01``): the
        *first three* stations (``C0D580, C0D550, 72D080``) must each be valid
        that calendar day; the per-day valid-station count is emitted.
      * Per-variable missing stations leave the averaged value as NaN; the rest
        of the calendar day is not imputed (forward-fill / bfill is forbidden).
3.  Weather is usable only when ``within_station_history == True``.
4.  Bao2 target is usable only when ``strict_target_usable == True`` AND
    ``storage_rate`` lies in ``[0, 100]``.  Targets are RAW observed values;
    no imputation, no bfill, no forward fill.
5.  A complete calendar index is materialised over the union of weather and
    Bao2 dates so that no day is silently dropped.
6.  Input window: a contiguous 14-day block ``[t-13, t]`` (inclusive) of
    single-day features ``{rainfall_avg_5, tx01_avg_3, tx02_avg_3, rh01_avg_3,
    wd01_avg_3, ps01_avg_3, storage}`` => a flat 98-dim input vector.  Every
    one of the 14 days must carry a strict Bao2 storage value AND a fully
    averaged weather row; otherwise the sample is excluded.
7.  Targets: the raw ``storage_rate`` at ``t + horizon`` for ``horizon in
    {7, 14}``.  No forward-fill, no back-fill.
8.  Time-series splits are driven by ``target_date`` (NOT by ``origin_date``):

        train : target_date in 2016..2022  (explicit; training does NOT start
                                              in 2014)
        val   : target_date in 2023
        test  : target_date in 2024..2025

9.  All models at a given ``(horizon, split)`` see the *same* samples.
10. Models (no test-set tuning, identical samples across models):
        * Persistence baseline: ``y_pred(t+h) = storage(t)`` -- the most
          recent observed storage in the input window.  Its "input" is the
          last-day storage column of the same ``X`` shared with the other
          models.  ``fit`` is a no-op.
        * ``Ridge(alpha=1.0)`` on z-scored features.
        * ``RandomForestRegressor(n_estimators=200, max_depth=10,
              min_samples_leaf=5, random_state=42, n_jobs=2)`` on raw
          features.  No test-set tuning of any kind.
11. ``StandardScaler`` is fit *per horizon* on that horizon's training inputs
    only (not across horizons, never on val/test) and reused at inference.
12. Metrics per ``(model, horizon, split)``: MAE, RMSE (percentage points),
    R^2, MAPE excluding zero actuals (with the count of excluded zero rows),
    sample count.  When ``n == 0`` the MAE/RMSE/MAPE/R^2 cells are empty (not
    NaN-codepoint strings).  The low-storage subset (``actual < 30`` pp) also
    gets its own MAE and RMSE reported per model, with the same empty-cell
    rule when the subset is empty.  An actual below 30 pp is reported as a
    REGIME tag only; this module never claims a "water shortage" event.
13. Artifacts written to ``data/research/recovery_baselines_v2`` (per-day
    predictions, metrics table, fixed config dump, input-file SHA256,
    code SHA256, package versions, daily data table, sample list, exclusion
    counts, joblib-serialised models and scalers).
14. A completed run is signalled by a ``COMPLETED.flag`` sentinel; subsequent
    invocations refuse to overwrite without an explicit ``--force`` flag.
    Pre-existing ``recovery_baselines_v1`` is left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Fixed configuration (mutable only via argparse; do not soft-code elsewhere)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data'

RECOVERY_DATE = '20260920'
DEFAULT_BAO2_CSV = DATA_ROOT / 'recovery' / RECOVERY_DATE / 'bao2_daily.csv'
DEFAULT_WEATHER_CSV = DATA_ROOT / 'recovery' / RECOVERY_DATE / 'research_weather_snapshot.csv'
# v1 lives at recovery_baselines_v1 (no-refusal with --force); the default
# default output for the persistence-augmented experiment is v2.
DEFAULT_OUTPUT = DATA_ROOT / 'research' / 'recovery_baselines_v2'

# Five fixed stations.  Order matters for the rainfall averaging rule.
STATIONS_ALL_FIVE: Tuple[str, ...] = ('C0D580', 'C0D550', '72D080', 'C1D410', 'C1D420')
# The first three stations are the averaging window for the non-rainfall
# variables; the trailing two act as supplementary rainfall observers.
STATIONS_FIRST_THREE: Tuple[str, ...] = STATIONS_ALL_FIVE[:3]

WEATHER_VARS: Tuple[str, ...] = ('pp01', 'tx01', 'tx02', 'rh01', 'wd01', 'ps01')
WEATHER_VARS_NON_RAIN: Tuple[str, ...] = WEATHER_VARS[1:]

# Three-week input window (t-13 .. t inclusive).
WINDOW_LEN: int = 14
HORIZONS: Tuple[int, ...] = (7, 14)

# Time-series splits keyed by TARGET date (the citable split rule).
SPLITS: Dict[str, Tuple[date, date]] = {
    'train': (date(2016, 1, 1), date(2022, 12, 31)),
    'val':   (date(2023, 1, 1), date(2023, 12, 31)),
    'test':  (date(2024, 1, 1), date(2025, 12, 31)),
}
SPLIT_NOTE = (
    'Splits driven by target_date year; train explicitly 2016..2022 (NOT 2014); '
    'val 2023; test 2024..2025.'
)

RANDOM_SEED: int = 42
RIDGE_ALPHA: float = 1.0
RF_KWARGS: Dict[str, object] = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=2,
)

# Below this percentage point an actual storage is treated as a low-storage
# REGIME in the metrics summary; this is a regime tag, NOT a drought / shortage
# event label.
LOW_STORAGE_THRESHOLD_PP: float = 30.0

COMPLETION_MARKER = 'COMPLETED.flag'


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def digest(path: Path) -> str:
    """SHA-256 of a file's bytes (streamed in 1 MiB blocks)."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_bao2(path: Path) -> pd.DataFrame:
    """Read Bao2 daily CSV; normalise dtypes; do NOT filter here."""
    df = pd.read_csv(path)
    df['data_date'] = pd.to_datetime(df['data_date']).dt.date
    df['observation_time'] = df['observation_time'].astype('string')
    df['strict_target_usable'] = df['strict_target_usable'].astype(bool)
    df['storage_rate'] = pd.to_numeric(df['storage_rate'], errors='coerce')
    return df


def load_weather(path: Path) -> pd.DataFrame:
    """Read weather snapshot CSV; coerce variable values to numeric; do NOT filter."""
    df = pd.read_csv(path)
    df['obs_date'] = pd.to_datetime(df['obs_date']).dt.date
    df['within_station_history'] = df['within_station_history'].astype(bool)
    for v in WEATHER_VARS:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Calendar panel construction
# ---------------------------------------------------------------------------

def _aggregate_cross_station(weather: pd.DataFrame) -> pd.DataFrame:
    """Build the per-calendar-day cross-station averaged weather panel.

    Rules (see module docstring §2):
      * pp01: all five stations valid that day => mean over the five; else NaN.
      * tx01/tx02/rh01/wd01/ps01: first-three stations valid that day => mean
        over the three; else NaN.
      * valid_stations_<var> reports the count of valid stations for that
        variable on that day (0..5 for rainfall, 0..3 for the others).
    """
    weather = weather[weather['within_station_history']]
    if weather.empty:
        return pd.DataFrame(columns=['date'])

    rows: List[dict] = []
    grouped = weather.groupby('obs_date', sort=True)
    for day, day_df in grouped:
        rec: Dict[str, object] = {'date': day}
        # Rainfall over the full five-station set.
        rain = day_df.set_index('stno')['pp01'].dropna()
        if len(rain) == 5 and set(rain.index) == set(STATIONS_ALL_FIVE):
            rec['pp01_avg'] = float(rain.reindex(STATIONS_ALL_FIVE).mean())
            rec['pp01_valid_stations'] = 5
        else:
            rec['pp01_avg'] = np.nan
            rec['pp01_valid_stations'] = int(len(set(rain.index)))

        # The other five variables, each over the first-three station set.
        first_three_df = day_df[day_df['stno'].isin(STATIONS_FIRST_THREE)]
        for v in WEATHER_VARS_NON_RAIN:
            ser = first_three_df.set_index('stno')[v].dropna()
            if len(ser) == 3 and set(ser.index) == set(STATIONS_FIRST_THREE):
                rec[f'{v}_avg'] = float(ser.reindex(STATIONS_FIRST_THREE).mean())
                rec[f'{v}_valid_stations'] = 3
            else:
                rec[f'{v}_avg'] = np.nan
                rec[f'{v}_valid_stations'] = int(len(set(ser.index)))
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_bao2_storage(bao2: pd.DataFrame) -> pd.DataFrame:
    """Return date, storage_rate, strict_target_usable (already typed in load_bao2)."""
    out = bao2[['data_date', 'storage_rate', 'strict_target_usable']].copy()
    out = out.rename(columns={'data_date': 'date'})
    # No de-duplication here: if multiple rows share a date (shouldn't happen
    # in bao2_daily) we keep all of them; the merge below would multiply rows.
    return out


def build_calendar_panel(bao2: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Materialise the complete calendar day panel across weather + Bao2 dates."""
    weather = weather[weather['within_station_history']]
    b_days = set(bao2['data_date'].dropna())
    w_days = set(weather['obs_date'].dropna())
    if not b_days and not w_days:
        return pd.DataFrame(columns=[
            'date', 'pp01_avg', 'pp01_valid_stations',
            'tx01_avg', 'tx01_valid_stations', 'tx02_avg', 'tx02_valid_stations',
            'rh01_avg', 'rh01_valid_stations', 'wd01_avg', 'wd01_valid_stations',
            'ps01_avg', 'ps01_valid_stations', 'storage', 'storage_strict',
        ])
    start = min(b_days | w_days)
    end = max(b_days | w_days)
    cal = pd.DataFrame({'date': list(_daterange(start, end))})

    wx = _aggregate_cross_station(weather)
    panel = cal.merge(wx, on='date', how='left')

    storage = _build_bao2_storage(bao2)
    panel = panel.merge(storage, on='date', how='left')

    panel['storage'] = panel['storage_rate']
    panel = panel[[
        'date',
        'pp01_avg', 'pp01_valid_stations',
        'tx01_avg', 'tx01_valid_stations',
        'tx02_avg', 'tx02_valid_stations',
        'rh01_avg', 'rh01_valid_stations',
        'wd01_avg', 'wd01_valid_stations',
        'ps01_avg', 'ps01_valid_stations',
        'storage', 'strict_target_usable',
    ]].sort_values('date').reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    origin_date: date
    target_date: date
    horizon: int


def _split_for_target(target: date) -> Optional[str]:
    for name, (start, end) in SPLITS.items():
        if start <= target <= end:
            return name
    return None


def _candidate_origins(horizons: Tuple[int, ...]) -> List[Tuple[date, int]]:
    """Enumerate candidate (origin, horizon) pairs that *could* fall in any split."""
    pairs: List[Tuple[date, int]] = []
    min_target = min(span[0] for span in SPLITS.values())
    max_target = max(span[1] for span in SPLITS.values())
    for h in horizons:
        earliest = min_target - timedelta(days=h)
        latest = max_target - timedelta(days=h)
        for d in _daterange(earliest, latest):
            pairs.append((d, h))
    return pairs


def _panel_storage_strict(panel: pd.DataFrame, target: date) -> bool:
    """Return True iff the target date has strict + 0..100 storage (no fill)."""
    rows = panel[panel['date'] == target]
    if rows.empty:
        return False
    row = rows.iloc[0]
    if pd.isna(row['storage']):
        return False
    val = float(row['storage'])
    if not (0.0 <= val <= 100.0):
        return False
    if not bool(row['strict_target_usable']):
        return False
    return True


def build_samples(
    panel: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    window_len: int = WINDOW_LEN,
) -> Tuple[Dict[int, List[Sample]], Dict[str, int]]:
    """Build samples per horizon with full audit of exclusions.

    Excluded candidates (counts cumulative across horizons):

        * ``excluded_storage_window_incomplete`` -- at least one day in
          [origin-13, origin] lacks strict Bao2 storage (raw, no fill).
        * ``excluded_weather_window_incomplete`` -- at least one day in
          [origin-13, origin] lacks any of the six averaged weather values.
        * ``excluded_storage_target_missing`` -- target_date origin+horizon
          lacks strict Bao2 storage (raw, no fill).

    All days in the input window must lie on the panel's calendar index;
    otherwise the candidate is also excluded (counted under the storage window
    reason since they are absent by construction).
    """
    panel_idx = panel.set_index('date').sort_index()
    if panel_idx.empty:
        empty_counts = dict(
            candidates_total=0,
            excluded_storage_window_incomplete=0,
            excluded_weather_window_incomplete=0,
            excluded_storage_target_missing=0,
            kept_total=0,
        )
        return {h: [] for h in horizons}, empty_counts

    weather_cols = ['pp01_avg', 'tx01_avg', 'tx02_avg', 'rh01_avg', 'wd01_avg', 'ps01_avg']

    samples: Dict[int, List[Sample]] = {h: [] for h in horizons}
    counts = dict(
        candidates_total=0,
        excluded_storage_window_incomplete=0,
        excluded_weather_window_incomplete=0,
        excluded_storage_target_missing=0,
        kept_total=0,
    )

    for origin, h in _candidate_origins(horizons):
        counts['candidates_total'] += 1
        window = [origin - timedelta(days=window_len - 1 - i) for i in range(window_len)]
        target = origin + timedelta(days=h)

        # Storage for every input day: strict + 0..100, no fill.
        ok_storage_window = True
        for d in window:
            if d not in panel_idx.index:
                ok_storage_window = False
                break
            row = panel_idx.loc[d]
            if pd.isna(row['storage']) or not bool(row['strict_target_usable']):
                ok_storage_window = False
                break
            val = float(row['storage'])
            if not (0.0 <= val <= 100.0):
                ok_storage_window = False
                break
        if not ok_storage_window:
            counts['excluded_storage_window_incomplete'] += 1
            continue

        # Weather for every input day: all 6 averaged features non-null.
        ok_weather_window = True
        for d in window:
            row = panel_idx.loc[d]
            for c in weather_cols:
                v = row[c]
                if pd.isna(v):
                    ok_weather_window = False
                    break
            if not ok_weather_window:
                break
        if not ok_weather_window:
            counts['excluded_weather_window_incomplete'] += 1
            continue

        # Target raw storage: strict + 0..100, no fill.
        if not _panel_storage_strict(panel, target):
            counts['excluded_storage_target_missing'] += 1
            continue

        samples[h].append(Sample(origin_date=origin, target_date=target, horizon=h))
        counts['kept_total'] += 1

    return samples, counts


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

# Single-day feature order; must stay aligned with the 98-dim flat input.
FEATURE_COLS: Tuple[str, ...] = (
    'pp01_avg', 'tx01_avg', 'tx02_avg', 'rh01_avg', 'wd01_avg', 'ps01_avg', 'storage',
)


def assemble_xy(
    panel: pd.DataFrame,
    samples: List[Sample],
    window_len: int = WINDOW_LEN,
) -> Tuple[np.ndarray, np.ndarray, List[date], List[date]]:
    """Stack 14-day * 7-feature windows for each sample.

    Returns ``(X, y, origin_dates, target_dates)``.  Targets are the raw,
    non-filled storage observation on the target date.
    """
    panel_idx = panel.set_index('date').sort_index()
    n_feat_per_day = len(FEATURE_COLS)
    n = len(samples)
    X = np.full((n, window_len * n_feat_per_day), np.nan, dtype=np.float64)
    y = np.full(n, np.nan, dtype=np.float64)
    origins: List[date] = []
    targets: List[date] = []
    for i, s in enumerate(samples):
        window = [s.origin_date - timedelta(days=window_len - 1 - k) for k in range(window_len)]
        for k, d in enumerate(window):
            row = panel_idx.loc[d]
            for f, col in enumerate(FEATURE_COLS):
                X[i, k * n_feat_per_day + f] = float(row[col])
        y[i] = float(panel_idx.loc[s.target_date, 'storage'])
        origins.append(s.origin_date)
        targets.append(s.target_date)
    return X, y, origins, targets


def split_samples(samples: List[Sample]) -> Dict[str, List[Sample]]:
    """Partition samples by the split that owns their target date."""
    out: Dict[str, List[Sample]] = {name: [] for name in SPLITS}
    out['out_of_range'] = []
    for s in samples:
        owner = _split_for_target(s.target_date)
        if owner is None:
            out['out_of_range'].append(s)
        else:
            out[owner].append(s)
    return out


# ---------------------------------------------------------------------------
# Models & metrics
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)
    return model


def fit_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(**RF_KWARGS)
    model.fit(X_train, y_train)
    return model


class PersistenceRegressor:
    """Persistence baseline: predict the most recent observed storage.

    The regressor consumes the SAME 98-dim input vector as Ridge and RF
    (samples, splits and horizons are shared with the other models), but only
    uses the last-day ``storage`` column to make its prediction.  ``fit`` is
    a no-op aside from recording the input column index.

    Saved with ``joblib.dump`` for inspection; pickle-safe serialisation.
    """

    LAST_FEATURE_COL_NAME = 'storage'

    def __init__(self) -> None:
        self.last_idx_: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PersistenceRegressor':
        # ``X`` has shape (n_samples, WINDOW_LEN * len(FEATURE_COLS)) = (n, 98).
        # The storage column is the last feature of each timestep, so the last
        # column overall is also the last-day storage.
        if X.ndim != 2:
            raise ValueError(f'PersistenceRegressor expects a 2-D matrix; got ndim={X.ndim}')
        self.last_idx_ = X.shape[1] - 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.last_idx_ is None:
            raise RuntimeError('PersistenceRegressor.predict called before .fit()')
        return np.asarray(X[:, self.last_idx_], dtype=np.float64)


def fit_persistence(X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> PersistenceRegressor:
    """Construct a :class:`PersistenceRegressor` and record the storage column index."""
    model = PersistenceRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Compute MAE / RMSE / R^2 / MAPE (zero-excluded) plus the low-storage
    sub-metrics (actual below ``LOW_STORAGE_THRESHOLD_PP``).

    Numeric fields (``mae_pp``, ``rmse_pp``, ``r2``, ``mape_percent``,
    ``mae_pp_low_storage``, ``rmse_pp_low_storage``, ``low_storage_fraction``)
    carry ``np.nan`` when they cannot be computed (sample empty, single
    sample, no zero-excluded MAPE support, low-storage subset empty, etc.).
    ``pandas.to_csv`` then renders ``NaN`` as an empty cell, satisfying the
    "leave the cell empty when n=0" requirement.  The returned counters
    (``n``, ``n_zero_actual_excluded_from_mape``, ``n_nonzero_for_mape``,
    ``low_storage_count``) are always integers.
    """
    nan = float('nan')
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = int(len(y_true))
    if n == 0:
        return dict(
            n=0,
            n_zero_actual_excluded_from_mape=0,
            n_nonzero_for_mape=0,
            mae_pp=nan, rmse_pp=nan, r2=nan, mape_percent=nan,
            low_storage_count=0,
            n_low_storage=0,
            mae_pp_low_storage=nan,
            rmse_pp_low_storage=nan,
            low_storage_fraction=nan,
        )
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if n > 1 else nan
    nonzero = y_true != 0.0
    n_zero = int((~nonzero).sum())
    n_nonzero = int(nonzero.sum())
    if n_nonzero > 0:
        mape = float(np.mean(np.abs(
            (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
    else:
        mape = nan
    low = y_true < LOW_STORAGE_THRESHOLD_PP
    low_n = int(low.sum())
    if low_n > 0:
        mae_low = float(mean_absolute_error(y_true[low], y_pred[low]))
        rmse_low = float(np.sqrt(mean_squared_error(y_true[low], y_pred[low])))
    else:
        mae_low = nan
        rmse_low = nan
    return dict(
        n=n,
        n_zero_actual_excluded_from_mape=n_zero,
        n_nonzero_for_mape=n_nonzero,
        mae_pp=mae,
        rmse_pp=rmse,
        r2=r2,
        mape_percent=mape,
        low_storage_count=low_n,
        n_low_storage=low_n,
        mae_pp_low_storage=mae_low,
        rmse_pp_low_storage=rmse_low,
        low_storage_fraction=float(low_n) / n,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _package_versions() -> Dict[str, str]:
    def _safe(modname: str) -> str:
        mod = sys.modules.get(modname)
        return getattr(mod, '__version__', 'unknown')

    return dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        pandas=_safe('pandas'),
        numpy=_safe('numpy'),
        sklearn=sklearn.__version__,
        joblib=_safe('joblib'),
    )


def _make_predictions_frame(
    horizon: int, model_name: str,
    origins: List[date], targets: List[date],
    split: str, y_true: np.ndarray, y_pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        'origin_date': origins,
        'target_date': targets,
        'split': split,
        'horizon': horizon,
        'model': model_name,
        'actual': y_true,
        'predicted': y_pred,
    })


def _hash_panel_text(panel: pd.DataFrame) -> str:
    """Stable text fingerprint of the panel for the daily-data-table artefact."""
    h = hashlib.sha256()
    cols = ['date', 'pp01_avg', 'pp01_valid_stations', 'tx01_avg', 'tx01_valid_stations',
            'tx02_avg', 'tx02_valid_stations', 'rh01_avg', 'rh01_valid_stations',
            'wd01_avg', 'wd01_valid_stations', 'ps01_avg', 'ps01_valid_stations',
            'storage', 'strict_target_usable']
    text = panel[cols].to_csv(index=False).encode()
    h.update(text)
    return h.hexdigest()


def _is_complete(outdir: Path) -> bool:
    return (outdir / COMPLETION_MARKER).exists()


def run(
    bao2_csv: Path = DEFAULT_BAO2_CSV,
    weather_csv: Path = DEFAULT_WEATHER_CSV,
    output: Path = DEFAULT_OUTPUT,
    force: bool = False,
) -> int:
    """End-to-end execution; returns process exit code."""
    if _is_complete(output) and not force:
        print(f'[run_baselines] {output} already completed ({COMPLETION_MARKER} present); '
              f'refusing to overwrite. Use --force to redo.')
        return 0

    output.mkdir(parents=True, exist_ok=True)
    # Drop a stale marker if we are rerunning under --force.
    marker = output / COMPLETION_MARKER
    if marker.exists():
        marker.unlink()

    # --- 1. hashes for inputs and code -------------------------------------
    bao2_sha = digest(bao2_csv) if bao2_csv.exists() else 'missing'
    weather_sha = digest(weather_csv) if weather_csv.exists() else 'missing'
    code_sha = digest(Path(__file__))

    # --- 2. load inputs ----------------------------------------------------
    bao2 = load_bao2(bao2_csv)
    weather = load_weather(weather_csv)

    # --- 3. complete calendar panel --------------------------------------
    panel = build_calendar_panel(bao2, weather)
    panel_sha = _hash_panel_text(panel)

    # --- 4. samples and exclusions ---------------------------------------
    samples_by_h, exclusion_counts = build_samples(panel)

    # --- 5. assemble per-horizon X / y and splits ------------------------
    horizons_data: Dict[int, Dict[str, dict]] = {}
    for h in HORIZONS:
        sample_list = samples_by_h[h]
        partitions = split_samples(sample_list)
        X, y, origins, targets = assemble_xy(panel, sample_list)
        split_payload: Dict[str, dict] = {}
        for split_name, part in partitions.items():
            # Index lookup by sample target date (origin is identical for same target date).
            idxs = sorted({s.origin_date for s in part})
            origins_s = [s.origin_date for s in part]
            targets_s = [s.target_date for s in part]
            # Map sample lists back to the matching rows of X, y.
            key_to_row: Dict[Tuple[date, date], int] = {(o, t): i for i, (o, t) in enumerate(zip(origins, targets))}
            rows = [key_to_row[(o, t)] for o, t in zip(origins_s, targets_s)]
            sub_X = X[rows] if rows else np.empty((0, X.shape[1] if X.ndim == 2 else 0))
            sub_y = y[rows] if rows else np.empty((0,))
            split_payload[split_name] = dict(
                samples=part,
                origin_dates=origins_s,
                target_dates=targets_s,
                X=sub_X, y=sub_y,
            )
        horizons_data[h] = dict(
            all_samples=sample_list,
            splits=split_payload,
            X=X, y=y,
            origin_dates=origins,
            target_dates=targets,
        )

    # --- 6. per-horizon StandardScaler fit ONLY on that horizon's train -
    scaler_per_h: Dict[int, StandardScaler] = {}
    for h in HORIZONS:
        train_payload = horizons_data[h]['splits'].get('train', {})
        Xt = train_payload.get('X')
        if Xt is not None and Xt.size:
            scaler_per_h[h] = fit_scaler(Xt)
            joblib.dump(scaler_per_h[h], output / f'scaler_ridge_h{h}.joblib')
        else:
            # No training data for this horizon: still emit an unscaled
            # scaler so consumers can detect the degenerate case.
            scaler_per_h[h] = StandardScaler()
            joblib.dump(scaler_per_h[h], output / f'scaler_ridge_h{h}.joblib')

    # --- 7. fit one Ridge (scaled), one RandomForest, one Persistence ----
    #       per horizon.  All fits use ONLY the train split; val/test never
    #       enter any fit step.
    fitted_per_h: Dict[int, Dict[str, object]] = {}

    for h in HORIZONS:
        hdata = horizons_data[h]
        train_payload = hdata['splits'].get('train', {})
        if not train_payload or not train_payload['X'].size:
            fitted_per_h[h] = {}
            joblib.dump(PersistenceRegressor(), output / f'model_persistence_h{h}.joblib')
            continue
        rows_idx = {(o, t): i for i, (o, t) in enumerate(zip(hdata['origin_dates'], hdata['target_dates']))}
        train_idxs = [rows_idx[(o, t)] for o, t in zip(train_payload['origin_dates'], train_payload['target_dates'])]
        X_train = hdata['X'][train_idxs]
        y_train = hdata['y'][train_idxs]

        scaler = scaler_per_h[h]

        # Ridge on StandardScaler(features).  The scaler was fit on this
        # horizon's training inputs only (step 6) and is reused at
        # inference for every split of this horizon.
        ridge = fit_ridge(scaler.transform(X_train), y_train)
        joblib.dump(ridge, output / f'model_ridge_h{h}.joblib')

        # RandomForest on raw features; the scaler is intentionally not applied.
        rf = fit_rf(X_train, y_train)
        joblib.dump(rf, output / f'model_random_forest_h{h}.joblib')

        # Persistence baseline: same X, predicts the last-day storage column.
        persistence = fit_persistence(X_train, y_train)
        joblib.dump(persistence, output / f'model_persistence_h{h}.joblib')

        fitted_per_h[h] = dict(ridge=ridge, random_forest=rf, persistence=persistence)

    # --- 8. predict on each (horizon, split) -----------------------------
    # All three models consume the SAME X for a given (horizon, split).
    metrics_rows: List[dict] = []
    predictions_frames: List[pd.DataFrame] = []
    for h in HORIZONS:
        hdata = horizons_data[h]
        fitted = fitted_per_h.get(h, {})
        if not fitted:
            continue
        rows_idx = {(o, t): i for i, (o, t) in enumerate(zip(hdata['origin_dates'], hdata['target_dates']))}
        scaler = scaler_per_h[h]
        for split_name in ('train', 'val', 'test'):
            payload = hdata['splits'].get(split_name)
            if not payload or not payload['X'].size:
                continue
            idxs = [rows_idx[(o, t)] for o, t in zip(payload['origin_dates'], payload['target_dates'])]
            X_part = hdata['X'][idxs]
            y_part = hdata['y'][idxs]
            origins_part = payload['origin_dates']
            targets_part = payload['target_dates']

            # Ridge inference always goes through this horizon's scaler.
            ridge_pred = np.asarray(fitted['ridge'].predict(scaler.transform(X_part)), dtype=np.float64)
            ridge_metrics = evaluate(y_part, ridge_pred)
            ridge_metrics.update(model='ridge', horizon=h, split=split_name)
            metrics_rows.append(ridge_metrics)
            predictions_frames.append(_make_predictions_frame(
                h, 'ridge', origins_part, targets_part, split_name, y_part, ridge_pred))

            # RandomForest inference uses the raw window features.
            rf_pred = np.asarray(fitted['random_forest'].predict(X_part), dtype=np.float64)
            rf_metrics = evaluate(y_part, rf_pred)
            rf_metrics.update(model='random_forest', horizon=h, split=split_name)
            metrics_rows.append(rf_metrics)
            predictions_frames.append(_make_predictions_frame(
                h, 'random_forest', origins_part, targets_part, split_name, y_part, rf_pred))

            # Persistence inference: identical samples, identical y, just a
            # different prediction column (last-day storage).
            persistence_pred = np.asarray(fitted['persistence'].predict(X_part), dtype=np.float64)
            persistence_metrics = evaluate(y_part, persistence_pred)
            persistence_metrics.update(model='persistence', horizon=h, split=split_name)
            metrics_rows.append(persistence_metrics)
            predictions_frames.append(_make_predictions_frame(
                h, 'persistence', origins_part, targets_part, split_name, y_part, persistence_pred))

    # --- 9. metrics + predictions ---------------------------------------
    METRIC_COLUMNS: Tuple[str, ...] = (
        'model', 'horizon', 'split',
        'n', 'n_zero_actual_excluded_from_mape', 'n_nonzero_for_mape',
        'mae_pp', 'rmse_pp', 'r2', 'mape_percent',
        'n_low_storage', 'mae_pp_low_storage', 'rmse_pp_low_storage',
        'low_storage_fraction',
    )
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        for col in METRIC_COLUMNS:
            if col not in metrics_df.columns:
                metrics_df[col] = np.nan
        metrics_df = metrics_df[list(METRIC_COLUMNS)]
    else:
        metrics_df = pd.DataFrame(columns=list(METRIC_COLUMNS))
    # Empty (NaN) cells render as empty strings in pandas.to_csv; this is
    # exactly the "leave the cell empty when n=0" requirement.
    metrics_df.to_csv(output / 'metrics.csv', index=False)
    predictions_df = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
    if not predictions_df.empty:
        predictions_df = predictions_df[['origin_date', 'target_date', 'split', 'horizon',
                                          'model', 'actual', 'predicted']]
    predictions_df.to_csv(output / 'predictions.csv', index=False)

    # --- 9. artefact tables and config -----------------------------------
    daily_table = panel[[
        'date',
        'pp01_avg', 'pp01_valid_stations',
        'tx01_avg', 'tx01_valid_stations',
        'tx02_avg', 'tx02_valid_stations',
        'rh01_avg', 'rh01_valid_stations',
        'wd01_avg', 'wd01_valid_stations',
        'ps01_avg', 'ps01_valid_stations',
        'storage', 'strict_target_usable',
    ]].copy()
    daily_table.to_csv(output / 'daily_data_table.csv', index=False)

    sample_records: List[dict] = []
    for h in HORIZONS:
        for s in horizons_data[h]['all_samples']:
            sample_records.append({
                'origin_date': s.origin_date,
                'target_date': s.target_date,
                'horizon': s.horizon,
                'split': _split_for_target(s.target_date) or 'out_of_range',
            })
    pd.DataFrame(sample_records,
                 columns=['origin_date', 'target_date', 'horizon', 'split']).to_csv(
        output / 'samples.csv', index=False)

    per_horizon_excl: Dict[str, Dict[str, int]] = {}
    # Re-derive per-horizon counts for the exclusion-csv artefact.
    for h in HORIZONS:
        per_horizon_excl[f'h{h}'] = dict(
            n_samples=len(horizons_data[h]['all_samples']),
        )

    config_payload = dict(
        fixed_config=dict(
            recovery_date=RECOVERY_DATE,
            stations_all_five=list(STATIONS_ALL_FIVE),
            stations_first_three=list(STATIONS_FIRST_THREE),
            weather_vars=list(WEATHER_VARS),
            window_len=WINDOW_LEN,
            input_features_per_day=list(FEATURE_COLS),
            input_dim=WINDOW_LEN * len(FEATURE_COLS),
            horizons=list(HORIZONS),
            splits={k: [v[0].isoformat(), v[1].isoformat()] for k, v in SPLITS.items()},
            split_note=SPLIT_NOTE,
            ridge_alpha=RIDGE_ALPHA,
            random_forest=RF_KWARGS,
            random_seed=RANDOM_SEED,
            low_storage_threshold_pp=LOW_STORAGE_THRESHOLD_PP,
        ),
        input_files=dict(
            bao2_csv=str(bao2_csv), bao2_sha256=bao2_sha,
            weather_csv=str(weather_csv), weather_sha256=weather_sha,
        ),
        code_sha256=code_sha,
        panel_sha256=panel_sha,
        package_versions=_package_versions(),
        exclusion_counts=exclusion_counts,
        per_horizon_sample_counts=per_horizon_excl,
        models=dict(
            note=('Three models per horizon, all fitted on TRAIN only and '
                  'evaluated on the IDENTICAL samples at each (horizon, split): '
                  'Persistence (predicts the last observed storage in the '
                  'input window; no fit), Ridge(alpha=1.0) on the per-horizon '
                  'StandardScaler(features), and RandomForest(n_estimators=200, '
                  'max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=2) '
                  'on raw features.  The StandardScaler is fit PER HORIZON on '
                  'that horizon\'s training inputs only (no cross-horizon '
                  'union, never on val/test).'),
            files=(
                [f'model_persistence_h{h}.joblib' for h in HORIZONS]
                + [f'model_ridge_h{h}.joblib' for h in HORIZONS]
                + [f'model_random_forest_h{h}.joblib' for h in HORIZONS]
                + [f'scaler_ridge_h{h}.joblib' for h in HORIZONS]
            ),
        ),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    (output / 'config.json').write_text(json.dumps(config_payload, indent=2, ensure_ascii=False))

    # --- 10. completion sentinel ----------------------------------------
    (output / COMPLETION_MARKER).write_text(json.dumps(dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=code_sha,
        bao2_sha256=bao2_sha,
        weather_sha256=weather_sha,
        exclusion_counts=exclusion_counts,
    ), indent=2, ensure_ascii=False))

    print(f'[run_baselines] completed -> {output}')
    print(f'[run_baselines] candidates={exclusion_counts["candidates_total"]} '
          f'kept={exclusion_counts["kept_total"]}')
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bao2-csv', type=Path, default=DEFAULT_BAO2_CSV)
    parser.add_argument('--weather-csv', type=Path, default=DEFAULT_WEATHER_CSV)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if the output dir already reports COMPLETED.flag.')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(bao2_csv=args.bao2_csv, weather_csv=args.weather_csv,
               output=args.output, force=args.force)


if __name__ == '__main__':
    raise SystemExit(main())
