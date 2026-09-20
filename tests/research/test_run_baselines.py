"""Unit tests for src.research.run_baselines.

These tests verify the binding specification points called out by the lead engineer:

  * calendar window completeness (no gap collapse)
  * target date drives the train/val/test split
  * the storage target is RAW (no forward-fill, no back-fill)
  * cross-station averaging requires 5 stations for rainfall, 3 for the others
  * per-day valid-station count is emitted
  * weather rows outside ``within_station_history`` are excluded
  * the 14-day input window must be strictly contiguous and fully valid
  * metrics: MAPE excludes zero actuals and reports the exclusion count
  * an actual below ``30`` carries its own MAE and RMSE per model
  * sample-empty cells stay empty (not NaN-codepoint) in the metrics CSV
  * three models per horizon (persistence, ridge, random_forest) share the
    same samples for evaluation
  * the StandardScaler is fit per horizon on that horizon's training inputs
    only, never across horizons and never on val/test
  * completed runs are not silently overwritten

The tests synthesise minimal in-memory CSVs so they do not depend on the
real recovery snapshot.
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Make the module importable from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from research import run_baselines  # noqa: E402  (path tweak above)
from research.run_baselines import (  # noqa: E402
    COMPLETION_MARKER,
    FEATURE_COLS,
    HORIZONS,
    LOW_STORAGE_THRESHOLD_PP,
    PersistenceRegressor,
    SPLITS,
    STATIONS_ALL_FIVE,
    STATIONS_FIRST_THREE,
    WEATHER_VARS,
    WEATHER_VARS_NON_RAIN,
    WINDOW_LEN,
    Sample,
    _aggregate_cross_station,
    assemble_xy,
    build_calendar_panel,
    build_samples,
    evaluate,
    fit_persistence,
    fit_ridge,
    fit_rf,
    fit_scaler,
    split_samples,
)


def _make_bao2_df(dates, *, strict=True, in_range=True, value=50.0):
    rows = []
    for d in dates:
        rows.append({
            'id': 0,
            'data_date': d,  # Python date to align with the production loader
            'reservoir_id': 23,
            'observation_time': f'{d} 22:00:00',
            'basin_rainfall_mm': '',
            'inflow_cms': '',
            'effective_storage': '',
            'outflow_cms': '',
            'water_level_m': '',
            'full_water_level_m': '',
            'storage_rate': 50.0 if in_range else 150.0,
            'strict_target_usable': bool(strict),
        })
    df = pd.DataFrame(rows)
    if value != 50.0:
        df['storage_rate'] = float(value)
    df['data_date'] = pd.to_datetime(df['data_date']).dt.date
    return df


def _make_weather_df(records):
    """records: list of dicts with stno, obs_date, within_station_history, vars..."""
    cols = ['source_sha256', 'row_number', 'stno', 'obs_date', 'source_priority',
            'pp01', 'tx01', 'tx02', 'rh01', 'wd01', 'ps01', 'quality',
            'stname', 'opened', 'closed', 'within_station_history']
    rows = []
    for r in records:
        rows.append({
            'source_sha256': 'deadbeef' * 8,
            'row_number': 0,
            'stno': r['stno'],
            'obs_date': r['obs_date'],  # Python date
            'source_priority': 20,
            'pp01': r.get('pp01', np.nan),
            'tx01': r.get('tx01', np.nan),
            'tx02': r.get('tx02', np.nan),
            'rh01': r.get('rh01', np.nan),
            'wd01': r.get('wd01', np.nan),
            'ps01': r.get('ps01', np.nan),
            'quality': '{}',
            'stname': r['stno'],
            'opened': date(2000, 1, 1),
            'closed': pd.NaT,
            'within_station_history': r.get('within_station_history', True),
        })
    df = pd.DataFrame(rows, columns=cols)
    # Normalise types like the real loader would (dates as Python dates).
    df['obs_date'] = pd.to_datetime(df['obs_date']).dt.date
    df['within_station_history'] = df['within_station_history'].astype(bool)
    for v in WEATHER_VARS:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    return df


def _full_weather_day(day, *, base=1.0, outside_history=False, drop_var=None):
    """Build a (nearly-)complete weather day for all five stations.

    Parameters
    ----------
    day : date
    base : float
        Numeric baseline for every variable.
    outside_history : list of str
        Stations whose ``within_station_history`` is False on this day.
    drop_var : list of str
        Variables to leave as NaN on this day (across all five stations).
    """
    drop_var = drop_var or []
    outside_history = [s for s in (outside_history if isinstance(outside_history, (list, tuple, set)) else [outside_history]) if s]
    records = []
    for stno in STATIONS_ALL_FIVE:
        rec = {'stno': stno, 'obs_date': day,
               'within_station_history': stno not in outside_history}
        for v in WEATHER_VARS:
            rec[v] = float('nan') if v in drop_var else base
        records.append(rec)
    return records


class TestCalendarCompleteness(unittest.TestCase):
    def test_complete_calendar_no_gap_collapse(self):
        day0 = date(2021, 6, 1)
        # Bao2 only covers 3 of the 5 days, weather only covers the other 2.
        bao2 = _make_bao2_df([day0, day0 + timedelta(days=1), day0 + timedelta(days=2)])
        wx = _make_weather_df(
            _full_weather_day(day0 + timedelta(days=2))
            + _full_weather_day(day0 + timedelta(days=3))
            + _full_weather_day(day0 + timedelta(days=4))
        )
        panel = build_calendar_panel(bao2, wx)
        # Calendar from min(bao2) to max(weather) must contain every calendar
        # day with zero holes.
        expected_dates = {(day0 + timedelta(days=i)).isoformat() for i in range(5)}
        self.assertEqual(set(panel['date'].astype(str)), expected_dates)
        # Days where Bao2 was absent collapse to NaN storage, no synthetic fill.
        missing_bao2_day = day0 + timedelta(days=4)
        row = panel[panel['date'].astype(str) == missing_bao2_day.isoformat()].iloc[0]
        self.assertTrue(pd.isna(row['storage']))
        # Days covered by Bao2 keep the observed raw storage value.
        covered_day = day0 + timedelta(days=1)
        row = panel[panel['date'].astype(str) == covered_day.isoformat()].iloc[0]
        self.assertEqual(float(row['storage']), 50.0)


class TestCrossStationRules(unittest.TestCase):
    def test_rainfall_5_valid_averaged(self):
        day = date(2024, 5, 1)
        wx = _make_weather_df(_full_weather_day(day, base=2.0))
        agg = _aggregate_cross_station(wx)
        rec = agg[agg['date'] == day].iloc[0]
        self.assertAlmostEqual(float(rec['pp01_avg']), 2.0, places=6)
        self.assertEqual(int(rec['pp01_valid_stations']), 5)

    def test_rainfall_partial_only_no_average_and_valid_count_emitted(self):
        day = date(2024, 5, 1)
        records = _full_weather_day(day, base=2.0)
        # Make the last station missing for pp01
        records[-1]['pp01'] = np.nan
        # And make the 4th station outside its station history entirely (so its
        # row is dropped before aggregation, reducing valid count).
        wx = _make_weather_df(records)
        agg = _aggregate_cross_station(wx)
        rec = agg[agg['date'] == day].iloc[0]
        self.assertTrue(pd.isna(rec['pp01_avg']))
        # Only four of the five stations contributed.
        self.assertEqual(int(rec['pp01_valid_stations']), 4)

    def test_others_require_first_three_valid(self):
        day = date(2024, 5, 1)
        # First three stations provide tx01; the trailing two provide tx02.
        records = []
        for stno in STATIONS_ALL_FIVE:
            row = {
                'stno': stno, 'obs_date': day,
                'within_station_history': True,
                'pp01': 1.0,
                'tx01': 10.0 if stno in STATIONS_FIRST_THREE else np.nan,
                'tx02': 20.0 if stno in STATIONS_FIRST_THREE else 30.0,
                'rh01': 80.0, 'wd01': 1.0, 'ps01': 1000.0,
            }
            records.append(row)
        wx = _make_weather_df(records)
        agg = _aggregate_cross_station(wx)
        rec = agg[agg['date'] == day].iloc[0]
        # tx01: only first three contribute => averaged over 3
        self.assertAlmostEqual(float(rec['tx01_avg']), 10.0, places=6)
        self.assertEqual(int(rec['tx01_valid_stations']), 3)
        # tx02: only first three contribute => averaged over 3
        self.assertAlmostEqual(float(rec['tx02_avg']), 20.0, places=6)
        self.assertEqual(int(rec['tx02_valid_stations']), 3)
        # Rainfall: all five stations have value 1.0
        self.assertAlmostEqual(float(rec['pp01_avg']), 1.0, places=6)

    def test_outside_station_history_excluded_from_aggregation(self):
        day = date(2024, 6, 1)
        records = _full_weather_day(day, base=7.0, outside_history=['72D080'])
        wx = _make_weather_df(records)
        agg = _aggregate_cross_station(wx)
        rec = agg[agg['date'] == day].iloc[0]
        # 72D080 is excluded from every count and average.
        self.assertTrue(pd.isna(rec['pp01_avg']))
        self.assertEqual(int(rec['pp01_valid_stations']), 4)
        # For others (first three include 72D080): now only 2/3 valid -> NaN
        self.assertTrue(pd.isna(rec['tx01_avg']))
        self.assertEqual(int(rec['tx01_valid_stations']), 2)


class TestSplit(unittest.TestCase):
    def test_split_keys_on_target_date(self):
        # Same target date, two horizons -> both map to the same split.
        s_h7  = Sample(origin_date=date(2022, 12, 20), target_date=date(2022, 12, 27), horizon=7)
        s_h14 = Sample(origin_date=date(2022, 12, 13), target_date=date(2022, 12, 27), horizon=14)
        # 2022-12-27 is in the train split (2016-2022 inclusive).
        self.assertEqual(split_samples([s_h7, s_h14])['train'], [s_h7, s_h14])

        # Distinct origin-date years, each sample goes to the split of its target.
        s_val  = Sample(origin_date=date(2022, 12, 28), target_date=date(2023, 1, 4), horizon=7)
        s_val2 = Sample(origin_date=date(2023, 1, 1), target_date=date(2023, 1, 8), horizon=7)
        s_test = Sample(origin_date=date(2024, 6, 1), target_date=date(2024, 6, 8), horizon=7)
        s_train = Sample(origin_date=date(2022, 6, 1), target_date=date(2022, 6, 8), horizon=7)
        out = split_samples([s_train, s_val, s_val2, s_test])
        self.assertEqual(out['train'], [s_train])
        self.assertEqual(out['val'], [s_val, s_val2])
        self.assertEqual(out['test'], [s_test])

    def test_explicit_train_range_2016_to_2022(self):
        # The constants are PUBLIC spec; ensure they encode the documented
        # contract (train NOT starting in 2014).
        self.assertEqual(SPLITS['train'][0], date(2016, 1, 1))
        self.assertEqual(SPLITS['train'][1], date(2022, 12, 31))
        self.assertEqual(SPLITS['val'][0], date(2023, 1, 1))
        self.assertEqual(SPLITS['val'][1], date(2023, 12, 31))
        self.assertEqual(SPLITS['test'][0], date(2024, 1, 1))
        self.assertEqual(SPLITS['test'][1], date(2025, 12, 31))


class TestTargetNotFilled(unittest.TestCase):
    def test_target_uses_raw_observation_no_fill(self):
        # Build a panel where exactly one candidate's target lands on a day
        # marked strict_target_usable=False.  The sample must be excluded
        # (raw, no ffill / no bfill), so the resulting kept list cannot
        # contain that (origin, target).
        start = date(2022, 5, 1)
        bao2_dates = [start + timedelta(days=i) for i in range(50)]
        bao2 = _make_bao2_df(bao2_dates, value=60.0)
        bad_day = start + timedelta(days=20)
        bao2.loc[bao2['data_date'] == bad_day, 'strict_target_usable'] = False
        wx_records = []
        for d in bao2_dates:
            wx_records.extend(_full_weather_day(d, base=1.0))
        wx = _make_weather_df(wx_records)
        panel = build_calendar_panel(bao2, wx)

        samples, counts = build_samples(panel, horizons=(7,))
        # The bad candidate has origin = bad_day - 7 (inside the calendar)
        # and its target is bad_day (non-strict).  Verify it never appears
        # in the kept list.
        expected_bad_origin = bad_day - timedelta(days=7)
        bad_pair = (expected_bad_origin, bad_day)
        kept_pairs = {(s.origin_date, s.target_date) for s in samples[7]}
        self.assertNotIn(bad_pair, kept_pairs)
        self.assertTrue(bool(panel.loc[panel['date'] == bad_day, 'strict_target_usable'].iloc[0]) is False,
                        'bad_day must remain marked strict_target_usable=False in panel')
        # And there must have been at least one excluded target storage row.
        self.assertGreaterEqual(counts['excluded_storage_target_missing'], 1)

    def test_window_storage_strictly_required_no_fill(self):
        # Only the first 13 of the 14 input days have storage; one is missing.
        start = date(2022, 5, 1)
        # Bao2 covers 13 out of the 14 days, then continues.
        present_dates = [start + timedelta(days=i) for i in range(13)] + \
                        [start + timedelta(days=14 + i) for i in range(40)]
        bao2 = _make_bao2_df(present_dates, value=55.0)
        wx_records = []
        for d in present_dates:
            wx_records.extend(_full_weather_day(d, base=1.0))
        # Fill the gap day with weather so only storage is missing.
        wx_records.extend(_full_weather_day(start + timedelta(days=13), base=1.0))
        wx = _make_weather_df(wx_records)
        panel = build_calendar_panel(bao2, wx)

        # Target a sample with h=14 and origin = day_after_gap - 14 so the
        # 14-day window straddles the missing storage day.
        samples, counts = build_samples(panel, horizons=(14,))
        self.assertGreater(counts['excluded_storage_window_incomplete'], 0)


class TestWindowGeometry(unittest.TestCase):
    def test_window_is_exactly_14_days_inclusive(self):
        # Build a continuous 40-day valid panel.
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        bao2 = _make_bao2_df(days, value=70.0)
        wx = _make_weather_df([rec for d in days for rec in _full_weather_day(d, base=1.0)])
        panel = build_calendar_panel(bao2, wx)

        # Use an origin mid-window.
        s = Sample(origin_date=date(2024, 1, 20), target_date=date(2024, 1, 27), horizon=7)
        X, y, _, _ = assemble_xy(panel, [s])
        self.assertEqual(X.shape, (1, WINDOW_LEN * len(FEATURE_COLS)))
        # 14 days x 7 features == 98.
        self.assertEqual(X.shape[1], 98)
        # Targets are raw values, not zero-filled.
        self.assertEqual(float(y[0]), 70.0)


class TestMetrics(unittest.TestCase):
    def test_mape_excludes_zero_actuals_and_counts_them(self):
        y_true = np.array([0.0, 50.0, 100.0, 25.0])
        y_pred = np.array([5.0, 45.0, 90.0, 30.0])
        m = evaluate(y_true, y_pred)
        self.assertEqual(m['n'], 4)
        self.assertEqual(m['n_zero_actual_excluded_from_mape'], 1)
        self.assertEqual(m['n_nonzero_for_mape'], 3)
        # MAPE computed over non-zero only: mean(|5/50, 10/100, 5/25|) * 100
        expected = float(np.mean([5/50, 10/100, 5/25])) * 100.0
        self.assertAlmostEqual(m['mape_percent'], expected, places=6)
        # RMSE and MAE are in percentage points (no division by 100).
        expected_mae = float(np.mean([5, 5, 10, 5]))
        self.assertAlmostEqual(m['mae_pp'], expected_mae, places=6)

    def test_low_storage_regime_reported_not_shortage(self):
        # Below the 30 pp threshold is a regime tag, NOT a shortage event.
        y_true = np.array([10.0, 20.0, 29.999, 30.0, 80.0])
        y_pred = np.array([15.0, 25.0, 35.0, 30.0, 80.0])
        m = evaluate(y_true, y_pred)
        self.assertEqual(m['low_storage_count'], 3)
        # Ensure no output key suggests a 'shortage' label.
        forbidden = [k for k in m if 'shortage' in k.lower() or 'drought' in k.lower()]
        self.assertEqual(forbidden, [])

    def test_low_storage_mae_rmse_emitted_when_actual_below_threshold(self):
        # Predict with deliberately large errors on the low-storage subset.
        y_true = np.array([5.0, 15.0, 60.0, 70.0])  # 5, 15 < 30
        y_pred = np.array([15.0, 5.0, 60.0, 70.0])  # errors |10|, |10|, 0, 0
        m = evaluate(y_true, y_pred)
        self.assertEqual(m['n_low_storage'], 2)
        # MAE on the low-storage subset: (|10| + |10|) / 2 = 10.
        self.assertAlmostEqual(m['mae_pp_low_storage'], 10.0, places=6)
        # RMSE on the low-storage subset: sqrt((100 + 100) / 2) = 10.
        self.assertAlmostEqual(m['rmse_pp_low_storage'], 10.0, places=6)
        # And the global metrics remain unchanged.
        self.assertAlmostEqual(m['mae_pp'], float(np.mean([10, 10, 0, 0])), places=6)

    def test_low_storage_metrics_empty_when_no_low_actual(self):
        y_true = np.array([60.0, 70.0, 80.0, 90.0])
        y_pred = np.array([62.0, 72.0, 82.0, 92.0])
        m = evaluate(y_true, y_pred)
        self.assertEqual(m['n_low_storage'], 0)
        # No low-storage sample -> MAE/RMSE on that subset are NaN (-> empty CSV cells).
        self.assertTrue(np.isnan(m['mae_pp_low_storage']))
        self.assertTrue(np.isnan(m['rmse_pp_low_storage']))

    def test_evaluate_empty_yields_nan_that_render_as_empty_cells(self):
        # n=0 -> all metrics are NaN; in the CSV these become empty cells.
        m = evaluate(np.array([]), np.array([]))
        self.assertEqual(m['n'], 0)
        self.assertEqual(m['n_low_storage'], 0)
        for key in ('mae_pp', 'rmse_pp', 'r2', 'mape_percent',
                    'mae_pp_low_storage', 'rmse_pp_low_storage'):
            self.assertTrue(np.isnan(m[key]), msg=f'{key} should be NaN for empty input')
        # Render via pandas.to_csv and check the empty-cell convention.
        csv_text = pd.DataFrame([m]).to_csv(index=False)
        # Each NaN numeric cell becomes an empty token between commas.
        empty_cell_count = csv_text.count(',,')
        self.assertGreater(empty_cell_count, 0)


class TestScalerLeakage(unittest.TestCase):
    def test_scaler_fit_only_on_training(self):
        # Training: constant 1.0. Validation: constant 5.0.
        # If scaler were refit on val the val std would become 0 and z-scores
        # would be ill-defined; with train-only fit the val mean is far off
        # but the model's intercept absorbs the bias predictably.
        X_train = np.ones((10, 3)) * 1.0
        y_train = np.array([10.0] * 10)
        X_val = np.ones((4, 3)) * 5.0
        y_val = np.array([50.0] * 4)

        scaler = fit_scaler(X_train)
        Xtr = scaler.transform(X_train)
        Xv = scaler.transform(X_val)

        # The scaler's mean is exactly the train mean; the val transformation
        # produces non-trivial values rather than inf or nan -> proves the
        # scaler carries train statistics only.
        self.assertTrue(np.allclose(Xtr.mean(axis=0), 0.0))
        self.assertFalse(np.any(np.isinf(Xv)))
        # And the model fitted on scaled-train predicts on scaled-val without
        # any refit.
        model = fit_ridge(Xtr, y_train)
        pred_val = model.predict(Xv)
        # Targets are in the same scale as y_train (10), so val predictions
        # should NOT match val targets (50) -- this is the expected behaviour
        # of an out-of-distribution prediction, NOT a leakage artefact.
        self.assertFalse(np.allclose(pred_val, y_val))


class TestModelDeterminism(unittest.TestCase):
    def test_random_forest_is_deterministic_with_seed(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(40, 5))
        y_train = X_train @ np.array([1.0, -1.0, 0.5, 0.0, -0.5]) + 0.1 * rng.normal(size=40)
        X_test = rng.normal(size=(8, 5))
        m1 = fit_rf(X_train, y_train)
        m2 = fit_rf(X_train, y_train)
        # Float reductions across n_jobs=2 threads can vary at machine-epsilon
        # (≈1e-16); we require equality up to that bound.
        np.testing.assert_allclose(m1.predict(X_test), m2.predict(X_test),
                                   rtol=1e-12, atol=1e-12)


class TestPersistence(unittest.TestCase):
    """The persistence baseline predicts the last observed storage in the window."""

    def test_predicts_last_storage_column(self):
        # 98-dim input: 14 days * 7 features per day.  Storage is the 7th
        # feature of each day, so the very last column is the storage of the
        # most recent day in the window.
        rng = np.random.default_rng(7)
        X = rng.normal(size=(6, WINDOW_LEN * len(FEATURE_COLS)))
        # Stagger last-day storage values so we can verify the column pick.
        last_storage = np.array([42.0, 51.5, 88.0, 17.3, 99.9, 33.3])
        X[:, -1] = last_storage

        model = fit_persistence(X)
        pred = model.predict(X)
        np.testing.assert_array_equal(pred, last_storage)
        self.assertIsInstance(model, PersistenceRegressor)

    def test_persistence_uses_full_x_layout(self):
        # Predict must return a column from X, regardless of which 7-feature
        # block is the "last day" -- the implementation picks the very last
        # column.  Verify with a synthetic layout where the last column is
        # deliberately set to a sentinel.
        X = np.zeros((3, WINDOW_LEN * len(FEATURE_COLS)))
        sentinel = 1_000.0
        X[:, -1] = sentinel
        model = fit_persistence(X)
        out = model.predict(X)
        np.testing.assert_array_equal(out, np.full(3, sentinel))

    def test_persistence_rejects_predict_before_fit(self):
        m = PersistenceRegressor()
        with self.assertRaises(RuntimeError):
            m.predict(np.zeros((2, 98)))


class TestPerHorizonScaler(unittest.TestCase):
    """The StandardScaler is fit per horizon on that horizon's train only."""

    def test_scaler_fit_per_horizon_with_distinct_train_data(self):
        # h7 train: constant 1.0; h14 train: constant 5.0.  Each scaler
        # must reflect its OWN training distribution; mean after transform
        # should be (close to) zero when applied to its own train, and
        # the OTHER scaler must not silently use a different horizon's
        # statistics.
        rng = np.random.default_rng(13)
        X_h7_train = rng.normal(loc=1.0, scale=0.5, size=(50, WINDOW_LEN * len(FEATURE_COLS)))
        X_h14_train = rng.normal(loc=5.0, scale=0.5, size=(40, WINDOW_LEN * len(FEATURE_COLS)))

        scaler_h7 = fit_scaler(X_h7_train)
        scaler_h14 = fit_scaler(X_h14_train)

        # Train-side means: h7 should be approx 1, h14 approx 5.
        self.assertAlmostEqual(scaler_h7.mean_.mean(), 1.0, places=1)
        self.assertAlmostEqual(scaler_h14.mean_.mean(), 5.0, places=1)

        # Applying h7's scaler to h14's train data must NOT centre it on zero
        # (otherwise the scaler was contaminated with h14 statistics).
        centred = scaler_h7.transform(X_h14_train)
        self.assertNotAlmostEqual(float(centred.mean()), 0.0, places=1)

        # And the inverse direction: h14 scaler applied to h7 train should
        # also leave a clear non-zero mean.
        centred_other = scaler_h14.transform(X_h7_train)
        self.assertNotAlmostEqual(float(centred_other.mean()), 0.0, places=1)


class TestRunEndToEnd(unittest.TestCase):
    """Execute the public ``run`` function in a temp directory."""

    def _fabricate_inputs(self, tmp_path: Path):
        """Create a small but well-formed synthetic snapshot.

        The fixture spans 2022-01-01 .. 2024-12-31 so that all three
        splits (train=2022, val=2023, test=2024) populate at least one row
        each in the resulting metrics file.
        """
        valid_days = [date(2022, 1, 1) + timedelta(days=i) for i in range(3 * 365 + 1)]
        bao2 = _make_bao2_df(valid_days, value=72.0)
        wx = _make_weather_df([rec for d in valid_days for rec in _full_weather_day(d, base=1.5)])
        bao2_csv = tmp_path / 'bao2_daily.csv'
        wx_csv = tmp_path / 'research_weather_snapshot.csv'
        # Cast dates to ISO strings for CSV serialisation (helper produced
        # Python ``date`` objects so the .dt accessor is unavailable).
        bao2['data_date'] = bao2['data_date'].apply(lambda d: d.isoformat())
        bao2['observation_time'] = bao2['data_date'] + ' 22:00:00'
        bao2.to_csv(bao2_csv, index=False)
        wx['obs_date'] = wx['obs_date'].apply(lambda d: d.isoformat())
        wx['opened'] = wx['opened'].apply(lambda d: d.isoformat() if hasattr(d, 'isoformat') else str(d))
        wx.to_csv(wx_csv, index=False)
        return bao2_csv, wx_csv

    def test_run_produces_artifacts_and_completes_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bao2_csv, wx_csv = self._fabricate_inputs(tmp)
            out = tmp / 'artifacts'

            rc = run_baselines.run(bao2_csv=bao2_csv, weather_csv=wx_csv, output=out)
            self.assertEqual(rc, 0)
            self.assertTrue((out / COMPLETION_MARKER).exists())
            self.assertTrue((out / 'predictions.csv').exists())
            self.assertTrue((out / 'metrics.csv').exists())
            self.assertTrue((out / 'samples.csv').exists())
            self.assertTrue((out / 'daily_data_table.csv').exists())
            self.assertTrue((out / 'config.json').exists())
            # v2 artefacts: per-horizon scalers and three model files per horizon.
            for h in HORIZONS:
                self.assertTrue((out / f'scaler_ridge_h{h}.joblib').exists())
                self.assertTrue((out / f'model_ridge_h{h}.joblib').exists())
                self.assertTrue((out / f'model_random_forest_h{h}.joblib').exists())
                self.assertTrue((out / f'model_persistence_h{h}.joblib').exists())
            self.assertFalse((out / 'scaler_ridge.joblib').exists(),
                             'v1 used a single scaler_ridge.joblib; v2 must be per-horizon')

            # Predictions file at same horizon/split must contain the SAME
            # (origin_date, target_date) rows for ALL THREE models -> shared
            # samples.
            preds = pd.read_csv(out / 'predictions.csv')
            keys_cols = ['origin_date', 'target_date', 'split', 'horizon']
            keys_persistence = preds[preds['model'] == 'persistence'][keys_cols]
            keys_ridge = preds[preds['model'] == 'ridge'][keys_cols]
            keys_rf = preds[preds['model'] == 'random_forest'][keys_cols]
            self.assertEqual(
                keys_persistence.sort_values(keys_cols).reset_index(drop=True).to_dict('records'),
                keys_ridge.sort_values(keys_cols).reset_index(drop=True).to_dict('records'),
            )
            self.assertEqual(
                keys_ridge.sort_values(keys_cols).reset_index(drop=True).to_dict('records'),
                keys_rf.sort_values(keys_cols).reset_index(drop=True).to_dict('records'),
            )

            # Metrics file must carry low-storage MAE/RMSE per (model, horizon, split).
            metrics = pd.read_csv(out / 'metrics.csv')
            self.assertIn('mae_pp_low_storage', metrics.columns)
            self.assertIn('rmse_pp_low_storage', metrics.columns)
            self.assertIn('n_low_storage', metrics.columns)
            for h in HORIZONS:
                for split in ('train', 'val', 'test'):
                    sub = metrics[(metrics['horizon'] == h) & (metrics['split'] == split)]
                    self.assertEqual(set(sub['model']), {'persistence', 'ridge', 'random_forest'},
                                     msg=f'expected 3 models for ({h}, {split})')

            # Re-running without --force is a no-op; the marker is preserved.
            config_path = out / 'config.json'
            before = config_path.read_text()
            rc2 = run_baselines.run(bao2_csv=bao2_csv, weather_csv=wx_csv, output=out)
            self.assertEqual(rc2, 0)
            self.assertEqual(config_path.read_text(), before)


if __name__ == '__main__':
    unittest.main()
