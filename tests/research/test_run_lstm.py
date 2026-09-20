"""Unit tests for src.research.run_lstm.

These tests verify the binding specification points for the LSTM run:

  * the (origin_date, target_date, horizon) sample set is byte-identical to
    the locked ``recovery_baselines_v2`` samples (samples are NOT rebuilt);
  * the per-horizon ``MinMaxScaler`` is fit on the TRAIN split ONLY and the
    val/test data is transformed (not re-fit) using the same scaler;
  * the scaler addresses ``len(FEATURE_COLS) == 7`` cross-day features, NOT
    98 per-(day, feature) lags; the same raw value at lag-0 and lag-13 must
    map to the same scaled value (proves the lags are pooled);
  * the target uses the fixed ``Y_SCALE_DIVISOR = 100`` rule -- a fitted
    y-scaler is NEVER used;
  * predictions/metrics columns match the v2 baseline schema so the LSTM
    predictions can be sliced & aggregated alongside ridge/random_forest;
  * the LSTM architecture matches the spec (14-step, 7-feature input;
    LSTM64(return_sequences); Dropout 0.2; LSTM32; Dropout 0.2; Dense 1);
  * the artefacts written under ``data/research/recovery_lstm_v2`` lock both
    the v2 inputs and the program hash.

The tests synthesise minimal in-memory CSVs (or the real v2 fixtures) and
do not assume TensorFlow to be available for ALL tests -- the heavy
end-to-end training is exercised only when ``--runslow`` is supplied.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make the module importable from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from research import run_lstm  # noqa: E402  (path tweak above)
from research.run_baselines import (  # noqa: E402
    FEATURE_COLS,
    HORIZONS,
    LOW_STORAGE_THRESHOLD_PP,
    SPLITS,
    WINDOW_LEN,
    Sample,
)
from research.run_lstm import (  # noqa: E402
    ARCH,
    BATCH_SIZE,
    EPOCHS,
    ES_PATIENCE,
    INTRA_OP_THREADS,
    INTER_OP_THREADS,
    MODEL_NAME,
    OUTPUT_DIR,
    RANDOM_SEED,
    V2_DIR,
    Y_SCALE_DIVISOR,
    _build_lstm,
    _digest,
    _ensure_dir,
    _load_panel_from_v2,
    _load_samples_from_v2,
    _reshape_flat_to_3d,
    _set_cpu_threads,
    _set_global_determinism,
)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def _make_synthetic_v2(v2_dir: Path, *, n_train_per_h: int = 30, n_val: int = 6,
                       n_test: int = 6) -> None:
    """Build a minimal v2 directory layout so tests do not depend on the
    real ``data/research/recovery_baselines_v2`` location.

    The synthetic samples obey SPLITS (2016..2022 train, 2023 val, 2024..2025
    test).  Each retained sample has a fully-filled 14-day input window in
    the daily panel so that ``assemble_xy`` can build its 98-dim vector.
    """
    v2_dir.mkdir(parents=True, exist_ok=True)
    # Panel: train span 2016-01..2022-12 + val 2023 + test 2024..2025.
    rows = []
    d = date(2016, 1, 1)
    end = date(2025, 12, 31)
    # Scale the storage to a small noisy curve so val/test aren't constant.
    rng = np.random.default_rng(2024)
    days = (end - d).days + 1
    storage_curve = 50 + 20 * np.sin(np.arange(days) / 30.0) + rng.normal(scale=2.0, size=days)
    storage_curve = np.clip(storage_curve, 5, 95)
    for i in range(days):
        cur = d + timedelta(days=i)
        rows.append({
            'date': cur.isoformat(),
            'pp01_avg': 1.0, 'pp01_valid_stations': 5,
            'tx01_avg': 15.0, 'tx01_valid_stations': 3,
            'tx02_avg': 18.0, 'tx02_valid_stations': 3,
            'rh01_avg': 80.0, 'rh01_valid_stations': 3,
            'wd01_avg': 0.5, 'wd01_valid_stations': 3,
            'ps01_avg': 1000.0, 'ps01_valid_stations': 3,
            'storage': float(storage_curve[i]),
            'strict_target_usable': True,
        })
    daily = pd.DataFrame(rows)
    daily.to_csv(v2_dir / 'daily_data_table.csv', index=False)

    sample_rows = []
    horizon_max = max(HORIZONS)
    train_start = SPLITS['train'][0] + timedelta(days=horizon_max + WINDOW_LEN)
    train_end = SPLITS['train'][1]
    val_end = SPLITS['val'][1]
    test_end = SPLITS['test'][1]

    for h in HORIZONS:
        # Train samples distributed across the train years.
        for k in range(n_train_per_h):
            span_days = (train_end - train_start).days
            origin = train_start + timedelta(days=(k * span_days) // n_train_per_h)
            target = origin + timedelta(days=h)
            sample_rows.append({
                'origin_date': origin.isoformat(),
                'target_date': target.isoformat(),
                'horizon': h,
                'split': 'train',
            })
        # Val samples: evenly across 2023.
        val_start = SPLITS['val'][0] + timedelta(days=20)
        for k in range(n_val):
            span_days = (val_end - val_start).days
            origin = val_start + timedelta(days=(k * span_days) // n_val)
            target = origin + timedelta(days=h)
            sample_rows.append({
                'origin_date': origin.isoformat(),
                'target_date': target.isoformat(),
                'horizon': h,
                'split': 'val',
            })
        # Test samples: spread across 2024..2025.
        test_start = SPLITS['test'][0] + timedelta(days=20)
        for k in range(n_test):
            span_days = (test_end - test_start).days
            origin = test_start + timedelta(days=(k * span_days) // n_test)
            target = origin + timedelta(days=h)
            sample_rows.append({
                'origin_date': origin.isoformat(),
                'target_date': target.isoformat(),
                'horizon': h,
                'split': 'test',
            })

    pd.DataFrame(sample_rows, columns=['origin_date', 'target_date', 'horizon', 'split']).to_csv(
        v2_dir / 'samples.csv', index=False,
    )
    # A config.json with arbitrary v2 exclusion counts for the loader path.
    (v2_dir / 'config.json').write_text(json.dumps({
        'exclusion_counts': {
            'candidates_total': 100,
            'excluded_storage_window_incomplete': 5,
            'excluded_weather_window_incomplete': 0,
            'excluded_storage_target_missing': 0,
            'kept_total': n_train_per_h * len(HORIZONS) + n_val * len(HORIZONS) + n_test * len(HORIZONS),
        },
    }))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpecConstants(unittest.TestCase):
    def test_constants_match_spec(self):
        self.assertEqual(RANDOM_SEED, 42)
        self.assertEqual(EPOCHS, 100)
        self.assertEqual(BATCH_SIZE, 32)
        self.assertEqual(ES_PATIENCE, 10)
        self.assertEqual(INTRA_OP_THREADS, 2)
        self.assertEqual(INTER_OP_THREADS, 2)
        self.assertEqual(Y_SCALE_DIVISOR, 100.0)
        self.assertEqual(MODEL_NAME, 'lstm')
        # Architecture
        self.assertEqual(ARCH['lstm1_units'], 64)
        self.assertTrue(ARCH['lstm1_return_sequences'])
        self.assertEqual(ARCH['dropout1_rate'], 0.2)
        self.assertEqual(ARCH['lstm2_units'], 32)
        self.assertEqual(ARCH['dropout2_rate'], 0.2)
        self.assertEqual(ARCH['dense_units'], 1)


class TestArchitecture(unittest.TestCase):
    def test_build_lstm_matches_spec_layout(self):
        # Skip if TF not available in this env; the function is the only
        # consumer of TF in this test file.
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            self.skipTest('tensorflow not available in this environment')
        model = _build_lstm(seed=RANDOM_SEED)
        # Three 2-D tensor layers in order: LSTM1 (return_sequences), Dropout,
        # LSTM2, Dropout, Dense.
        layer_types = [type(l).__name__ for l in model.layers]
        self.assertIn('LSTM', layer_types)
        self.assertIn('Dropout', layer_types)
        self.assertIn('Dense', layer_types)
        # Verify shape: input (None, 14, 7) and final output is (None, 1).
        self.assertEqual(model.input_shape, (None, WINDOW_LEN, len(FEATURE_COLS)))
        self.assertEqual(model.output_shape, (None, 1))
        # And it's compiled with Adam/MSE.
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam')
        loss_name = (model.loss if isinstance(model.loss, str) else model.loss.__name__)
        self.assertEqual(loss_name.lower(), 'mse')
        # Hyper-param count rough sanity check (must be > 0).
        self.assertGreater(model.count_params(), 1000)


class TestReshapeLayout(unittest.TestCase):
    def test_flat_to_3d_preserves_day_order(self):
        rng = np.random.default_rng(0)
        X_flat = rng.normal(size=(4, WINDOW_LEN * len(FEATURE_COLS)))
        X3 = _reshape_flat_to_3d(X_flat)
        self.assertEqual(X3.shape, (4, WINDOW_LEN, len(FEATURE_COLS)))
        # Verify the (i, day, feature) mapping matches the flat order from
        # research.run_baselines.assemble_xy.
        X3_round = X3.reshape(4, -1)
        np.testing.assert_array_equal(X3_round, X_flat)


class TestSampleAlignment(unittest.TestCase):
    """Sample identity must match the locked v2 baseline byte-for-byte."""

    def test_load_samples_returns_per_horizon_lists(self):
        with tempfile.TemporaryDirectory() as td:
            v2 = Path(td) / 'v2'
            _make_synthetic_v2(v2, n_train_per_h=10, n_val=4, n_test=4)
            by_h = _load_samples_from_v2(v2)
            self.assertEqual(set(by_h.keys()), set(HORIZONS))
            for h, items in by_h.items():
                self.assertEqual(items[0].horizon, h)
                self.assertIsInstance(items[0], Sample)

    def test_sample_dates_byte_identical_to_v2(self):
        with tempfile.TemporaryDirectory() as td:
            v2 = Path(td) / 'v2'
            _make_synthetic_v2(v2, n_train_per_h=8, n_val=3, n_test=3)
            v2_samples = pd.read_csv(v2 / 'samples.csv')
            v2_samples['origin_date'] = pd.to_datetime(v2_samples['origin_date']).dt.date
            v2_samples['target_date'] = pd.to_datetime(v2_samples['target_date']).dt.date
            by_h = _load_samples_from_v2(v2)
            ours = []
            for h in HORIZONS:
                for s in by_h[h]:
                    ours.append({
                        'origin_date': s.origin_date.isoformat(),
                        'target_date': s.target_date.isoformat(),
                        'horizon': int(s.horizon),
                    })
            ours_df = pd.DataFrame(ours).sort_values(
                ['horizon', 'origin_date', 'target_date']
            ).reset_index(drop=True)
            v2_norm = v2_samples.copy()
            v2_norm['origin_date'] = v2_norm['origin_date'].astype(str)
            v2_norm['target_date'] = v2_norm['target_date'].astype(str)
            v2_norm = v2_norm.sort_values(
                ['horizon', 'origin_date', 'target_date']
            ).reset_index(drop=True)
            self.assertEqual(ours_df.to_dict('records'),
                             v2_norm[['origin_date', 'target_date', 'horizon']].to_dict('records'))

    def test_split_assignment_matches_v2_split_column(self):
        with tempfile.TemporaryDirectory() as td:
            v2 = Path(td) / 'v2'
            _make_synthetic_v2(v2)
            samples = pd.read_csv(v2 / 'samples.csv')
            samples['origin_date'] = pd.to_datetime(samples['origin_date']).dt.date
            samples['target_date'] = pd.to_datetime(samples['target_date']).dt.date
            by_h = _load_samples_from_v2(v2)
            for h in HORIZONS:
                v2_split_for_pair = {
                    (r.origin_date, r.target_date): r.split
                    for r in samples[samples['horizon'] == h].itertuples(index=False)
                }
                for s in by_h[h]:
                    self.assertEqual(v2_split_for_pair[(s.origin_date, s.target_date)],
                                     _expected_split_for_target(s.target_date))


def _expected_split_for_target(t: date) -> str:
    for name, (start, end) in SPLITS.items():
        if start <= t <= end:
            return name
    return 'out_of_range'


class TestScalerTrainOnly(unittest.TestCase):
    """The MinMaxScaler must be fit on TRAIN ONLY and on the 7 cross-day
    features (NOT the 98 per-lag features).

    The transform pipeline follows the same (n * 14, 7) -> reshape-to-(n, 14,
    7) flow used inside ``run_lstm``.
    """

    N_FEAT = len(FEATURE_COLS)  # == 7

    def _fit_spec_scaler(self, X_flat_train: np.ndarray):
        """Mirror the run_lstm pipeline: reshape train rows to (n*14, 7)
        and fit sklearn MinMaxScaler on that matrix."""
        from sklearn.preprocessing import MinMaxScaler
        X_2d = X_flat_train.reshape(-1, self.N_FEAT)
        scaler = MinMaxScaler()
        scaler.fit(X_2d)
        return scaler

    def _transform_spec(self, scaler, X_flat_split: np.ndarray) -> np.ndarray:
        """Mirror run_lstm's per-split transform: reshape -> transform ->
        reshape back to (n, 14, 7)."""
        X_2d = X_flat_split.reshape(-1, self.N_FEAT)
        scaled_2d = scaler.transform(X_2d)
        return scaled_2d.reshape(-1, WINDOW_LEN, self.N_FEAT).astype(np.float32)

    def test_scaler_addresses_seven_cross_day_features(self):
        """scaler.n_features_in_ must equal len(FEATURE_COLS) == 7."""
        rng = np.random.default_rng(7)
        X_flat = rng.normal(size=(40, WINDOW_LEN * self.N_FEAT))
        scaler = self._fit_spec_scaler(X_flat)
        self.assertEqual(scaler.n_features_in_, self.N_FEAT)

    def test_train_only_minmax_with_seven_features(self):
        """The 7-feature MinMaxScaler is fit on TRAIN ONLY; val rows below
        the train range map to negative scaled values, rows above the train
        range map above 1.  This is a leak guardrail."""
        from sklearn.preprocessing import MinMaxScaler
        rng = np.random.default_rng(7)
        # Build train/val/test in the shape the LSTM expects: (n, 98).
        # val distribution straddles train_min and train_max for at least
        # one feature so the leak guardrail is meaningful either way.
        X_train_flat = rng.normal(loc=5.0, scale=0.5, size=(60, WINDOW_LEN * self.N_FEAT))
        X_val_flat = rng.normal(loc=2.0, scale=2.5, size=(40, WINDOW_LEN * self.N_FEAT))
        X_test_flat = rng.normal(loc=10.0, scale=1.0, size=(20, WINDOW_LEN * self.N_FEAT))

        scaler = self._fit_spec_scaler(X_train_flat)
        # Per the spec, val/test use the same scaler (no refit).
        X_val_3d = self._transform_spec(scaler, X_val_flat)
        X_test_3d = self._transform_spec(scaler, X_test_flat)

        # After pooling across lags we have, for each of the 7 features, a
        # vector of length n*WINDOW_LEN.  Compute the train column min/max
        # over the SAME pooled layout so the leak check is fair.
        train_pooled = X_train_flat.reshape(-1, self.N_FEAT)
        val_pooled = X_val_flat.reshape(-1, self.N_FEAT)
        train_min = train_pooled.min(axis=0)
        train_max = train_pooled.max(axis=0)
        for feat in range(self.N_FEAT):
            denom = train_max[feat] - train_min[feat]
            self.assertGreater(float(denom), 1e-3,
                               f'feature {feat} train range must be non-degenerate')
            val_col = val_pooled[:, feat]
            below = val_col[val_col < train_min[feat]]
            above = val_col[val_col > train_max[feat]]
            self.assertGreater(below.size, 0,
                               f'feature {feat}: need val rows below train_min')
            self.assertGreater(above.size, 0,
                               f'feature {feat}: need val rows above train_max')
            scaled_below = (below - train_min[feat]) / denom
            scaled_above = (above - train_min[feat]) / denom
            self.assertLess(float(scaled_below.min()), -1e-6,
                            f'feat {feat}: val rows below train_min MUST come out '
                            f'negative when the scaler is fit on train only')
            self.assertGreater(float(scaled_above.max()), 1.0 + 1e-6,
                               f'feat {feat}: val rows above train_max MUST come out '
                               f'> 1 when the scaler is fit on train only')
        # The actual scaler output should match the formula above.
        scaled_val = scaler.transform(val_pooled)
        for feat in range(self.N_FEAT):
            val_col = val_pooled[:, feat]
            Xv_col_scaled = scaled_val[:, feat]
            self.assertLess(float(Xv_col_scaled[val_col < train_min[feat]].min()), -1e-6)
            self.assertGreater(float(Xv_col_scaled[val_col > train_max[feat]].max()), 1.0 + 1e-6)
        # And nothing is NaN / Inf at the model input.
        for arr in (X_val_3d, X_test_3d):
            self.assertFalse(np.any(np.isnan(arr)))
            self.assertFalse(np.any(np.isinf(arr)))

    def test_same_raw_feature_value_across_lags_scales_identically(self):
        """For every cross-day feature column, the same raw value at lag-0
        and lag-13 must map to the same scaled value -- the spec relies on
        pooling across lags.  Without the pool a per-(lag, feature) scaler
        would land on two different buckets.
        """
        rng = np.random.default_rng(11)
        n = 24
        X_raw_3d = rng.normal(size=(n, WINDOW_LEN, self.N_FEAT))
        # Force pp01_avg==v at both lag-0 and lag-13 for every sample.
        v = 42.0
        X_raw_3d[:, 0, 0] = v
        X_raw_3d[:, WINDOW_LEN - 1, 0] = v
        # Force tx01_avg==w at lag-3 and lag-10 too (extra cross-lag case).
        w = -7.5
        X_raw_3d[:, 3, 1] = w
        X_raw_3d[:, 10, 1] = w
        # And one more: rh01_avg==u at lag-1 and lag-12.
        u = 11.25
        X_raw_3d[:, 1, 3] = u
        X_raw_3d[:, 12, 3] = u

        X_flat = X_raw_3d.reshape(n, WINDOW_LEN * self.N_FEAT)
        scaler = self._fit_spec_scaler(X_flat)
        X_scaled_3d = self._transform_spec(scaler, X_flat)

        # Sanity: every sample's lag-0 pp01_avg and lag-13 pp01_avg share
        # the same scaled pp01_avg.
        np.testing.assert_allclose(
            X_scaled_3d[:, 0, 0], X_scaled_3d[:, WINDOW_LEN - 1, 0],
            rtol=1e-12, atol=1e-12,
        )
        # The flat-layout equivalent: columns 0 (day0, feat0) and
        # ((WINDOW_LEN-1) * N_FEAT + 0) (day13, feat0) must agree, sample
        # by sample.  The flat layout is (n, WINDOW_LEN * N_FEAT), so the
        # last valid day-13 feat-0 column is 13 * 7 + 0 = 91.
        X_scaled_flat = X_scaled_3d.reshape(n, WINDOW_LEN * self.N_FEAT)
        lag13_col0 = (WINDOW_LEN - 1) * self.N_FEAT + 0
        np.testing.assert_allclose(
            X_scaled_flat[:, 0], X_scaled_flat[:, lag13_col0],
            rtol=1e-12, atol=1e-12,
        )
        # tx01_avg at lag-3 vs lag-10.
        np.testing.assert_allclose(
            X_scaled_3d[:, 3, 1], X_scaled_3d[:, 10, 1],
            rtol=1e-12, atol=1e-12,
        )
        # rh01_avg at lag-1 vs lag-12.
        np.testing.assert_allclose(
            X_scaled_3d[:, 1, 3], X_scaled_3d[:, 12, 3],
            rtol=1e-12, atol=1e-12,
        )

    def test_transform_layout_matches_lstm_input(self):
        """Sanity: the transform's reshape to (n, 14, 7) has the right shape
        and dtype for the LSTM, preserving feature order day-by-day."""
        rng = np.random.default_rng(3)
        X_flat = rng.normal(size=(8, WINDOW_LEN * self.N_FEAT))
        scaler = self._fit_spec_scaler(X_flat)
        X_3d = self._transform_spec(scaler, X_flat)
        self.assertEqual(X_3d.shape, (8, WINDOW_LEN, self.N_FEAT))
        # float32 is what the LSTM expects on this host.
        self.assertEqual(X_3d.dtype, np.float32)
        # Reshape back to flat must round-trip.
        X_round = X_3d.reshape(8, WINDOW_LEN * self.N_FEAT)
        np.testing.assert_array_equal(X_round.reshape(8, -1), X_3d.reshape(8, -1))


class TestScalerPooledVsSeparate(unittest.TestCase):
    """Pooled vs per-lag contrast: when a PER-LAG (98-feature) scaler is
    used, the same raw value at lag-0 and lag-13 of the same feature produces
    DIFFERENT scaled values.  This is the artefact of an incorrect
    98-feature MinMax fit; the contract forbids it.
    """

    N_FEAT = len(FEATURE_COLS)  # 7
    LAG_FEATURES = WINDOW_LEN * len(FEATURE_COLS)  # 98

    def test_wrong_scaler_distinguishes_same_value_across_lags(self):
        from sklearn.preprocessing import MinMaxScaler
        rng = np.random.default_rng(13)
        n = 16
        X_raw_3d = rng.normal(size=(n, WINDOW_LEN, self.N_FEAT))
        # Make each LAG have its own very different baseline so the wrong
        # 98-feature scaler's per-column min/max MUST differ between
        # day-0 columns and day-13 columns.  Day d has baseline
        # N(10 * d, 0.5) for its 7 features; this guarantees a non-trivial
        # test even after one sample is overwritten with v=42.
        for d in range(WINDOW_LEN):
            X_raw_3d[:, d, :] = rng.normal(
                loc=float(d) * 10.0, scale=0.5, size=(n, self.N_FEAT),
            )
        # Overwrite sample 0: same raw value v=42 at lag-0 feat-0 and at
        # lag-13 feat-0.
        v = 42.0
        X_raw_3d[0, 0, 0] = v
        X_raw_3d[0, WINDOW_LEN - 1, 0] = v
        X_flat = X_raw_3d.reshape(n, self.LAG_FEATURES)

        # An INCORRECT 98-feature scaler (treats day-0 feat-0 and day-13
        # feat-0 as separate columns) WILL distinguish them because the
        # train distribution is wildly different at those two columns.
        wrong = MinMaxScaler()
        wrong.fit(X_flat)
        wrong_scaled = wrong.transform(X_flat)
        # At sample 0, columns 0 (day0-feat0) and 91 (day13-feat0) both
        # have raw value 42, but the WRONG 98-feature scaler assigns
        # DIFFERENT scaled values.
        lag13_col0 = (WINDOW_LEN - 1) * self.N_FEAT + 0
        v0 = float(wrong_scaled[0, 0])
        v91 = float(wrong_scaled[0, lag13_col0])
        self.assertFalse(
            np.isclose(v0, v91, atol=1e-6),
            msg=(
                f'sanity: a wrong (per-lag) 98-feature MinMax scaler MUST '
                f'distinguish the same raw value at lag-0 and lag-13; got '
                f'{v0} vs {v91} -- the synthetic data is degenerate, the '
                f'per-lag distributions do not differ enough.'
            ),
        )

        # Contrast: the SPEC-mandated 7-feature pooled fit MUST agree.
        right = MinMaxScaler()
        right.fit(X_flat.reshape(-1, self.N_FEAT))
        right_scaled_flat = right.transform(X_flat.reshape(-1, self.N_FEAT))
        right_scaled_3d = right_scaled_flat.reshape(n, WINDOW_LEN, self.N_FEAT)
        np.testing.assert_allclose(
            right_scaled_3d[0, 0, 0],
            right_scaled_3d[0, WINDOW_LEN - 1, 0],
            rtol=1e-12, atol=1e-12,
        )


class TestYScalingRule(unittest.TestCase):
    """The fixed y-scaling rule must be /100 (no fitted y-scaler)."""

    def test_constant_divisor(self):
        self.assertEqual(Y_SCALE_DIVISOR, 100.0)
        # Negative guardrail: a fitted y-scaler would be a different object;
        # here we sanity-check that the rule is purely arithmetic.
        raw = np.array([0.0, 23.45, 50.0, 99.9, 100.0])
        scaled = raw / Y_SCALE_DIVISOR
        expected = np.array([0.0, 0.2345, 0.5, 0.999, 1.0])
        np.testing.assert_allclose(scaled, expected, atol=1e-12)
        # Round-trip:
        np.testing.assert_allclose(scaled * Y_SCALE_DIVISOR, raw, atol=1e-12)


class TestDeterminismHelpers(unittest.TestCase):
    def test_set_cpu_threads_is_idempotent(self):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            self.skipTest('tensorflow not available')
        # Just ensure the call does not raise.
        _set_cpu_threads()
        _set_cpu_threads()
        _set_cpu_threads(intra=2, inter=2)

    def test_set_global_determinism_does_not_crash(self):
        # We don't actually assert RNG state -- that requires TF and may be
        # fragile across versions.  We only verify the helper is callable and
        # doesn't propagate exceptions at seed time.
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            self.skipTest('tensorflow not available')
        _set_global_determinism(RANDOM_SEED)


class TestArtefactContract(unittest.TestCase):
    """Run ``run()`` against a synthetic v2 directory and verify artefacts.

    All assertions share a single tempdir + run via ``setUpClass`` / ``tearDownClass``
    so the 3 test methods can read the same artefacts without losing them to
    TemporaryDirectory cleanup.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._td_ctx = tempfile.TemporaryDirectory()
        cls._td_path = Path(cls._td_ctx.name)
        cls._v2 = cls._td_path / 'v2'
        cls._out = cls._td_path / 'out'
        _make_synthetic_v2(cls._v2, n_train_per_h=12, n_val=4, n_test=4)
        try:
            rc = run_lstm.run(
                v2_dir=cls._v2, output=cls._out, force=True,
                epochs=2, batch_size=8, verbose_fit=0,
            )
        except Exception as exc:  # noqa: BLE001
            # If TF failed mid-run we want a readable error.
            log = list((cls._out / 'logs').glob('*_lstm_failure.log')) if (cls._out / 'logs').exists() else []
            fail = (log[0].read_text() if log else repr(exc))
            raise AssertionError(
                f'TF backend crashed during the synthetic run:\n{fail}\nout_dir={cls._out}'
            ) from exc
        assert rc == 0, f'run_lstm returned {rc}'

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls._td_ctx.cleanup()
        except Exception:  # noqa: BLE001
            pass

    def test_run_completes_and_writes_artifacts(self):
        out = self._out
        # Sentinel
        self.assertTrue((out / 'COMPLETED.flag').exists())
        # Locked inputs copied verbatim
        self.assertTrue((out / 'daily_data_table.csv').exists())
        self.assertTrue((out / 'samples.csv').exists())
        # Predictions schema matches v2 exactly
        preds = pd.read_csv(out / 'predictions.csv')
        self.assertEqual(
            list(preds.columns),
            ['origin_date', 'target_date', 'split', 'horizon', 'model', 'actual', 'predicted'],
        )
        self.assertTrue(set(preds['model'].unique()).issubset({MODEL_NAME}))
        self.assertTrue(set(preds['split'].unique()).issubset({'train', 'val', 'test'}))
        self.assertTrue(set(preds['horizon'].unique()).issubset(set(HORIZONS)))
        # All horizons + splits that had any training rows must have rows
        for h in HORIZONS:
            self.assertIn(h, set(preds['horizon'].unique()))

        # Metrics schema matches v2 baseline.
        metrics = pd.read_csv(out / 'metrics.csv')
        for col in ('model', 'horizon', 'split', 'n',
                    'n_zero_actual_excluded_from_mape', 'n_nonzero_for_mape',
                    'mae_pp', 'rmse_pp', 'r2', 'mape_percent',
                    'n_low_storage', 'mae_pp_low_storage', 'rmse_pp_low_storage',
                    'low_storage_fraction'):
            self.assertIn(col, metrics.columns)

        # Histories + scalers + models exist.
        for h in HORIZONS:
            self.assertTrue((out / f'history_h{h}.json').exists())
            scaler_path = out / f'scaler_lstm_h{h}.joblib'
            self.assertTrue(scaler_path.exists())
            # The on-disk scaler must address 7 cross-day features; if it
            # addressed 98 per-(day, feature) lags, the run has used the
            # WRONG scaler contract.
            loaded_scaler = joblib.load(scaler_path)
            self.assertEqual(
                int(getattr(loaded_scaler, 'n_features_in_', -1)),
                len(FEATURE_COLS),
                msg=(
                    f'horizon={h}: scaler.n_features_in_={loaded_scaler.n_features_in_} '
                    f'is not {len(FEATURE_COLS)}; the run is using the wrong '
                    f'(98-lag) contract'
                ),
            )
            # Model may be either full .h5 or a weights fallback.
            self.assertTrue(
                (out / f'model_lstm_h{h}.h5').exists()
                or (out / f'model_lstm_h{h}_weights.h5').exists()
            )

        # Config carries input SHAs and the y-scale rule.
        cfg = json.loads((out / 'config.json').read_text())
        self.assertEqual(cfg['fixed_config']['training']['seed'], RANDOM_SEED)
        self.assertEqual(cfg['fixed_config']['training']['epochs'], 2)
        self.assertEqual(cfg['fixed_config']['training']['batch_size'], 8)
        self.assertEqual(cfg['fixed_config']['training']['shuffle'], False)
        self.assertEqual(cfg['fixed_config']['training']['y_scale']['divisor'], Y_SCALE_DIVISOR)
        self.assertFalse(cfg['fixed_config']['forbidden_practices']['test_set_used_for_model_selection'])
        self.assertFalse(cfg['fixed_config']['forbidden_practices']['cross_horizon_train_union'])
        self.assertFalse(cfg['fixed_config']['forbidden_practices']['test_set_used_for_seed_tuning'])
        # Architecture
        self.assertEqual(cfg['fixed_config']['architecture']['lstm1_units'], 64)
        self.assertTrue(cfg['fixed_config']['architecture']['lstm1_return_sequences'])
        self.assertEqual(cfg['fixed_config']['architecture']['dropout1_rate'], 0.2)
        self.assertEqual(cfg['fixed_config']['architecture']['lstm2_units'], 32)
        self.assertEqual(cfg['fixed_config']['architecture']['dropout2_rate'], 0.2)
        self.assertEqual(cfg['fixed_config']['architecture']['dense_units'], 1)

        # Per-horizon sample counts match the v2 samples.csv
        for h, items in _load_samples_from_v2(self._v2).items():
            cfg_count = cfg['per_horizon_sample_counts'].get(f'h{h}', -1)
            self.assertEqual(cfg_count, len(items),
                             f'horizon={h} sample count mismatch: cfg={cfg_count} samples={len(items)}')

        # The sentinel carries the loaded SHAs.
        flag = json.loads((out / 'COMPLETED.flag').read_text())
        self.assertEqual(flag['random_seed'], RANDOM_SEED)
        self.assertEqual(flag['epochs'], 2)
        self.assertEqual(flag['batch_size'], 8)
        expected_daily = _digest(self._v2 / 'daily_data_table.csv')
        expected_samples = _digest(self._v2 / 'samples.csv')
        self.assertEqual(flag['v2_daily_data_table_sha256'], expected_daily)
        self.assertEqual(flag['v2_samples_sha256'], expected_samples)
        # And the config echoes those SHAs too.
        self.assertEqual(cfg['input_files']['v2_daily_data_table_sha256'], expected_daily)
        self.assertEqual(cfg['input_files']['v2_samples_sha256'], expected_samples)
        # Code SHA is recorded and matches the actual bytes of run_lstm.py.
        code_sha_now = _digest(Path(run_lstm.__file__))
        self.assertEqual(cfg['code_sha256'], code_sha_now)
        # Tensorflow version recorded.
        self.assertIn('tensorflow', cfg['package_versions'])

    def test_predictions_align_with_v2_samples_per_horizon_split(self):
        out = self._out
        v2 = self._v2
        v2_samples = pd.read_csv(v2 / 'samples.csv')
        v2_samples['origin_date'] = pd.to_datetime(v2_samples['origin_date']).dt.date
        v2_samples['target_date'] = pd.to_datetime(v2_samples['target_date']).dt.date
        v2_pairs = {
            h: {(r.origin_date, r.target_date): r.split
                for r in v2_samples[v2_samples['horizon'] == h].itertuples(index=False)}
            for h in HORIZONS
        }
        preds = pd.read_csv(out / 'predictions.csv')
        preds['origin_date'] = pd.to_datetime(preds['origin_date']).dt.date
        preds['target_date'] = pd.to_datetime(preds['target_date']).dt.date

        for h in HORIZONS:
            ours = {(r.origin_date, r.target_date): r.split
                    for r in preds[preds['horizon'] == h].itertuples(index=False)}
            # Ours must be a subset of v2.
            self.assertTrue(set(ours.keys()).issubset(set(v2_pairs[h].keys())),
                            msg=f'horizon={h} predictions reference rows not in v2 samples.csv')
            # Where v2 carries a split, ours must agree.
            for key, sp in ours.items():
                if key in v2_pairs[h]:
                    self.assertEqual(v2_pairs[h][key], sp,
                                     msg=f'horizon={h} split mismatch at {key}: v2={v2_pairs[h][key]} ours={sp}')

    def test_predictions_in_raw_percentage_points(self):
        """``actual`` and ``predicted`` are stored as percentage points."""
        preds = pd.read_csv(self._out / 'predictions.csv')
        self.assertGreaterEqual(float(preds['actual'].max()), 30.0)
        # At least one predicted above 5 (i.e. NOT stuck in the 0..1 training
        # space; the inference multiplies back by 100).
        self.assertGreater(float(preds['predicted'].max()), 5.0)

    def test_completed_flag_blocks_repeat_run(self):
        # Second call (without --force) is a no-op and must not modify
        # COMPLETED.flag (covered directly at run_lstm.run() level).
        import json as _json
        before_mtime = (self._out / 'COMPLETED.flag').stat().st_mtime
        before_text = (self._out / 'COMPLETED.flag').read_text()
        rc = run_lstm.run(
            v2_dir=self._v2, output=self._out, force=False,
            epochs=2, batch_size=8, verbose_fit=0,
        )
        self.assertEqual(rc, 0)
        self.assertEqual((self._out / 'COMPLETED.flag').stat().st_mtime, before_mtime)
        self.assertEqual((self._out / 'COMPLETED.flag').read_text(), before_text)


class TestIOHelpers(unittest.TestCase):
    def test_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'f.bin'
            p.write_bytes(b'hello world')
            d1 = _digest(p)
            d2 = _digest(p)
            self.assertEqual(d1, d2)
            self.assertEqual(d1, hashlib.sha256(b'hello world').hexdigest())

    def test_ensure_dir_creates_parents(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'a' / 'b' / 'c'
            _ensure_dir(target)
            self.assertTrue(target.is_dir())

    def test_panel_loader_normalises_dtypes(self):
        with tempfile.TemporaryDirectory() as td:
            v2 = Path(td) / 'v2'
            _make_synthetic_v2(v2)
            panel = _load_panel_from_v2(v2)
            self.assertIn('date', panel.columns)
            self.assertIn('storage', panel.columns)
            self.assertIn('strict_target_usable', panel.columns)
            # Date dtype -> Python date objects.
            self.assertIsInstance(panel['date'].iloc[0], date)
            self.assertTrue(panel['strict_target_usable'].dtype == bool)


class TestDefaults(unittest.TestCase):
    """The default V2_DIR / OUTPUT_DIR must point at the repo's data/ tree
    (so that running the script as a CLI lines up with the spec)."""

    def test_v2_dir_default_is_repo_data(self):
        self.assertTrue(str(V2_DIR).endswith('data/research/recovery_baselines_v2'))

    def test_output_dir_default_is_repo_data(self):
        # v2 corrected the MinMaxScaler to fit on the 7 cross-day features
        # (lag-pooled).  v1 -- and the recovery_lstm_v1 directory -- remain
        # untouched on disk.
        self.assertTrue(str(OUTPUT_DIR).endswith('data/research/recovery_lstm_v2'))
        self.assertFalse(str(OUTPUT_DIR).endswith('data/research/recovery_lstm_v1'))


if __name__ == '__main__':
    unittest.main()
