"""LSTM baseline for Bao2 reservoir storage prediction.

This module is a *research reconstruction* (NOT a June-reproduction).  It
consumes the locked sample set emitted by ``recovery_baselines_v2`` and trains
a single LSTM per horizon with explicit constraints so the experiment cannot
accidentally become a test-set model selection exercise.

Specification (binding; do NOT loosen without consulting the lead engineer):

1.  The (horizon, sample) set is **identical** to the baseline v2 run.  We
    load ``data/research/recovery_baselines_v2/daily_data_table.csv`` and
    ``samples.csv`` (rather than rebuilding the panel from bao2+weather); the
    panels and samples there are the citable version that the baselines fixed.
2.  ``assemble_xy``, ``split_samples``, ``evaluate``, ``Sample``,
    ``FEATURE_COLS``, ``HORIZONS``, ``WINDOW_LEN``, ``SPLITS``,
    ``LOW_STORAGE_THRESHOLD_PP`` from ``research.run_baselines`` are reused as-is
    -- the LSTM shares the same input layout, the same target columns, and the
    same metrics contract as the v2 baselines.  Only the model class differs.
3.  Per-horizon preprocessing:
        * Feature scaler is a ``sklearn.preprocessing.MinMaxScaler`` fit *only*
          on the train split of that horizon, on the **7 cross-day features**
          (not on 98 per-(day, feature) lags).  Concretely the scaler is fit
          on a matrix ``(n_train * WINDOW_LEN, len(FEATURE_COLS)) == (n_train
          * 14, 7)`` obtained by reshaping train rows from ``(n, 98)`` to
          ``(n, 14, 7)`` then to ``(n*14, 7)``.  ``scaler.n_features_in_``
          is therefore ``7``.  Inference-time ``transform`` follows the same
          ``(n_split * 14, 7)`` reshape and the result is reshaped back to
          ``(n_split, 14, 7)`` for the LSTM.  As a consequence, the same raw
          feature value at lag-0 and lag-13 maps to the same scaled value
          (the scaler pools across lags).
        * Val and test use the SAME fitted scaler (never re-fit).  No
          cross-horizon union of train rows.
        * Target (raw storage in percentage points) is rescaled by a fixed
          divisor ``Y_SCALE_DIVISOR = 100`` -- i.e.  ``y_scaled = y / 100`` at
          fit time and ``y_pred_pp = y_pred * 100`` at inference.  This fixed
          scaling is documented in ``config.json``; no y-scaler is fitted.
4.  Model architecture (per horizon, identical copy):

        Input  : (WINDOW_LEN=14, len(FEATURE_COLS)=7)
        LSTM   : 64 units, return_sequences=True
        Dropout: 0.2
        LSTM   : 32 units
        Dropout: 0.2
        Dense  : 1 unit, linear activation (regression head)

    Compiled with ``optimizer='adam'`` and ``loss='mse'``.
5.  Training schedule:
        * ``seed=42`` is re-applied at the **start of every horizon** via
          ``tf.keras.utils.set_random_seed`` (Python ``random``, ``numpy`` and
          TF graph seeds).  This deliberately decouples h14 from h7's training
          time / RNG state; there is **no** seed search or seed averaging,
          and the test set never participates in seed selection.
        * ``epochs=100``, ``batch_size=32``, ``shuffle=False``.
        * ``EarlyStopping(monitor='val_loss', patience=10,
          restore_best_weights=True)``.
        * ``validation_data`` is the val split of that horizon, never the
          test split.
        * Test data is **never** used for model selection, checkpointing or
          seed averaging -- it is only used to emit the held-out metrics.
        * Two CPU threads via ``TF_NUM_INTRAOP_THREADS`` /
          ``TF_NUM_INTEROP_THREADS`` environment variables (set BEFORE the TF
          import) and via ``tf.config.threading.set_*_parallelism_threads``.
6.  Artefacts under ``data/research/recovery_lstm_v2``:
        * ``config.json`` -- inputs (SHA256), program SHA256, package versions,
          exclusion counts (loaded from v2), per-horizon sample counts,
          architecture and training hyper-params, y-scale rule, completion
          timestamp.  The payload also records the fitted scaler shape
          (``n_features_in_`` = ``len(FEATURE_COLS)`` = 7) and the seed-reset
          policy.
        * ``predictions.csv`` -- per-day ``(origin_date, target_date, split,
          horizon, model='lstm', actual, predicted)`` rows.  ``actual`` and
          ``predicted`` are in **raw percentage points**, matching the v2
          baseline schema.
        * ``metrics.csv`` -- identical column set to the v2 baseline
          (model/horizon/split/n/.../mae_pp_low_storage/...).  Empty cells for
          metrics that cannot be computed (n=0).
        * ``history_h{h}.json`` -- Keras ``History.history`` dict per horizon.
        * ``model_lstm_h{h}.h5`` -- full Keras model (architecture + weights).
        * ``scaler_lstm_h{h}.joblib`` -- per-horizon MinMaxScaler, fitted on
          ``(n_train * WINDOW_LEN, len(FEATURE_COLS)) == (n_train * 14, 7)``.
        * ``daily_data_table.csv`` -- copied verbatim from v2 (locked input).
        * ``samples.csv`` -- copied verbatim from v2 (locked input).
        * ``run.log`` -- training log (epoch summaries + final metrics).
        * ``logs/<ts>_lstm_failure.log`` -- only emitted on a non-zero exit.
        * ``COMPLETED.flag`` -- sentinel; subsequent invocations refuse to
          overwrite unless ``--force`` is supplied.
7.  Pre-existing ``recovery_baselines_v1``, ``recovery_baselines_v2`` and
    ``recovery_lstm_v1`` are read but never modified.  ``recovery_lstm_v2``
    is the citable artefact for the 7-feature MinMaxScaler spec; an
    opt-in ``--force`` re-run rewrites only this directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Thread / log-level hints MUST be set BEFORE the tensorflow import.
# ---------------------------------------------------------------------------
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import joblib  # noqa: E402  (after env-var setup)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

import tensorflow as tf  # noqa: E402  (host-installed TF 2.11.0)

# Reuse the v2 baseline public API -- sample identity, feature order, split
# rules and metrics contract are identical.
from research.run_baselines import (  # noqa: E402
    FEATURE_COLS,
    HORIZONS,
    LOW_STORAGE_THRESHOLD_PP,
    SPLITS,
    WINDOW_LEN,
    Sample,
    assemble_xy,
    evaluate,
    split_samples,
)


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data'
V2_DIR = DATA_ROOT / 'research' / 'recovery_baselines_v2'
# v1 used a per-(day, feature) 98-dim MinMaxScaler; v2 corrects that to a
# 7-feature ("feature columns pooled across lags") MinMaxScaler per spec.
OUTPUT_DIR = DATA_ROOT / 'research' / 'recovery_lstm_v2'

# Seed is applied ONCE for the entire run; there is no per-horizon / per-seed
# search space.  Test set never participates in model selection.
RANDOM_SEED: int = 42

# CPU threading budget (host has plenty of cores; we deliberately cap to 2).
INTRA_OP_THREADS: int = 2
INTER_OP_THREADS: int = 2

# Fixed target scaling rule -- documented in config.json.
Y_SCALE_DIVISOR: float = 100.0
Y_SCALE_NOTE = (
    f'Raw target storage (percentage points) is divided by a constant '
    f'Y_SCALE_DIVISOR={Y_SCALE_DIVISOR:g} at fit time and multiplied back at '
    f'inference.  This fixed rule is the only y-side scaling; NO y-scaler is '
    f'fitted and NO statistics from val/test influence it.'
)

# Training schedule.
EPOCHS: int = 100
BATCH_SIZE: int = 32
ES_PATIENCE: int = 10

# Architecture (per horizon, identical copy).
ARCH = dict(
    lstm1_units=64,
    lstm1_return_sequences=True,
    dropout1_rate=0.2,
    lstm2_units=32,
    dropout2_rate=0.2,
    dense_units=1,
)

MODEL_NAME: str = 'lstm'
COMPLETION_MARKER: str = 'COMPLETED.flag'


# ---------------------------------------------------------------------------
# Small helpers (mirror baselines where possible, but local copies).
# ---------------------------------------------------------------------------

def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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
        tensorflow=_safe('tensorflow'),
        joblib=_safe('joblib'),
    )


def _set_global_determinism(seed: int) -> None:
    """Apply the seed once for the entire run (no per-horizon reseeding)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    # tf.keras.utils.set_random_seed sets python_random, numpy and tf.
    tf.keras.utils.set_random_seed(seed)


def _set_cpu_threads(intra: int = INTRA_OP_THREADS, inter: int = INTER_OP_THREADS) -> None:
    """Cap both intra- and inter-op parallelism to the documented budget."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError:
        # already initialised: that's OK; the env-var hint above is enough.
        pass


def _build_lstm(seed: int = RANDOM_SEED) -> tf.keras.Model:
    """Construct the LSTM head used by every horizon."""
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(WINDOW_LEN, len(FEATURE_COLS)), name='lstm_input')
    x = tf.keras.layers.LSTM(
        ARCH['lstm1_units'], return_sequences=ARCH['lstm1_return_sequences'],
        name='lstm_1',
    )(inp)
    x = tf.keras.layers.Dropout(ARCH['dropout1_rate'], name='dropout_1')(x)
    x = tf.keras.layers.LSTM(ARCH['lstm2_units'], name='lstm_2')(x)
    x = tf.keras.layers.Dropout(ARCH['dropout2_rate'], name='dropout_2')(x)
    out = tf.keras.layers.Dense(ARCH['dense_units'], name='dense')(x)
    model = tf.keras.Model(inp, out, name='lstm')
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
    return model


def _load_panel_from_v2(v2_dir: Path) -> pd.DataFrame:
    """Read ``daily_data_table.csv`` from v2 and ensure date dtypes."""
    df = pd.read_csv(v2_dir / 'daily_data_table.csv')
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['strict_target_usable'] = df['strict_target_usable'].astype(bool)
    return df


def _load_samples_from_v2(v2_dir: Path) -> Dict[int, List[Sample]]:
    """Read ``samples.csv`` from v2 and group by horizon as Sample objects.

    The v2 ``samples.csv`` already encodes the ``split`` per row; we keep
    that split verbatim.  ``split_samples`` is then run as a *cross-check*
    on the loaded samples.
    """
    samples_df = pd.read_csv(v2_dir / 'samples.csv')
    samples_df['origin_date'] = pd.to_datetime(samples_df['origin_date']).dt.date
    samples_df['target_date'] = pd.to_datetime(samples_df['target_date']).dt.date
    samples_df['horizon'] = samples_df['horizon'].astype(int)
    samples_df['split'] = samples_df['split'].astype(str)
    by_h: Dict[int, List[Sample]] = {int(h): [] for h in HORIZONS}
    for r in samples_df.itertuples(index=False):
        if int(r.horizon) not in by_h:
            # A horizon outside our known list -- skip rather than silently extend.
            continue
        by_h[int(r.horizon)].append(
            Sample(origin_date=r.origin_date, target_date=r.target_date, horizon=int(r.horizon))
        )
    return by_h


def _reshape_flat_to_3d(X_flat: np.ndarray) -> np.ndarray:
    """(n, WINDOW_LEN * len(FEATURE_COLS)) -> (n, WINDOW_LEN, len(FEATURE_COLS))."""
    return X_flat.reshape(-1, WINDOW_LEN, len(FEATURE_COLS))


def _make_predictions_frame(
    horizon: int,
    origins: List[date], targets: List[date],
    split: str,
    y_true_pp: np.ndarray, y_pred_pp: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        'origin_date': origins,
        'target_date': targets,
        'split': split,
        'horizon': horizon,
        'model': MODEL_NAME,
        'actual': y_true_pp,
        'predicted': y_pred_pp,
    })


def _is_complete(outdir: Path) -> bool:
    return (outdir / COMPLETION_MARKER).exists()


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    v2_dir: Path = V2_DIR,
    output: Path = OUTPUT_DIR,
    force: bool = False,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose_fit: int = 1,
) -> int:
    """End-to-end execution; returns process exit code.

    On any uncaught exception a ``logs/<ts>_lstm_failure.log`` file is written
    under ``output`` so the failure is preserved even though we never overwrite
    ``COMPLETED.flag``.
    """
    output = _ensure_dir(output)
    log_dir = _ensure_dir(output / 'logs')

    # Set up a logger that always writes to run.log inside the output dir.
    logger = logging.getLogger('run_lstm')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_file = output / 'run.log'
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(sh)

    try:
        return _run_locked(
            v2_dir=v2_dir, output=output, force=force,
            epochs=epochs, batch_size=batch_size, verbose_fit=verbose_fit,
            logger=logger,
        )
    except BaseException as exc:  # noqa: BLE001  we want to log EVERY failure
        ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = log_dir / f'{ts}_lstm_failure.log'
        try:
            path.write_text(
                'LSTM run failed before COMPLETED.flag could be written.\n'
                f'Exception type: {type(exc).__name__}\n'
                f'Exception text: {exc}\n'
                f'Traceback:\n{__import__("traceback").format_exc()}\n',
                encoding='utf-8',
            )
            logger.exception('run_lstm failed; log saved to %s', path)
        finally:
            # Re-raise so the caller still sees the exception.
            raise


def _run_locked(
    *,
    v2_dir: Path,
    output: Path,
    force: bool,
    epochs: int,
    batch_size: int,
    verbose_fit: int,
    logger: logging.Logger,
) -> int:
    if _is_complete(output) and not force:
        logger.info(
            '[run_lstm] %s already completed (%s present); refusing to overwrite. '
            'Use --force to redo.', output, COMPLETION_MARKER,
        )
        return 0

    if _is_complete(output) and force:
        (output / COMPLETION_MARKER).unlink()

    # --- 1. Reference v2 inputs ------------------------------------------
    daily_v2 = v2_dir / 'daily_data_table.csv'
    samples_v2 = v2_dir / 'samples.csv'
    daily_sha = _digest(daily_v2) if daily_v2.exists() else 'missing'
    samples_sha = _digest(samples_v2) if samples_v2.exists() else 'missing'

    # --- 2. Load + persist locked v2 inputs -----------------------------
    panel = _load_panel_from_v2(v2_dir)
    samples_by_h = _load_samples_from_v2(v2_dir)

    # Snapshot the locked v2 inputs into our output, so the run is fully
    # self-describing.
    shutil.copy2(daily_v2, output / 'daily_data_table.csv')
    shutil.copy2(samples_v2, output / 'samples.csv')

    exclusion_counts_loaded = {
        # The v2 excluded counts describe why a CANDIDATE was dropped, not the
        # counts of KEPT samples.  We only emit kept_total / per-horizon
        # n_samples here; the v2 config.json carries the full exclusion table
        # and we mirror that into ours below.
        'exclusion_counts_source': 'recovery_baselines_v2/config.json',
    }
    try:
        v2_config = json.loads((v2_dir / 'config.json').read_text())
        exclusion_counts_loaded['v2_kept_total'] = v2_config.get(
            'exclusion_counts', {}
        ).get('kept_total')
    except Exception:  # noqa: BLE001
        v2_config = {}

    per_horizon_sample_counts: Dict[str, int] = {
        f'h{h}': len(samples_by_h[h]) for h in HORIZONS
    }

    # --- 3. Apply CPU cap exactly ONCE (env-var + tf.config) -----------
    _set_cpu_threads()
    logger.info(
        'CPU threads: intra_op=%d inter_op=%d (budgeted via env-var + tf.config).',
        INTRA_OP_THREADS, INTER_OP_THREADS,
    )

    # --- 4. Per-horizon assemble + fit + predict ------------------------
    # Re-seed inside the loop so h14 does not depend on h7's RNG state or
    # training time.  There is NO seed search and the test set never enters
    # the seed decision.
    history_per_h: Dict[int, Dict[str, list]] = {}
    metrics_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    artefact_index: Dict[str, object] = {
        'config': [], 'history': [], 'models': [], 'scalers': [],
    }
    scaler_n_features_per_h: Dict[int, int] = {}

    for h in HORIZONS:
        # ----- Per-horizon seed reset -----
        _set_global_determinism(RANDOM_SEED)
        logger.info(
            'horizon=%d: seed=%d re-applied (per-horizon reset policy).',
            h, RANDOM_SEED,
        )

        sample_list = list(samples_by_h[h])
        partitions = split_samples(sample_list)
        # Cross-check: partitions must agree with v2's ``split`` column for
        # every kept sample.  Emit a warning rather than silently disagreeing.
        v2_split_lookup: Dict[Tuple[date, date], str] = {}
        v2_samples_df = pd.read_csv(v2_dir / 'samples.csv')
        v2_samples_df['origin_date'] = pd.to_datetime(
            v2_samples_df['origin_date']).dt.date
        v2_samples_df['target_date'] = pd.to_datetime(
            v2_samples_df['target_date']).dt.date
        for r in v2_samples_df.itertuples(index=False):
            if int(r.horizon) == h:
                v2_split_lookup[(r.origin_date, r.target_date)] = r.split

        for split_name in ('train', 'val', 'test'):
            for s in partitions.get(split_name, []):
                key = (s.origin_date, s.target_date)
                ours = split_name
                theirs = v2_split_lookup.get(key)
                if theirs is not None and theirs != ours:
                    logger.warning(
                        'horizon=%d split mismatch for %s: v2=%s ours=%s',
                        h, key, theirs, ours,
                    )
                elif theirs is None:
                    logger.warning(
                        'horizon=%d sample %s not present in v2 samples.csv',
                        h, key,
                    )

        X_flat, y, origins, targets = assemble_xy(panel, sample_list)
        n = len(sample_list)
        if n == 0:
            logger.warning('horizon=%d has no samples; skipping.', h)
            history_per_h[h] = {}
            continue

        # Index lookup by (origin, target) -> row.
        rows_idx: Dict[Tuple[date, date], int] = {
            (o, t): i for i, (o, t) in enumerate(zip(origins, targets))
        }
        train_part = partitions.get('train', [])
        val_part = partitions.get('val', [])
        test_part = partitions.get('test', [])
        train_rows = [rows_idx[(s.origin_date, s.target_date)] for s in train_part]
        val_rows = [rows_idx[(s.origin_date, s.target_date)] for s in val_part]
        test_rows = [rows_idx[(s.origin_date, s.target_date)] for s in test_part]

        if not train_rows:
            logger.warning('horizon=%d has empty train split; skipping.', h)
            history_per_h[h] = {}
            continue

        # ----- 4a. Train-only MinMaxScaler on the 7 cross-day features -
        # Spec: scaler is fit on the 7 daily features pooled across lags,
        # NOT on the 98 per-(day, feature) lags.  Reshape the train rows from
        # (n_train, WINDOW_LEN * 7) -> (n_train * WINDOW_LEN, 7) so a single
        # raw value at lag-0 and lag-13 of the same feature column lands in
        # the same scaled bucket.
        scaler = MinMaxScaler()
        X_train_flat = X_flat[train_rows]
        X_train_flat = np.nan_to_num(X_train_flat, nan=0.0)
        X_train_2d = X_train_flat.reshape(-1, len(FEATURE_COLS))  # (n_train * 14, 7)
        scaler.fit(X_train_2d)
        # Hard guardrail: the fitted scaler must address 7 columns.
        assert scaler.n_features_in_ == len(FEATURE_COLS), (
            f'horizon={h}: scaler.n_features_in_={scaler.n_features_in_} != '
            f'{len(FEATURE_COLS)} (spec contract violation)'
        )
        scaler_n_features_per_h[h] = int(scaler.n_features_in_)
        scaler_path = output / f'scaler_lstm_h{h}.joblib'
        joblib.dump(scaler, scaler_path)
        artefact_index['scalers'].append(scaler_path.name)

        # Inference: same scaler; reshape (n_split, 98) -> (n_split * 14, 7)
        # -> transform -> reshape back to (n_split, 14, 7).
        def _scaled_3d(rows: List[int]) -> np.ndarray:
            if not rows:
                return np.empty(
                    (0, WINDOW_LEN, len(FEATURE_COLS)), dtype=np.float32,
                )
            sub = X_flat[rows]
            sub = np.nan_to_num(sub, nan=0.0)
            sub_2d = sub.reshape(-1, len(FEATURE_COLS))   # (n_split * 14, 7)
            scaled_2d = scaler.transform(sub_2d)           # (n_split * 14, 7)
            scaled_3d = scaled_2d.reshape(-1, WINDOW_LEN, len(FEATURE_COLS))
            return scaled_3d.astype(np.float32)

        X_train_3d = _scaled_3d(train_rows)
        X_val_3d = _scaled_3d(val_rows)
        X_test_3d = _scaled_3d(test_rows)

        # y fixed-scaling (no y-scaler).
        y_all = y / Y_SCALE_DIVISOR  # 0..1.
        y_train = y_all[train_rows]
        y_val = y_all[val_rows]
        y_test = y_all[test_rows]

        # ----- 4b. Build + train the LSTM ----------------------------
        logger.info('horizon=%d: build model (seed=%d)', h, RANDOM_SEED)
        model = _build_lstm(seed=RANDOM_SEED)
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ES_PATIENCE, restore_best_weights=True,
        )
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if val_rows:
            validation_data = (X_val_3d, y_val)
        else:
            # No val split at this horizon: call fit without validation_data
            # so the EarlyStopping monitor never fires.  The cap is still
            # ``epochs`` -- no test-based selection.
            es = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=ES_PATIENCE, restore_best_weights=True,
            )

        t0 = datetime.now(timezone.utc)
        hist = model.fit(
            X_train_3d, y_train,
            validation_data=validation_data,
            epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose_fit,
            callbacks=[es],
        )
        train_seconds = (datetime.now(timezone.utc) - t0).total_seconds()
        history_per_h[h] = {k: list(map(float, v)) for k, v in hist.history.items()}
        logger.info(
            'horizon=%d: training finished in %.1fs (epochs run=%d)',
            h, train_seconds, len(history_per_h[h].get('loss', [])),
        )

        model_path = output / f'model_lstm_h{h}.h5'
        try:
            model.save(str(model_path))
        except Exception:  # noqa: BLE001  -- fall back to weights-only on the very rare h5 fail.
            model_path = output / f'model_lstm_h{h}_weights.h5'
            model.save_weights(str(model_path))
        artefact_index['models'].append(model_path.name)

        history_path = output / f'history_h{h}.json'
        history_path.write_text(json.dumps(
            {k: list(map(float, v)) for k, v in hist.history.items()},
            ensure_ascii=False, indent=2,
        ))
        artefact_index['history'].append(history_path.name)

        # ----- 4c. Evaluate on train/val/test -------------------------
        split_payload = {
            'train': (train_rows, train_part, X_train_3d),
            'val':   (val_rows, val_part, X_val_3d),
            'test':  (test_rows, test_part, X_test_3d),
        }

        for split_name in ('train', 'val', 'test'):
            rows, part, X_3d = split_payload[split_name]
            if not rows:
                continue
            y_pred_scaled = np.asarray(model.predict(X_3d, verbose=0), dtype=np.float64).flatten()
            y_pred_pp = y_pred_scaled * Y_SCALE_DIVISOR
            y_true_pp = np.asarray(y[rows], dtype=np.float64)

            split_y_origins = [s.origin_date for s in part]
            split_y_targets = [s.target_date for s in part]
            prediction_frames.append(_make_predictions_frame(
                h, split_y_origins, split_y_targets, split_name, y_true_pp, y_pred_pp,
            ))

            m = evaluate(y_true_pp, y_pred_pp)
            m.update(model=MODEL_NAME, horizon=h, split=split_name)
            metrics_rows.append(m)

    # --- 5. Persist metrics + predictions + config + sentinel ----------
    METRIC_COLUMNS = (
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
    metrics_df.to_csv(output / 'metrics.csv', index=False)

    if prediction_frames:
        preds_df = pd.concat(prediction_frames, ignore_index=True)
    else:
        preds_df = pd.DataFrame(
            columns=['origin_date', 'target_date', 'split', 'horizon', 'model', 'actual', 'predicted']
        )
    preds_df.to_csv(output / 'predictions.csv', index=False)

    code_path = Path(__file__).resolve()
    code_sha = _digest(code_path)
    now = datetime.now(timezone.utc)

    config_payload = dict(
        fixed_config=dict(
            window_len=WINDOW_LEN,
            input_features_per_day=list(FEATURE_COLS),
            input_dim=WINDOW_LEN * len(FEATURE_COLS),
            horizons=list(HORIZONS),
            splits={k: [v[0].isoformat(), v[1].isoformat()] for k, v in SPLITS.items()},
            low_storage_threshold_pp=LOW_STORAGE_THRESHOLD_PP,
            model_name=MODEL_NAME,
            architecture=ARCH,
            training=dict(
                optimizer='adam',
                loss='mse',
                seed=RANDOM_SEED,
                seed_reset_policy='per-horizon (set_random_seed invoked at the '
                                  'start of every horizon so h14 does not '
                                  'depend on h7 RNG/fit history)',
                seed_search=False,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                early_stopping=dict(
                    monitor='val_loss',
                    patience=ES_PATIENCE,
                    restore_best_weights=True,
                ),
                cpu_threads=dict(
                    intra_op=INTRA_OP_THREADS,
                    inter_op=INTER_OP_THREADS,
                ),
                y_scale=dict(
                    divisor=Y_SCALE_DIVISOR,
                    note=Y_SCALE_NOTE,
                ),
                feature_scaler=dict(
                    type='sklearn.preprocessing.MinMaxScaler',
                    fit=(
                        'train split of the same horizon only; '
                        'never across horizons; never on val/test; '
                        'fit matrix is (n_train * WINDOW_LEN, '
                        'len(FEATURE_COLS)) == (n_train * 14, 7) so '
                        'scaler.n_features_in_ == 7'
                    ),
                    n_features_in_expected=len(FEATURE_COLS),
                    note='A single 7-feature scaler per horizon; the same raw '
                         'value at lag-0 and lag-13 must map to the same '
                         'scaled value (verified in test_run_lstm).',
                ),
            ),
            forbidden_practices=dict(
                test_set_used_for_model_selection=False,
                test_set_used_for_seed_tuning=False,
                cross_horizon_train_union=False,
                y_scaler_fit=False,
            ),
            note=(
                'This run is a research reconstruction (NOT a June-reproduction). '
                'Samples are locked to recovery_baselines_v2; val and test sets '
                'are aligned by (origin_date, target_date, horizon) with that '
                'baseline.  Only the model class differs.'
            ),
        ),
        input_files=dict(
            v2_dir=str(v2_dir),
            v2_daily_data_table=str(daily_v2),
            v2_daily_data_table_sha256=daily_sha,
            v2_samples=str(samples_v2),
            v2_samples_sha256=samples_sha,
            v2_config=str(v2_dir / 'config.json'),
        ),
        code_sha256=code_sha,
        package_versions=_package_versions(),
        exclusion_counts_loaded=exclusion_counts_loaded,
        per_horizon_sample_counts=per_horizon_sample_counts,
        per_horizon_scaler_n_features_in={
            f'h{h}': scaler_n_features_per_h.get(h) for h in HORIZONS
        },
        scaler_contract=dict(
            expected_n_features_in=len(FEATURE_COLS),
            # 7 means the scaler pools the 7 cross-day feature columns across
            # all 14 lags; the same raw value at lag-0 and lag-13 must map to
            # the same scaled value.
            verifications=(
                'test_run_lstm.TestScalerTrainOnly: '
                '(a) n_features_in_ == len(FEATURE_COLS) == 7; '
                '(b) identical raw value at lag-0 vs lag-13 maps to identical '
                'scaled value; '
                '(c) val rows below train_min yield strictly negative scaled '
                'values; val rows above train_max yield >1.'
            ),
        ),
        artefacts=dict(
            metrics='metrics.csv',
            predictions='predictions.csv',
            daily_data_table='daily_data_table.csv (copy of v2)',
            samples='samples.csv (copy of v2)',
            histories=[f'history_h{h}.json' for h in HORIZONS],
            models=[f'model_lstm_h{h}.h5' for h in HORIZONS],
            scalers=[f'scaler_lstm_h{h}.joblib' for h in HORIZONS],
        ),
        generated_at_utc=now.isoformat(),
    )
    (output / 'config.json').write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False)
    )

    (output / COMPLETION_MARKER).write_text(json.dumps(dict(
        completed_at_utc=now.isoformat(),
        code_sha256=code_sha,
        v2_daily_data_table_sha256=daily_sha,
        v2_samples_sha256=samples_sha,
        per_horizon_sample_counts=per_horizon_sample_counts,
        per_horizon_scaler_n_features_in={
            f'h{h}': scaler_n_features_per_h.get(h) for h in HORIZONS
        },
        random_seed=RANDOM_SEED,
        epochs=epochs,
        batch_size=batch_size,
    ), indent=2, ensure_ascii=False))

    logger.info('[run_lstm] completed -> %s', output)
    for h in HORIZONS:
        n = per_horizon_sample_counts.get(f'h{h}', 0)
        nfeat = scaler_n_features_per_h.get(h)
        logger.info(
            'horizon=%d n_samples=%d scaler.n_features_in_=%s',
            h, n, nfeat,
        )
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-dir', type=Path, default=V2_DIR,
                        help='Source directory holding recovery_baselines_v2 outputs')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR,
                        help='Target output directory (will be created if missing)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if COMPLETED.flag already exists.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--verbose', type=int, default=1,
                        help='Keras fit verbosity (0=silent, 1=progress bar, 2=line/Epoch).')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(
        v2_dir=args.v2_dir,
        output=args.output,
        force=args.force,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose_fit=args.verbose,
    )


if __name__ == '__main__':
    raise SystemExit(main())
