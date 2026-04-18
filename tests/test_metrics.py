#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 models/utils_metrics.py 的評估指標計算

執行：pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 把 models/ 加入 Python 路徑
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models"))

from utils_metrics import mae, rmse, nse, kge, pbias, evaluate_all


# =============================================================================
# MAE / RMSE
# =============================================================================
class TestMAE:
    def test_perfect_prediction_returns_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mae(y, y) == 0.0

    def test_constant_error(self):
        y = np.array([100.0, 200.0, 300.0])
        yhat = y + 10.0
        assert mae(y, yhat) == pytest.approx(10.0)

    def test_ignores_nan(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        yhat = np.array([1.0, 2.0, 3.0, 5.0])
        # 只有最後一筆 (|4-5|=1) 進入計算（第三筆被 NaN 過濾）
        assert mae(y, yhat) == pytest.approx(1.0 / 3.0)


class TestRMSE:
    def test_perfect_prediction_returns_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        assert rmse(y, y) == 0.0

    def test_penalizes_large_errors_more_than_mae(self):
        y = np.array([0.0, 0.0, 0.0])
        yhat = np.array([1.0, 1.0, 10.0])
        # MAE = 4.0, RMSE = √(102/3) ≈ 5.83
        assert rmse(y, yhat) > mae(y, yhat)


# =============================================================================
# NSE - Nash-Sutcliffe Efficiency
# =============================================================================
class TestNSE:
    def test_perfect_prediction_nse_equals_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nse(y, y) == pytest.approx(1.0)

    def test_predicting_mean_gives_nse_zero(self):
        """NSE=0 的定義：模型等於「永遠預測訓練集均值」"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yhat = np.full_like(y, y.mean())
        assert nse(y, yhat) == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean_gives_negative_nse(self):
        """比「猜均值」還差 → NSE < 0（論文中是不合格的）"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yhat = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        assert nse(y, yhat) < 0


# =============================================================================
# KGE - Kling-Gupta Efficiency
# =============================================================================
class TestKGE:
    def test_perfect_prediction_kge_equals_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert kge(y, y) == pytest.approx(1.0)

    def test_slight_underestimation_gives_reasonable_kge(self):
        """
        注意：當預測值全部相同時（std=0），相關係數是 NaN，KGE 會 NaN。
        這裡改用「整體縮放 0.9 倍」的情境，結果應該是 KGE 接近但小於 1。
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yhat = y * 0.9
        score = kge(y, yhat)
        assert 0.5 < score < 1.0, f"KGE 值超出預期範圍：{score}"

    def test_constant_prediction_returns_nan(self):
        """
        當預測值全都一樣（如均值），std=0 → 相關係數未定義 → KGE 為 NaN。
        這是合理的數學行為，記錄起來避免未來誤改。
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yhat = np.full_like(y, y.mean())
        score = kge(y, yhat)
        assert np.isnan(score)


# =============================================================================
# PBIAS - Percent Bias
# =============================================================================
class TestPBIAS:
    def test_no_bias_returns_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        assert pbias(y, y) == pytest.approx(0.0)

    def test_systematic_underestimation_gives_negative(self):
        """模型低估 → PBIAS < 0"""
        y = np.array([100.0, 200.0, 300.0])
        yhat = y * 0.9  # 低估 10%
        assert pbias(y, yhat) == pytest.approx(-10.0)

    def test_systematic_overestimation_gives_positive(self):
        """模型高估 → PBIAS > 0"""
        y = np.array([100.0, 200.0, 300.0])
        yhat = y * 1.2  # 高估 20%
        assert pbias(y, yhat) == pytest.approx(20.0)


# =============================================================================
# evaluate_all - 整合函式
# =============================================================================
class TestEvaluateAll:
    def test_returns_all_expected_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_all(y, y, label="test")
        expected_keys = {"label", "n", "MAE", "RMSE", "NSE", "KGE", "PBIAS_%"}
        assert set(result.keys()) == expected_keys

    def test_n_reflects_non_nan_count(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        yhat = np.array([1.0, 2.0, 3.0, 4.0])
        result = evaluate_all(y, yhat)
        assert result["n"] == 3


# =============================================================================
# 錯誤處理
# =============================================================================
class TestErrorHandling:
    def test_mismatched_lengths_raises(self):
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="長度不一致"):
            mae(y, yhat)

    def test_accepts_python_list(self):
        """應該能吃 list，不只 numpy array"""
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_accepts_pandas_series(self):
        """應該能吃 pandas Series"""
        pd = pytest.importorskip("pandas")
        y = pd.Series([1.0, 2.0, 3.0])
        yhat = pd.Series([1.0, 2.0, 3.0])
        assert mae(y, yhat) == 0.0
