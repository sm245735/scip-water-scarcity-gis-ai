#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試訓練集 CSV 的結構正確性

前置條件：先跑過 `python models/01_build_training_set.py`
產出 `data/寶二訓練集_v1.csv`

若 CSV 不存在，測試會自動跳過（不會紅色失敗）。

執行：pytest tests/test_training_set.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = PROJECT_ROOT / "data" / "寶二訓練集_v1.csv"

# 如果資料檔還沒產生，整組測試跳過（不視為失敗）
pytestmark = pytest.mark.skipif(
    not TRAINING_CSV.exists(),
    reason=f"訓練集尚未產生（請先跑 01_build_training_set.py）：{TRAINING_CSV}",
)


@pytest.fixture(scope="module")
def df():
    """載入訓練集，整個檔案共用（避免每個測試都重讀）"""
    return pd.read_csv(TRAINING_CSV, parse_dates=["data_date"])


# ============================================================================
# 結構檢查
# ============================================================================
class TestStructure:
    def test_has_expected_columns(self, df):
        expected = {
            "data_date", "year", "month", "day_of_year",
            "doy_sin", "doy_cos",
            "basin_rainfall_tccip_mm",
            "rainfall_7d_sum",        # 衍生特徵：7 天累積降雨
            "rainfall_30d_sum",       # 衍生特徵：30 天累積降雨
            "storage_diff_1d",       # 衍生特徵：前日蓄水變化
            "basin_rainfall_self_mm",
            "inflow_cms", "outflow_cms",
            "effective_storage",
            "is_imputed_rainfall_self",
            "is_imputed_inflow",
            "is_imputed_outflow",
            "is_imputed_storage",
        }
        assert set(df.columns) == expected

    def test_label_is_present(self, df):
        """effective_storage 存在於欄位中，LSTM 訓練時明確指定即可"""
        assert "effective_storage" in df.columns


# ============================================================================
# 日期連續性（LSTM 不能有斷日）
# ============================================================================
class TestDateContinuity:
    def test_exactly_2922_rows(self, df):
        """2016-01-01 ~ 2023-12-31 = 2,922 天（含兩個閏年 2016/2020）"""
        assert len(df) == 2922

    def test_dates_are_sorted(self, df):
        assert df["data_date"].is_monotonic_increasing

    def test_no_duplicate_dates(self, df):
        assert df["data_date"].nunique() == len(df)

    def test_no_missing_dates_in_range(self, df):
        """確認每一天都在，沒有斷日"""
        all_days = pd.date_range("2016-01-01", "2023-12-31", freq="D")
        assert set(df["data_date"]) == set(all_days)


# ============================================================================
# 數值合理性
# ============================================================================
class TestValueSanity:
    def test_doy_sin_range(self, df):
        """sin/cos 值域一定在 [-1, 1]"""
        assert df["doy_sin"].between(-1.0, 1.0).all()
        assert df["doy_cos"].between(-1.0, 1.0).all()

    def test_rainfall_non_negative(self, df):
        """降雨量不可能是負的（缺值 NaN 例外）"""
        rain = df["basin_rainfall_tccip_mm"].dropna()
        assert (rain >= 0).all()

    def test_storage_in_plausible_range(self, df):
        """寶二有效蓄水量設計值 3,100 萬立方公尺，實際觀測值應在 0~3500 之間"""
        storage = df["effective_storage"].dropna()
        assert storage.min() >= 0
        assert storage.max() <= 3500, f"蓄水量出現異常大值：{storage.max()}"

    def test_month_range(self, df):
        assert df["month"].between(1, 12).all()

    def test_day_of_year_range(self, df):
        assert df["day_of_year"].between(1, 366).all()


# ============================================================================
# 資料品質（警告性質，不一定要通過）
# ============================================================================
class TestDataQuality:
    def test_rainfall_coverage_above_95_percent(self, df):
        """TCCIP 降雨應該覆蓋率接近 100%"""
        coverage = df["basin_rainfall_tccip_mm"].notna().mean()
        assert coverage > 0.95, f"降雨覆蓋率只有 {coverage:.1%}"

    def test_storage_coverage_above_90_percent(self, df):
        """蓄水量覆蓋率若低於 90%，LSTM 很難訓練，要先想辦法補值"""
        coverage = df["effective_storage"].notna().mean()
        assert coverage > 0.90, (
            f"蓄水量覆蓋率只有 {coverage:.1%}，需要處理缺值策略"
        )
