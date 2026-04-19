#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Step 1：組裝寶山第二水庫 LSTM 訓練集（v2.0.0）
=============================================================================
版本歷史：
    v1.0.0 (2026-04-19)：初版，含 is_imputed 邏輯
    v2.0.0 (2026-04-19)：新增 3 個衍生特徵（rainfall_7d_sum, rainfall_30d_sum, storage_diff_1d）

設計決策：
功能：
    1. 從 reservoir_daily 撈寶二（reservoir_id=23）2016-2023 每日水情
    2. JOIN TCCIP 頭前溪集水區每日降雨（空間平均）
    3. 補齊連續日期（無條件對齊到 2,922 天）
    4. 產生 is_imputed_* 缺值標記（在補值前先標記）
    5. 執行補值策略（≤7 天缺口線性內插，>7 天保留 NaN）
    6. 加入時間週期特徵（doy_sin / doy_cos）
    7. 輸出乾淨的 CSV → data/寶二訓練集_v1.csv

設計決策：
    - 降雨以 TCCIP 空間加權平均（basin_rainfall_tccip_mm）為主要 feature
    - reservoir_daily 自帶的 basin_rainfall_mm 改名為 _self_mm 作為對照欄位（不进模型）
    - reservoir_daily 保持「事實表」語意（NULL 為 NULL，不在 DB 層補值）
    - is_imputed_* 打在訓練集 CSV 這層，隔離「分析決策」與「資料事實」
    - 時間範圍鎖 2016-01-01 ~ 2023-12-31（跟 TCCIP 對齊）

執行：
    docker exec thesis_python_dev python /app/models/01_build_training_set.py

輸出：
    data/寶二訓練集_v1.csv  (約 2,922 列 × 11 欄)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

# =============================================================================
# 路徑與設定（自動偵測專案根目錄，不要硬編碼絕對路徑）
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_SAMPLES_DIR = PROJECT_ROOT / "data_samples"
DATA_DIR.mkdir(exist_ok=True)

# DB config（優先用環境變數，符合現有 src/ 腳本的慣例）
DB_USER = os.getenv("DB_USER", "sm245735")
DB_PASS_RAW = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "thesis_analysis")

# 研究設定
RESERVOIR_ID = 23                 # 寶山第二水庫
START_DATE = "2016-01-01"
END_DATE = "2023-12-31"           # 跟 TCCIP 降雨對齊

# I/O
OUTPUT_FILE = DATA_DIR / "寶二訓練集_v1.csv"
TCCIP_CSV = DATA_SAMPLES_DIR / "新竹頭前溪集水區每日降雨量_2016_2023.csv"


# =============================================================================
# 資料載入
# =============================================================================
def fetch_reservoir_daily() -> pd.DataFrame:
    """
    從 reservoir_daily 撈寶二時間序列。

    注意：
        - 刻意不選 water_level_m / storage_rate / full_water_level_m，避免 data leakage
          （這三個欄位都是 effective_storage 的函數）
        - basin_rainfall_mm 改名為 _self_mm 跟 TCCIP 那版區分
    """
    from sqlalchemy import create_engine, text

    if not DB_PASS_RAW:
        sys.exit("❌ DB_PASSWORD 環境變數未設定，請先 export 或在 docker-compose 填入")

    db_url = (
        f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASS_RAW)}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    sql = text("""
        SELECT
            data_date,
            basin_rainfall_mm  AS basin_rainfall_self_mm,
            inflow_cms,
            outflow_cms,
            effective_storage
        FROM reservoir_daily
        WHERE reservoir_id = :rid
          AND data_date BETWEEN :start AND :end
        ORDER BY data_date
    """)

    engine = create_engine(db_url)
    try:
        df = pd.read_sql(
            sql, engine,
            params={"rid": RESERVOIR_ID, "start": START_DATE, "end": END_DATE},
        )
    finally:
        engine.dispose()

    df["data_date"] = pd.to_datetime(df["data_date"])
    return df


def load_tccip_rainfall() -> pd.DataFrame:
    """
    讀取預先計算好的頭前溪集水區 TCCIP 每日平均降雨。

    該 CSV 由 src/gis_analysis/ 相關腳本產生（ST_Union + ST_Intersects），
    共 2,922 筆（2016-01-01 ~ 2023-12-31，43 個 TCCIP 網格）。
    """
    if not TCCIP_CSV.exists():
        sys.exit(f"❌ 找不到 TCCIP 降雨檔：{TCCIP_CSV}")
    df = pd.read_csv(TCCIP_CSV, parse_dates=["data_date"])
    df = df[["data_date", "avg_rainfall"]].rename(
        columns={"avg_rainfall": "basin_rainfall_tccip_mm"}
    )
    return df


# =============================================================================
# 組裝訓練集
# =============================================================================
def build_training_set() -> pd.DataFrame:
    print("=" * 70)
    print("Step 1：組裝寶二水庫 LSTM 訓練集")
    print("=" * 70)

    print(f"\n→ 從 DB 撈寶二（reservoir_id={RESERVOIR_ID}）"
          f" {START_DATE} ~ {END_DATE}")
    df_res = fetch_reservoir_daily()
    print(f"  reservoir_daily：{len(df_res):,} 筆")

    print(f"\n→ 讀 TCCIP 頭前溪降雨 CSV")
    df_rain = load_tccip_rainfall()
    print(f"  TCCIP 頭前溪：{len(df_rain):,} 筆")

    # 合併
    df = df_res.merge(df_rain, on="data_date", how="left")

    # 強制補齊連續日期（LSTM 不能有斷日）
    all_days = pd.DataFrame({
        "data_date": pd.date_range(START_DATE, END_DATE, freq="D")
    })
    df = all_days.merge(df, on="data_date", how="left")
    expected = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1
    print(f"\n→ 補齊連續日期：{len(df):,} 筆（應為 {expected} 筆）")
    assert len(df) == expected, "連續日期補齊後筆數不對，請檢查"

    # Step A：產生 is_imputed 標記（在補值「之前」就標記好）
    # 注意：補值決策是「分析行為」而非「資料事實」，打在訓練集這層而非資料庫
    print("\n→ 產生缺值標記 is_imputed_*（補值前旗標）")
    feature_cols = [
        "basin_rainfall_tccip_mm",
        "basin_rainfall_self_mm",
        "inflow_cms",
        "outflow_cms",
        "effective_storage",
    ]
    # 對應的 flag 欄位名（符合 reviewer 建議的命名）
    flag_map = {
        "basin_rainfall_tccip_mm":  "is_imputed_rainfall_tccip",
        "basin_rainfall_self_mm":   "is_imputed_rainfall_self",
        "inflow_cms":               "is_imputed_inflow",
        "outflow_cms":              "is_imputed_outflow",
        "effective_storage":        "is_imputed_storage",
    }
    for col in feature_cols:
        df[flag_map[col]] = df[col].isna().astype(int)

    # Step B：執行補值策略
    print("\n→ 執行補值（線性內插，連續缺口 ≤7 天，>7 天保留 NaN）")
    for col in feature_cols:
        before = df[col].isna().sum()
        df[col] = df[col].interpolate(method="linear", limit=7, limit_area="inside")
        after = df[col].isna().sum()
        imputed = before - after
        print(f"    {col:30s}: 補值 {imputed:4d} 筆, 仍 NaN {after:4d} 筆")

    # Step C：補值統計（論文方法章節會用到）
    print("\n→ 補值統計（供論文方法章節引用）")
    for col, flag_col in flag_map.items():
        total_imputed = df[flag_col].sum()
        pct = total_imputed / len(df) * 100
        still_na = df[col].isna().sum()
        print(f"    {col:30s}: 補值 {total_imputed:4d} 筆 ({pct:5.1f}%), 仍 NaN {still_na:4d} 筆")

    # 時間週期特徵（LSTM 要吃得出年週期）
    df["year"] = df["data_date"].dt.year
    df["month"] = df["data_date"].dt.month
    df["day_of_year"] = df["data_date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # =========================================================
    # 衍生特徵（基於已清洗的欄位計算，可選擇性進入 LSTM）
    # =========================================================
    print("\n→ 產生衍生特徵（rolling sums 與 lag differences）")

    # 累積降雨：用 TCCIP 為主（可信度 100%）
    df["rainfall_7d_sum"] = df["basin_rainfall_tccip_mm"].rolling(window=7, min_periods=1).sum()
    df["rainfall_30d_sum"] = df["basin_rainfall_tccip_mm"].rolling(window=30, min_periods=1).sum()

    # 蓄水量變化率：前一日差值（第一日會是 NaN，需處理）
    df["storage_diff_1d"] = df["effective_storage"].diff()
    df["storage_diff_1d"] = df["storage_diff_1d"].fillna(0)  # 第一日無前值，補 0 合理

    # 檢查
    print(f"  rainfall_7d_sum  : 範圍 [{df['rainfall_7d_sum'].min():.1f}, {df['rainfall_7d_sum'].max():.1f}] mm")
    print(f"  rainfall_30d_sum : 範圍 [{df['rainfall_30d_sum'].min():.1f}, {df['rainfall_30d_sum'].max():.1f}] mm")
    print(f"  storage_diff_1d : 範圍 [{df['storage_diff_1d'].min():.1f}, {df['storage_diff_1d'].max():.1f}] 萬m³")

    # 欄位排序（feature / label / flag 分組排列，LSTM 訓練時明確指定欄位）
    # 訓練時請用：
    #   LABEL_COL = "effective_storage"
    #   FEATURE_COLS = [
    #       "basin_rainfall_tccip_mm",   # 當日降雨
    #       "rainfall_7d_sum",           # 7 天累積降雨
    #       "rainfall_30d_sum",          # 30 天累積降雨
    #       "effective_storage",         # 當前蓄水量
    #       "storage_diff_1d",           # 昨日蓄水變化
    #       "doy_sin",                   # 年週期 sin
    #       "doy_cos",                   # 年週期 cos
    #   ]
    #   注意：basin_rainfall_self_mm 是水利署單點值，僅供 sanity check，不進模型
    #   注意：inflow_cms / outflow_cms 缺值 97.8%，保留於 CSV 但不進模型（口試可說明原因）
    col_order = [
        "data_date",
        "year", "month", "day_of_year", "doy_sin", "doy_cos",
        "basin_rainfall_tccip_mm",   # Feature（主）— TCCIP 空間平均
        "rainfall_7d_sum",           # Feature（衍生）— 7 天累積降雨
        "rainfall_30d_sum",          # Feature（衍生）— 30 天累積降雨
        "storage_diff_1d",           # Feature（衍生）— 前日蓄水變化
        "basin_rainfall_self_mm",    # Feature（對照）— Statistics.aspx 自報
        "inflow_cms",                # Feature（對照）— 保留，不進模型
        "outflow_cms",               # Feature（對照）— 保留，不進模型
        "effective_storage",         # LABEL
        # is_imputed_rainfall_tccip 移除：TCCIP CSV 預先計算完整，永遠為 0，是雜訊
        "is_imputed_rainfall_self",   # 補值旗標（reservoir_daily.basin_rainfall_mm 可能真的有缺）
        "is_imputed_inflow",           # 補值旗標
        "is_imputed_outflow",          # 補值旗標
        "is_imputed_storage",          # 補值旗標
    ]
    df = df[col_order]
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"✅ 輸出：{OUTPUT_FILE}")
    print("=" * 70)
    print(f"筆數：{len(df):,}")
    print(f"欄位：{list(df.columns)}")

    print("\n前 5 筆：")
    print(df.head().to_string(index=False))

    print("\n後 5 筆：")
    print(df.tail().to_string(index=False))

    print("\n數值欄位統計摘要：")
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["year", "month", "day_of_year"], errors="ignore"
    )
    print(numeric.describe().T.to_string())


def main() -> None:
    df = build_training_set()
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print_summary(df)


if __name__ == "__main__":
    main()
